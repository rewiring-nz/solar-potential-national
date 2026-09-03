"""
Train a model that predicts roof FACES, not roof lines.

WHY A DIFFERENT TARGET RATHER THAN A BETTER LINE MODEL. The line detector finds
about as many creases per roof as Josh draws -- 19.2 against 18.0 over 299
buildings -- and its output cannot be turned into geometry, because 98.5% of its
endpoints dangle against 56.4% of his and only 2.6% of its line pairs touch
against his 24.3%. Fragments cannot be assembled into a partition: merging,
snapping and bridging them reached 85.7% dangling and stopped, since roof lines
are mostly parallel and a ray along a ridge never meets another ridge.

Predicting regions removes the assembly step. Two heads:

    INTERIOR   is this pixel inside some roof face
    BOUNDARY   is this pixel on the edge between two faces

Thresholding interior-minus-boundary gives seeds, and a watershed grown from
those seeds under the boundary map yields CLOSED regions by construction. There
is no disconnected output to reassemble, which is the entire point.

WHY TWO HEADS AND NOT JUST INSTANCE IDS. Face ids are arbitrary -- the same roof
labelled twice can number its faces differently -- so a network cannot learn
them directly. Interior and boundary are label-order independent, and the
watershed recovers instances afterwards.

THE HONEST RISK. 92 roofs is not much. The line model's learning curve was flat
from 60 to 74 roofs, which said more data would not have saved it; whether the
same ceiling applies to a region target is exactly what --curve measures here.
The scratch encoder is not offered: on this much data it memorised, and the
pretrained one is the only version that ever generalised.

Usage:
    python tools/train_face_model.py --epochs 40
    python tools/train_face_model.py --curve
    python tools/train_face_model.py --epochs 40 --save data/models/roof_faces_v1.pt
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

DATA = ROOT / "data" / "training_faces"


def load_split(manifest, split, keep=None):
    import numpy as np
    xs, ys, ws = [], [], []
    for rec in manifest["patches"]:
        if rec["split"] != split:
            continue
        if keep is not None and rec["building_id"] not in keep:
            continue
        d = np.load(DATA / rec["file"])
        inst = d["faces"]
        interior = (inst > 0).astype("float32")
        edge = (d["edges"] > 0).astype("float32")
        # A boundary pixel is not interior: the two heads must disagree there,
        # or the watershed has nothing to cut on.
        interior = np.clip(interior - edge, 0, 1)
        # RGB and the three height channels, stacked. The height is not a
        # nicety: a crease is a slope discontinuity, and colour only shows one
        # where the sun happens to have cast a line.
        img = d["image"].astype("float32") / 255.0
        if "height" in d.files:
            img = np.concatenate([img, d["height"].astype("float32") / 255.0],
                                 axis=-1)
        xs.append(img)
        ys.append(np.stack([interior, edge], axis=-1))
        ws.append((d["weight"] > 0).astype("float32"))
    if not xs:
        return None
    import torch
    # .contiguous() is not cosmetic: permute leaves a strided view, and
    # autograd refuses to build the backward graph over one ("view size is not
    # compatible with input tensor's size and stride").
    return (torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous(),
            torch.from_numpy(np.stack(ys)).permute(0, 3, 1, 2).contiguous(),
            torch.from_numpy(np.stack(ws))[:, None].contiguous())


def build(pretrained=True, in_ch=6):
    import torch
    import torch.nn as nn
    from train_line_model import build_unet
    m = build_unet(pretrained=pretrained)
    # same body, two heads instead of three kinds
    if hasattr(m, "out"):
        m.out = nn.Conv2d(m.out.in_channels, 2, 1)
    # WIDEN THE STEM to take height alongside RGB. The ImageNet weights are
    # kept for the colour channels and copied (halved, so the activation
    # scale is unchanged) into the new ones -- training the stem from scratch
    # would throw away the pretrained encoder that is the only reason this
    # generalises on 92 roofs.
    if in_ch != 3 and hasattr(m, "stem"):
        old = m.stem[0]
        new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size,
                        old.stride, old.padding, bias=old.bias is not None)
        with torch.no_grad():
            w = old.weight
            reps = (in_ch + 2) // 3
            new.weight.copy_(w.repeat(1, reps, 1, 1)[:, :in_ch] * (3.0 / in_ch))
            if old.bias is not None:
                new.bias.copy_(old.bias)
        m.stem[0] = new
    return m


def loss_fn(logits, y, w):
    import torch
    import torch.nn.functional as F
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = (bce * w).sum() / w.sum().clamp(min=1) / y.shape[1]
    p = torch.sigmoid(logits)
    num = 2 * (p * y * w).sum(dim=(0, 2, 3)) + 1.0
    den = ((p + y) * w).sum(dim=(0, 2, 3)) + 1.0
    dice = (1 - num / den).mean()
    return bce + dice


def evaluate(model, val, device, thr=0.5):
    import torch
    x, y, w = val
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), 8):
            outs.append(torch.sigmoid(model(x[i:i + 8].to(device))).cpu())
    p = torch.cat(outs)
    res = {}
    for i, name in enumerate(("interior", "boundary")):
        pred = (p[:, i] > thr).float() * w[:, 0]
        gt = y[:, i] * w[:, 0]
        tp = (pred * gt).sum()
        f1 = (2 * tp / (pred.sum() + gt.sum()).clamp(min=1)).item()
        iou = (tp / (pred + gt - pred * gt).sum().clamp(min=1)).item()
        res[name] = {"f1": f1, "iou": iou}
    return res


def train_once(train, val, device, epochs, seed=0, quiet=False, pretrained=True):
    import torch
    torch.manual_seed(seed)
    model = build(pretrained, in_ch=train[0].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    x, y, w = train
    n = len(x)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 8):
            idx = perm[i:i + 8]
            xb, yb, wb = x[idx].to(device), y[idx].to(device), w[idx].to(device)
            opt.zero_grad()
            l = loss_fn(model(xb), yb, wb)
            l.backward()
            opt.step()
            tot += l.item() * len(idx)
        sched.step()
        if not quiet and (ep + 1) % 5 == 0:
            r = evaluate(model, val, device)
            print(f"  epoch {ep+1:>3}  loss {tot/n:.4f}  "
                  f"interior F1 {r['interior']['f1']:.3f}  "
                  f"boundary F1 {r['boundary']['f1']:.3f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--curve", action="store_true",
                    help="is 92 roofs the ceiling, or would more labels help?")
    ap.add_argument("--save", type=str, default=None)
    a = ap.parse_args()

    import torch
    mf = DATA / "manifest.json"
    if not mf.exists():
        print("no data -- run tools/export_face_training.py first")
        return 1
    manifest = json.loads(mf.read_text())
    train = load_split(manifest, "train")
    val = load_split(manifest, "val")
    if train is None or val is None:
        print("empty split")
        return 1
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(train[0])} train patches, {len(val[0])} val patches, device {device}")

    if a.curve:
        ids = sorted({r["building_id"] for r in manifest["patches"]
                      if r["split"] == "train"})
        print("\nLEARNING CURVE -- does more labelling help, or is this the ceiling?")
        for frac in (0.25, 0.5, 0.75, 1.0):
            keep = set(ids[:max(1, int(len(ids) * frac))])
            tr = load_split(manifest, "train", keep)
            if tr is None:
                continue
            m = train_once(tr, val, device, a.epochs, quiet=True)
            r = evaluate(m, val, device)
            print(f"  {len(keep):>3} roofs ({len(tr[0]):>4} patches): "
                  f"interior F1 {r['interior']['f1']:.3f}  "
                  f"boundary F1 {r['boundary']['f1']:.3f}")
        print("\n  Flat means more roofs will not help and the limit is the")
        print("  model or the target, not the labelling.")
        return 0

    model = train_once(train, val, device, a.epochs)
    r = evaluate(model, val, device)
    print(f"\nHELD OUT ({len(manifest['val_building_ids'])} roofs the model never saw)")
    for k, v in r.items():
        print(f"  {k:9s} F1 {v['f1']:.3f}  IoU {v['iou']:.3f}")
    if a.save:
        p = Path(a.save)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "pretrained": True,
                    "heads": ["interior", "boundary"]}, p)
        print(f"\nsaved {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

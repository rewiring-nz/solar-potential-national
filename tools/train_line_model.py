"""
Train a roof-line detector, and answer whether more labels would help.

The model is deliberately small: a 4-level U-Net predicting three channels --
ridge, valley, cliff -- per pixel. With ~150 distinct roofs a structured
wireframe parser is not on the table; a per-pixel detector is, and it plugs into
the existing seam. src/roof_line_source.py already accepts predicted lines and
hands them to the SAME LiDAR gate the Hough detector feeds, so the model
proposes and the LiDAR still disposes.

TWO THINGS DECIDE WHETHER THIS IS HONEST.

  THE SPLIT IS BY ROOF, done in export_training_data.py. Patches from one
  building overlap; splitting at random puts near-identical patches in both sets
  and the validation score stops meaning anything.

  ACCURACY IS A USELESS METRIC HERE. Only ~3.4% of pixels sit on a line, so
  predicting all background scores 96.6%. The loss is Dice plus positively
  weighted BCE, and the reported metric is per-kind F1 at a matched threshold,
  never accuracy.

THE LEARNING CURVE IS THE POINT OF --curve. Josh asked whether all 156 roofs are
needed or whether 100 would do. That is answerable rather than guessable: train
on 20, 40, 60... roofs against a FIXED validation set and look at the shape. If
validation F1 is still climbing steeply at the largest size, more labels help;
if it has flattened, they will not, and the labelling effort is better spent
elsewhere -- on marking roofs COMPLETE, for instance, which is what precision
rests on.

Usage:
    python tools/train_line_model.py --curve          # does more data help?
    python tools/train_line_model.py --epochs 40      # train on everything
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "training"
KINDS = ["ridge", "valley", "cliff"]


def load_split(manifest, split, keep_buildings=None):
    import numpy as np
    xs, ys, ws = [], [], []
    for p in manifest["patches"]:
        if p["split"] != split:
            continue
        if keep_buildings is not None and p["building_id"] not in keep_buildings:
            continue
        z = np.load(DATA / p["file"])
        xs.append(z["image"])
        ys.append(z["lines"])
        ws.append(z["weight"])
    if not xs:
        return None
    import torch
    x = torch.from_numpy(np.stack(xs)).float().permute(0, 3, 1, 2) / 255.0
    y = torch.from_numpy(np.stack(ys)).float().permute(0, 3, 1, 2) / 255.0
    w = torch.from_numpy(np.stack(ws)).float().unsqueeze(1) / 255.0
    return x, y, w


def build_unet(pretrained=False):
    """A U-Net, optionally on a pretrained ImageNet encoder.

    From scratch, 74 roofs is not enough: training loss halves between epochs 50
    and 90 while validation F1 sits flat at 0.24. That is memorisation, and no
    amount of extra epochs fixes it. A pretrained encoder does not need to learn
    what an edge looks like from 74 roofs -- it already knows -- so it spends the
    data on what a ROOF CREASE looks like instead."""
    if pretrained:
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18, ResNet18_Weights

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        class ResUNet(nn.Module):
            def __init__(self):
                super().__init__()
                r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)   # /2, 64
                self.pool = r.maxpool                                # /4
                self.l1, self.l2 = r.layer1, r.layer2                # 64, 128
                self.l3 = r.layer3                                   # 256
                self.u3 = nn.ConvTranspose2d(256, 128, 2, 2)
                self.c3 = block(256, 128)
                self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
                self.c2 = block(128, 64)
                self.u1 = nn.ConvTranspose2d(64, 64, 2, 2)
                self.c1 = block(128, 64)
                self.u0 = nn.ConvTranspose2d(64, 32, 2, 2)
                self.c0 = block(32, 32)
                self.out = nn.Conv2d(32, 3, 1)

            def forward(self, x):
                s = self.stem(x)            # /2
                p = self.pool(s)            # /4
                a = self.l1(p)              # /4  64
                b = self.l2(a)              # /8  128
                c = self.l3(b)              # /16 256
                x = self.c3(torch.cat([self.u3(c), b], 1))    # /8
                x = self.c2(torch.cat([self.u2(x), a], 1))    # /4
                x = self.c1(torch.cat([self.u1(x), s], 1))    # /2
                x = self.c0(self.u0(x))                       # /1
                return self.out(x)

        return ResUNet()
    return _build_scratch_unet()


def _build_scratch_unet():
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self, base=32):
            super().__init__()
            self.d1, self.d2 = block(3, base), block(base, base * 2)
            self.d3, self.d4 = block(base * 2, base * 4), block(base * 4, base * 8)
            self.pool = nn.MaxPool2d(2)
            self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
            self.c3 = block(base * 8, base * 4)
            self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
            self.c2 = block(base * 4, base * 2)
            self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
            self.c1 = block(base * 2, base)
            self.out = nn.Conv2d(base, 3, 1)

        def forward(self, x):
            import torch
            a = self.d1(x); b = self.d2(self.pool(a))
            c = self.d3(self.pool(b)); d = self.d4(self.pool(c))
            x = self.c3(torch.cat([self.u3(d), c], 1))
            x = self.c2(torch.cat([self.u2(x), b], 1))
            x = self.c1(torch.cat([self.u1(x), a], 1))
            return self.out(x)

    return UNet()


def loss_fn(logits, y, w, pos_weight):
    """Dice plus roof-weighted BCE.

    BCE alone collapses to all-background at 3.4% positives; Dice alone is
    unstable early. The weight mask keeps the street out of the loss."""
    import torch
    import torch.nn.functional as F
    bce = F.binary_cross_entropy_with_logits(
        logits, y, reduction="none", pos_weight=pos_weight)
    bce = (bce * w).mean()
    p = torch.sigmoid(logits) * w
    t = y * w
    num = 2 * (p * t).sum((0, 2, 3)) + 1.0
    den = p.sum((0, 2, 3)) + t.sum((0, 2, 3)) + 1.0
    return bce + (1 - num / den).mean()


def evaluate(model, val, device, thr=0.5):
    import torch
    x, y, w = val
    model.eval()
    f1s = []
    with torch.no_grad():
        preds = []
        for i in range(0, len(x), 32):
            preds.append(torch.sigmoid(model(x[i:i + 32].to(device))).cpu())
        p = torch.cat(preds)
    for k in range(3):
        pk = ((p[:, k] > thr).float() * w[:, 0])
        tk = (y[:, k] * w[:, 0])
        tp = (pk * tk).sum()
        f1 = (2 * tp / (pk.sum() + tk.sum() + 1e-6)).item()
        f1s.append(f1)
    return f1s


def train_once(train, val, device, epochs, seed=0, quiet=False, pretrained=False):
    import torch
    torch.manual_seed(seed)
    model = build_unet(pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    x, y, w = train
    pos = torch.tensor([8.0, 8.0, 8.0], device=device).view(1, 3, 1, 1)
    n = len(x)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=max(1, epochs * ((n + 15) // 16)))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 16):
            idx = perm[i:i + 16]
            xb, yb, wb = x[idx].to(device), y[idx].to(device), w[idx].to(device)
            # dihedral augmentation, free and the only thing standing between
            # 150 roofs and immediate overfitting
            if torch.rand(1).item() < 0.5:
                xb, yb, wb = [t.flip(-1) for t in (xb, yb, wb)]
            if torch.rand(1).item() < 0.5:
                xb, yb, wb = [t.flip(-2) for t in (xb, yb, wb)]
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                xb, yb, wb = [torch.rot90(t, k, (-2, -1)) for t in (xb, yb, wb)]
            # flip and rot90 return views with awkward strides; the backward
            # pass needs them laid out contiguously
            xb, yb, wb = xb.contiguous(), yb.contiguous(), wb.contiguous()
            opt.zero_grad()
            l = loss_fn(model(xb), yb, wb, pos)
            l.backward()
            opt.step()
            sched.step()
            tot += l.item()
        if not quiet and (ep + 1) % 5 == 0:
            f1 = evaluate(model, val, device)
            print(f"    epoch {ep+1:>3}  loss {tot/max(1,n//16):.3f}  "
                  f"val F1 " + " ".join(f"{k[:3]} {v:.2f}" for k, v in zip(KINDS, f1)))
    return model, evaluate(model, val, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--curve", action="store_true",
                    help="train at several dataset sizes to see if more helps")
    ap.add_argument("--sizes", nargs="*", type=int, default=None)
    ap.add_argument("--pretrained", action="store_true",
                    help="ImageNet-pretrained ResNet18 encoder")
    ap.add_argument("--save", type=str, default=None,
                    help="train on EVERY labelled roof and save the weights")
    a = ap.parse_args()

    if not (DATA / "manifest.json").exists():
        print("no training data -- run tools/export_training_data.py first")
        return 1
    import torch
    manifest = json.loads((DATA / "manifest.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")

    val = load_split(manifest, "val")
    train_ids = sorted({p["building_id"] for p in manifest["patches"]
                        if p["split"] == "train"})
    if val is None or not train_ids:
        print("not enough data")
        return 1
    print(f"device {device} | {len(train_ids)} train roofs, "
          f"{len(val[0])} val patches from "
          f"{len({p['building_id'] for p in manifest['patches'] if p['split']=='val'})} roofs\n")

    if a.save:
        # For production there is no holdout: every labelled roof is training
        # data. The held-out numbers above are what the model is worth; this
        # just gets the most out of the labels for inference.
        import torch
        tr = load_split(manifest, "train")
        va = load_split(manifest, "val")
        allx = (torch.cat([tr[0], va[0]]), torch.cat([tr[1], va[1]]),
                torch.cat([tr[2], va[2]]))
        print(f"training on ALL {len(allx[0])} patches for inference "
              f"(no holdout -- see the scores above for what it is worth)")
        model, _ = train_once(allx, val, device, a.epochs, quiet=True,
                              pretrained=a.pretrained)
        out = Path(a.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "pretrained": a.pretrained, "kinds": KINDS,
                    "patch": manifest["patch"]}, out)
        print(f"wrote {out}")
        return 0

    if not a.curve:
        train = load_split(manifest, "train")
        t0 = time.time()
        _, f1 = train_once(train, val, device, a.epochs, pretrained=a.pretrained)
        print(f"\nfinal val F1: " +
              "  ".join(f"{k} {v:.3f}" for k, v in zip(KINDS, f1)) +
              f"   mean {sum(f1)/3:.3f}   ({time.time()-t0:.0f}s)")
        return 0

    sizes = a.sizes or [s for s in (10, 20, 40, 60, len(train_ids))
                        if s <= len(train_ids)]
    sizes = sorted(set(sizes))
    print("LEARNING CURVE -- does labelling more roofs still help?\n")
    print(f"{'train roofs':>12}{'ridge':>8}{'valley':>8}{'cliff':>8}{'mean F1':>10}")
    prev = None
    for nsz in sizes:
        keep = set(train_ids[:nsz])
        tr = load_split(manifest, "train", keep)
        if tr is None:
            continue
        _, f1 = train_once(tr, val, device, a.epochs, quiet=True, pretrained=a.pretrained)
        m = sum(f1) / 3
        delta = "" if prev is None else f"   {m - prev:+.3f}"
        print(f"{nsz:>12}{f1[0]:>8.3f}{f1[1]:>8.3f}{f1[2]:>8.3f}{m:>10.3f}{delta}")
        prev = m
    print("\nRead the last column. If it is still climbing at the largest size,")
    print("more labelled roofs are worth the effort. If it has flattened, they")
    print("are not -- and the same hours are better spent marking roofs COMPLETE,")
    print("which is what precision rests on and where the sample is thinnest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

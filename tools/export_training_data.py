"""
Turn drawn labels into something a model can train on.

Needed whatever architecture we land on, so it is built first and kept
independent of that choice: image patches plus per-pixel line masks, split by
ROOF rather than by patch.

WHY THE SPLIT IS BY ROOF. Patches from one roof overlap and share the same
building, the same imagery capture, the same roof material. Split them at random
and near-identical patches land in both train and validation, the model memorises
the roof, and the validation score becomes a fiction that only collapses on real
data. This is the single easiest way to produce an impressive number that means
nothing, so the split is by building id and never negotiable.

WHAT IS EXPORTED PER PATCH:
    image   RGB from the same orthophoto the labeller was looking at
    lines   one mask channel per kind (ridge / valley / cliff), drawn a few
            pixels wide -- a 1px line is nearly unlearnable and every published
            line detector widens its targets
    weight  a mask of pixels that are actually roof, so the loss ignores the
            street. Without it most of the target is background and the model
            learns to predict nothing everywhere, which scores well and is
            useless.

DELIBERATELY NOT EXPORTED: roofs flagged absent, not_building or unclear. Those
carry no geometry anyone believes, and "cannot read the image" in particular
would teach the model to hallucinate lines out of blur.

Usage:
    python tools/export_training_data.py
    python tools/export_training_data.py --patch 128 --stride 64 --val-frac 0.2
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

LABELS = ROOT / "data" / "roof_labels.json"
OUT = ROOT / "data" / "training"
KINDS = ["ridge", "valley", "cliff"]
SKIP_FLAGS = {"absent", "not_building", "unclear"}
LINE_WIDTH_PX = 3          # targets are widened; a 1px line is nearly unlearnable
PAD_M = 4.0


def rasterise(shape, segments, bounds, width_px):
    """Draw world-coordinate segments into a mask at the crop's resolution."""
    import numpy as np
    from PIL import Image, ImageDraw
    minx, miny, maxx, maxy = bounds
    h, w = shape
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    for a, b in segments:
        ax = (a[0] - minx) / (maxx - minx) * w
        ay = (1 - (a[1] - miny) / (maxy - miny)) * h
        bx = (b[0] - minx) / (maxx - minx) * w
        by = (1 - (b[1] - miny) / (maxy - miny)) * h
        d.line([ax, ay, bx, by], fill=255, width=width_px)
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-line-px", type=int, default=40,
                    help="drop patches with almost no line in them")
    a = ap.parse_args()

    import numpy as np
    import geopandas as gpd
    import rasterio
    import rasterio.windows
    import shapely
    from PIL import Image, ImageDraw
    from src.region_build import area_paths
    from score_geometry import _line_points

    labels = json.loads(LABELS.read_text())["buildings"]
    usable = {k: v for k, v in labels.items()
              if v.get("problem") not in SKIP_FLAGS and v.get("lines")}
    ids = sorted(usable, key=lambda k: int(k))
    if not ids:
        print("no usable labelled roofs")
        return 1

    # Split by ROOF, deterministically, so re-running does not quietly reshuffle
    # what the model has already seen.
    rng = np.random.default_rng(20260902)
    order = rng.permutation(len(ids))
    n_val = max(1, int(len(ids) * a.val_frac))
    val_ids = {ids[i] for i in order[:n_val]}

    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (OUT / split).mkdir(exist_ok=True)

    ctxs = {}
    counts = {"train": 0, "val": 0}
    skipped_empty = 0
    manifest = []

    for k in ids:
        lab = usable[k]
        bid = int(k)
        area = lab.get("area")
        if area not in ctxs:
            p = area_paths(area)
            ctxs[area] = None if not p["imagery"].exists() else {
                "gdf": gpd.read_file(
                    p["dir"] / "building_outlines_dedup.geojson"
                    if (p["dir"] / "building_outlines_dedup.geojson").exists()
                    else p["outlines"]).set_index("building_id", drop=False),
                "img": rasterio.open(p["imagery"]),
            }
        ctx = ctxs[area]
        if not ctx or bid not in ctx["gdf"].index:
            continue
        geom = ctx["gdf"].loc[bid].geometry
        minx, miny, maxx, maxy = geom.bounds
        bounds = (minx - PAD_M, miny - PAD_M, maxx + PAD_M, maxy + PAD_M)

        win = rasterio.windows.from_bounds(*bounds, ctx["img"].transform)
        rgb = np.moveaxis(ctx["img"].read([1, 2, 3], window=win,
                                          boundless=True, fill_value=0), 0, -1)
        if rgb.shape[0] < a.patch or rgb.shape[1] < a.patch:
            continue
        shape = rgb.shape[:2]

        # one channel per kind, so the model can be asked which it found
        masks = []
        for kind in KINDS:
            segs = [(l["a"], l["b"]) for l in lab["lines"]
                    if l.get("kind") == kind and l.get("a") and l.get("b")]
            masks.append(rasterise(shape, segs, bounds, LINE_WIDTH_PX))
        mask = np.stack(masks, axis=-1)

        # roof-only weight, so the loss is not dominated by street and garden
        wimg = Image.new("L", (shape[1], shape[0]), 0)
        wd = ImageDraw.Draw(wimg)
        ring = list(geom.exterior.coords)
        wd.polygon([((x - bounds[0]) / (bounds[2] - bounds[0]) * shape[1],
                     (1 - (y - bounds[1]) / (bounds[3] - bounds[1])) * shape[0])
                    for x, y in ring], fill=255)
        weight = np.array(wimg)

        split = "val" if k in val_ids else "train"
        for top in range(0, shape[0] - a.patch + 1, a.stride):
            for left in range(0, shape[1] - a.patch + 1, a.stride):
                sl = (slice(top, top + a.patch), slice(left, left + a.patch))
                m = mask[sl]
                if m.sum() / 255 < a.min_line_px:
                    skipped_empty += 1
                    continue
                stem = f"{bid}_{top}_{left}"
                np.savez_compressed(
                    OUT / split / f"{stem}.npz",
                    image=rgb[sl].astype("uint8"),
                    lines=m.astype("uint8"),
                    weight=weight[sl].astype("uint8"))
                counts[split] += 1
                manifest.append({"file": f"{split}/{stem}.npz", "building_id": bid,
                                 "area": area, "split": split})

    (OUT / "manifest.json").write_text(json.dumps(
        {"kinds": KINDS, "patch": a.patch, "stride": a.stride,
         "line_width_px": LINE_WIDTH_PX,
         "val_building_ids": sorted(val_ids, key=int),
         "patches": manifest}, indent=1))

    print(f"{len(ids)} labelled roofs usable "
          f"({len(labels) - len(usable)} skipped: flagged, or no lines)")
    print(f"  split by ROOF: {len(ids) - len(val_ids)} train, {len(val_ids)} val")
    print(f"  patches written: {counts['train']} train, {counts['val']} val")
    print(f"  patches dropped for having almost no line: {skipped_empty}")
    print(f"\nwrote {OUT}")
    print("\nThe split is by building, not by patch. Patches from one roof "
          "overlap\nand share a building, so splitting at random would put "
          "near-identical\npatches in both sets and the validation score would "
          "mean nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

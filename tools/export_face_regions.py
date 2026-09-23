"""Export the drawn FACES as a dense region-learning target.

WHY THIS EXISTS. The markup already trains a detector -- but only to light
up thin LINE pixels, which a long chain then has to close into polygons.
That chain is where the inventions come from: a line that fails to close
into a cell is silently dropped (#4735237: 264 panels laid across drawn
ridges), and when the polygons come out badly a parametric shape vocabulary
guesses instead, which is how #4735292 got a flat top it does not have.

Matching simple shapes to buildings may be the wrong approach now that a
markup dataset exists: the model could just learn the markup instead of
trying to fit shapes.

The drawn faces are ground-truth POLYGONS. Learning them as REGIONS removes both
failure modes at once: a region cannot fail to close, and there is no
vocabulary to invent from.

WHAT IS EXPORTED, one sample per roof (not per patch: a 12 m patch can sit
entirely inside one face, where "distance to the nearest boundary" is
unknowable from the pixels in view -- faces are roof-scale objects and the
model has to see the whole roof):

    image   7 channels, both sensors fused at the INPUT rather than by
            reconciling two fragile outputs downstream:
              0-2  RGB from the orthophoto the labeller was looking at
              3    height above the roof's own base, /10 m
              4    slope, /45 deg
              5,6  sin/cos of aspect -- the signal that reads a pyramid
                   unambiguously (see the aspect map on #4735292) and that
                   the imagery-only detector keeps missing
    target  2 channels -- BOUNDARY (the drawn face edges, a few px wide) and CORE
            (the drawn faces eroded). Scale-tolerant, unlike a raw distance
            transform, which saturates near zero on narrow faces and trains
            badly; and between them they define a watershed exactly: cores
            are the seeds, boundaries the barriers. Still dense supervision,
            unlike thin lines where ~2% of pixels carry the signal.
    weight  the drawn faces themselves -- the loss ignores street and garden.

Each roof is SCALE-NORMALISED into the frame (footprint's long side spans
FILL_PX) so the model learns roof STRUCTURE rather than absolute size, which
is what makes ~130 roofs go far enough to train on.

Split is by ROOF and pinned to data/bench_ids.txt, so the benchmark roofs are
never trained on and the held-out number stays honest.

Usage:
    python tools/export_face_regions.py
    python tools/export_face_regions.py --ids 4735292 --preview
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
import rasterio.windows
import geopandas as gpd
import shapely
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.spatial import cKDTree

from src.region_build import area_paths
from src.pointcloud_source import PointCloudSource

OUT = ROOT / "data" / "face_regions"
SIZE = 256          # frame the model sees
FILL_PX = 208       # footprint long side spans this many pixels
BOUNDARY_PX = 3     # a 1 px target line is nearly unlearnable
CORE_ERODE_PX = 4   # seed region inside each face
EAVE_ERASE_PX = 5   # how wide a band of footprint edge to erase from it
VOID = {"absent", "not_building", "unclear"}


def roof_sample(geom, faces, img_ds, pc):  # noqa: C901
    """(image HxWx7 float32, target HxW float32, weight HxW bool) or None."""
    minx, miny, maxx, maxy = geom.bounds
    span = max(maxx - minx, maxy - miny)
    if span <= 0:
        return None
    m_per_px = span / FILL_PX
    half = SIZE / 2 * m_per_px
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    b = (cx - half, cy - half, cx + half, cy + half)

    win = rasterio.windows.from_bounds(*b, img_ds.transform)
    rgb = np.moveaxis(img_ds.read([1, 2, 3], window=win, boundless=True,
                                  fill_value=0, out_shape=(3, SIZE, SIZE)),
                      0, -1).astype(np.float32) / 255.0

    def to_px(x, y):
        return ((x - b[0]) / (b[2] - b[0]) * SIZE,
                (1 - (y - b[1]) / (b[3] - b[1])) * SIZE)

    # --- LiDAR channels: grid z, then differentiate -------------------
    pts = pc.points_in_bbox(b[0], b[1], b[2], b[3], building_only=True)
    z = np.zeros((SIZE, SIZE), np.float32)
    if pts is not None and len(pts) >= 50:
        gx, gy = np.meshgrid(
            np.linspace(b[0] + m_per_px / 2, b[2] - m_per_px / 2, SIZE),
            np.linspace(b[3] - m_per_px / 2, b[1] + m_per_px / 2, SIZE))
        tree = cKDTree(pts[:, :2])
        dist, idx = tree.query(np.c_[gx.ravel(), gy.ravel()], k=1)
        z = pts[idx, 2].reshape(SIZE, SIZE).astype(np.float32)
        z[dist.reshape(SIZE, SIZE) > 2.5] = np.nan
    base = np.nanpercentile(z, 5) if np.isfinite(z).any() else 0.0
    zf = np.nan_to_num(z - base, nan=0.0)
    # SMOOTH BEFORE DIFFERENTIATING. The survey is ~4.9 returns/m2, so a
    # nearest-neighbour grid at 0.1 m/px is a staircase and its per-pixel
    # gradient is confetti (visible in the first preview). Blur at roughly
    # the sample spacing first -- the same 1.5 m-ish neighbourhood a local
    # plane fit would use, at a fraction of the cost.
    zf = gaussian_filter(zf, max(2.0, 1.2 / m_per_px))
    dzy, dzx = np.gradient(zf, m_per_px)
    slope = np.degrees(np.arctan(np.hypot(dzx, dzy)))
    mag = np.hypot(dzx, dzy) + 1e-9
    chans = np.dstack([
        rgb,
        np.clip(zf / 10.0, 0, 1),
        np.clip(slope / 45.0, 0, 1),
        (-dzx / mag + 1) / 2,           # downhill east/west
        (dzy / mag + 1) / 2,            # downhill north/south (row axis flips)
    ]).astype(np.float32)

    # --- target: distance to the nearest drawn face boundary ----------
    wim = Image.new("L", (SIZE, SIZE), 0)
    wd = ImageDraw.Draw(wim)
    bim = Image.new("L", (SIZE, SIZE), 0)
    bd = ImageDraw.Draw(bim)
    for f in faces:
        ring = [to_px(x, y) for x, y in f]
        wd.polygon(ring, fill=255)
        bd.line(ring + [ring[0]], fill=255, width=BOUNDARY_PX)
    # DO NOT ASK THE MODEL TO PREDICT THE EAVES. The footprint is known
    # exactly at inference, and eave pixels outnumber crease pixels several
    # to one, so training on them spends the capacity of a 96-roof dataset
    # on the one boundary that needs no learning. Measured 18 Sep: with
    # eaves in the target the model returned the outline and little else
    # (#4734678 one face where three were drawn, #4735104 two where
    # eight). Erasing them leaves only the interior creases -- the actual
    # unknown -- and the watershed gets the outline from the roof mask.
    ed = ImageDraw.Draw(bim)
    fr = [to_px(x, y) for x, y in geom.exterior.coords]
    ed.line(fr + [fr[0]], fill=0, width=BOUNDARY_PX + EAVE_ERASE_PX)
    weight = np.array(wim) > 0
    if weight.sum() < 400:
        return None
    boundary = np.array(bim) > 0
    core = binary_erosion(weight & ~boundary,
                          np.ones((3, 3), bool), iterations=CORE_ERODE_PX)
    target = np.dstack([boundary, core]).astype(np.float32)
    return chans, target, weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--preview", action="store_true",
                    help="write a PNG per roof instead of training data")
    a = ap.parse_args()

    B = json.loads((ROOT / "data/roof_labels.json").read_text())["buildings"]
    val_ids = set((ROOT / "data/bench_ids.txt").read_text().split())
    OUT.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val"):
        (OUT / s).mkdir(exist_ok=True)

    pc = PointCloudSource(max_cached_tiles=3)
    ctx, n = {}, {"train": 0, "val": 0}
    skipped = {"no_imagery": 0, "no_faces": 0, "too_small": 0}
    for k, v in sorted(B.items(), key=lambda kv: int(kv[0])):
        bid = int(k)
        if a.ids and bid not in a.ids:
            continue
        if v.get("problem") in VOID:
            continue
        faces = [[(float(x), float(y)) for x, y in f["ring"]]
                 for f in (v.get("faces") or [])
                 if f.get("ring") and len(f["ring"]) >= 3]
        if not faces:
            skipped["no_faces"] += 1
            continue
        area = v.get("area")
        if area not in ctx:
            p = area_paths(area)
            dd = p["dir"] / "building_outlines_dedup.geojson"
            ctx[area] = None
            if p["imagery"].exists():
                ctx[area] = (gpd.read_file(dd if dd.exists() else p["outlines"])
                             .set_index("building_id", drop=False),
                             rasterio.open(p["imagery"]))
        if ctx[area] is None or bid not in ctx[area][0].index:
            skipped["no_imagery"] += 1
            continue
        gdf, img_ds = ctx[area]
        out = roof_sample(gdf.loc[bid].geometry, faces, img_ds, pc)
        if out is None:
            skipped["too_small"] += 1
            continue
        chans, target, weight = out
        if a.preview:
            im = Image.fromarray((chans[:, :, :3] * 255).astype("uint8"))
            heat = Image.fromarray(
                (np.dstack([target[:, :, 0], target[:, :, 1],
                            np.zeros_like(target[:, :, 0])]) * 255
                 ).astype("uint8"))
            asp = Image.fromarray(
                (np.dstack([chans[:, :, 5], chans[:, :, 4],
                            chans[:, :, 6]]) * 255).astype("uint8"))
            sheet = Image.new("RGB", (SIZE * 3, SIZE))
            sheet.paste(im, (0, 0)); sheet.paste(asp, (SIZE, 0))
            sheet.paste(heat, (SIZE * 2, 0))
            sheet.save(f"/tmp/fr_{bid}.png")
            print(f"#{bid}: preview -> /tmp/fr_{bid}.png ({len(faces)} faces)")
            continue
        split = "val" if k in val_ids else "train"
        np.savez_compressed(OUT / split / f"{bid}.npz",
                            image=(chans * 255).astype("uint8"),
                            target=(target * 255).astype("uint8"),
                            weight=weight.astype("uint8"))
        n[split] += 1
    if not a.preview:
        (OUT / "manifest.json").write_text(json.dumps(
            {"size": SIZE, "fill_px": FILL_PX,
             "channels": ["r", "g", "b", "z", "slope", "asp_x", "asp_y"],
             "train": n["train"], "val": n["val"]}, indent=1))
        print(f"exported train {n['train']}  val {n['val']}   skipped {skipped}")


if __name__ == "__main__":
    sys.exit(main() or 0)

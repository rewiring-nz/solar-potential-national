"""
Export Josh's roof FACES as training targets, not his lines.

WHY THE TARGET HAS TO CHANGE. The line detector works, in the narrow sense that
it finds about as many creases per roof as Josh draws -- 19.2 against 18.0 over
299 buildings. It is useless for geometry anyway, because of topology:

                        lines/roof   touching pairs   dangling ends
    Josh's markup           18.0         24.3%            56.4%
    model predictions       19.2          2.6%            98.5%

He draws a ridge until it meets a hip, so his lines form a graph with real
junctions and the faces between them are implied. The model activates along a
crease and the stroke ends where the activation faded, so its output is a
scatter that never closes anything. Merging, snapping and bridging those
fragments was tried and moved dangling ends only to 85.7%: roof lines are
mostly parallel, and a ray along a ridge never meets another ridge.

A model that predicts REGIONS cannot fail this way. There is no such thing as a
disconnected face -- every pixel it labels belongs to some region, so the output
is a partition by construction rather than by inference.

THE TARGETS ALREADY EXIST. roof_labels.json carries a `faces` array per roof:
rings the labelling tool derives from his lines in the browser, with an area and
a usable flag. 85 completed roofs carry them, and nothing had read them until
now. So this needs no new labelling work -- his existing markup, re-projected
into the target a segmentation model wants.

WHAT IS WRITTEN PER PATCH:
    image     RGB from the same orthophoto he was looking at
    height    normalised DSM height plus its two gradient components, on the
              same grid -- a crease is a gradient discontinuity, and until now
              the model was never shown it
    faces     an integer instance mask, one id per face, 0 for not-roof
    edges     the face boundaries, 2 px wide -- a boundary-aware loss needs
              them and they are free to compute here
    usable    a binary mask of faces he did NOT mark "no panels here"
    weight    roof-only, so the loss ignores the street

SPLIT BY ROOF, never by patch, for the same reason as the line exporter:
patches from one roof share a building, imagery capture and roof material, so a
random split puts near-identical patches in train and validation and the score
becomes fiction.

Usage:
    python tools/export_face_training.py
    python tools/export_face_training.py --patch 192 --stride 96
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LABELS = ROOT / "data" / "roof_labels.json"
OUT = ROOT / "data" / "training_faces"
SKIP_FLAGS = {"absent", "not_building", "unclear"}
PAD_M = 4.0
EDGE_PX = 2
MIN_FACE_PX = 60          # a face smaller than this in-patch teaches nothing


def _rasterise_faces(shape, faces, bounds):
    """Instance mask (uint16), edge mask, usable mask -- all at patch scale."""
    import numpy as np
    from PIL import Image, ImageDraw

    minx, miny, maxx, maxy = bounds
    h, w = shape

    def to_px(p):
        return ((p[0] - minx) / (maxx - minx) * w,
                (1 - (p[1] - miny) / (maxy - miny)) * h)

    inst = Image.new("I", (w, h), 0)
    usable = Image.new("L", (w, h), 0)
    edges = Image.new("L", (w, h), 0)
    di, du, de = ImageDraw.Draw(inst), ImageDraw.Draw(usable), ImageDraw.Draw(edges)
    for i, f in enumerate(faces, start=1):
        ring = [to_px(p) for p in f["ring"]]
        if len(ring) < 3:
            continue
        di.polygon(ring, fill=i)
        if f.get("usable", True):
            du.polygon(ring, fill=255)
        de.line(ring + [ring[0]], fill=255, width=EDGE_PX)
    return (np.array(inst).astype("uint16"),
            np.array(edges), np.array(usable))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--val-frac", type=float, default=0.2)
    a = ap.parse_args()

    import numpy as np
    import geopandas as gpd
    import rasterio
    import rasterio.windows
    from PIL import Image, ImageDraw
    from src.region_build import area_paths

    labels = json.loads(LABELS.read_text())["buildings"]
    usable_roofs = {k: v for k, v in labels.items()
                    if v.get("problem") not in SKIP_FLAGS
                    and v.get("complete") and (v.get("faces") or [])}
    ids = sorted(usable_roofs, key=lambda k: int(k))
    if not ids:
        print("no completed roofs carry faces -- nothing to export")
        return 1

    rng = np.random.default_rng(20260903)
    order = rng.permutation(len(ids))
    n_val = max(1, int(len(ids) * a.val_frac))
    val_ids = {ids[i] for i in order[:n_val]}

    OUT.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val"):
        (OUT / s).mkdir(exist_ok=True)

    ctxs = {}
    counts = {"train": 0, "val": 0}
    n_faces = 0
    manifest = []

    for k in ids:
        lab = usable_roofs[k]
        bid = int(k)
        area = lab.get("area")
        if area not in ctxs:
            p = area_paths(area)
            dd = p["dir"] / "building_outlines_dedup.geojson"
            ctxs[area] = None if not p["imagery"].exists() else {
                "gdf": gpd.read_file(dd if dd.exists() else p["outlines"]
                                     ).set_index("building_id", drop=False),
                "img": rasterio.open(p["imagery"]),
                # THE HEIGHT DATA THE MODEL HAS NEVER SEEN. A ridge is a slope
                # discontinuity, which the LiDAR measures directly, and both
                # exporters fed RGB only -- so the model was asked to infer
                # roof geometry from colour and shadow while the geometry sat
                # unused on disk. That is very likely why boundary F1 is 0.185
                # while interior F1 is 0.83: roof-vs-street is a colour
                # question, a crease is not.
                "dsm": rasterio.open(p["dsm"]) if p["dsm"].exists() else None}
        ctx = ctxs[area]
        if not ctx or bid not in ctx["gdf"].index:
            continue
        geom = ctx["gdf"].loc[bid].geometry
        minx, miny, maxx, maxy = geom.bounds
        bounds = (minx - PAD_M, miny - PAD_M, maxx + PAD_M, maxy + PAD_M)

        win = rasterio.windows.from_bounds(*bounds, ctx["img"].transform)
        try:
            rgb = np.moveaxis(ctx["img"].read([1, 2, 3], window=win,
                                              boundless=True, fill_value=0), 0, -1)
        except Exception:
            continue
        if rgb.shape[0] < a.patch or rgb.shape[1] < a.patch:
            continue
        shape = rgb.shape[:2]

        # Height, resampled onto the imagery grid, as three derived channels:
        # normalised height (where the roof is high), and the two gradient
        # components (which way the surface tilts). A crease is where the
        # gradient CHANGES, so handing the network the gradient rather than raw
        # z means the discontinuity is one convolution away instead of buried.
        hgt = np.zeros(shape + (3,), dtype="float32")
        if ctx.get("dsm") is not None:
            try:
                dwin = rasterio.windows.from_bounds(*bounds,
                                                    ctx["dsm"].transform)
                z = ctx["dsm"].read(1, window=dwin, boundless=True,
                                    fill_value=np.nan,
                                    out_shape=shape).astype("float32")
                fin = np.isfinite(z)
                if fin.sum() > 50:
                    lo = np.nanpercentile(z[fin], 2)
                    hi = np.nanpercentile(z[fin], 98)
                    zz = np.clip((np.nan_to_num(z, nan=lo) - lo)
                                 / max(hi - lo, 1e-3), 0, 1)
                    gy, gx = np.gradient(zz)
                    hgt[..., 0] = zz
                    hgt[..., 1] = np.clip(gx * 8 + 0.5, 0, 1)
                    hgt[..., 2] = np.clip(gy * 8 + 0.5, 0, 1)
            except Exception:
                pass

        faces = [f for f in lab["faces"] if len(f.get("ring") or []) >= 3]
        if not faces:
            continue
        inst, edges, usable = _rasterise_faces(shape, faces, bounds)

        wimg = Image.new("L", (shape[1], shape[0]), 0)
        wd = ImageDraw.Draw(wimg)
        ring = [((x - bounds[0]) / (bounds[2] - bounds[0]) * shape[1],
                 (1 - (y - bounds[1]) / (bounds[3] - bounds[1])) * shape[0])
                for x, y in geom.exterior.coords]
        wd.polygon(ring, fill=255)
        weight = np.array(wimg)

        split = "val" if k in val_ids else "train"
        for top in range(0, shape[0] - a.patch + 1, a.stride):
            for left in range(0, shape[1] - a.patch + 1, a.stride):
                sl = (slice(top, top + a.patch), slice(left, left + a.patch))
                pi = inst[sl]
                # a patch with no face, or only a sliver of one, teaches nothing
                if int((pi > 0).sum()) < MIN_FACE_PX:
                    continue
                stem = f"{bid}_{top}_{left}"
                np.savez_compressed(
                    OUT / split / f"{stem}.npz",
                    image=rgb[sl].astype("uint8"),
                    height=(hgt[sl] * 255).astype("uint8"),
                    faces=pi.astype("uint16"),
                    edges=edges[sl].astype("uint8"),
                    usable=usable[sl].astype("uint8"),
                    weight=weight[sl].astype("uint8"))
                counts[split] += 1
                manifest.append({"file": f"{split}/{stem}.npz",
                                 "building_id": bid, "area": area,
                                 "split": split})
        n_faces += len(faces)

    (OUT / "manifest.json").write_text(json.dumps(
        {"patch": a.patch, "stride": a.stride, "edge_px": EDGE_PX,
         "val_building_ids": sorted(val_ids, key=int),
         "patches": manifest}, indent=1))

    print(f"{len(ids)} completed roofs carry tool-derived faces "
          f"({n_faces} faces total)")
    print(f"  split by ROOF: {len(ids) - len(val_ids)} train, {len(val_ids)} val")
    print(f"  patches: {counts['train']} train, {counts['val']} val")
    print(f"\nwrote {OUT}")
    print("\nThese are REGION targets. A model trained on them cannot emit the")
    print("disconnected strokes the line detector does -- 98.5% of its endpoints")
    print("dangle, against 56.4% of Josh's, which is why fragments could never be")
    print("assembled into a roof.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

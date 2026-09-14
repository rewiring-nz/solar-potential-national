"""RID2 -> our 4-channel line-training patches (pretraining corpus).

Josh: "Why can't the detector see hips? ... It needs to see hips and
edges/cliffs" -- and, weeks earlier, the agreed lever: pre-train on public
labelled roofs, fine-tune on his markups. RID2 (Krapf et al., CC-BY-4.0,
zenodo.org/records/14062580) provides ~4,764 Dutch aerial tiles at 0.08 m
with per-segment orientation polygons. Fold LINES are derived from segment
adjacency:

  - two segments whose azimuths oppose (~180 deg)      -> ridge
  - azimuths ~90 deg apart, junction on a convex run   -> hip
  - azimuths ~90 deg apart, junction at a reflex bend  -> valley
  - anything touching 'flat' or background             -> no line (and the
    CLIFF channel is unsupervised on RID patches: no heights in the data)

Output matches tools/export_training_data.py exactly (128 px patches,
stride 64, 3 px lines, 4 channels) plus a per-patch channel weight vector
`cw` so the trainer can zero the cliff channel's loss here while Josh's
patches keep all four supervised.

    .venv-sam/bin/python tools/export_rid_training.py --inspect   # layout
    .venv-sam/bin/python tools/export_rid_training.py             # export
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RID_DIR = ROOT / "data" / "rid2" / "dataset"
OUT_DIR = ROOT / "data" / "training_rid2"
KINDS = ["ridge", "valley", "cliff", "hip"]
PATCH, STRIDE, LINE_WIDTH_PX = 128, 64, 3
RES_M = 0.08

AZ = {"N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
      "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225,
      "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}


def classify_edge(az_a, az_b, shared, union):
    """ridge / hip / valley / None for the shared edge of two segments."""
    import numpy as np
    if az_a is None or az_b is None:
        return None
    d = abs((az_a - az_b + 180) % 360 - 180)
    if d >= 135:
        return "ridge"
    if d < 45:
        return None            # near-parallel faces: seam, not a fold type
    # ~90 deg: hip at convex junctions, valley at reflex ones. Test the
    # union outline's turn direction at the vertex nearest the shared
    # edge's outer endpoint.
    try:
        ext = list(union.exterior.coords[:-1])
        import numpy as np
        pts = [np.array(shared.coords[0]), np.array(shared.coords[-1])]
        best_i, best_d = None, 1e9
        for p in pts:
            for i, c in enumerate(ext):
                dd = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
                if dd < best_d:
                    best_d, best_i = dd, i
        if best_d > 1.5 ** 2:
            return "valley"     # neither end reaches the outline: interior
        a = np.array(ext[best_i - 1])
        b = np.array(ext[best_i])
        c = np.array(ext[(best_i + 1) % len(ext)])
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        # union.exterior is counter-clockwise for shapely-valid polygons:
        # positive cross = convex vertex
        return "hip" if cross > 0 else "valley"
    except Exception:
        return "hip" if d >= 45 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.inspect or not RID_DIR.exists():
        base = ROOT / "data" / "rid2"
        for p in sorted(base.rglob("*"))[:40]:
            print(p.relative_to(base))
        return

    import numpy as np
    from PIL import Image, ImageDraw
    from shapely.geometry import shape as shp, LineString
    from shapely.ops import unary_union

    # annotations: one geojson with segment polygons + orientation class
    cands = list(RID_DIR.rglob("*.json")) + list(RID_DIR.rglob("*.geojson"))
    print("annotation candidates:", [c.name for c in cands[:10]])
    seg_files = [c for c in cands if "segment" in c.name.lower()]
    if not seg_files:
        print("no segment geojson found -- inspect the layout")
        return
    feats = []
    for sf in seg_files:
        d = json.load(open(sf))
        feats.extend(d.get("features") or [])
    print(f"{len(feats)} segment polygons from {len(seg_files)} file(s)")

    # group by image tile via geo images: each tif named cx_cy in EPSG28992
    geo_dir = next((d for d in RID_DIR.rglob("geo_images") if d.is_dir()),
                   None)
    if geo_dir is None:
        print("no geo_images dir found")
        return
    import rasterio
    from shapely.strtree import STRtree
    polys = []
    metas = []
    for f in feats:
        try:
            g = shp(f["geometry"])
            if g.geom_type != "Polygon" or g.area < 2.0:
                continue
            props = f.get("properties") or {}
            cls = (props.get("class") or props.get("label")
                   or props.get("orientation") or props.get("azimuth"))
            polys.append(g)
            metas.append(str(cls))
        except Exception:
            continue
    tree = STRtree(polys)
    print(f"{len(polys)} usable segment polygons; classes sample:",
          sorted(set(metas))[:20])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_patch = 0
    tifs = sorted(geo_dir.glob("*.tif"))
    if a.limit:
        tifs = tifs[:a.limit]
    for ti, tif in enumerate(tifs):
        try:
            ds = rasterio.open(tif)
            b = ds.bounds
            rgb = np.moveaxis(ds.read([1, 2, 3]), 0, -1)
        except Exception:
            continue
        h, w = rgb.shape[:2]
        from shapely.geometry import box
        idxs = tree.query(box(*b))
        local = [(polys[i], metas[i]) for i in np.atleast_1d(idxs)
                 if polys[i].intersects(box(*b))]
        if len(local) < 2:
            continue
        # derive typed lines from adjacent segment pairs
        lines = []
        for i in range(len(local)):
            for j in range(i + 1, len(local)):
                gi, ci = local[i]
                gj, cj = local[j]
                if gi.distance(gj) > 0.3:
                    continue
                inter = gi.buffer(0.15).intersection(gj.buffer(0.15))
                if inter.is_empty or inter.area < 0.2:
                    continue
                # shared edge as the intersection's long axis
                mrr = inter.minimum_rotated_rectangle
                cc = list(mrr.exterior.coords)[:4]
                e = sorted(((np.hypot(cc[(k + 1) % 4][0] - cc[k][0],
                                      cc[(k + 1) % 4][1] - cc[k][1]), k)
                            for k in range(4)), reverse=True)
                L, k0 = e[0]
                if L < 1.2:
                    continue
                mid1 = ((np.array(cc[k0]) + np.array(cc[(k0 + 3) % 4])) / 2)
                mid2 = ((np.array(cc[(k0 + 1) % 4])
                         + np.array(cc[(k0 + 2) % 4])) / 2)
                shared = LineString([tuple(mid1), tuple(mid2)])
                az_i = AZ.get(str(ci).upper().strip())
                az_j = AZ.get(str(cj).upper().strip())
                try:
                    union = unary_union([gi, gj])
                    if union.geom_type != "Polygon":
                        union = max(union.geoms, key=lambda q: q.area)
                except Exception:
                    continue
                kind = classify_edge(az_i, az_j, shared, union)
                if kind:
                    lines.append((kind, shared))
        if not lines:
            continue
        # rasterize channels in pixel space
        def to_px(x, y):
            return ((x - b.left) / (b.right - b.left) * w,
                    (1 - (y - b.bottom) / (b.top - b.bottom)) * h)
        chans = []
        for kind in KINDS:
            im = Image.new("L", (w, h), 0)
            dr = ImageDraw.Draw(im)
            for k2, ln in lines:
                if k2 != kind:
                    continue
                (x1, y1), (x2, y2) = ln.coords[0], ln.coords[-1]
                dr.line([*to_px(x1, y1), *to_px(x2, y2)],
                        fill=255, width=LINE_WIDTH_PX)
            chans.append(np.array(im))
        mask = np.stack(chans, axis=-1)
        # weight: union of all roof segments in the tile
        wim = Image.new("L", (w, h), 0)
        wd = ImageDraw.Draw(wim)
        for g, _c in local:
            try:
                wd.polygon([to_px(*c) for c in g.exterior.coords], fill=255)
            except Exception:
                pass
        weight = np.array(wim)
        cw = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)  # no cliffs
        for py in range(0, h - PATCH + 1, STRIDE):
            for px in range(0, w - PATCH + 1, STRIDE):
                m = mask[py:py + PATCH, px:px + PATCH]
                if m.max() == 0:
                    continue
                np.savez_compressed(
                    OUT_DIR / f"rid_{ti}_{py}_{px}.npz",
                    image=rgb[py:py + PATCH, px:px + PATCH],
                    lines=m, weight=weight[py:py + PATCH, px:px + PATCH],
                    cw=cw)
                n_patch += 1
        if ti % 200 == 0:
            print(f"  {ti}/{len(tifs)} tiles, {n_patch} patches", flush=True)
    print(f"done: {n_patch} patches -> {OUT_DIR}")


if __name__ == "__main__":
    main()

"""
Draw what the segmenter thinks against what the labeller drew, on one roof.

The scorecard says 19 Camp Street agrees with Josh on 0% of its lines and flags
60 m2 of obstruction where he marked none. A number like that says something is
wrong but not what, and the difference between "it found a different roof",
"it found the right roof and mislabelled it" and "the labels are in the wrong
place" is not recoverable from a percentage.

So render it: imagery underneath, drawn lines by kind, predicted facet edges,
detected obstructions, marked obstructions. One picture per roof, and the
failure is usually obvious in it.

Usage:
    python tools/render_disagreement.py 5372566
    python tools/render_disagreement.py --worst 6      # lowest-F1 labelled roofs
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
OUT_DIR = ROOT / "data" / "disagreement"

KIND_COLOUR = {"ridge": "#1fff7a", "valley": "#35b6ff", "cliff": "#ff3b30"}
PAD_M = 4.0


def draw_one(bid, lab, ctx, pc, ax):
    import numpy as np
    import rasterio.windows
    from score_geometry import _line_points, _obs_ring
    from src.roof_segmentation import segment_building_best
    from src.obstruction_detection import detect_obstructions_combined

    geom = ctx["gdf"].loc[bid].geometry
    minx, miny, maxx, maxy = geom.bounds
    b = (minx - PAD_M, miny - PAD_M, maxx + PAD_M, maxy + PAD_M)

    if ctx["img"] is not None:
        w = rasterio.windows.from_bounds(*b, ctx["img"].transform)
        rgb = np.moveaxis(ctx["img"].read([1, 2, 3], window=w,
                                          boundless=True, fill_value=0), 0, -1)
        ax.imshow(rgb.astype("uint8"), extent=[b[0], b[2], b[1], b[3]])
    ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])

    x, y = geom.exterior.xy
    ax.plot(x, y, color="#ffd400", lw=1.6, label="outline")

    facets = segment_building_best(ctx["dsm"], pc, geom, bid,
                                   imagery_ds=ctx["img"]) or []
    for i, f in enumerate(facets):
        fx, fy = f["geometry"].exterior.xy
        ax.plot(fx, fy, color="#ffffff", lw=1.1, alpha=0.85,
                label="segmenter facets" if i == 0 else None)

    # obstructions the detector found
    n_found = 0
    for f in facets:
        if f.get("plane_a") is None:
            continue
        try:
            found = detect_obstructions_combined(
                ctx["img"], pc, f["geometry"],
                (f["plane_a"], f["plane_b"], f["plane_c"]),
                roof_geom=f.get("building_geometry")) or []
        except Exception:
            found = []
        for g in found:
            gg = g["geometry"] if isinstance(g, dict) else g
            polys = gg.geoms if gg.geom_type == "MultiPolygon" else [gg]
            for pgon in polys:
                px, py = pgon.exterior.xy
                ax.fill(px, py, color="#ff8a00", alpha=0.30,
                        label="detected obstruction" if n_found == 0 else None)
                n_found += 1

    # what was marked
    seen = set()
    for l in lab.get("lines", []):
        pts = _line_points(l)
        if not pts:
            continue
        k = l.get("kind", "ridge")
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=KIND_COLOUR.get(k, "#fff"), lw=2.6,
                path_effects=None, label=(f"drawn {k}" if k not in seen else None))
        seen.add(k)
    for j, o in enumerate(lab.get("obstructions", [])):
        ring = _obs_ring(o)
        if ring and len(ring) >= 3:
            ax.fill([p[0] for p in ring], [p[1] for p in ring],
                    facecolor="none", edgecolor="#c06bff", lw=1.8,
                    label="marked obstruction" if j == 0 else None)

    ax.set_title(f"#{bid}  {lab.get('address') or lab.get('area','')}",
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return len(facets), n_found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", type=int)
    ap.add_argument("--worst", type=int, default=0,
                    help="render the N lowest-F1 labelled roofs instead")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import rasterio
    from src.region_build import area_paths
    from src.pointcloud_source import PointCloudSource
    from score_geometry import (line_scores, predicted_lines_from_facets,
                                _line_points)
    from src.roof_segmentation import segment_building_best

    labels = json.loads(LABELS.read_text())["buildings"]
    pc = PointCloudSource()
    ctxs = {}

    def ctx_for(area):
        if area not in ctxs:
            p = area_paths(area)
            ctxs[area] = None if not (p["outlines"].exists() and p["dsm"].exists()) else {
                "gdf": gpd.read_file(p["outlines"]).set_index("building_id", drop=False),
                "dsm": rasterio.open(p["dsm"]),
                "img": rasterio.open(p["imagery"]) if p["imagery"].exists() else None,
            }
        return ctxs[area]

    ids = list(a.ids)
    if a.worst:
        scored = []
        for k, lab in labels.items():
            bid = int(k)
            ctx = ctx_for(lab.get("area"))
            if not ctx or bid not in ctx["gdf"].index:
                continue
            tl = [t for t in (_line_points(l) for l in lab.get("lines", [])) if t]
            if not tl:
                continue
            geom = ctx["gdf"].loc[bid].geometry
            try:
                f = segment_building_best(ctx["dsm"], pc, geom, bid,
                                          imagery_ds=ctx["img"]) or []
                s = line_scores(predicted_lines_from_facets(f, geom), tl, 0.75)
            except Exception:
                continue
            if s:
                scored.append((s["f1"], bid))
        scored.sort()
        ids = [b for _, b in scored[:a.worst]]
        print("lowest-F1 roofs:", ids)

    if not ids:
        print("give building ids, or --worst N")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = np.atleast_1d(axes).ravel() if n > 1 else [axes]

    for ax, bid in zip(axes, ids):
        lab = labels.get(str(bid))
        if not lab:
            continue
        ctx = ctx_for(lab.get("area"))
        if not ctx or bid not in ctx["gdf"].index:
            continue
        nf, no = draw_one(bid, lab, ctx, pc, ax)
        print(f"  #{bid}: {nf} facets, {no} detected obstructions, "
              f"{len(lab.get('lines', []))} drawn lines")
    for ax in axes[len(ids):]:
        ax.axis("off")
    axes[0].legend(fontsize=7, loc="upper right", framealpha=0.8)

    out = OUT_DIR / ("disagreement_" + "_".join(str(i) for i in ids[:4]) + ".png")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    import numpy as np  # noqa: E402  (used by main's axes handling)
    sys.exit(main())

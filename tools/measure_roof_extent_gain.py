"""
What would using the LiDAR roof extent instead of the surveyed outline actually
gain, in panels?

Area lost is not the same as capacity lost. An eave is 0.45 m and the panel edge
setback already eats 0.3 m of it, so most of the ubiquitous overhang is worth
nothing. The attached carports and verandas might be worth a great deal. The
only way to tell them apart is to run the real fitter both ways.

That is what this does: the same segmentation and the same panel fitting, once
with the LINZ outline and once with src.roof_extent.roof_region, on the roofs
Josh has actually marked. No pipeline code is modified -- the boundary is
substituted per call -- so this can run while a district build is in flight.

The number it prints is the one that decides whether a 16-hour rebuild is worth
it.

Usage:
    python tools/measure_roof_extent_gain.py
    python tools/measure_roof_extent_gain.py --ids 4719759 4735623
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
LABELS = DATA_DIR / "roof_labels.json"


def fit_all(facets, pc, imagery_ds, axis_ref=None):
    """Panels for one building, through the shipped fitter."""
    from src.obstruction_detection import detect_obstructions_combined
    from src.panel_fitting import fit_panels_on_facet
    total = 0
    for f in facets:
        if f.get("plane_a") is None:
            continue
        # EXTENT and AXIS REFERENCE are different questions, and panel_fitting
        # conflates them in one field. building_geometry decides where the roof
        # ends AND what angle to rack flat arrays at -- and for the second job
        # the surveyed outline is better precisely because it is crisp and
        # rectilinear, where a rasterised LiDAR boundary is not.
        if axis_ref is not None:
            f = dict(f)
            f["building_geometry"] = axis_ref
        try:
            obs = detect_obstructions_combined(
                imagery_ds, pc, f["geometry"],
                (f["plane_a"], f["plane_b"], f["plane_c"]),
                roof_geom=f.get("building_geometry"))
        except Exception:
            obs = []
        siblings = [o for o in facets if o is not f]
        try:
            total += len(fit_panels_on_facet(f, obstructions=obs,
                                             sibling_facets=siblings) or [])
        except Exception:
            pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    a = ap.parse_args()

    import geopandas as gpd
    import rasterio
    from shapely.ops import unary_union
    from src.region_build import area_paths
    from src.roof_segmentation import segment_building_best
    from src.pointcloud_source import PointCloudSource
    from src.roof_extent import roof_region, PAD_M

    if not LABELS.exists():
        print("no labels yet")
        return 2
    labels = json.loads(LABELS.read_text())["buildings"]
    ids = a.ids or sorted(int(k) for k in labels)

    pc = PointCloudSource()
    ctxs = {}
    rows = []

    print(f"panels with the OUTLINE vs the LiDAR ROOF EXTENT, {len(ids)} roofs\n")
    print(f"{'building':>10}{'outline m2':>11}{'extent m2':>10}{'+area':>7}"
          f"{'now':>8}{'naive':>8}{'axis-fix':>8}   address")

    for bid in ids:
        lab = labels[str(bid)]
        area = lab.get("area")
        if area not in ctxs:
            p = area_paths(area)
            if not (p["outlines"].exists() and p["dsm"].exists()):
                continue
            ctxs[area] = {
                "gdf": gpd.read_file(p["outlines"]).set_index("building_id", drop=False),
                "dsm": rasterio.open(p["dsm"]),
                "img": rasterio.open(p["imagery"]) if p["imagery"].exists() else None,
            }
        ctx = ctxs[area]
        gdf = ctx["gdf"]
        if bid not in gdf.index:
            continue
        outline = gdf.loc[bid].geometry

        near = gdf[gdf.geometry.intersects(outline.buffer(PAD_M + 1))]
        near = near[near["building_id"] != bid]
        others = unary_union(list(near.geometry)) if len(near) else None

        try:
            extent = roof_region(pc, outline, neighbours=others)
        except Exception as e:
            print(f"{bid:>10}   roof_region failed: {type(e).__name__}: {e}")
            continue

        try:
            f_now = segment_building_best(ctx["dsm"], pc, outline, bid,
                                          imagery_ds=ctx["img"]) or []
            p_now = fit_all(f_now, pc, ctx["img"])
            if extent.equals(outline):
                f_new, p_new, p_axis = f_now, p_now, p_now
            else:
                f_new = segment_building_best(ctx["dsm"], pc, extent, bid,
                                              imagery_ds=ctx["img"]) or []
                p_new = fit_all(f_new, pc, ctx["img"])
                p_axis = fit_all(f_new, pc, ctx["img"], axis_ref=outline)
        except Exception as e:
            print(f"{bid:>10}   fit failed: {type(e).__name__}: {e}")
            continue

        rows.append({"id": bid, "a_now": outline.area, "a_new": extent.area,
                     "p_now": p_now, "p_new": p_new, "p_axis": p_axis})
        print(f"{bid:>10}{outline.area:>11.0f}{extent.area:>10.0f}"
              f"{100 * (extent.area / outline.area - 1):>6.0f}%"
              f"{p_now:>8}{p_new:>+8}{p_axis:>+8}   {(lab.get('address') or '')[:22]}")

    if not rows:
        print("nothing measured")
        return 1

    an, aw = sum(r["a_now"] for r in rows), sum(r["a_new"] for r in rows)
    pn, pw = sum(r["p_now"] for r in rows), sum(r["p_new"] for r in rows)
    pa = sum(r["p_axis"] for r in rows)
    gained = [r for r in rows if r["p_axis"] > r["p_now"]]
    lost = [r for r in rows if r["p_axis"] < r["p_now"]]
    lost_naive = [r for r in rows if r["p_new"] < r["p_now"]]

    print(f"\nover {len(rows)} buildings:")
    print(f"  boundary area   {an:>8.0f} -> {aw:>8.0f} m2  "
          f"({100 * (aw / an - 1):+.1f}%)")
    print(f"  panels, outline today          {pn:>8}")
    print(f"  panels, extent (naive)         {pw:>8}  "
          f"({100 * (pw / pn - 1) if pn else 0:+.1f}%)   "
          f"lost on {len(lost_naive)} buildings")
    print(f"  panels, extent + outline axis  {pa:>8}  "
          f"({100 * (pa / pn - 1) if pn else 0:+.1f}%)   "
          f"lost on {len(lost)} buildings")
    print(f"  gained on {len(gained)} buildings, lost on {len(lost)}, "
          f"unchanged on {len(rows) - len(gained) - len(lost)}")
    if lost:
        print("\n  still losing after the axis fix -- these are worth reading:")
        for r in sorted(lost, key=lambda r: r["p_axis"] - r["p_now"])[:5]:
            print(f"    #{r['id']}: {r['p_now']} -> {r['p_axis']}")
    print(f"\n  Area is the easy number; panels are the one that matters, "
          f"because the\n  0.3 m edge setback already consumes most of a "
          f"0.45 m eave.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

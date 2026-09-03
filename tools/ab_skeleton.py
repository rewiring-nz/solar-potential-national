"""
Isolate the skeleton-roof reconstruction and measure what it costs or buys.

WHY THIS EXISTS. Panels crossing Josh's drawn lines went from 20.4% (30 Aug
build) to 23.8% (1 Sep build). The obvious suspect was the vision-cut work, and
the obvious suspect is wrong: the 1 Sep build was written three minutes before
the vision seam was even committed, so imagery cuts were not running in either
build. The only substantial pipeline geometry change in that window is
612c095 -- skeleton-roof reconstruction, touching roof_partition,
roof_segmentation and roof_skeleton.

That commit moved three things at once, which is why it needs isolating rather
than arguing about. roof_segmentation picks the skeleton whenever it explains
the roof within SKELETON_TIE_MARGIN (0.05) of the partition's score, so setting
that margin unreachable turns the skeleton off and changes nothing else. Same
code, same points, same panel fitter, one knob.

TRUTH IS UNCHANGED AND INDEPENDENT of the knob: panels measured against the
lines Josh drew, exactly as tools/measure_panel_crossings.py does it. Both arms
are scored the same way, so whatever the arms disagree about is the skeleton.

WHAT WOULD MAKE THIS MEANINGLESS. Running it on roofs where the skeleton never
wins: the two arms would be identical and the answer would be a confident zero.
So the number of roofs where the choice actually differed is reported, and it is
the first thing to read -- a result over 4 roofs is not a result.

Usage:
    python tools/ab_skeleton.py --limit 25
    python tools/ab_skeleton.py --ids 5372566
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
GRAZE_M2 = 0.10 * 0.10
SKIP_FLAGS = {"absent", "not_building", "unclear"}


def _crossings(panels, lines_union):
    n = 0
    for p in panels:
        if not p.intersects(lines_union):
            continue
        try:
            pieces = p.difference(lines_union.buffer(0.01))
        except Exception:
            continue
        geoms = list(getattr(pieces, "geoms", [pieces]))
        if len([g for g in geoms if g.area > GRAZE_M2]) >= 2:
            n += 1
    return n


def build_roof(ctx, geom, bid, pc):
    """Segment + fit panels for one roof, with whatever settings are live."""
    from src.roof_segmentation import segment_building_best
    from src.obstruction_detection import detect_obstructions_combined
    from src.panel_fitting import fit_panels_on_facet

    facets = segment_building_best(ctx["dsm"], pc, geom, bid,
                                   imagery_ds=ctx["img"]) or []
    panels = []
    for f in facets:
        if f.get("plane_a") is None:
            continue
        try:
            obs = detect_obstructions_combined(
                ctx["img"], pc, f["geometry"],
                (f["plane_a"], f["plane_b"], f["plane_c"]),
                roof_geom=f.get("building_geometry")) or []
        except Exception:
            obs = []
        sib = [o for o in facets if o is not f]
        try:
            for pnl in (fit_panels_on_facet(f, obstructions=obs,
                                            sibling_facets=sib) or []):
                panels.append(pnl["geometry"] if isinstance(pnl, dict) else pnl)
        except Exception:
            pass
    return facets, panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--limit", type=int, default=25)
    a = ap.parse_args()

    import geopandas as gpd
    import rasterio
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    from src.region_build import area_paths
    from src.pointcloud_source import PointCloudSource
    import src.roof_segmentation as RS
    from triage_roofs import _drawn_lines

    labels = json.loads(LABELS.read_text())["buildings"]
    ids = a.ids or sorted(
        int(k) for k, v in labels.items()
        if v.get("complete") and v.get("problem") not in SKIP_FLAGS)
    if a.limit:
        ids = ids[:a.limit]

    pc = PointCloudSource()
    ctxs = {}
    tot = {"on": [0, 0], "off": [0, 0]}      # [panels, crossing]
    differed = 0
    rows = []

    original_margin = RS.SKELETON_TIE_MARGIN

    for bid in ids:
        lab = labels.get(str(bid))
        if not lab:
            continue
        segs = _drawn_lines(lab)
        if not segs:
            continue
        area = lab.get("area")
        if area not in ctxs:
            p = area_paths(area)
            ctxs[area] = None if not (p["outlines"].exists()
                                      and p["dsm"].exists()) else {
                "gdf": gpd.read_file(p["outlines"]).set_index("building_id",
                                                             drop=False),
                "dsm": rasterio.open(p["dsm"]),
                "img": rasterio.open(p["imagery"]) if p["imagery"].exists()
                else None,
            }
        ctx = ctxs[area]
        if not ctx or bid not in ctx["gdf"].index:
            continue
        geom = ctx["gdf"].loc[bid].geometry
        lines = unary_union([LineString(s) for s in segs])

        res = {}
        for arm, margin in (("on", original_margin), ("off", -99.0)):
            RS.SKELETON_TIE_MARGIN = margin
            try:
                facets, panels = build_roof(ctx, geom, bid, pc)
            except Exception:
                res[arm] = None
                continue
            res[arm] = (len(facets), len(panels), _crossings(panels, lines))
        RS.SKELETON_TIE_MARGIN = original_margin

        if not res.get("on") or not res.get("off"):
            continue
        if res["on"] != res["off"]:
            differed += 1
        for arm in ("on", "off"):
            tot[arm][0] += res[arm][1]
            tot[arm][1] += res[arm][2]
        rows.append((bid, res["on"], res["off"]))

    if not rows:
        print("nothing measured")
        return 1

    print(f"\nSKELETON RECONSTRUCTION, ON vs OFF, over {len(rows)} labelled roofs")
    print(f"\n  roofs where the two arms actually differed: {differed}")
    if differed < 5:
        print("  -> too few to conclude anything. The skeleton rarely won on")
        print("     this set, so any difference below is noise, not evidence.")
    print()
    for arm, name in (("on", "skeleton ON  (current)"),
                      ("off", "skeleton OFF (pre-612c095)")):
        p, c = tot[arm]
        print(f"  {name:28s} {p:>6} panels, {c:>5} crossing "
              f"({100 * c / max(p, 1):.1f}%)")
    dp = tot["on"][0] - tot["off"][0]
    dc = (100 * tot["on"][1] / max(tot["on"][0], 1)
          - 100 * tot["off"][1] / max(tot["off"][0], 1))
    print(f"\n  skeleton changes panel count by {dp:+d} "
          f"and crossing rate by {dc:+.1f} points")
    print("\n  A change is only bad if crossings rise. Fewer panels with fewer")
    print("  crossings may be the skeleton correctly refusing bad placements.")
    if differed:
        print("\n  roofs where it mattered most:")
        worst = sorted((r for r in rows if r[1] != r[2]),
                       key=lambda r: -(r[1][2] - r[2][2]))[:6]
        for bid, on, off in worst:
            print(f"    #{bid}: ON {on[0]}f/{on[1]}p/{on[2]}x   "
                  f"OFF {off[0]}f/{off[1]}p/{off[2]}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())

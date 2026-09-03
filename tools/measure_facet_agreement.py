"""
Do the facets look like the roof Josh drew?

Panel-crossing rate (tools/measure_panel_crossings.py) asks whether panels
straddle a fold. This asks the question he actually raised looking at the map:
"you still got the facets wrong ... completely different to the mark up I gave
you". Those are different failures and only this one sees the second.

TWO DIRECTIONS, AND THE SECOND IS THE ONE THAT WAS MISSING. Measuring only
whether his lines are covered by some facet edge scores 7 Anderson Heights at
94.8% while the roof on screen is visibly a jumble -- because a partition can
find every line he drew AND add a dozen he did not. So:

  FOUND      how much of his drawn line length has a facet edge running within
             TOL of it. Low means creases were missed.
  UNDRAWN    how much of the INTERIOR facet-edge length has no drawn line near
             it. Edges along the building outline are excluded, since those are
             the footprint rather than anything he claimed. High means the roof
             was chopped up along lines he never saw.
  CLUTTER    total interior edge length. Two partitions can both score well and
             one of them draws twice as many metres of line to do it.

A roof is only right when FOUND is high and UNDRAWN is low. Reporting either
alone is how a fragmented roof passes for a good one.

Usage:
    python tools/measure_facet_agreement.py
    python tools/measure_facet_agreement.py --ids 5371108 --ab
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
TOL_M = 1.0                 # hand-drawn lines carry this much slop
OUTLINE_TOL_M = 0.6         # an edge this close to the footprint is the footprint
SKIP = {"absent", "not_building", "unclear"}


def agreement(faces, segs, outline, tol=TOL_M):
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    if not faces or not segs:
        return None
    edges = unary_union([f["geometry"].boundary for f in faces])
    interior = edges.difference(outline.boundary.buffer(OUTLINE_TOL_M))
    drawn = unary_union([LineString([(s[0], s[1]), (s[2], s[3])]) for s in segs])
    tot = cov = 0.0
    for s in segs:
        ls = LineString([(s[0], s[1]), (s[2], s[3])])
        tot += ls.length
        cov += ls.intersection(edges.buffer(tol)).length
    ilen = interior.length
    undrawn = interior.difference(drawn.buffer(tol)).length
    return {"found": cov / max(tot, 1e-9),
            "undrawn": undrawn / max(ilen, 1e-9),
            "clutter_m": ilen, "facets": len(faces)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ab", action="store_true",
                    help="compare with Josh's lines wired in vs ignored")
    a = ap.parse_args()

    import geopandas as gpd
    import rasterio
    from src.region_build import area_paths
    from src.roof_segmentation import segment_building_best
    from src.pointcloud_source import PointCloudSource
    import src.roof_line_source as RLS

    labels = json.loads(LABELS.read_text())["buildings"]
    ids = a.ids or sorted(int(k) for k, v in labels.items()
                          if v.get("complete") and v.get("problem") not in SKIP)
    if a.limit:
        ids = ids[:a.limit]

    pc = PointCloudSource()
    orig = RLS.drawn_segments
    orig_faces = RLS.drawn_faces
    ctxs = {}
    # Both label-derived inputs are toggled together. Toggling only
    # drawn_segments once made an A/B report identical arms, because the path
    # under test read drawn_faces and never saw the switch.
    arms = (("OFF", lambda b: []), ("ON", orig)) if a.ab else (("ON", orig),)
    tot = {k: {"found": [], "undrawn": [], "clutter": [], "facets": []}
           for k, _ in arms}

    for bid in ids:
        lab = labels.get(str(bid))
        if not lab:
            continue
        segs = orig(bid)
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
                else None}
        ctx = ctxs[area]
        if not ctx or bid not in ctx["gdf"].index:
            continue
        geom = ctx["gdf"].loc[bid].geometry
        for name, fn in arms:
            RLS.drawn_segments = fn
            RLS.drawn_faces = (orig_faces if fn is orig else (lambda b: []))
            RLS._LABELS_CACHE[0] = None
            try:
                faces = segment_building_best(ctx["dsm"], pc, geom, bid,
                                              imagery_ds=ctx["img"]) or []
            except Exception:
                continue
            st = agreement(faces, segs, geom)
            if not st:
                continue
            tot[name]["found"].append(st["found"])
            tot[name]["undrawn"].append(st["undrawn"])
            tot[name]["clutter"].append(st["clutter_m"])
            tot[name]["facets"].append(st["facets"])
        RLS.drawn_segments = orig
        RLS.drawn_faces = orig_faces

    n = len(tot[arms[-1][0]]["found"])
    if not n:
        print("nothing measured")
        return 1
    print(f"\nFACET AGREEMENT WITH JOSH'S MARKUP, over {n} roofs he marked "
          f"complete\n")
    print(f"  {'':6s} {'your lines found':>17s} {'edges you did NOT draw':>24s} "
          f"{'interior edge m':>16s} {'facets':>8s}")
    for name, _ in arms:
        t = tot[name]
        if not t["found"]:
            continue
        f = 100 * sum(t["found"]) / len(t["found"])
        u = 100 * sum(t["undrawn"]) / len(t["undrawn"])
        c = sum(t["clutter"]) / len(t["clutter"])
        fc = sum(t["facets"]) / len(t["facets"])
        print(f"  {name:6s} {f:>16.1f}% {u:>23.1f}% {c:>16.0f} {fc:>8.1f}")
    print("\n  A roof is right only when the first number is high AND the")
    print("  second is low. 7 Anderson scored 94.8% on the first while looking")
    print("  like a jumble on screen -- every line he drew was found, and a")
    print("  dozen he never drew were added alongside them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

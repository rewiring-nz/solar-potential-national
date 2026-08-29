"""
Score every building's layout on the things that make it look wrong.

Josh found 21 bad buildings by clicking around for an afternoon. Every one
was real, and the audit scripts we already had (edge overlap, facet escape,
lumpy, z-split) flagged none of them, because those measure GEOMETRIC
correctness -- is the panel inside its facet, is the surface under it flat --
and what he was reacting to was different: roofs that are half empty, arrays
broken into confetti, two angles on one roof, obstructions eating a third of
a building. This measures those instead.

Per building:
  fill        panel plan area / resolved roof area. Median across the
              district was 54% when this was written.
  islands     connected groups of touching panels. A real install is one or
              two compact arrays; twelve islands is confetti.
  angles      distinct panel bearings (5-degree bins). More than one on a
              building usually means facets disagreed about which way to
              rack -- Josh on 26 Isle St, "it should just fill consistently
              in the same direction".
  obstr       obstruction area as a share of resolved roof. Over ~35% is
              almost always over-carve, not equipment.
  stranded    panels not part of any group of >= MIN_CLEAN_ARRAY panels --
              the lone singles and pairs that read as noise.

Usage:
  python src/audit_layout_quality.py [--area NAME] [--top 25]
  python src/audit_layout_quality.py --worst fill      # rank by one metric
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyproj
from shapely.geometry import shape
from shapely.ops import transform as shp_transform, unary_union
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.region_build import DATA_DIR, area_paths

TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True).transform
MIN_CLEAN_ARRAY = 3      # a group smaller than this reads as stragglers, not an array
TOUCH_TOL_M = 0.35       # panels this close count as the same array (tile gaps are 4cm)
ANGLE_BIN_DEG = 5


def _bearing(poly):
    """Bearing of the panel's long edge, 0-180."""
    c = list(poly.exterior.coords)
    best, blen = 0.0, -1.0
    for i in range(min(4, len(c) - 1)):
        dx, dy = c[i + 1][0] - c[i][0], c[i + 1][1] - c[i][1]
        d = dx * dx + dy * dy
        if d > blen:
            blen, best = d, np.degrees(np.arctan2(dx, dy)) % 180
    return best


def _islands(panels):
    """Connected components over 'panels touch within TOUCH_TOL_M'."""
    if not panels:
        return []
    tree = STRtree(panels)
    seen, groups = set(), []
    for i in range(len(panels)):
        if i in seen:
            continue
        stack, group = [i], []
        seen.add(i)
        while stack:
            k = stack.pop()
            group.append(k)
            for j in tree.query(panels[k].buffer(TOUCH_TOL_M)):
                j = int(j)
                if j not in seen and panels[k].buffer(TOUCH_TOL_M).intersects(panels[j]):
                    seen.add(j)
                    stack.append(j)
        groups.append(group)
    return groups


def audit(path, limit=None):
    d = json.loads(Path(path).read_text())
    by = defaultdict(lambda: {"panel": [], "facet": [], "obstruction": []})
    for f in d["features"]:
        k = f["properties"].get("kind")
        if k in ("panel", "facet", "obstruction"):
            by[f["properties"]["building_id"]][k].append(f["geometry"])

    rows = []
    for bid, g in by.items():
        if not g["facet"]:
            continue
        try:
            facets = unary_union([shp_transform(TO_NZTM, shape(x)).buffer(0) for x in g["facet"]])
            roof = facets.area
            if roof < 40:            # sheds and garages are not the problem here
                continue
            panels = [shp_transform(TO_NZTM, shape(x)).buffer(0) for x in g["panel"]]
            obst = unary_union([shp_transform(TO_NZTM, shape(x)).buffer(0)
                                for x in g["obstruction"]]) if g["obstruction"] else None
            groups = _islands(panels)
            big = [gr for gr in groups if len(gr) >= MIN_CLEAN_ARRAY]
            stranded = sum(len(gr) for gr in groups if len(gr) < MIN_CLEAN_ARRAY)
            angles = {int(_bearing(p) // ANGLE_BIN_DEG) for p in panels}
            rows.append({
                "building_id": int(bid),
                "roof_m2": round(roof, 1),
                "panels": len(panels),
                "fill": round(unary_union(panels).area / roof, 3) if panels else 0.0,
                "islands": len(groups),
                "clean_arrays": len(big),
                "stranded": stranded,
                "angles": len(angles),
                "obstr": round(obst.intersection(facets).area / roof, 3) if obst is not None else 0.0,
            })
        except Exception:
            continue
        if limit and len(rows) >= limit:
            break
    return rows


def report(rows, top=25, worst="fill"):
    if not rows:
        print("nothing to audit")
        return
    fill = np.array([r["fill"] for r in rows])
    isl = np.array([r["islands"] for r in rows])
    ang = np.array([r["angles"] for r in rows])
    ob = np.array([r["obstr"] for r in rows])
    st = np.array([r["stranded"] for r in rows])
    print(f"{len(rows):,} buildings with >=40 m2 of resolved roof\n")
    print(f"  fill      median {np.median(fill):5.0%}   below 40%: {(fill < 0.4).sum():5,}"
          f"  ({100 * (fill < 0.4).mean():4.1f}%)")
    print(f"  islands   median {np.median(isl):5.0f}   more than 3: {(isl > 3).sum():5,}"
          f"  ({100 * (isl > 3).mean():4.1f}%)")
    print(f"  angles    median {np.median(ang):5.0f}   more than 1: {(ang > 1).sum():5,}"
          f"  ({100 * (ang > 1).mean():4.1f}%)")
    print(f"  obstr     median {np.median(ob):5.1%}   over 35%:    {(ob > 0.35).sum():5,}"
          f"  ({100 * (ob > 0.35).mean():4.1f}%)")
    print(f"  stranded  total  {st.sum():5,} panels in groups of < {MIN_CLEAN_ARRAY}"
          f"  ({100 * st.sum() / max(sum(r['panels'] for r in rows), 1):.1f}% of all panels)")

    key = {"fill": lambda r: r["fill"], "islands": lambda r: -r["islands"],
           "angles": lambda r: -r["angles"], "obstr": lambda r: -r["obstr"],
           "stranded": lambda r: -r["stranded"]}[worst]
    print(f"\nworst {top} by {worst}:")
    print(f"  {'building':>10}  {'roof m2':>8} {'panels':>6} {'fill':>5} {'isl':>4} "
          f"{'ang':>4} {'obstr':>6} {'strand':>6}")
    for r in sorted(rows, key=key)[:top]:
        print(f"  {r['building_id']:>10}  {r['roof_m2']:8.0f} {r['panels']:6d} {r['fill']:5.0%} "
              f"{r['islands']:4d} {r['angles']:4d} {r['obstr']:6.0%} {r['stranded']:6d}")


def main():
    argv = sys.argv[1:]
    area = top = None
    worst = "fill"
    if "--area" in argv:
        area = argv[argv.index("--area") + 1]
    if "--top" in argv:
        top = int(argv[argv.index("--top") + 1])
    if "--worst" in argv:
        worst = argv[argv.index("--worst") + 1]
    path = area_paths(area)["panel_layouts"] if area else DATA_DIR / "panel_layouts.geojson"
    print(f"auditing {path}\n")
    rows = audit(path)
    (DATA_DIR / "audit_layout_quality.json").write_text(json.dumps(rows))
    report(rows, top=top or 25, worst=worst)


if __name__ == "__main__":
    main()

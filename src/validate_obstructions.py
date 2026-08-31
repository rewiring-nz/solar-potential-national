"""
Score obstruction detection against a LABELLED set, both directions at once.

Two previous attempts at this were reverted because they were tuned on one
symptom: fix the roof that misses its plant, and the validated equipment
reference collapses; fix the roof that carves 47% of itself away, and real
ducting stops being detected. The detector fails in BOTH directions and a
change is only good if it moves both.

So the set carries three kinds of case, every one of them from a real report:

  OVER_CARVE     obstruction covers a large share of a roof that is mostly
                 clear. Score = obstruction area / roof area; LOWER is better.
  UNDER_DETECT   real plant that panels are being placed on top of. Score =
                 panels sitting on LiDAR that stands >0.25m proud of the roof
                 plane with most returns above it; LOWER is better.
  REFERENCE      known-good detections that must survive. Score = detected
                 obstruction area; must stay near its baseline.

Usage:
  python src/validate_obstructions.py              # score everything
  python src/validate_obstructions.py --save base  # record a baseline to diff against
"""

import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from shapely.ops import unary_union
from shapely.strtree import STRtree

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.region_build import DATA_DIR, all_areas, area_paths
from src.roof_segmentation import segment_building_best
from src.pointcloud_source import PointCloudSource
from src.obstruction_detection import detect_obstructions_combined
from src.panel_fitting import fit_panels_on_facet
from src.audit_layouts import plane_from_facet_points

OVER_CARVE = {
    4735250: "121 Hallenstein St -- 47% of roof carved, roof is mostly clear",
    4735242: "5 Church St -- 34% carved",
    4679079: "1/12 Hawthorne Dr -- 37.7% carved, flat white section",
    4735275: "11B Henry St -- spurious blobs breaking the array",
    5372585: "45 Camp St -- 'unnecessary obstructions disrupting a clean array'",
    4734932: "found by audit -- 61% of roof carved",
    5370372: "found by audit -- 52% of roof carved",
    # HEIGHT-derived over-carve. The colour cap does not touch these: their
    # obstruction is 139-279 m2 of height evidence, and the cause is upstream
    # -- segmentation fitting ONE plane across a multi-gabled roof, so most of
    # the surface legitimately reads as "above the plane". Fixing it by
    # thresholding obstruction area is the trap: the validated reference roof
    # is 45% of its own facet and 17 Cardigan St is 63%, so any cap that
    # separates them is sitting between two real numbers with nothing in
    # between. The fix belongs in segmentation.
    4719759: "17 Cardigan St -- 140 m2 of a 222 m2 roof, multi-gabled house",
    4730591: "2 Hawthorne Dr -- 279 m2 height obstruction",
    4729642: "55 Arrowtown-Lake Hayes Rd -- 175 m2 height obstruction",
}
UNDER_DETECT = {
    5371121: "1 Earl St -- 11 panels on structure up to 0.95m proud",
    4734769: "28 Rees St -- missed ducting (bug doc #4)",
    5370360: "17 Marine Pde -- panels placed OVER real obstructions (bug doc #6)",
    5372587: "35 Shotover St -- 'panels clearly overlapping all sorts of obstructions'",
}
REFERENCE = {
    5370338: "equipment reference -- 223 m2 of genuine rooftop ducting, must survive",
}

RAISED_MARGIN_M = 0.25
RAISED_FRACTION = 0.6
_CTX = {}
_AREA_OF = {}


def _area_of(bid):
    if bid in _AREA_OF:
        return _AREA_OF[bid]
    for area in all_areas():
        p = area_paths(area)["panel_layouts"]
        if p.exists():
            t = p.read_text()
            if f'"building_id": {bid}' in t or f'"building_id":{bid}' in t:
                _AREA_OF[bid] = area
                return area
    _AREA_OF[bid] = None
    return None


def _ctx(area):
    if area in _CTX:
        return _CTX[area]
    paths = area_paths(area)
    dd = paths["dir"] / "building_outlines_dedup.geojson"
    gdf = gpd.read_file(dd if dd.exists() else paths["outlines"]).set_index("building_id", drop=False)
    dsm = rasterio.open(paths["dsm"])
    img = rasterio.open(paths["imagery"]) if paths["imagery"].exists() else None
    _CTX[area] = {"gdf": gdf, "dsm": dsm, "img": img, "pc": PointCloudSource()}
    return _CTX[area]


def score(bid):
    area = _area_of(bid)
    if area is None:
        return None
    c = _ctx(area)
    facets = segment_building_best(c["dsm"], c["pc"], c["gdf"].loc[bid].geometry, bid,
                                   imagery_ds=c["img"])
    if not facets:
        return None
    roof = unary_union([f["geometry"] for f in facets])
    all_obst, all_panels, planes, geoms = [], [], [], []
    for f in facets:
        plane = (f["plane_a"], f["plane_b"], f["plane_c"])
        ob = detect_obstructions_combined(c["img"], c["pc"], f["geometry"], plane,
                                          roof_geom=f.get("building_geometry"))
        all_obst.extend(ob)
        all_panels.extend(fit_panels_on_facet(f, obstructions=ob,
                                              sibling_facets=[o for o in facets if o is not f]))
        geoms.append(f["geometry"])
        minx, miny, maxx, maxy = f["geometry"].bounds
        pts = c["pc"].points_in_bbox(minx, miny, maxx, maxy, building_only=True)
        pl = None
        if len(pts) >= 12:
            inside = shapely.contains_xy(f["geometry"], pts[:, 0], pts[:, 1])
            fp = pts[inside]
            if len(fp) >= 12:
                pl = plane_from_facet_points(fp)
        planes.append(pl)

    ob_area = unary_union([o.buffer(0) for o in all_obst]).intersection(roof).area if all_obst else 0.0
    tree = STRtree(geoms) if geoms else None
    raised = 0
    for p in all_panels:
        g = p["geometry"]
        fi = None
        if tree is not None:
            for idx in tree.query(g):
                if geoms[int(idx)].contains(g.centroid):
                    fi = int(idx)
                    break
        if fi is None or planes[fi] is None or planes[fi][2] is None:
            continue
        minx, miny, maxx, maxy = g.bounds
        pts = c["pc"].points_in_bbox(minx, miny, maxx, maxy, building_only=True)
        if len(pts) < 6:
            continue
        inside = shapely.contains_xy(g, pts[:, 0], pts[:, 1])
        pp = pts[inside]
        if len(pp) < 6:
            continue
        x0, y0, cf = planes[fi]
        res = pp[:, 2] - (cf[0] * (pp[:, 0] - x0) + cf[1] * (pp[:, 1] - y0) + cf[2])
        if float(np.percentile(res, 75)) > RAISED_MARGIN_M and \
           float((res > RAISED_MARGIN_M).mean()) > RAISED_FRACTION:
            raised += 1
    return {"roof_m2": roof.area, "obstr_m2": ob_area,
            "obstr_frac": ob_area / max(roof.area, 1e-9),
            "panels": len(all_panels), "panels_on_raised": raised}


def main():
    save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    base_path = DATA_DIR / "obstruction_validation_baseline.json"
    base = json.loads(base_path.read_text()) if base_path.exists() else {}
    out = {}
    for label, group in (("OVER-CARVE  (want obstr% LOW)", OVER_CARVE),
                         ("UNDER-DETECT (want raised LOW)", UNDER_DETECT),
                         ("REFERENCE   (must survive)", REFERENCE)):
        print(f"\n{label}")
        for bid, why in group.items():
            r = score(bid)
            if r is None:
                print(f"  #{bid}  not found")
                continue
            out[str(bid)] = r
            b = base.get(str(bid))
            d = ""
            if b:
                d = (f"   [was obstr {b['obstr_frac']:.0%}, raised {b['panels_on_raised']}, "
                     f"panels {b['panels']}]")
            print(f"  #{bid}  roof {r['roof_m2']:6.0f} m2   obstr {r['obstr_frac']:4.0%} "
                  f"({r['obstr_m2']:6.1f} m2)   panels {r['panels']:4d}   "
                  f"on-raised {r['panels_on_raised']:3d}   {why}{d}")
    if save:
        base_path.write_text(json.dumps(out))
        print(f"\nbaseline saved to {base_path.name}")


if __name__ == "__main__":
    main()

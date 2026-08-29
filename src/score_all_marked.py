"""Score the CURRENT pipeline against every roof Josh has marked. Run before
any push: it is the regression gate that stops the 'count matched, structure
wrong' trap from shipping again.

Usage: python src/score_all_marked.py
"""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, geopandas as gpd, rasterio, shapely.vectorized
from shapely.geometry import Polygon, box, Point
from shapely.ops import unary_union
from src.region_build import area_paths
from src.pointcloud_source import PointCloudSource
from src.roof_segmentation import segment_building_best
from src.obstruction_detection import detect_obstructions_combined
from src.roof_partition import _points_in, _inlier_fraction

TRUTH = json.load(open(Path(__file__).resolve().parent.parent / "data/roof_truth.json"))
AREA_OF = {4719759: "arrowtown_millbrook"}     # everything else is pilot

rows, warns = [], []
_ctx = {}
def ctx(area):
    if area not in _ctx:
        p = area_paths(area)
        _ctx[area] = (gpd.read_file(p["outlines"]).set_index("building_id", drop=False),
                      rasterio.open(p["dsm"]), rasterio.open(p["imagery"]))
    return _ctx[area]

pc = PointCloudSource()
for r in TRUTH["roofs"]:
    bid = r["building_id"]
    exp = r.get("faces")
    area = AREA_OF.get(bid, "pilot")
    gdf, dsm, img = ctx(area)
    if bid not in gdf.index:
        warns.append(f"{bid} not in {area}"); continue
    g = gdf.loc[bid].geometry
    facets = segment_building_best(dsm, pc, g, bid, imagery_ds=img)
    roof = unary_union([f["geometry"] for f in facets]) if facets else None
    mnx, mny, mxx, mxy = g.bounds
    raw = pc.points_in_bbox(mnx-1, mny-1, mxx+1, mxy+1, building_only=True)
    pts = raw[shapely.vectorized.contains(g, raw[:, 0], raw[:, 1])]
    fits = []
    for f in facets:
        pl = np.array([f["plane_a"], f["plane_b"], f["plane_c"]])
        sub = _points_in(f["geometry"], pts)
        if len(sub) >= 5:
            fits.append((_inlier_fraction(sub, pl), f["geometry"].area))
    wtd = sum(a*b for a, b in fits)/max(sum(b for _, b in fits), 1e-9) if fits else 0
    worst = min(a for a, _ in fits) if fits else 0

    obs = []
    for f in facets:
        obs += detect_obstructions_combined(img, pc, f["geometry"],
                (f["plane_a"], f["plane_b"], f["plane_c"]),
                roof_geom=f.get("building_geometry"))
    carve = (unary_union([o.buffer(0) for o in obs]).intersection(roof).area
             if obs and roof is not None else 0.0)

    # structure checks where traced geometry exists (Josh's metric: AREA, not counts)
    extra = ""
    t = r.get("traced_lines_nztm")
    if t and "middle_section_corners" in t:
        band = Polygon(t["middle_section_corners"])
        spanning = sum(1 for f in facets
                       if f["geometry"].intersection(band).area > 0.5
                       and f["geometry"].area > 2 * band.area)
        extra = f" mid-span={spanning}"
    to = r.get("traced_obstructions_nztm")
    if to:
        marked = unary_union([box(*v) for k, v in to.items() if not k.startswith("_")])
        got = (unary_union([o.buffer(0) for o in obs]).intersection(marked).area
               if obs else 0.0)
        extra += f" marked-ob-found={got/max(marked.area,1e-9):.0%}"
    rows.append((bid, r.get("address", "?")[:20], exp, len(facets), wtd, worst,
                 carve, roof.area if roof is not None else 0, extra))

print(f"{'building':>9} {'address':<20} {'faces':>9} {'wtd fit':>8} {'worst':>6} "
      f"{'carve':>6}  notes")
fails = 0
for bid, addr, exp, got, wtd, worst, carve, ra, extra in rows:
    ok = exp is None or abs(got - exp) <= 2
    if not ok: fails += 1
    print(f"{bid:>9} {addr:<20} {got:>3}/{'?' if exp is None else exp:<3}"
          f"{'' if ok else '!'}  {wtd:>7.1%} {worst:>6.0%} "
          f"{carve/max(ra,1e-9):>6.1%} {extra}")
for w in warns: print("  WARN:", w)
print(f"\n{len(rows)} marked roofs scored; {fails} outside +/-2 of Josh's count.")
print("Compare against the last committed run in data/marked_scores.json before pushing.")
prev = Path("data/marked_scores.json")
cur = {str(b): {"faces": g, "wtd": round(w, 3), "worst": round(wo, 3),
                "carve": round(c/max(ra2,1e-9), 3)}
       for b, _, _, g, w, wo, c, ra2, _ in rows}
if prev.exists():
    old = json.load(open(prev))
    for k, v in cur.items():
        if k in old:
            dw = v["wtd"] - old[k]["wtd"]
            if dw < -0.03:
                print(f"  REGRESSION {k}: weighted fit {old[k]['wtd']:.1%} -> {v['wtd']:.1%}")
json.dump(cur, open(prev, "w"), indent=1)
print("scores saved to data/marked_scores.json")

"""
Score the shipped segmenter and the reconstruction against the labelled roofs.

This exists because every automatic metric available rewards cutting a roof
into more pieces -- off-plane residual always falls when you add a plane -- so
tuning against one trades roofs against each other indefinitely. These labels
are what is actually right, given by Josh.

Two kinds of label, because a count is not always honest:
  count       how many distinct PLANES the roof has. Scored as absolute error
              against each model. Planes, not polygons: one flat deck split by
              a lift overrun is three polygons and one plane.
  preference  which model got closer, for roofs too complicated to count.
              Scored as a win.

Usage: python src/score_labels.py [--area pilot]
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyproj
import shapely.vectorized
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pointcloud_source import PointCloudSource
from src.region_build import area_paths
from src.roof_reconstruct import n_planes, reconstruct

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True).transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="pilot")
    a = ap.parse_args()

    labels = json.loads((DATA_DIR / f"roof_labels_{a.area}.json").read_text())
    raw = json.loads(area_paths(a.area)["outlines"].read_text())
    outl = {}
    for f in raw["features"]:
        g = shape(f["geometry"])
        outl[int(f["properties"]["building_id"])] = (
            shp_transform(TO_NZTM, g) if abs(g.bounds[0]) <= 180 else g).buffer(0)
    lay = json.loads(area_paths(a.area)["panel_layouts"].read_text())
    shipped = defaultdict(int)
    for f in lay["features"]:
        if f["properties"].get("kind") == "facet":
            shipped[int(f["properties"]["building_id"])] += 1
    sp = json.loads(area_paths(a.area)["solar_potential"].read_text())
    addr = {int(f["properties"]["building_id"]): f["properties"].get("address", "")
            for f in sp["features"]}

    pc = PointCloudSource()
    counted, prefs, excluded, pending = [], [], 0, 0
    for bid, lab in labels.items():
        if lab.get("exclude"):
            excluded += 1
            continue
        if not lab.get("confirmed"):
            pending += 1
            continue
        o = outl.get(int(bid))
        if o is None:
            continue
        minx, miny, maxx, maxy = o.bounds
        pts = pc.points_in_bbox(minx - 1, miny - 1, maxx + 1, maxy + 1, building_only=True)
        pts = pts[shapely.vectorized.contains(o.buffer(0.3), pts[:, 0], pts[:, 1])]
        try:
            fac, _ = reconstruct(int(bid), o, pts)
        except Exception as e:
            print(f"  {bid}: reconstruct failed ({type(e).__name__})")
            continue
        rec, ship = n_planes(fac), shipped[int(bid)]
        row = {"bid": int(bid), "addr": addr.get(int(bid), ""), "ship": ship,
               "rec": rec, "polys": len(fac), "label": lab.get("label", "")}
        if lab.get("kind") == "preference":
            row["prefer"] = lab["prefer"]
            prefs.append(row)
        elif lab.get("n") is not None:
            row["truth"] = lab["n"]
            # Some roofs have a genuine near-tie -- 34 Belfast Terrace has two
            # faces so nearly coplanar that Josh said 4 would not be wrong
            # either. Scoring that as a full miss would punish the right answer.
            row["alt"] = lab.get("n_alt")
            row["uncertain"] = bool(lab.get("uncertain"))
            counted.append(row)

    def miss(v, r):
        """Absolute error, forgiving an explicitly acceptable alternative."""
        e = abs(v - r["truth"])
        return min(e, abs(v - r["alt"])) if r.get("alt") is not None else e

    print(f"{'roof':<24}{'truth':>6}{'shipped':>9}{'polys':>7}{'planes':>8}   err ship/rec")
    es = er = 0
    for r in sorted(counted, key=lambda r: r["addr"]):
        es += miss(r["ship"], r)
        er += miss(r["rec"], r)
        mark = " ?" if r.get("uncertain") else ""
        if r.get("alt") is not None:
            mark += f" (or {r['alt']})"
        print(f"{(r['addr'] or r['bid']):<24}{r['truth']:>6}{r['ship']:>9}{r['polys']:>7}"
              f"{r['rec']:>8}   {r['ship']-r['truth']:+d} / {r['rec']-r['truth']:+d}{mark}")
    n = len(counted)
    if n:
        # Polygons scored alongside planes on purpose. assign_plane_ids fuses
        # pieces of one surface, which is right on 1 Memorial (10 polygons ->
        # 7 planes, exactly the truth) and wrong on 32 Park (8 -> 7, when 8 was
        # already correct). Neither count dominates, so hiding one would hide
        # where the remaining error actually lives.
        ep = sum(miss(r["polys"], r) for r in counted)
        exact_s = sum(1 for r in counted if miss(r["ship"], r) == 0)
        exact_r = sum(1 for r in counted if miss(r["rec"], r) == 0)
        exact_p = sum(1 for r in counted if miss(r["polys"], r) == 0)
        print(f"\n{n} counted roofs")
        print(f"  total absolute error   shipped {es:>3}    planes {er:>3}    polygons {ep:>3}")
        print(f"  mean absolute error    shipped {es/n:>5.2f}  planes {er/n:>5.2f}  polygons {ep/n:>5.2f}")
        print(f"  exact matches          shipped {exact_s}/{n}    planes {exact_r}/{n}    polygons {exact_p}/{n}")
        under = sum(1 for r in counted if r["rec"] < r["truth"] and miss(r["rec"], r) > 0)
        over = sum(1 for r in counted if r["rec"] > r["truth"] and miss(r["rec"], r) > 0)
        print(f"  reconstruction bias    {under} under, {over} over, {n-under-over} exact")
    if prefs:
        wr = sum(1 for r in prefs if r["prefer"] == "reconstruction")
        print(f"\n{len(prefs)} preference roofs: reconstruction preferred on {wr}, "
              f"shipped on {len(prefs)-wr}")
        for r in prefs:
            print(f"    {(r['addr'] or r['bid'])}: prefers {r['prefer']} "
                  f"(shipped {r['ship']}, reconstruction {r['rec']})")
    unc = sum(1 for r in counted if r.get("uncertain"))
    if unc:
        print(f"  ({unc} marked ? -- label given with reservations, treat as provisional)")
    print(f"\n{pending} still awaiting a label, {excluded} excluded")
    if n < 10:
        print("NOTE: too few labels to draw a conclusion from -- this is a running tally, "
              "not a result.")


if __name__ == "__main__":
    main()

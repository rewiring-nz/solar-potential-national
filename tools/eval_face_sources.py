"""Which reading of a roof is closest to the drawn faces?

Scores every candidate generator on the SAME held-out roofs, with the same
measure, so "the learned model is better" is a number and not an impression.

Per roof, per source:
    agree    mean over the drawn faces of the best IoU any candidate face
             achieves against it (the drawn face is the unit, so missing one
             costs in full and no amount of extra faces can pay it back)
    matched  drawn faces matched at IoU >= 0.5 -- "did it find this face"
    nface    how many faces the source claims, against how many were drawn;
             over-claiming is the recurring failure

Only roofs in data/bench_ids.txt are scored: those are pinned out of the
face-region model's training set, so the comparison is honest.

    python tools/eval_face_sources.py
    python tools/eval_face_sources.py --sources learned hypothesis
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon

from src.region_build import area_paths
from src.pointcloud_source import PointCloudSource

VOID = {"absent", "not_building", "unclear"}


def agree_and_match(cand, drawn):
    if not drawn:
        return None, None
    if not cand:
        return 0.0, 0.0
    tot = matched = 0.0
    for d in drawn:
        best = 0.0
        for c in cand:
            u = c.union(d).area
            if u > 0:
                best = max(best, c.intersection(d).area / u)
        tot += best
        matched += 1.0 if best >= 0.5 else 0.0
    return tot / len(drawn), matched / len(drawn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*",
                    default=["selected", "learned"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    B = json.loads((ROOT / "data/roof_labels.json").read_text())["buildings"]
    bench = [k for k in (ROOT / "data/bench_ids.txt").read_text().split()
             if k in B and B[k].get("problem") not in VOID
             and (B[k].get("faces") or [])]
    if a.limit:
        bench = bench[:a.limit]
    pc = PointCloudSource(max_cached_tiles=3)
    ctx = {}
    acc = {s: {"agree": [], "matched": [], "dn": [], "n": 0}
           for s in a.sources}
    print(f"{len(bench)} held-out marked roofs\n")
    for k in bench:
        bid = int(k)
        v = B[k]
        drawn = [Polygon([(x, y) for x, y in f["ring"]])
                 for f in v["faces"] if f.get("ring") and len(f["ring"]) >= 3]
        drawn = [d for d in drawn if d.is_valid and d.area > 1.0]
        if not drawn:
            continue
        ar = v.get("area")
        if ar not in ctx:
            p = area_paths(ar)
            dd = p["dir"] / "building_outlines_dedup.geojson"
            ctx[ar] = (gpd.read_file(dd if dd.exists() else p["outlines"])
                       .set_index("building_id", drop=False),
                       rasterio.open(p["imagery"])) \
                if p["imagery"].exists() else None
        if ctx[ar] is None:
            continue
        gdf, img = ctx[ar]
        if bid not in gdf.index:
            continue
        geom = gdf.loc[bid].geometry
        mnx, mny, mxx, mxy = geom.bounds
        pts = pc.points_in_bbox(mnx - 2, mny - 2, mxx + 2, mxy + 2,
                                building_only=True)
        for s in a.sources:
            cand = []
            if s == "learned":
                from src.face_regions import learned_faces
                cand = learned_faces(geom, img, pts)
            elif s == "selected":
                fp = ROOT / f"data/selected_faces/{bid}.json"
                if fp.exists():
                    cand = [Polygon(r) for r in
                            json.loads(fp.read_text())["faces"]]
            cand = [c for c in cand if c.is_valid and c.area > 0.5]
            ag, mt = agree_and_match(cand, drawn)
            if ag is None:
                continue
            acc[s]["agree"].append(ag)
            acc[s]["matched"].append(mt)
            acc[s]["dn"].append(len(cand) - len(drawn))
            acc[s]["n"] += 1
    print(f"{'source':12s} {'roofs':>5s} {'agree':>7s} {'matched':>8s} "
          f"{'faces vs drawn':>14s}")
    for s in a.sources:
        d = acc[s]
        if not d["n"]:
            print(f"{s:12s}     0")
            continue
        dn = np.array(d["dn"])
        print(f"{s:12s} {d['n']:5d} {np.mean(d['agree']):7.3f} "
              f"{np.mean(d['matched'])*100:7.1f}% "
              f"{np.mean(dn):+8.1f} (over {int((dn>0).sum())}, "
              f"under {int((dn<0).sum())})")


if __name__ == "__main__":
    sys.exit(main() or 0)

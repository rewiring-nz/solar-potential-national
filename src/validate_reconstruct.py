"""
Does reconstruction hold up on ORDINARY HOUSES?

The ten buildings the prototype was developed against were the ten worst in
the district -- a barrel vault, a terrace, a curved fan roof. Tuning on those
and shipping would be exactly the mistake this project has already made twice:
fix the roof that misses its plant, break the validated reference set.

Josh, 26 Aug: "households are the more important check, they are the rooftops
you have failed on more often." So this samples plain houses by roof area and
reports the DISTRIBUTION, not the winners.

Per building, shipped model vs reconstruction:
  off-plane   share of roof points more than 0.35m from the plane they sit on.
              The accuracy number -- it says whether the model describes the
              roof at all.
  coverage    share of the building outline the facets actually account for.
              The shipped segmenter leaves holes; roof it never resolved is
              roof that can never carry a panel.
  facets      count. Going up is not automatically bad (a hip roof IS four
              faces) but confetti is, so the median matters more than any one.

A change is only good if off-plane falls, coverage does not, and the facet
count stays sane. All three, or it is not an improvement.

Usage:
  python src/validate_reconstruct.py --area pilot --n 120
  python src/validate_reconstruct.py --area pilot --n 60 --min-roof 250 --max-roof 5000
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
from shapely.ops import transform as shp_transform, unary_union

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pointcloud_source import PointCloudSource
from src.region_build import area_paths
from src.roof_reconstruct import fit_plane, reconstruct, residuals

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True).transform
OFF_PLANE_M = 0.35


def off_plane_share(facets, pts):
    """Points more than OFF_PLANE_M from the plane of the facet containing them."""
    bad = tot = 0
    for f in facets:
        g = f["geometry"] if isinstance(f, dict) else f
        pp = pts[shapely.vectorized.contains(g, pts[:, 0], pts[:, 1])]
        if len(pp) < 8:
            continue
        if isinstance(f, dict) and "plane_a" in f:
            r = f["plane_a"] * pp[:, 0] + f["plane_b"] * pp[:, 1] + f["plane_c"] - pp[:, 2]
        else:
            # Shipped facets from the layout file carry no plane, so give them
            # the best plane they could have had. That FLATTERS the shipped
            # model; the comparison is deliberately not tilted our way.
            r = residuals(fit_plane(pp), pp)
        bad += int((np.abs(r) > OFF_PLANE_M).sum())
        tot += len(pp)
    return (bad / tot) if tot else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="pilot")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--min-roof", type=float, default=60.0)
    ap.add_argument("--max-roof", type=float, default=250.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lay = json.loads(area_paths(a.area)["panel_layouts"].read_text())
    shipped = defaultdict(list)
    for f in lay["features"]:
        if f["properties"].get("kind") == "facet":
            shipped[int(f["properties"]["building_id"])].append(
                shp_transform(TO_NZTM, shape(f["geometry"])).buffer(0))

    raw = json.loads(area_paths(a.area)["outlines"].read_text())
    outlines = {}
    for f in raw["features"]:
        g = shape(f["geometry"])
        if abs(g.bounds[0]) <= 180:
            g = shp_transform(TO_NZTM, g)
        outlines[int(f["properties"]["building_id"])] = g.buffer(0)

    cand = [b for b, o in outlines.items()
            if a.min_roof <= o.area <= a.max_roof and b in shipped]
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(cand, size=min(a.n, len(cand)), replace=False)
    print(f"{len(cand)} buildings in {a.min_roof:.0f}-{a.max_roof:.0f} m2; sampling {len(pick)}\n")

    pc = PointCloudSource()
    rows = []
    for k, bid in enumerate(pick, 1):
        bid = int(bid)
        outline = outlines[bid]
        minx, miny, maxx, maxy = outline.bounds
        pts = pc.points_in_bbox(minx - 1, miny - 1, maxx + 1, maxy + 1, building_only=True)
        pts = pts[shapely.vectorized.contains(outline.buffer(0.3), pts[:, 0], pts[:, 1])]
        if len(pts) < 40:
            continue
        old = shipped[bid]
        try:
            new, obs = reconstruct(bid, outline, pts)
        except Exception as e:
            print(f"  {bid}: FAILED {type(e).__name__}: {e}")
            rows.append({"building_id": bid, "failed": True})
            continue
        o_old, o_new = off_plane_share(old, pts), off_plane_share(new, pts)
        cov_old = unary_union(old).area / outline.area if old else 0.0
        cov_new = unary_union([f["geometry"] for f in new]).area / outline.area if new else 0.0
        rows.append({"building_id": bid, "roof_m2": round(outline.area, 1),
                     "off_old": o_old, "off_new": o_new,
                     "cov_old": round(cov_old, 3), "cov_new": round(cov_new, 3),
                     "n_old": len(old), "n_new": len(new), "n_obs": len(obs)})
        if k % 20 == 0:
            print(f"  ...{k}/{len(pick)}")

    ok = [r for r in rows if not r.get("failed") and r["off_old"] is not None
          and r["off_new"] is not None]
    print(f"\n{len(ok)} buildings compared, {sum(1 for r in rows if r.get('failed'))} failed\n")
    if not ok:
        return

    def med(key):
        return float(np.median([r[key] for r in ok]))

    print(f"{'':<16}{'shipped':>10}{'reconstructed':>16}")
    print(f"{'off-plane (med)':<16}{med('off_old'):>9.1%}{med('off_new'):>16.1%}")
    print(f"{'coverage (med)':<16}{med('cov_old'):>9.1%}{med('cov_new'):>16.1%}")
    print(f"{'facets (med)':<16}{med('n_old'):>9.0f}{med('n_new'):>16.0f}")
    print(f"{'obstructions':<16}{'':>9}{med('n_obs'):>16.0f}")

    better = [r for r in ok if r["off_new"] < r["off_old"] - 0.02]
    worse = [r for r in ok if r["off_new"] > r["off_old"] + 0.02]
    print(f"\noff-plane: {len(better)} better, {len(worse)} worse, "
          f"{len(ok) - len(better) - len(worse)} level")
    cov_lost = [r for r in ok if r["cov_new"] < r["cov_old"] - 0.05]
    print(f"coverage:  {len(cov_lost)} buildings lost more than 5 points of roof")
    busy = [r for r in ok if r["n_new"] > max(8, 2 * r["n_old"])]
    print(f"facets:    {len(busy)} buildings more than doubled past 8 facets")

    if worse:
        print(f"\nworst regressions:")
        print(f"  {'building':>10} {'roof':>7} {'off old':>8} {'off new':>8} "
              f"{'cov old':>8} {'cov new':>8} {'facets':>12}")
        for r in sorted(worse, key=lambda r: r["off_old"] - r["off_new"])[:12]:
            print(f"  {r['building_id']:>10} {r['roof_m2']:>7.0f} {r['off_old']:>7.0%} "
                  f"{r['off_new']:>8.0%} {r['cov_old']:>8.0%} {r['cov_new']:>8.0%} "
                  f"{r['n_old']:>5} -> {r['n_new']:<4}")

    out = Path(a.out) if a.out else DATA_DIR / f"validate_reconstruct_{a.area}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

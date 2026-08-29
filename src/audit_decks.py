"""
Find roof area that is not roof: decks, balconies and terraces inside the
building outline.

Josh, on 1/49 Belfast Terrace: "you are counting a balcony area in the outline
of the building, really the roof is only taking up half the building outline."
Confirmed there -- the model resolved facets over 99% of a 228 m2 outline when
about half of it is open deck. That inflates capacity AND places panels on
balconies, and it is invisible to every metric we have, because a deck is a
real, flat, well-sampled surface that looks exactly like a good roof from
above.

The discriminator is height above the ground beneath it. A roof clears the
building; a deck sits at floor level. Neither an absolute nor a relative test
works alone -- a single-storey roof is genuinely low, and a split-level house
genuinely has roof sections at different heights -- so a patch has to be BOTH
low in absolute terms AND well below the same building's main roof.

Ground comes from LiDAR ground-class returns around the building, NOT from
data/dem_wide_mosaic.tif. That DEM is 8 m per cell -- one cell spans a whole
house -- and a first version of this audit used it and produced roof heights of
0.9 m and even -2.08 m. pointcloud_source's own docstring already said as much:
the placement gate measures against ground returns "rather than against
building-class flags or a smoothed DEM".

STATUS: DOES NOT WORK. Kept for the dead ends it records, not for its output.

Three attempts, each wrong for a different reason:

 1. Ground from data/dem_wide_mosaic.tif. That DEM is 8 m per cell -- one cell
    spans a whole house -- and it produced roof heights of 0.9 m and -2.08 m.
 2. Ground from LiDAR ground-class returns, flagging anything low. Flagged 52%
    of buildings, because a single-storey eave sits near 2.5 m, so the lower
    half of any ordinary pitched roof falls under the bar.
 3. Adding a flatness test, on the reasoning that decks are flat and eaves are
    not. Down to 20.7% of buildings -- and rendering the flagged points over
    imagery showed they were large flat commercial ROOFS, not one balcony
    among them.

The reason 3 fails is terrain. Queenstown is built on slopes, so one flat roof
sits at very different heights above the ground beneath its own footprint; the
uphill portion reads as low-and-flat and gets called a deck. Height above
ground cannot separate deck from roof here at all.

What would: imagery. A deck reads differently from a roof -- decking texture,
railings, furniture, colour -- at 0.1 m, where the LiDAR is 0.42 m and blind to
all of it. That is the next thing to try, and it is the same
LiDAR-plus-imagery reconciliation this project keeps arriving back at.

The problem itself is real and confirmed by Josh on 1/49 Belfast Terrace, where
facets covered 99% of a 228 m2 outline and about half of it is open deck. Its
PREVALENCE is still unmeasured.

Reads only static inputs (outlines, point cloud), so it is safe to run while a
rebuild has the pipeline files open.

Usage: python src/audit_decks.py [--area pilot] [--n 400]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pyproj
import shapely.vectorized
from scipy.spatial import cKDTree
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pointcloud_source import PointCloudSource
from src.region_build import area_paths

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True).transform
GROUND_RING_M = 12.0   # how far out to look for ground returns
GROUND_K = 8           # ground returns averaged per query point, to ride out noise
DECK_MAX_ABOVE_GROUND_M = 3.4   # below a plausible single-storey eave
DECK_MIN_BELOW_ROOF_M = 1.5     # ...and clearly below this building's own roof
MIN_DECK_M2 = 4.0               # smaller than this is noise or a step
# Being low is not enough. A single-storey eave sits around 2.5 m, so the lower
# half of an ordinary pitched roof falls under the height bar -- a first pass
# flagged 52% of buildings, which is eaves, not balconies. A deck is FLAT and a
# pitched roof is not, so the low area must also be near-horizontal.
DECK_MAX_SLOPE_DEG = 12.0
DECK_SLOPE_K = 10               # neighbours used to fit the local surface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="pilot")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()

    raw = json.loads(area_paths(a.area)["outlines"].read_text())
    outl = {}
    for f in raw["features"]:
        g = shape(f["geometry"])
        outl[int(f["properties"]["building_id"])] = (
            shp_transform(TO_NZTM, g) if abs(g.bounds[0]) <= 180 else g).buffer(0)

    pc = PointCloudSource()
    rng = np.random.default_rng(a.seed)
    cand = [b for b, o in outl.items() if o.area >= 60]
    pick = rng.choice(cand, size=min(a.n, len(cand)), replace=False)

    rows = []
    for bid in pick:
        bid = int(bid)
        o = outl[bid]
        minx, miny, maxx, maxy = o.bounds
        pts = pc.points_in_bbox(minx - 1, miny - 1, maxx + 1, maxy + 1, building_only=True)
        pts = pts[shapely.vectorized.contains(o, pts[:, 0], pts[:, 1])]
        if len(pts) < 60:
            continue
        # Ground from LiDAR ground-class returns in a ring around the
        # building. Averaging the nearest few rides out individual noisy
        # returns while still following a slope, which matters here -- half of
        # Queenstown is built on one.
        gpts = pc.ground_points_in_bbox(minx - GROUND_RING_M, miny - GROUND_RING_M,
                                        maxx + GROUND_RING_M, maxy + GROUND_RING_M)
        if len(gpts) < GROUND_K:
            continue
        gtree = cKDTree(gpts[:, :2])
        _, idx = gtree.query(pts[:, :2], k=min(GROUND_K, len(gpts)))
        gz = np.median(gpts[idx, 2], axis=1) if idx.ndim > 1 else gpts[idx, 2]
        above = pts[:, 2] - gz
        roof_h = float(np.percentile(above, 85))     # this building's own roof
        low = above < min(DECK_MAX_ABOVE_GROUND_M, roof_h - DECK_MIN_BELOW_ROOF_M)
        if low.sum() >= 8:
            # Local slope at each low point, from a plane through its nearest
            # neighbours. Flat -> deck; sloped -> the low end of a real roof.
            btree = cKDTree(pts[:, :2])
            k = min(DECK_SLOPE_K, len(pts))
            _, nb = btree.query(pts[low][:, :2], k=k)
            flat = np.zeros(int(low.sum()), bool)
            for j in range(int(low.sum())):
                nbr = pts[nb[j]] if nb.ndim > 1 else pts[[nb[j]]]
                if len(nbr) < 4:
                    continue
                A = np.column_stack([nbr[:, 0] - nbr[:, 0].mean(),
                                     nbr[:, 1] - nbr[:, 1].mean(),
                                     np.ones(len(nbr))])
                try:
                    coef, *_ = np.linalg.lstsq(A, nbr[:, 2], rcond=None)
                except Exception:
                    continue
                flat[j] = np.degrees(np.arctan(np.hypot(coef[0], coef[1]))) <= DECK_MAX_SLOPE_DEG
            idx_low = np.where(low)[0]
            low = np.zeros(len(pts), bool)
            low[idx_low[flat]] = True
        if low.sum() < 8:
            rows.append((bid, o.area, roof_h, 0.0, 0.0))
            continue
        # Convert the low points to an area estimate at the sampling density
        # rather than hulling them -- a hull over scattered points would
        # overstate a deck as badly as it overstates an obstruction.
        density = len(pts) / o.area
        deck_m2 = low.sum() / max(density, 1e-6)
        if deck_m2 < MIN_DECK_M2:
            deck_m2 = 0.0
        rows.append((bid, o.area, roof_h, deck_m2, deck_m2 / o.area))
    R = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    n = len(R)
    print(f"{n} buildings sampled in {a.area}\n")
    print(f"median roof height above ground: {np.median(R[:, 1]):.1f} m")
    share = R[:, 3]
    for lo, hi, lab in ((0.0, 0.02, "no deck found"), (0.02, 0.10, "up to 10%"),
                        (0.10, 0.25, "10-25%"), (0.25, 0.50, "25-50%"),
                        (0.50, 1.01, "over half")):
        k = int(((share >= lo) & (share < hi)).sum())
        print(f"  outline that is low-lying, {lab:<14} {k:4d}  ({100 * k / n:4.1f}%)")
    affected = share >= 0.10
    print(f"\nbuildings with 10%+ of their outline sitting at floor level: "
          f"{int(affected.sum())} of {n} ({100 * affected.mean():.1f}%)")
    print(f"total low-lying area across the sample: {R[:, 2].sum():,.0f} m2 "
          f"of {R[:, 0].sum():,.0f} m2 outline ({100 * R[:, 2].sum() / R[:, 0].sum():.1f}%)")
    print("\nIf that area is carrying panels today, both capacity and economics are")
    print("overstated by roughly that share on the affected buildings.")
    worst = sorted(rows, key=lambda r: -r[4])[:10]
    print(f"\n{'building':>10} {'outline':>9} {'roof h':>8} {'low area':>10} {'share':>7}")
    for bid, area, rh, dm, sh in worst:
        print(f"{bid:>10} {area:>8.0f}m {rh:>7.1f}m {dm:>9.0f}m {sh:>6.0%}")
    out = DATA_DIR / f"audit_decks_{a.area}.json"
    out.write_text(json.dumps([{"building_id": r[0], "outline_m2": round(r[1], 1),
                                "roof_h_m": round(r[2], 2), "low_m2": round(r[3], 1),
                                "low_share": round(r[4], 3)} for r in rows], indent=1))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

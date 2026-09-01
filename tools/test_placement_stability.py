"""
How much does the panel count move when the building outline barely does?

Found while measuring whether the LiDAR roof extent would gain panels: a
STRICTLY LARGER boundary lost panels on 23 of 42 buildings, which is impossible
if placement responds sensibly to the region it is given. Chasing that produced
the number below, which matters far more than the roof-extent question did.

  A +/-10 cm nudge to the outline moves the panel count by a median of 26%,
  worst case 88%, with 11 of 12 buildings moving more than 10%.

10 cm is an order of magnitude finer than the outlines themselves are surveyed
to. So the per-building panel count is, to a large extent, arbitrary: not wrong
exactly, but one sample from a wide distribution of equally defensible answers.

RULED OUT FIRST, because it would explain the same observation: the segmenter is
deterministic. The same boundary gives the identical count on repeated runs, so
this is real sensitivity and not RANSAC noise.

WHY IT HAPPENS. The partition is a greedy recursive search -- roof_partition
scores a cut, keeps it if it improves the fit, and recurses. Every one of those
decisions has a threshold, and a boundary nudge flips the marginal ones. One
flipped cut changes the facets beneath it, which changes their axes, setbacks
and packing, and the difference compounds all the way down.

WHAT IT MEANS IN PRACTICE:
  - District totals are fine. This noise is unbiased and averages out over
    thousands of buildings.
  - A single building's number carries roughly this much uncertainty, which is
    worth knowing before defending one to a homeowner.
  - Any A/B test of a pipeline change on a handful of buildings is measuring
    mostly this. Comparisons need either many buildings or the same boundary.

THERE IS AN OPPORTUNITY IN IT. We publish rooftop POTENTIAL, so the best valid
packing is the honest answer, not whichever one a greedy search happened to
reach. Taking the best of a few perturbed runs raises the total.

MIND THE HARNESS THOUGH. This tool measures through segment_building_best and
fit_panels_on_facet, which is NOT the whole pipeline: the real build also runs
the panel gate, per-panel shading and the deep-shade veto, and those absorb most
of the variance because they strip marginal panels whichever partition produced
them. The gain here reads ~13%; through the real pipeline it is +3.45%. Trust
the latter -- it is implemented behind config.LAYOUT_PERTURBATIONS_M -- and read
the numbers below as an upper bound on what perturbation could ever buy.

Usage:
    python tools/test_placement_stability.py
    python tools/test_placement_stability.py --n 20 --nudges 0.05 0.1 -0.05
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="buildings to test")
    ap.add_argument("--nudges", nargs="*", type=float,
                    default=[0.05, 0.10, -0.05],
                    help="metres to buffer the outline by")
    ap.add_argument("--check-determinism", action="store_true",
                    help="run the base boundary three times first")
    a = ap.parse_args()

    import numpy as np
    import geopandas as gpd
    import rasterio
    from src.region_build import area_paths
    from src.roof_segmentation import segment_building_best
    from src.pointcloud_source import PointCloudSource
    sys.path.insert(0, str(ROOT / "tools"))
    from measure_roof_extent_gain import fit_all

    if not LABELS.exists():
        print("no labels yet")
        return 2
    labels = json.loads(LABELS.read_text())["buildings"]
    ids = sorted(int(k) for k in labels)[:a.n]

    pc = PointCloudSource()
    ctxs = {}

    def ctx_for(area):
        if area not in ctxs:
            p = area_paths(area)
            if not (p["outlines"].exists() and p["dsm"].exists()):
                ctxs[area] = None
            else:
                ctxs[area] = {
                    "gdf": gpd.read_file(p["outlines"]).set_index("building_id", drop=False),
                    "dsm": rasterio.open(p["dsm"]),
                    "img": rasterio.open(p["imagery"]) if p["imagery"].exists() else None,
                }
        return ctxs[area]

    def count(ctx, geom, bid):
        f = segment_building_best(ctx["dsm"], pc, geom, bid,
                                  imagery_ds=ctx["img"]) or []
        return fit_all(f, pc, ctx["img"])

    if a.check_determinism:
        print("same boundary, three runs -- any variation is nondeterminism:\n")
        allsame = True
        for bid in ids[:5]:
            ctx = ctx_for(labels[str(bid)].get("area"))
            if not ctx or bid not in ctx["gdf"].index:
                continue
            g = ctx["gdf"].loc[bid].geometry
            c = [count(ctx, g, bid) for _ in range(3)]
            same = len(set(c)) == 1
            allsame &= same
            print(f"  {bid}: {c}  {'identical' if same else 'VARIES'}")
        print(f"\n  deterministic: {allsame}\n")

    head = "".join(f"{f'{n:+.2f}m':>8}" for n in a.nudges)
    print(f"{'building':>10}{'base':>8}{head}{'spread':>9}{'best':>7}")
    spreads, base_tot, best_tot = [], 0, 0
    per_building = []

    for bid in ids:
        ctx = ctx_for(labels[str(bid)].get("area"))
        if not ctx or bid not in ctx["gdf"].index:
            continue
        o = ctx["gdf"].loc[bid].geometry
        try:
            base = count(ctx, o, bid)
        except Exception:
            continue
        got = []
        for nudge in a.nudges:
            g = o.buffer(nudge)
            if g.is_empty or not g.is_valid:
                got.append(None)
                continue
            try:
                got.append(count(ctx, g, bid))
            except Exception:
                got.append(None)
        vals = [base] + [v for v in got if v is not None]
        spread = 100 * (max(vals) - min(vals)) / max(base, 1)
        spreads.append(spread)
        base_tot += base
        best_tot += max(vals)
        per_building.append({"id": bid, "base": base, "got": got})
        cells = "".join(f"{str(v) if v is not None else '-':>8}" for v in got)
        print(f"{bid:>10}{base:>8}{cells}{spread:>8.0f}%{max(vals):>7}")

    if not spreads:
        print("nothing measured")
        return 1
    print(f"\n  median spread {np.median(spreads):.1f}%   "
          f"worst {max(spreads):.0f}%   "
          f"moving >10%: {sum(1 for s in spreads if s > 10)}/{len(spreads)}")
    print(f"  panels: base {base_tot}, best-of-perturbations {best_tot} "
          f"({100 * (best_tot / base_tot - 1):+.1f}%)")

    # WHAT IT WOULD COST. Every extra perturbation is another full layout pass,
    # so the question is not which set wins but which wins per unit of compute.
    if per_building:
        print("\n  keeping the best of N layouts, and what each N costs:")
        combos = [("base only", [])]
        for i, nudge in enumerate(a.nudges):
            combos.append((f'base + "{nudge:+.2f}m"', [i]))
        if len(a.nudges) >= 2:
            combos.append((f'base + "{a.nudges[0]:+.2f}m" + "{a.nudges[-1]:+.2f}m"',
                           [0, len(a.nudges) - 1]))
        combos.append(("all of them", list(range(len(a.nudges)))))
        for name, idx in combos:
            tot = sum(max([r["base"]] + [r["got"][i] for i in idx
                                         if r["got"][i] is not None])
                      for r in per_building)
            cost = 1 + len(idx)
            gain = 100 * (tot / base_tot - 1) if base_tot else 0
            print(f"    {name:<34}{tot:>7}  ({gain:+5.1f}%)  {cost}x layout"
                  f"   {gain / cost:>5.1f}% per unit cost")
        wins = {}
        for r in per_building:
            vals = {"base": r["base"]}
            for i, nudge in enumerate(a.nudges):
                if r["got"][i] is not None:
                    vals[f"{nudge:+.2f}m"] = r["got"][i]
            wins[max(vals, key=vals.get)] = wins.get(max(vals, key=vals.get), 0) + 1
        print(f"    which one wins: {wins}")
    print("\n  A nudge far smaller than the outlines' own survey accuracy should")
    print("  not move the answer. That it moves this much means a single")
    print("  building's count is one sample from a wide distribution, and that")
    print("  small-sample A/B tests of pipeline changes measure mostly this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

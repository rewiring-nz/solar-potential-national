"""
Per-module output invariants: statements that must be true of a finished build.

The counterpart to preflight.py. Preflight checks a stage's INPUTS before it
runs; this checks the OUTPUTS after. Both exist because the failures that cost
this project the most were not crashes -- they were plausible-looking numbers
that nobody could see were wrong.

WHY PER-MODULE RATHER THAN ONE "VALIDATION MODULE": an invariant is only
expressible in the vocabulary of the module that produces it. "A facet's slope
must be under 55 degrees" means nothing outside roof geometry. "Annual yield per
kWp belongs in 900-1700 kWh for New Zealand" means nothing outside the yield
conversion. Collecting them into one place would separate each check from the
code and the reasoning it is about, which is exactly how the existing checks
drifted out of date. So the checks live here grouped BY MODULE, and each names
the module it belongs to.

These are WATCHLISTS, not build failures. Every one of them flags things that
are suspicious rather than certainly wrong, and a real roof occasionally is
strange. The output is a ranked list to look at, in the spirit of the truth
scorecard -- run it after a build, before pushing.

The panel-coverage check earns its place by history. On 1 Sep, 9 Henry Street
was claiming 44 panels -- 19.4 kW -- with every one of them on two 67-degree
faces that were walls, not roof. Josh spotted it by eye on the map. This check
finds it, and 58 others like it, in about a second.

Usage:
    python src/invariants.py                 # whole district
    python src/invariants.py pilot           # one region
    python src/invariants.py --top 30
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# --- module 7, panel placement -------------------------------------------
# Panels cannot cover more of a building than the building has. In plan view a
# real roof loses area to edge setbacks, ridge setbacks, obstructions and the
# foreshortening of any pitch, so the honest ceiling is well under 1.0.
#
# Measured across 14,525 buildings with panels, the coverage ratio runs:
#     p50 0.51   p90 0.71   p95 0.76   p99 0.85   p99.5 0.89   max 1.78
# 0.90 sits just above the 99.5th percentile: it flags 59 buildings (0.4%),
# which is a watchlist rather than a flood, and it catches 9 Henry Street at
# 0.94. A stricter 1.00 -- "physically impossible" -- flags only 15 and would
# have MISSED the building this check was written for.
PANEL_COVERAGE_MAX = 0.90

# --- module 6, roof geometry ----------------------------------------------
# A building whose modelled roof is a small fraction of its footprint has not
# been understood. 9 Henry Street modelled 19 m2 of an 85 m2 footprint (23%)
# once its wall-facets were correctly rejected -- the panels were wrong AND the
# roof was missing, and only the first was visible.
ROOF_COVERAGE_MIN = 0.35

# --- module 4, solar potential conversion ---------------------------------
# Annual AC yield per kWp installed.
#
# The ceiling is DERIVED, not guessed: no roof can beat an unshaded panel at
# the best orientation the lookup table knows, so the model itself supplies the
# bound (about 1257 kWh/kWp here, at a steep north pitch). Anything above it is
# arithmetic that has gone wrong somewhere, not a very good roof. 2% of headroom
# absorbs binning and rounding.
#
# The floor is deliberately far below the ordinary population. A first attempt
# used 800, which flagged 2,973 buildings -- 19.6% -- and was simply wrong: the
# district median is 898 kWh/kWp, because it includes every south-facing and
# tree-shaded roof, and the model puts an unshaded 35-degree SOUTH roof at 721.
# The low tail is real. 300 catches only genuine absurdity.
#
# The lesson is worth keeping next to the constants: a threshold picked from
# intuition flagged a fifth of the district, and the data corrected it. Set
# these from the distribution or from physics, never from a feeling about what
# looks normal.
YIELD_PER_KWP_MIN = 300.0
YIELD_CEILING_TOLERANCE = 1.02


def _physical_yield_ceiling():
    """Best unshaded kWh/kWp this model can produce at any orientation."""
    from src.solar_model import SolarModel
    pv = config.PV_ASSUMPTIONS
    m = SolarModel()
    dc2ac = (pv["inverter_efficiency_pct"] / 100.0) * (1 - pv["system_derate_pct"] / 100.0)
    m2_per_kwp = pv["panel_area_m2"] / (pv["panel_rated_power_w"] / 1000.0)
    eff = pv["panel_efficiency_pct"] / 100.0
    best = max(m.annual_poa_kwh_per_m2(t, a)
               for t in range(0, config.MAX_ROOF_SLOPE_DEG + 1, 5)
               for a in range(0, 360, 15))
    return best * m2_per_kwp * eff * dc2ac * YIELD_CEILING_TOLERANCE


# --- module 7 again: the two files must agree ------------------------------
# solar_potential's fill_panels_100 is baked FROM panel_layouts, so they must
# hold the same number of panels for every building. On 1 Sep they did not, for
# 67.8% of buildings -- the deciles had been baked from an earlier panel set and
# nothing noticed. District-wide the dashboard was quoting 750,672 panels while
# the map drew 774,642.
#
# Josh found it by counting panels on one roof: 15 Kent Street read "25 panels"
# against 79 actually placed. Re-running bake_density_deciles fixed every one,
# so the failure is ORDERING, not arithmetic -- which is exactly the kind of
# thing a build does silently and a check catches in a second.
def _decile_layout_agreement(regions):
    """Buildings where fill_panels_100 disagrees with the actual panel count."""
    from collections import Counter
    lay_path = DATA_DIR / "panel_layouts.geojson"
    if not lay_path.exists():
        return None
    actual = Counter()
    for f in json.loads(lay_path.read_text())["features"]:
        if f["properties"].get("kind") == "panel":
            actual[f["properties"]["building_id"]] += 1
    out = []
    for f in json.loads((DATA_DIR / "solar_potential.geojson").read_text())["features"]:
        p = f["properties"]
        d = p.get("fill_panels_100")
        if d is None:
            continue
        a = actual.get(p["building_id"], 0)
        if a != d:
            out.append((abs(a - d), int(p["building_id"]), (p.get("address") or "").strip(),
                        f"dashboard says {d} panels, the map draws {a}"))
    return out


def _footprints(regions):
    import geopandas as gpd
    from src.region_build import area_paths
    out = {}
    for a in regions:
        p = area_paths(a)
        if not p["outlines"].exists():
            continue
        g = gpd.read_file(p["outlines"])
        for bid, geom in zip(g["building_id"], g.geometry):
            out[int(bid)] = geom.area          # NZTM, already square metres
    return out


def check(regions=None, top=15):
    from src.region_build import all_areas
    regions = regions or list(all_areas())
    sp_path = DATA_DIR / "solar_potential.geojson"
    if not sp_path.exists():
        print(f"no {sp_path} -- run the build first")
        return 2

    sp = json.loads(sp_path.read_text())
    foot = _footprints(regions)
    panel_m2 = config.PV_ASSUMPTIONS["panel_area_m2"]
    yield_max = _physical_yield_ceiling()

    findings = {"7 panel placement": [], "6 roof geometry": [], "4 yield conversion": [],
                "7 deciles vs layouts": []}
    checked = 0

    for f in sp["features"]:
        p = f["properties"]
        bid = int(p.get("building_id", 0))
        area = foot.get(bid)
        if area is None or area <= 0:
            continue
        checked += 1
        addr = (p.get("address") or "").strip()
        n = p.get("panel_count") or 0

        if n:
            cov = n * panel_m2 / area
            if cov > PANEL_COVERAGE_MAX:
                findings["7 panel placement"].append(
                    (cov, bid, addr,
                     f"panels cover {cov:.0%} of the footprint "
                     f"({n} panels, {area:.0f} m2 building)"))

        roof = p.get("facet_area_m2")
        if roof and n:
            frac = roof / area
            if frac < ROOF_COVERAGE_MIN:
                findings["6 roof geometry"].append(
                    (1 - frac, bid, addr,
                     f"only {frac:.0%} of the footprint is modelled as roof "
                     f"({roof:.0f} of {area:.0f} m2) yet it carries {n} panels"))

        kwp, kwh = p.get("kwp"), p.get("ac_kwh_year")
        if kwp and kwh and kwp > 0.5:
            per = kwh / kwp
            if per > yield_max:
                findings["4 yield conversion"].append(
                    (per, bid, addr,
                     f"{per:.0f} kWh per kWp -- ABOVE the {yield_max:.0f} an "
                     f"unshaded panel at the best orientation can reach"))
            elif per < YIELD_PER_KWP_MIN:
                findings["4 yield conversion"].append(
                    (yield_max - per, bid, addr,
                     f"{per:.0f} kWh per kWp -- implausibly low "
                     f"(floor {YIELD_PER_KWP_MIN:.0f})"))

    agree = _decile_layout_agreement(regions)
    if agree is not None:
        findings["7 deciles vs layouts"] = agree

    total = sum(len(v) for v in findings.values())
    print(f"invariant check over {checked:,} buildings in "
          f"{len(regions)} region(s)\n")
    for module in ("7 deciles vs layouts", "6 roof geometry", "7 panel placement",
                   "4 yield conversion"):
        hits = sorted(findings[module], reverse=True)
        pct = 100 * len(hits) / checked if checked else 0
        print(f"module {module}: {len(hits):,} flagged ({pct:.2f}%)")
        for _, bid, addr, why in hits[:top]:
            print(f"    #{bid}  {addr[:34]:34s} {why}")
        if len(hits) > top:
            print(f"    ... and {len(hits) - top:,} more")
        print()

    print(f"{total:,} flags in total. These are WATCHLISTS, not failures -- a "
          f"real roof is\noccasionally strange. Look at the top of each list "
          f"before pushing a build.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regions", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    return check(a.regions or None, a.top)


if __name__ == "__main__":
    sys.exit(main())

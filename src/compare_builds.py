"""
Diff a finished build against the previous one, per building.

Why this exists: every placement regression in this project so far was
found the same way -- Josh looked at a roof on the live map and said "this
is wrong". Three separate gate rules each deleted real panels from real
houses (4 Abbottswood Ln 61->6, 6 Shotover St 72kW->4, 7 Cedar Dr 69->6),
and nothing in the pipeline noticed, because every run prints healthy
totals whether or not it just destroyed a suburb. Totals hide it: a rule
that wipes 155 panels off one commercial roof moves a 743,303-panel total
by 0.02%.

So compare per BUILDING, and rank by what changed most. A rebuild that
intends to change one thing should show a short, explainable list.

Usage:
    # before the rebuild's merge overwrites data/solar_potential.geojson
    python src/compare_builds.py --snapshot
    # ...rebuild...
    python src/compare_builds.py --watchlist   # the bug-doc buildings by name
    python src/compare_builds.py [--top 40] [--min-loss 5]

--snapshot writes data/build_snapshot_prev.json (gitignored, ~0.7MB).
"""

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOLAR = DATA_DIR / "solar_potential.geojson"
SNAP = DATA_DIR / "build_snapshot_prev.json"


# Which field actually reflects a rebuild matters, and getting it wrong makes
# this tool worse than useless. fill_panels_100 / fill_kwh_100 are baked from
# THIS run's panel_layouts by src/bake_density_deciles.py -- they are what the
# map draws. panel_count / kwp come from the solar-model pass, which
# run_layouts_regate.sh deliberately skips, so they can be many builds stale.
# The first version of this script compared panel_count and cheerfully
# reported "+0, no change" on a rebuild that had just removed 23,575 panels.
PANELS, KWH, LEGACY_COUNT, ADDRESS = 0, 1, 2, 3


def _read_current():
    d = json.loads(SOLAR.read_text())
    out = {}
    for f in d["features"]:
        p = f["properties"]
        out[str(p["building_id"])] = [p.get("fill_panels_100", 0), p.get("fill_kwh_100", 0),
                                      p.get("panel_count", 0), p.get("address", "")]
    return out


def snapshot():
    cur = _read_current()
    SNAP.write_text(json.dumps(cur))
    print(f"snapshot: {len(cur)} buildings, {sum(v[PANELS] for v in cur.values()):,} placed panels, "
          f"{sum(v[KWH] for v in cur.values()) / 1e6:.1f} GWh/yr -> {SNAP.name}")


def compare(top=40, min_loss=5):
    if not SNAP.exists():
        raise SystemExit(f"no {SNAP.name} -- run with --snapshot before the rebuild")
    prev, cur = json.loads(SNAP.read_text()), _read_current()

    p_tot, c_tot = sum(v[PANELS] for v in prev.values()), sum(v[PANELS] for v in cur.values())
    have_prev_kwh = all(v[KWH] is not None for v in prev.values())
    p_kwh = sum(v[KWH] or 0 for v in prev.values())
    c_kwh = sum(v[KWH] for v in cur.values())
    print(f"buildings     {len(prev):,} -> {len(cur):,}")
    print(f"placed panels {p_tot:,} -> {c_tot:,}  ({c_tot - p_tot:+,}, {100 * (c_tot - p_tot) / max(p_tot, 1):+.1f}%)")
    if have_prev_kwh:
        print(f"annual output {p_kwh / 1e6:.1f} -> {c_kwh / 1e6:.1f} GWh/yr  ({(c_kwh - p_kwh) / 1e6:+.1f})")
    else:
        print(f"annual output {c_kwh / 1e6:.1f} GWh/yr  (no comparable figure in this snapshot)")

    # panel_count/kwp ride along from an older solar-model pass. When they
    # drift far from the layouts, the choropleth and the building table are
    # describing a build the map is no longer drawing.
    legacy = sum(v[LEGACY_COUNT] for v in cur.values())
    if legacy and abs(legacy - c_tot) > 0.02 * max(c_tot, 1):
        print(f"\nNOTE: solar_potential's panel_count totals {legacy:,} against {c_tot:,} "
              f"actually placed ({100 * (legacy - c_tot) / max(c_tot, 1):+.1f}%). Those fields "
              f"(and kwp, which colours the choropleth) come from the solar-model pass and are "
              f"stale until a full build re-runs it.")

    gone = [b for b in prev if b not in cur]
    new = [b for b in cur if b not in prev]
    if gone:
        print(f"\n{len(gone)} buildings DISAPPEARED from the build "
              f"(dedupe ownership change, or a region that failed): {gone[:10]}")
    if new:
        print(f"{len(new)} buildings are new to the build: {new[:10]}")

    deltas = []
    for b, c in cur.items():
        p = prev.get(b)
        if p is None:
            continue
        d = c[PANELS] - p[PANELS]
        if abs(d) >= min_loss:
            deltas.append((d, b, p, c))
    losses = sorted(d for d in deltas if d[0] < 0)
    gains = sorted((d for d in deltas if d[0] > 0), reverse=True)

    # Losses first and always: a panel that vanished is the failure mode that
    # has actually bitten this project, repeatedly. Gains are usually the
    # intended effect of whatever changed.
    for label, rows in (("LOST panels", losses), ("GAINED panels", gains)):
        print(f"\n{len(rows)} buildings {label} (>= {min_loss}); worst {min(top, len(rows))}:")
        for d, b, p, c in rows[:top]:
            print(f"  {d:+5d}  {p[PANELS]:4d} -> {c[PANELS]:4d} panels, "
                  f"{c[KWH] / 1000:7.1f} MWh/yr now  #{b}  {c[ADDRESS] or p[ADDRESS]}")

    wiped = [r for r in losses if r[3][PANELS] == 0 and r[2][PANELS] > 0]
    if wiped:
        print(f"\nWARNING: {len(wiped)} buildings went to ZERO panels having had some before. "
              f"That is the shape of every gate regression so far -- check these on the map "
              f"before deploying:")
        for d, b, p, c in wiped[:20]:
            print(f"  #{b}  {p[PANELS]} -> 0 panels  {p[ADDRESS]}")


# The 11 buildings from Josh's first bug doc (docs/bugdoc-2026-08-22.md).
# Every rebuild should be checked against these by name, not just in the
# aggregate -- they are the cases that defined what "wrong" looks like here.
WATCHLIST = {
    5370339: "7 Duke St (curved/parapet commercial)",
    4735316: "101/8 Duke St (whole roof was one obstruction)",
    4735237: "24 Beach St (most of roof obstruction)",
    4734769: "28 Rees St (whole roof obstruction, 0 panels)",
    4726050: "22 Earl St (whole complex obstruction)",
    5370360: "17 Marine Pde (panels OVER real obstructions)",
    5370328: "9 Marine Pde (large flat section all obstruction)",
    4725584: "32 Frankton Rd (most of complex obstruction)",
    5372566: "19 Camp St (curved fan roof)",
    4750979: "9 Leeds Ln (panels overlap ridges/hips)",
    4750998: "10B Belfast Tce (roof levels not split)",
}


def watchlist():
    prev = json.loads(SNAP.read_text()) if SNAP.exists() else {}
    cur = _read_current()
    print("bug-doc watchlist (docs/bugdoc-2026-08-22.md):")
    for bid, why in WATCHLIST.items():
        c = cur.get(str(bid))
        p = prev.get(str(bid))
        if c is None:
            print(f"  #{bid}  MISSING from this build  -- {why}")
            continue
        was = f"{p[PANELS]:4d} panels" if p else "(no snapshot)"
        print(f"  #{bid}  {was}  ->  {c[PANELS]:4d} panels   {why}")


def main():
    argv = sys.argv[1:]
    if "--snapshot" in argv:
        return snapshot()
    if "--watchlist" in argv:
        return watchlist()
    top = int(argv[argv.index("--top") + 1]) if "--top" in argv else 40
    min_loss = int(argv[argv.index("--min-loss") + 1]) if "--min-loss" in argv else 5
    compare(top=top, min_loss=min_loss)


if __name__ == "__main__":
    main()

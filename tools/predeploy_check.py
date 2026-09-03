"""
Compare a build against the LIVE site before deploying it.

Three regressions were caught this way on 3 Sep, all of them introduced by
fixes that measured well:

  #4725584  465 panels -> 0   a borrowed plane dragged _area_weighted_inlier
                              below MIN_ROOF_CONFIDENCE and the roof was
                              withheld
  #4735242  156 panels -> 0   a 15 m2 markup on a 1,054 m2 building replaced
                              the whole partition
  23 buildings                withheld as low_confidence that had panels before

NONE of them were visible in tools/measure_facet_agreement.py, which is what
had been driving the work all day. That tool scores facet SHAPE against Josh's
drawings; it has no idea whether a building ships any panels at all. A roof can
match his markup beautifully and show nothing on the map.

WHY AGAINST LIVE, NOT AGAINST THE LAST BUILD. `compare_builds.py --snapshot`
overwrites its baseline every run, so after two rebuilds in a day it compares a
build to itself and reports no change -- which happened. The deployed site is
the only baseline that cannot be overwritten by the thing being tested, and it
is also the thing Josh is actually looking at.

WHAT IS FLAGGED, worst first:
  ZEROED     had panels, now has none. The most visible failure there is: a
             building that simply disappears from the map.
  BIG DROP   lost more than DROP_FRAC of its panels AND more than DROP_MIN.
             Proportional alone flags every small shed; absolute alone flags
             every large roof.
  WITHHELD   newly carries a no_estimate_reason having been estimated before.

A drop is not automatically wrong -- better geometry legitimately removes
panels that were overlapping a ridge, and Josh said so himself: "on some faces
this will add more panels, on others it will reduce them". This does not judge.
It surfaces what a person should look at before pushing.

Usage:
    python tools/predeploy_check.py
    python tools/predeploy_check.py --live-url https://.../solar_potential.geojson
"""

import argparse
import json
import sys
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

LIVE_URL = ("https://rewiring-nz.github.io/nz-solar-potential/"
            "data/solar_potential.geojson")
NEW = ROOT / "data" / "solar_potential.geojson"

ZERO_MIN = 5        # ignore a building that only ever had a panel or two
DROP_FRAC = 0.30    # lost this share of its panels...
DROP_MIN = 20       # ...and at least this many


def _panels(p):
    return p.get("fill_panels_100", p.get("panel_count", 0)) or 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-url", default=LIVE_URL)
    ap.add_argument("--new", default=str(NEW))
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    newp = Path(a.new)
    if not newp.exists():
        print(f"no build at {newp}")
        return 2
    print(f"fetching the live build from {a.live_url.split('/data/')[0]} ...")
    try:
        with urllib.request.urlopen(a.live_url, timeout=120) as r:
            live_doc = json.loads(r.read().decode())
    except Exception as e:
        print(f"could not fetch live: {type(e).__name__}: {e}")
        print("Refusing to pass a check that did not run.")
        return 2

    live = {int(f["properties"]["building_id"]): f["properties"]
            for f in live_doc["features"]
            if f["properties"].get("building_id") is not None}
    new = {int(f["properties"]["building_id"]): f["properties"]
           for f in json.loads(newp.read_text())["features"]
           if f["properties"].get("building_id") is not None}
    common = sorted(set(live) & set(new))
    if not common:
        print("no buildings in common -- is this the same district?")
        return 2

    zeroed, dropped, withheld = [], [], []
    tot_live = tot_new = 0
    for b in common:
        wl, wn = _panels(live[b]), _panels(new[b])
        tot_live += wl
        tot_new += wn
        if wl >= ZERO_MIN and wn == 0:
            zeroed.append((b, wl, wn))
        elif wl > 0 and (wl - wn) >= DROP_MIN and (wl - wn) / wl >= DROP_FRAC:
            dropped.append((b, wl, wn))
        if new[b].get("no_estimate_reason") and not live[b].get("no_estimate_reason") \
                and wl >= ZERO_MIN:
            withheld.append((b, wl, new[b]["no_estimate_reason"]))

    print(f"\n{len(common)} buildings compared")
    print(f"  panels   {tot_live:,} -> {tot_new:,}  "
          f"({tot_new - tot_live:+,}, {100 * (tot_new - tot_live) / max(tot_live, 1):+.1f}%)")

    def show(title, rows, fmt):
        print(f"\n  {title}: {len(rows)}")
        for r in sorted(rows, key=lambda r: -r[1])[:a.top]:
            print("    " + fmt(r))

    show("ZEROED (had panels, now none)", zeroed,
         lambda r: f"#{r[0]}  {r[1]} -> 0")
    show(f"BIG DROP (>{100*DROP_FRAC:.0f}% and >{DROP_MIN} panels)", dropped,
         lambda r: f"#{r[0]}  {r[1]} -> {r[2]}  ({100*(r[2]-r[1])/r[1]:+.0f}%)")
    show("NEWLY WITHHELD", withheld,
         lambda r: f"#{r[0]}  had {r[1]} panels, now: {r[2]}")

    bad = len(zeroed) + len(withheld)
    print(f"\n  {'LOOK AT THESE BEFORE DEPLOYING' if bad else 'nothing zeroed or newly withheld'}")
    print("  A drop is not automatically wrong -- better geometry removes panels")
    print("  that were overlapping a ridge. This surfaces, it does not judge.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""Re-run face prediction for the buildings the hypothesis area-cap raise
(450 -> 2000 m2, 17 Sep) can actually change, then patch the ones it did.

The new gate only alters buildings where ALL of:
  - footprint area in (450, 2000] m2
  - a selected_faces file exists whose source is not already hypothesis
  - every incumbent score < 0.50 (hypothesis is barred from shipping
    otherwise, so rerunning anything else is wasted compute)

For each region: list candidates, rerun tools/predict_faces.py --ids on
them, and report which buildings' selected reading changed source. Pass
--patch to also rebuild those buildings' layouts (patch_buildings,
--skip-tiles per region; rebuild tiles once yourself afterwards).

    .venv/bin/python tools/redo_hyp_gate.py                # dry: list only
    .venv/bin/python tools/redo_hyp_gate.py --regions pilot
    .venv/bin/python tools/redo_hyp_gate.py --patch        # full run
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEL = Path("data/selected_faces")
PY = sys.executable



from src.region_build import all_areas as _all_regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=None)
    ap.add_argument("--patch", action="store_true")
    a = ap.parse_args()
    import config
    regions = a.regions or _all_regions()

    total_changed = []
    for region in regions:
        ids = candidates(region)
        print(f"{region}: {len(ids)} candidates", flush=True)
        if not ids:
            continue
        before = {i: json.loads((SEL / f"{i}.json").read_text()).get("source")
                  for i in ids}
        r = subprocess.run(
            [PY, "tools/predict_faces.py", "--region", region,
             "--ids", *map(str, ids)],
            env={**os.environ, "SOLAR_SELECTED_FACES": "1"})
        if r.returncode != 0:
            print(f"{region}: predict_faces FAILED rc={r.returncode}",
                  flush=True)
            continue
        changed = []
        for i in ids:
            p = SEL / f"{i}.json"
            now = json.loads(p.read_text()).get("source") if p.exists() else None
            if now == "hypothesis" and before[i] != "hypothesis":
                changed.append(i)
        print(f"{region}: {len(changed)} now hypothesis: "
              f"{changed[:20]}{'...' if len(changed) > 20 else ''}", flush=True)
        total_changed.append((region, changed))
        if a.patch and changed:
            r = subprocess.run(
                [PY, "src/patch_buildings.py", *map(str, changed),
                 "--area", region, "--skip-tiles"],
                env={**os.environ, "SOLAR_SELECTED_FACES": "1"})
            print(f"{region}: patch rc={r.returncode}", flush=True)

    n = sum(len(c) for _, c in total_changed)
    print(f"TOTAL {n} buildings changed to hypothesis", flush=True)


if __name__ == "__main__":
    main()

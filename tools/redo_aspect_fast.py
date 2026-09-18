"""Re-predict ONLY what a selector change can move, in parallel.

The first pass at this re-predicted all 15,353 buildings serially on one
of sixteen cores and was 7 hours in with 11 more to go. Two facts make it
minutes instead:

  * A change to the FORM CONTEST can only move a building where the simple
    form is allowed to ship at all -- source already hypothesis, or every
    incumbent reading under 0.50. That is 3,802 of 13,376 buildings; the
    rest are decided by a reading the contest never reaches.
  * predict_faces is one process. Regions are independent, so they run
    side by side. Patching is NOT independent -- every region's patch
    rewrites the shared district geojson -- so that phase stays serial.

Phase 1 predicts in parallel, phase 2 patches serially, and the changed
set is a real diff of the face rings, not an assumption.

    python tools/redo_aspect_fast.py --skip <regions already done>
"""

import argparse, hashlib, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SEL = Path("data/selected_faces")
PY_ = sys.executable


def fingerprint(bid):
    p = SEL / f"{bid}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    rings = [[(round(x, 2), round(y, 2)) for x, y in f] for f in d["faces"]]
    return hashlib.md5(json.dumps([d.get("source"), sorted(rings)],
                                  sort_keys=True).encode()).hexdigest()


def eligible(bid):
    """Could the form contest possibly change this building's reading?"""
    p = SEL / f"{bid}.json"
    if not p.exists():
        return True          # no reading yet: the contest is all it has
    try:
        d = json.loads(p.read_text())
    except Exception:
        return True
    if d.get("source") == "hypothesis":
        return True
    return max(d.get("score_sam", 0), d.get("score_line", 0),
               d.get("score_lidar", 0)) < 0.50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--predict-only", action="store_true")
    a = ap.parse_args()
    import config
    import geopandas as gpd
    from src.region_build import area_paths

    work = []
    for region in config.REGIONS:
        if region in a.skip:
            continue
        op = area_paths(region)["outlines"]
        if not Path(op).exists():
            continue
        ids = [int(x) for x in gpd.read_file(op)["building_id"]]
        el = [b for b in ids if eligible(b)]
        if el:
            work.append((region, ids, el))
    print(f"{len(work)} regions, "
          f"{sum(len(e) for _, _, e in work)} eligible buildings", flush=True)

    befores = {}

    def predict(item):
        region, ids, el = item
        befores[region] = {b: fingerprint(b) for b in el}
        for i in range(0, len(el), 400):
            chunk = el[i:i + 400]
            r = subprocess.run([PY_, "tools/predict_faces.py", "--region",
                                region, "--ids", *map(str, chunk)],
                               env={**os.environ, "SOLAR_SELECTED_FACES": "1",
                                    "OMP_NUM_THREADS": "2"},
                               capture_output=True)
            if r.returncode != 0:
                print(f"{region}: predict rc={r.returncode} "
                      f"{r.stderr.decode()[-200:]}", flush=True)
                return region, []
        changed = [b for b in el if fingerprint(b) != befores[region][b]]
        print(f"{region}: {len(changed)} of {len(el)} eligible changed",
              flush=True)
        return region, changed

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        results = list(ex.map(predict, work))

    total = sum(len(c) for _, c in results)
    print(f"PREDICT DONE: {total} buildings changed", flush=True)
    if a.predict_only:
        return

    for region, changed in results:      # serial: shared district geojson
        if not changed:
            continue
        for i in range(0, len(changed), 60):
            chunk = changed[i:i + 60]
            rc = subprocess.run([PY_, "src/patch_buildings.py",
                                 *map(str, chunk), "--area", region,
                                 "--skip-tiles"],
                                env={**os.environ,
                                     "SOLAR_SELECTED_FACES": "1"}).returncode
            print(f"{region}: patch chunk rc={rc}", flush=True)
    print(f"TOTAL {total} buildings rebuilt", flush=True)


if __name__ == "__main__":
    main()

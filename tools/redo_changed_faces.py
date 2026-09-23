"""Re-run face prediction district-wide, then patch only the buildings whose
SELECTED READING ACTUALLY CHANGED.

Used after a change to the selector or the form contest (e.g. the eave-based
aspect term, 18 Sep). Re-predicting is cheap relative to rebuilding layouts,
so predict everything, diff the face rings, and rebuild only real changes.

    .venv/bin/python tools/redo_changed_faces.py              # dry run
    .venv/bin/python tools/redo_changed_faces.py --patch
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SEL = Path("data/selected_faces")
PY_ = sys.executable



from src.region_build import all_areas as _all_regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", action="store_true")
    ap.add_argument("--regions", nargs="*", default=None)
    a = ap.parse_args()
    import config
    import geopandas as gpd
    from src.region_build import area_paths

    total = 0
    for region in (a.regions or _all_regions()):
        op = area_paths(region)["outlines"]
        if not Path(op).exists():
            continue
        ids = [int(x) for x in gpd.read_file(op)["building_id"]]
        before = {b: fingerprint(b) for b in ids}
        r = subprocess.run([PY_, "tools/predict_faces.py", "--region", region,
                            "--ids", *map(str, ids)],
                           env={**os.environ, "SOLAR_SELECTED_FACES": "1"})
        if r.returncode != 0:
            print(f"{region}: predict FAILED rc={r.returncode}", flush=True)
            continue
        changed = [b for b in ids if fingerprint(b) != before[b]]
        print(f"{region}: {len(changed)} of {len(ids)} changed", flush=True)
        total += len(changed)
        if a.patch and changed:
            for i in range(0, len(changed), 60):   # bounded argv, bounded blast
                chunk = changed[i:i + 60]
                rc = subprocess.run(
                    [PY_, "src/patch_buildings.py", *map(str, chunk),
                     "--area", region, "--skip-tiles"],
                    env={**os.environ, "SOLAR_SELECTED_FACES": "1"}).returncode
                print(f"{region}: chunk {i//60} rc={rc}", flush=True)
    print(f"TOTAL {total} buildings changed", flush=True)


if __name__ == "__main__":
    main()

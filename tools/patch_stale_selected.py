"""Rebuild every building the layouts were not built from.

RESUME-SAFE BY CONSTRUCTION, which redo_aspect_fast is not: that driver
fingerprints each building, predicts, and patches whatever changed. Restart
it after a preemption and the already-predicted buildings fingerprint as
UNCHANGED -- they were changed by the run that died -- so they are silently
never patched. The VM is preemptible and this has now bitten twice.

patch_buildings records a hash of the reading each building was built from
(data/built_from.json). Anything whose current reading hashes differently is
work not yet applied -- true however many times the run is interrupted, and
unlike file mtimes it is not destroyed by patching rewriting the layouts.

Usage: python tools/patch_stale_selected.py [--patch]
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SEL = Path("data/selected_faces")
STATE = Path("data/built_from.json")
PY_ = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", action="store_true")
    a = ap.parse_args()
    import geopandas as gpd
    import config
    from src.region_build import area_paths
    import hashlib
    try:
        state = json.loads(STATE.read_text())
    except Exception:
        state = {}
    stale = set()
    for p in SEL.glob("*.json"):
        h = hashlib.md5(p.read_bytes()).hexdigest()[:12]
        if state.get(p.stem) != h:
            stale.add(int(p.stem))
    print(f"{len(stale)} readings the layouts were not built from", flush=True)
    if not stale:
        return
    d = Path("data/regions")
    regions = sorted({p.name for p in d.iterdir() if p.is_dir()}
                     | set(config.REGIONS))
    total = 0
    for region in regions:
        op = Path(area_paths(region)["outlines"])
        if not op.exists():
            continue
        ids = sorted(stale & {int(x) for x in gpd.read_file(op)["building_id"]})
        if not ids:
            continue
        print(f"{region}: {len(ids)} to rebuild", flush=True)
        total += len(ids)
        if not a.patch:
            continue
        for i in range(0, len(ids), 60):
            chunk = ids[i:i + 60]
            rc = subprocess.run(
                [PY_, "src/patch_buildings.py", *map(str, chunk),
                 "--area", region, "--skip-tiles", "--skip-bake"],
                env={**os.environ, "SOLAR_SELECTED_FACES": "1"}).returncode
            print(f"{region}: chunk rc={rc}", flush=True)
    # Once, at the end, instead of once per 60-building chunk: it is a
    # district-wide pass over the merged layouts and only the last run counts.
    if a.patch and total:
        subprocess.run([PY_, "src/bake_density_deciles.py"], check=True)
    print(f"TOTAL {total}", flush=True)


if __name__ == "__main__":
    main()

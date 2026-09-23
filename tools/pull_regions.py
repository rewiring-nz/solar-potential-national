"""Fetch published regions' emitted output from the bucket, for the combine.

    python tools/pull_regions.py                 # every region in done/
    python tools/pull_regions.py kingston hawea  # just these

Workers publish to gs://<bucket>/build/regions/<r>/out/ (src/publish_region.py)
and never hold the whole build; the machine that runs src/combine_regions.py
needs data/out/<r>/ for each region it combines, and this is how it gets them.
Only out/ is pulled -- the region GeoJSONs and readings stay in the bucket
unless asked for (--with-region), because the combine does not read them.

gcloud storage rsync skips what is already identical, so re-running after a
partial pull costs only the listing.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gcs_queue import ls, url

OUT_ROOT = ROOT / "data" / "out"
REGIONS_DIR = ROOT / "data" / "regions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regions", nargs="*")
    ap.add_argument("--with-region", action="store_true",
                    help="also pull the region GeoJSONs into data/regions/<r>/")
    a = ap.parse_args()
    regions = a.regions or sorted(n[:-5] for n in ls("done") if n.endswith(".json"))
    if not regions:
        print("nothing in done/ and no regions given")
        return 1
    ok = 0
    for r in regions:
        dest = OUT_ROOT / r
        dest.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(["gcloud", "storage", "rsync", "-r", url("regions", r, "out"), str(dest)],
                             capture_output=True, text=True)
        if res.returncode != 0 or not (dest / "summary.json").exists():
            print(f"  {r}: FAILED ({res.stderr.strip().splitlines()[-1] if res.stderr.strip() else 'no summary'})")
            continue
        if a.with_region:
            rd = REGIONS_DIR / r
            rd.mkdir(parents=True, exist_ok=True)
            subprocess.run(["gcloud", "storage", "rsync", "-r", url("regions", r, "region"), str(rd)],
                           capture_output=True, text=True)
        ok += 1
        print(f"  {r}: pulled")
    print(f"{ok}/{len(regions)} regions under {OUT_ROOT} -- now: python src/combine_regions.py")
    return 0 if ok == len(regions) else 1


if __name__ == "__main__":
    sys.exit(main())

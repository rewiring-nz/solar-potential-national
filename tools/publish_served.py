"""Push the combined, served data set to the tiles bucket, under a version.

    python tools/publish_served.py 38            # -> gs://rewiring-solar-tiles/v38/data/
    python tools/publish_served.py 38 --public   # ...and grant public read once

Then in site-config.js:  dataBase: "https://storage.googleapis.com/rewiring-solar-tiles/v38/",
and dataVersion: "38". The page prefixes every data URL with dataBase
(preview.html, DATA_BASE), so nothing else changes.

WHY A VERSION FOLDER. Tiles are fetched by byte range and cached by the
browser; replacing a file in place under a URL the page already holds is how
a client ends up reading the header of one build and the tiles of another.
A new version is a new folder; the old one is deleted when nobody points at it.

WHY A SEPARATE BUCKET. gs://rewiring-solar-data holds models and build inputs
and must stay private. gs://rewiring-solar-tiles holds only what the page
serves, has CORS for Range requests, and is the one that is made public.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BUCKET = "gs://rewiring-solar-tiles"
SERVED_FILES = ["panel_layouts.pmtiles", "buildings.pmtiles", "building_cells.pmtiles",
                "addresses.json", "assumptions.json", "seasonal_curves.json",
                "build_summary.json", "markup_lines.geojson"]
SERVED_DIRS = ["building_detail", "heatmap_tiles", "addresses", "seasonal_curves", "summaries"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("--public", action="store_true",
                    help="grant allUsers objectViewer on the bucket (idempotent)")
    a = ap.parse_args()
    base = f"{BUCKET}/v{a.version}/data"
    missing = [f for f in ("panel_layouts.pmtiles", "buildings.pmtiles", "building_cells.pmtiles")
               if not (DATA / f).exists()]
    if missing:
        print(f"not a combined build: missing {missing} -- run src/combine_regions.py first")
        return 2
    for f in SERVED_FILES:
        p = DATA / f
        if p.exists():
            subprocess.run(["gcloud", "storage", "cp", "-q", str(p), f"{base}/{f}"], check=True)
    for d in SERVED_DIRS:
        p = DATA / d
        if p.is_dir():
            subprocess.run(["gcloud", "storage", "rsync", "-r", "-q", str(p), f"{base}/{d}"], check=True)
    if a.public:
        subprocess.run(["gcloud", "storage", "buckets", "add-iam-policy-binding", BUCKET,
                        "--member=allUsers", "--role=roles/storage.objectViewer", "-q"],
                       check=True, capture_output=True)
        print("public read granted on", BUCKET)
    print(f"served set at {base}/")
    print(f'site-config.js:  dataBase: "https://storage.googleapis.com/rewiring-solar-tiles/v{a.version}/",  '
          f'dataVersion: "{a.version}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Pull a region's raster inputs back from the bucket (or LINZ as fallback).

The heavy inputs -- imagery mosaics, point-cloud tiles -- live in
gs://rewiring-solar-data, not on this Mac. patch_buildings and any local build
need them present for the regions being touched; run this first:

  python src/restore_region_data.py frankton_flats arrowtown_millbrook
  python src/restore_region_data.py --pointcloud    # the .laz tiles

Requires gcloud auth (the claude-batch service key) or falls back to telling
you to run the fetch scripts against LINZ.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUCKET = "gs://rewiring-solar-data"

def main():
    args = [a for a in sys.argv[1:]]
    if "--pointcloud" in args:
        args.remove("--pointcloud")
        dest = ROOT / "data/pointcloud"
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcloud", "storage", "rsync", f"{BUCKET}/pointcloud", str(dest)],
                       check=True)
    for region in args:
        dest = ROOT / "data/regions" / region
        dest.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["gcloud", "storage", "rsync", "-r",
                            f"{BUCKET}/regions/{region}", str(dest)])
        if r.returncode != 0:
            print(f"{region}: not in bucket yet -- fall back to "
                  f"'python src/fetch_regions.py {region}' (LINZ, slower)")

if __name__ == "__main__":
    main()

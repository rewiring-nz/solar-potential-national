"""
Downscaled ("overview") copies of the regional heat-map rasters.

Why this exists -- measured, not guessed. The heat map is served as MapLibre
IMAGE sources, one per region, and an image source is not tiled: attaching one
uploads the ENTIRE png as a single GPU texture at full resolution, whatever
the current zoom. The rasters are built at 0.4 m/px so a roof reads sharply at
z18, which makes speargrass_hayes 13007x15989 = 832 MB of RGBA on its own. At
district zoom every region intersects the viewport, so 20 of the 24 attach at
once: ~4.1 GB of texture for a view where a whole roof is two pixels wide.
That is the single biggest cause of the map feeling heavy.

Two problems, one fix:
- Memory. A 4096-max-dimension copy is 1/15th the pixels of the worst case.
- Portability. 15989px is under the 16384 texture limit of a typical desktop
  GPU but over the 8192 limit plenty of hardware still reports, where the
  region's heat map simply fails to appear. 4096 is safe essentially
  everywhere.

The frontend uses these below DETAIL_ZOOM and the full-resolution originals
above it, so nothing is lost where the detail is actually visible.

Run after merge_regions.py (it reads data/heatmaps/, written by that step).
Idempotent: an up-to-date LOD (newer than its source) is left alone.

Usage: python src/build_heatmap_lod.py [--force]
"""

import json
import sys
from pathlib import Path

from PIL import Image

# These are legitimately huge rasters we produced ourselves; the decompression
# bomb guard is aimed at untrusted input and only gets in the way here.
Image.MAX_IMAGE_PIXELS = None

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HEATMAPS_DIR = DATA_DIR / "heatmaps"
LOD_DIR = HEATMAPS_DIR / "lod"
MAX_DIM = 4096


def main(force=None):
    # Callable as a library (src/merge_regions.py runs it right after it
    # rewrites the manifest), so don't read argv unless nobody said.
    if force is None:
        force = "--force" in sys.argv[1:]
    manifest_path = HEATMAPS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path} -- run src/merge_regions.py first")
    manifest = json.loads(manifest_path.read_text())
    LOD_DIR.mkdir(parents=True, exist_ok=True)

    saved_px = 0
    for entry in manifest:
        src = DATA_DIR.parent / entry["png"]
        if not src.exists():
            print(f"  WARNING: {entry['name']} png missing at {src}, skipping")
            continue
        dst = LOD_DIR / f"{entry['name']}.png"
        entry["png_lod"] = f"data/heatmaps/lod/{entry['name']}.png"
        # Pixel dimensions go in the manifest so the frontend can budget GPU
        # texture memory before attaching anything, instead of discovering the
        # cost after the upload.
        with Image.open(src) as im:
            entry["size"] = list(im.size)
        if dst.exists() and not force and dst.stat().st_mtime >= src.stat().st_mtime:
            with Image.open(dst) as im:
                entry["size_lod"] = list(im.size)
            print(f"  {entry['name']}: up to date")
            continue
        with Image.open(src) as im:
            w, h = im.size
            scale = min(1.0, MAX_DIM / max(w, h))
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            # LANCZOS, not NEAREST: this is a continuous colour field, and a
            # nearest-neighbour shrink of a 4x oversampled raster drops most of
            # the roofs entirely (a 2m building lands between sample points).
            out = im.convert("RGBA").resize((nw, nh), Image.LANCZOS)
            out.save(dst, optimize=True)
        entry["size_lod"] = [nw, nh]
        saved_px += (w * h - nw * nh)
        print(f"  {entry['name']}: {w}x{h} -> {nw}x{nh} "
              f"({dst.stat().st_size / 1e6:.1f}MB)")

    manifest_path.write_text(json.dumps(manifest))
    print(f"manifest.json updated with png_lod for {len(manifest)} regions; "
          f"{saved_px * 4 / 1e9:.1f} GB of RGBA texture saved at overview zoom")


if __name__ == "__main__":
    main()

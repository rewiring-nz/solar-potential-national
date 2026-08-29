"""
Prototype: extract dominant straight roofline edges directly from the 0.1m
RGB imagery (Canny edge detection + probabilistic Hough transform), instead
of relying solely on the 1m DSM to define facet shape.

This is deliberately a standalone diagnostic, not wired into the real
pipeline yet -- the question being tested is whether image-based edges are
even a strong enough signal to build on before committing to integrating
them. Renders detected lines overlaid on the source imagery for one
building so the result can be checked by eye against the true roofline.

Usage: python src/roofline_prototype.py <building_id>
"""
import sys
import warnings
from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CANNY_LOW, CANNY_HIGH = 40, 120
HOUGH_THRESHOLD = 25
HOUGH_MIN_LINE_LENGTH_PX = 15  # ~1.5m at 0.1m/px
HOUGH_MAX_LINE_GAP_PX = 8


def detect_roofline_segments(rgb, building_mask):
    """rgb: HxWx3 uint8. building_mask: HxW bool (restrict edges to roughly
    the building footprint so eave shadows/neighbouring roofs don't
    dominate). Returns Nx4 array of (x1,y1,x2,y2) pixel-space segments."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    edges[~building_mask] = 0
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH_PX, maxLineGap=HOUGH_MAX_LINE_GAP_PX,
    )
    return edges, (lines.reshape(-1, 4) if lines is not None else np.empty((0, 4)))


def main(building_id):
    gdf = gpd.read_file(DATA_DIR / "building_outlines.geojson")
    row = gdf[gdf["building_id"] == building_id].iloc[0]
    imagery_ds = rasterio.open(DATA_DIR / "imagery_mosaic.tif")

    pad = 2
    minx, miny, maxx, maxy = row.geometry.bounds
    window = rasterio.windows.from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, imagery_ds.transform)
    arr = imagery_ds.read([1, 2, 3], window=window)
    rgb = np.moveaxis(arr, 0, -1)
    wt = imagery_ds.window_transform(window)

    # Rasterize the building footprint into this same pixel grid so edge
    # detection can be restricted to roughly the roof, not the street/lawn
    # around it (a building's own eave line is the target -- everything
    # else is noise for this purpose).
    from rasterio.features import rasterize
    building_mask = rasterize([(row.geometry, 1)], out_shape=rgb.shape[:2], transform=wt).astype(bool)
    building_mask = cv2.dilate(building_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    edges, segments = detect_roofline_segments(rgb, building_mask)
    print(f"Building #{building_id}: {len(segments)} line segments detected")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("Source imagery (0.1m)")
    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title(f"Canny edges (masked to footprint), {CANNY_LOW}-{CANNY_HIGH}")
    axes[2].imshow(rgb)
    for x1, y1, x2, y2 in segments:
        axes[2].plot([x1, x2], [y1, y2], color="#00ff88", linewidth=1.5)
    axes[2].set_title(f"Hough line segments ({len(segments)})")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out_path = DATA_DIR / f"roofline_prototype_{building_id}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main(int(sys.argv[1]))

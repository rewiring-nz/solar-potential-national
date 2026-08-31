"""
Find roof-plane boundaries in the 0.1 m imagery, not the 0.42 m LiDAR.

Why this exists (Josh, 27 Aug): "estimate the roof shape from the image data,
and then estimate the slopes based on that image. Then only if the image is too
confusing ... process a combination of lidar and image ... most rooftops do look
pretty clear in the imagery, so you might get most rooftop planes more
accurately defined, large roof changes, not small slivers, roof areas like they
are in the real world."

Everything measured yesterday points the same way:

- imagery is 0.1 m, LiDAR ~0.42 m. Four times the resolution exactly where
  LiDAR is weakest, which is edges.
- three separate failures wanted it: deck-vs-roof (audit_decks.py, three
  methods all failed on LiDAR alone), obstruction footprints drawn too wide,
  and ridges landing off the visible ridgeline.
- and the decisive one: the realism merge showed that slivers CANNOT be merged
  away after the fact. They are real planes the point cloud supports. Few large
  blocky faces have to come from the segmenter, and the point cloud does not
  carry the evidence to produce them.

The division of labour is the point. Imagery says WHERE the boundaries are --
a ridge is usually an obvious tonal step between two faces catching different
light. LiDAR says WHAT ANGLE each face sits at, which imagery cannot give at
all. Neither alone is enough.

This module is the first half only: boundaries. It is a prototype and is not
wired into anything.

Usage: python src/roof_from_imagery.py <building_id> [more ids...]
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
import shapely
from scipy import ndimage
from shapely.geometry import LineString, Polygon

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# A roof face is a region of consistent tone; a ridge is the step between two.
# Working in a smoothed gradient rather than raw pixels because roof cladding is
# ribbed, and the ribs are a far stronger local signal than the ridge is.
SMOOTH_PX = 3.0            # ~0.3 m: kills rib texture, keeps a ridge
MIN_FACE_M2 = 8.0          # below this it is not a face worth racking
MERGE_TONE_DELTA = 0.06    # neighbouring regions this close in tone are one face


def _read_roof(imagery, outline, pad_m=1.0):
    minx, miny, maxx, maxy = outline.bounds
    win = rasterio.windows.from_bounds(minx - pad_m, miny - pad_m,
                                       maxx + pad_m, maxy + pad_m, imagery.transform)
    rgb = imagery.read([1, 2, 3], window=win, boundless=True, fill_value=0).astype(float)
    wt = imagery.window_transform(win)
    grey = rgb.mean(axis=0) / 255.0
    rows, cols = np.mgrid[0:grey.shape[0], 0:grey.shape[1]]
    xs = wt.c + (cols + 0.5) * wt.a
    ys = wt.f + (rows + 0.5) * wt.e
    inside = shapely.contains_xy(outline, xs, ys)
    return grey, inside, xs, ys, abs(wt.a)


def faces_from_imagery(imagery, outline):
    """Segment the roof into tonal regions. Returns a list of (mask, mean_tone).

    Watershed on the gradient: ridges are gradient ridges, faces are the basins
    between them. This is the classic way to cut an image at its edges without
    needing the edges to form closed loops, which real ridgelines rarely do --
    they fade where two faces happen to catch similar light.
    """
    grey, inside, xs, ys, px = _read_roof(imagery, outline)
    if inside.sum() < 50:
        return [], None, None, px
    sm = ndimage.gaussian_filter(grey, SMOOTH_PX)
    gy, gx = np.gradient(sm)
    grad = np.hypot(gx, gy)

    # Seeds: local tone plateaus well away from any gradient.
    flat = grad < np.percentile(grad[inside], 40)
    seeds, nseed = ndimage.label(flat & inside)
    if nseed == 0:
        return [], grad, inside, px

    # Grow the seeds over the gradient. ndimage has no watershed, so this is a
    # priority flood done cheaply: assign every roof pixel to the seed whose
    # basin reaches it first through the LOWEST gradient path, approximated by
    # iterative dilation ordered by gradient threshold.
    labels = seeds.copy()
    for q in (50, 60, 70, 80, 90, 100):
        allow = inside & (grad <= np.percentile(grad[inside], q))
        for _ in range(6):
            grown = ndimage.grey_dilation(labels, size=3)
            take = (labels == 0) & allow & (grown > 0)
            if not take.any():
                break
            labels[take] = grown[take]
    labels[~inside] = 0

    area_px = MIN_FACE_M2 / (px * px)
    out = []
    for lab in range(1, labels.max() + 1):
        m = labels == lab
        if m.sum() < area_px:
            continue
        out.append((m, float(grey[m].mean())))
    # Merge neighbours of near-identical tone: one face split by a seam.
    merged = True
    while merged and len(out) > 1:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                if abs(out[i][1] - out[j][1]) > MERGE_TONE_DELTA:
                    continue
                if not (ndimage.binary_dilation(out[i][0]) & out[j][0]).any():
                    continue
                m = out[i][0] | out[j][0]
                tone = (out[i][1] * out[i][0].sum() + out[j][1] * out[j][0].sum()) / m.sum()
                out = [out[k] for k in range(len(out)) if k not in (i, j)] + [(m, tone)]
                merged = True
                break
            if merged:
                break
    return out, grad, inside, px


def main():
    import geopandas as gpd
    from src.region_build import area_paths
    paths = area_paths("pilot")
    gdf = gpd.read_file(paths["outlines"])
    imagery = rasterio.open(paths["dir"] / "imagery_mosaic.tif")
    for bid in (int(a) for a in sys.argv[1:]):
        row = gdf[gdf["building_id"] == bid]
        if row.empty:
            print(f"{bid}: not in outlines")
            continue
        o = row.iloc[0].geometry
        faces, grad, inside, px = faces_from_imagery(imagery, o)
        areas = sorted((m.sum() * px * px for m, _ in faces), reverse=True)
        print(f"{bid}: outline {o.area:6.0f} m2 -> {len(faces)} tonal faces, "
              f"areas {[round(a) for a in areas[:6]]}"
              f"{' ...' if len(areas) > 6 else ''}")
    imagery.close()


if __name__ == "__main__":
    main()

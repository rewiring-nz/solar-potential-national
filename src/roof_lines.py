"""
Candidate roof cut-lines detected in the 0.1 m imagery.

The partition places its cuts by sweeping offsets at 25 cm and keeping whichever
fits the LiDAR best. That is the wrong sensor for the job. The Queenstown point
cloud is 2021 at about 7.8 pts/m2 -- roughly 0.42 m between points, and fewer
returns on a roof -- so it cannot localise a ridge better than its own spacing.
The imagery is 0.1 m captured February-March 2026: a ridge is a sharp intensity
edge running the length of a roof, four times finer than the LiDAR spacing and
five years newer.

So let the imagery PROPOSE lines and the LiDAR DISPOSE of them. Every line found
here is only a candidate; the partition scores it exactly like a swept cut and
keeps it only if it explains the points better. A spurious line -- a gutter, a
shadow, a parked car, a path -- simply loses on score and costs nothing. That
asymmetry is what makes it safe to be generous here: the cost of a false line is
one scoring pass, and the cost of a missed line is a cut placed by guesswork.

Lines are returned as (angle_deg, offset) in the convention roof_partition._cut
already uses -- angle of the line's direction, and signed perpendicular distance
from the polygon's centroid -- so a detected ridge and a swept candidate are
interchangeable to the caller.
"""

import math
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import LineString, Point

warnings.filterwarnings("ignore")
# ...but never deprecations. A blanket ignore is exactly how 68 calls to
# shapely.vectorized -- an API documented for REMOVAL, under an unpinned
# shapely>=2.0 -- stayed invisible until 31 Aug. Third-party noise stays
# suppressed; a countdown to the pipeline breaking does not.
warnings.filterwarnings("default", category=DeprecationWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Canny + Hough was the first attempt and Josh's verdict on it was exact:
# "you are drawing lines that are not in the underlying image, and then even
# when there are clearly defined straight line ridges on the roof in the image,
# you are missing them." Both halves of that have a cause.
#
# FALSE LINES came from roof texture. Tiles, corrugations and shingle courses
# are strong, regular, straight edges at 0.1 m -- exactly what an edge detector
# is built to find. A bilateral filter removes them while keeping structural
# edges, because it smooths within regions and not across them.
#
# MISSED RIDGES came from two things. A ridge between two roof faces is often a
# gentle intensity STEP rather than a sharp edge -- the faces differ in
# brightness because they differ in angle to the sun -- and Canny's fixed
# thresholds drop those. LSD works from local gradient orientation instead and
# keeps them. And Hough scored each fragment separately, so a real ridge broken
# into six pieces by a chimney or texture scored six weak votes rather than one
# strong one; fragments of one line are now summed.
PAD_M = 2.0
CLAHE_CLIP = 2.0              # local contrast, to lift soft ridges out of a flat roof
CLAHE_GRID = 8
BILATERAL_D = 7               # texture suppression that does not blur across a ridge
BILATERAL_SIGMA_COLOR = 40
BILATERAL_SIGMA_SPACE = 7
LSD_SCALE = 0.8
MIN_FRAGMENT_M = 0.8          # below this a segment is noise, not part of anything

MIN_LENGTH_M = 2.0            # total evidence, summed over fragments of one line
BOUNDARY_EXCLUSION_M = 0.8    # a segment hugging the footprint edge is the eave or a
# gutter, and the footprint already provides that edge exactly -- re-cutting on a
# blurry copy of it can only be worse than the surveyed line

# Two candidates this close in angle AND in perpendicular offset are the same
# physical line seen twice (Hough returns fragments), not two roof features.
CLUSTER_ANGLE_DEG = 6.0
CLUSTER_OFFSET_M = 0.7

MAX_CANDIDATES = 24           # scoring is O(candidates); a roof with more real lines
# than this is past what the partition's face budget would use anyway


def _angle_offset(seg, cx, cy):
    """LineString -> (angle_deg in [0,180), signed perpendicular offset from
    (cx, cy)), matching roof_partition._cut's convention."""
    (x1, y1), (x2, y2) = seg.coords[0], seg.coords[-1]
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
    theta = np.radians(ang)
    n = np.array([-np.sin(theta), np.cos(theta)])
    off = float((np.array([x1, y1]) - np.array([cx, cy])) @ n)
    return float(ang), off


def _merge_collinear(cands, keep_length=False):
    """Fragments of one physical line become one line carrying their TOTAL
    length.

    This is the fix for missed ridges. A ridge crossed by a chimney, a vent or
    a patch of texture arrives as several short segments; scoring them
    separately buries a real 12 m ridge under a 3 m gutter. Summed, it outranks
    everything on the roof, which is what it should do."""
    clusters = []   # [angle, offset, total_length, weight] -- angle/offset length-weighted
    for ang, off, length in sorted(cands, key=lambda c: -c[2]):
        hit = None
        for c in clusters:
            da = min(abs(ang - c[0]), 180.0 - abs(ang - c[0]))
            if da < CLUSTER_ANGLE_DEG and abs(off - c[1]) < CLUSTER_OFFSET_M:
                hit = c
                break
        if hit is None:
            clusters.append([ang, off, length, length])
        else:
            w = hit[3] + length
            hit[0] = (hit[0] * hit[3] + ang * length) / w
            hit[1] = (hit[1] * hit[3] + off * length) / w
            hit[2] += length
            hit[3] = w
    clusters = [c for c in clusters if c[2] >= MIN_LENGTH_M]
    clusters.sort(key=lambda c: -c[2])
    out = [(c[0], c[1], c[2]) for c in clusters[:MAX_CANDIDATES]]
    return out if keep_length else [(a, o) for a, o, _ in out]


# How much evidence before a line is trusted WITHOUT the LiDAR agreeing. It has
# to scale with the roof: a 4 m crease across a 211 m2 house is a primary
# feature, the same line on a 743 m2 commercial roof is a detail. Measured, the
# lines this admits are the ones a person points at -- 7 Anderson Heights' hip
# creases (4.3-7.1 m on a 211 m2 roof), 5 Isle St's two main ridges (10.3 and
# 9.9 m) and nothing else on that roof, 29 Edinburgh Dr's three (13.3, 12.9,
# 10.5 m).
STRONG_LINE_MIN_M = 4.0
STRONG_LINE_AREA_COEF = 0.30      # of sqrt(roof area)
MAX_STRONG_LINES = 8              # a roof does not have many primary creases


def strong_roof_lines(imagery_ds, footprint):
    """Only the lines long enough to act on WITHOUT the LiDAR agreeing.

    Ordinary candidates are offered to the partition and kept only if they
    improve the fit. That is the right test when both sensors can see the
    feature, and useless when only one can. 7 Anderson Heights is the case: two
    hipped sections whose creases are unmistakable in the imagery and almost
    absent from the point cloud, which is near-flat across the whole roof. Every
    LiDAR-scored candidate there is rejected because cutting does not improve a
    fit that was never wrong about the height -- so the faces run straight over
    both hips and the panels follow.

    A line carrying this much total evidence is not noise, and on a roof the
    LiDAR cannot resolve it is the only evidence there is."""
    cands = _merge_collinear(_raw_fragments(imagery_ds, footprint), keep_length=True)
    bar = max(STRONG_LINE_MIN_M, STRONG_LINE_AREA_COEF * math.sqrt(max(footprint.area, 1.0)))
    strong = [(a, o) for a, o, ln in cands if ln >= bar]
    return strong[:MAX_STRONG_LINES]


def _raw_fragments(imagery_ds, footprint):
    """Every straight fragment found on this roof, as (angle, offset, length)."""
    if imagery_ds is None or footprint.is_empty:
        return []
    minx, miny, maxx, maxy = footprint.bounds
    try:
        window = rasterio.windows.from_bounds(minx - PAD_M, miny - PAD_M,
                                              maxx + PAD_M, maxy + PAD_M,
                                              imagery_ds.transform)
        arr = imagery_ds.read([1, 2, 3], window=window)
    except Exception:
        return []
    if arr.size == 0 or arr.shape[1] < 8 or arr.shape[2] < 8:
        return []

    rgb = np.moveaxis(arr, 0, -1).astype(np.uint8)
    wt = imagery_ds.window_transform(window)

    # Confine to the roof itself. Without this the strongest lines on the page
    # are kerbs, paths and neighbouring rooflines.
    mask = rasterize([(footprint, 1)], out_shape=rgb.shape[:2], transform=wt).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                           tileGridSize=(CLAHE_GRID, CLAHE_GRID)).apply(gray)
    gray = cv2.bilateralFilter(gray, BILATERAL_D, BILATERAL_SIGMA_COLOR,
                               BILATERAL_SIGMA_SPACE)
    gray = np.where(mask > 0, gray, 0).astype(np.uint8)

    try:
        lsd = cv2.createLineSegmentDetector(scale=LSD_SCALE)
        detected = lsd.detect(gray)[0]
    except Exception:
        return []
    if detected is None or len(detected) == 0:
        return []
    lines = detected.reshape(-1, 4)

    cx, cy = footprint.centroid.x, footprint.centroid.y
    boundary = footprint.exterior
    out = []
    for x1, y1, x2, y2 in lines:
        wx1, wy1 = wt * (float(x1), float(y1))
        wx2, wy2 = wt * (float(x2), float(y2))
        seg = LineString([(wx1, wy1), (wx2, wy2)])
        if seg.length < MIN_FRAGMENT_M:
            continue
        if (boundary.distance(Point(wx1, wy1)) < BOUNDARY_EXCLUSION_M
                and boundary.distance(Point(wx2, wy2)) < BOUNDARY_EXCLUSION_M):
            continue          # tracing the eave; the surveyed outline is better
        ang, off = _angle_offset(seg, cx, cy)
        out.append((ang, off, seg.length))
    return out


def roof_line_candidates(imagery_ds, footprint):
    """(angle_deg, offset) candidates for cutting this footprint. [] if the
    imagery is missing or nothing survives -- the caller falls back to sweeping."""
    return _merge_collinear(_raw_fragments(imagery_ds, footprint))

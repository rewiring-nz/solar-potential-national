"""
Build a roof as a PLANAR PARTITION of the surveyed footprint, cut only by
straight lines, refined until the planes explain the LiDAR.

The design constraint is Josh's, stated directly:

    "Most roof shapes are relatively simple. They aren't fuzzy, they are
    straight lines, generally a few different angles."

    "Those unique shapes are still generally made up of the same principles as
    household roofs, just more of them on the same building footprint. For
    example a hotel of apartments with many household like roofs in the same
    building footprint. Or a big warehouse roof with multiple angle roof
    sections... clear flat angled sections at differing pitches, with differing
    cut offs, but in general the roof shape principles remain the same."

    "The roof will almost never be some type of organic shape."

Two consequences, and they are the whole module.

STRAIGHT BY CONSTRUCTION. Every facet boundary is either a footprint edge
(surveyed by LINZ, already straight) or a cut line. No boundary is ever traced
from a raster. That alone removes the defect Josh has now reported four times:
the shipped facets on 29 Edinburgh Dr carry 1335-1835 vertices each on a roof
that is four rectangles, and on 1/5 Sydney St 874-1035 each. A partition cannot
produce those shapes -- the vertex count is bounded by the number of cuts.

COMPLEXITY EARNED, NOT ASSUMED. Fitting one template per building would do
exactly what Josh warned against -- "you should not fit a simple roof when the
underlying roof is actually more complex". So nothing is assumed about how many
faces a roof has. A region is kept whole when one plane already explains its
points; it is cut when a plane does not, and each half is then asked the same
question. A simple gable stops after one cut. A hotel keeps going. The stopping
rule is fit quality, so complexity is spent only where the roof actually has it.

Cut directions come from the footprint itself. Roofs are built on walls, so
ridges, hips and valleys run parallel or perpendicular to the walls below far
more often than not, and the surveyed outline is a better source for those
angles than anything recoverable from a 5.7 pts/m2 cloud.
"""

import sys
import warnings
from pathlib import Path

import time

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split as shapely_split, snap, unary_union

warnings.filterwarnings("ignore")
# ...but never deprecations. A blanket ignore is exactly how 68 calls to
# shapely.vectorized -- an API documented for REMOVAL, under an unpinned
# shapely>=2.0 -- stayed invisible until 31 Aug. Third-party noise stays
# suppressed; a countdown to the pipeline breaking does not.
warnings.filterwarnings("default", category=DeprecationWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# A facet is accepted when this share of its points lie within the band below.
# 0.15 m is a little over twice the DSM's own noise, so a genuine plane clears
# it easily and a plane spanning two faces cannot.
ACCEPT_INLIER = 0.85
INLIER_BAND_M = 0.15

MIN_FACET_M2 = 6.0          # below ~3 panels a face is not worth racking
MIN_PIECE_M2 = 4.0          # a cut may not leave a sliver smaller than this
# Depth alone is the WRONG stopping rule and it was quietly ruining the hardest
# roofs. Cuts do not halve a region evenly -- a ridge shaves a strip off one
# side -- so a large remainder can descend seven levels while barely shrinking,
# exhaust its budget and be returned whole. On 1/5 Sydney St that surrendered a
# 138 m2 face at 15% on-plane, 40% of the footprint, even though _best_cut had
# a 25% cut available for it. Depth is now generous and the real brakes are
# size and a per-building face cap, both of which track what is actually being
# spent rather than how many times the function has recursed.
MAX_DEPTH = 14
MAX_FACES = 60              # a hotel needs many; nothing real needs more
MIN_POINTS = 25

# A wall is not a roof. 26 Panorama Terrace is a stepped house and the pipeline
# put 36.8 m2 of it -- 31% of the modelled roof -- on faces of 42.3 and 44.5
# degrees, which on that building are the risers between levels. 1 Church St had
# a 10.2 m2 face at 42.8 degrees fitting its own points at 29.6%. District-wide
# 10.3% of every panel, carrying 9.2% of the claimed generation, sits above 35
# degrees.
#
# But slope alone must not disqualify a face, and the measurement says why:
# across 510 faces from 90 random pilot buildings, faces at 40 degrees or more
# fit under 70% in 32% of cases against 11% for shallower ones -- so steep faces
# are three times as likely to be wrong, and two thirds of them are still fine.
# Queenstown has genuinely steep roofs. Josh, asked to choose: drop a face only
# when it is BOTH steep and badly fitted.
STEEP_FACE_DEG = 40.0
STEEP_FACE_MIN_FIT = 0.70

# A WELL-FITTED wall. The filter above cannot catch the real risers, and the
# measurement on 26 Panorama Terrace shows why: its two step risers sit at 42.3
# and 44.5 degrees and fit 89.0% and 89.8% -- a wall is a perfect plane, so the
# better the wall, the higher the fit. What separates a riser from a steep roof
# is its RUN: the horizontal extent along its own gradient. A riser is a strip
# standing up -- its z-range divided by tan(slope) is the height of one storey
# step over its pitch, 1.4-2.5 m on every riser measured -- while a genuine
# steep roof face runs 3 m or more downhill. Measured runs: Panorama risers
# 1.6 m and 2.5 m, 15B Frankton's 37.5-degree strip 1.4 m.
STEEP_RUN_DEG = 35.0         # faces this steep are tested as possible risers
STEEP_MIN_RUN_M = 3.0        # a riser has less downhill run than this
RISER_SLOPE_GAP_DEG = 15.0   # ...and every face it touches is this much flatter

# A wall-clock ceiling on the cut search for ONE building.
#
# Bounding the search by a count of candidate cuts was tried first and it is the
# wrong meter: the cost of one evaluation is a polygon split plus two plane fits
# over whatever points fall in the region, so it scales with point density. A
# 219 m2 house spends 4,782 evaluations and finishes in under a second, while
# 4722059 (16,010 m2, 226 m across, 73,226 points) spends ~3 ms on each one. Any
# count low enough to bound the big roof also truncates ordinary buildings --
# measured: a 12,000 cap left a 1,725 m2 building clipped mid-search while it
# had been finishing comfortably in 14 s.
#
# Time bounds the actual pathology and nothing else. Every roof that completes
# quickly is bit-for-bit unchanged, because the deadline is never reached; only
# a roof that is genuinely running long gets stopped, and it keeps the faces it
# has already found -- still watertight, still a valid partition, just coarser.
# Anything that ends up badly fitted is withheld by the confidence gate in
# build_layout_geojson rather than published wrong.
CUT_TIME_BUDGET_S = 60.0     # base; scaled up with roof area, see partition_roof.
# 60s flat starved exactly the roofs Josh cares most about: 32 Frankton Road
# (4,032 m2, 31,776 points) hit the deadline before the search made one good
# cut and shipped as a 5,128 m2 sheet fitting 12.9% -- with 1,138 panels on it.
# The budget exists to stop a stall, not to decide segmentation quality, so it
# scales with the work: a house keeps 60s it never uses, a big commercial roof
# gets up to 240s.
CUT_TIME_BUDGET_MAX_S = 240.0
CUT_TIME_PER_M2_S = 0.05
_cut_deadline = [None]
_cut_evals = [0]
CUT_BUDGET_EXHAUSTED = [0]

ANGLE_TOL_DEG = 4.0         # footprint edge directions this close are one direction
MAX_OFFSETS_PER_DIRECTION = 120   # cap on how many cut positions one direction is
# tried at. The sweep costs a polygon split and two plane fits PER POSITION, at
# every node of the recursion, so its cost grows with the span of the roof --
# fine at house scale, ruinous above it. Building 4722059 on Frankton Flats is
# 16,010 m2: ~200 m across is ~800 positions per direction per node, and it ran
# for over an hour at 100% CPU without finishing, stalling two full rebuilds.
# The cap only binds above a ~30 m span (120 * 0.25 m), so every house and most
# commercial roofs sweep at exactly the old 0.25 m and are bit-for-bit
# unchanged; only genuinely large roofs are coarsened, and 0.25 m precision on a
# 200 m warehouse was never meaningful anyway.
OFFSET_STEP_M = 0.25        # how finely each candidate direction is swept. A ridge cut
# landing half a metre off leaves a strip of the WRONG plane on both sides of it,
# which drags the fit down on exactly the roofs whose structure was found correctly.
# A cut only has to help at all, not pay for itself immediately. Requiring a
# real gain from each single cut was tried and it makes the recursion blind:
# on 1/5 Sydney St, a twelve-unit roof, the whole footprint fits one plane at
# 13% and the BEST available single cut reaches only 16%, because both halves
# are still multi-plane messes. That roof needs about ten cuts before the fit
# improves sharply, and a one-step-lookahead test can never see past the first.
# So the stopping rule is "this region is now explained" -- or too small, or too
# deep -- and over-splitting is handled afterwards by _merge_bridgeable, which
# undoes any cut a panel could lie across anyway.
MIN_SPLIT_GAIN = 0.005
# How much usable area a cut may cost, PER POINT of fit it buys. An absolute cap
# was tried first and is wrong: one legitimate ridge cut across a 15 m roof
# already costs about 7 m2 of setback, so any flat cap tight enough to stop
# over-fragmentation also stops the first honest cut, and every roof collapsed
# to a single facet at 38% on-plane.
#
# Scaling it by the fit gain is the honest trade. A cut that takes a roof from
# 30% to 90% on-plane has fixed the building and has earned a lot of setback; a
# cut that buys two points has not earned any.
# Swept, not guessed. 0.5 was shipped first and was badly wrong: it cost 20
# points of fit on complex roofs (64% on-plane against 86% with the rule off)
# to buy a facet count that was not needed -- 47 Stanley St fell from 22 facets
# at 96% to 4 at 53%, which is under-modelling a roof, not making it blocky.
#
#   cost/fit    hard roofs              random roofs
#   0.5         64% on-plane,  4 facets   83%,  4
#   2.0         84%,          11          93%,  6
#   unlimited   86%,          12          93%,  6
#
# 2.0 takes essentially all the available fit while houses still come back at
# about six faces, which is the blockiness Josh asked for. Past 2.0 nothing
# changes, so the rule is only ever binding on the roofs where it should be.
SETBACK_COST_PER_FIT = 2.0

# A "fold" test was tried here -- treat a face carrying points far off its own
# plane as containing a physical drop, and cut it regardless of setback cost.
# It does not work, and the reason is worth keeping: the signal does not
# separate the cases. 5 Isle St, which Josh confirms is correctly three faces,
# has a face with 10.6% of its points more than 0.5 m off plane; 7 Anderson
# Heights, which he says is wrong, has 10.0% and 10.2%. Any threshold that cuts
# one cuts the other. Nor does WHERE those points sit help -- on both roofs they
# cluster within half a metre of a facet edge, which is just bleed from the
# neighbouring plane.


def _fit_plane(pts):
    """Least-squares plane through points, as (a, b, c) with z = ax + by + c.

    Solved about the points' own centroid, then shifted back. Solving on raw
    NZTM coordinates -- x near 1.2 million, y near 5 million, against a column
    of ones -- is a condition number around 1e6, and it does not merely lose a
    little precision: it silently returns planes that do not fit their own
    points. Two faces of 5 Isle St measured 0.1 degrees apart with a 0.00 m step
    at their join, and the plane fitted to their union scored 16% on-plane
    against 99% for each of them separately, which blocked a merge that should
    obviously have happened and left that roof at 5 faces where Josh counted 3."""
    x0, y0 = pts[:, 0].mean(), pts[:, 1].mean()
    A = np.column_stack([pts[:, 0] - x0, pts[:, 1] - y0, np.ones(len(pts))])
    coef, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a, b, float(coef[2]) - a * x0 - b * y0


def _fit_plane_robust(pts, iterations=6):
    """Trimmed fit, so a chimney or a parapet does not tilt the whole face."""
    keep = np.ones(len(pts), bool)
    plane = _fit_plane(pts)
    for _ in range(iterations):
        r = pts[:, 2] - (plane[0] * pts[:, 0] + plane[1] * pts[:, 1] + plane[2])
        nxt = np.abs(r - np.median(r[keep])) < INLIER_BAND_M * 2
        if nxt.sum() < MIN_POINTS or (nxt == keep).all():
            break
        keep = nxt
        plane = _fit_plane(pts[keep])
    return plane


def _inlier_fraction(pts, plane):
    r = pts[:, 2] - (plane[0] * pts[:, 0] + plane[1] * pts[:, 1] + plane[2])
    return float((np.abs(r - np.median(r)) < INLIER_BAND_M).mean())


def _usable(poly, setback=None):
    """Area left after the ridge setback -- what panel packing actually gets."""
    setback = config.RIDGE_SETBACK_M if setback is None else setback
    if poly.is_empty:
        return 0.0
    try:
        return float(poly.buffer(-setback).area)
    except Exception:
        return 0.0


def _points_in(poly, pts):
    if len(pts) == 0:
        return pts
    return pts[shapely.contains_xy(poly, pts[:, 0], pts[:, 1])]


def _edge_directions(poly):
    """Distinct directions of the footprint's own edges, longest first.

    Roofs are built on walls: ridges and hips run parallel or perpendicular to
    the outline far more often than not, and the LINZ outline is surveyed, so
    these angles are exact in a way nothing recovered from the point cloud is."""
    coords = np.asarray(poly.exterior.coords)
    segs = coords[1:] - coords[:-1]
    lens = np.hypot(segs[:, 0], segs[:, 1])
    angs = np.degrees(np.arctan2(segs[:, 1], segs[:, 0])) % 180.0
    order = np.argsort(-lens)
    out = []
    for i in order:
        if lens[i] < 1.0:
            continue
        a = angs[i]
        if all(min(abs(a - b), 180 - abs(a - b)) > ANGLE_TOL_DEG for b in out):
            out.append(float(a))
    # Every wall direction, its perpendicular, and its two diagonals. The
    # diagonals are not decoration: a hip BISECTS the corner between two walls,
    # so a hip line runs at 45 degrees to both, and without them the partition
    # cannot cut a hip roof at all -- it is forced to approximate one with
    # rectangular cuts, which is why 5 Isle St stalled at two faces and 45%
    # on-plane. Josh named this exact geometry: "a 45 degree mitre type roof
    # joint... they are very common on roof geometry".
    perp = [(a + 90.0) % 180.0 for a in out]
    diag = [(a + 45.0) % 180.0 for a in out] + [(a + 135.0) % 180.0 for a in out]
    seen, uniq = [], []
    for a in out + perp + diag:
        if all(min(abs(a - b), 180 - abs(a - b)) > ANGLE_TOL_DEG for b in seen):
            seen.append(a)
            uniq.append(a)
    return uniq


def _cut(poly, angle_deg, offset):
    """Split a polygon with an infinite line at this angle and offset."""
    theta = np.radians(angle_deg)
    d = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-d[1], d[0]])
    c = np.array(poly.centroid.coords[0])
    span = max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]) * 2 + 10
    mid = c + n * offset
    line = LineString([mid - d * span, mid + d * span])
    try:
        parts = list(shapely_split(poly, line).geoms)
    except Exception:
        return []
    return [p for p in parts if isinstance(p, Polygon) and p.area >= MIN_PIECE_M2]


def _score(poly, pts):
    """Area-weighted inlier fraction if this region were one facet."""
    sub = _points_in(poly, pts)
    if len(sub) < MIN_POINTS:
        return None, 0.0
    plane = _fit_plane_robust(sub)
    return plane, _inlier_fraction(sub, plane)



# A fold is not "many points off the plane" -- that test does not work, and the
# numbers say so plainly. Measured on the two roofs Josh judged opposite ways:
#
#   7 Anderson (he calls WRONG)  faces at 11.0% and 8.0% of points beyond 0.5 m
#   5 Isle     (he calls RIGHT)  faces at  2.3%, 6.6% and 10.6% beyond 0.5 m
#
# 5 Isle's correct 44 m2 face has a LONGER tail than Anderson's wrong 64.6 m2
# face. Any threshold that cuts one shatters the other, which is why the earlier
# fold-fraction and fold-location attempts were abandoned.
#
# What separates them is the SHAPE of the off-plane set. On 5 Isle the low
# points are rooftop clutter: compact blobs scattered over a surface that really
# is one plane. On 7 Anderson they are the valleys between three pyramid hips --
# long connected bands running clean across the face. So the test is spatial: a
# fold is a connected low region that is elongated AND spans most of the way
# across the region it sits in. Clutter is neither.
FOLD_DROP_M = 0.40           # how far below the plane counts as "low"
FOLD_CELL_M = 0.6            # grid the low points onto this, then label components
FOLD_MIN_CELLS = 8           # smaller connected sets are clutter, not structure
FOLD_MIN_SPAN_SHARE = 0.55   # a fold runs most of the way across the face
FOLD_MIN_ELONGATION = 2.2    # and it is long and thin, not a blob


def _fold_evidence(poly, pts, plane):
    """True when the points below `plane` form a band crossing `poly`.

    Detects a roof that drops through the middle of a face -- the defect Josh
    reported repeatedly on 7 Anderson Heights -- while ignoring the scattered
    low clutter that sits on a face which really is one plane."""
    sub = _points_in(poly, pts)
    if len(sub) < MIN_POINTS:
        return False
    r = sub[:, 2] - (plane[0] * sub[:, 0] + plane[1] * sub[:, 1] + plane[2])
    r = r - np.median(r)
    low = sub[r < -FOLD_DROP_M]
    if len(low) < FOLD_MIN_CELLS:
        return False
    try:
        from scipy import ndimage
    except Exception:
        return False
    minx, miny, maxx, maxy = poly.bounds
    nx = max(2, int(np.ceil((maxx - minx) / FOLD_CELL_M)))
    ny = max(2, int(np.ceil((maxy - miny) / FOLD_CELL_M)))
    if nx * ny > 400000:
        return False
    grid = np.zeros((ny, nx), dtype=bool)
    ix = np.clip(((low[:, 0] - minx) / FOLD_CELL_M).astype(int), 0, nx - 1)
    iy = np.clip(((low[:, 1] - miny) / FOLD_CELL_M).astype(int), 0, ny - 1)
    grid[iy, ix] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return False
    face_span = max(maxx - minx, maxy - miny)
    for lab in range(1, n + 1):
        ys, xs = np.nonzero(labels == lab)
        if len(xs) < FOLD_MIN_CELLS:
            continue
        px = xs * FOLD_CELL_M + minx
        py = ys * FOLD_CELL_M + miny
        c = np.column_stack([px - px.mean(), py - py.mean()])
        # principal axes of the connected low region
        try:
            axes = np.linalg.svd(c, full_matrices=False)[2]
        except Exception:
            continue
        proj = c @ axes.T
        length = proj[:, 0].max() - proj[:, 0].min()
        width = proj[:, 1].max() - proj[:, 1].min() if proj.shape[1] > 1 else 0.0
        if length < FOLD_MIN_SPAN_SHARE * face_span:
            continue
        if width > 1e-6 and length / max(width, FOLD_CELL_M) < FOLD_MIN_ELONGATION:
            continue
        return True
    return False



# Where the ridge drops, the roof changes section, and that is a cut whether or
# not the plane fit asks for one.
#
# Josh, twice, on 7 Anderson Heights: "two panel planes placed and both
# overlapping a roof ridge where it drops in the middle", and later "you are
# also not accounting for the dip in the middle we have already spoken about."
# Measured on that roof, taking the 97th percentile of height in 1 m slices
# along the building's long axis: the ridge sits at 381.42 m from -8.5 to -3.5,
# falls to 380.26 m at -0.5, and returns to 381.47 m from 1.5 to 6.5. A 1.16 m
# dip in the middle of a 21 m roof, with the north slope running straight over
# it as one 64.6 m2 face.
#
# The plane fit cannot see this on its own: both sections have the same pitch
# and aspect, so a plane through both scores well. The signal is in the upper
# envelope, not the residuals, which is why it is measured separately here and
# offered to _best_cut as an extra candidate position rather than left to the
# generic sweep to stumble on.
RIDGE_SLICE_M = 1.0          # along-axis resolution of the ridge profile
RIDGE_TOP_PCT = 97           # what counts as "the ridge" inside one slice
RIDGE_DROP_M = 0.45          # a dip this far below both neighbouring peaks is a section change
RIDGE_EDGE_MARGIN_M = 2.0    # ignore dips at the very ends, those are the hips


def _ridge_drop_offsets(poly, pts):
    """Positions along the region's long axis where the ridge line dips.

    Returns (angle_deg, [offsets]) for a cut perpendicular to that axis, in the
    same convention _cut uses, or None when the ridge is continuous."""
    sub = _points_in(poly, pts)
    if len(sub) < 4 * MIN_POINTS:
        return None
    try:
        rect = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)
    except Exception:
        return None
    e = rect[1:] - rect[:-1]
    L = np.hypot(e[:, 0], e[:, 1])
    if len(L) == 0 or L.max() <= 0:
        return None
    i = int(np.argmax(L))
    u = e[i] / L[i]
    c = np.array(poly.centroid.coords[0])
    s = (sub[:, :2] - c) @ u
    lo, hi = s.min(), s.max()
    if hi - lo < 4 * RIDGE_SLICE_M:
        return None
    edges = np.arange(lo, hi + 1e-9, RIDGE_SLICE_M)
    mids, tops = [], []
    for k in range(len(edges) - 1):
        m = (s >= edges[k]) & (s < edges[k + 1])
        if m.sum() < 5:
            continue
        mids.append(0.5 * (edges[k] + edges[k + 1]))
        tops.append(np.percentile(sub[m, 2], RIDGE_TOP_PCT))
    if len(tops) < 5:
        return None
    mids, tops = np.array(mids), np.array(tops)
    offsets = []
    for k in range(1, len(tops) - 1):
        if mids[k] - lo < RIDGE_EDGE_MARGIN_M or hi - mids[k] < RIDGE_EDGE_MARGIN_M:
            continue
        if tops[k] > tops[k - 1] or tops[k] > tops[k + 1]:
            continue                      # not a local minimum
        drop = min(tops[:k].max() - tops[k], tops[k + 1:].max() - tops[k])
        if drop >= RIDGE_DROP_M:
            offsets.append(float(mids[k]))
    if not offsets:
        return None
    # _cut takes the direction of the LINE; perpendicular to the long axis
    angle = float((np.degrees(np.arctan2(u[1], u[0])) + 90.0) % 180.0)
    n = np.array([-np.sin(np.radians(angle)), np.cos(np.radians(angle))])
    # express each offset in the same projection _cut uses
    return angle, [float(o * (u @ n)) for o in offsets]


def _best_cut(poly, pts, base_score):
    """The straight line that best explains this region as two planes."""
    if _cut_deadline[0] is not None and time.process_time() > _cut_deadline[0]:
        return None
    best = None

    # A ridge drop is a section change and gets its own candidate positions,
    # scored the same way as every other cut so it only wins if it explains the
    # surface better.
    rd = _ridge_drop_offsets(poly, pts)
    if rd is not None:
        angle, offs = rd
        for off in offs:
            parts = _cut(poly, angle, off)
            if len(parts) < 2:
                continue
            tot = num = 0.0
            worst = 1.0
            ok = True
            for part in parts:
                pl, sc = _score(part, pts)
                if pl is None:
                    ok = False
                    break
                num += sc * part.area
                tot += part.area
                worst = min(worst, sc)
            if ok and tot > 0:
                combined = num / tot
                if combined > base_score + MIN_SPLIT_GAIN and (best is None or combined > best[0]):
                    best = (combined, parts, worst)

    for angle in _edge_directions(poly):
        theta = np.radians(angle)
        n = np.array([-np.sin(theta), np.cos(theta)])
        coords = np.asarray(poly.exterior.coords)
        c = np.array(poly.centroid.coords[0])
        proj = (coords - c) @ n
        lo, hi = proj.min(), proj.max()
        if hi - lo < 2 * OFFSET_STEP_M:
            continue
        step = max(OFFSET_STEP_M, (hi - lo) / MAX_OFFSETS_PER_DIRECTION)
        for off in np.arange(lo + step, hi - step + 1e-9, step):
            _cut_evals[0] += 1
            if _cut_deadline[0] is not None and time.process_time() > _cut_deadline[0]:
                CUT_BUDGET_EXHAUSTED[0] += 1
                return best
            parts = _cut(poly, angle, float(off))
            if len(parts) < 2:
                continue
            tot = num = 0.0
            ok = True
            for part in parts:
                pl, sc = _score(part, pts)
                if pl is None:
                    ok = False
                    break
                num += sc * part.area
                tot += part.area
            if not ok or tot <= 0:
                continue
            combined = num / tot
            if combined > base_score + MIN_SPLIT_GAIN and (best is None or combined > best[0]):
                best = (combined, parts)
    return best


def _cut_on_line(poly, A, B, C, cx, cy):
    """Split a polygon along A(x-cx) + B(y-cy) + C = 0.

    Everything is in coordinates local to (cx, cy), and both reasons are
    NZTM's fault. Anchoring the line at its closest point to the ORIGIN puts
    that anchor about 5,000 km away, so a line segment a few tens of metres
    long never reaches the building and the split silently does nothing. And
    a plane's intercept is its height at x=0, y=0, which for NZTM is an
    astronomical number, so differencing two of them loses all the precision
    that matters. Both vanish once the origin is the polygon itself."""
    n = np.hypot(A, B)
    if n < 1e-9:
        return []
    d = np.array([-B, A]) / n              # direction along the line
    origin = np.array([cx, cy])
    pt0 = origin - np.array([A, B]) * (C / (n ** 2))   # nearest point ON the line
    span = max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]) * 2 + 10
    line = LineString([pt0 - d * span, pt0 + d * span])
    try:
        parts = list(shapely_split(poly, line).geoms)
    except Exception:
        return []
    return [q for q in parts if isinstance(q, Polygon) and q.area >= MIN_PIECE_M2]


def _refine_cut(poly, pts, parts):
    """Move a swept cut onto the two planes' own intersection line.

    The sweep can only place a cut to within OFFSET_STEP_M, and a ridge that
    lands even 25 cm off leaves a strip of the WRONG plane on both sides of it,
    which is what kept fit lagging while structure was already correct. But two
    planes that meet do so along an exact line -- that line IS the ridge or hip,
    and it is available in closed form. Solve for it and re-cut there.

    Only for planes that actually intersect: near-parallel faces at different
    heights are a step, not a fold, and their intersection is meaningless or
    infinitely far away, so the swept cut stands."""
    if len(parts) != 2:
        return parts
    pa, _ = _score(parts[0], pts)
    pb, _ = _score(parts[1], pts)
    if pa is None or pb is None:
        return parts
    cx, cy = poly.centroid.x, poly.centroid.y
    A, B = pa[0] - pb[0], pa[1] - pb[1]
    # height gap between the two planes AT the centroid, not at x=0,y=0
    C = ((pa[0] * cx + pa[1] * cy + pa[2]) - (pb[0] * cx + pb[1] * cy + pb[2]))
    if np.hypot(A, B) < 1e-3:
        return parts        # parallel: a step in height, keep the swept cut
    refined = _cut_on_line(poly, A, B, C, cx, cy)
    if len(refined) < 2:
        return parts

    def weighted(ps):
        tot = num = 0.0
        for q in ps:
            _, sc = _score(q, pts)
            num += sc * q.area
            tot += q.area
        return num / tot if tot else 0.0

    return refined if weighted(refined) > weighted(parts) else parts


# Big planes first; their intersections ARE the roof lines.
#
# Josh, after seeing an edge detector draw a maze over a simple roof: "You need
# a way to clearly detect and define big flat planes that make up roof shapes,
# generally all these planes are large in size, and connect smoothly at angled
# edges most of the time."
#
# That inverts the problem. Hunting edges in imagery and inferring planes from
# them is backwards and fails in both directions -- texture produces lines that
# are not roof features, and a real ridge that is a soft intensity step
# produces no line at all. But two planes that meet do so along their exact
# analytic intersection: no detection, no threshold, no false positives. Find
# the planes and the ridges, hips and valleys come out for free, straight and
# in the right place.
#
# The size prior is the whole point and is what previous attempts at this got
# wrong -- roof_reconstruct fitted planes to whatever the points supported and
# shattered roofs into strips. A plane has to be BIG to exist at all here.
PLANE_MIN_AREA_M2 = 12.0        # a face smaller than ~6 panels is not a roof plane
PLANE_MIN_FOOTPRINT_SHARE = 0.04
PLANE_TOL_M = 0.20              # a point this close to a plane is on it
PLANE_MAX = 10                  # real roofs are simple; past this it is not planes any more
PLANE_RANSAC_ITERS = 250
PLANE_SAMPLE_RADIUS_M = 6.0     # 3-point samples drawn locally, or a plane gets fitted
# through three points on three different faces and describes nothing


def _detect_large_planes(pts, footprint, rng, max_planes=None):
    """Greedy RANSAC for the few LARGE planes a roof is actually made of.

    Take the best-supported plane, remove its points, repeat -- stopping as soon
    as the best remaining plane is too small to be a roof face rather than
    grinding on until every leftover point has one."""
    if len(pts) < MIN_POINTS:
        return []
    min_area = max(PLANE_MIN_AREA_M2, PLANE_MIN_FOOTPRINT_SHARE * footprint.area)
    pt_area = footprint.area / max(len(pts), 1)      # plan area each point stands for
    min_pts = max(MIN_POINTS, int(min_area / max(pt_area, 1e-9)))

    remaining = pts
    planes = []
    tree = None
    cap = PLANE_MAX if max_planes is None else max_planes
    while len(planes) < cap and len(remaining) >= min_pts:
        from scipy.spatial import cKDTree
        tree = cKDTree(remaining[:, :2])
        best, best_n = None, 0
        for _ in range(PLANE_RANSAC_ITERS):
            i = rng.integers(len(remaining))
            near = tree.query_ball_point(remaining[i, :2], PLANE_SAMPLE_RADIUS_M)
            if len(near) < 3:
                continue
            pick = rng.choice(near, size=3, replace=False)
            trio = remaining[pick]
            v1, v2 = trio[1] - trio[0], trio[2] - trio[0]
            nrm = np.cross(v1, v2)
            if abs(nrm[2]) < 1e-6:
                continue
            a, b = -nrm[0] / nrm[2], -nrm[1] / nrm[2]
            c = trio[0, 2] - a * trio[0, 0] - b * trio[0, 1]
            n = int((np.abs(remaining[:, 2] - (a * remaining[:, 0] + b * remaining[:, 1] + c))
                     < PLANE_TOL_M).sum())
            if n > best_n:
                best, best_n = (a, b, c), n
        if best is None or best_n < min_pts:
            break
        inl = np.abs(remaining[:, 2] - (best[0] * remaining[:, 0]
                                        + best[1] * remaining[:, 1] + best[2])) < PLANE_TOL_M
        planes.append(_fit_plane_robust(remaining[inl]))
        remaining = remaining[~inl]
    return planes


def _plane_intersection_cuts(planes, poly):
    """(angle, offset) for every pair of planes that meet, in _cut's convention.

    This is the replacement for detecting roof lines. Two planes intersect along
    one exact line; near-parallel pairs are skipped because their intersection is
    meaningless or far away -- those are steps in height, not folds."""
    cx, cy = poly.centroid.x, poly.centroid.y
    out = []
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            pa, pb = planes[i], planes[j]
            A, B = pa[0] - pb[0], pa[1] - pb[1]
            if np.hypot(A, B) < 1e-3:
                continue
            C = ((pa[0] * cx + pa[1] * cy + pa[2]) - (pb[0] * cx + pb[1] * cy + pb[2]))
            n = np.hypot(A, B)
            ang = float((np.degrees(np.arctan2(A, -B))) % 180.0)   # line dir _|_ to (A,B)
            off = float(-C / n)
            out.append((ang, off))
    return out


def _partition(poly, pts, depth=0, budget=None):
    if budget is None:
        budget = [MAX_FACES]
    plane, score = _score(poly, pts)
    if plane is None:
        return []

    # Acceptance needs BOTH: enough points near the plane, and no fold.
    #
    # The inlier fraction alone is blind to how far the outliers are, and that
    # is not a small blind spot. 7 Anderson Heights had a 73 m2 face scoring
    # exactly 85% -- the acceptance bar -- while 10% of its points sat more than
    # half a metre off it and nearly 5% over a metre. That is a roof dropping
    # through the middle of a face, and it was accepted as one plane, so the
    # recursion never even asked whether a cut would help. Josh: "two panel
    # planes placed and both overlapping a roof ridge where it drops in the
    # middle."
    #
    # A tail that far out is structure, not noise. Roughness raises the count of
    # points just outside the band; a fold puts them metres away.
    # A region is one face only if it fits a plane, does not fold, and its ridge
    # does not drop. The ridge test has to sit in ACCEPTANCE and not merely in
    # the cut search: two sections of the same pitch and aspect fit a common
    # plane well, so 7 Anderson's north slope scored 86.8% -- above the bar --
    # and was accepted whole before any cut was considered, running straight
    # over a 1.16 m dip.
    folded = _fold_evidence(poly, pts, plane)
    drops = _ridge_drop_offsets(poly, pts) if not folded else None
    if ((score >= ACCEPT_INLIER and not folded and drops is None) or depth >= MAX_DEPTH
            or poly.area < 2 * MIN_FACET_M2 or budget[0] <= 1):
        return [(poly, plane)]
    best = _best_cut(poly, pts, score)
    if best is None:
        return [(poly, plane)]      # no straight cut explains it better -- keep it whole
    parts = _refine_cut(poly, pts, best[1])

    # A cut has to earn the panel area it costs. Every facet is eroded by the
    # ridge setback before packing, so cutting one face into two loses a strip
    # down the middle permanently -- and a fit improvement that cannot be used,
    # because the pieces are too narrow to rack panels on, is worth nothing.
    #
    # Without this the recursion buys fit indefinitely: measured on random pilot
    # roofs it produced 20 facets on a 255 m2 house and 25 on 333 m2, about
    # 13 m2 each. Real houses have two to eight planes. Josh, twice: "they need
    # to be large and blocky most of the time like real rooftops", and "it's
    # highly unlikely there would ever be very many vertices on a house".
    gain = max(0.0, best[0] - score)
    cost = _usable(poly) - sum(_usable(q) for q in parts)
    # The setback economics protect a face that is already a plane from being
    # fragmented for a fit gain too small to be worth the racking area. They must
    # not protect a FOLDED face: there the cut is not buying fit, it is putting
    # the ridge where the roof actually bends, and paying a setback strip for
    # that is the whole point. Measured on 7 Anderson Heights: the fold test
    # correctly flagged its 75.4 m2 face, the recursion asked for a cut, and this
    # veto threw it away (cost 2.81 m2 against a 1.97 m2 threshold) -- so the
    # fold detection changed nothing until the veto learned to stand down.
    if not folded and drops is None and cost > SETBACK_COST_PER_FIT * gain * max(_usable(poly), 1e-9):
        return [(poly, plane)]
    out = []
    budget[0] -= 1          # this cut spends one face from the building's budget
    for part in parts:
        out.extend(_partition(part, _points_in(part, pts), depth + 1, budget))
    return out or [(poly, plane)]


def _slope_aspect(plane):
    a, b, _ = plane
    slope = float(np.degrees(np.arctan(np.hypot(a, b))))
    aspect = float((np.degrees(np.arctan2(-a, -b)) + 360.0) % 360.0)
    return slope, aspect


def _plane_angle(p, q):
    na = np.array([-p[0], -p[1], 1.0]); na /= np.linalg.norm(na)
    nb = np.array([-q[0], -q[1], 1.0]); nb /= np.linalg.norm(nb)
    return float(np.degrees(np.arccos(np.clip(abs(na @ nb), -1.0, 1.0))))


BRIDGE_MAX_STEP_M = 0.10

# Two faces produced by the same cut can end up separated by a hairline crack --
# coincident edges whose vertices differ in the last bits of floating point. GEOS
# reports them as touching but unions them into a MultiPolygon, and the bridge
# merge then refuses the pair because "the union is not a Polygon".
#
# That is not a corner case here, it is Josh's complaint about 7 Anderson
# Heights: "the edge roof plane triangles are triangles, but you are cutting
# them into two smaller triangles by continuing the main roof ridgeline through
# them." Both hip ends were split by the ridge cut and both pairs were perfectly
# coplanar -- aspects 315.2/315.2 and 138.4/136.5, angle well inside the bridge
# limit, sharing an 8.4 m and a 6.7 m edge at zero distance -- and both merges
# were thrown away over a sliver of about 0.02 m2.
#
# Snapping one face's vertices onto the other's within a centimetre closes the
# crack without moving any edge a distance anyone could see.
MERGE_SNAP_M = 0.01

# Two pieces of ONE plane, separated only because something was carved between
# them, are one face. The bridge merge normally has to earn back racking area --
# gain > 0.5 m2 of setback -- which is the right bar when the question is
# whether a genuine seam is worth keeping. It is the wrong bar when there is no
# seam at all.
#
# 7 Anderson Heights: the north slope is notched by the recessed feature in the
# middle of the roof, so its two parts meet along a neck of just 0.77 m. Merging
# them recovers 0.008 m2 of setback and the economics threw it away, leaving 9
# faces where Josh drew 8. They are 0.78 degrees apart with a 0.001 m step at
# the join, and merged they fit 93.7% -- better than the 86.8% a single
# uncut face scored. That is one plane by every measure that matters.
SAME_PLANE_ANGLE_DEG = 1.5   # tighter than the bridge angle: this is "identical", not "close"
SAME_PLANE_STEP_M = 0.05


def _step_at_join(pa, pb, poly_a, poly_b):
    """Height gap between two planes WHERE THEY ADJOIN.

    Without this the merge is wrong in a way an angle test cannot see: two
    parallel faces at different levels have identical normals, so they read as
    0 degrees apart and get merged straight across a step. That is exactly what
    happened here -- the recursion built 5 correct faces on 5 Isle St at 87-99%
    on-plane and the merge collapsed them to 2 at 45%. Third time this same bug
    has appeared in this codebase; comparing planes at their shared boundary
    rather than comparing normals is the only thing that catches it."""
    shared = poly_a.buffer(0.3).intersection(poly_b.buffer(0.3))
    if shared.is_empty:
        return float("inf")
    c = shared.centroid
    za = pa[0] * c.x + pa[1] * c.y + pa[2]
    zb = pb[0] * c.x + pb[1] * c.y + pb[2]
    return abs(float(za - zb))


def _merge_bridgeable(faces, pts):
    """Undo cuts a panel could lie across.

    A cut that buys fit but not enough to stop a panel spanning it costs the
    ridge setback on both sides for nothing. 5 degrees over a 1.7 m panel is a
    15 cm rise, past what a rigid frame bridges."""
    # Pair testing is memoised. The loop used to restart the whole O(N^2) sweep
    # after every successful merge, re-running _points_in and _fit_plane_robust
    # on pairs it had already rejected -- O(N^3) in the expensive geometry ops.
    # It still terminated (a merge always drops the face count by one), so it
    # never showed up as a hang in testing, just as a build that sat at 100% CPU
    # on one worker. On a Frankton Flats roof at the 60-face cap that is ~100k
    # plane fits and about an hour on one building, which is what stalled the
    # 28 Aug rebuild twice. A rejected pair can only become viable again if one
    # of its two faces is itself replaced by a merge, so remembering rejections
    # by face identity is exact -- this is a speedup, not an approximation.
    faces = list(faces)
    uid = {id(f): n for n, f in enumerate(faces)}
    next_uid = len(faces)
    rejected = set()
    changed = True
    while changed and len(faces) > 1:
        changed = False
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                key = (uid[id(faces[i])], uid[id(faces[j])])
                if key in rejected:
                    continue
                (pi, li), (pj, lj) = faces[i], faces[j]
                if not pi.buffer(0.25).intersects(pj):
                    rejected.add(key)
                    continue
                if _plane_angle(li, lj) > 5.0:
                    rejected.add(key)
                    continue
                if _step_at_join(li, lj, pi, pj) > BRIDGE_MAX_STEP_M:
                    rejected.add(key)
                    continue
                u = unary_union([pi, snap(pj, pi, MERGE_SNAP_M)])
                if u.geom_type != "Polygon":
                    closed = unary_union([pi.buffer(MERGE_SNAP_M),
                                          pj.buffer(MERGE_SNAP_M)]).buffer(-MERGE_SNAP_M)
                    if closed.geom_type != "Polygon":
                        rejected.add(key)
                        continue
                    u = Polygon(closed.exterior, [r for r in closed.interiors])
                gain = (u.buffer(-config.RIDGE_SETBACK_M).area
                        - pi.buffer(-config.RIDGE_SETBACK_M).area
                        - pj.buffer(-config.RIDGE_SETBACK_M).area)
                same_plane = (_plane_angle(li, lj) <= SAME_PLANE_ANGLE_DEG
                              and _step_at_join(li, lj, pi, pj) <= SAME_PLANE_STEP_M)
                if gain <= 0.5 and not same_plane:
                    rejected.add(key)
                    continue
                sub = _points_in(u, pts)
                if len(sub) < MIN_POINTS:
                    rejected.add(key)
                    continue
                pl = _fit_plane_robust(sub)
                # The merged face has to still be a plane. Angle, step and area
                # gain all pass on faces that individually fit well but whose
                # UNION does not -- a gentle curve is exactly that, every
                # adjacent pair within the bridge angle while the whole sweep is
                # not one plane. Unchecked, this built a 138 m2 face on 1/5
                # Sydney St sitting at 15% on-plane out of pieces that were
                # each fine, and nothing downstream could recover from it
                # because the recursion had already finished.
                merged_fit = _inlier_fraction(sub, pl)
                worst_before = min(_inlier_fraction(_points_in(pi, pts), li),
                                   _inlier_fraction(_points_in(pj, pts), lj))
                # A merge made on "same plane" grounds rather than on recovered
                # area has to clear the acceptance bar outright -- it is claiming
                # the two pieces ARE one plane, so the union must look like one.
                bar = ACCEPT_INLIER if (gain <= 0.5) else min(ACCEPT_INLIER, worst_before - 0.02)
                if merged_fit < bar:
                    rejected.add(key)
                    continue
                merged = (u, pl)
                faces = [f for k, f in enumerate(faces) if k not in (i, j)] + [merged]
                uid[id(merged)] = next_uid
                next_uid += 1
                changed = True
                break
            if changed:
                break
    return faces


# How many planes a roof actually has, from Josh: "there are generally not going
# to be many planes on a roof, most probably only have between 1 and 10 or so.
# Unless a big hotel or business roof, but then still... Likely between 1 and 30
# or so." That is the prior this whole module was missing, and it is why a
# 93%-on-plane score could sit on a roof he called clearly wrong: fit says
# nothing about whether a shape looks like a roof.
PLANES_TYPICAL_MAX = 10          # a house
PLANES_LARGE_MAX = 30            # a hotel or a large commercial roof
PLANES_LARGE_ROOF_M2 = 600.0     # above this footprint, allow the higher count
MAX_INTERSECTION_LINES = 24      # cutting by every pair explodes; keep the best


def top_surface(pts, cell_m=0.5, drop_m=0.5):
    """Keep only the points on the TOP surface of each plan cell. Under an eave
    of a multi-level building LiDAR records both the upper roof edge and the
    lower roof beneath it at the same plan location; a roof model is a function
    z(x, y), so those lower returns are unexplainable by construction and they
    poison plane fits, region labels and any points-explained metric alike
    (measured on #5119630: two roof levels, 29% of points on the lower one,
    facet planes fitted through the mixture explaining 0% of their polygons).
    The lower level's EXPOSED area keeps its points -- there it IS the top."""
    if len(pts) == 0:
        return pts
    ix = np.floor(pts[:, 0] / cell_m).astype(np.int64)
    iy = np.floor(pts[:, 1] / cell_m).astype(np.int64)
    key = (ix - ix.min()) * (iy.max() - iy.min() + 1) + (iy - iy.min())
    order = np.argsort(key, kind="stable")
    ks, zs = key[order], pts[order, 2]
    top = np.empty(len(pts))
    start = 0
    for end in np.append(np.where(np.diff(ks) != 0)[0] + 1, len(ks)):
        top[start:end] = zs[start:end].max()
        start = end
    keep_sorted = zs > top - drop_m
    keep = np.zeros(len(pts), dtype=bool)
    keep[order] = keep_sorted
    return pts[keep]


def dedupe_planes(planes, footprint, angle_tol_deg=5.0, height_tol_m=0.35):
    """Drop planes that are the same surface fitted twice. Two genuinely
    separate faces sharing pitch AND aspect AND height are coplanar -- one
    plane serves both, and the adjacency merge separates their polygons."""
    cx, cy = footprint.centroid.x, footprint.centroid.y
    kept = []
    for p in planes:
        dup = False
        for q in kept:
            na = np.array([-p[0], -p[1], 1.0]); na /= np.linalg.norm(na)
            nb = np.array([-q[0], -q[1], 1.0]); nb /= np.linalg.norm(nb)
            ang = np.degrees(np.arccos(np.clip(na @ nb, -1, 1)))
            dz = abs((p[0] * cx + p[1] * cy + p[2]) - (q[0] * cx + q[1] * cy + q[2]))
            if ang < angle_tol_deg and dz < height_tol_m:
                dup = True
                break
        if not dup:
            kept.append(p)
    return kept


def explained_fraction(faces, pts, band=INLIER_BAND_M):
    """Share of ALL the building's points a facet set explains: a point counts
    when it falls inside some facet whose plane passes within `band` of it.
    One number rewarding both coverage and correctness -- a wedge spanning two
    real faces covers its points but explains few of them, and a correct facet
    set that abandons half the roof explains at most the half it kept."""
    if len(pts) == 0:
        return 0.0
    ok = np.zeros(len(pts), dtype=bool)
    for f in faces:
        poly = f["geometry"] if isinstance(f, dict) else f[0]
        a, b, c = ((f["plane_a"], f["plane_b"], f["plane_c"]) if isinstance(f, dict)
                   else f[1])
        m = shapely.contains_xy(poly, pts[:, 0], pts[:, 1])
        if not m.any():
            continue
        resid = np.abs(pts[m, 2] - (a * pts[m, 0] + b * pts[m, 1] + c))
        idx = np.where(m)[0]
        ok[idx[resid < band]] = True
    return float(ok.mean())


def _reflex_corner_cuts(poly):
    """Cut lines through each REFLEX (concave) footprint corner, along both
    adjoining wall directions. An L- or U-shaped building's wings meet at its
    reflex corners; the roof sections separate along lines through them. The
    plane-intersection cuts cannot draw these boundaries when the two wings'
    faces are near-parallel (same pitch, different wing) -- parallel planes
    have no intersection line -- and that is exactly the geometry of every
    multi-wing hip roof, so without these the cells straddle wings and whole
    faces get merged across the valley (measured on #5119630: two 52 m2
    cross-wing sheets, explained fraction 0.51)."""
    ring = np.asarray(poly.exterior.coords[:-1])
    if len(ring) < 4:
        return []
    # shoelace: positive = CCW
    area2 = float(np.sum(ring[:, 0] * np.roll(ring[:, 1], -1)
                         - np.roll(ring[:, 0], -1) * ring[:, 1]))
    ccw = area2 > 0
    c = np.array(poly.centroid.coords[0])
    out = []
    n_pts = len(ring)
    for i in range(n_pts):
        p0, p1, p2 = ring[i - 1], ring[i], ring[(i + 1) % n_pts]
        v1, v2 = p1 - p0, p2 - p1
        if np.hypot(*v1) < 0.5 or np.hypot(*v2) < 0.5:
            continue        # survey jitter, not a wall corner
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if (cross < 0) != ccw:
            continue        # convex corner in this orientation
        for v in (v1, v2):
            ang = float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)
            theta = np.radians(ang)
            nrm = np.array([-np.sin(theta), np.cos(theta)])
            out.append((ang, float((p1 - c) @ nrm)))
    return out


def partition_with_labels(building_id, footprint, pts, labels, planes):
    """Arrangement partition where the ASSIGNMENT comes from labelled points
    (region growing), not residual voting. Residual voting cannot tell two
    coplanar faces on different wings apart -- same plane, same residuals --
    and unary_union then welds them into one cross-valley sheet. Labels are
    contiguous regions by construction, so a wing can only ever claim cells
    its own points sit in.

    labels: per-point region id, -1 for unassigned. planes: (a,b,c) per id."""
    inside_mask_pts = pts
    inside = _points_in(footprint, inside_mask_pts)
    if len(inside) < MIN_POINTS:
        return []
    # labels must correspond to pts row-for-row; recompute the footprint mask
    # the same way _points_in does so they stay aligned.
    m = shapely.contains_xy(footprint, pts[:, 0], pts[:, 1])
    pin, lin = pts[m], np.asarray(labels)[m]

    # Cut lines: every pair of genuinely DISTINCT planes (their intersection is
    # a ridge/hip/valley) plus the wing separations at reflex corners.
    uniq = dedupe_planes(list(planes), footprint)
    cuts = (_plane_intersection_cuts(uniq, footprint)[:MAX_INTERSECTION_LINES]
            + _reflex_corner_cuts(footprint))
    cells = [footprint]
    for ang, off in cuts:
        nxt = []
        for c in cells:
            parts = _cut(c, ang, off)
            nxt.extend(parts if len(parts) >= 2 else [c])
        cells = nxt
        if len(cells) > 200:
            break

    labelled = []
    for cell in cells:
        if cell.area < MIN_PIECE_M2:
            continue
        mm = shapely.contains_xy(cell, pin[:, 0], pin[:, 1])
        votes = lin[mm]
        votes = votes[votes >= 0]
        if len(votes) < 4:
            labelled.append((cell, None))
            continue
        ids, counts = np.unique(votes, return_counts=True)
        labelled.append((cell, int(ids[np.argmax(counts)])))
    known = [(g, i) for g, i in labelled if i is not None]
    if not known:
        return []
    for k, (g, i) in enumerate(labelled):
        if i is None:
            labelled[k] = (g, min(known, key=lambda t: t[0].distance(g))[1])

    out = []
    for rid in sorted({i for _, i in labelled}):
        mine = [g for g, i in labelled if i == rid]
        merged = unary_union(mine)
        for poly in (merged.geoms if merged.geom_type == "MultiPolygon" else [merged]):
            if poly.area < MIN_FACET_M2:
                continue
            sub = _points_in(poly, pin)
            plane = (_fit_plane_robust(sub) if len(sub) >= MIN_POINTS
                     else tuple(planes[rid]))
            slope, aspect = _slope_aspect(plane)
            if slope > config.MAX_ROOF_SLOPE_DEG:
                continue
            if slope >= STEEP_FACE_DEG and _inlier_fraction(sub, plane) < STEEP_FACE_MIN_FIT:
                continue
            out.append({
                "building_id": building_id,
                "geometry": Polygon(poly.exterior, [r for r in poly.interiors]),
                "plane_a": plane[0], "plane_b": plane[1], "plane_c": plane[2],
                "slope_deg": slope, "aspect_deg": aspect,
                "area_m2": float(poly.area), "point_count": int(len(sub)),
            })
    return out


def partition_by_planes(building_id, footprint, pts, seed=0, planes=None):
    """Big planes from the LiDAR, trimmed by each other and by the building edge.

    Josh's description exactly: "make big planes based on detectable roof angles
    with the lidar, and then trim those planes by either the edge of the building
    or another plane."

    Nothing is detected in imagery and no boundary is traced. A plane's extent is
    decided by where it stops being the best explanation of the points -- which
    is either where another plane takes over, along the exact line the two
    intersect, or the surveyed footprint edge. Both are straight by construction,
    so the result cannot be fuzzy and the faces meet cleanly at real angles.

    `planes` lets a caller supply the plane hypotheses (region growing finds
    much cleaner planes on complex hip roofs than the greedy detector here --
    greedy RANSAC's compromise planes at the joins were exactly why this path
    sat unused); None keeps the self-detecting behaviour."""
    rng = np.random.default_rng(seed)
    inside = _points_in(footprint, pts)
    if len(inside) < MIN_POINTS:
        return []
    if planes is None:
        cap = PLANES_LARGE_MAX if footprint.area > PLANES_LARGE_ROOF_M2 else PLANES_TYPICAL_MAX
        planes = _detect_large_planes(inside, footprint, rng, max_planes=cap)
    planes = dedupe_planes(planes, footprint)
    if not planes:
        return []

    # Cut by where the planes meet. Ordered by how much roof each pair actually
    # separates, so if the cap bites it is the least important joins that go.
    cuts = (_plane_intersection_cuts(planes, footprint)[:MAX_INTERSECTION_LINES]
            + _reflex_corner_cuts(footprint))
    cells = [footprint]
    for ang, off in cuts:
        nxt = []
        for c in cells:
            parts = _cut(c, ang, off)
            nxt.extend(parts if len(parts) >= 2 else [c])
        cells = nxt
        if len(cells) > 200:      # runaway guard on a pathological roof
            break

    # Each cell goes to the plane its own points support best; cells too sparse
    # to vote take the plane of the nearest cell that could.
    labelled = []
    for cell in cells:
        if cell.area < MIN_PIECE_M2:
            continue
        sub = _points_in(cell, inside)
        if len(sub) < 6:
            labelled.append((cell, None))
            continue
        res = [np.median(np.abs(sub[:, 2] - (a * sub[:, 0] + b * sub[:, 1] + c)))
               for a, b, c in planes]
        labelled.append((cell, int(np.argmin(res))))
    known = [(g, i) for g, i in labelled if i is not None]
    if not known:
        return []
    for k, (g, i) in enumerate(labelled):
        if i is None:
            labelled[k] = (g, min(known, key=lambda t: t[0].distance(g))[1])

    # Adjacent cells on the same plane are one face.
    out = []
    for pi in range(len(planes)):
        mine = [g for g, i in labelled if i == pi]
        if not mine:
            continue
        merged = unary_union(mine)
        for poly in (merged.geoms if merged.geom_type == "MultiPolygon" else [merged]):
            if poly.area < MIN_FACET_M2:
                continue
            sub = _points_in(poly, inside)
            plane = _fit_plane_robust(sub) if len(sub) >= MIN_POINTS else planes[pi]
            slope, aspect = _slope_aspect(plane)
            if slope > config.MAX_ROOF_SLOPE_DEG:
                continue
            if slope >= STEEP_FACE_DEG and _inlier_fraction(sub, plane) < STEEP_FACE_MIN_FIT:
                continue        # steep AND not a plane: a wall, not a roof face
            out.append({
                "building_id": building_id,
                "geometry": Polygon(poly.exterior, [r for r in poly.interiors]),
                "plane_a": plane[0], "plane_b": plane[1], "plane_c": plane[2],
                "slope_deg": slope, "aspect_deg": aspect,
                "area_m2": float(poly.area), "point_count": int(len(sub)),
            })
    return out


# The LINZ outline is the BUILDING, not the roof.
#
# Josh drew the true roof outline on 7 Anderson Heights and it does not follow
# the footprint: the roof overhangs it on one side and sits inside it on
# another. Measured across five buildings, 6.6% to 18.8% of roof-height points
# fall outside the footprint, by up to 2 m. Those are eaves.
#
# This module cuts the footprint into faces, so every perimeter face was wrong
# at its edge before any ridge logic ran, and roof area was understated
# everywhere. The footprint is still the right SKELETON -- it is surveyed and
# straight -- so the roof outline is built by pushing it out to where the roof
# actually stops rather than by tracing points, which would reintroduce the
# fuzz the whole approach exists to avoid.
# Deliberately timid, because a uniform buffer grows toward the NEIGHBOURS too
# and their roofs sit at similar heights, so a loose test walks straight onto
# them: at 2.0 m and a 50% share this grew 7 Anderson Heights by 66% and 2/8
# Wakatipu by 81%, when Josh's drawn roof outline is nearer 10% larger than the
# footprint. Held to a typical eave, and needing the ring to be almost entirely
# roof-height before it is accepted.
# OFF pending better work -- see roof_outline. The FINDING is solid and matters:
# 6.6% to 18.8% of roof-height points fall outside the LINZ footprint, by up to
# 2 m, so roof area is understated everywhere and every perimeter face is wrong
# at its edge. But a uniform buffer is the wrong instrument. Josh's drawn outline
# on 7 Anderson Heights is not the footprint grown evenly -- the roof overhangs
# on one side and sits INSIDE it on another -- and growing uniformly took that
# roof to 16 faces against the 8 he counted. This needs per-edge treatment:
# decide independently for each footprint edge how far the roof runs past it.
# The eave is added AFTER partitioning, never before -- see _extend_to_eave.
# Partitioning a grown outline was tried and is wrong -- 6.6% to 18.8% of roof-height points
# fall outside the LINZ footprint by up to 2 m, so roof area is understated
# everywhere -- and per-edge measurement matches Josh's drawn outlines exactly
# (7 Anderson Heights runs 1.25 m past one long edge, 0.0 past the opposite one,
# 2.0 m past one end). But growing the outline makes face counts WORSE, in both
# the uniform and the per-edge form: with imagery cuts active, Anderson goes
# 10 -> 17 faces against Josh's 8 and 29 Edinburgh 4 -> 7 against his 5, because
# the eave strips are then sliced by the same lines into slivers. Fixing this
# which is why the ring is now merged into the faces that already exist instead.
EAVE_MAX_M = 2.0
EAVE_MIN_EDGE_M = 1.5            # shorter edges are corner chamfers, not roof sides
EAVE_MIN_BAND_POINTS = 3         # roof points needed in a strip to keep walking out
EAVE_HEIGHT_SLACK_M = 0.4        # how far outside the roof's own height range still counts
EAVE_CORNER_CLOSE_M = 0.3
EAVE_STEP_M = 0.25
EAVE_MIN_POINT_SHARE = 0.85


def roof_outline(footprint, pts):
    """Footprint pushed out to the real roof edge, ONE EDGE AT A TIME.

    A uniform buffer cannot represent this and was tried first: Josh's drawn
    roof outlines run past the footprint on some sides and sit inside it on
    others, and growing evenly took 7 Anderson Heights to 16 faces against the
    8 he counted, and 2/8 Wakatipu Heights up 33% in area. Measured per edge,
    Anderson runs 1.25 m past one long edge and 0.0 past the opposite one, and
    2.0 m past one end -- there is no single number.

    Each edge is walked outward in short steps for as long as roof-height points
    keep appearing, then the strip it gained is unioned on. Every edge stays a
    straight line, so this cannot reintroduce the traced-boundary fuzz the whole
    module exists to avoid."""
    if len(pts) < MIN_POINTS or footprint.is_empty:
        return footprint
    inside = _points_in(footprint, pts)
    if len(inside) < MIN_POINTS:
        return footprint
    lo, hi = np.percentile(inside[:, 2], [5, 95])
    lo -= EAVE_HEIGHT_SLACK_M
    hi += EAVE_HEIGHT_SLACK_M
    at_roof = (pts[:, 2] >= lo) & (pts[:, 2] <= hi)
    if at_roof.sum() < MIN_POINTS:
        return footprint

    coords = np.asarray(footprint.exterior.coords)
    strips = []
    for i in range(len(coords) - 1):
        a, b = coords[i], coords[i + 1]
        seg = b - a
        length = float(np.hypot(*seg))
        if length < EAVE_MIN_EDGE_M:
            continue
        d = seg / length
        n = np.array([d[1], -d[0]])
        if footprint.contains(Point(*((a + b) / 2 + n * 0.3))):
            n = -n                      # make sure it points outward
        reach = 0.0
        for step in np.arange(EAVE_STEP_M, EAVE_MAX_M + 1e-9, EAVE_STEP_M):
            band = Polygon([a + n * (step - EAVE_STEP_M), b + n * (step - EAVE_STEP_M),
                            b + n * step, a + n * step])
            if band.is_empty or not band.is_valid:
                break
            m = at_roof & shapely.contains_xy(band, pts[:, 0], pts[:, 1])
            if int(m.sum()) < EAVE_MIN_BAND_POINTS:
                break
            reach = float(step)
        if reach > 0:
            strips.append(Polygon([a, b, b + n * reach, a + n * reach]))

    if not strips:
        return footprint
    grown = unary_union([footprint] + strips)
    if grown.geom_type == "MultiPolygon":
        grown = max(grown.geoms, key=lambda q: q.area)
    if grown.geom_type != "Polygon":
        return footprint
    # close the small notches left at corners where two strips meet
    grown = grown.buffer(EAVE_CORNER_CLOSE_M).buffer(-EAVE_CORNER_CLOSE_M)
    if grown.geom_type != "Polygon" or grown.is_empty:
        return footprint
    return Polygon(grown.exterior).simplify(0.05)


# A strong imagery line is only cut if the roof actually CHANGES there.
#
# Cutting every strong line unconditionally fixed 7 Anderson Heights, whose hip
# creases the LiDAR cannot resolve, and broke 5 Isle St, which went from the 3
# faces Josh confirms to 4. His note on that roof is the clue: "The largest
# plane on this roof has a significant angle/slope on it. I haven't marked that.
# It's also still all one plane though." The imagery sees a line there -- a
# seam, a stain, a shadow, a tonal band across a sloped surface -- and it is not
# a roof line.
#
# The earlier mistake was asking the LiDAR the wrong question. "Does cutting
# improve the fit" is answerable only where the LiDAR already resolves the
# feature, so it rejected Anderson's hips. "Do the two sides have different
# ORIENTATIONS" is answerable from the same coarse data, because it compares two
# large samples rather than resolving a crease: a ridge or hip turns the roof,
# a stain does not. A height step counts too -- parallel faces at different
# levels are a real edge.
LINE_MIN_TURN_DEG = 8.0       # plane orientation must change by this across the line
LINE_MIN_STEP_M = 0.20        # ...or the two sides sit at different heights
LINE_MIN_SIDE_POINTS = 30


def _line_is_real(poly, pts, ang, off):
    """Does the roof change across this line, or is it only visible in the image?"""
    parts = _cut(poly, ang, off)
    if len(parts) != 2:
        return False
    a, b = (_points_in(parts[0], pts), _points_in(parts[1], pts))
    if len(a) < LINE_MIN_SIDE_POINTS or len(b) < LINE_MIN_SIDE_POINTS:
        return False
    pa, pb = _fit_plane_robust(a), _fit_plane_robust(b)
    if _plane_angle(pa, pb) >= LINE_MIN_TURN_DEG:
        return True
    return _step_at_join(pa, pb, parts[0], parts[1]) >= LINE_MIN_STEP_M


# The footprint can also be BIGGER than the roof, and that half was missed.
#
# Josh drew it on 7 Anderson Heights -- "the roof overhangs it on the upper-left
# and sits INSIDE it on the lower-left" -- and it is what put a panel over the
# edge of 62 Ballarat St. Measured there, three edges overshoot the real roof by
# 0.5 to 1.25 m: about 29 m2 of dead ground inside the footprint, carrying
# panels on roof that does not exist.
#
# Trimmed BEFORE partitioning, unlike the eave, because dead ground must never
# become part of a face at all.
TRIM_MAX_M = 2.0
TRIM_STEP_M = 0.25
TRIM_MIN_BAND_POINTS = 3


def trim_to_roof(footprint, pts):
    """Footprint pulled IN wherever it overshoots the roof, edge by edge."""
    if len(pts) < MIN_POINTS or footprint.is_empty:
        return footprint
    inside = _points_in(footprint, pts)
    if len(inside) < MIN_POINTS:
        return footprint
    lo, hi = np.percentile(inside[:, 2], [5, 95])
    lo -= EAVE_HEIGHT_SLACK_M
    hi += EAVE_HEIGHT_SLACK_M
    at_roof = (pts[:, 2] >= lo) & (pts[:, 2] <= hi)
    if at_roof.sum() < MIN_POINTS:
        return footprint

    coords = np.asarray(footprint.exterior.coords)
    cuts = []
    for i in range(len(coords) - 1):
        a, b = coords[i], coords[i + 1]
        seg = b - a
        length = float(np.hypot(*seg))
        if length < EAVE_MIN_EDGE_M:
            continue
        d = seg / length
        n = np.array([d[1], -d[0]])
        if footprint.contains(Point(*((a + b) / 2 + n * 0.3))):
            n = -n                       # outward
        gap = 0.0
        for step in np.arange(0.0, TRIM_MAX_M + 1e-9, TRIM_STEP_M):
            band = Polygon([a - n * step, b - n * step,
                            b - n * (step + TRIM_STEP_M), a - n * (step + TRIM_STEP_M)])
            if band.is_empty or not band.is_valid:
                break
            m = at_roof & shapely.contains_xy(band, pts[:, 0], pts[:, 1])
            if int(m.sum()) >= TRIM_MIN_BAND_POINTS:
                break                    # roof starts here
            gap = float(step + TRIM_STEP_M)
        if gap > 0:
            cuts.append(Polygon([a, b, b - n * gap, a - n * gap]))

    if not cuts:
        return footprint
    trimmed = footprint.difference(unary_union(cuts))
    if trimmed.geom_type == "MultiPolygon":
        trimmed = max(trimmed.geoms, key=lambda q: q.area)
    if (trimmed.geom_type != "Polygon" or trimmed.is_empty
            or trimmed.area < 0.5 * footprint.area):
        return footprint                 # never lose half a building to this
    return Polygon(trimmed.exterior).simplify(0.05)


def _extend_to_eave(faces, footprint, pts):
    """Give each finished face the strip of roof that overhangs beside it.

    The eave is real -- 6.6% to 18.8% of roof-height points fall outside the
    LINZ footprint, by up to 2 m -- but partitioning a grown outline is the
    wrong way to capture it. The imagery lines then slice the new strips into
    slivers and the face count balloons: 7 Anderson Heights went 10 -> 17
    against Josh's 8, 29 Edinburgh 4 -> 7 against his 5.

    Adding it here instead cannot create a face. The ring between footprint and
    roof edge is cut up and each piece joins whichever finished face it already
    touches, so the count is exactly what the partition decided and the roof
    simply reaches its true edge."""
    if not faces:
        return faces
    outline = roof_outline(footprint, pts)
    ring = outline.difference(footprint)
    if ring.is_empty or ring.area < 0.5:
        return faces
    pieces = list(ring.geoms) if ring.geom_type == "MultiPolygon" else [ring]
    out = [(g, pl) for g, pl in faces]
    for piece in pieces:
        if piece.is_empty or piece.area < 0.05:
            continue
        # Which face this strip belongs to is decided by the strip's own points,
        # not by which face happens to touch it most. Shared boundary alone
        # attaches an overhang to whatever is beside it even when the roof there
        # lies on a different plane: on 7 Anderson Heights that grew the upper
        # slope from 64.6 m2 at 86.8% on-plane to 98 m2 at 78%, by gluing a strip
        # of the hip onto it. Among the faces this strip actually touches, take
        # the one whose plane the strip's own returns sit closest to.
        touching = [k for k, (g, _pl) in enumerate(out)
                    if g.buffer(0.05).intersection(piece).area > 0]
        if not touching:
            nearest = min(range(len(out)), key=lambda k: out[k][0].distance(piece))
            touching = [nearest]
        strip_pts = _points_in(piece, pts)
        if len(strip_pts) >= 4:
            best = max(touching, key=lambda k: _inlier_fraction(strip_pts, out[k][1]))
        else:
            best = max(touching,
                       key=lambda k: out[k][0].buffer(0.05).intersection(piece).area)
        g, pl = out[best]
        merged = unary_union([g, piece])
        if merged.geom_type != "Polygon":
            continue
        out[best] = (Polygon(merged.exterior, [r for r in merged.interiors]), pl)
    return out



# A recessed section -- a length of roof sitting BELOW the surface around it --
# has to come out as its own region before anything is cut, because no line that
# spans the whole roof can isolate it.
#
# Josh drew one on 7 Anderson Heights and confirmed it is recessed. Measured:
# the ridge runs at 381.42 m, drops to 380.26 m across the middle, and returns
# to 381.47 m. Earlier attempts hunted for its edges as shoulders in a 1-D ridge
# profile and put the cuts in the wrong place every time. In 2-D it is simply
# the set of points below the roof's own plane envelope.
#
# Two details matter. A pitched roof's surface is the LOWER envelope of its
# planes, so the reference has to be min() over the main faces and not any one
# plane -- against a single plane the hips read as recessed too, and 78 m2 of a
# 211 m2 roof got flagged. And the depth has to be a BAND: where the surveyed
# footprint overruns the roof, the points beyond the eave are wall or ground
# sitting about 2 m down, and they connect to the real recess and swallow it.
# Restricting to 0.45-1.8 m below separates them -- overlap with Josh's traced
# section went from IoU 0.24 to 0.55.
RECESS_MIN_DEPTH_M = 0.45
RECESS_MAX_DEPTH_M = 1.80
RECESS_CELL_M = 0.75
RECESS_MIN_AREA_M2 = 6.0
RECESS_MAX_AREA_SHARE = 0.35    # more than this is not a section, it is the roof
RECESS_MIN_POINTS = 40
RECESS_MAX_FACES = 2       # Josh: "only two faces in the middle recession"


def _recessed_region(footprint, pts, faces):
    """The recessed section of a roof, as a rectangle in the roof's own frame.

    `faces` is a first-pass partition, used only for its planes. Returns None
    when the roof has no such section."""
    if len(faces) < 2 or len(pts) < RECESS_MIN_POINTS:
        return None
    try:
        from scipy import ndimage
    except Exception:
        return None
    planes = [pl for _poly, pl in sorted(faces, key=lambda t: -t[0].area)[:4]]
    env = np.minimum.reduce([p[0] * pts[:, 0] + p[1] * pts[:, 1] + p[2] for p in planes])
    resid = pts[:, 2] - env
    core = pts[(resid < -RECESS_MIN_DEPTH_M) & (resid > -RECESS_MAX_DEPTH_M)]
    if len(core) < RECESS_MIN_POINTS:
        return None
    minx, miny, maxx, maxy = footprint.bounds
    nx = max(2, int(np.ceil((maxx - minx) / RECESS_CELL_M)))
    ny = max(2, int(np.ceil((maxy - miny) / RECESS_CELL_M)))
    if nx * ny > 200000:
        return None
    grid = np.zeros((ny, nx), dtype=bool)
    ix = np.clip(((core[:, 0] - minx) / RECESS_CELL_M).astype(int), 0, nx - 1)
    iy = np.clip(((core[:, 1] - miny) / RECESS_CELL_M).astype(int), 0, ny - 1)
    grid[iy, ix] = True
    grid = ndimage.binary_closing(grid, structure=np.ones((3, 3), dtype=bool))
    lab, n = ndimage.label(grid, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return None
    best = max(range(1, n + 1), key=lambda k: int(np.sum(lab == k)))
    if int(np.sum(lab == best)) * RECESS_CELL_M ** 2 < RECESS_MIN_AREA_M2:
        return None
    ys, xs = np.nonzero(lab == best)
    cx = minx + (xs + 0.5) * RECESS_CELL_M
    cy = miny + (ys + 0.5) * RECESS_CELL_M
    # a rectangle in the roof's own frame: roofs are straight lines, not blobs
    try:
        rect = np.asarray(footprint.minimum_rotated_rectangle.exterior.coords)
    except Exception:
        return None
    e = rect[1:] - rect[:-1]
    L = np.hypot(e[:, 0], e[:, 1])
    if len(L) == 0 or L.max() <= 0:
        return None
    u = e[int(np.argmax(L))] / L.max()
    v = np.array([-u[1], u[0]])
    c = np.array(footprint.centroid.coords[0])
    xy = np.column_stack([cx, cy]) - c
    a, b = xy @ u, xy @ v
    # Measured on both axes, NOT stretched across the roof. Extending it eave to
    # eave was tried, on my own reading of Josh's traced markup, and it is wrong
    # twice over: his section is about 28 m2 where the full width of that roof
    # would be nearer 55, and forcing the stretched region into the two faces he
    # describes gave planes fitting 50% and 43% -- worse than not modelling it.
    # The traced corners came out aligned to north rather than to the roof, so
    # that reading was an artefact of tracing his line by eye.
    poly = Polygon([c + u * aa + v * bb for aa, bb in
                    ((a.min(), b.min()), (a.max(), b.min()),
                     (a.max(), b.max()), (a.min(), b.max()))])
    try:
        poly = poly.intersection(footprint)
    except Exception:
        return None
    if (poly.is_empty or poly.geom_type != "Polygon"
            or poly.area < RECESS_MIN_AREA_M2
            or poly.area > RECESS_MAX_AREA_SHARE * footprint.area):
        return None
    return poly


def partition_roof(building_id, footprint, pts, imagery_ds=None):
    """Surveyed footprint + point cloud -> straight-edged, plane-backed facets.

    Strong imagery lines, if imagery is supplied, are cut FIRST and without the
    point cloud getting a vote. Everywhere else in this module a cut has to earn
    itself against the LiDAR, which is right when both sensors can see the
    feature and wrong when only one can. 7 Anderson Heights is the case that
    forced it: two hipped sections whose creases are unmistakable in 0.1 m
    imagery and almost absent from a point cloud that is near-flat across the
    whole roof. Every LiDAR-scored candidate there was rejected -- correctly, by
    its own logic, since cutting did not improve a fit that was never wrong
    about height -- so the faces ran straight over both hips and the panels
    followed. Josh: "two panel planes placed and both overlapping a roof ridge
    where it drops in the middle."

    Only lines carrying enough evidence to be a primary crease qualify; see
    roof_lines.strong_roof_lines."""
    _cut_evals[0] = 0          # the budget is per building, not per process
    # CPU time, not wall time. A wall-clock deadline is stolen by whatever else
    # is running: during a district rebuild the harness measured 32 Frankton Rd
    # at 16.5% weighted fit while the same code on an idle machine reached 33%,
    # because the other workers ate its 240 seconds. Each partition runs on one
    # thread, so process CPU time measures ITS work regardless of load.
    _cut_deadline[0] = time.process_time() + min(
        CUT_TIME_BUDGET_MAX_S, max(CUT_TIME_BUDGET_S, footprint.area * CUT_TIME_PER_M2_S))
    footprint = trim_to_roof(footprint, pts)
    inside = _points_in(footprint, pts)
    if len(inside) < MIN_POINTS:
        return []

    # Imagery cuts are OFF. Measured against the four roofs Josh has drawn, they
    # are actively harmful:
    #
    #                    Josh   cuts ON   cuts OFF
    #   7 Anderson         8       12         8
    #   5 Isle St          3        3         3
    #   29 Edinburgh       5        3         5
    #   2/8 Wakatipu       8        9         8
    #
    # Off matches all four exactly; on matches one. The idea is still right --
    # 7 Anderson's hip creases are unmistakable in 0.1 m imagery and nearly
    # absent from a point cloud that is flat across that roof -- but cutting a
    # whole cell with a line detected over part of it fragments the roof faster
    # than it fixes it, and the rendered result is a jumble of arbitrary
    # polygons. Josh, looking at exactly that output: "the whole placement is
    # wrong."
    #
    # What is missing is a way to cut only the stretch a crease actually covers.
    # Clipping the cut to the detected segment's extent was tried and did not
    # help, because on this roof the creases span most of the building anyway.
    USE_IMAGERY_CUTS = False

    cells = [footprint]
    if imagery_ds is not None and USE_IMAGERY_CUTS:
        # NOT a bare except. A rewrite of roof_outline above once deleted
        # _line_is_real while leaving this call site, and a broad except turned
        # that into "imagery cuts silently do nothing" -- the measurements looked
        # plausible and were meaningless. Import and geometry failures are the
        # only ones worth tolerating here.
        try:
            # Via roof_line_source so a vision model can propose these
            # instead. With no model prediction on disk this is the same call
            # it always was -- see that module's header for the fusion rule.
            from src.roof_line_source import strong_lines
            for ang, off in strong_lines(imagery_ds, footprint, building_id):
                nxt = []
                for c in cells:
                    parts = (_cut(c, ang, off)
                             if _line_is_real(c, _points_in(c, inside), ang, off) else [])
                    nxt.extend(parts if len(parts) >= 2 else [c])
                cells = nxt
        except (ImportError, ValueError, AttributeError) as exc:
            print(f"  roof_partition: imagery cuts unavailable ({exc!r})", flush=True)
            cells = [footprint]

    faces = []
    for cell in cells:
        faces.extend(_partition(cell, _points_in(cell, inside)))
    if not faces:
        return []

    # Second pass: if the first one reveals a recessed section, partition the
    # section and the roof around it separately. A cut that spans the region can
    # never isolate something in the middle of it, so this cannot be done by
    # offering another candidate to _best_cut -- it has to change what is being
    # cut. Only re-run when a section is actually found, so ordinary roofs pay
    # one plane-envelope evaluation and nothing else.
    recess = _recessed_region(footprint, inside, faces)
    if recess is not None:
        rest = footprint.difference(recess)
        pieces = [recess] + [q for q in getattr(rest, "geoms", [rest])
                             if isinstance(q, Polygon) and q.area >= MIN_PIECE_M2]
        if len(pieces) >= 2:
            regrown = []
            for piece in pieces:
                if piece is recess:
                    # Josh: "There are only two faces in the middle recession. It
                    # is two four sided shapes connected in the middle." A budget
                    # of two allows exactly one cut -- the line where they join.
                    # Unbudgeted, the recursion split the section into four.
                    regrown.extend(_partition(piece, _points_in(piece, inside),
                                              budget=[RECESS_MAX_FACES]))
                else:
                    regrown.extend(_partition(piece, _points_in(piece, inside)))
            if regrown:
                faces = regrown

    faces = _merge_bridgeable(faces, inside)

    # Step risers on stepped houses remain an OPEN problem, deliberately.
    # 26 Panorama Terrace carries 36.8 m2 of 42-44 degree strips between its
    # near-flat bands, and every local-geometry rule tried on 29 Aug also
    # deleted a real roof somewhere else. The measurements, so this is not
    # retried blind: (1) fit cannot reject a wall -- Panorama's risers fit 89%,
    # a wall is a perfect plane. (2) slope + short downhill run also matches
    # small real gables: 4725529 is a genuine 44-degree pair with a 2.9 m run
    # and 4734699 an entire 39-degree house. (3) "every neighbour much flatter"
    # fails because the risers touch a smeared 24-degree transition face.
    # (4) requiring a candidate pair to share a ridge does not separate them:
    # Panorama's two strips SHARE a 13.4 m edge at the top of both, exactly like
    # a gable pair -- at 0.42 m point spacing a smeared step edge is
    # geometrically a small steep roof. The remaining signal is imagery: a real
    # gable shows a sunlit/shaded pair, a step shows a shadow line.

    # NO eave extension. Josh's decision (29 Aug): roof beyond the LINZ
    # footprint stays out of the model. It was also placing panels on air --
    # 45 Camp St showed panels overlapping the edge 'floating with nothing
    # underneath them', because a face grown past the footprint carries its
    # panel grid with it. _extend_to_eave is kept for reference but not called.

    out = []
    for poly, plane in faces:
        if poly.area < MIN_FACET_M2:
            continue
        sub = _points_in(poly, inside)
        slope, aspect = _slope_aspect(plane)
        if slope > config.MAX_ROOF_SLOPE_DEG:
            continue
        if slope >= STEEP_FACE_DEG and _inlier_fraction(sub, plane) < STEEP_FACE_MIN_FIT:
            continue        # steep AND not a plane: a wall, not a roof face
        out.append({
            "building_id": building_id,
            "geometry": Polygon(poly.exterior, [r for r in poly.interiors]),
            "plane_a": plane[0], "plane_b": plane[1], "plane_c": plane[2],
            "slope_deg": slope, "aspect_deg": aspect,
            "area_m2": float(poly.area), "point_count": int(len(sub)),
        })
    return out

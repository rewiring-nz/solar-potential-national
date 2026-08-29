"""
Per building footprint: extract the DSM patch, split into planar roof
facets, drop facets over config.MAX_ROOF_SLOPE_DEG or too small to hold a
panel.

Method: multi-plane RANSAC directly on the DSM's pixel grid (each valid
pixel inside the footprint treated as an (x, y, z) point). This is the
grid-based variant of the standard LiDAR-roof-segmentation approach from
the literature -- we don't have the raw point cloud locally (only the DSM
raster), so pixel centres stand in for points. At 1m resolution a small
garage roof is only a handful of pixels, which is the real precision
ceiling of this approach; it's fine for the pilot, and swapping in the
raw LAZ point cloud later (higher point density) would be a drop-in
upgrade to `points_from_window` without touching the RANSAC/vectorize code.
"""

import sys
from pathlib import Path

import cv2
import math
import numpy as np
import rasterio
import shapely.vectorized
from rasterio.features import rasterize, shapes as rasterio_shapes
from rasterio.mask import mask as rasterio_mask
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as sparse_connected_components
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPoint, Point, Polygon, shape as shapely_shape
from shapely.ops import polygonize, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# RANSAC needs randomness, but a single shared RNG instance makes a building's result depend on
# how many other buildings were processed before it in the same run -- same input, different output
# depending on unrelated history, which is a debugging trap (bit us once: a standalone repro of one
# building disagreed with its result inside a full pipeline run, purely from RNG state drift). Each
# building gets its own RNG seeded from its own id instead, so results are independent of run order.

MIN_FACET_AREA_M2 = 3.0  # below this, can't usefully fit even one setback-shrunk panel
# Vertical residual to count as an inlier. Was 0.15 (~the DSM's raw noise
# floor), which sounds like the "correct" physical value but was far too
# tight in practice: real roofing has enough small-scale texture (seams,
# ribs, snow, minor sensor noise) that a single true flat plane routinely
# failed to pass a 0.15m-tolerance fit as one piece, fragmenting into many
# small, spurious "planes" instead -- exactly the "dozens of tiny facet
# outlines on an obviously simple roof" pattern reported against the live
# map. Raised to 0.35 after directly measuring the tradeoff on a 120-
# building sample: coverage rises 57%->71% and facets/building *drops*
# (3.1->2.7, i.e. less fragmentation, not less precision) between 0.15 and
# 0.35, with no increase in a same-building proxy for "wrongly merged two
# real roof planes into one" (11/120 flagged at both 0.30 and 0.35) --
# that failure mode only shows up past ~0.40, where it climbs to 13-14/120.
RANSAC_DISTANCE_THRESHOLD_M = 0.35
# On a *shallow*-pitched multi-face roof (a gentle hip/pyramid, ~10-11 deg
# per face -- confirmed directly from the DSM on a reported building:
# ~1.9m of true elevation change peak-to-eave), 0.35m of *vertical*
# tolerance corresponds to a wide *lateral* bleed zone across a ridge
# (0.35 / tan(11 deg) ~= 1.8m) -- enough for a plane fit to one face to
# also swallow points from an adjacent, differently-facing one near the
# hip line, producing one wrong "flat" facet across the whole roof.
# Tried fixing this with a second, tighter inlier-acceptance threshold
# (flat, and separately slope-proportional) -- both measurably fixed the
# reported building, but both also caused real, measured regressions on
# already-working buildings (e.g. one dropped from 978 to 719 panels).
# Reverted rather than ship a net-negative trade -- this specific pattern
# (shallow multi-face roofs, face-to-face slope differences small relative
# to 1m-DSM noise) is left as a known limitation for now, same category as
# the sawtooth-roof resolution ceiling documented above.
RANSAC_ITERATIONS = 300
RANSAC_MIN_INLIERS = 6  # pixels; below this a "plane" is just noise, not a real facet
RANSAC_SAMPLE_RADIUS_M = 3.0  # max plan-view spread of a 3-point candidate sample -- see ransac_planes
# Silently capped large/complex buildings: a big multi-wing institutional
# roof genuinely needs 15-20+ distinct planes, and this hard cap stopped
# RANSAC after 6, leaving ~40% of one real building's roof area (including
# an entire clean flat section) with no facet, no obstruction check, and
# no panels -- not because it wasn't viable, but because segmentation gave
# up before reaching it. 40 comfortably covers what real buildings in the
# pilot needed (checked: one large complex fully resolved by ~20); RANSAC
# still stops earlier on its own once no significant plane remains, so
# this doesn't slow down the many small/simple buildings at all.
MAX_PLANES_PER_BUILDING = 40


def points_from_window(dsm_array, window_transform, nodata):
    """dsm_array: 2D window clipped to (roughly) one building. Returns
    (points[N,3] in x,y,z world coords, row_idx[N], col_idx[N])."""
    rows, cols = np.where(dsm_array != nodata)
    if len(rows) == 0:
        return np.empty((0, 3)), rows, cols
    xs, ys = rasterio.transform.xy(window_transform, rows, cols)
    zs = dsm_array[rows, cols]
    points = np.column_stack([xs, ys, zs])
    return points, rows, cols


def fit_plane_lstsq(points):
    """points[N,3] -> (a, b, c) minimizing sum((a*x+b*y+c - z)^2).

    Solved about the points' own centroid and shifted back, so callers see the
    same (a, b, c) convention. This used to solve on raw NZTM coordinates -- x
    near 1.2 million, y near 5 million, against a column of ones, a condition
    number around 1e6 -- and the failure mode is not a little lost precision but
    planes that do not fit their own points. Found in the partition, where the
    plane fitted to the union of two faces measured 0.1 degrees apart with a
    0.00 m step at their join scored 16% on-plane against 99% for each face
    alone; that blocked a merge that should plainly have happened, and left
    5 Isle St at five faces where Josh counted three. Fixing it there took that
    roof to exactly three, and 47 Stanley St from 18 faces at 93% to 11 at 97%.

    This function has eleven call sites and the centred variant below has one,
    so the same fault was reachable through most of the segmenter."""
    x0, y0 = points[:, 0].mean(), points[:, 1].mean()
    A = np.column_stack([points[:, 0] - x0, points[:, 1] - y0, np.ones(len(points))])
    coeffs, *_ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    return np.array([a, b, float(coeffs[2]) - a * x0 - b * y0])


def fit_plane_lstsq_centered(points):
    """Exact alias of fit_plane_lstsq, kept only so existing call sites and
    saved references keep working.

    This was once a genuinely different function: fit_plane_lstsq solved on raw
    NZTM coordinates and this one centred first. That difference is gone --
    fit_plane_lstsq now centres internally -- and the two are bit-for-bit
    identical on random roof-scale inputs. The old docstring here still claimed
    it existed *because* the other one was un-centred, which would send the next
    reader looking for a numerical difference that no longer exists.
    """
    return fit_plane_lstsq(points)


def plane_residuals(points, plane):
    a, b, c = plane
    pred = a * points[:, 0] + b * points[:, 1] + c
    return np.abs(pred - points[:, 2])


def ransac_planes(points, rng, distance_threshold=RANSAC_DISTANCE_THRESHOLD_M,
                   iterations=RANSAC_ITERATIONS, min_inliers=RANSAC_MIN_INLIERS,
                   max_planes=MAX_PLANES_PER_BUILDING):
    """Iteratively extract dominant planes. Returns list of (plane, inlier_mask_into_points)."""
    remaining_idx = np.arange(len(points))
    planes = []

    while len(remaining_idx) >= min_inliers and len(planes) < max_planes:
        pts = points[remaining_idx]
        best_inlier_local = None

        if len(pts) < 3:
            break

        # 3-point samples are drawn from a small spatial neighbourhood (see
        # RANSAC_SAMPLE_RADIUS_M below), not from anywhere in the whole
        # point cloud -- found directly on a real hip/pyramid roof (~4m of
        # true elevation change, individual faces around 20-30 deg): fully
        # random sampling let a 3-point sample straddle multiple true faces
        # and construct a spurious near-flat "compromise" plane that, on a
        # roughly symmetric multi-face roof, can rack up MORE inliers within
        # tolerance than any single true face does (many points sit near the
        # roof's average elevation even though they're on four different
        # slopes) -- exactly the "thinks it's all one flat plane" failure
        # reported against a real building. A spatially-local sample can't
        # span two faces of a normal-sized roof, so it can't construct that
        # cross-face compromise plane in the first place.
        tree = cKDTree(pts[:, :2])
        for _ in range(iterations):
            anchor = rng.integers(len(pts))
            neighbor_idx = tree.query_ball_point(pts[anchor, :2], RANSAC_SAMPLE_RADIUS_M)
            if len(neighbor_idx) < 3:
                continue
            sample_idx = rng.choice(neighbor_idx, size=3, replace=False)
            sample = pts[sample_idx]
            # Skip near-degenerate (collinear) samples -- cross product of
            # two edge vectors near zero means no well-defined plane normal.
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            normal = np.cross(v1, v2)
            if np.linalg.norm(normal[:2]) > 1e6 or abs(normal[2]) < 1e-9:
                continue
            try:
                plane = fit_plane_lstsq(sample)
            except np.linalg.LinAlgError:
                continue
            residuals = plane_residuals(pts, plane)
            inlier_local = residuals < distance_threshold
            if best_inlier_local is None or inlier_local.sum() > best_inlier_local.sum():
                best_inlier_local = inlier_local

        if best_inlier_local is None or best_inlier_local.sum() < min_inliers:
            break

        # Refit on all inliers for a stabler plane, then recompute the
        # inlier set once against that refit plane.
        refit_plane = fit_plane_lstsq(pts[best_inlier_local])
        residuals = plane_residuals(pts, refit_plane)
        inlier_local = residuals < distance_threshold
        if inlier_local.sum() < min_inliers:
            break

        global_inlier_idx = remaining_idx[inlier_local]
        planes.append((refit_plane, global_inlier_idx))
        remaining_idx = remaining_idx[~inlier_local]

    return planes


def slope_aspect_from_plane(a, b):
    """z = a*x + b*y + c. Returns (slope_deg, aspect_deg). Aspect is the
    compass bearing (0=N, 90=E, clockwise) the surface faces -- i.e. the
    downhill direction, which is also the direction a mounted panel would
    face. Derivation: downhill vector = -(a, b); bearing = atan2(east, north)."""
    slope_deg = np.degrees(np.arctan(np.hypot(a, b)))
    aspect_deg = np.degrees(np.arctan2(-a, -b)) % 360
    return slope_deg, aspect_deg


RECT_FIT_MIN_FILL_FRACTION = 0.7  # hull area / its minimum-rotated-rectangle area
SHAPE_FIT_TOLERANCE_M = 1.0  # how far past the traced pixel footprint a fitted
# rectangle/hull may reach when snapping to clean edges -- enough to smooth
# 1m-grid staircase noise and small dropout notches, not enough for a
# handful of stray far-flung inlier points (RANSAC noise, or a coincidental
# planar match on a separate, physically unconnected roof wing) to balloon
# the fitted shape across the whole building. Found by direct testing: an
# earlier, unbounded version of this fit let one dominant plane's rectangle
# bleed across and eat several other, physically separate roof wings on a
# real building, leaving them as thin useless slivers.


def component_shape(points_xy, component_mask, window_transform):
    """Reconstruct one connected component's facet boundary geometrically
    from its actual inlier point locations, instead of tracing the raw
    per-pixel mask -- but bounded to stay near where pixels were actually
    claimed.

    Tracing the pixel mask (the original approach) makes the facet boundary
    exactly as noisy as the 1m DSM grid and whatever RANSAC noise excluded
    individual pixels near the edges -- it produces blobby, undersized
    shapes that don't look like the roof plane they represent, and panel
    packing (which aligns to the facet's own minimum-rotated-rectangle)
    inherits that same misalignment. A real roof plane's footprint is
    almost always a simple, mostly-rectangular polygon, so: take the convex
    hull of the inlier points; if it already fills most (>=70%) of its own
    minimum-rotated-rectangle, the "gaps" are just edge noise/small
    dropouts, not a genuine non-rectangular notch, so snap to the clean
    rectangle. Otherwise (a genuinely L-shaped/hipped point cloud) keep the
    hull. Either way, clip the result to the traced pixel footprint buffered
    by SHAPE_FIT_TOLERANCE_M -- the hull/rectangle is fit from point
    *locations* alone, with no idea how sparse or outlier-driven those
    points are, so left unbounded it can reach far past the real plane."""
    if len(points_xy) < 3:
        return None
    hull = MultiPoint(points_xy).convex_hull
    if hull.geom_type != "Polygon" or hull.area <= 0:
        return None  # collinear/degenerate -- not a usable facet shape
    min_rect = hull.minimum_rotated_rectangle
    if min_rect.geom_type == "Polygon" and min_rect.area > 0 and hull.area / min_rect.area >= RECT_FIT_MIN_FILL_FRACTION:
        candidate = min_rect
    else:
        candidate = hull

    traced = [
        shapely_shape(geom)
        for geom, val in rasterio_shapes(component_mask.astype(np.uint8), mask=component_mask, transform=window_transform)
        if val == 1
    ]
    if not traced:
        return None
    bound = unary_union(traced).buffer(SHAPE_FIT_TOLERANCE_M, join_style="mitre", mitre_limit=5)
    bounded = candidate.intersection(bound)
    if bounded.is_empty:
        return None
    if bounded.geom_type == "MultiPolygon":
        bounded = max(bounded.geoms, key=lambda p: p.area)
    elif bounded.geom_type != "Polygon":
        return None
    return bounded


def _dedupe_overlaps(facets, min_area):
    """Facet shapes now come from a fitted rectangle/hull over each plane's
    points (see component_shape) rather than the raw claimed-pixel mask, so
    two facets from different planes can end up overlapping where one
    plane's fitted rectangle bleeds past its actual pixels into a
    neighbour's territory -- pixel-mask tracing couldn't do that (each
    pixel could only belong to one plane), but a geometric fit can.
    Processes largest-first and subtracts already-claimed area from each
    smaller facet, so no roof area (and no panel) is ever double-counted
    across two overlapping facets."""
    ordered = sorted(facets, key=lambda f: -f["area_m2"])
    claimed = None
    result = []
    for f in ordered:
        geom = f["geometry"] if claimed is None else f["geometry"].difference(claimed)
        if not geom.is_empty:
            pieces = list(geom.geoms) if geom.geom_type in ("MultiPolygon", "GeometryCollection") else [geom]
            for piece in pieces:
                if piece.geom_type == "Polygon" and piece.area >= min_area:
                    new_f = dict(f)
                    new_f["geometry"] = piece
                    new_f["area_m2"] = piece.area
                    result.append(new_f)
        claimed = f["geometry"] if claimed is None else claimed.union(f["geometry"])
    return result


MERGE_SLOPE_DIFF_DEG = 5.0
MERGE_ASPECT_DIFF_DEG = 20.0
MERGE_LOW_SLOPE_DEG = 7.0  # below this, aspect is noise (near-flat roof) -- ignore it for merging
MERGE_BUFFER_M = 0.5  # facets within this gap still count as "adjacent" (grid/RANSAC edge noise)


def _circular_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def merge_similar_facets(facets):
    """Two RANSAC passes over the same physical plane (common on large
    near-flat roofs, where residual noise lets a second near-duplicate
    plane pass the inlier threshold) leave behind two adjacent facets with
    near-identical slope/aspect. Merge those back into one."""
    if len(facets) < 2:
        return facets

    n = len(facets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = facets[i], facets[j]
            slope_close = abs(fi["slope_deg"] - fj["slope_deg"]) <= MERGE_SLOPE_DIFF_DEG
            both_flat = fi["slope_deg"] < MERGE_LOW_SLOPE_DEG and fj["slope_deg"] < MERGE_LOW_SLOPE_DEG
            aspect_close = both_flat or _circular_diff(fi["aspect_deg"], fj["aspect_deg"]) <= MERGE_ASPECT_DIFF_DEG
            if not (slope_close and aspect_close):
                continue
            if fi["geometry"].buffer(MERGE_BUFFER_M).intersects(fj["geometry"].buffer(MERGE_BUFFER_M)):
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for idx_group in groups.values():
        if len(idx_group) == 1:
            merged.append(facets[idx_group[0]])
            continue
        group_facets = [facets[i] for i in idx_group]
        total_pts = sum(f["point_count"] for f in group_facets) or 1
        merged_geom = unary_union([f["geometry"] for f in group_facets])
        if merged_geom.geom_type == "MultiPolygon":
            # The buffer-touches check that grouped these said they were
            # adjacent, but the union came out disconnected anyway (touching
            # only at a single point, or the buffer tolerance let through a
            # near-miss) -- keep the largest part rather than pass a
            # MultiPolygon downstream, which panel_fitting and the renderer
            # both assume never happens for a single facet.
            merged_geom = max(merged_geom.geoms, key=lambda p: p.area)
        a = sum(f["plane_a"] * f["point_count"] for f in group_facets) / total_pts
        b = sum(f["plane_b"] * f["point_count"] for f in group_facets) / total_pts
        c = sum(f["plane_c"] * f["point_count"] for f in group_facets) / total_pts
        slope_deg, aspect_deg = slope_aspect_from_plane(a, b)
        merged.append({
            "building_id": group_facets[0]["building_id"],
            "plane_a": a, "plane_b": b, "plane_c": c,
            "slope_deg": slope_deg,
            "aspect_deg": aspect_deg,
            "area_m2": merged_geom.area,
            "point_count": total_pts,
            "geometry": merged_geom,
        })

    return merged


def segment_building(dsm_ds, building_geom, building_id, ransac_distance_threshold=None, min_facet_area_m2=None):
    """Returns a list of facet dicts for one building footprint (shapely
    geometry, in the DSM's CRS). ransac_distance_threshold/min_facet_area_m2
    override the module defaults -- exposed so a caller (e.g. the live
    parameter-tuning server) can experiment without editing config."""
    try:
        window_array, window_transform = rasterio_mask(
            dsm_ds, [building_geom], crop=True, nodata=dsm_ds.nodata, filled=True
        )
    except ValueError:
        return []  # geometry doesn't overlap the raster at all
    return _segment_from_window(window_array[0], window_transform, dsm_ds.nodata, building_geom, building_id,
                                 ransac_distance_threshold, min_facet_area_m2)


def segment_building_from_pointcloud(pc_source, building_geom, building_id, resolution=0.3,
                                      ransac_distance_threshold=None, min_facet_area_m2=None):
    """Same segmentation, sourced from the raw LiDAR point cloud (rasterized
    onto a fine grid, see pointcloud_source.rasterize_pointcloud_window)
    instead of the 1m DSM -- everything past that point is identical,
    exactly the "drop-in upgrade to points_from_window" this module's own
    docstring anticipated from the start."""
    from src.pointcloud_source import rasterize_pointcloud_window
    window_array, window_transform, nodata = rasterize_pointcloud_window(pc_source, building_geom, resolution)
    if window_array is None:
        return []
    return _segment_from_window(window_array[0], window_transform, nodata, building_geom, building_id,
                                 ransac_distance_threshold, min_facet_area_m2)


def _segment_from_window(window_array, window_transform, nodata, building_geom, building_id,
                          ransac_distance_threshold=None, min_facet_area_m2=None):
    min_facet_area_m2 = MIN_FACET_AREA_M2 if min_facet_area_m2 is None else min_facet_area_m2
    points, rows, cols = points_from_window(window_array, window_transform, nodata)
    if len(points) < RANSAC_MIN_INLIERS:
        return []

    rng = np.random.default_rng(building_id)
    ransac_kwargs = {} if ransac_distance_threshold is None else {"distance_threshold": ransac_distance_threshold}
    planes = ransac_planes(points, rng, **ransac_kwargs)

    facets = []
    for plane, inlier_idx in planes:
        a, b, c = plane
        slope_deg, aspect_deg = slope_aspect_from_plane(a, b)
        if slope_deg > config.MAX_ROOF_SLOPE_DEG:
            continue

        facet_mask = np.zeros(window_array.shape, dtype=bool)
        facet_mask[rows[inlier_idx], cols[inlier_idx]] = True
        labeled, n_components = ndimage.label(facet_mask)
        point_labels = labeled[rows[inlier_idx], cols[inlier_idx]]

        for label_id in range(1, n_components + 1):
            component_idx = inlier_idx[point_labels == label_id]
            component_mask = labeled == label_id
            polygon = component_shape(points[component_idx, :2], component_mask, window_transform)
            if polygon is None:
                continue

            # The fitted rectangle/hull is built from DSM point locations,
            # which can reach slightly past the true roofline; the building
            # outline is imagery-derived (0.1m) and traces the real edge,
            # so clip back to it.
            polygon = polygon.intersection(building_geom)
            if polygon.is_empty:
                continue
            if polygon.geom_type == "MultiPolygon":
                polygon = max(polygon.geoms, key=lambda p: p.area)
            elif polygon.geom_type not in ("Polygon",):
                continue  # intersection degenerated to a line/point -- not a usable facet
            if polygon.area < min_facet_area_m2:
                continue

            facets.append({
                "building_id": building_id,
                "plane_a": a, "plane_b": b, "plane_c": c,
                "slope_deg": slope_deg,
                "aspect_deg": aspect_deg,
                "area_m2": polygon.area,
                "point_count": len(component_idx),
                "geometry": polygon,
            })

    facets = _dedupe_overlaps(facets, min_facet_area_m2)
    return merge_similar_facets(facets)


# --- Image-guided segmentation ---------------------------------------------
#
# The RANSAC approach above asks the 1m DSM to do two things at once: find
# where a roof divides into separate faces (the boundary) and find each
# face's slope/aspect (the fit) -- and it's the *boundary* discovery that
# breaks down on shallow multi-face roofs (see RANSAC_DISTANCE_THRESHOLD_M's
# comment): a compromise plane spanning several true faces can out-compete
# any single real face for inlier count. But roof *edges* are usually
# clearly visible in the 0.1m RGB imagery even when the DSM can't resolve
# the slope difference across them -- confirmed directly: on a real ~10-11
# deg hip/pyramid roof RANSAC kept merging into one flat facet, Canny edge
# detection + a Hough transform on the same building's imagery found all
# four ridge lines cleanly.
#
# So: detect candidate interior ridge/hip lines from the imagery, partition
# the building polygon along them, then fit each resulting wedge's
# slope/aspect from just the DSM points inside it -- a much easier problem
# than discovering the wedge boundary and its slope simultaneously. Falls
# back to the pure-DSM segment_building() at several points whenever the
# image doesn't yield a confident partition, rather than trusting a
# possibly-wrong line: no interior lines found, a wedge's own points don't
# fit a plane well (the "ridge line" was probably noise -- a shadow edge,
# a roofing seam, not a real slope change), or the result doesn't cover
# enough of the footprint to be worth preferring over the fallback.

ROOFLINE_CANNY_LOW, ROOFLINE_CANNY_HIGH = 40, 120
ROOFLINE_HOUGH_THRESHOLD = 25
ROOFLINE_HOUGH_MIN_LINE_LENGTH_PX = 15  # ~1.5m at 0.1m/px
ROOFLINE_HOUGH_MAX_LINE_GAP_PX = 8
# A detected segment whose whole length sits this close to the building's
# own outline is just re-finding the eave/edge, which we already have
# precisely from the (0.1m, imagery-derived) building outline polygon --
# only *interior* divisions add anything.
ROOFLINE_BOUNDARY_EXCLUSION_M = 1.0
ROOFLINE_MIN_INTERIOR_LENGTH_M = 2.0  # shorter than this, after clipping to the footprint, isn't a real dividing line
ROOFLINE_CLUSTER_ANGLE_DEG = 8.0  # candidate segments this close in angle and...
ROOFLINE_CLUSTER_DIST_M = 1.5  # ...this close in perpendicular offset are treated as the same real line, not two
ROOFLINE_WEDGE_MIN_POINTS = 6
ROOFLINE_WEDGE_MAX_RMS_RESIDUAL_M = 0.35  # if a wedge's best-fit plane doesn't explain its own points this well,
# the line that created it probably wasn't a real ridge -- distrust the whole partition rather than one wedge,
# since a wrong line usually means neighbouring wedges are wrong too (they share that boundary)
ROOFLINE_MIN_COVERAGE_FRACTION = 0.5  # image-guided facets must explain at least this much of the footprint
# NOT SAFE TO RAISE AS A FIX: found in testing that any threshold high enough to catch the real
# failure mode this is meant to guard against (a bad partition silently dropping a whole legitimate
# wing of a multi-section building, while still clearing 50%) also rejects the one repeatedly-
# verified genuine improvement this whole feature produced (a shallow hip roof whose real facets
# only explain ~51% of its footprint, because the rest is a separate lower structure the wedge
# partition was never meant to cover). A single scalar coverage threshold can't tell "genuinely
# incomplete but correct" apart from "wrongly dropped real area" -- that needs the partition to
# reconcile against what it *didn't* explain, not just total up what it did. Left low and this
# function is NOT wired into the live pipeline (see README) rather than shipping on an
# unconvincing safety net.
# to be preferred over falling back to the RANSAC result


def _segment_angle_deg(seg):
    (x1, y1), (x2, y2) = seg.coords[0], seg.coords[-1]
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def _perp_distance_to_line(point_xy, seg):
    (x1, y1), (x2, y2) = seg.coords[0], seg.coords[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = np.hypot(dx, dy)
    if norm < 1e-9:
        return seg.distance(Point(point_xy))
    px, py = point_xy
    return abs(dx * (y1 - py) - dy * (x1 - px)) / norm


def _detect_interior_roof_lines(imagery_ds, building_geom):
    """Returns candidate interior ridge/hip LineStrings (world CRS) from
    Canny + Hough on the 0.1m RGB imagery, excluding anything that's just
    tracing the building's own outline."""
    pad = 2
    minx, miny, maxx, maxy = building_geom.bounds
    try:
        window = rasterio.windows.from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, imagery_ds.transform)
        arr = imagery_ds.read([1, 2, 3], window=window)
    except Exception:
        return []
    if arr.size == 0 or arr.shape[1] < 5 or arr.shape[2] < 5:
        return []
    rgb = np.moveaxis(arr, 0, -1).astype(np.uint8)
    wt = imagery_ds.window_transform(window)

    building_mask = rasterize([(building_geom, 1)], out_shape=rgb.shape[:2], transform=wt).astype(np.uint8)
    building_mask = cv2.dilate(building_mask, np.ones((5, 5), np.uint8))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, ROOFLINE_CANNY_LOW, ROOFLINE_CANNY_HIGH)
    edges[building_mask == 0] = 0

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=ROOFLINE_HOUGH_THRESHOLD,
        minLineLength=ROOFLINE_HOUGH_MIN_LINE_LENGTH_PX, maxLineGap=ROOFLINE_HOUGH_MAX_LINE_GAP_PX,
    )
    if lines is None:
        return []

    boundary = building_geom.exterior
    candidates = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        wx1, wy1 = wt * (x1, y1)
        wx2, wy2 = wt * (x2, y2)
        seg = LineString([(wx1, wy1), (wx2, wy2)])
        if seg.length < 1.0:
            continue
        if (boundary.distance(Point(wx1, wy1)) < ROOFLINE_BOUNDARY_EXCLUSION_M
                and boundary.distance(Point(wx2, wy2)) < ROOFLINE_BOUNDARY_EXCLUSION_M):
            continue
        candidates.append(seg)
    return candidates


def _cluster_lines(candidates):
    """Dedupe near-duplicate detections of the same real ridge (Hough
    often returns several overlapping segments along one true line) --
    keep the longest in each angle+offset cluster."""
    accepted = []
    for seg in sorted(candidates, key=lambda s: -s.length):
        ang = _segment_angle_deg(seg)
        mid = seg.interpolate(0.5, normalized=True)
        if any(
            min(abs(ang - a_ang), 180 - abs(ang - a_ang)) < ROOFLINE_CLUSTER_ANGLE_DEG
            and _perp_distance_to_line((mid.x, mid.y), a_seg) < ROOFLINE_CLUSTER_DIST_M
            for a_seg, a_ang in accepted
        ):
            continue
        accepted.append((seg, ang))
    return [s for s, _ in accepted]


def _extend_and_clip(seg, building_geom):
    """Hough segments usually stop short of the true ridge's full extent
    (wherever contrast happened to drop) -- extend along the detected
    direction far past the building, then clip back to its true span."""
    (x1, y1), (x2, y2) = seg.coords[0], seg.coords[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = np.hypot(dx, dy)
    if norm < 1e-9:
        return None
    ux, uy = dx / norm, dy / norm
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bminx, bminy, bmaxx, bmaxy = building_geom.bounds
    diag = np.hypot(bmaxx - bminx, bmaxy - bminy) + 1
    extended = LineString([(cx - ux * diag, cy - uy * diag), (cx + ux * diag, cy + uy * diag)])
    clipped = extended.intersection(building_geom.buffer(0.01))
    if clipped.is_empty:
        return None
    if clipped.geom_type == "MultiLineString":
        clipped = max(clipped.geoms, key=lambda l: l.length)
    if clipped.geom_type != "LineString" or clipped.length < ROOFLINE_MIN_INTERIOR_LENGTH_M:
        return None
    return clipped


def _partition_by_lines(building_geom, lines):
    merged = unary_union([building_geom.exterior, *lines])
    polys = [p for p in polygonize(merged) if building_geom.buffer(0.05).contains(p.representative_point())]
    return polys


def _fit_wedge_facet(wedge_poly, points, rows, cols, window_shape, window_transform,
                      building_id, building_geom, min_facet_area_m2):
    wedge_mask = rasterize([(wedge_poly, 1)], out_shape=window_shape, transform=window_transform).astype(bool)
    in_wedge = wedge_mask[rows, cols]
    wedge_points = points[in_wedge]
    if len(wedge_points) < ROOFLINE_WEDGE_MIN_POINTS:
        return None
    plane = fit_plane_lstsq(wedge_points)
    rms = np.sqrt(np.mean(plane_residuals(wedge_points, plane) ** 2))
    if rms > ROOFLINE_WEDGE_MAX_RMS_RESIDUAL_M:
        return None
    a, b, c = plane
    slope_deg, aspect_deg = slope_aspect_from_plane(a, b)
    if slope_deg > config.MAX_ROOF_SLOPE_DEG:
        return None
    clipped = wedge_poly.intersection(building_geom)
    if clipped.is_empty:
        return None
    if clipped.geom_type == "MultiPolygon":
        clipped = max(clipped.geoms, key=lambda p: p.area)
    elif clipped.geom_type != "Polygon":
        return None
    if clipped.area < min_facet_area_m2:
        return None
    return {
        "building_id": building_id,
        "plane_a": a, "plane_b": b, "plane_c": c,
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "area_m2": clipped.area,
        "point_count": int(in_wedge.sum()),
        "geometry": clipped,
    }


ROOFLINE_WEDGE_MIN_AREA_M2 = 30.0  # a kept wedge must be at least this big -- the plain
# MIN_FACET_AREA_M2 floor (3.0m2, sized only for "can a panel fit") is far too permissive here: a
# small dormer/gable-end detail can legitimately pass every other check (a real slope difference,
# a clean per-side fit) and still not be worth its own facet. Tried this as a *fraction* of the
# building's footprint first (0.08) -- wrong shape for the problem: on a large multi-wing building,
# 8% of the total footprint is bigger than most of that building's own legitimately-small real
# facets (55 facets averaging ~52m2 each on one real building), rejecting genuine wings, not just
# dormer noise. Dormer/gable-end artifacts are small in *absolute* terms regardless of the
# building's overall size, so an absolute floor is the right shape of gate; this filters those out
# without needing to solve the harder problem of classifying "real major ridge" vs "real minor
# architectural detail" directly. Value chosen empirically: swept 15/20/30/40/60 against two
# buildings with confirmed false-ridge regressions (a plain gable and a zigzag-edged roof, both
# repeatedly fragmented by earlier, looser validation attempts) and four with confirmed real
# improvements -- 30 is the largest floor that still stabilises both bad cases while keeping the
# one real improvement (a shallow hip roof) that survives at every floor tested; the other three
# "improvements" turned out to only exist below this floor too, meaning they very likely shared the
# same over-fragmentation problem rather than being genuine wins -- losing them here is the correct
# outcome, not a regression.

ROOFLINE_VALIDATION_MIN_SLOPE_DIFF_DEG = 4.0
ROOFLINE_VALIDATION_MIN_ASPECT_DIFF_DEG = 15.0
ROOFLINE_VALIDATION_MIN_SIDE_POINTS = 10
# How far from the candidate line a point can be and still count towards
# testing it. First version of this check split the *whole* building by
# each line and compared the two halves -- too coarse: on a building with
# other real 3D features (dormers, a chimney, a neighbouring wing at a
# different pitch), each "half" bundles all of that together, so the
# comparison can show a large difference that has nothing to do with
# whether *this specific line* sits on a real ridge. Restricting to a
# narrow corridor tests only what actually changes right at the line.
ROOFLINE_VALIDATION_CORRIDOR_M = 3.0


def _line_side_mask(points_xy, seg):
    (x1, y1), (x2, y2) = seg.coords[0], seg.coords[-1]
    dx, dy = x2 - x1, y2 - y1
    cross = dx * (points_xy[:, 1] - y1) - dy * (points_xy[:, 0] - x1)
    return cross > 0


def _perp_distances_to_line(points_xy, seg):
    (x1, y1), (x2, y2) = seg.coords[0], seg.coords[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = np.hypot(dx, dy)
    if norm < 1e-9:
        return np.full(len(points_xy), np.inf)
    return np.abs(dx * (y1 - points_xy[:, 1]) - dy * (x1 - points_xy[:, 0])) / norm


def _line_is_real_ridge(seg, points):
    """Reject a candidate line unless the DSM itself shows a real slope or
    aspect difference between its two sides, tested using only points in a
    narrow corridor either side of the line -- found in testing that a
    plausible-looking image edge is often a shadow line or a fine roofing
    seam, not an actual change in pitch, and letting those through
    fragmented genuinely-uniform roofs into disconnected slivers (a plain
    gable house dropped from 73 correctly-packed panels to 30 scattered
    ones; `merge_similar_facets` was supposed to catch this after the fact
    by re-merging near-identical adjacent wedges, but small malformed
    wedges from a bad split have too few points for a stable fit, so their
    slope/aspect estimates scatter enough to dodge that check). An earlier
    version of this function compared the *whole* building split in two by
    each line, which was too coarse -- see ROOFLINE_VALIDATION_CORRIDOR_M."""
    perp = _perp_distances_to_line(points[:, :2], seg)
    near = perp <= ROOFLINE_VALIDATION_CORRIDOR_M
    side = _line_side_mask(points[:, :2], seg)
    pts_a, pts_b = points[near & side], points[near & ~side]
    if len(pts_a) < ROOFLINE_VALIDATION_MIN_SIDE_POINTS or len(pts_b) < ROOFLINE_VALIDATION_MIN_SIDE_POINTS:
        return False  # can't confidently test this line -- don't trust what can't be verified
    slope_a, aspect_a = slope_aspect_from_plane(*fit_plane_lstsq(pts_a)[:2])
    slope_b, aspect_b = slope_aspect_from_plane(*fit_plane_lstsq(pts_b)[:2])
    # A "line" whose corridor produces an implausibly steep side (found in
    # testing: 60+ deg, well past any real roof pitch) isn't dividing two
    # roof planes at all -- it's the base of a small dormer/gable-end
    # feature, where the corridor sample on that side is mostly the
    # feature's own near-vertical wall, not a second roof surface. That's
    # a real slope difference (the check below would happily pass it) but
    # not a useful *facet* division. MAX_ROOF_SLOPE_DEG is the pipeline's
    # own existing definition of "not roof" elsewhere in this file.
    if slope_a > config.MAX_ROOF_SLOPE_DEG or slope_b > config.MAX_ROOF_SLOPE_DEG:
        return False
    if slope_a < MERGE_LOW_SLOPE_DEG and slope_b < MERGE_LOW_SLOPE_DEG:
        return False  # both sides near-flat -- aspect is noise here, and there's no slope difference either
    return (abs(slope_a - slope_b) >= ROOFLINE_VALIDATION_MIN_SLOPE_DIFF_DEG
            or _circular_diff(aspect_a, aspect_b) >= ROOFLINE_VALIDATION_MIN_ASPECT_DIFF_DEG)


def segment_building_image_guided(dsm_ds, imagery_ds, building_geom, building_id,
                                   ransac_distance_threshold=None, min_facet_area_m2=None):
    """Image-guided segmentation with a safe fallback to the pure-DSM
    segment_building() -- see the module comment above `_segment_angle_deg`
    for the rationale. imagery_ds=None always uses the fallback (so
    existing callers that don't have imagery open keep working unchanged)."""
    min_facet_area_m2 = MIN_FACET_AREA_M2 if min_facet_area_m2 is None else min_facet_area_m2

    def fallback():
        return segment_building(dsm_ds, building_geom, building_id, ransac_distance_threshold, min_facet_area_m2)

    if imagery_ds is None:
        return fallback()

    try:
        window_array, window_transform = rasterio_mask(
            dsm_ds, [building_geom], crop=True, nodata=dsm_ds.nodata, filled=True
        )
    except ValueError:
        return fallback()
    window_array = window_array[0]
    points, rows, cols = points_from_window(window_array, window_transform, dsm_ds.nodata)
    if len(points) < RANSAC_MIN_INLIERS:
        return fallback()

    raw_lines = _detect_interior_roof_lines(imagery_ds, building_geom)
    clustered = _cluster_lines(raw_lines)
    clipped_lines = [l for l in (_extend_and_clip(s, building_geom) for s in clustered) if l is not None]
    validated_lines = [l for l in clipped_lines if _line_is_real_ridge(l, points)]
    if not validated_lines:
        return fallback()

    wedges = _partition_by_lines(building_geom, validated_lines)
    if len(wedges) < 2:
        return fallback()

    # Keep wedges whose own points fit a clean plane; drop the rest rather
    # than distrusting the whole partition on one bad wedge -- found in
    # testing that when the image detects only some of a roof's true ridge
    # lines (e.g. 3 of 4 on a hip roof, one lost to noise/low contrast),
    # the resulting partition has a mix of clean single-face wedges (which
    # fit beautifully) and a few malformed slivers straddling the missing
    # ridge (which correctly fail the fit check) -- rejecting everything
    # over those slivers would throw away the good wedges too.
    facets = [
        f for f in (
            _fit_wedge_facet(wedge, points, rows, cols, window_array.shape, window_transform,
                              building_id, building_geom,
                              max(min_facet_area_m2, ROOFLINE_WEDGE_MIN_AREA_M2))
            for wedge in wedges
        ) if f is not None
    ]

    if sum(f["area_m2"] for f in facets) < ROOFLINE_MIN_COVERAGE_FRACTION * building_geom.area:
        return fallback()

    return merge_similar_facets(facets)


# --- Point-native segmentation -----------------------------------------
#
# segment_building/_segment_from_window rasterize onto a regular grid --
# originally the DSM's own 1m grid, and ndimage.label (connected
# components) plus a rasterized pixel trace (bounding each facet's fitted
# shape) both assume that grid exists. That grid dependency is exactly
# what broke when denser point-cloud data was fed into it: re-rasterizing
# point-cloud points onto either a fine or a matching-resolution grid,
# confirmed directly, produced results that swung between over-merged and
# over-fragmented depending on threshold -- because the pipeline's
# thresholds were tuned against the DSM raster's specific noise/gridding
# characteristics, which a rasterized-then-regridded point cloud doesn't
# share, no matter how the regridding is tuned.
#
# This works directly on an irregular (x, y, z) point set instead, no
# grid anywhere. Same interface for points sourced from a rasterized DSM
# window (points_from_window) or a raw point-cloud query -- important for
# scaling beyond one pilot region: LiDAR point-cloud coverage isn't
# universal across NZ yet, so a national pipeline needs regions with only
# a DSM available to degrade to that without a separate code path.

POINTCLOUD_CLUSTER_RADIUS_M = 1.5  # two points this close (or connected via a chain of such
# gaps) count as the same physical patch of roof -- replaces the raster-grid adjacency
# ndimage.label used to provide, needed here since there's no grid at all to be adjacent on.


# --- Compromise-facet splitting via local surface normals ------------------
#
# RANSAC_DISTANCE_THRESHOLD_M's own comment documents the residual failure
# mode this targets: on a shallow multi-face roof, a plane correctly seeded
# from one real face can still have its *inlier acceptance* (a global
# residual test against every remaining point, with no idea which face a
# point is actually on) reach across a real ridge and absorb points from an
# adjacent, differently-facing plane -- because near the ridge, both faces'
# true heights sit within the necessarily-loose vertical tolerance of a
# single "compromise" plane. Two earlier fixes for this were tried and
# reverted: a globally tighter/slope-proportional threshold measurably
# fixed the reported case but caused real regressions elsewhere (one
# building dropped from 978 to 719 panels) -- reverted rather than ship a
# net-negative trade. Image-based ridge detection (segment_building_image_
# guided, above) found real ridges from RGB Canny/Hough edges, but shadows
# and roofing seams produce visually identical false edges, and repeated
# validation attempts couldn't separate the two reliably enough to ship.
#
# This instead asks the *point cloud itself*, not one global fit or an
# image, whether a facet's own points actually support a single plane:
# fit each point its own small local plane from just its nearest few
# neighbours, and look for a spatially coherent boundary between two
# neighbourhoods of differing local slope/aspect -- a real ridge is exactly
# that kind of boundary; a shadow or seam has no reason to line up with one
# in the 3D geometry, since it's a lighting/material artifact, not a slope
# change. This can still blur right at the ridge itself (a point's local
# neighbourhood there straddles both faces), but confirming a split needs a
# spatially *significant* cluster on each side, not a clean classification
# of every single point, so a thin blurred seam along the ridge doesn't
# defeat it as long as each face's bulk is far enough from the ridge to
# read cleanly (checked directly: true on every reported failing case).

LOCAL_NORMAL_K = 10  # neighbours for one point's local plane fit
LOCAL_NORMAL_RADIUS_M = 2.0  # cap on how far a "local" neighbour can be -- kept close to
# POINTCLOUD_CLUSTER_RADIUS_M/RANSAC_SAMPLE_RADIUS_M's own precedent for "still the same patch of
# roof"; wider would blur the very ridge this is trying to detect, narrower risks too few points
# per estimate on sparser parts of the point cloud
LOCAL_NORMAL_MIN_NEIGHBOURS = 5  # fewer than this and a local plane fit is just noise
SPLIT_MIN_POINTS = 20  # a facet needs enough points to support two independently-confident
# sub-planes, not just one -- below this, not worth the analysis
SPLIT_MIN_AREA_M2 = 15.0  # below this a facet can't plausibly hide a second real face of any
# useful size (each side would need to individually clear MIN_FACET_AREA_M2 downstream anyway)
SPLIT_CONNECT_RADIUS_M = LOCAL_NORMAL_RADIUS_M  # two points can only join the same sub-cluster
# if they were also close enough to plausibly share a local normal estimate in the first place
SPLIT_LOCAL_SLOPE_DIFF_DEG = 6.0
SPLIT_LOCAL_ASPECT_DIFF_DEG = 25.0
SPLIT_MIN_SUBCLUSTER_POINTS = RANSAC_MIN_INLIERS
SPLIT_MIN_SUBCLUSTER_AREA_M2 = 20.0  # a point-count floor alone lets a real but spatially-tiny
# texture artifact (a corrugated/ribbed roofing material's own alternating micro-facets, confirmed
# directly: a densely-sampled patch can pack dozens of points into under a square metre) through as
# "significant" -- an area floor this size is well above any single corrugation rib's own footprint
# but still well under a real minor roof wing, so it screens out texture noise without needing to
# tell texture and structure apart by any other means


def _local_normals(points):
    """Per-point local (slope_deg, aspect_deg) from a small least-squares
    plane fit over each point's own nearest neighbours. Physically grounded
    alternative to image-based ridge detection: local surface normals come
    straight from the 3D geometry, so they can't be fooled by a shadow edge
    or roofing seam the way Canny-on-RGB was (see the module comment
    above). Returns (slope_deg[N], aspect_deg[N]); a point with too few
    nearby neighbours gets NaN in both -- not enough local support to trust
    a normal there, rather than guessing from a diluted/distant sample."""
    n = len(points)
    slopes = np.full(n, np.nan)
    aspects = np.full(n, np.nan)
    if n < LOCAL_NORMAL_MIN_NEIGHBOURS + 1:
        return slopes, aspects
    tree = cKDTree(points[:, :2])
    k = min(LOCAL_NORMAL_K + 1, n)  # +1: a point's own nearest neighbour is itself, at distance 0
    dists, idxs = tree.query(points[:, :2], k=k)
    for i in range(n):
        row_dists, row_idx = np.atleast_1d(dists[i]), np.atleast_1d(idxs[i])
        mask = (row_dists <= LOCAL_NORMAL_RADIUS_M) & (row_idx != i)
        neighbor_idx = row_idx[mask]
        if len(neighbor_idx) < LOCAL_NORMAL_MIN_NEIGHBOURS:
            continue
        try:
            a, b, _ = fit_plane_lstsq(points[neighbor_idx])
        except np.linalg.LinAlgError:
            continue
        slopes[i], aspects[i] = slope_aspect_from_plane(a, b)
    return slopes, aspects


def _maybe_split_compromise_facet(comp_points, parent_plane):
    """Given one spatially-connected component of a RANSAC plane's inlier
    points (plus that plane's own (a,b,c)), check whether it's actually two
    or more real roof faces wrongly merged into one compromise plane (see
    the module comment above). Returns a list of (points_subset, plane)
    pairs: the single input unchanged if no confident split is found, or
    2+ freshly-refit groups otherwise.

    Splits are only kept if the resulting groups' *global* refit slope/
    aspect differ by more than merge_similar_facets' own thresholds would
    still merge back together later in the same pipeline run -- using the
    identical thresholds both directions on purpose, so a split can never
    produce two facets the very next step would just undo, which would
    otherwise waste the work and make the final boundary depend on
    incidental split/merge ordering instead of a real, confirmed
    difference."""
    if len(comp_points) < SPLIT_MIN_POINTS:
        return [(comp_points, parent_plane)]
    if MultiPoint(comp_points[:, :2]).convex_hull.area < SPLIT_MIN_AREA_M2:
        return [(comp_points, parent_plane)]

    local_slope, local_aspect = _local_normals(comp_points)
    valid_idx = np.where(~np.isnan(local_slope))[0]
    if len(valid_idx) < SPLIT_MIN_POINTS:
        return [(comp_points, parent_plane)]

    vpts = comp_points[valid_idx]
    vslope = local_slope[valid_idx]
    vaspect = local_aspect[valid_idx]

    tree = cKDTree(vpts[:, :2])
    pairs = tree.query_pairs(SPLIT_CONNECT_RADIUS_M, output_type="ndarray")
    if len(pairs) == 0:
        return [(comp_points, parent_plane)]

    both_flat = (vslope[pairs[:, 0]] < MERGE_LOW_SLOPE_DEG) & (vslope[pairs[:, 1]] < MERGE_LOW_SLOPE_DEG)
    slope_close = np.abs(vslope[pairs[:, 0]] - vslope[pairs[:, 1]]) <= SPLIT_LOCAL_SLOPE_DIFF_DEG
    araw = np.abs(vaspect[pairs[:, 0]] - vaspect[pairs[:, 1]]) % 360
    aspect_diff = np.minimum(araw, 360 - araw)
    aspect_close = both_flat | (aspect_diff <= SPLIT_LOCAL_ASPECT_DIFF_DEG)
    pairs = pairs[slope_close & aspect_close]

    n = len(vpts)
    if len(pairs) == 0:
        sub_components = [np.array([i]) for i in range(n)]
    else:
        row = np.concatenate([pairs[:, 0], pairs[:, 1]])
        col = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
        n_comp, labels = sparse_connected_components(graph, directed=False)
        sub_components = [np.where(labels == i)[0] for i in range(n_comp)]

    def _sub_area(c):
        if len(c) < 3:
            return 0.0
        hull = MultiPoint(vpts[c, :2]).convex_hull
        return hull.area if hull.geom_type == "Polygon" else 0.0

    significant = [c for c in sub_components
                   if len(c) >= SPLIT_MIN_SUBCLUSTER_POINTS and _sub_area(c) >= SPLIT_MIN_SUBCLUSTER_AREA_M2]
    if len(significant) < 2:
        return [(comp_points, parent_plane)]

    # Map back to indices into comp_points, then fold any leftover points
    # (too-small a cluster, or normal estimation failed) into whichever
    # kept cluster is spatially nearest -- dropping them instead would
    # silently shrink the split facets' own claimed area versus the
    # original unsplit one.
    sub_groups_local = [valid_idx[c] for c in significant]
    assigned = np.full(len(comp_points), -1)
    for gi, idxs in enumerate(sub_groups_local):
        assigned[idxs] = gi
    unassigned = np.where(assigned == -1)[0]
    if len(unassigned):
        centroids = np.array([comp_points[idxs, :2].mean(axis=0) for idxs in sub_groups_local])
        for i in unassigned:
            d = np.linalg.norm(centroids - comp_points[i, :2], axis=1)
            assigned[i] = int(np.argmin(d))

    candidate_groups = [comp_points[assigned == gi] for gi in range(len(sub_groups_local))]

    refit_planes = []
    for g in candidate_groups:
        try:
            refit_planes.append(fit_plane_lstsq(g))
        except np.linalg.LinAlgError:
            return [(comp_points, parent_plane)]

    def _distinct(pi, pj):
        si, ai = slope_aspect_from_plane(pi[0], pi[1])
        sj, aj = slope_aspect_from_plane(pj[0], pj[1])
        slope_close = abs(si - sj) <= MERGE_SLOPE_DIFF_DEG
        both_flat = si < MERGE_LOW_SLOPE_DEG and sj < MERGE_LOW_SLOPE_DEG
        aspect_close = both_flat or _circular_diff(ai, aj) <= MERGE_ASPECT_DIFF_DEG
        return not (slope_close and aspect_close)

    if not any(_distinct(refit_planes[i], refit_planes[j])
               for i in range(len(refit_planes)) for j in range(i + 1, len(refit_planes))):
        return [(comp_points, parent_plane)]

    return list(zip(candidate_groups, refit_planes))


def _cluster_points_spatially(points_xy, radius):
    """Connected-components over a point set via a spatial graph (edge
    between any two points within `radius`) instead of raster adjacency.
    Returns a list of index arrays (into points_xy), one per component."""
    n = len(points_xy)
    if n == 0:
        return []
    if n == 1:
        return [np.array([0])]
    tree = cKDTree(points_xy)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return [np.array([i]) for i in range(n)]
    row = np.concatenate([pairs[:, 0], pairs[:, 1]])
    col = np.concatenate([pairs[:, 1], pairs[:, 0]])
    graph = coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    n_components, labels = sparse_connected_components(graph, directed=False)
    return [np.where(labels == i)[0] for i in range(n_components)]


def component_shape_from_points(points_xy):
    """Like component_shape, but for a properly spatially-clustered point
    set with no raster trace to bound against. That bound existed to stop
    a rectangle/hull fit from reaching past stray, far-flung inlier points
    that shouldn't have counted as part of the same facet -- clustering
    itself already prevents that here, since a stray point wouldn't be in
    the same spatially-connected component to begin with."""
    if len(points_xy) < 3:
        return None
    hull = MultiPoint(points_xy).convex_hull
    if hull.geom_type != "Polygon" or hull.area <= 0:
        return None
    min_rect = hull.minimum_rotated_rectangle
    if min_rect.geom_type == "Polygon" and min_rect.area > 0 and hull.area / min_rect.area >= RECT_FIT_MIN_FILL_FRACTION:
        return min_rect
    return hull


def segment_points(points, building_geom, building_id, ransac_distance_threshold=None, min_facet_area_m2=None):
    """Point-native equivalent of segment_building -- see the module
    comment above for why this exists. `points` is an Nx3 (x, y, z) array
    in the building's world CRS, from any source."""
    min_facet_area_m2 = MIN_FACET_AREA_M2 if min_facet_area_m2 is None else min_facet_area_m2
    if len(points) < RANSAC_MIN_INLIERS:
        return []

    rng = np.random.default_rng(building_id)
    ransac_kwargs = {} if ransac_distance_threshold is None else {"distance_threshold": ransac_distance_threshold}
    planes = ransac_planes(points, rng, **ransac_kwargs)

    facets = []
    for plane, inlier_idx in planes:
        a, b, c = plane
        slope_deg, aspect_deg = slope_aspect_from_plane(a, b)
        if slope_deg > config.MAX_ROOF_SLOPE_DEG:
            continue

        # Tried inserting _maybe_split_compromise_facet here (per-component,
        # before shaping) to catch RANSAC's remaining compromise-plane
        # blind spot -- see that function's module comment. Direct
        # diagnostic testing on the reported failing buildings (#5371200,
        # #4734944, #4735310) found the local-normal signal is dominated by
        # roofing material micro-texture (corrugated/ribbed metal, tile
        # overlap) at the point densities available here, not real
        # macro-scale ridges: 90%+ of points read as "locally steep" in a
        # diffuse, non-clustered pattern, and the only sub-clusters that
        # passed a point-count-only significance filter were sub-10m2
        # noise. Adding an area floor big enough to reject that texture
        # noise made the split fire on none of the reported cases at all --
        # confirmed those "improvements" were the same noise, not real
        # splits (same self-correcting pattern documented for the
        # abandoned image-guided approach above). Left defined and unwired
        # rather than shipped: a third dead end on this specific problem,
        # not a threshold this needs tuned further.
        inlier_points = points[inlier_idx]
        for comp_local_idx in _cluster_points_spatially(inlier_points[:, :2], POINTCLOUD_CLUSTER_RADIUS_M):
            comp_points = inlier_points[comp_local_idx]
            if len(comp_points) < 3:
                continue
            polygon = component_shape_from_points(comp_points[:, :2])
            if polygon is None:
                continue

            polygon = polygon.intersection(building_geom)
            if polygon.is_empty:
                continue
            if polygon.geom_type == "MultiPolygon":
                polygon = max(polygon.geoms, key=lambda p: p.area)
            elif polygon.geom_type not in ("Polygon",):
                continue
            if polygon.area < min_facet_area_m2:
                continue

            facets.append({
                "building_id": building_id,
                "plane_a": a, "plane_b": b, "plane_c": c,
                "slope_deg": slope_deg,
                "aspect_deg": aspect_deg,
                "area_m2": polygon.area,
                "point_count": len(comp_points),
                "geometry": polygon,
            })

    facets = _dedupe_overlaps(facets, min_facet_area_m2)
    return merge_similar_facets(facets)


def segment_building_points_native(dsm_ds, building_geom, building_id,
                                    ransac_distance_threshold=None, min_facet_area_m2=None):
    """segment_points, sourced from the DSM (same points_from_window
    extraction segment_building uses) -- lets the point-native path be
    validated against the existing DSM-based pipeline on equal terms
    before pointing it at denser point-cloud data."""
    try:
        window_array, window_transform = rasterio_mask(
            dsm_ds, [building_geom], crop=True, nodata=dsm_ds.nodata, filled=True
        )
    except ValueError:
        return []
    points, _, _ = points_from_window(window_array[0], window_transform, dsm_ds.nodata)
    return segment_points(points, building_geom, building_id, ransac_distance_threshold, min_facet_area_m2)


def segment_building_from_pointcloud_native(pc_source, building_geom, building_id, pad_m=2.0,
                                             ransac_distance_threshold=None, min_facet_area_m2=None):
    """segment_points, sourced directly from the raw point cloud -- no
    rasterization step at all, unlike segment_building_from_pointcloud."""
    minx, miny, maxx, maxy = building_geom.bounds
    points = pc_source.points_in_bbox(minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m, building_only=True)
    return segment_points(points, building_geom, building_id, ransac_distance_threshold, min_facet_area_m2)


# --- Orientation-clustered segmentation (repetitive roof forms) ------------
#
# Region growing walks a neighbourhood graph, so it needs a contiguous
# surface; on a sawtooth it dies at the first fold and returns almost
# nothing (Turner St: 1 facet, 18m2 of 339m2). The global solver survives
# but fits planes ACROSS the teeth, producing shallow averaged facets.
#
# Yet the raw per-point normals resolve those teeth perfectly -- Turner St's
# are cleanly bimodal at ~135/315 deg with a 52 deg median slope, which is
# exactly what the heat map renders correctly while the facets do not. So
# for repetitive forms, cluster points BY ORIENTATION first, then split each
# orientation into spatially connected pieces: one facet per tooth face,
# with the tooth's true slope instead of an average across the fold.
ORIENT_BIN_DEG = 20.0           # aspect histogram resolution
ORIENT_MIN_MODE_SHARE = 0.12    # a mode must hold this share of steep points
ORIENT_ASSIGN_TOL_DEG = 30.0    # point-to-mode assignment half-width
ORIENT_LINK_DIST_M = 1.0        # plan-view spacing that keeps a face connected
ORIENT_MIN_FACET_PTS = 25


def segment_building_orientation_clustered(pc_source, building_geom, building_id,
                                            min_facet_area_m2=None):
    # callers may pass None to mean "use the module default"
    if min_facet_area_m2 is None:
        min_facet_area_m2 = MIN_FACET_AREA_M2
    import shapely.vectorized as _sv
    from scipy.sparse import coo_matrix as _coo
    from scipy.sparse.csgraph import connected_components as _cc

    pts = pc_source.points_in_bbox(*building_geom.bounds, building_only=True)
    if len(pts) < 60:
        return []
    inside = _sv.contains(building_geom, pts[:, 0], pts[:, 1])
    p = pts[inside]
    if len(p) < 60:
        return []
    normals, _rms = _pca_normals(p, k=RG_NORMAL_K)
    sgn = np.sign(normals[:, 2]); sgn[sgn == 0] = 1
    slope = np.degrees(np.arccos(np.clip(np.abs(normals[:, 2]), 0, 1)))
    aspect = (np.degrees(np.arctan2(normals[:, 0] * sgn, normals[:, 1] * sgn)) + 360) % 360
    steep = slope >= STRUCTURED_MIN_SLOPE_DEG
    if steep.sum() < 60:
        return []

    nbins = int(round(360 / ORIENT_BIN_DEG))
    hist, edges = np.histogram(aspect[steep], bins=nbins, range=(0, 360))
    modes = []
    for i in np.argsort(hist)[::-1]:
        if hist[i] < ORIENT_MIN_MODE_SHARE * steep.sum():
            break
        centre = (edges[i] + edges[i + 1]) / 2
        if any(abs(((centre - m + 180) % 360) - 180) <= ORIENT_ASSIGN_TOL_DEG for m in modes):
            continue
        modes.append(centre)
    if len(modes) < 2:
        return []  # not a repetitive form; leave it to the other methods

    facets = []
    for m in modes:
        sel = steep & (np.abs(((aspect - m + 180) % 360) - 180) <= ORIENT_ASSIGN_TOL_DEG)
        if sel.sum() < ORIENT_MIN_FACET_PTS:
            continue
        q = p[sel]
        tree = cKDTree(q[:, :2])
        pairs = tree.query_pairs(r=ORIENT_LINK_DIST_M, output_type="ndarray")
        if len(pairs) == 0:
            continue
        n_q = len(q)
        graph = _coo((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_q, n_q))
        ncomp, labels = _cc(graph, directed=False)
        for c in range(ncomp):
            comp = q[labels == c]
            if len(comp) < ORIENT_MIN_FACET_PTS:
                continue
            facet = _facet_from_points(comp, building_id, min_facet_area_m2)
            if facet is not None:
                facets.append(facet)
    return facets


def _facet_from_points(comp, building_id, min_facet_area_m2):
    """Concave-ish footprint + least-squares plane for one cluster."""
    from shapely.geometry import MultiPoint as _MP
    a, b, c0 = fit_plane_lstsq(comp)
    poly = _MP([(x, y) for x, y in comp[:, :2]]).buffer(0.6).buffer(-0.45)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.is_empty or poly.geom_type != "Polygon" or poly.area < min_facet_area_m2:
        return None
    slope_deg = np.degrees(np.arctan(np.hypot(a, b)))
    aspect_deg = (np.degrees(np.arctan2(-a, -b)) + 360) % 360
    return {
        "geometry": poly, "building_id": building_id,
        "plane_a": a, "plane_b": b, "plane_c": c0,
        "slope_deg": float(slope_deg), "aspect_deg": float(aspect_deg),
        "point_count": int(len(comp)),
    }


STRUCTURED_ASPECT_TOL_DEG = 25.0   # half-width of an aspect mode
STRUCTURED_MAX_MODES = 3           # sawtooth = 2; allow a little slack
STRUCTURED_MIN_AREA_SHARE = 0.7    # of total facet area inside those modes
STRUCTURED_MIN_SLOPE_DEG = 8.0     # near-flat facets have meaningless aspects


def _facets_are_structured(facets):
    """True when facet orientations collapse into a few tight modes, i.e.
    the segmentation found a repeating roof form (sawtooth, ribs, a run of
    identical dormers) rather than fragmenting on noise. Area-weighted so a
    dusting of slivers can't fake structure."""
    pitched = [f for f in facets if f.get("slope_deg", 0) >= STRUCTURED_MIN_SLOPE_DEG]
    total = sum(f["geometry"].area for f in pitched)
    if len(pitched) < 4 or total <= 0:
        return False
    # greedy area-weighted clustering of aspects on the circle
    remaining = sorted(pitched, key=lambda f: -f["geometry"].area)
    modes, covered = [], 0.0
    for f in remaining:
        a = f["aspect_deg"]
        if any(abs(((a - m + 180) % 360) - 180) <= STRUCTURED_ASPECT_TOL_DEG for m in modes):
            continue
        if len(modes) >= STRUCTURED_MAX_MODES:
            break
        modes.append(a)
    for f in pitched:
        if any(abs(((f["aspect_deg"] - m + 180) % 360) - 180) <= STRUCTURED_ASPECT_TOL_DEG for m in modes):
            covered += f["geometry"].area
    return (covered / total) >= STRUCTURED_MIN_AREA_SHARE


# ---------------------------------------------------------------------------
# Planarity repair
#
# Nothing in this pipeline ever checked that a returned "plane" is planar. It
# turns out that omission is the single largest source of bad layouts: a scan
# of the pilot area found 9% of facets with a plane-residual standard deviation
# over 1 m, and those facets carry 22% of all panels. A real roof plane sits
# near 0.1-0.2 m, which is just DSM noise.
#
# The failure mode is a stepped building. A staircase of flat levels has an
# overall downward trend, and a single tilted plane fits that trend with
# residuals RANSAC is willing to accept. 93 Beach St came out as ONE 2249 m2
# facet at 11.8 degrees spanning a 21 m height range with six distinct storeys
# in its height histogram -- on a building whose roof is flat. Panels were then
# packed across all of it, which is what Josh reported: "these are not part of
# the main flat roof plane which is likely the only place the panels should be
# on this roof".
#
# The trigger is the residual spread, but the SPLIT is on genuine height gaps.
# That distinction is what makes this safe: a real facet that is merely rough
# or noisy has no gap in its residual distribution and comes back untouched, so
# only facets with actual discontinuities are ever cut. A facet that cannot be
# repaired is returned as it was -- this pass may not cost a building its
# segmentation.
# The trigger is an INLIER FRACTION, not a standard deviation. sd was the first
# attempt and it is not robust: a perfectly good flat facet that happens to
# contain a few wall or vegetation returns 15 m away reads sd 0.7-1.3 while
# 85% of its points sit inside 10 cm of its plane. Measured on pilot, sd
# flagged 406 of 1,066 buildings, most of them fine. The fraction of points
# lying within a band of the plane cannot be moved by a handful of outliers:
# genuine roof facets measure 85-99%, and the ones that are actually a tilted
# sheet through a stepped building measure 22-36%.
PLANARITY_INLIER_BAND_M = 0.30      # "on the plane", generously -- DSM noise is ~0.1 m
PLANARITY_MIN_INLIER_FRACTION = 0.70
PLANARITY_BAND_GAP_M = 0.60     # a real step between roof levels. Below this it is
# noise or a genuine slope, and the facet is left alone. A panel is 1.7 m long, so a
# 0.6 m step is far past anything a mounting frame spans.
PLANARITY_BIN_M = 0.25          # residual histogram resolution
PLANARITY_MIN_MODE_FRACTION = 0.15   # a bump below this share of the tallest is noise
PLANARITY_VALLEY_FRACTION = 0.40     # the dip between two modes must be this much
# lower than the smaller of them, or they are one level rather than two
PLANARITY_MIN_PART_AREA_M2 = 8.0
PLANARITY_MIN_PART_POINTS = 12
# A raised roof LEVEL and a run of rooftop ducting both sit above the main
# plane, and banding on height alone cannot tell them apart -- the first
# version of this pass promoted 223 m2 of genuine ducting on #5370338 to a
# facet and put 10 panels on top of it. What separates them is that you cannot
# stand a row of panels on a duct: a storey is broad, ducting is narrow. So a
# repaired part has to survive being eroded by half a panel width. Aggregate
# area is no help here -- that ducting run totals 223 m2.
PLANARITY_MIN_HALF_WIDTH_M = 0.9
PLANARITY_MIN_CORE_FRACTION = 0.30
PLANARITY_MIN_CORE_AREA_M2 = 4.0
PLANARITY_MAX_PASSES = 2


def _facet_points(pc_source, geom):
    """Point-cloud points inside one facet footprint."""
    minx, miny, maxx, maxy = geom.bounds
    pts = pc_source.points_in_bbox(minx, miny, maxx, maxy)
    if pts is None or len(pts) == 0:
        return np.empty((0, 3))
    pts = np.asarray(pts)
    try:
        from shapely import contains_xy
        keep = contains_xy(geom, pts[:, 0], pts[:, 1])
    except Exception:
        from matplotlib.path import Path as _Path
        ext = np.asarray(geom.exterior.coords)
        keep = _Path(ext).contains_points(pts[:, :2])
    return pts[keep]


def _height_bands(resid, gap=PLANARITY_BAND_GAP_M):
    """Cut a height distribution at genuine valleys between dense modes.

    Two things had to be got right here, and both were found by measurement on
    93 Beach St rather than by reasoning.

    First, the axis is RAW HEIGHT, not distance from the facet's own plane. The
    trigger for this repair is that the plane is untrustworthy, so residuals
    measured against it are meaningless -- on 93 Beach the residual histogram
    is nearly flat, with valleys only 3-7% below their surrounding peaks, and
    no split is detectable. In raw height the same points show six storeys with
    valleys 0-8% of their peaks. That is the physically right axis anyway: a
    step between roof levels is a VERTICAL discontinuity.

    Second, an empty-gap test is not enough. A tall building returns points off
    its WALLS as well as its roofs, and those partly fill the space between
    storeys. What separates roof levels is density, not emptiness -- roofs are
    dense, walls are sparse.

    Returns [] when the distribution is unimodal, which is the common case and
    means the facet is left exactly as it was. A genuine pitched roof spreads
    its heights broadly and evenly, so it has no deep valley to cut at."""
    lo, hi = float(resid.min()), float(resid.max())
    if hi - lo < gap:
        return []
    nbins = max(8, int(np.ceil((hi - lo) / PLANARITY_BIN_M)))
    counts, edges = np.histogram(resid, bins=nbins, range=(lo, hi))
    # box-smooth so single-bin noise is not mistaken for a mode or a valley
    k = np.ones(3) / 3.0
    dens = np.convolve(counts.astype(float), k, mode="same")

    peaks = [i for i in range(1, len(dens) - 1)
             if dens[i] >= dens[i - 1] and dens[i] > dens[i + 1]]
    if dens[0] > dens[1]:
        peaks.insert(0, 0)
    if dens[-1] > dens[-2]:
        peaks.append(len(dens) - 1)
    peaks = [i for i in peaks if dens[i] >= PLANARITY_MIN_MODE_FRACTION * dens.max()]
    if len(peaks) < 2:
        return []

    centres = (edges[:-1] + edges[1:]) / 2.0
    cuts = []
    for a, b in zip(peaks, peaks[1:]):
        if centres[b] - centres[a] < gap:
            continue          # two bumps on one roof level, not two levels
        v = a + int(np.argmin(dens[a:b + 1]))
        if dens[v] <= PLANARITY_VALLEY_FRACTION * min(dens[a], dens[b]):
            cuts.append(centres[v])
    if not cuts:
        return []
    bounds = [-np.inf] + cuts + [np.inf]
    return [(resid > a) & (resid <= b) for a, b in zip(bounds, bounds[1:])]


def _repair_one_facet(f, pc_source, depth=0):
    """One facet in, one or more planar facets out. The original on failure."""
    geom = f["geometry"]
    if depth >= PLANARITY_MAX_PASSES or geom.is_empty:
        return [f]
    pts = _facet_points(pc_source, geom)
    if len(pts) < PLANARITY_MIN_PART_POINTS * 2:
        return [f]

    resid = pts[:, 2] - (f["plane_a"] * pts[:, 0] + f["plane_b"] * pts[:, 1] + f["plane_c"])
    inlier = float((np.abs(resid - np.median(resid)) < PLANARITY_INLIER_BAND_M).mean())
    if inlier >= PLANARITY_MIN_INLIER_FRACTION:
        return [f]   # the plane describes its points -- nothing to repair

    # Suspect the plane, so band on raw height -- see _height_bands.
    bands = _height_bands(pts[:, 2] - np.median(pts[:, 2]))
    if not bands:
        return [f]   # rough, but continuous -- not a stepped facet

    out = []
    for mask in bands:
        if int(mask.sum()) < PLANARITY_MIN_PART_POINTS:
            continue
        part = _facet_from_points(pts[mask], f["building_id"], PLANARITY_MIN_PART_AREA_M2)
        if part is None:
            continue
        # _facet_from_points dilates to close gaps between points; never let a
        # repaired part claim ground the parent facet did not hold.
        clipped = part["geometry"].intersection(geom)
        if clipped.geom_type == "MultiPolygon":
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if clipped.is_empty or clipped.geom_type != "Polygon" or clipped.area < PLANARITY_MIN_PART_AREA_M2:
            continue
        core = clipped.buffer(-PLANARITY_MIN_HALF_WIDTH_M)
        if (core.is_empty or core.area < PLANARITY_MIN_CORE_AREA_M2
                or core.area < PLANARITY_MIN_CORE_FRACTION * clipped.area):
            continue   # too narrow to be a roof level -- ducting, parapet, plant
        part["geometry"] = clipped
        part["area_m2"] = float(clipped.area)
        out.extend(_repair_one_facet(part, pc_source, depth + 1))

    return out or [f]


def repair_nonplanar_facets(facets, pc_source):
    """Split any facet whose plane does not actually describe its points."""
    repaired = []
    for f in facets:
        try:
            repaired.extend(_repair_one_facet(f, pc_source))
        except Exception:
            repaired.append(f)   # never cost a building its segmentation
    return repaired


# Applied to every segmenter's output or none: see merge_uneconomic_splits at
# the end of this file. Off by default until it is measured on a full area.
APPLY_REALISM_MERGE = True


# Balconies on a stepped apartment building are not roof.
#
# Josh has given the same instruction twice, on two different buildings. On 93
# Beach St: "these are not part of the main flat roof plane which is likely the
# only place the panels should be on this roof". On 7 Panorama Tce: "it's a
# roof with a clear flat plane, and then lots of stepped apartment balconies
# and you are placing panels on balconies incorrectly".
#
# 7 Panorama shows the signature plainly. Two large clean roofs -- 469 m2 and
# 263 m2, both 95% on-plane, at 363.3 m and 360.9 m -- and then ten small
# surfaces at 352.95-357.13 m, up to TEN METRES below, with inliers of 14-69%.
# Six of those ten sit within 7 cm of each other: one floor's balconies,
# repeated along the building.
#
# All three parts of that signature are needed. Height alone would throw away
# the genuinely stepped roof levels on 93 Beach that DO deserve panels. Poor
# planarity alone is just a bad facet. Small alone is a dormer. A surface that
# is far below a large clean roof AND does not fit a plane AND is small
# compared to that roof is a balcony: railings, furniture and partial occlusion
# are what make its points refuse to lie flat.
#
# Deliberately conservative -- it only engages when the building HAS an obvious
# main roof to be measured against, so an ordinary house or a genuine
# multi-level roof is never touched.
BALCONY_MAIN_MIN_INLIER = 0.85     # the main roof has to be convincingly planar
BALCONY_MAIN_MIN_AREA_M2 = 120.0   # ...and big enough to be the building's roof
BALCONY_MIN_DROP_M = 2.5           # a balcony sits this far below it, at least
BALCONY_MAX_INLIER = 0.75          # ...does not lie on a plane...
BALCONY_MAX_AREA_SHARE = 0.25      # ...and is small next to the main roof


# A rooftop plant deck is not a roof level either.
#
# The balcony filter below catches surfaces far BELOW the main roof. This is the
# same error the other way up, and it appeared the moment the partition started
# tiling the whole footprint: on the equipment reference building the LARGEST
# face is 178.7 m2 sitting 0.65 m above the roof, at 79% on-plane -- a duct
# platform modelled as roof, with panels laid on top of it. Obstruction
# detection cannot save it, because the deck really is flat; it is a perfectly
# good plane that happens not to be a roof.
#
# The discriminator is height, and it is not a close call. A real additional
# storey is 2.5 m or more. Rooftop equipment -- ducting, plant, condensers,
# lift overruns -- sits a few tens of centimetres to about a metre proud. There
# is almost nothing in between, so a surface raised into that band, and smaller
# than the roof it sits on, is equipment.
PLANT_MIN_RISE_M = 0.25       # below this it is just the main roof, with noise
PLANT_MAX_RISE_M = 1.60       # above this it is a genuine upper level, keep it
PLANT_MAX_AREA_SHARE = 1.00   # ...but it can out-cover any SINGLE face beneath it
# Height and area alone CANNOT tell a duct platform from a stepped roof level --
# the reference's plant deck is 178 m2 at +0.65 m and 5 Isle St's genuine upper
# roof is 191 m2 at +1.15 m. Shipping without this test cost 5 Isle two of the
# three faces Josh counted on it. What separates them is that a plant deck is
# CLUTTERED: ducting, condensers, rails and walkways leave its points refusing
# to lie flat (79% on-plane on the reference) while a real roof section is clean
# (95-99% on 5 Isle). Same reasoning as the balcony filter, where occlusion and
# railings do the same thing.
PLANT_MAX_INLIER = 0.88


def drop_plant_decks(facets, pc_source):
    """Remove raised equipment platforms modelled as roof faces."""
    if len(facets) < 2:
        return facets
    stats = []
    for f in facets:
        pts = _facet_points(pc_source, f["geometry"])
        if len(pts) < 12:
            stats.append((f, None, None))
            continue
        r = pts[:, 2] - (f["plane_a"] * pts[:, 0] + f["plane_b"] * pts[:, 1] + f["plane_c"])
        inl = float((np.abs(r - np.median(r)) < PLANARITY_INLIER_BAND_M).mean())
        stats.append((f, float(np.median(pts[:, 2])), inl))

    known = [(f, h) for f, h, _ in stats if h is not None]
    if len(known) < 2:
        return facets

    # The main roof LEVEL, not the biggest single face. Taking the largest face
    # picks the plant deck itself on exactly the roofs this is meant to fix: on
    # the equipment reference the duct platform is 178 m2 and the largest piece
    # of actual roof under it is 133 m2. Group faces into height bands and take
    # the band holding the most roof.
    bands = []   # [height, total_area, faces]
    for f, h in sorted(known, key=lambda t: t[1]):
        if bands and abs(h - bands[-1][0]) <= PLANT_MIN_RISE_M:
            bands[-1][1] += f["geometry"].area
            bands[-1][2].append(f)
        else:
            bands.append([h, f["geometry"].area, [f]])
    main_h, main_area, main_faces = max(bands, key=lambda b: b[1])
    main_ids = {id(f) for f in main_faces}

    kept = []
    for f, h, inl in stats:
        if (h is not None and id(f) not in main_ids
                and PLANT_MIN_RISE_M < (h - main_h) <= PLANT_MAX_RISE_M
                and f["geometry"].area < PLANT_MAX_AREA_SHARE * main_area
                and inl is not None and inl < PLANT_MAX_INLIER):
            continue
        kept.append(f)
    return kept or facets


def drop_balcony_levels(facets, pc_source):
    """Remove stepped balcony surfaces from a building that has a clear main roof."""
    if len(facets) < 2:
        return facets
    stats = []
    for f in facets:
        pts = _facet_points(pc_source, f["geometry"])
        if len(pts) < 12:
            stats.append((f, None, None))
            continue
        r = pts[:, 2] - (f["plane_a"] * pts[:, 0] + f["plane_b"] * pts[:, 1] + f["plane_c"])
        inl = float((np.abs(r - np.median(r)) < PLANARITY_INLIER_BAND_M).mean())
        stats.append((f, inl, float(np.median(pts[:, 2]))))

    mains = [(f, i, h) for f, i, h in stats
             if i is not None and i >= BALCONY_MAIN_MIN_INLIER
             and f["geometry"].area >= BALCONY_MAIN_MIN_AREA_M2]
    if not mains:
        return facets
    main = max(mains, key=lambda t: t[0]["geometry"].area)
    main_area, main_h = main[0]["geometry"].area, main[2]

    kept = []
    for f, inl, h in stats:
        if (inl is not None and h is not None
                and h < main_h - BALCONY_MIN_DROP_M
                and inl < BALCONY_MAX_INLIER
                and f["geometry"].area < BALCONY_MAX_AREA_SHARE * main_area):
            continue
        kept.append(f)
    return kept or facets


# Compact features -- recessed valleys, raised housings, lightwells -- are cut
# OUT of the face they sit in. See src/roof_features.py for why this needs to be
# a region rather than a line: a line through a face gives two half-planes, and
# no sequence of them ever encloses a shape in the middle of one. That is why
# four attempts at 7 Anderson Heights' central feature all failed, and why Josh
# kept seeing panels laid straight across it.
def drop_roof_features(facets, pc_source):
    """Remove compact non-roof regions from the faces containing them."""
    if not facets:
        return facets
    try:
        from src.roof_features import extract_features
    except Exception:
        return facets
    out = []
    for f in facets:
        pts = _facet_points(pc_source, f["geometry"])
        if len(pts) < 40:
            out.append(f)
            continue
        try:
            feats = extract_features(f["geometry"], pts,
                                     (f["plane_a"], f["plane_b"], f["plane_c"]))
        except Exception:
            feats = []
        if not feats:
            out.append(f)
            continue
        keep = f["geometry"].difference(unary_union([q for q, _ in feats]))
        if keep.is_empty:
            out.append(f)
            continue
        pieces = list(keep.geoms) if keep.geom_type == "MultiPolygon" else [keep]
        added = False
        for q in pieces:
            if q.geom_type != "Polygon" or q.area < 4.0:
                continue
            g = dict(f)
            g["geometry"] = Polygon(q.exterior, [r for r in q.interiors])
            g["area_m2"] = float(q.area)
            out.append(g)
            added = True
        if not added:
            out.append(f)
    return out


def _attach_building_geometry(facets, building_geom, pc_source=None, building_id=None):
    """Panel packing needs the building outline to align rows on flat roofs
    (a facet's own hull has no reliable orientation there). Attached once
    here so every caller inherits it without changing call sites.

    Also the single choke point every segment_building_best return passes
    through, which is where the realism merge belongs -- one place rather than
    five, so no strategy can quietly skip it."""
    if facets and pc_source is not None and building_id is not None:
        facets = _maybe_reconstruct(facets, pc_source, building_geom, building_id)
    if facets and pc_source is not None:
        facets = repair_nonplanar_facets(facets, pc_source)
        facets = drop_balcony_levels(facets, pc_source)
        facets = drop_plant_decks(facets, pc_source)
        facets = drop_roof_features(facets, pc_source)
    if APPLY_REALISM_MERGE and facets:
        try:
            facets = merge_uneconomic_splits(facets)
        except Exception as exc:
            # A bad merge must never cost a building its whole segmentation, but
            # it must not be invisible either -- a merge that always throws would
            # otherwise look exactly like a merge that never applies.
            _note_fallback("merge_uneconomic_splits", building_id, exc)
    for f in facets:
        f["building_geometry"] = building_geom
    return facets


FLAT_WINNER_MAX_SLOPE_DEG = 3.0      # "the area winner is a flat sheet" -- see the guard at
# the end of segment_building_best()
FLAT_WINNER_MIN_SLOPE_GAIN_DEG = 6.0  # 3.0 was tried: it fixed 55 Arrowtown-Lake
# Hayes Rd on paper (35%% obstruction -> 0%%) but by switching to a candidate covering
# 187 m2 LESS roof, and it put 45 panels onto raised structure -- trading a visible
# over-carve for an invisible under-detect. Keep the stricter bar.  # ...and the alternative has to find real pitch, not noise
FLAT_WINNER_MIN_AREA_SHARE = 0.45     # ...over a decent share of the same roof


# Reconstruction is the primary segmenter (Josh, 26 Aug: "just roll out the new
# 3D option and then we can improve on that rather than me providing feedback on
# the old way again"). It builds the roof as planes joined along their
# intersection lines and clipped to the surveyed outline, instead of tracing each
# facet independently -- so facets come out straight-edged, sharing exact
# borders, and covering more of the roof (median 96.9% vs 93.4% on a household
# sample).
#
# It scored level with the strategies below on PLANE COUNT against Josh's 20
# labelled roofs (24 total error each). That was the wrong basis to judge it on
# and is not why it is here: a count cannot see straight edges, shared ridges or
# coverage, which are the things panel placement actually sits on.
#
# Falls through to the best-of-five below whenever it returns nothing, which
# keeps the never-worse-than-before guarantee that path already provides for
# buildings with poor point-cloud coverage.
# OFF. Rolled out, measured on pilot, and reverted the same night.
#
# On the ten buildings it was developed against it resolved MORE roof than the
# existing segmenter (96.9% vs 93.4% median coverage) with straight edges. Run
# across all 1,066 pilot buildings it did the opposite: 71,852 panels -> 51,560
# (-28%), 20 buildings lost every panel, and 25 Brecon St went from 352 panels
# on a cleanly covered roof to 8, resolving 140 m2 of a 1,073 m2 outline.
#
# The facets it produces there carry 300+ vertices, so the arrangement is not
# yielding straight-edged cells at all on large commercial roofs -- it is
# unioning many tiny cells, most of which then fall under MIN_FACET_M2 and are
# dropped. That is a different failure from anything the ten-building set
# showed, which is the whole lesson: ten hand-picked roofs did not represent
# 1,066.
#
# Do not switch this back on without running a full area and checking panel
# COUNT, not just facet shape.
# Reconstruction is not better everywhere, so it is not chosen everywhere.
# Measured on the six roofs Josh called out for bad roof shape on 27 Aug, the
# area-weighted share of points lying within 30 cm of their own facet's plane:
#
#   5 Isle St        segmenter 24%  ->  reconstruct 82%   (he said "3 planes,
#                                                          one is not detected")
#   47 Stanley St    segmenter 59%  ->  reconstruct 97%   (mitre joints)
#   53 Hallenstein   segmenter 75%  ->  reconstruct 98%   ("fuzzy outlines")
#   2/8 Wakatipu     segmenter 58%  ->  reconstruct 49%
#   4 Pinnacle Pl    segmenter 88%  ->  reconstruct 75%
#   29 Park St       segmenter 80%  ->  reconstruct 91%   but 7 -> 17 facets
#
# Three large wins, two losses, and one that trades plane fidelity for
# fragmentation. A global flag has to pick one of those outcomes for every roof
# in the district; picking per building on the measured evidence does not.
#
# So: run the segmenter, measure it, and only reach for reconstruction when the
# segmenter is doing badly -- then keep whichever result actually describes the
# roof better. The facet-count guard is what stops it trading a bad plane for a
# shattered one, which is the failure Josh rejected this module for in the
# first place ("they need to be large and blocky most of the time").
USE_RECONSTRUCTION = False        # unconditional use -- still off, and should stay off
RECONSTRUCT_MIN_POINTS = 40
RECONSTRUCT_WHEN_INLIER_BELOW = 0.70   # only consider it for roofs the segmenter fits badly
RECONSTRUCT_MIN_INLIER_GAIN = 0.10     # ...and only switch on a clear win, not noise
RECONSTRUCT_MIN_USABLE_SHARE = 0.90    # ...that does not shatter the roof to get there.
# Counting facets was tried first and is the wrong guard: on 5 Isle St the
# segmenter returns ONE facet, so any reconstruction at all exceeds a ratio
# bound, including the 3 planes Josh says that roof actually has. What matters
# is not how many faces there are but whether the extra edges cost panel area,
# and that can be measured directly -- every facet is eroded by the ridge
# setback before packing, so total usable area IS the fragmentation cost.


def _area_weighted_inlier(facets, pc_source):
    """Share of a roof's points lying within 30 cm of their own facet's plane,
    weighted by facet area. The same measure the defect scanner ranks on."""
    if not facets:
        return 0.0
    tot = num = 0.0
    for f in facets:
        pts = _facet_points(pc_source, f["geometry"])
        if len(pts) < 12:
            continue
        r = pts[:, 2] - (f["plane_a"] * pts[:, 0] + f["plane_b"] * pts[:, 1] + f["plane_c"])
        inl = float((np.abs(r - np.median(r)) < PLANARITY_INLIER_BAND_M).mean())
        a = f["geometry"].area
        num += inl * a
        tot += a
    return num / tot if tot else 0.0


def _usable_area(facets):
    """Total area left after each facet is eroded by the ridge setback -- what
    panel packing actually gets to use. Fragmenting a roof shows up here as
    lost area, however many or few faces it ends up with."""
    tot = 0.0
    for f in facets:
        try:
            tot += max(0.0, f["geometry"].buffer(-config.RIDGE_SETBACK_M).area)
        except Exception:
            continue
    return tot


def _maybe_reconstruct(facets, pc_source, building_geom, building_id):
    """Swap in reconstruction only where it demonstrably describes the roof
    better. Returns the facets to use."""
    try:
        base = _area_weighted_inlier(facets, pc_source)
        if base >= RECONSTRUCT_WHEN_INLIER_BELOW:
            return facets
        alt = _reconstruct_facets(pc_source, building_geom, building_id)
        if not alt:
            return facets
        if _area_weighted_inlier(alt, pc_source) - base < RECONSTRUCT_MIN_INLIER_GAIN:
            return facets
        base_usable = _usable_area(facets)
        if base_usable > 0 and _usable_area(alt) < RECONSTRUCT_MIN_USABLE_SHARE * base_usable:
            return facets   # better planes, but paid for by shattering the roof
        return alt
    except Exception as exc:
        _note_fallback("repair_nonplanar", None, exc)
        return facets


# --- fallback visibility -------------------------------------------------
# Every silent `except Exception: return []` here is a place where the primary
# roof model can stop running while the build still produces plausible-looking
# output. That has already cost this project twice: a deleted helper turned
# imagery cuts into a no-op and two full comparison tables were run against the
# broken state before anyone noticed. Geometry from awkward roofs is a real and
# expected failure; a NameError is a bug. Distinguish them and never fall back
# in silence.
_BUG_EXCEPTIONS = (NameError, AttributeError, TypeError, ImportError,
                   UnboundLocalError, IndexError, KeyError, ZeroDivisionError)
FALLBACK_COUNTS = {}


def _note_fallback(where, building_id, exc):
    """Record a fallback and make it visible on stderr."""
    key = f"{where}:{type(exc).__name__}"
    FALLBACK_COUNTS[key] = FALLBACK_COUNTS.get(key, 0) + 1
    if isinstance(exc, _BUG_EXCEPTIONS):
        # A code defect, not a hard roof. Print the traceback the first few
        # times so it cannot hide behind a fallback that looks like a result.
        if FALLBACK_COUNTS[key] <= 3:
            import traceback
            print(f"[BUG] {where} building={building_id} {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
    elif FALLBACK_COUNTS[key] <= 3:
        print(f"[fallback] {where} building={building_id} "
              f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def _reconstruct_facets(pc_source, building_geom, building_id):
    """roof_reconstruct in the shape segment_building_best expects. Returns []
    on anything unexpected so the caller falls back rather than failing a
    whole area's build for one awkward roof."""
    try:
        import shapely.vectorized
        from src.roof_reconstruct import reconstruct
        minx, miny, maxx, maxy = building_geom.bounds
        pts = pc_source.points_in_bbox(minx - 1, miny - 1, maxx + 1, maxy + 1,
                                       building_only=True)
        if len(pts) < RECONSTRUCT_MIN_POINTS:
            return []
        inside = shapely.vectorized.contains(building_geom.buffer(0.3), pts[:, 0], pts[:, 1])
        pts = pts[inside]
        if len(pts) < RECONSTRUCT_MIN_POINTS:
            return []
        facets, _obstructions = reconstruct(building_id, building_geom.buffer(0), pts)
        # panel_fitting reads building_geometry to align panels to the building
        # outline on blobby or near-flat facets. reconstruct does not set it, and
        # without it every reconstructed roof would silently lose that alignment.
        for f in facets:
            f["building_geometry"] = building_geom
        # Obstructions are deliberately dropped here: obstruction_detection has
        # its own labelled validation set and its own both-directions test, and
        # swapping two things at once would make a regression unattributable.
        return facets
    except Exception as exc:
        _note_fallback("reconstruct", building_id, exc)
        return []


# The planar partition (src/roof_partition.py) is the primary roof model.
#
# Measured on 21 RANDOM pilot buildings, not hand-picked ones: coverage-adjusted
# share of points lying on their own facet's plane is better under the partition
# on 19 of them, the segmenter on 2. Worst facet outline: segmenter median 12
# vertices and max 2,594, partition median 9 and max 15. And on the thing that
# actually matters, layouts: 761 panels against 700, with panels not sitting on
# a plane down from 12.7% to 7.8% -- more panels AND fewer bad ones, which
# usually trade against each other.
#
# It also fixes the defect Josh has now reported five times. 1/5 Sydney St came
# out of a full rebuild with twelve facets carrying 593-1,035 vertices each:
# "pretty clearly still fuzzy and incorrect... This should be a cleanly defined
# roof, it is lots of straight planes and angles together." A partition cannot
# produce that shape -- every boundary is a surveyed footprint edge or a cut
# line, so vertex count is bounded by the number of cuts.
#
# The old segmenter stays as the fallback for anything the partition cannot
# model at all, which is a real case: too few points, or a footprint it cannot
# cut.
USE_PARTITION = True


def _partition_facets(pc_source, building_geom, building_id, imagery_ds=None):
    """roof_partition in the shape segment_building_best returns. [] on
    anything unexpected, so the caller falls back rather than failing a whole
    area's build for one awkward roof."""
    try:
        from src.roof_partition import partition_roof
        minx, miny, maxx, maxy = building_geom.bounds
        pts = pc_source.points_in_bbox(minx - 1, miny - 1, maxx + 1, maxy + 1,
                                       building_only=True)
        if len(pts) < RECONSTRUCT_MIN_POINTS:
            return []
        inside = shapely.vectorized.contains(building_geom, pts[:, 0], pts[:, 1])
        pts = pts[inside]
        if len(pts) < RECONSTRUCT_MIN_POINTS:
            return []
        return partition_roof(building_id, building_geom.buffer(0), pts, imagery_ds=imagery_ds)
    except Exception as exc:
        _note_fallback("partition", building_id, exc)
        return []


_IMAGERY_UNSET = object()


def segment_building_best(dsm_ds, pc_source, building_geom, building_id,
                           ransac_distance_threshold=None, min_facet_area_m2=None,
                           imagery_ds=_IMAGERY_UNSET):
    """Runs the point-cloud global solver, the (greedy) point-cloud-native
    segmentation, and the DSM-raster fallback, and keeps whichever explains
    more real roof area. Verified directly on a 400-building sample: the
    point-cloud path alone gives a large net improvement over DSM-only
    (coverage 77%->91%, less fragmented: 1064->897 total facets) but has a
    small residual failure rate (~4% notably worse, ~2.5% finding nothing
    at all -- some buildings just don't have clean point-cloud coverage,
    e.g. sitting on a source-tile edge, or a roof material/colour the
    point cloud's classifier doesn't tag as "building" reliably). Falling
    back to the DSM result per-building whenever it does better removes
    that entire residual category while keeping every genuine improvement.
    This is also the natural place a national rollout degrades gracefully
    -- LiDAR point-cloud coverage isn't universal across NZ yet, so any
    building outside point-cloud coverage (pc_source finds no points at
    all) automatically and silently uses the DSM path instead.

    The global solver (see its own module comment above) is a genuine
    algorithmic upgrade over the greedy point-cloud path specifically for
    real multi-plane roofs it under-segments -- confirmed directly on six
    reported buildings, each recovering substantially more real pitched
    roof area (e.g. one building: 0 resolved facets under greedy RANSAC
    beyond one dominant flat plane -> 14-16 facets covering most of the
    roof at real, varied slopes under the global solver). Both point-cloud
    paths are still computed and compared by area rather than trusting the
    global solver outright -- it's a heuristic global-*approximate* solver
    (ICM, not an exact optimum), not guaranteed to never do worse on any
    given roof, and this keeps the same never-worse-than-before guarantee
    the DSM fallback already provides."""
    # imagery_ds has no default on purpose. It used to default to None, and the
    # result was five separate tools -- anderson2, scan_defects,
    # validate_obstructions, build_heatmap, compare_layouts -- plus live_server
    # quietly analysing a weaker pipeline than the one being shipped, and
    # rendering pictures Josh was asked to judge. Passing None is still fine and
    # means "this area genuinely has no imagery"; forgetting to pass it is not.
    if imagery_ds is _IMAGERY_UNSET:
        raise TypeError(
            "segment_building_best: pass imagery_ds explicitly (None if the area "
            "has no imagery). Defaulting it silently changes which roof model runs.")

    if USE_PARTITION:
        facets_pt = _partition_facets(pc_source, building_geom, building_id, imagery_ds)
        if facets_pt:
            return _attach_building_geometry(facets_pt, building_geom, pc_source, building_id)
        # nothing partitioned -- fall through to the old strategies below

    if USE_RECONSTRUCTION:
        facets_rc = _reconstruct_facets(pc_source, building_geom, building_id)
        if facets_rc:
            return facets_rc

    facets_rg = segment_building_from_pointcloud_regiongrow(
        pc_source, building_geom, building_id, min_facet_area_m2=min_facet_area_m2,
    )
    facets_global = segment_building_from_pointcloud_global(
        pc_source, building_geom, building_id, min_facet_area_m2=min_facet_area_m2,
    )
    facets_pc = segment_building_from_pointcloud_native(
        pc_source, building_geom, building_id, ransac_distance_threshold=ransac_distance_threshold,
        min_facet_area_m2=min_facet_area_m2,
    )
    facets_dsm = segment_building(dsm_ds, building_geom, building_id, ransac_distance_threshold, min_facet_area_m2)
    facets_orient = segment_building_orientation_clustered(pc_source, building_geom, building_id,
                                                           min_facet_area_m2)
    area_rg = sum(f["area_m2"] for f in facets_rg)
    area_global = sum(f["area_m2"] for f in facets_global)
    area_pc = sum(f["area_m2"] for f in facets_pc)
    area_dsm = sum(f["area_m2"] for f in facets_dsm)

    # A correctly-split multi-plane roof almost always explains *slightly*
    # less total area than a wrongly-merged single flat compromise plane --
    # confirmed directly: real facets exclude a few scattered points that
    # don't confidently join any one clean plane, while one loose greedy
    # plane just claims everything within its tolerance. A strict "keep
    # whichever has more area" comparison would silently prefer the *wrong*,
    # under-segmented result almost every time the better methods' whole
    # reason for existing actually fires. So, in order of trust:
    #
    # 1. Region growing (segment_points_regiongrow) -- structurally immune
    #    to the cross-face compromise-plane failure every plane-competition
    #    method here shares (see its section comment; confirmed on a real
    #    hip roof no competition-based tuning could fix). Preferred whenever
    #    it found multi-plane structure and explains a comparable share of
    #    the roof.
    # 2. The ICM global solver, same rule -- still better than greedy on
    #    multi-plane roofs region growing declines (e.g. too few clean
    #    seeds on very sparse point coverage).
    # 3. Strict area-max across everything -- catches both methods' real
    #    failure modes (no usable point coverage at all, area near zero)
    #    and keeps the never-worse-than-DSM guarantee.
    best_alternative_area = max(area_global, area_pc, area_dsm)
    # Shatter guard: on curved or noisy roofs region growing can fragment into
    # dozens of tiny patches at roughly the same total area (measured: one real
    # building at 100 facets vs the global solver's 12, same 890m2) -- many
    # tiny facets with ridge setbacks between them is worse panel-layout output
    # than the alternative's coarser split, so fall back when RG's facet count
    # balloons (>3x the global solver's) AND its facets are much smaller on
    # average (<0.3x) -- both conditions, so a genuine fine-grained split of a
    # wrongly-merged roof (few facets -> several real ones) is never rejected.
    rg_shattered = (
        len(facets_rg) > 3 * max(1, len(facets_global))
        and len(facets_global) > 0
        and (area_rg / len(facets_rg)) < 0.3 * (area_global / len(facets_global))
        # ...unless the many facets are STRUCTURED rather than scattered. A
        # sawtooth/ribbed roof legitimately segments into dozens of small
        # facets whose aspects collapse into two tight modes; noise
        # fragmentation points every which way. Rejecting the structured case
        # cost us every sawtooth roof in town -- Turner St's teeth (point
        # normals cleanly bimodal at ~135/315 deg, median slope 52 deg) were
        # discarded in favour of 18 fragments with planes fitted ACROSS the
        # teeth at 7-25 deg. See _facets_are_structured.
        and not _facets_are_structured(facets_rg)
    )
    # Repetitive-form override: when a roof's orientations collapse into a
    # couple of tight modes AND the best alternative reports much shallower
    # facets, the alternative is averaging planes across folds (a sawtooth
    # read as a flat-ish sheet). Trust the orientation clustering there even
    # though it covers a little less area -- correct geometry beats coverage,
    # because everything downstream (aspect, yield, row direction) inherits it.
    if facets_orient and _facets_are_structured(facets_orient):
        area_orient = sum(f["geometry"].area for f in facets_orient)
        alt_best = max([(area_global, facets_global), (area_rg, facets_rg),
                        (area_pc, facets_pc), (area_dsm, facets_dsm)], key=lambda t: t[0])
        if alt_best[1]:
            med_alt = float(np.median([f["slope_deg"] for f in alt_best[1]]))
            med_or = float(np.median([f["slope_deg"] for f in facets_orient]))
            if med_or - med_alt >= 15.0 and area_orient >= 0.6 * alt_best[0]:
                return _attach_building_geometry(facets_orient, building_geom, pc_source, building_id)

    if (not rg_shattered) and len(facets_rg) > 1 and area_rg >= GLOBAL_AREA_TOLERANCE * best_alternative_area:
        return _attach_building_geometry(facets_rg, building_geom, pc_source, building_id)
    if len(facets_global) > 1 and area_global >= GLOBAL_AREA_TOLERANCE * max(area_pc, area_dsm):
        return _attach_building_geometry(facets_global, building_geom, pc_source, building_id)

    candidates = [facets_rg, facets_global, facets_pc, facets_dsm]
    areas = [area_rg, area_global, area_pc, area_dsm]
    winner = int(np.argmax(areas))

    # Last guard before area decides: a SINGLE near-flat facet covering the
    # whole footprint always wins on area, because one loose plane claims
    # everything within its tolerance while a correct split declines the points
    # that do not confidently belong. On a genuinely pitched or gabled roof
    # that winner is a horizontal sheet laid over the top of it, and everything
    # downstream inherits the error: the real gables then read as standing
    # ABOVE the plane, so obstruction detection carves them out.
    #
    # 17 Cardigan St, a multi-gabled house, is the case that found this --
    # 140 m2 of its 222 m2 roof was being carved away as "obstruction", and
    # every alternative had it right:
    #     region growing        13 facets  147 m2  slope median 14.2
    #     global multi-plane     3 facets  112 m2  slope median  9.2
    #     DSM ransac             3 facets  169 m2  slope median  1.4
    #     chosen (greedy pc)     1 facet   222 m2  slope median  0.5
    # Same on 2 Hawthorne Dr and 55 Arrowtown-Lake Hayes Rd.
    #
    # Deliberately narrow: it only fires when the winner is ONE facet and
    # essentially flat, and only for an alternative that finds real slope over
    # a decent share of the same roof. A genuinely flat commercial roof has no
    # such alternative, so it keeps its single facet.
    if len(candidates[winner]) == 1 and candidates[winner][0]["slope_deg"] < FLAT_WINNER_MAX_SLOPE_DEG:
        best_alt, best_alt_score = None, 0.0
        for fs, ar in zip(candidates, areas):
            if len(fs) < 2 or ar < FLAT_WINNER_MIN_AREA_SHARE * areas[winner]:
                continue
            med = float(np.median([f["slope_deg"] for f in fs]))
            if med - candidates[winner][0]["slope_deg"] < FLAT_WINNER_MIN_SLOPE_GAIN_DEG:
                continue
            if ar > best_alt_score:
                best_alt, best_alt_score = fs, ar
        if best_alt is not None:
            return _attach_building_geometry(best_alt, building_geom, pc_source, building_id)

    return _attach_building_geometry(candidates[winner], building_geom, pc_source, building_id)


# --- Global multi-plane solver (candidate pool + joint label assignment) ----
#
# Every plane-fitting approach above -- this file's own RANSAC, and the two
# abandoned attempts documented near segment_building_image_guided and
# _maybe_split_compromise_facet -- shares one structural weakness: each is
# *greedy*. RANSAC extracts one dominant plane, locks it in by removing its
# inliers, and never reconsiders; the (abandoned) image and local-normal
# approaches only ever look for a *split* of an already-greedily-formed
# facet, one boundary at a time. None of them jointly consider "what's the
# best explanation for ALL these points at once, across ALL candidate
# planes at once" -- confirmed directly to matter on a real building
# (#5371112, a clean L-shaped roof with an obvious ridge direction change):
# one dominant plane wins early during greedy extraction and nothing in a
# greedy method ever forces it to reconsider once the second wing's
# evidence is sitting right there.
#
# This instead: (1) generates an over-complete pool of candidate planes
# from many spatially-local samples (same anti-compromise-plane sampling
# as ransac_planes above, just not stopping at one winner), then (2) jointly
# assigns every point a plane label via Iterated Conditional Modes (ICM,
# Besag 1986) -- minimising a cost that combines "how well does this point
# fit its assigned plane" against "do this point's spatial neighbours
# mostly share its label" (a Potts smoothness prior), iterating until
# stable. A real ridge is exactly where the data term overrides the
# smoothness prior (two genuinely different planes fit their own sides far
# better than either fits the other side); noise/texture isn't, because it
# has no consistent spatial pattern for the smoothness prior to reinforce.
#
# Deliberately NOT full alpha-expansion graph-cut (the textbook globally-
# near-optimal MRF solver for exactly this kind of problem): its pairwise
# term construction needs auxiliary nodes handled exactly right, and a
# subtly wrong implementation fails silently -- it still produces a
# plausible-looking segmentation, just a wrong one, which is a worse
# outcome than not attempting it. ICM is weaker (can settle into a local
# optimum a global solver would escape) but every step is a direct,
# checkable "does this single point's label improve its own cost" -- much
# lower risk of an undetected correctness bug while still being a genuine
# joint/global method rather than a fourth variation on greedy extraction.

GLOBAL_CANDIDATE_SAMPLES = 600  # random local 3-point samples tried when building the candidate pool
GLOBAL_MAX_CANDIDATES = 8  # cap on distinct candidate planes kept after dedup. Was 14 -- found
# directly to be the dominant cause of "way too many facets, clearly wrong" reports on ordinary
# buildings: with a large enough candidate pool, DSM/point-cloud noise reliably produces several
# extra candidates that are each individually well-supported (many inlier points, so no other
# filter catches them) but represent the same real plane as another candidate already kept, just
# outside the dedup thresholds (MERGE_SLOPE_DIFF_DEG/MERGE_ASPECT_DIFF_DEG) -- ICM then correctly,
# faithfully assigns different noisy sub-patches of one real roof section to whichever of these
# near-duplicates fits each patch's own local noise best, fragmenting one real facet into several.
# Swept 4-14 directly against both known-over-segmented buildings (#5372567: 50->1 facets across
# that range) and the reference buildings the global solver exists to correctly split
# (#5371112's "roof that changes direction" case): 8 was the largest reduction in over-segmentation
# (150-building random sample: mean facets/building 5.01->4.30, buildings with >8 facets 29->18,
# total segmented area *unchanged* at +1.7%) that still left every reference multi-plane building's
# facet count exactly as before. Below 8 (tested 6, 4), reference buildings start losing real
# splits too, including collapsing #5372567 itself to a single facet -- too blunt a cut.
GLOBAL_NEIGHBOR_RADIUS_M = POINTCLOUD_CLUSTER_RADIUS_M  # smoothness-prior graph edges
GLOBAL_MAX_RESIDUAL_M = 1.0  # data-cost truncation -- caps how much a single point can "pull"
# the optimisation towards a plane that fits it badly, same spirit as a robust loss function
GLOBAL_SMOOTHNESS_WEIGHT = 0.08  # relative weight of "do my neighbours agree with me" against
# "how well do I personally fit my assigned plane" -- found by direct testing (swept 0.02-0.3
# against the reported failing buildings): too low and ICM barely differs from independently
# nearest-fitting each point (noisy, salt-and-pepper labels); too high and it over-smooths real
# ridges away, pulling the whole roof back towards one dominant label exactly like the greedy
# methods this is meant to fix
GLOBAL_ICM_MAX_ITERS = 12
GLOBAL_AREA_TOLERANCE = 0.75  # see segment_building_best -- how much of the best alternative's
# total area the global solver must still explain to be preferred over it outright whenever it
# also found genuine multi-plane structure (more than one facet)
GLOBAL_MIN_LABEL_POINTS = RANSAC_MIN_INLIERS


def _generate_candidate_planes(points, rng, n_samples=GLOBAL_CANDIDATE_SAMPLES,
                                distance_threshold=RANSAC_DISTANCE_THRESHOLD_M,
                                max_candidates=GLOBAL_MAX_CANDIDATES):
    """Over-complete pool of candidate (a, b, c) planes from many spatially-
    local 3-point samples (see ransac_planes' own comment for why local,
    not fully random, samples) -- scored by inlier count like RANSAC, but
    every sample tried is kept as a candidate rather than only the single
    best, then deduplicated. Returns a list of (a, b, c) tuples."""
    n = len(points)
    if n < 3:
        return []
    tree = cKDTree(points[:, :2])
    raw_candidates = []
    for _ in range(n_samples):
        anchor = rng.integers(n)
        neighbor_idx = tree.query_ball_point(points[anchor, :2], RANSAC_SAMPLE_RADIUS_M)
        if len(neighbor_idx) < 3:
            continue
        sample_idx = rng.choice(neighbor_idx, size=3, replace=False)
        sample = points[sample_idx]
        v1, v2 = sample[1] - sample[0], sample[2] - sample[0]
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal[:2]) > 1e6 or abs(normal[2]) < 1e-9:
            continue
        try:
            plane = fit_plane_lstsq(sample)
        except np.linalg.LinAlgError:
            continue
        inliers = plane_residuals(points, plane) < distance_threshold
        if inliers.sum() < RANSAC_MIN_INLIERS:
            continue
        refit = fit_plane_lstsq(points[inliers])
        raw_candidates.append((refit, int(inliers.sum())))

    raw_candidates.sort(key=lambda c: -c[1])
    kept = []
    for plane, inlier_count in raw_candidates:
        a, b, c = plane
        slope_deg, aspect_deg = slope_aspect_from_plane(a, b)
        if slope_deg > config.MAX_ROOF_SLOPE_DEG:
            continue
        is_dup = False
        for kplane, _, _ in kept:
            ka, kb, kc = kplane
            kslope, kaspect = slope_aspect_from_plane(ka, kb)
            slope_close = abs(slope_deg - kslope) <= MERGE_SLOPE_DIFF_DEG
            both_flat = slope_deg < MERGE_LOW_SLOPE_DEG and kslope < MERGE_LOW_SLOPE_DEG
            aspect_close = both_flat or _circular_diff(aspect_deg, kaspect) <= MERGE_ASPECT_DIFF_DEG
            if slope_close and aspect_close:
                is_dup = True
                break
        if not is_dup:
            kept.append((plane, slope_deg, aspect_deg))
        if len(kept) >= max_candidates:
            break

    return [p for p, _, _ in kept]


def _icm_assign_labels(points, candidate_planes, neighbor_radius=GLOBAL_NEIGHBOR_RADIUS_M,
                        max_residual=GLOBAL_MAX_RESIDUAL_M, smoothness_weight=GLOBAL_SMOOTHNESS_WEIGHT,
                        max_iters=GLOBAL_ICM_MAX_ITERS):
    """Jointly assigns every point a candidate-plane label via ICM (see
    module comment). Returns an int array of label indices into
    candidate_planes, one per point."""
    n = len(points)
    n_labels = len(candidate_planes)

    # data_cost[i, k] = how badly point i fits candidate plane k, truncated so one
    # badly-fit point can't dominate the optimisation for its whole neighbourhood
    data_cost = np.empty((n, n_labels))
    for k, plane in enumerate(candidate_planes):
        data_cost[:, k] = np.minimum(plane_residuals(points, plane), max_residual) ** 2

    labels = np.argmin(data_cost, axis=1)  # independent nearest-fit initialisation

    tree = cKDTree(points[:, :2])
    pairs = tree.query_pairs(neighbor_radius, output_type="ndarray")
    if len(pairs) == 0:
        return labels  # no spatial structure to smooth over -- nearest-fit is already the ICM answer

    # adjacency as a per-point neighbour list -- ICM updates one point at a time against
    # its neighbours' *current* labels, so a flat pair list is resolved into lists once
    neighbors = [[] for _ in range(n)]
    for i, j in pairs:
        neighbors[i].append(j)
        neighbors[j].append(i)

    order = np.arange(n)
    for _ in range(max_iters):
        changed = 0
        np.random.default_rng(0).shuffle(order)  # fixed shuffle seed -- deterministic given the
        # same candidate pool/points, so a building's result doesn't depend on unrelated RNG state
        for i in order:
            nbr = neighbors[i]
            if not nbr:
                new_label = np.argmin(data_cost[i])
            else:
                nbr_labels = labels[nbr]
                smooth_cost = np.array([
                    np.count_nonzero(nbr_labels != k) for k in range(n_labels)
                ]) * smoothness_weight
                new_label = np.argmin(data_cost[i] + smooth_cost)
            if new_label != labels[i]:
                labels[i] = new_label
                changed += 1
        if changed == 0:
            break

    return labels


GLOBAL_BUILDING_MARGIN_M = 0.5  # points are clipped to within this margin of the building's own
# footprint before candidate generation/ICM even run -- found directly on a real attached
# row-house unit (#5371112): the padded bbox query points_in_bbox callers use for point-cloud
# access (pad_m=2.0, generous on purpose so a facet's shape isn't starved of context right at its
# own edge) can pull in a *neighbouring* building's roof points on a row house, and unlike the old
# greedy RANSAC (whose spatially-local seed sampling rarely happens to land a whole hypothesis on
# next door's roof before running out of iterations), this global solver actively searches the
# entire input for good candidate planes and will happily find and assign points to a
# neighbour's plane if it fits acceptably -- confirmed directly: components built from those
# labels clip to fully empty against this building's own footprint, wasting real point evidence
# instead of ever contributing to this building's own segmentation
GLOBAL_REFIT_MAX_SLOPE_DRIFT_DEG = 15.0  # if refitting a component on just its own points swings
# the slope more than this from the candidate plane's, the refit isn't trustworthy -- confirmed
# directly: small or spatially elongated components (a thin sliver along a hip/valley) give
# fit_plane_lstsq an ill-conditioned, noise-dominated problem, producing wildly steep nonsense
# (45-58 degrees from candidates around 17-28) rather than a genuinely different, valid plane

GLOBAL_SATELLITE_MAX_AREA_M2 = 6.0  # a same-label spatial component smaller than this (or than
# GLOBAL_SATELLITE_MAX_AREA_FRACTION of its label's largest component) is folded into that largest
# component instead of becoming its own separate facet. Confirmed directly as the dominant real
# cause of the "clearly wrong, way too many facets" reports: ICM assigns per POINT, and a few
# points near the middle of one real plane flipping to a neighbouring label's better-fitting
# candidate (label noise, not a real physical break) splits what is genuinely one connected roof
# region into several disconnected same-label islands -- e.g. building #5372567 carried five
# separate facets all at aspect=325.4, slope~9.6 before this fix, none of them touching each other
# closely enough for merge_similar_facets (which only merges facets that are already spatially
# adjacent) to combine them, because they're the *same* label fragmented, not two different labels
# that independently converged to a similar plane. Absorbing them here (same label = same plane
# hypothesis by construction, so unioning is not a stretch) instead of gating merge_similar_facets
# more loosely keeps that function's own adjacency requirement meaningful for its real job:
# catching two genuinely different RANSAC passes that happened to land on the same plane.
GLOBAL_SATELLITE_MAX_AREA_FRACTION = 0.2


def segment_points_global(points, building_geom, building_id, min_facet_area_m2=None):
    """Global-solver equivalent of segment_points -- see the module comment
    above for why this exists and how it differs from the greedy RANSAC
    approach. Same interface and same downstream shape-fitting/filtering
    as segment_points, only the plane-assignment step itself differs."""
    min_facet_area_m2 = MIN_FACET_AREA_M2 if min_facet_area_m2 is None else min_facet_area_m2

    if len(points) > 0:
        footprint = building_geom.buffer(GLOBAL_BUILDING_MARGIN_M)
        inside = shapely.vectorized.contains(footprint, points[:, 0], points[:, 1])
        points = points[inside]
    if len(points) < RANSAC_MIN_INLIERS:
        return []

    rng = np.random.default_rng(building_id)
    candidates = _generate_candidate_planes(points, rng)
    if not candidates:
        return []

    labels = _icm_assign_labels(points, candidates)

    facets = []
    for label_id, plane in enumerate(candidates):
        member_idx = np.where(labels == label_id)[0]
        if len(member_idx) < GLOBAL_MIN_LABEL_POINTS:
            continue
        label_points = points[member_idx]

        label_facets = []
        for comp_local_idx in _cluster_points_spatially(label_points[:, :2], POINTCLOUD_CLUSTER_RADIUS_M):
            comp_points = label_points[comp_local_idx]
            if len(comp_points) < 3:
                continue
            # Refit on this component's own points -- the candidate plane was fit from its
            # original 3-point sample's local neighbourhood, but ICM may have grown or shrunk
            # its membership since, so a fresh fit is usually more accurate than the seed plane.
            # Not trusted blindly though: a small or spatially elongated component (a thin
            # sliver along a hip/valley) gives least-squares an ill-conditioned, noise-dominated
            # problem -- confirmed directly producing wildly steep nonsense (45-58 degrees from
            # candidates around 17-28) that the *candidate* plane never had. Refit slope drifting
            # far from the candidate's own is the tell; fall back to the candidate plane itself
            # rather than trust an unstable fit.
            candidate_slope_deg, _ = slope_aspect_from_plane(plane[0], plane[1])
            try:
                a, b, c = fit_plane_lstsq(comp_points)
                refit_slope_deg, _ = slope_aspect_from_plane(a, b)
                if abs(refit_slope_deg - candidate_slope_deg) > GLOBAL_REFIT_MAX_SLOPE_DRIFT_DEG:
                    a, b, c = plane
            except np.linalg.LinAlgError:
                a, b, c = plane
            slope_deg, aspect_deg = slope_aspect_from_plane(a, b)
            if slope_deg > config.MAX_ROOF_SLOPE_DEG:
                continue

            polygon = component_shape_from_points(comp_points[:, :2])
            if polygon is None:
                continue
            polygon = polygon.intersection(building_geom)
            if polygon.is_empty:
                continue
            if polygon.geom_type == "MultiPolygon":
                polygon = max(polygon.geoms, key=lambda p: p.area)
            elif polygon.geom_type not in ("Polygon",):
                continue
            if polygon.area < min_facet_area_m2:
                continue

            label_facets.append({
                "building_id": building_id,
                "plane_a": a, "plane_b": b, "plane_c": c,
                "slope_deg": slope_deg,
                "aspect_deg": aspect_deg,
                "area_m2": polygon.area,
                "point_count": len(comp_points),
                "geometry": polygon,
            })

        # ICM labels per point, not per region -- a few points near the middle of one real,
        # spatially-continuous plane flipping to a neighbouring label (noise, not a real physical
        # break) fragments that one label's own points into several disconnected spatial
        # components, each of which would otherwise become its own tiny separate facet. Since
        # they share this same label -- the same plane hypothesis, by construction -- absorb any
        # small fragment into this label's own largest component instead of keeping it as a
        # separate facet; merge_similar_facets (below, spatial-adjacency-gated) is a different,
        # narrower check for two independently-converged labels landing on the same plane.
        if len(label_facets) > 1:
            primary = max(label_facets, key=lambda f: f["area_m2"])
            absorb_threshold = max(GLOBAL_SATELLITE_MAX_AREA_M2,
                                    GLOBAL_SATELLITE_MAX_AREA_FRACTION * primary["area_m2"])
            keep, absorbed = [], []
            for f in label_facets:
                (keep if f is primary or f["area_m2"] >= absorb_threshold else absorbed).append(f)
            if absorbed:
                primary["geometry"] = unary_union([primary["geometry"]] + [f["geometry"] for f in absorbed])
                primary["area_m2"] = primary["geometry"].area
                primary["point_count"] += sum(f["point_count"] for f in absorbed)
            label_facets = keep

        facets.extend(label_facets)

    facets = _dedupe_overlaps(facets, min_facet_area_m2)
    return merge_similar_facets(facets)


def segment_building_from_pointcloud_global(pc_source, building_geom, building_id, pad_m=2.0,
                                             min_facet_area_m2=None):
    """segment_points_global, sourced directly from the raw point cloud --
    the global-solver counterpart to segment_building_from_pointcloud_native."""
    minx, miny, maxx, maxy = building_geom.bounds
    points = pc_source.points_in_bbox(minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m, building_only=True)
    return segment_points_global(points, building_geom, building_id, min_facet_area_m2)


# --- Region-growing segmentation (normal-field + contiguity) ------------------
#
# The global solver above fixed greedy RANSAC's under-segmentation, but it (and
# every plane-competition method in this file) shares one deeper structural
# flaw, confirmed directly on a real pyramid/hip roof (#4735241): candidate
# planes are INFINITE, and points are assigned by residual competition. On a
# roof whose faces converge at a central peak, a near-flat "compromise" plane
# at roughly mean roof height grazes a band across ALL four real faces at once
# and collects MORE within-tolerance points than any single true face --
# measured directly: 737 inliers at a tight 0.1m threshold vs 471-697 for each
# real face. No threshold, smoothness weight, or seeding scheme fixes that
# (each was tried and measured), because the flat plane genuinely does fit
# more points -- the failure is the competition model itself, not its tuning.
#
# Region growing eliminates that failure class by construction rather than by
# tuning: a facet is grown outward from a locally-planar seed through spatial
# adjacency, admitting a point only if BOTH its own local surface normal
# agrees with the region's plane AND it lies on that plane. A region
# physically cannot jump across a ridge to graze another face -- the ridge
# points' own local normals disagree with both sides and halt growth -- and
# there is no global residual competition anywhere. Each point ends up owned
# by exactly one region, so facet polygons come from disjoint point sets and
# the fitted-polygon-overlap problems downstream mostly vanish at the source.

RG_NORMAL_K = 12  # neighbours per local PCA normal -- enough averaging to be noise-robust on
# ~5-10 pts/m^2 LiDAR (neighbourhood radius ~0.6-0.9m), small enough not to straddle a whole
# narrow facet
RG_ADJ_K = 10  # growth adjacency: each point's k nearest plan-view neighbours
RG_SEED_MAX_RMS_M = 0.06  # only genuinely planar neighbourhoods may found a region (sqrt of the
# PCA smallest eigenvalue ~ RMS distance of the neighbourhood to its own best plane)
RG_ANGLE_TOL_DEG = 10.0  # max angle between a point's local normal and the region plane's normal.
# Was 20 -- confirmed directly on the reference shallow (~9.5 deg) pyramid roof that 20 admits
# EVERYTHING: two faces of a pyramid at slope theta and aspects 90 deg apart have normals only
# ~13.5 deg apart (19 deg for opposite faces) at theta=9.5, so a 20-degree tolerance never rejects
# a cross-face point and the incrementally-refitting region plane "creeps" across the apex to
# swallow the whole roof (final refit: 0.2 deg flat over 2798 points spanning all four faces).
# Same-face normal noise in angle terms is only ~2-4 deg at shallow slopes (aspect noise barely
# moves the normal when slope is small) and ~8-9 deg at steep slopes (where cross-face separation
# is huge, 45+ deg) -- so 10 separates the two populations across the whole slope range, with the
# one known soft spot being near-flat roofs (2-5 deg) where faces are indistinguishable anyway and
# merging them is harmless for POA.
RG_DIST_TOL_M = 0.20  # max point-to-region-plane distance during growth
RG_LEFTOVER_DIST_TOL_M = 0.35  # looser bound for the post-pass that folds ridge/edge points
# (whose own local normals were too noisy to pass growth) into an adjacent region
RG_REFIT_EVERY = 20  # incremental plane refit cadence during growth
RG_MIN_REGION_POINTS = 10  # below this a region is dissolved back into the leftover pool


def _pca_normals(points, k=RG_NORMAL_K):
    """Batched local surface normals: for every point, PCA over its k nearest
    plan-view neighbours. Returns (normals[n,3] unit, upward; rms[n] ~ RMS
    plane-fit residual of each neighbourhood, i.e. local planarity)."""
    n = len(points)
    k = min(k, n)
    tree = cKDTree(points[:, :2])
    _, nbr_idx = tree.query(points[:, :2], k=k)
    if k == 1:
        nbr_idx = nbr_idx[:, None]
    nbh = points[nbr_idx]  # (n, k, 3)
    centered = nbh - nbh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k
    evals, evecs = np.linalg.eigh(cov)  # ascending eigenvalues
    normals = evecs[:, :, 0]  # smallest-eigenvalue eigenvector = surface normal
    flip = normals[:, 2] < 0
    normals[flip] *= -1.0
    rms = np.sqrt(np.maximum(evals[:, 0], 0.0))
    return normals, rms


def _plane_from_accumulators(S):
    """Solve the normal equations for z = a*x + b*y + c from running sums.
    S = [Sxx, Sxy, Syy, Sx, Sy, Sxz, Syz, Sz, count]."""
    Sxx, Sxy, Syy, Sx, Sy, Sxz, Syz, Sz, cnt = S
    A = np.array([[Sxx, Sxy, Sx], [Sxy, Syy, Sy], [Sx, Sy, cnt]])
    b = np.array([Sxz, Syz, Sz])
    return np.linalg.solve(A, b)


def _grow_regions(points, normals, rms, angle_tol_deg=RG_ANGLE_TOL_DEG,
                   dist_tol=RG_DIST_TOL_M, seed_max_rms=RG_SEED_MAX_RMS_M,
                   min_region_points=RG_MIN_REGION_POINTS):
    """Core region growing. Returns (labels[n] int, planes list) -- label >= 0
    indexes into planes; -1/-2 = unclaimed (never grown / grown but region
    dissolved as too small)."""
    n = len(points)
    tree = cKDTree(points[:, :2])
    _, adj = tree.query(points[:, :2], k=min(RG_ADJ_K + 1, n))
    adj = adj[:, 1:]  # drop self

    cos_tol = np.cos(np.radians(angle_tol_deg))
    labels = np.full(n, -1, dtype=int)
    planes = []
    order = np.argsort(rms)  # most-planar seeds first

    from collections import deque
    for seed in order:
        if labels[seed] != -1 or rms[seed] > seed_max_rms:
            continue
        rid = len(planes)
        # All plane math for this region runs in seed-centered coordinates --
        # accumulating raw normal equations on ~1e6-magnitude NZTM coordinates
        # squares an already-large condition number (~2.5e11 measured on a real
        # building) and was confirmed to corrupt the incremental refit planes
        # badly enough that growth admitted cross-face points under a
        # numerically-drifted plane, silently re-creating the exact cross-face
        # creep this whole method exists to prevent.
        x0, y0 = points[seed, 0], points[seed, 1]
        # Bootstrap the region plane from the seed's own local normal:
        # normal (nx,ny,nz) with nz>0  =>  z = a*dx + b*dy + cc (dx,dy seed-relative)
        nx, ny, nz = normals[seed]
        if nz < 1e-6:
            continue  # near-vertical local surface -- not a roof plane seed
        a, b = -nx / nz, -ny / nz
        cc = points[seed, 2]  # at the seed, dx = dy = 0
        plane = np.array([a, b, cc])

        S = np.zeros(9)
        members = []
        frontier = deque([seed])
        labels[seed] = rid
        while frontier:
            p = frontier.popleft()
            members.append(p)
            dx, dy, z = points[p, 0] - x0, points[p, 1] - y0, points[p, 2]
            S += (dx * dx, dx * dy, dy * dy, dx, dy, dx * z, dy * z, z, 1.0)
            if len(members) >= 3 and len(members) % RG_REFIT_EVERY == 0:
                try:
                    plane = _plane_from_accumulators(S)
                except np.linalg.LinAlgError:
                    pass
            pa, pb, pcc = plane
            denom = np.sqrt(pa * pa + pb * pb + 1.0)
            plane_normal = np.array([-pa, -pb, 1.0]) / denom
            for q in adj[p]:
                if labels[q] != -1:
                    continue
                if normals[q] @ plane_normal < cos_tol:
                    continue
                if abs(pa * (points[q, 0] - x0) + pb * (points[q, 1] - y0) + pcc - points[q, 2]) > dist_tol:
                    continue
                labels[q] = rid
                frontier.append(q)

        if len(members) < min_region_points:
            labels[np.array(members)] = -2  # dissolved -- reclaimable by the leftover pass only
            continue
        try:
            plane = _plane_from_accumulators(S)
        except np.linalg.LinAlgError:
            pass
        # Convert back to global coordinates for downstream use:
        # z = a*dx + b*dy + cc  =>  c_global = cc - a*x0 - b*y0
        planes.append(np.array([plane[0], plane[1], plane[2] - plane[0] * x0 - plane[1] * y0]))

    # Compact label ids (dissolved regions left gaps in numbering only if a
    # dissolved region ever got an id -- it didn't: ids are assigned by
    # len(planes) and planes only appended for kept regions. But growth wrote
    # rid into labels before the size check, so remap those: any label >=
    # len(planes) means its region was dissolved after a later region already
    # appended -- impossible by construction (rid == len(planes) at claim
    # time, and a dissolved region appends nothing, so the NEXT region reuses
    # the same rid). Guard against that reuse: dissolved members were reset
    # to -2 above, so no stale rid survives.
    #
    # Leftover pass: ridge/edge/noisy points (labels < 0) join an ADJACENT
    # region whose plane fits them -- local, contiguity-bounded assignment
    # only, never a global competition. Two hops max.
    for _ in range(2):
        unclaimed = np.where(labels < 0)[0]
        if len(unclaimed) == 0:
            break
        changed = False
        new_labels = labels.copy()
        for p in unclaimed:
            best_rid, best_err = -1, RG_LEFTOVER_DIST_TOL_M
            for q in adj[p]:
                rid = labels[q]
                if rid < 0:
                    continue
                pa, pb, pc = planes[rid]
                err = abs(pa * points[p, 0] + pb * points[p, 1] + pc - points[p, 2])
                if err < best_err:
                    best_err, best_rid = err, rid
            if best_rid >= 0:
                new_labels[p] = best_rid
                changed = True
        labels = new_labels
        if not changed:
            break

    return labels, planes


def segment_points_regiongrow(points, building_geom, building_id, min_facet_area_m2=None):
    """Region-growing counterpart to segment_points_global -- same interface,
    same downstream shape fitting/merging, different (contiguity-based)
    plane-assignment core. See the section comment above for why."""
    min_facet_area_m2 = MIN_FACET_AREA_M2 if min_facet_area_m2 is None else min_facet_area_m2

    if len(points) > 0:
        footprint = building_geom.buffer(GLOBAL_BUILDING_MARGIN_M)
        inside = shapely.vectorized.contains(footprint, points[:, 0], points[:, 1])
        points = points[inside]
    if len(points) < RG_MIN_REGION_POINTS:
        return []

    normals, rms = _pca_normals(points)
    labels, planes = _grow_regions(points, normals, rms)

    facets = []
    for rid, plane in enumerate(planes):
        member_idx = np.where(labels == rid)[0]
        if len(member_idx) < RG_MIN_REGION_POINTS:
            continue
        member_points = points[member_idx]
        # Final refit on the region's full membership (leftover pass may have
        # added ridge/edge points since the last incremental refit).
        try:
            a, b, c = fit_plane_lstsq_centered(member_points)
        except np.linalg.LinAlgError:
            a, b, c = plane
        slope_deg, aspect_deg = slope_aspect_from_plane(a, b)
        if slope_deg > config.MAX_ROOF_SLOPE_DEG:
            continue

        # A region is spatially contiguous by construction, but the polygon
        # step still clusters defensively: the leftover pass can very rarely
        # attach a satellite clump via a chain of mutual-kNN links thinner
        # than the clustering radius.
        for comp_local_idx in _cluster_points_spatially(member_points[:, :2], POINTCLOUD_CLUSTER_RADIUS_M):
            comp_points = member_points[comp_local_idx]
            if len(comp_points) < RG_MIN_REGION_POINTS:
                continue
            polygon = component_shape_from_points(comp_points[:, :2])
            if polygon is None:
                continue
            polygon = polygon.intersection(building_geom)
            if polygon.is_empty:
                continue
            if polygon.geom_type == "MultiPolygon":
                polygon = max(polygon.geoms, key=lambda p: p.area)
            elif polygon.geom_type not in ("Polygon",):
                continue
            if polygon.area < min_facet_area_m2:
                continue
            facets.append({
                "building_id": building_id,
                "plane_a": a, "plane_b": b, "plane_c": c,
                "slope_deg": slope_deg,
                "aspect_deg": aspect_deg,
                "area_m2": polygon.area,
                "point_count": len(comp_points),
                "geometry": polygon,
            })

    facets = _dedupe_overlaps(facets, min_facet_area_m2)
    return merge_similar_facets(facets)


def segment_building_from_pointcloud_regiongrow(pc_source, building_geom, building_id, pad_m=2.0,
                                                 min_facet_area_m2=None):
    """segment_points_regiongrow, sourced directly from the raw point cloud."""
    minx, miny, maxx, maxy = building_geom.bounds
    points = pc_source.points_in_bbox(minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m, building_only=True)
    return segment_points_regiongrow(points, building_geom, building_id, min_facet_area_m2)


# --- Realism pass: roofs are few large faces, not many slivers -------------
#
# Josh, 26 Aug, after judging ten before/after layouts: "They need to be large
# and blocky most of the time like real rooftops. It's a lot more common for
# rooftops to be clear large flat surfaces on a few different angles and slopes,
# than it is to have lots of small changes."
#
# This is not only about looking realistic. panel_fitting erodes every facet by
# RIDGE_SETBACK_M and panels cannot span two facets, so every split costs usable
# area, and the smaller the piece the worse the rate:
#
#      6 m2 face ->  57% of it usable      150 m2 face -> 90%
#     25 m2 face ->  77%                   400 m2 face -> 94%
#
# Measured on the live pilot: 36% of facets are under 15 m2, and the way roofs
# are currently split costs 11.0% of all roof area to erosion, against 6.0% if
# each building were a single face. So there is about 5.6% of usable roof to be
# recovered by splitting less.
#
# The criterion is therefore derived rather than tuned. Merging two adjacent
# faces GAINS the erosion area their shared boundary was costing, and LOSES
# yield on the smaller face because it is now modelled at the wrong angle. Merge
# when the gain exceeds the loss. The only modelled assumption is that yield
# falls with the cosine of the angle between the two plane normals, which is a
# first-order approximation of the projected-irradiance loss.
SLIVER_CONSIDER_MAX_M2 = 40.0   # above this a face is worth keeping on its own
SLIVER_MIN_SHARED_M = 0.5       # must actually adjoin, not just touch at a corner
# A HARD cap, checked before the area trade-off and not tradeable against it.
#
# The first version had only the cosine yield term, and it merged across real
# ridges: measured over 120 buildings, the median accepted merge was 19.6 deg
# and half were 20 deg or more. Josh saw the result immediately -- panels
# "going over ridge or roof edges ... edge of a roof section".
#
# The error was modelling a ridge crossing as lost YIELD. It is not. A panel is
# rigid: across a join of angle t it lifts about (panel_length / 2) * tan(t) off
# the roof -- 30 cm at 20 deg on a 1.7 m panel. That panel cannot be installed
# at all, so no amount of recovered setback area buys it. Only joins shallow
# enough for a panel to actually lie across may be merged.
SLIVER_MAX_MERGE_ANGLE_DEG = 4.0   # ~6 cm lift on a 1.7 m panel
# A STEP is invisible to the angle test and must be checked separately.
#
# _plane_angle_deg compares plane NORMALS, so two parallel roof sections at
# different heights read as 0 degrees apart and sail through the cap above.
# Merging them puts panels across a vertical step -- Josh, on 6 Shotover St:
# "clearly overlapping roof ridges". The reconstruction module had this test
# and this one did not; the omission is mine.
#
# Measured at the shared boundary, because two planes that genuinely fold
# together are identical there and only diverge away from it.
SLIVER_MAX_MERGE_STEP_M = 0.15


def _usable_after_setback(area_m2, setback_m):
    """Area left after eroding a face by the ridge setback, approximating the
    face as a square. Exact for a square, close enough for the trade-off, and
    it does not depend on the polygon being well behaved."""
    side = math.sqrt(max(area_m2, 0.0))
    return max(side - 2.0 * setback_m, 0.0) ** 2


def _plane_angle_deg(f, g):
    """Angle between two facets' plane normals."""
    n1 = np.array([-f["plane_a"], -f["plane_b"], 1.0])
    n2 = np.array([-g["plane_a"], -g["plane_b"], 1.0])
    c = float(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def merge_uneconomic_splits(facets, setback_m=None):
    """Fuse adjacent faces whose split costs more usable area than the yield it
    buys. Repeats until nothing else qualifies, smallest face first."""
    if setback_m is None:
        setback_m = getattr(config, "RIDGE_SETBACK_M", 0.25)
    if len(facets) < 2:
        return facets
    facets = list(facets)
    changed = True
    while changed and len(facets) > 1:
        changed = False
        order = sorted(range(len(facets)), key=lambda i: facets[i]["geometry"].area)
        for i in order:
            f = facets[i]
            if f["geometry"].area > SLIVER_CONSIDER_MAX_M2:
                continue
            best, best_shared = None, SLIVER_MIN_SHARED_M
            for j in range(len(facets)):
                if i == j:
                    continue
                shared = f["geometry"].buffer(0.05).intersection(
                    facets[j]["geometry"].boundary).length
                if shared > best_shared:
                    best, best_shared = j, shared
            if best is None:
                continue
            g = facets[best]
            merged = unary_union([f["geometry"], g["geometry"]]).buffer(0.02).buffer(-0.02)
            if merged.geom_type != "Polygon" or merged.is_empty:
                continue
            gain = (_usable_after_setback(merged.area, setback_m)
                    - _usable_after_setback(f["geometry"].area, setback_m)
                    - _usable_after_setback(g["geometry"].area, setback_m))
            theta = _plane_angle_deg(f, g)
            if theta > SLIVER_MAX_MERGE_ANGLE_DEG:
                continue   # a real ridge: a panel cannot lie across it at any price
            shared = f["geometry"].buffer(0.05).intersection(g["geometry"].boundary)
            if shared.is_empty:
                continue
            c = shared.centroid
            zf = f["plane_a"] * c.x + f["plane_b"] * c.y + f["plane_c"]
            zg = g["plane_a"] * c.x + g["plane_b"] * c.y + g["plane_c"]
            if abs(zf - zg) > SLIVER_MAX_MERGE_STEP_M:
                continue   # parallel but stepped: a panel cannot bridge it either
            # The smaller face is the one that ends up mis-oriented.
            small = min(f, g, key=lambda x: x["geometry"].area)
            loss = small["geometry"].area * (1.0 - math.cos(math.radians(theta)))
            if gain <= loss:
                continue
            # Keep the larger face's plane -- it carries more of the roof.
            keep = g if g["geometry"].area >= f["geometry"].area else f
            merged_facet = dict(keep)
            merged_facet["geometry"] = merged
            if "area_m2" in merged_facet:
                merged_facet["area_m2"] = float(merged.area)
            facets = [facets[k] for k in range(len(facets)) if k not in (i, best)]
            facets.append(merged_facet)
            changed = True
            break
    return facets

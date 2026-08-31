"""Skeleton-roof reconstruction: build the roof UP from the footprint.

Josh's #5119630 (Island Bay) is the type case. Its top-surface LiDAR shows a
single connected hip-and-valley form: every face climbs inward from its eave
at a common ~23 degree pitch, and the ridge network is nothing more than
where those inclined planes meet. Point-cluster methods keep drawing organic
blobs over it and cut-based partitions keep slicing across it, because both
try to DISCOVER boundaries that are in fact CONSTRUCTIBLE: for a roof of this
family, footprint + per-edge eave height + pitch determines the whole 3D
shape.

So construct it. For each (merged, non-trivial) footprint edge, incline a
plane from the edge's measured eave height at the fitted pitch. The roof
surface is the LOWER ENVELOPE of those surfaces -- using distance to the edge
SEGMENT (not its infinite line), which adds a cone at each corner and makes
the envelope behave on non-convex outlines. Every plan point is labelled by
the edge whose surface is lowest there; a facet is a label region, its plane
refit from the actual points it contains. Hips, ridges and valleys land on
the label boundaries by construction.

The envelope is verified, never trusted: the caller compares the result
against other strategies on points-explained. Flat-topped sections, gables
and dormers make the envelope wrong locally -- the per-region refit absorbs
small errors, and a roof that is not this family at all simply scores badly
and loses.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

warnings.filterwarnings("ignore")
# ...but never deprecations. A blanket ignore is exactly how 68 calls to
# shapely.vectorized -- an API documented for REMOVAL, under an unpinned
# shapely>=2.0 -- stayed invisible until 31 Aug. Third-party noise stays
# suppressed; a countdown to the pipeline breaking does not.
warnings.filterwarnings("default", category=DeprecationWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.roof_partition import (_fit_plane_robust, _inlier_fraction, _points_in,
                                _slope_aspect, MIN_FACET_M2, MIN_POINTS,
                                STEEP_FACE_DEG, STEEP_FACE_MIN_FIT)

GRID_RES_M = 0.25
EDGE_MIN_M = 1.0                # shorter footprint edges are survey jitter
COLLINEAR_TOL_DEG = 4.0
EAVE_BAND_IN_M = 1.5            # how far inside an edge its eave height is read
EAVE_PCTL = 15                  # low percentile of the band = eave height
EAVE_MIN_PTS = 6
PITCH_CANDIDATES = np.arange(8.0, 42.1, 2.0)
PITCH_REFINE_STEP = 0.5
ENVELOPE_BAND_M = 0.20          # a point this close to the envelope is explained
SIMPLIFY_M = 0.35               # label-region boundary simplification


def _merged_edges(poly):
    """Footprint edges, consecutive collinear runs merged, tiny ones dropped.
    Returns [(p0, p1)] as np arrays."""
    ring = np.asarray(poly.exterior.coords)      # closed
    segs = []
    for i in range(len(ring) - 1):
        segs.append([ring[i], ring[i + 1]])
    # merge collinear consecutive
    merged = []
    for seg in segs:
        if merged:
            a0, a1 = merged[-1]
            v1 = a1 - a0
            v2 = seg[1] - seg[0]
            ang = abs(np.degrees(np.arctan2(v1[0]*v2[1]-v1[1]*v2[0], v1 @ v2)))
            if ang < COLLINEAR_TOL_DEG or ang > 180 - COLLINEAR_TOL_DEG:
                merged[-1] = [a0, np.asarray(seg[1])]
                continue
        merged.append([np.asarray(seg[0]), np.asarray(seg[1])])
    # wrap-around merge (first and last may be one wall split at ring start)
    if len(merged) > 1:
        a0, a1 = merged[-1]
        b0, b1 = merged[0]
        v1, v2 = a1 - a0, b1 - b0
        ang = abs(np.degrees(np.arctan2(v1[0]*v2[1]-v1[1]*v2[0], v1 @ v2)))
        if ang < COLLINEAR_TOL_DEG:
            merged[0] = [a0, b1]
            merged.pop()
    return [(p0, p1) for p0, p1 in merged if np.hypot(*(p1 - p0)) >= EDGE_MIN_M]


def _seg_distance(px, py, p0, p1):
    """Vectorised distance from points to a segment."""
    d = p1 - p0
    L2 = float(d @ d)
    t = np.clip(((px - p0[0]) * d[0] + (py - p0[1]) * d[1]) / max(L2, 1e-9), 0.0, 1.0)
    qx, qy = p0[0] + t * d[0], p0[1] + t * d[1]
    return np.hypot(px - qx, py - qy)


def _eave_heights(edges, pts, footprint):
    """Eave z per edge: a low percentile of the top-surface points in a band
    just inside the edge. Edges with no evidence inherit the building's
    lowest measured eave (better than inventing one)."""
    out = []
    for p0, p1 in edges:
        d = _seg_distance(pts[:, 0], pts[:, 1], p0, p1)
        band = pts[d < EAVE_BAND_IN_M]
        out.append(float(np.percentile(band[:, 2], EAVE_PCTL)) if len(band) >= EAVE_MIN_PTS else None)
    known = [z for z in out if z is not None]
    if not known:
        return None
    fallback = min(known)
    return [z if z is not None else fallback for z in out]


def _envelope(edges, eaves, pitch_deg, px, py):
    """Lower-envelope height and argmin edge label at plan points."""
    t = np.tan(np.radians(pitch_deg))
    best_z = np.full(len(px), np.inf)
    label = np.full(len(px), -1, dtype=np.int32)
    for i, ((p0, p1), ez) in enumerate(zip(edges, eaves)):
        z = ez + t * _seg_distance(px, py, p0, p1)
        m = z < best_z
        best_z[m] = z[m]
        label[m] = i
    return best_z, label


def _fit_pitch(edges, eaves, pts):
    """The single pitch whose envelope explains the most points, coarse scan
    then a local refine. Returns (pitch, explained_fraction_at_pitch)."""
    px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]
    def score(pitch):
        z, _ = _envelope(edges, eaves, pitch, px, py)
        return float((np.abs(pz - z) < ENVELOPE_BAND_M).mean())
    best = max(PITCH_CANDIDATES, key=score)
    lo, hi = best - 2.0, best + 2.0
    fine = np.arange(lo, hi + 1e-9, PITCH_REFINE_STEP)
    best = max(fine, key=score)
    return float(best), score(best)


def _split_levels(pts, min_gap_m=0.8, min_share=0.12):
    """Split points into height LEVELS at the widest empty band in the z
    histogram. A two-storey house with a lower wrap-around section puts its
    two roof systems in cleanly separated z bands; each gets its own skeleton.
    Returns a list of point arrays, LOWEST level first, or [pts] if unimodal."""
    z = np.sort(pts[:, 2])
    if len(z) < 40:
        return [pts]
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    if hi - lo < min_gap_m * 2:
        return [pts]
    bins = np.arange(lo, hi + 0.301, 0.3)
    hist, edges_ = np.histogram(z, bins=bins)
    # widest run of empty bins with enough mass on both sides
    best = None
    i = 0
    while i < len(hist):
        if hist[i] == 0:
            j = i
            while j < len(hist) and hist[j] == 0:
                j += 1
            below = hist[:i].sum()
            above = hist[j:].sum()
            width = (j - i) * 0.3
            if (width >= min_gap_m and below >= min_share * len(pts)
                    and above >= min_share * len(pts)):
                if best is None or width > best[0]:
                    best = (width, (edges_[i] + edges_[j]) / 2)
            i = j
        else:
            i += 1
    if best is None:
        return [pts]
    cut = best[1]
    return [pts[pts[:, 2] <= cut], pts[pts[:, 2] > cut]]


def _upper_outline(upper_pts, footprint):
    """Plan outline of an upper storey from its own points, clipped to the
    building. Simplified hard -- an upper storey is walls, not fuzz."""
    from src.roof_segmentation import component_shape_from_points
    poly = component_shape_from_points(upper_pts[:, :2])
    if poly is None:
        return None
    poly = poly.buffer(0.15).simplify(0.5).buffer(0)
    poly = poly.intersection(footprint)
    if poly.is_empty:
        return None
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)
    if poly.geom_type != "Polygon" or poly.area < MIN_FACET_M2:
        return None
    return poly


def _skeleton_one_level(building_id, outline, region, pts, min_envelope_fit):
    """Skeleton-fit ONE level: planes rise from `outline`'s edges, facets are
    clipped to `region` (the plan area this level's roof actually occupies --
    for an upper storey they coincide; for a lower skirt the region is the
    ring around the upper storey)."""
    if len(pts) < MIN_POINTS * 2:
        return []
    edges = _merged_edges(outline)
    if len(edges) < 3:
        return []
    eaves = _eave_heights(edges, pts, outline)
    if eaves is None:
        return []
    pitch, fit = _fit_pitch(edges, eaves, pts)
    if fit < min_envelope_fit:
        return []
    return _facets_from_envelope(building_id, edges, eaves, pitch, region, pts)


def skeleton_roof(building_id, footprint, pts, min_envelope_fit=0.55):
    """Facets for a (possibly multi-level) common-pitch skeleton roof, or []
    when the building is not one. Levels are split on the z histogram; the
    upper storey gets its own outline (from its own points) and its own
    skeleton; the lower level keeps the ring around it."""
    levels = _split_levels(pts)
    if len(levels) == 2:
        lower_pts, upper_pts = levels
        upper = _upper_outline(upper_pts, footprint)
        if upper is not None:
            out = _skeleton_one_level(building_id, upper, upper, upper_pts,
                                      min_envelope_fit)
            ring = footprint.difference(upper.buffer(0.1))
            if not ring.is_empty and ring.area >= MIN_FACET_M2:
                out += _skeleton_one_level(building_id, footprint, ring,
                                           lower_pts, min_envelope_fit)
            return out
    return _skeleton_one_level(building_id, footprint, footprint, pts,
                               min_envelope_fit)


def _facets_from_envelope(building_id, edges, eaves, pitch, region, pts):

    # Label a plan grid by the envelope, vectorise the label regions.
    minx, miny, maxx, maxy = region.bounds
    xs = np.arange(minx, maxx + GRID_RES_M, GRID_RES_M)
    ys = np.arange(miny, maxy + GRID_RES_M, GRID_RES_M)
    gx, gy = np.meshgrid(xs, ys)
    flat_x, flat_y = gx.ravel(), gy.ravel()
    inside = shapely.contains_xy(region, flat_x, flat_y)
    _, lab = _envelope(edges, eaves, pitch, flat_x, flat_y)
    lab[~inside] = -1
    lab_grid = lab.reshape(gx.shape)

    from rasterio import features as rfeatures
    from rasterio.transform import from_origin
    transform = from_origin(minx - GRID_RES_M / 2, ys[-1] + GRID_RES_M / 2,
                            GRID_RES_M, GRID_RES_M)
    out = []
    for rid in range(len(edges)):
        mask = (np.flipud(lab_grid) == rid)
        if not mask.any():
            continue
        polys = []
        for geomdict, val in rfeatures.shapes(mask.astype(np.uint8), mask=mask,
                                              transform=transform):
            p = Polygon(geomdict["coordinates"][0],
                        geomdict["coordinates"][1:])
            if p.is_valid and p.area >= MIN_FACET_M2 * 0.5:
                polys.append(p)
        if not polys:
            continue
        merged = unary_union(polys).intersection(region)
        geoms = merged.geoms if isinstance(merged, MultiPolygon) else [merged]
        for poly in geoms:
            if not isinstance(poly, Polygon) or poly.area < MIN_FACET_M2:
                continue
            poly = poly.simplify(SIMPLIFY_M).buffer(0)
            if poly.is_empty or poly.geom_type != "Polygon" or poly.area < MIN_FACET_M2:
                continue
            sub = _points_in(poly, pts)
            if len(sub) >= MIN_POINTS:
                plane = _fit_plane_robust(sub)
            else:
                # construct the edge's own plane at the fitted pitch
                p0, p1 = edges[rid]
                d = (p1 - p0) / np.hypot(*(p1 - p0))
                n = np.array([-d[1], d[0]])
                # inward normal: point slightly inside must be positive
                mid = (p0 + p1) / 2
                if not region.buffer(0.5).contains(shapely.geometry.Point(*(mid + n * 0.3))):
                    n = -n
                t = np.tan(np.radians(pitch))
                a, b = t * n[0], t * n[1]
                c = eaves[rid] - a * mid[0] - b * mid[1]
                plane = (float(a), float(b), float(c))
            slope, aspect = _slope_aspect(plane)
            if slope > config.MAX_ROOF_SLOPE_DEG:
                continue
            if slope >= STEEP_FACE_DEG and _inlier_fraction(sub, plane) < STEEP_FACE_MIN_FIT:
                continue
            out.append({
                "building_id": building_id,
                "geometry": poly,
                "plane_a": plane[0], "plane_b": plane[1], "plane_c": plane[2],
                "slope_deg": slope, "aspect_deg": aspect,
                "area_m2": float(poly.area), "point_count": int(len(sub)),
            })
    return out


import shapely.geometry  # used in the constructed-plane fallback above

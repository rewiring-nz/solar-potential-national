"""Move each shared ridge onto the crest the LiDAR actually shows.

2 Preston Drive shipped panels over a ridge line. Measured: a symmetric 24/24 degree gable whose two
faces met 3.66 m across an 8.4 m wing -- 0.8 m west of the crest, which the
point cloud puts at 4.44 m on every 1 m slice along the wing. The east face
was 115 m2 to the west's 88, and its first column of panels sat astride the
real ridge. Across arrowtown_millbrook, 2,171 opposite-facing facet pairs:
the crest offset is unbiased (median 0.00 m) but 36% are more than 0.5 m
off and 15% more than 1 m -- one gable in six has a ridge a panel can
straddle.

WHY THE PARTITION GETS IT WRONG. The boundary between two faces is placed
where their two fitted planes intersect (roof_partition._refine_cut). On
sparse LiDAR (Arrowtown is ~4 returns/m2) a plane's height is uncertain by
a few centimetres per fit, and two 24-degree planes that are each 10 cm off
in opposite directions meet 45 cm from where they should. The crest itself
is far better determined than either plane: it is the maximum of the
profile, and every return along the ridge votes on it.

WHAT THIS DOES. For each pair of pitched facets facing away from each
other (aspects >= 150 degrees apart) that share at least RIDGE_MIN_LEN_M
of boundary, take the returns within CREST_SEARCH_M either side of that
boundary and fit a TENT: z = z0 + g*along - s1*max(0, c-x) - s2*max(0, x-c),
two planes meeting on a line parallel to the boundary at across-offset c.
The best c on a 5 cm grid is the crest. If it lies between SNAP_MIN_M and
SNAP_MAX_M from the boundary, both slopes fall away from it, the fit is
clean, and the per-slice crests agree (a crest that wanders is not a
line), every vertex on that boundary -- of either facet and of any hip
piece whose apex sits on it -- moves by the offset. The facets keep their
fitted planes; only the line where one hands over to the other moves.

Drawn roofs are not touched: the markup wins (build_layout_geojson,
"A DRAWN ROOF IS NOT WITHHELD").
"""

import os

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union

RIDGE_MIN_LEN_M = 4.0
RIDGE_MIN_SLOPE_DEG = 10.0
RIDGE_OPPOSITE_DEG = 150.0
CREST_SEARCH_M = 3.0          # returns this far either side of the boundary vote
CREST_GRID_M = 0.05
SNAP_MIN_M = 0.3              # a boundary this close to the crest stands
SNAP_MAX_M = 2.5              # further than this is a different roof feature
SLOPE_MIN = np.tan(np.radians(5.0))   # both sides must fall away from the crest
FIT_RMS_MAX_M = 0.35
SLICE_M = 2.0
SLICE_AGREE_M = 0.3           # a slice agrees with the majority within this
SLICE_AGREE_SHARE = 0.6       # ...and the majority must be at least this share
SLICE_RMS_MAX_M = 0.2         # the agreeing slices must fit their tent this well
SLICE_DRIFT_MAX_M = 0.6       # end-to-end drift of the agreeing crests along the run
MIN_POINTS = 24
VERTEX_TOL_M = 0.3
MIN_FACET_M2 = 1.0
GAP_BRIDGE_M = 0.12           # faces cut from buffered slivers sit up to ~10 cm apart
SIMPLIFY_M = 0.03             # straighten the bridge's rounded ends off the outline
FOOTPRINT_TOL_M2 = 0.5        # the facets' union may not grow or shrink by more
OVERLAP_TOL_M2 = 0.3          # nor may they start overlapping each other

SNAP_STATS = {"pairs": 0, "measured": 0, "snapped": 0, "moved_m": [],
              "recut_failed": 0, "third_facet_failed": 0, "guard_reverted": 0}


def _tent_fit(along, across, z, c_lo, c_hi, strict=True):
    """Best crest offset c and the fit at it; None when the model does not hold.
    strict=False only asks WHERE the crest is (the per-slice agreement check);
    the slope and residual guards are applied to the whole-ridge fit."""
    best = None
    for c in np.arange(c_lo, c_hi + 1e-9, CREST_GRID_M):
        A = np.column_stack([np.ones_like(z), along,
                             -np.maximum(0.0, c - across), -np.maximum(0.0, across - c)])
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        r = z - A @ coef
        rms = float(np.sqrt((r * r).mean()))
        if best is None or rms < best[0]:
            best = (rms, float(c), coef)
    rms, c, coef = best
    if strict and (coef[2] < SLOPE_MIN or coef[3] < SLOPE_MIN or rms > FIT_RMS_MAX_M):
        return None
    return c, rms


def _ridge_line(ga, gb):
    """(origin, unit along, unit normal toward ga, length) of the shared
    boundary, or None when the two do not share a straight run of it."""
    shared = ga.buffer(0.05).intersection(gb.buffer(0.05))
    if shared.is_empty or shared.area < 0.2:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):   # shapely's envelope on exact rectangles
        mrr = shared.minimum_rotated_rectangle
    if mrr.geom_type != "Polygon":
        return None
    cs = np.asarray(mrr.exterior.coords)[:4]
    edges = [(np.linalg.norm(cs[k + 1] - cs[k]), k) for k in range(3)]
    length, k = max(edges)
    if length < RIDGE_MIN_LEN_M:
        return None
    u = (cs[k + 1] - cs[k]) / length
    n = np.array([-u[1], u[0]])
    if (np.asarray(ga.centroid.coords[0]) - np.asarray(shared.centroid.coords[0])) @ n < 0:
        n = -n
    # origin: the shared run's own start, not the rectangle corner
    c0 = np.asarray(shared.centroid.coords[0]) - u * (length / 2)
    return c0, u, n, length


def _points_near(pc_source, dsm, c0, u, n, length):
    corners = np.array([c0 - n * (CREST_SEARCH_M + .5), c0 + n * (CREST_SEARCH_M + .5),
                        c0 + u * length - n * (CREST_SEARCH_M + .5), c0 + u * length + n * (CREST_SEARCH_M + .5)])
    minx, miny = corners.min(0)
    maxx, maxy = corners.max(0)
    pts = None
    if pc_source is not None:
        try:
            pts = pc_source.points_in_bbox(minx, miny, maxx, maxy, building_only=True)
        except TypeError:
            pts = pc_source.points_in_bbox(minx, miny, maxx, maxy)
    if (pts is None or len(pts) < MIN_POINTS) and dsm is not None:
        from src.roof_segmentation import _dsm_points_in
        pts = _dsm_points_in(box(minx, miny, maxx, maxy), dsm)
    if pts is None or len(pts) < MIN_POINTS:
        return None
    return np.asarray(pts, dtype=float)


def crest_offset(pc_source, dsm, ga, gb):
    """How far (m, toward ga) the LiDAR crest sits from the ga/gb boundary,
    or None when there is no clean straight crest to measure."""
    rl = _ridge_line(ga, gb)
    if rl is None:
        return None
    c0, u, n, length = rl
    pts = _points_near(pc_source, dsm, c0, u, n, length)
    if pts is None:
        return None
    rel = pts[:, :2] - c0
    along, across, z = rel @ u, rel @ n, pts[:, 2]
    keep = (along > -0.5) & (along < length + 0.5) & (np.abs(across) <= CREST_SEARCH_M)
    if keep.sum() < MIN_POINTS:
        return None
    along, across, z = along[keep], across[keep], z[keep]
    fit = _tent_fit(along, across, z, -SNAP_MAX_M, SNAP_MAX_M)
    if fit is None:
        return None
    c, _ = fit
    # A crest is a LINE. Where the run passes a junction -- the wing meets a
    # hip section, a dormer sits on it -- the tent does not hold and those
    # slices land anywhere (2 Preston Drive: 12 m of slices at +0.80 with
    # rms 0.03, two junction slices at -0.75 and -0.15). So: the crest is
    # where the majority of slices agree, it must be clean there, and it
    # must not drift along the run. The junction slices are simply outvoted.
    def slices(slice_m):
        out = []
        for a in np.arange(0.0, length - slice_m + 1e-9, slice_m):
            m = (along >= a) & (along < a + slice_m)
            if m.sum() < 8:
                continue
            f = _tent_fit(along[m], across[m], z[m], c - 1.5, c + 1.5, strict=False)
            if f is not None:
                out.append((a + slice_m / 2, f[0], f[1]))
        return out
    sl = slices(SLICE_M)
    if len(sl) < 2:
        sl = slices(2 * SLICE_M)        # sparse returns: longer slices
    if len(sl) < 2:
        return None
    sl = np.asarray(sl)
    med = np.median(sl[:, 1])
    agree = sl[np.abs(sl[:, 1] - med) <= SLICE_AGREE_M]
    if len(agree) < 2 or len(agree) < SLICE_AGREE_SHARE * len(sl):
        return None
    if np.median(agree[:, 2]) > SLICE_RMS_MAX_M:
        return None
    if len(agree) >= 3:
        drift = abs(np.polyfit(agree[:, 0], agree[:, 1], 1)[0]) * length
        if drift > SLICE_DRIFT_MAX_M:
            return None
    return float(np.median(agree[:, 1])), rl


def _shift_vertices(geom, c0, u, n, length, d):
    """Move every vertex on the boundary run by d along n."""
    def ring(coords):
        out = []
        for x, y in coords:
            rel = np.array([x, y]) - c0
            a, t = rel @ u, rel @ n
            if abs(t) <= VERTEX_TOL_M and -0.5 <= a <= length + 0.5:
                out.append((x + n[0] * d, y + n[1] * d))
            else:
                out.append((x, y))
        return out
    if geom.geom_type == "Polygon":
        return Polygon(ring(geom.exterior.coords), [ring(i.coords) for i in geom.interiors])
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([_shift_vertices(g, c0, u, n, length, d) for g in geom.geoms])
    return geom


def _polys(g):
    """The polygon parts of any geometry, largest first."""
    if g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    parts = [q for q in getattr(g, "geoms", []) if q.geom_type == "Polygon" and q.area > 1e-6]
    return sorted(parts, key=lambda q: -q.area)


def _recut(gi, gj, c0, u, n, length, d):
    """Hand the band between the old boundary and the crest (over the ridge's
    own run) from one face to the other. Local by design: a full-length cut of
    the two faces' union failed on 113 of 253 ridges -- the union is often a
    MultiPolygon by a floating-point sliver, and an L-shaped face gets cut
    twice. A strip touches only what lies between the two lines.

    Two things go wrong with a plain transfer and are handled here. The faces
    were cut from buffered slivers and can sit 5-10 cm apart, so the band
    does not touch its new owner and a union leaves it as a separate part:
    the gap is bridged where the band meets the taker. And the giver's
    remainder can come apart -- a corner beyond the run's end detached from
    the main face -- so detached remainders go to the taker too. Either way
    the two faces cover exactly what they covered before."""
    import shapely
    lo, hi = -1.0, length + 1.0
    strip = Polygon([tuple(c0 + u * lo), tuple(c0 + u * hi),
                     tuple(c0 + u * hi + n * d), tuple(c0 + u * lo + n * d)])
    if not strip.is_valid:
        strip = strip.buffer(0)
    giver, taker = (gi, gj) if d > 0 else (gj, gi)
    try:
        piece = unary_union(_polys(giver.intersection(strip)))
        if piece.is_empty or piece.area < 0.05:
            return None
        remainder = _polys(giver.difference(piece))
        if not remainder:
            return None
        new_giver, extras = remainder[0], remainder[1:]
        # The bridge fills the faces' gap ONLY between the band's own ends: a
        # bare buffer protruded past them as 20 cm round bumps, and the
        # building frame -- which reads bearings off every facet edge --
        # took those stray edges seriously (2 Preston Drive: 81 panels to
        # 56, on facets the snap never touched).
        rel = np.asarray(piece.exterior.coords)[:, :2] - c0 if piece.geom_type == "Polygon" else \
            np.vstack([np.asarray(q.exterior.coords)[:, :2] for q in piece.geoms]) - c0
        a_lo, a_hi = float((rel @ u).min()), float((rel @ u).max())
        sgn = 1.0 if d > 0 else -1.0
        band = Polygon([tuple(c0 + u * a_lo - n * sgn * GAP_BRIDGE_M), tuple(c0 + u * a_hi - n * sgn * GAP_BRIDGE_M),
                        tuple(c0 + u * a_hi + n * d), tuple(c0 + u * a_lo + n * d)])
        if not band.is_valid:
            band = band.buffer(0)
        bridge = piece.buffer(GAP_BRIDGE_M).intersection(taker.buffer(GAP_BRIDGE_M)).intersection(band)
        new_taker = unary_union([taker, piece, bridge] + extras)
        new_taker = shapely.set_precision(new_taker, 1e-3)
        parts = _polys(new_taker)
        if not parts or sum(q.area for q in parts[1:]) > 0.05:
            return None
        # the bridge is a buffer and leaves its rounded ends on the outline:
        # 73 vertices on 2 Preston Drive's west face and 56 panels where
        # there had been 81. Straighten it back.
        new_taker = parts[0].simplify(SIMPLIFY_M, preserve_topology=True)
    except Exception:
        return None
    for g in (new_giver, new_taker):
        if not g.is_valid or g.area < MIN_FACET_M2:
            return None
    return (new_giver, new_taker) if d > 0 else (new_taker, new_giver)


def _ridge_vertices(geoms, c0, u, n, length):
    """The vertices of the two faces that lie on the shared run."""
    pts = []
    for g in geoms:
        for x, y in g.exterior.coords:
            rel = np.array([x, y]) - c0
            if abs(rel @ n) <= VERTEX_TOL_M and -0.5 <= rel @ u <= length + 0.5:
                pts.append((x, y))
    return np.asarray(pts) if pts else np.zeros((0, 2))


def _shift_coincident(geom, ridge_pts, n, d):
    """Move the vertices of geom that coincide (5 cm) with a ridge vertex."""
    if not len(ridge_pts) or geom.geom_type != "Polygon":
        return geom
    changed = False

    def ring(coords):
        nonlocal changed
        out = []
        for x, y in coords:
            if np.min(np.hypot(ridge_pts[:, 0] - x, ridge_pts[:, 1] - y)) <= 0.05:
                out.append((x + n[0] * d, y + n[1] * d))
                changed = True
            else:
                out.append((x, y))
        return out
    ext = ring(geom.exterior.coords)
    ints = [ring(i.coords) for i in geom.interiors]
    return Polygon(ext, ints) if changed else geom


def snap_ridges_to_crest(facets, pc_source, dsm=None):
    """Return facets with each shared ridge moved onto the measured crest.
    SOLAR_RIDGE_SNAP=0 in the environment turns the pass off, so a golden or
    bench run can show exactly what it moved."""
    if os.environ.get("SOLAR_RIDGE_SNAP", "1") == "0":
        return facets
    if not facets or any(f.get("from_labels") for f in facets):
        return facets
    facets = [dict(f) for f in facets]
    for i in range(len(facets)):
        for j in range(i + 1, len(facets)):
            fi, fj = facets[i], facets[j]
            if fi.get("slope_deg", 0) < RIDGE_MIN_SLOPE_DEG or fj.get("slope_deg", 0) < RIDGE_MIN_SLOPE_DEG:
                continue
            da = abs(((fi.get("aspect_deg", 0) - fj.get("aspect_deg", 0)) + 180) % 360 - 180)
            if da < RIDGE_OPPOSITE_DEG:
                continue
            SNAP_STATS["pairs"] += 1
            r = crest_offset(pc_source, dsm, fi["geometry"], fj["geometry"])
            if r is None:
                continue
            SNAP_STATS["measured"] += 1
            d, (c0, u, n, length) = r
            if abs(d) < SNAP_MIN_M or abs(d) > SNAP_MAX_M:
                continue
            # The two faces are RE-CUT along the shifted line. Sliding their
            # vertices sideways instead put a gable-end vertex off its wall
            # and changed the footprint on 48 of 253 ridges; a cut cannot.
            recut = _recut(fi["geometry"], fj["geometry"], c0, u, n, length, d)
            if recut is None:
                SNAP_STATS["recut_failed"] += 1
                continue
            gi, gj = recut
            moved = [(i, gi), (j, gj)]
            ok = True
            claimed = unary_union([gi, gj])
            # A third facet follows the ridge only where it SHARES a vertex
            # with it -- a hip apex, a dormer corner sitting on the ridge --
            # and then gives up whatever now lies under the two faces.
            # Anything looser (every vertex within 0.3 m of the line) dragged
            # neighbouring faces along and lost footprint on 92 of 253 ridges.
            ridge_pts = _ridge_vertices([fi["geometry"], fj["geometry"]], c0, u, n, length)
            for k, f in enumerate(facets):
                if k in (i, j):
                    continue
                g = _shift_coincident(f["geometry"], ridge_pts, n, d)
                if g is not f["geometry"] and not g.is_valid:
                    g = g.buffer(0)
                try:
                    under = g.intersection(claimed).area
                    if g is f["geometry"] and under < 0.05:
                        continue
                    if under > 0.05:
                        g = g.difference(claimed)
                except Exception:
                    ok = False
                    break
                if g.geom_type != "Polygon" or not g.is_valid or g.is_empty or g.area < MIN_FACET_M2:
                    ok = False
                    break
                moved.append((k, g))
            if not ok:
                SNAP_STATS["third_facet_failed"] += 1
                continue
            # The roof must still tile the same ground: no footprint change,
            # no overlap opened. Revert those.
            before = [f["geometry"] if f["geometry"].is_valid else f["geometry"].buffer(0) for f in facets]
            after = list(before)
            for k, g in moved:
                after[k] = g
            try:
                u0, u1 = unary_union(before), unary_union(after)
            except Exception:                 # a topology GEOS cannot resolve: leave the roof alone
                SNAP_STATS["guard_reverted"] += 1
                continue
            overlap0 = sum(g.area for g in before) - u0.area
            overlap1 = sum(g.area for g in after) - u1.area
            if abs(u1.area - u0.area) > FOOTPRINT_TOL_M2 or overlap1 - overlap0 > OVERLAP_TOL_M2:
                SNAP_STATS["guard_reverted"] += 1
                continue
            for k, g in moved:
                facets[k]["geometry"] = g
                if "area_m2" in facets[k]:
                    facets[k]["area_m2"] = g.area
            SNAP_STATS["snapped"] += 1
            SNAP_STATS["moved_m"].append(abs(d))
    return facets

"""
Per roof facet: fit the maximum number of 1x2m panels, respecting an edge
setback.

The key subtlety: a facet's polygon (from roof_segmentation) is in
plan-view (map) coordinates, but a panel lying flush on a tilted roof
covers *more* roof surface than its plan-view footprint suggests -- the
dimension running up/down the slope is foreshortened in plan view by
cos(slope). Packing panels directly against the plan-view polygon would
under- or over-fit real panels.

Fix: "unroll" the facet into its own 2D on-surface coordinate frame
(u = along the ridge/contour direction, unaffected by tilt; v = up the
slope, plan-view length scaled by 1/cos(slope) to recover true surface
length), pack real 1x2m rectangles there, then map the fitted panel
corners back to plan-view world coordinates for output/mapping.

Panels are packed in uniform aligned rows (the way real installations are
racked), not general 2D bin-packing -- a handful of row/column start
offsets are tried per orientation (portrait/landscape) and the
best-fitting configuration is kept. This will sometimes miss a panel that
true irregular bin-packing could squeeze in; documented approximation,
not hidden.
"""

import sys
from pathlib import Path

import warnings

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

RASTER_RESOLUTION_M = 0.1  # occupancy grid cell size in surface space; 10 cells/m
LANDSCAPE_WIN_MARGIN = 0.10  # landscape must fit >10% more panels than portrait to be
# chosen. Below that the tidier, conventional orientation is worth more than the extra panel,
# and -- more importantly -- two near-identical halves of one roof then agree with each other.
OFFSET_STEPS = 10  # vertical row-start offsets tried per orientation (columns are scanned exhaustively, see _pack_orientation)


FLAT_SLOPE_DEG = 10.0        # below this, slope direction barely constrains racking
FACET_RECTANGULARITY_MIN = 0.7  # facet area / its own bounding rectangle's area
SETBACK_LADDER_MIN_GAIN_PANELS = 2  # a tighter edge setback has to win at least this many more
# panels to be worth taking. Two, not a percentage: a percentage sounds principled but scales
# with facet size, so it blocked a genuine extra row on a small roof while waving through a
# handful of edge-jammed panels on a big one. The point of the rule is only to refuse the
# single panel squeezed hard against a boundary -- Josh's "clean install over every square
# inch" -- not to refuse real capacity.


def _edge_aligned_axes(facet_polygon, aspect_deg, slope_deg=None, building_polygon=None):
    """Real installers rack panels parallel to the roof edge, not to
    whatever direction the RANSAC-fit plane's aspect happens to point --
    and the fitted aspect can be off by a few degrees from the facet's
    actual eave/ridge line (segmentation noise). This finds that real
    edge direction via the polygon's minimum-rotated-rectangle (its two
    edges are, for a roughly rectangular/parallelogram roof facet, a
    good estimate of the true eave-parallel and slope-parallel
    directions) and uses that for the packing axes instead of raw aspect.
    Falls back to the pure-aspect axes if the polygon is degenerate.
    Returns (u_hat, v_hat) unit vectors in world (east, north)."""
    theta = np.radians(aspect_deg)
    fallback_v = np.array([np.sin(theta), np.cos(theta)])
    fallback_u = np.array([np.cos(theta), -np.sin(theta)])

    # On a near-flat roof there is no slope direction to rack against, and
    # the FACET polygon is a hull-derived blob whose minimum rectangle can
    # sit at any angle -- which is why flat-roof rows came out skew to the
    # parapets. The building outline is crisp and rectilinear, so use it as
    # the reference there. Pitched roofs keep using their own facet edges
    # (eave/ridge lines), which is the correct reference for them.
    # Only override when the facet's own shape is an unreliable guide: a
    # low-slope facet whose outline is blobby rather than rectangular (the
    # hull of a segmented flat roof). A clean rectangular facet already
    # agrees with the building and keeps its own edges; a pitched roof always
    # does, since its eave/ridge lines are the correct reference.
    # The slope test that used to gate this is gone. It only allowed the
    # building-outline fallback below FLAT_SLOPE_DEG, on the theory that a
    # pitched facet always has trustworthy eave/ridge lines. It does not: a
    # BLOBBY facet's minimum rotated rectangle is an unreliable axis estimate
    # at any pitch, and two examples from Josh show it plainly -- 26 Ballarat
    # St's bad facet sits at 12.0 deg (rectangularity 0.63) and 18 Ballarat
    # St's at 29.1 deg (rectangularity 0.44, its axis running 41.9 deg against
    # the building's own 132.3 deg, a 90-degree cross-grain block). On 111
    # Hallenstein St one plane of a roof is racked correctly and the other,
    # same roof, is skewed. Blobbiness is the right test; slope was never part
    # of it. A clean facet still keeps its own edges, which is what the
    # rectangularity check protects.
    reference = facet_polygon
    if building_polygon is not None:
        try:
            rect = facet_polygon.minimum_rotated_rectangle.area
            blobby = rect > 0 and (facet_polygon.area / rect) < FACET_RECTANGULARITY_MIN
        except Exception:
            blobby = False
        # NEAR-FLAT facets always defer to the building, blobby or not. A flat
        # roof has no real slope direction, so its "aspect" is noise, and two
        # facets of the SAME flat roof can end up racked at different angles --
        # Josh on 26 Isle St: "this roof filled with two different angles of
        # panels when it should just fill consistently in the same direction".
        # Deferring to the building outline is what makes every array on one
        # building share an angle, which is the thing that reads as a real
        # install. A pitched facet still uses its own eave/ridge lines, because
        # there the slope direction IS real -- unless the facet is blobby, in
        # which case its own rectangle is not to be trusted at any pitch.
        if blobby or (slope_deg is not None and slope_deg < FLAT_SLOPE_DEG):
            reference = building_polygon

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mrr = reference.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        if len(coords) < 4:
            return fallback_u, fallback_v
        edge1 = np.array(coords[1]) - np.array(coords[0])
        edge2 = np.array(coords[2]) - np.array(coords[1])
    except Exception:
        return fallback_u, fallback_v

    if (not np.all(np.isfinite(edge1)) or not np.all(np.isfinite(edge2))
            or np.linalg.norm(edge1) < 1e-6 or np.linalg.norm(edge2) < 1e-6):
        return fallback_u, fallback_v
    edge1, edge2 = edge1 / np.linalg.norm(edge1), edge2 / np.linalg.norm(edge2)

    # Pick whichever of the two perpendicular edge directions best matches
    # the aspect-derived contour direction -- that one becomes u_hat (sign
    # doesn't matter for u), the other becomes v_hat, whose sign is then
    # set to point downslope so the foreshortening correction lands on
    # the right axis.
    if abs(np.dot(edge1, fallback_u)) >= abs(np.dot(edge2, fallback_u)):
        u_hat, v_hat = edge1, edge2
    else:
        u_hat, v_hat = edge2, edge1
    if np.dot(v_hat, fallback_v) < 0:
        v_hat = -v_hat
    return u_hat, v_hat


def _surface_transform(u_hat, v_hat, slope_deg, origin):
    """Returns (to_surface, to_world) coordinate-transform functions between
    plan-view world (x, y) and the facet's own (u, v) on-surface frame,
    where u = edge-aligned contour direction, v = true surface distance
    up-slope. u_hat/v_hat: orthonormal world-frame unit vectors from
    _edge_aligned_axes."""
    slope_rad = np.radians(slope_deg)
    cos_slope = max(np.cos(slope_rad), 1e-6)  # guard near-vertical, shouldn't happen post slope-filter
    x0, y0 = origin

    def to_surface(x, y, z=None):
        x, y = np.asarray(x), np.asarray(y)
        dx, dy = x - x0, y - y0
        u = dx * u_hat[0] + dy * u_hat[1]
        v_plan = dx * v_hat[0] + dy * v_hat[1]
        v = v_plan / cos_slope
        return u, v

    def to_world(u, v):
        u, v = np.asarray(u), np.asarray(v)
        v_plan = v * cos_slope
        dx = u * u_hat[0] + v_plan * v_hat[0]
        dy = u * u_hat[1] + v_plan * v_hat[1]
        return x0 + dx, y0 + dy

    return to_surface, to_world


ALIGN_LOSS_TOLERANCE = 0.05  # column-aligned packing is preferred unless it fits more than
# max(1, this fraction) fewer panels than the free per-row scan -- real installers rack rows
# with columns lined up, so a small capacity cost buys a much more realistic layout, but a
# jagged/angled facet edge where rigid columns strand serious usable area still falls back


def _pack_orientation(occupancy, res, w, h, offset_steps=OFFSET_STEPS):
    """occupancy: boolean grid, True = usable. w, h in metres (grid cells).

    Two packing strategies, best-of:
    1. Column-aligned grid (preferred): panels sit at a shared column pitch
       across every row -- every column phase is tried, so the grid snaps to
       whichever registration fits the facet best. This is how real arrays
       are racked: rows with their vertical edges lined up, a blocked slot
       skipped rather than the whole row sliding sideways.
    2. Free per-row scan (fallback): each row-band independently scans every
       column start, packing maximum panels at the cost of staggered,
       unrealistic column seams -- kept only for facets where rigid columns
       genuinely strand usable area (see ALIGN_LOSS_TOLERANCE).
    A handful of vertical (row) start offsets are tried for both, since where
    the first row-band starts can gain or lose a whole extra row lower down."""
    rows, cols = occupancy.shape
    w_cells, h_cells = max(1, round(w / res)), max(1, round(h / res))
    if w_cells > cols or h_cells > rows:
        return []

    # Precompute a summed-area table so any rectangle's "all occupied?"
    # check is O(1) instead of O(w_cells*h_cells) -- matters once this runs
    # across thousands of facets.
    sat = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    sat[1:, 1:] = np.cumsum(np.cumsum(occupancy.astype(np.int32), axis=0), axis=1)

    def rect_fully_occupied(r0, r1, c0, c1):
        total = sat[r1, c1] - sat[r0, c1] - sat[r1, c0] + sat[r0, c0]
        return total == (r1 - r0) * (c1 - c0)

    # Every row phase, not a sample of them. This used to try offset_steps=10 of
    # the h_cells possible alignments (10 of 20 for a portrait panel at 0.1m
    # cells), so whether a facet got its best row alignment was partly luck --
    # and the luck changed whenever the usable SHAPE changed, which made
    # otherwise-strict improvements upstream look like small regressions. The
    # extra phases cost one more sweep each and the summed-area table makes a
    # sweep cheap.
    row_offsets = range(h_cells) if h_cells <= 2 * offset_steps else \
        np.linspace(0, h_cells, offset_steps, endpoint=False, dtype=int)

    best_aligned = []
    for r_off in row_offsets:
        for c_off in range(w_cells):
            placed = []
            r0 = r_off
            while r0 + h_cells <= rows:
                c0 = c_off
                while c0 + w_cells <= cols:
                    if rect_fully_occupied(r0, r0 + h_cells, c0, c0 + w_cells):
                        placed.append((r0, c0, r0 + h_cells, c0 + w_cells))
                    c0 += w_cells  # always step by the grid pitch -- columns stay aligned
                r0 += h_cells
            if len(placed) > len(best_aligned):
                best_aligned = placed

    best_free = []
    for r_off in row_offsets:
        placed = []
        r0 = r_off
        while r0 + h_cells <= rows:
            c0 = 0
            while c0 + w_cells <= cols:
                if rect_fully_occupied(r0, r0 + h_cells, c0, c0 + w_cells):
                    placed.append((r0, c0, r0 + h_cells, c0 + w_cells))
                    c0 += w_cells  # jump past the panel just placed
                else:
                    c0 += 1  # fine-grained search for the next valid start in this row
            r0 += h_cells
        if len(placed) > len(best_free):
            best_free = placed

    allowed_loss = max(1, int(np.ceil(ALIGN_LOSS_TOLERANCE * len(best_free))))
    best = best_aligned if len(best_aligned) >= len(best_free) - allowed_loss else best_free
    return best, w_cells, h_cells


SHALLOW_SEAM_DEG = 12.0        # a fold gentler than this needs no ridge cap
SHALLOW_SEAM_SETBACK_M = 0.05  # token clearance so two grids do not collide
SHALLOW_SEAM_REACH_M = 1.0     # how far in from such a seam the relief applies


def _shallow_seams(facet, sibling_facets):
    """Region near boundaries shared with near-coplanar neighbours, or None.

    Panels still cannot SPAN the seam -- each facet keeps its own grid -- but
    they do not need to stand half a metre back from a 6 degree change in a roof
    that has no ridge along it."""
    if not sibling_facets:
        return None
    from shapely.ops import unary_union as _uu
    na = np.array([-facet["plane_a"], -facet["plane_b"], 1.0])
    na /= np.linalg.norm(na)
    near = []
    for other in sibling_facets:
        nb = np.array([-other["plane_a"], -other["plane_b"], 1.0])
        nb /= np.linalg.norm(nb)
        ang = float(np.degrees(np.arccos(np.clip(abs(na @ nb), -1.0, 1.0))))
        if ang > SHALLOW_SEAM_DEG:
            continue
        shared = facet["geometry"].buffer(0.05).intersection(other["geometry"])
        if shared.is_empty:
            continue
        near.append(other["geometry"].buffer(SHALLOW_SEAM_REACH_M))
    if not near:
        return None
    region = _uu(near).intersection(facet["geometry"])
    return None if region.is_empty else region


def fit_panels_on_facet(facet, panel_width=config.PANEL_WIDTH_M, panel_height=config.PANEL_HEIGHT_M,
                         setback=config.PANEL_EDGE_SETBACK_M, resolution=RASTER_RESOLUTION_M,
                         obstructions=None, sibling_facets=None, ridge_setback=config.RIDGE_SETBACK_M,
                         fallback_setback=config.PANEL_EDGE_SETBACK_FALLBACK_M):
    """Returns list of panel dicts: {geometry (world XY Polygon), facet_id fields}.
    obstructions: optional list of world-XY Polygons (e.g. from
    obstruction_detection.detect_obstructions) to exclude from the usable
    area -- subtracted before the setback buffer, in plan-view world
    coordinates, before anything gets unrolled into surface space.
    sibling_facets: kept for signature compatibility; ridge clearance now
    comes from eroding the facet's own boundary (see below), which covers a
    shared ridge and any other facet edge alike.

    Two DIFFERENT clearances, applied to two different boundaries:
      - the full edge setback, measured from the building's real outer edge
      - the smaller ridge setback, measured from this facet's own boundary

    They used to compound. The facet was eroded by the edge setback AND the
    siblings were separately buffered out by the ridge setback, so an
    internal seam lost 0.55m on EACH side -- a 1.10m gap between panels
    across a ridge, where a real install leaves about 0.3m. Measured over
    the pilot area: 12,697 m2, 5.6% of all roof, and up to 20% of an
    individual complex residential roof. It also split every multi-facet
    roof into separate islands, which is most of the confetti look.

    fallback_setback: the ladder's tighter rung -- see the loop below."""
    geom = facet["geometry"]
    aspect_deg, slope_deg = facet["aspect_deg"], facet["slope_deg"]
    building_polygon = facet.get("building_geometry")

    if obstructions:
        geom = geom.difference(unary_union(obstructions))
        if geom.is_empty:
            return []

    origin = (facet["geometry"].centroid.x, facet["geometry"].centroid.y)
    u_hat, v_hat = _edge_aligned_axes(facet["geometry"], aspect_deg, slope_deg,
                                       facet.get("building_geometry"))
    to_surface, to_world = _surface_transform(u_hat, v_hat, slope_deg, origin)

    surface_poly = shapely_transform(lambda x, y, z=None: to_surface(x, y), geom)
    # The facet's own boundary carries the RIDGE clearance: that is what keeps a
    # panel off a shared hip/valley and out of the next facet's grid, and it is
    # applied once, not once per neighbour.
    # ...but scaled to the fold it actually is. The clearance exists because a
    # rigid panel cannot lie across a fold and a ridge needs its cap; both are
    # about how sharply the roof turns. Applied flat, a 45 degree gable ridge and
    # a 6 degree seam get the same 0.25 m each side -- a 0.5 m gap between
    # arrays. Measured on 7 Malaghan St, whose faces differ by 5.6-6.7 degrees
    # with height steps of 0.06-0.09 m, the setback costs 71 m2 of a 474 m2 roof
    # while packing inside the usable area is already 84-87% efficient. Josh on
    # that building: "lots of empty space not used".
    #
    # So a boundary shared with a near-coplanar neighbour keeps only a token
    # clearance, and a real ridge keeps the full amount.
    surface_ridge = surface_poly.buffer(-ridge_setback)
    shallow = _shallow_seams(facet, sibling_facets)
    if shallow is not None:
        relief = shapely_transform(lambda x, y, z=None: to_surface(x, y), shallow)
        surface_ridge = unary_union([surface_ridge,
                                     surface_poly.buffer(-SHALLOW_SEAM_SETBACK_M)
                                                 .intersection(relief)])
        if surface_ridge.geom_type not in ("Polygon", "MultiPolygon"):
            surface_ridge = surface_poly.buffer(-ridge_setback)
    # The building's real outer edge carries the full EDGE clearance. Where the
    # facet meets that edge the stricter of the two wins, which is the setback;
    # at an internal seam only the ridge clearance applies.
    surface_building = None
    if building_polygon is not None and not building_polygon.is_empty:
        surface_building = shapely_transform(lambda x, y, z=None: to_surface(x, y),
                                              building_polygon)

    # Try the whole setback ladder and keep whichever fits the most panels,
    # rather than only dropping to the fallback when the generous setback fits
    # ZERO. Measured on #4733121: that facet is a 3.4m-wide strip, so a 0.3m
    # setback each side leaves 2.8m -- exactly two 1m panel rows with 0.8m
    # stranded. At 0.2m it leaves 3.0m and takes three. The packing strategy
    # was never the problem there (a global lattice, a per-row column phase and
    # an exhaustive row phase all placed the same 12 panels, against a
    # geometric ceiling of 14 for two rows); the usable SHAPE was.
    #
    # This is Josh's call, and it is the right one under "place them everywhere
    # that is technically feasible": "maybe there is too tight of a tolerance on
    # panels going next to each other that could be relaxed a bit". The generous
    # setback stays the default because it wins whenever it can; it just no
    # longer strands a whole row to keep a margin nobody asked for.
    best = []
    for sb in sorted({setback, fallback_setback}, reverse=True):
        if surface_building is not None:
            usable = surface_ridge.intersection(surface_building.buffer(-sb))
        else:
            usable = surface_poly.buffer(-sb)   # no outline: fall back to the old behaviour
        candidate = _pack_usable(usable, panel_width, panel_height, resolution, to_world, facet)
        # The generous setback is tried first and kept unless a tighter one is a
        # REAL gain -- a whole extra row, not one squeezed panel. Josh: "it's
        # less about maximising every inch of roof space, and more about
        # building a clean install". One extra panel hard against an edge is
        # exactly the scrappiness he is asking us not to produce.
        if len(candidate) >= len(best) + SETBACK_LADDER_MIN_GAIN_PANELS:
            best = candidate
    return best


def _pack_usable(usable, panel_width, panel_height, resolution, to_world, facet):
    """`usable` is already fully eroded -- edge clearance from the building's
    outer edge, ridge clearance from the facet's own boundary, obstructions
    removed. This just lays the lattice on it."""
    if usable.is_empty:
        return []

    min_part_area = min(panel_width, panel_height) * max(panel_width, panel_height) * 0.9
    all_parts = list(usable.geoms) if usable.geom_type == "MultiPolygon" else [usable]
    parts = [p for p in all_parts if p.area >= min_part_area]
    if not parts:
        return []

    # ONE grid for the whole facet, never one grid per disconnected piece.
    # Packing each piece separately gave every piece its own grid origin AND its
    # own independent choice of portrait/landscape, so a roof split in two by a
    # small vent came out as one tidy row and one broken, offset row at a
    # different orientation. Josh reported exactly that ("why can't you have a
    # consistent and clean two rows here, rather than one consistent row and
    # then another one broken up"), and it is why a 3.3 m2 obstruction could
    # wreck a 90 m2 layout. A single grid spans the exclusion: rows stay
    # collinear across it and on both sides of it, which is how a real install
    # is racked.
    u_min, v_min, u_max, v_max = unary_union(parts).bounds
    cols = max(1, int(np.ceil((u_max - u_min) / resolution)))
    rows = max(1, int(np.ceil((v_max - v_min) / resolution)))
    transform = Affine(resolution, 0, u_min, 0, resolution, v_min)
    occupancy = rasterize([(p, 1) for p in parts], out_shape=(rows, cols), transform=transform,
                           fill=0, dtype=np.uint8).astype(bool)
    # Distance (metres) from each usable cell to the nearest excluded one (an obstruction,
    # a ridge/edge setback, the facet boundary) -- used below as a per-panel "confidence"
    # score for the density slider: a panel comfortably in the middle of a big clean area
    # scores higher than one hugging right up against an exclusion zone.
    clearance = ndimage.distance_transform_edt(occupancy) * resolution

    # Portrait -- short edge to the ridge -- unless landscape wins by a real
    # margin. Picking whichever orientation fits one more panel is what put
    # different orientations on MIRRORED HALVES of the same roof face, because
    # a centimetre of difference between two nearly identical halves flips the
    # winner. Josh on 7 York St: "the change in angles of vertical or
    # horizontal panels on mirrored sides of the same roof face. A normal
    # install would have these oriented in the same way, likely usually
    # portrait with the small edge facing the top roof ridge."
    #
    # u runs along the eave/ridge and v up-slope (see facet_axes), so the first
    # candidate, 1 m across by 2 m up, IS portrait-with-short-edge-to-the-ridge.
    # Landscape still wins where the geometry genuinely calls for it -- a wide
    # shallow strip fits far more panels lying down -- but it has to earn it
    # rather than win a tie.
    candidates = []
    for is_portrait, (w, h) in ((True, (panel_width, panel_height)),
                                (False, (panel_height, panel_width))):
        result = _pack_orientation(occupancy, resolution, w, h)
        if result:
            placed_o, wc, hc = result
            candidates.append((is_portrait, placed_o, wc, hc))

    panels = []
    if candidates:
        by_orient = {c[0]: c for c in candidates}
        port, land = by_orient.get(True), by_orient.get(False)
        if port and land:
            chosen = land if len(land[1]) > len(port[1]) * (1.0 + LANDSCAPE_WIN_MARGIN) else port
        else:
            chosen = port or land
        _, placed, w_cells, h_cells = chosen

        for r0, c0, r1, c1 in placed:
            u0, v0 = u_min + c0 * resolution, v_min + r0 * resolution
            u1, v1 = u_min + c1 * resolution, v_min + r1 * resolution
            corners_u = [u0, u1, u1, u0]
            corners_v = [v0, v0, v1, v1]
            wx, wy = to_world(corners_u, corners_v)
            panel_poly = Polygon(zip(wx, wy))
            panels.append({
                "building_id": facet["building_id"],
                "facet_aspect_deg": facet["aspect_deg"],
                "facet_slope_deg": facet["slope_deg"],
                "geometry": panel_poly,
                "area_m2": panel_width * panel_height,  # true panel area, not plan-view (foreshortened) area
                "clearance_m": float(clearance[r0:r1, c0:c1].min()),
                # Placement sequence within this facet (row-major across parts) -- the density
                # filter fills in this order so a partial layout is contiguous rows, like a real
                # staged install, not a scatter of individually-scored panels. facet_key groups
                # a facet's panels through the flat cross-facet sort even when two facets share
                # an identical binned POA (common: two parallel strips of the same roof plane),
                # which would otherwise interleave their per-facet order sequences.
                "order": len(panels),
                "facet_key": id(facet),
            })

    return panels


MAIN_ARRAY_MIN_PANELS = 10  # straggler banding only applies when the building's largest
# array is at least this big: a "big commercial main array" exists. Below it (residential),
# a couple of 2-panel blocks IS the install -- never banded (direct user feedback).
STRAGGLER_RANK_FLOOR = 80  # stragglers rank 81..100: the 80% default density shows exactly
# the arrays an installer would quote; sliding past 80 progressively adds the extras.
MINOR_ARRAY_MIN_PANELS = 4  # a straggler group smaller than this is dropped (see below) --
# roughly the smallest string a real installer bothers mounting and wiring separately
MINOR_ARRAY_MIN_FRACTION = 0.25  # ...unless it's still a meaningful share of the building's
# largest array, which keeps legitimately tiny roofs (a 2-3 panel cottage) fully intact
MINOR_ARRAY_ALWAYS_KEEP_PANELS = 20  # ...and an array this size is a real install whatever
# else is on the roof. The relative test alone does not scale to big commercial roofs: 25% of
# 29 Park St's 399-panel main array is 100 panels, which called its 72-panel secondary array a
# fragment. That is a ~32 kW array. Josh: "maybe filling in big secondary roofs if there is
# ample space for big arrays".


def drop_minor_arrays(facet_panels):
    """facet_panels: list of per-facet panel lists for ONE building. Returns
    the same structure with straggler groups emptied out.

    Real installers concentrate on the good contiguous areas; a couple of
    lone panels on a far corner of the roof, while the main array sits
    elsewhere, is visual noise and not how systems get quoted or built
    (direct user feedback, matching what the Brisbane real-installation
    survey showed: installs are one or two compact arrays, not confetti).
    A facet's group is dropped when it's both small in absolute terms
    (< MINOR_ARRAY_MIN_PANELS) and small relative to the building's largest
    group (< MINOR_ARRAY_MIN_FRACTION of it) -- the relative test is what
    protects a genuinely small roof whose "largest array" is itself 2-3
    panels: there, 2 panels IS the install, not a straggler."""
    # Softened (user feedback, bug-doc cycle 22 Aug): stragglers are no longer
    # DELETED -- they're tagged, and assign_fill_ranks banishes them to fill
    # ranks above STRAGGLER_RANK_FLOOR. The 80% default density therefore
    # shows main arrays only, while 100% still shows every feasible panel.
    # Banding only happens when a big main array exists (>= MAIN_ARRAY_MIN_PANELS):
    # on a small residential roof, scattered 2-panel blocks ARE the install.
    if not facet_panels:
        return facet_panels
    largest = max(len(panels) for panels in facet_panels)
    if largest >= MAIN_ARRAY_MIN_PANELS:
        for panels in facet_panels:
            n = len(panels)
            if 0 < n < max(MINOR_ARRAY_MIN_PANELS, MINOR_ARRAY_MIN_FRACTION * largest):
                for panel in panels:
                    panel["straggler"] = True
    return facet_panels


def _erosion_order(panels, poa_key):
    """Fill order for the density slider, built by reverse erosion: strip the
    WORST panel from the full layout repeatedly, then reverse that sequence.
    Worst = least sunny, then closest to the array's edge, then furthest from
    the surviving cluster's centre.

    Why: filling row-major peels row-by-row, so a reduced system can end up a
    thin strip hugging a parapet. A real small install on a big roof is a
    compact block in the sunniest deep part of the roof (Josh's spec). The
    edge term is normalised by the building's own array extent, so on a small
    house -- where every panel is near an edge -- it vanishes and placement
    stays realistic rather than being pushed artificially inward.
    """
    if len(panels) <= 2:
        return sorted(panels, key=lambda p: (-p[poa_key], p["facet_key"], p["order"]))

    pts = np.array([[p["geometry"].centroid.x, p["geometry"].centroid.y] for p in panels])
    span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    if span < 1e-6:
        return sorted(panels, key=lambda p: (-p[poa_key], p["facet_key"], p["order"]))

    poa = np.array([p[poa_key] for p in panels], dtype=float)
    poa_norm = (poa - poa.min()) / (np.ptp(poa) or 1.0)

    # Iterative erosion is O(n^2); on the biggest roofs (the airport fits
    # ~6,000 panels) that is minutes per building across 15k buildings. Above
    # this size, score once against the whole array's centre and sort -- same
    # compact-core behaviour, O(n log n).
    if len(panels) > 600:
        centre = pts.mean(axis=0)
        d = np.linalg.norm(pts - centre, axis=1)
        edginess = d / (d.max() or 1.0)
        score = poa_norm - 0.55 * edginess
        order = np.argsort(-score)
        return [panels[i] for i in order]

    alive = np.ones(len(panels), dtype=bool)
    removal = []
    tree_pts = pts
    for _ in range(len(panels) - 1):
        idx = np.flatnonzero(alive)
        live = tree_pts[idx]
        centre = live.mean(axis=0)
        # distance to the live cluster's edge, approximated by how far each
        # panel sits from the centre relative to the cluster's own reach
        d = np.linalg.norm(live - centre, axis=1)
        reach = d.max() or 1.0
        edginess = d / reach                     # 0 centre .. 1 rim
        score = poa_norm[idx] - 0.55 * edginess  # higher = keep longer
        worst_local = int(np.argmin(score))
        worst = int(idx[worst_local])
        removal.append(worst)
        alive[worst] = False
    removal.append(int(np.flatnonzero(alive)[0]))
    # reversed removal = fill order (last removed is filled first)
    return [panels[i] for i in reversed(removal)]


def _tag_fragment_arrays(panels):
    """Re-tag stragglers using TRUE contiguous array size.

    drop_minor_arrays makes this judgement per FACET -- it treats "every panel
    on this facet" as one array. On 29 Park St that is wrong in the way that
    matters: the roof is curved, segmentation splits it into three sections,
    each section holds plenty of panels, so nothing is a straggler -- while
    within each section the panels are a big clean block PLUS a scatter of
    lone panels and 2-3 panel fragments. Josh: "lots of lonely panels and
    small arrays of panels surrounding a large array". Facet size cannot see
    those; contiguous array size can.

    Same two thresholds as drop_minor_arrays, and the same relative test that
    protects a genuinely small roof where a 2-panel block IS the install."""
    if not panels:
        return panels
    largest = max(p.get("array_size", 1) for p in panels)
    if largest < MAIN_ARRAY_MIN_PANELS:
        return panels   # residential: scattered small blocks are the install
    cutoff = min(MINOR_ARRAY_ALWAYS_KEEP_PANELS,
                 max(MINOR_ARRAY_MIN_PANELS, MINOR_ARRAY_MIN_FRACTION * largest))
    for p in panels:
        p["straggler"] = p.get("array_size", 1) < cutoff
    return panels


def _order_by_array(panels, poa_key):
    """Fill order that finishes one array before starting the next.

    The old order eroded the building's whole panel set at once, scored by
    sunniness. So reducing the density slider stripped the least sunny SIDE
    first -- a large clean array -- while lone panels on the sunny side
    survived, because sunniness is a per-panel property and being a fragment
    is not. Josh: "the panels from one side get removed before all the lonely
    panels and small arrays from other areas get removed. That's unrealistic."

    Arrays are taken in order of total yield, so the main array fills first
    and a big secondary roof follows, while fragments -- already tagged
    stragglers by _tag_fragment_arrays -- sit in the band above. Within an
    array the existing reverse-erosion order still applies, so a partly-filled
    array is a compact block in its sunniest deep part rather than a thin
    strip. Reducing therefore cleans up the ragged edges of one array at a
    time before it starts removing whole arrays."""
    if not panels:
        return []
    groups = {}
    for p in panels:
        groups.setdefault(p.get("array_id", 0), []).append(p)
    # MEAN yield per panel, not total. Total let a big shaded array outrank a
    # small sunny one -- 40 panels x poor sun beats 12 x full sun on total --
    # so lowering the density slider stripped the SUNNY panels first. Josh,
    # live-testing (#4740662): "you are removing panels from sunnier areas
    # first... You should remove from the shadiest, lowest producing panels
    # first." Fragments are already fenced into the straggler band above, so
    # mean cannot promote a two-panel scrap over a real array; among real
    # arrays the sunniest fills first, which is also the order an installer
    # would actually build them.
    ordered = sorted(groups.values(),
                     key=lambda g: (-sum(q[poa_key] for q in g) / len(g), -len(g)))
    out = []
    for g in ordered:
        out.extend(_erosion_order(g, poa_key))
    return out


def assign_fill_ranks(panels, poa_key="poa_kwh_m2_yr"):
    """Writes p["fill_rank"] (1..100, integer percentile) onto every panel of
    ONE building, following exactly apply_panel_density's fill order
    (sunniest facet first, then row-major within the facet). The frontend
    filters panels to fill_rank <= density% client-side, which is what makes
    the density slider work on the static deployed site with no server."""
    if not panels:
        return panels
    # Array membership is computed FIRST because the ordering below depends on
    # it. It used to run at the end, purely as metadata for the frontend, which
    # meant the pipeline had a correct notion of "one contiguous array" and
    # then ranked panels without using it.
    _assign_array_membership(panels)
    _tag_fragment_arrays(panels)
    main = _order_by_array(([p for p in panels if not p.get("straggler")]), poa_key)
    extras = sorted((p for p in panels if p.get("straggler")),
                    key=lambda p: (-p[poa_key], p["facet_key"], p["order"]))
    # Main arrays occupy ranks 1..STRAGGLER_RANK_FLOOR, stragglers the band
    # above -- guaranteeing the default density cut excludes exactly the
    # stragglers regardless of their share of the building's panels.
    for i, p in enumerate(main):
        p["fill_rank"] = int(np.ceil((i + 1) / len(main) * STRAGGLER_RANK_FLOOR))
    for j, p in enumerate(extras):
        p["fill_rank"] = STRAGGLER_RANK_FLOOR + int(np.ceil((j + 1) / len(extras) * (100 - STRAGGLER_RANK_FLOOR)))

    # fill_order is the same sequence as an EXACT COUNT, 1..N, not a percentile.
    # A percentile cannot express "the best 14 panels", and that is the question
    # a real quote asks: an installer looks for an easy spot for a 6kW or 9kW
    # system, finds the best place for an array that size, and quotes on it
    # (Josh). Because the order comes from reverse erosion -- repeatedly strip
    # the worst panel, then reverse -- the first N of it is already a compact
    # block in the sunniest deep part of the roof, which is exactly the array
    # such an installer would pick. So a target system size becomes
    # "fill_order <= ceil(target_kW / panel_kW)", client-side, with no rebuild
    # needed to change the targets.
    for i, p in enumerate(main + extras):
        p["fill_order"] = i + 1
    return panels


ARRAY_TOUCH_TOL_M = 0.35   # panels this close are the same physical array (tile gaps are 4cm)


def _assign_array_membership(panels):
    """Writes array_id and array_size onto every panel: which contiguous block
    of touching panels it belongs to, and how big that block is.

    This is what lets the frontend express "clean arrays only, nothing under N
    panels" as a filter, instead of it having to be a baked-in placement rule.
    Both are per-building integers and cost one small field each in the tiles,
    so the rules can be retuned without a three-hour rebuild."""
    if not panels:
        return panels
    geoms = [p["geometry"] for p in panels]
    tree = STRtree(geoms)
    seen, gid = {}, 0
    for i in range(len(geoms)):
        if i in seen:
            continue
        gid += 1
        stack, members = [i], []
        seen[i] = gid
        while stack:
            k = stack.pop()
            members.append(k)
            probe = geoms[k].buffer(ARRAY_TOUCH_TOL_M)
            for j in tree.query(probe):
                j = int(j)
                if j not in seen and probe.intersects(geoms[j]):
                    seen[j] = gid
                    stack.append(j)
        for k in members:
            panels[k]["array_id"] = gid
            panels[k]["array_size"] = len(members)
    return panels


def apply_panel_density(panels, density_pct, poa_key="poa_kwh_m2_yr"):
    """Keeps only the top density_pct% of panels across a building's *whole*
    panel list (spanning every facet), ranked sunniest-facet-first and,
    within a facet, in row-major placement order -- so a partial layout is
    the sunniest facet filling up contiguously, row by row, the way a real
    staged install grows, rather than a scatter of individually-scored
    panels (the original clearance-ranked version looked exactly like that
    scatter). density_pct=100 returns every panel unchanged --
    fit_panels_on_facet's own output is already "every feasible panel", so
    this only ever removes panels, never adds ones that placement itself
    ruled out (an obstruction, an edge, a roof join) -- density controls
    *how much* of the feasible area is used, not a relaxation of what
    counts as feasible in the first place. Each panel dict must carry
    poa_key (annual POA irradiance for its own facet's slope/aspect, e.g.
    from SolarModel.annual_poa_kwh_per_m2) plus the order and facet_key
    fields fit_panels_on_facet already attaches; facet_key keeps a facet's
    panels grouped through the sort when two facets share an identical
    binned POA."""
    if density_pct >= 100 or not panels:
        return panels
    density_pct = max(0.0, density_pct)
    ranked = sorted(panels, key=lambda p: (-p[poa_key], p["facet_key"], p["order"]))
    keep_n = int(round(len(ranked) * density_pct / 100))
    return ranked[:keep_n]

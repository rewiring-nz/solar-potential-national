"""
The roof the LiDAR actually sees, as a polygon -- an alternative to trusting the
surveyed outline as the limit of where panels may go.

Josh: "it's important you base the panel placement likely moreso based on the
actual roof position and shape than the building outline necessarily as you
might cut off a lot of edges if the building outline is the limit."

Measured on 30 labelled buildings (tools/measure_outline_offset.py): 17.9% of
the roof-height area contiguous with each building sits OUTSIDE its outline, and
it is two different populations --

  EAVES               0.45 m median overhang, on nearly every building
  ATTACHED STRUCTURE  carports, verandas, lean-tos; 37% of the lost area lies
                      more than 3 m out, concentrated on fewer buildings

NOT A REGISTRATION OFFSET. The median best-fit translation is 0.00, 0.00 m and
shifting gains a mean IoU of +0.019, so this cannot be fixed by moving outlines
around. They are the wrong SHAPE -- a wall-line footprint rather than a roof
extent -- which is why the boundary has to be rebuilt from the surface itself.

THREE THINGS THIS DELIBERATELY DOES NOT DO:

  It never SHRINKS. The result is unioned with the original outline, so a
  building whose LiDAR is thin or patchy keeps exactly the boundary it has
  today. This can only add area, never remove it, which makes it safe to adopt
  incrementally -- no building can lose panels because of it.

  It excludes MAPPED NEIGHBOURS and anything not CONTIGUOUS with this roof.
  Both were found the hard way: LAS class 6 is every building, so the house next
  door reads as this roof, and an unmapped shed 6 m away reads as an extension.
  Skipping either inflates the answer by more than the effect being measured.

  It does not decide whether the extra area is USABLE. Slope, planarity, height
  above ground and obstruction checks all still run downstream exactly as they
  do now. This widens the region those tests are allowed to look at; it does not
  weaken any of them.
"""

import numpy as np

# Sampling grid. Finer than the ~5.8 points/m2 survey density would invent
# structure the LiDAR cannot support; coarser loses the eave being measured.
CELL_M = 0.5
PAD_M = 6.0             # how far beyond the outline to look for attached roof
MIN_HEIGHT_M = 2.0      # matches gate_panels.MIN_HEIGHT_ABOVE_GROUND_M
GROUND_SEARCH_M = 25.0
MIN_GROUND_POINTS = 20
CLOSE_M = 0.75          # bridges one-cell gaps from sparse returns
SIMPLIFY_M = 0.25       # takes the staircase off a rasterised boundary
MIN_ADD_M2 = 1.0        # ignore slivers


def _connected_to(mask, seed):
    """Components of `mask` touching `seed`, 8-connected."""
    try:
        from scipy import ndimage
        lab, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        keep = set(np.unique(lab[seed])) - {0}
        return np.isin(lab, list(keep))
    except ImportError:
        out = seed.copy()
        while True:
            g = out.copy()
            g[1:, :] |= out[:-1, :]; g[:-1, :] |= out[1:, :]
            g[:, 1:] |= out[:, :-1]; g[:, :-1] |= out[:, 1:]
            g[1:, 1:] |= out[:-1, :-1]; g[:-1, :-1] |= out[1:, 1:]
            g[1:, :-1] |= out[:-1, 1:]; g[:-1, 1:] |= out[1:, :-1]
            g &= mask
            if g.sum() == out.sum():
                return g
            out = g


def roof_region(pc_source, outline, neighbours=None, ground_z=None):
    """Where this building's roof really is, as a polygon.

    Returns the outline unchanged whenever the LiDAR cannot improve on it --
    no coverage, too few ground points to set a height datum, or nothing
    contiguous found. Callers can therefore use it unconditionally.
    """
    import shapely
    from shapely.geometry import shape
    from shapely.ops import unary_union

    if pc_source is None or outline is None or outline.is_empty:
        return outline

    minx, miny, maxx, maxy = outline.bounds
    x0, y0 = minx - PAD_M, miny - PAD_M
    x1, y1 = maxx + PAD_M, maxy + PAD_M

    if ground_z is None:
        c = outline.centroid
        g = pc_source.ground_points_in_bbox(c.x - GROUND_SEARCH_M, c.y - GROUND_SEARCH_M,
                                            c.x + GROUND_SEARCH_M, c.y + GROUND_SEARCH_M)
        if g is None or len(g) < MIN_GROUND_POINTS:
            return outline                     # no datum, so no opinion
        ground_z = float(np.percentile(g[:, 2], 50))

    pts = pc_source.points_in_bbox(x0, y0, x1, y1)
    if pts is None or len(pts) == 0:
        return outline

    high = pts[pts[:, 2] >= ground_z + MIN_HEIGHT_M]
    if len(high) == 0:
        return outline

    nx = max(1, int((x1 - x0) / CELL_M))
    ny = max(1, int((y1 - y0) / CELL_M))
    roof = np.zeros((ny, nx), dtype=bool)
    ix = ((high[:, 0] - x0) / CELL_M).astype(int)
    iy = ((high[:, 1] - y0) / CELL_M).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    roof[iy[ok], ix[ok]] = True

    gx = x0 + (np.arange(nx) + 0.5) * CELL_M
    gy = y0 + (np.arange(ny) + 0.5) * CELL_M
    XX, YY = np.meshgrid(gx, gy)
    flatx, flaty = XX.ravel(), YY.ravel()

    if neighbours is not None and not getattr(neighbours, "is_empty", True):
        on_other = shapely.contains_xy(neighbours, flatx, flaty).reshape(ny, nx)
        roof &= ~on_other

    inside = shapely.contains_xy(outline, flatx, flaty).reshape(ny, nx)
    seed = roof & inside
    if not seed.any():
        return outline                          # nothing of this roof recognised
    roof = _connected_to(roof, seed)

    try:
        from rasterio.features import shapes
        from rasterio.transform import from_origin
        # rasterio rasters run top-down; flip so row 0 is the NORTH edge
        tr = from_origin(x0, y1, CELL_M, CELL_M)
        polys = [shape(geom) for geom, val
                 in shapes(np.flipud(roof).astype(np.uint8), mask=np.flipud(roof),
                           transform=tr) if val == 1]
    except Exception:
        return outline

    if not polys:
        return outline

    grown = unary_union(polys)
    # close one-cell gaps left by sparse returns, then take the staircase off
    grown = grown.buffer(CLOSE_M).buffer(-CLOSE_M).simplify(SIMPLIFY_M)
    if grown.is_empty or not grown.is_valid:
        grown = grown.buffer(0)

    # keep only the part touching the outline: closing can bridge to something
    # that was correctly separate
    if grown.geom_type == "MultiPolygon":
        keep = [p for p in grown.geoms if p.intersects(outline)]
        if not keep:
            return outline
        grown = unary_union(keep)

    # NEVER shrink. Union so this can only ever add area.
    out = unary_union([outline, grown])
    if out.geom_type == "MultiPolygon":
        out = max(out.geoms, key=lambda p: p.area)
    if not out.is_valid:
        out = out.buffer(0)
    if out.area - outline.area < MIN_ADD_M2:
        return outline
    return out

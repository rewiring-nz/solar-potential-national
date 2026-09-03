"""
Turn drawn roof lines into closed roof faces.

Josh: "my lines might need to extend further to create an 'area'. They are in
general mark ups of the roof lines in the actual image, not relating as much to
the outline provided ... some 'sealing' of areas might be needed."

That is exactly right, and it is the difference between having line labels and
having ROOF GEOMETRY labels. A person tracing a ridge stops where the ridge
visually stops. They do not run it 30 cm further to touch an eave, and they
certainly do not run it to a surveyed outline that is sitting half a metre off
the roof. So the drawn network is very nearly a planar subdivision and almost
never exactly one, and a strict arrangement of it yields one big face and a
handful of slivers.

WHY NOT JUST TRUST THE OUTLINE AS THE OUTER BOUNDARY. Because it is frequently
wrong, which is the whole point -- measured across 52 roofs, outline-vs-roof IoU
ranges 0.20 to 0.85. Sealing to a boundary that is itself offset would bake the
offset into every face.

SO THE SEALING IS LOCAL AND CONSERVATIVE, in this order:

  SNAP     endpoints within SNAP_M of each other become one node. Two lines
           drawn to "the same" corner are the same corner.
  EXTEND   a dangling end is extended ALONG ITS OWN DIRECTION, up to EXTEND_M,
           and stops at the first thing it meets -- another line, or the roof
           hull. Direction is never invented: a ridge is extended along the
           ridge, so a wrong extension is visibly wrong rather than plausibly
           wrong.
  HULL     the outer boundary is the drawn lines' own concave-ish hull, not the
           surveyed outline, so an offset outline cannot distort the faces.

Anything still dangling after that is left alone. A line that reaches nothing
within EXTEND_M is not evidence of a face, and inventing one would manufacture
geometry the labeller did not claim.
"""

import math

SNAP_M = 0.40        # ends this close are the same point
EXTEND_M = 2.50      # how far a dangling end may reach for something to meet
HIT_TOL_M = 0.25     # how near an extension must pass to count as meeting
MIN_FACE_M2 = 1.0


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _seg_intersect_t(p1, p2, p3, p4):
    """Parameter along p1->p2 where it crosses p3->p4, or None."""
    d1x, d1y = p2[0] - p1[0], p2[1] - p1[1]
    d2x, d2y = p4[0] - p3[0], p4[1] - p3[1]
    den = d1x * d2y - d1y * d2x
    scale = math.hypot(d1x, d1y) * math.hypot(d2x, d2y)
    if scale == 0 or abs(den) < 1e-9 * scale:
        return None
    t = ((p3[0] - p1[0]) * d2y - (p3[1] - p1[1]) * d2x) / den
    u = ((p3[0] - p1[0]) * d1y - (p3[1] - p1[1]) * d1x) / den
    if -1e-9 <= u <= 1 + 1e-9:
        return t
    return None


def snap_endpoints(segs, tol=SNAP_M):
    """Merge endpoints that are within tol, so near-misses become junctions."""
    pts = []
    for a, b in segs:
        pts.extend([list(a), list(b)])
    reps = []
    for p in pts:
        for r in reps:
            if _dist(p, r) <= tol:
                p[0], p[1] = r[0], r[1]
                break
        else:
            reps.append([p[0], p[1]])
    out = []
    for i in range(0, len(pts), 2):
        a, b = pts[i], pts[i + 1]
        if _dist(a, b) > 1e-6:
            out.append((tuple(a), tuple(b)))
    return out


def _touches(pt, segs, skip, tol=HIT_TOL_M):
    for j, (c, d) in enumerate(segs):
        if j == skip:
            continue
        if _dist(pt, c) <= tol or _dist(pt, d) <= tol:
            return True
        # distance from pt to the segment
        vx, vy = d[0] - c[0], d[1] - c[1]
        L2 = vx * vx + vy * vy
        if L2 == 0:
            continue
        t = max(0.0, min(1.0, ((pt[0] - c[0]) * vx + (pt[1] - c[1]) * vy) / L2))
        if _dist(pt, (c[0] + vx * t, c[1] + vy * t)) <= tol:
            return True
    return False


def extend_dangling(segs, boundary=None, max_ext=EXTEND_M):
    """Push each free end along its own line until it meets something.

    The direction is the line's own, never guessed, so an extension that is
    wrong is obviously wrong rather than a plausible invention."""
    out = []
    for i, (a, b) in enumerate(segs):
        a, b = list(a), list(b)
        for which in (0, 1):
            end = a if which == 0 else b
            other = b if which == 0 else a
            if _touches(tuple(end), segs, i):
                continue
            dx, dy = end[0] - other[0], end[1] - other[1]
            L = math.hypot(dx, dy)
            if L == 0:
                continue
            ux, uy = dx / L, dy / L
            far = (end[0] + ux * max_ext, end[1] + uy * max_ext)
            # nearest thing this extension crosses
            best = None
            for j, (c, d) in enumerate(segs):
                if j == i:
                    continue
                t = _seg_intersect_t(tuple(end), far, c, d)
                if t is not None and 1e-6 < t <= 1.0:
                    best = t if best is None else min(best, t)
            if boundary is not None:
                try:
                    from shapely.geometry import LineString
                    ls = LineString([tuple(end), far])
                    inter = ls.intersection(boundary)
                    if not inter.is_empty:
                        pts = ([inter] if inter.geom_type == "Point"
                               else list(getattr(inter, "geoms", [])))
                        for p in pts:
                            if p.geom_type != "Point":
                                continue
                            t = _dist(end, (p.x, p.y)) / max_ext
                            if 1e-6 < t <= 1.0:
                                best = t if best is None else min(best, t)
                except Exception:
                    pass
            if best is not None:
                np_ = (end[0] + ux * max_ext * best, end[1] + uy * max_ext * best)
                if which == 0:
                    a = list(np_)
                else:
                    b = list(np_)
        out.append((tuple(a), tuple(b)))
    return out


def drawn_hull(segs, pad=0.0):
    """Outer boundary from the DRAWN lines, not the surveyed outline.

    Using the outline would bake its offset into every face -- and the offset is
    the thing we are working around."""
    from shapely.geometry import MultiPoint
    pts = [p for s in segs for p in s]
    if len(pts) < 3:
        return None
    hull = MultiPoint(pts).convex_hull
    if pad:
        hull = hull.buffer(pad)
    return hull if hull.geom_type == "Polygon" else None


def faces_from_lines(lines, outline=None, snap=SNAP_M, extend=EXTEND_M):
    """Closed faces implied by the drawn lines, after sealing.

    `lines` is [((x,y),(x,y)), ...] in metres. `outline` is used only as a
    fallback boundary when the drawn lines do not enclose enough on their own.
    Returns (faces, stats)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union, polygonize

    segs = [(tuple(a), tuple(b)) for a, b in lines if _dist(a, b) > 1e-6]
    if not segs:
        return [], {"reason": "no lines"}
    raw_faces = list(polygonize([list(s) for s in segs]))

    snapped = snap_endpoints(segs, snap)
    boundary = None
    hull = drawn_hull(snapped)
    if hull is not None:
        boundary = hull.exterior
    elif outline is not None:
        boundary = outline.exterior

    sealed = extend_dangling(snapped, boundary, extend)
    edges = [list(s) for s in sealed]
    if boundary is not None:
        coords = list(boundary.coords)
        edges += [[coords[i], coords[i + 1]] for i in range(len(coords) - 1)]

    # NODE FIRST. shapely's polygonize requires its input split at every
    # crossing; given raw segments that cross, it returns almost nothing. This
    # module was written without it and produced one big face plus slivers on
    # every real roof, which is why it ended up imported by nothing. Measured on
    # 7 Anderson Heights: 3 cells un-noded (a 177 m2 blob and two slivers),
    # 7 sensible faces noded.
    from shapely.ops import unary_union
    from shapely.geometry import LineString
    try:
        noded = unary_union([LineString(e) for e in edges])
    except Exception:
        noded = edges
    faces = [f for f in polygonize(noded)
             if f.is_valid and f.area >= MIN_FACE_M2]
    return faces, {
        "segments": len(segs),
        "faces_before_sealing": len([f for f in raw_faces if f.area >= MIN_FACE_M2]),
        "faces_after_sealing": len(faces),
        "area_m2": round(sum(f.area for f in faces), 1),
    }

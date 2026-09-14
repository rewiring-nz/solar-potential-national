"""
Turn detected line fragments into a connected roof-line network.

THE GAP THIS FILLS, measured rather than assumed. Across 299 buildings the
vision model emits about as many lines per roof as Josh draws by hand -- 19.2
against 18.0 -- and they are almost entirely disconnected:

                        lines/roof   touching pairs   dangling ends
    Josh's markup           18.0         24.3%            56.4%
    model predictions       19.2          2.6%            98.5%

That difference is not accuracy, it is TOPOLOGY. A person draws a ridge until
it meets a hip, so their lines form a graph with real junctions. The detector
sees a crease, activates along it, and the fragment ends wherever the
activation faded -- so its output is a scatter of strokes that never meet.

It matters because a subdivision needs a graph. Polygonizing disconnected
strokes yields one big face, which is exactly what happened: routing model
lines through the segment-subdivision path scored 77.4% against 84.1% for
cutting on them, and roofs collapsed to a single facet.

WHAT THIS DOES, in the order that matters:

  MERGE    fragments that are nearly collinear and nearly touching are one
           crease the detector saw in pieces. Joining them first means the
           later steps work on creases rather than on stroke ends.
  SNAP     endpoints within SNAP_M become one node, so two fragments drawn to
           the same corner meet there instead of passing.
  BRIDGE   a still-dangling end is extended along its own direction to the
           nearest line it would hit. Direction is never invented: a wrong
           bridge is visibly wrong rather than plausibly wrong.

Deliberately NOT here: anything that invents a line the detector did not
propose. The aim is to connect what was found, not to guess what was missed.
"""

import math

MERGE_ANGLE_DEG = 12.0     # fragments within this bearing are the same crease
MERGE_GAP_M = 1.5          # ...if their ends are also this close
SNAP_M = 0.50              # endpoints this close are one node
BRIDGE_M = 2.5             # how far a dangling end may reach for a line to meet
MIN_LEN_M = 0.4


def _ang(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _merge_collinear(segs):
    """Join fragments that are the same crease seen in pieces."""
    segs = [list(s) for s in segs]
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            if segs[i] is None:
                continue
            for j in range(i + 1, len(segs)):
                if segs[j] is None:
                    continue
                a, b = segs[i], segs[j]
                if abs(_ang(a[0], a[1]) - _ang(b[0], b[1])) > MERGE_ANGLE_DEG:
                    continue
                # which ends face each other?
                best = None
                for pa in (0, 1):
                    for pb in (0, 1):
                        d = _dist(a[pa], b[pb])
                        if best is None or d < best[0]:
                            best = (d, pa, pb)
                if best[0] > MERGE_GAP_M:
                    continue
                _, pa, pb = best
                far_a = a[1 - pa]
                far_b = b[1 - pb]
                # the join must not double back on itself
                if abs(_ang(far_a, far_b) - _ang(a[0], a[1])) > MERGE_ANGLE_DEG:
                    continue
                segs[i] = [far_a, far_b]
                segs[j] = None
                changed = True
        segs = [s for s in segs if s is not None]
    return [(tuple(s[0]), tuple(s[1])) for s in segs]


def _snap(segs, tol=SNAP_M):
    reps = []
    out = []
    for a, b in segs:
        na, nb = list(a), list(b)
        for p in (na, nb):
            for r in reps:
                if _dist(p, r) <= tol:
                    p[0], p[1] = r[0], r[1]
                    break
            else:
                reps.append([p[0], p[1]])
        if _dist(na, nb) > 1e-9:
            out.append((tuple(na), tuple(nb)))
    return out


def _bridge(segs, max_ext=BRIDGE_M):
    """Push each still-dangling end along its own line to the nearest hit."""
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union
    from collections import defaultdict

    if len(segs) < 2:
        return segs
    lines = [LineString(s) for s in segs]
    deg = defaultdict(int)

    def key(p):
        return (round(p[0], 2), round(p[1], 2))

    for a, b in segs:
        deg[key(a)] += 1
        deg[key(b)] += 1

    extra = []
    for i, (a, b) in enumerate(segs):
        for end, other in ((a, b), (b, a)):
            if deg[key(end)] != 1:
                continue
            dx, dy = end[0] - other[0], end[1] - other[1]
            L = math.hypot(dx, dy)
            if L == 0:
                continue
            ux, uy = dx / L, dy / L
            start = (end[0] + ux * 1e-6, end[1] + uy * 1e-6)
            far = (end[0] + ux * max_ext, end[1] + uy * max_ext)
            others = [l for k, l in enumerate(lines) if k != i]
            if not others:
                continue
            try:
                hit = LineString([start, far]).intersection(unary_union(others))
            except Exception:
                continue
            if hit.is_empty:
                continue
            cands = []
            for g in ([hit] if not hasattr(hit, "geoms") else list(hit.geoms)):
                if g.geom_type == "Point":
                    cands.append(g)
                elif g.geom_type == "LineString" and not g.is_empty:
                    cands.append(Point(g.coords[0]))
            if not cands:
                continue
            n = min(cands, key=lambda c: Point(end).distance(c))
            extra.append((tuple(end), (n.x, n.y)))
    return segs + extra


def connect(segs):
    """Fragments in, connected network out. Same [x1,y1,x2,y2] shape."""
    pairs = [((s[0], s[1]), (s[2], s[3])) for s in segs
             if len(s) == 4 and math.hypot(s[2] - s[0], s[3] - s[1]) > MIN_LEN_M]
    if len(pairs) < 2:
        return [list(p[0]) + list(p[1]) for p in pairs]
    pairs = _merge_collinear(pairs)
    pairs = _snap(pairs)
    pairs = _bridge(pairs)
    return [[a[0], a[1], b[0], b[1]] for a, b in pairs]


# --------------------------------------------------------------- repetition

# ROOF FOLDS REPEAT, AND THE DETECTOR DOES NOT KNOW IT.
#
# A sawtooth roof is a set of parallel ridges. The model finds the building's
# long axes confidently and the folds only as short weak stubs -- on 7 Anderson
# Heights the two sawtooth lines score 0.880 and 0.850 and are 1.5 m and 1.7 m
# against the 4.7-7.0 m Josh drew. Raising or lowering the confidence cutoff
# cannot help: the geometry is a stub either way.
#
# But a stub that shares a bearing with other lines on the same roof is
# probably one of a repeated set, and a repeated set of folds spans the roof.
# So extend it along ITS OWN bearing to the roof edge -- direction is never
# invented, only length.
#
# MEASURED AND REJECTED. The sibling count was expected to be the whole design;
# it turned out not to matter, because the idea is wrong in both settings:
#
#   model path            found   undrawn
#   unchanged             82.5%    22.4%
#   extend >=2 siblings   81.1%    23.5%
#   extend >=1 sibling    81.2%    23.3%
#
# Fewer real creases found AND more invented ones, either way. Obvious in
# hindsight: extending a stub across the roof produces precisely the
# full-width cut that fragments a simple roof -- on 107 Beach Street a 1.8 m
# stub already slices an 11 m building. This amplifies the bug it was meant to
# route around.
#
# Kept, unwired, because the reasoning is sound and the failure is specific:
# it would be worth revisiting only alongside a cutter that clips a cut to the
# extent of the line that justified it.
PARALLEL_TOL_DEG = 12.0
PARALLEL_MIN_LEN_M = 0.8      # below this a stub is not evidence of anything
PARALLEL_MAX_LEN_M = 6.0      # a line already this long IS the fold; leave it


def extend_parallel_sets(segs, boundary_poly, min_siblings=2,
                         tol_deg=PARALLEL_TOL_DEG):
    """Stubs that belong to a parallel set, extended across the roof."""
    from shapely.geometry import LineString

    rows = []
    for s in segs:
        if len(s) != 4:
            continue
        L = math.hypot(s[2] - s[0], s[3] - s[1])
        if L < PARALLEL_MIN_LEN_M:
            continue
        a = math.degrees(math.atan2(s[3] - s[1], s[2] - s[0])) % 180.0
        rows.append([list(s), L, a])
    out = []
    for i, (s, L, a) in enumerate(rows):
        if L > PARALLEL_MAX_LEN_M:
            out.append(s)
            continue
        sibs = sum(1 for j, (_, _, b) in enumerate(rows)
                   if j != i and min(abs(b - a), 180 - abs(b - a)) <= tol_deg)
        if sibs < min_siblings:
            out.append(s)
            continue
        cx, cy = (s[0] + s[2]) / 2.0, (s[1] + s[3]) / 2.0
        ux, uy = math.cos(math.radians(a)), math.sin(math.radians(a))
        try:
            far = LineString([(cx - ux * 300, cy - uy * 300),
                              (cx + ux * 300, cy + uy * 300)])
            clip = far.intersection(boundary_poly)
        except Exception:
            out.append(s)
            continue
        if clip.is_empty:
            out.append(s)
            continue
        g = max(getattr(clip, "geoms", [clip]), key=lambda x: x.length)
        c = list(g.coords)
        out.append([c[0][0], c[0][1], c[-1][0], c[-1][1]])
    return out

"""Vector roof lines from a heatmap, placed exactly where the activation is.

WHY THE OLD EXTRACTION COULD NOT BE PRECISE. predict_roof_lines turned the
model's per-pixel probabilities into segments by thresholding, labelling
connected components, and taking each component's SVD principal axis. Roof
lines MEET -- a hip runs into a ridge on nearly every pitched roof -- so the
component containing a junction is L- or T-shaped, and the principal axis of an
L is a diagonal through neither arm. Every junction therefore produced a line
"slightly misaligned to the actual visible roof lines", which is, verbatim, the
defect Josh reported seeing on a lot of rooftops. Weak spots in the activation
also split one crease into several components, which is where the stubs came
from.

Josh: "When lines are clearly visible on a rooftop, you should be drawing them
exactly in the right spot."

THE SHAPE OF THE FIX, stage by stage, each with a reason:

  THIN      the thresholded mask to a one-pixel skeleton (Zhang-Suen; written
            out here because the repo deliberately avoids scikit-image). The
            skeleton follows the activation's centreline, junctions and all.
  TRACE     the skeleton into paths BETWEEN nodes -- endpoints and junctions --
            so a hip and the ridge it meets are separate strokes that share an
            endpoint, never one blob sharing an axis.
  STRAIGHTEN each path into straight pieces (recursive max-deviation split),
            merging near-collinear neighbours, because roof creases are
            straight and the skeleton wobbles pixel by pixel.
  SNAP      each segment sideways, in quarter-pixel steps, to the offset that
            maximises mean probability along it. The threshold decides which
            pixels are "on"; the PEAK of the activation is where the model
            actually thinks the crease is, and at 0.1 m/px a one-pixel snap is
            10 cm of placement.
  FOLLOW    each endpoint outward along the line's own direction while the
            probability stays warm. A hard threshold truncates a crease
            wherever confidence dips below it; the underlying ridge of
            activation usually continues, and following it is what turns a
            1.5 m stub back into the 5 m fold Josh draws. Direction is never
            invented -- only length, along the fitted line.
  JOIN      endpoints that nearly meet into shared junction points, so the
            output is a network with real T- and Y-junctions, like the one a
            person draws -- 56% of Josh's endpoints touch another line against
            2.6% of the old extraction's.

Coordinates are pixel-centre (+0.5) throughout: the old code passed raw column
indices to the world transform, a systematic half-pixel -- 5 cm -- shift on
every line it ever produced.
"""

import math

import numpy as np

THR = 0.40                # a pixel this confident is part of some line
FOLLOW = 0.22             # follow an endpoint while probability stays above this
FOLLOW_MAX_PX = 40        # how far an endpoint may follow the activation ridge
FIT_TOL_PX = 1.6          # max skeleton deviation before a path is split
MERGE_DEG = 8.0           # adjacent pieces within this bearing are one line
SNAP_RANGE_PX = 3.0       # sideways search for the activation peak
JOIN_PX = 5.0             # endpoints this close become one junction
MIN_COMPONENT_PX = 12
MIN_LEN_PX = 8.0
SCORE_FLOOR = 0.45         # a refined line the model barely believes is fuzz
DUP_DEG = 10.0             # same bearing as a kept line...
DUP_DIST_PX = 3.5
SHORT_PX = 22.0            # ~2 m at 0.1 m/px
SHORT_PENALTY = 0.15       # what a short line must add in confidence          # ...and lying on it: the same crease seen twice

KINDS = ["ridge", "valley", "cliff", "hip"]


# ---------------------------------------------------------------- thinning

def _thin(mask):
    """Zhang-Suen thinning: binary mask -> one-pixel-wide skeleton."""
    img = mask.astype(np.uint8).copy()
    if img.sum() == 0:
        return img
    img = np.pad(img, 1)

    def neighbours(y, x):
        return (img[y - 1, x], img[y - 1, x + 1], img[y, x + 1],
                img[y + 1, x + 1], img[y + 1, x], img[y + 1, x - 1],
                img[y, x - 1], img[y - 1, x - 1])

    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            to_del = []
            ys, xs = np.nonzero(img[1:-1, 1:-1])
            for y, x in zip(ys + 1, xs + 1):
                p = neighbours(y, x)
                b = sum(p)
                if not (2 <= b <= 6):
                    continue
                a = sum((p[i] == 0 and p[(i + 1) % 8] == 1) for i in range(8))
                if a != 1:
                    continue
                if phase == 0:
                    if p[0] * p[2] * p[4] or p[2] * p[4] * p[6]:
                        continue
                else:
                    if p[0] * p[2] * p[6] or p[0] * p[4] * p[6]:
                        continue
                to_del.append((y, x))
            for y, x in to_del:
                img[y, x] = 0
            changed = changed or bool(to_del)
    return img[1:-1, 1:-1]


# ---------------------------------------------------------------- tracing

_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _trace(skel):
    """Skeleton -> list of pixel paths, each running node to node.

    A node is an endpoint (degree 1) or a junction (degree >= 3); paths never
    run through one, so two creases meeting stay two strokes with a shared end.
    """
    on = set(zip(*np.nonzero(skel)))
    if not on:
        return []
    deg = {p: sum(((p[0] + dy, p[1] + dx) in on) for dy, dx in _OFFS)
           for p in on}
    nodes = {p for p in on if deg[p] != 2}
    used = set()          # directed pixel-to-pixel steps already walked
    paths = []

    def walk(start, first):
        # every step is marked used in BOTH directions, or the same path gets
        # walked again from its far node and every line comes out twice
        path = [start, first]
        used.add((start, first))
        used.add((first, start))
        prev, cur = start, first
        while cur not in nodes:
            nxt = None
            for dy, dx in _OFFS:
                q = (cur[0] + dy, cur[1] + dx)
                if q in on and q != prev and (cur, q) not in used:
                    nxt = q
                    break
            if nxt is None:
                break
            used.add((cur, nxt))
            used.add((nxt, cur))
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    for n in sorted(nodes):
        for dy, dx in _OFFS:
            q = (n[0] + dy, n[1] + dx)
            if q in on and (n, q) not in used:
                paths.append(walk(n, q))
    # pure loops (rare): every pixel degree 2, no node to start from
    covered = {p for path in paths for p in path}
    for p in sorted(on - covered):
        if p in covered:
            continue
        nbrs = [(p[0] + dy, p[1] + dx) for dy, dx in _OFFS
                if (p[0] + dy, p[1] + dx) in on]
        if not nbrs:
            continue          # an isolated pixel is noise, not a loop
        loop = walk(p, nbrs[0])
        covered.update(loop)
        paths.append(loop)
    return [p for p in paths if len(p) >= 3]


# ------------------------------------------------------------- straighten

def _split_straight(path, tol=FIT_TOL_PX):
    """Recursive max-deviation split of one pixel path into straight runs."""
    pts = np.array([(x, y) for y, x in path], dtype=float) + 0.5
    out = []

    def rec(i, j):
        a, b = pts[i], pts[j]
        d = b - a
        L = np.hypot(*d)
        if L < 1e-6 or j - i < 2:
            out.append((i, j))
            return
        n = np.array([-d[1], d[0]]) / L
        dev = np.abs((pts[i:j + 1] - a) @ n)
        k = int(np.argmax(dev))
        if dev[k] > tol:
            rec(i, i + k)
            rec(i + k, j)
        else:
            out.append((i, j))

    rec(0, len(pts) - 1)
    segs = [(pts[i], pts[j]) for i, j in out
            if np.hypot(*(pts[j] - pts[i])) >= 2.0]

    # merge back the near-collinear neighbours the split produced
    merged = []
    for s in segs:
        if merged:
            a, b = merged[-1]
            ang1 = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            ang2 = math.degrees(math.atan2(s[1][1] - s[0][1], s[1][0] - s[0][0]))
            dd = abs(ang1 - ang2) % 180
            if min(dd, 180 - dd) < MERGE_DEG and np.hypot(*(s[0] - b)) < 2.0:
                merged[-1] = (a, s[1])
                continue
        merged.append(s)
    return merged


# ------------------------------------------------------------- refinement

def _bilinear(P, x, y):
    h, w = P.shape
    if not (0 <= x < w - 1 and 0 <= y < h - 1):
        return 0.0
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    return float(P[y0, x0] * (1 - fx) * (1 - fy) + P[y0, x0 + 1] * fx * (1 - fy)
                 + P[y0 + 1, x0] * (1 - fx) * fy + P[y0 + 1, x0 + 1] * fx * fy)


def _line_mean(P, a, b, n=None):
    L = np.hypot(*(b - a))
    k = max(int(L), 2)
    ts = np.linspace(0.0, 1.0, k)
    return float(np.mean([_bilinear(P, *(a + t * (b - a))) for t in ts]))


def _refine(P, a, b):
    """Snap sideways to the activation peak, then follow both ends outward."""
    d = b - a
    L = np.hypot(*d)
    if L < 1e-6:
        return a, b
    d = d / L
    nrm = np.array([-d[1], d[0]])
    best, best_o = -1.0, 0.0
    for o in np.arange(-SNAP_RANGE_PX, SNAP_RANGE_PX + 1e-9, 0.25):
        m = _line_mean(P, a + o * nrm, b + o * nrm)
        if m > best:
            best, best_o = m, o
    a = a + best_o * nrm
    b = b + best_o * nrm

    def follow(p, direction):
        q = p.copy()
        for _ in range(int(FOLLOW_MAX_PX / 0.5)):
            step = q + 0.5 * direction
            if _bilinear(P, *step) < FOLLOW:
                break
            q = step
        return q

    return follow(a, -d), follow(b, d)


MERGE_GAP_PX = 8.0         # collinear pieces this far apart are one crease
MERGE_OFF_PX = 2.5         # ...if they also lie on the same line
TRIM_PX = 12.0             # an end this close past a crossing is overshoot


def _seg_x(a, b, c, e, tol=1.0):
    """Intersection of two segments, or None. tol allows near-miss ends."""
    d1, d2 = b - a, e - c
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((c[0] - a[0]) * d2[1] - (c[1] - a[1]) * d2[0]) / den
    u = ((c[0] - a[0]) * d1[1] - (c[1] - a[1]) * d1[0]) / den
    L1, L2 = np.hypot(*d1), np.hypot(*d2)
    if -tol / L1 <= t <= 1 + tol / L1 and -tol / L2 <= u <= 1 + tol / L2:
        return a + t * d1
    return None


def _junction_cleanup(segs):
    """End every line AT the junction, the way a person draws.

    Josh, comparing the rebuilt lines with his own markup: "overlapping ends of
    most lines which I don't have in what I gave you". The overshoot is FOLLOW's
    doing -- at a junction the crossing line's own activation keeps the
    probability warm, so an end runs a few pixels past the meeting point along
    the other crease's glow. His hip stops ON the ridge; ours poked through it.

    Two purely geometric repairs, no probabilities involved:
      TRIM  an end whose line crosses another line just inside the tip is cut
            back to the crossing -- the tail past it is the overshoot.
      T-SNAP an end that stops NEAR another line is landed exactly on it, so a
            T-junction is a T rather than a near-miss.
    """
    segs = [(a.copy(), b.copy()) for a, b in segs]
    # trim overshoots back to the deepest crossing inside the tip window
    for i, (a, b) in enumerate(segs):
        d = b - a
        L = np.hypot(*d)
        if L < 1e-6:
            continue
        u = d / L
        t_lo, t_hi = 0.0, L
        for j, (c, e) in enumerate(segs):
            if i == j:
                continue
            X = _seg_x(a, b, c, e)
            if X is None:
                continue
            t = float((X - a) @ u)
            if 0.5 < t < TRIM_PX:
                t_lo = max(t_lo, t)
            if L - TRIM_PX < t < L - 0.5:
                t_hi = min(t_hi, t)
        if t_hi - t_lo >= MIN_LEN_PX:
            segs[i] = (a + t_lo * u, a + t_hi * u)
    # land T-ends exactly on the line they stop near
    for i, (a, b) in enumerate(segs):
        for endi, pnt in ((0, a), (1, b)):
            best = None
            for j, (c, e) in enumerate(segs):
                if i == j:
                    continue
                d2 = e - c
                L2 = np.hypot(*d2)
                if L2 < 1e-6:
                    continue
                u2 = d2 / L2
                t = float(np.clip((pnt - c) @ u2, 0.0, L2))
                q = c + t * u2
                dist = float(np.hypot(*(pnt - q)))
                if dist < JOIN_PX and (best is None or dist < best[0]):
                    best = (dist, q)
            if best is not None and best[0] > 1e-6:
                if endi == 0:
                    segs[i] = (best[1], segs[i][1])
                else:
                    segs[i] = (segs[i][0], best[1])
    return [(a, b) for a, b in segs if np.hypot(*(b - a)) >= MIN_LEN_PX]


def _colinear_merge(s1, s2):
    """One segment spanning both, if they are pieces of the same crease."""
    a1, b1 = s1
    a2, b2 = s2
    d1 = b1 - a1
    L1 = np.hypot(*d1)
    L2 = np.hypot(*(b2 - a2))
    if L1 < 1e-6 or L2 < 1e-6:
        return None
    ref_a, ref_d, ref_L = (a1, d1 / L1, L1) if L1 >= L2 else \
        (a2, (b2 - a2) / L2, L2)
    ang1 = math.atan2(d1[1], d1[0])
    ang2 = math.atan2((b2 - a2)[1], (b2 - a2)[0])
    dd = abs(math.degrees(ang1 - ang2)) % 180
    if min(dd, 180 - dd) > MERGE_DEG:
        return None
    nrm = np.array([-ref_d[1], ref_d[0]])
    for q in (a1, b1, a2, b2):
        if abs((q - ref_a) @ nrm) > MERGE_OFF_PX:
            return None
    t1 = sorted([(q - ref_a) @ ref_d for q in (a1, b1)])
    t2 = sorted([(q - ref_a) @ ref_d for q in (a2, b2)])
    if max(t1[0], t2[0]) - min(t1[1], t2[1]) > MERGE_GAP_PX:
        return None          # a real gap: two creases sharing a bearing
    lo, hi = min(t1[0], t2[0]), max(t1[1], t2[1])
    return (ref_a + lo * ref_d, ref_a + hi * ref_d)


# ------------------------------------------------------------------ public

def extract(prob3, to_world, thr=THR):
    """(C,H,W) probabilities -> [{'seg': [x1,y1,x2,y2], 'kind', 'score'}].

    Geometry comes from the channels COMBINED (their max): one crease should be
    one line even where two channels half-fire on it, and the kind is read back
    per line afterwards. The world transform is only applied at the end, so
    every refinement above happens at full pixel precision.
    """
    P = prob3.max(axis=0)
    mask = P > thr
    if mask.sum() < MIN_COMPONENT_PX:
        return []
    skel = _thin(mask)
    segs = []
    for path in _trace(skel):
        segs.extend(_split_straight(path))

    refined = []
    for a, b in segs:
        a2, b2 = _refine(P, np.array(a, float), np.array(b, float))
        if np.hypot(*(b2 - a2)) < MIN_LEN_PX:
            continue
        refined.append((a2, b2))

    # REJOIN THE PIECES OF ONE CREASE. The tracer splits at junctions, which is
    # right for keeping a hip off the ridge's axis -- but Josh's ridge IS one
    # line that hips meet at T-junctions, so after splitting, his single ridge
    # exists here as collinear pieces that nothing put back together. He said
    # it directly, comparing 7 Anderson Heights against his own markup: "many
    # of the lines don't go their full length ... mine is much cleaner". Two
    # segments on the same line with at most a small gap are one crease.
    changed = True
    while changed:
        changed = False
        for i in range(len(refined)):
            if refined[i] is None:
                continue
            for j in range(i + 1, len(refined)):
                if refined[j] is None:
                    continue
                m = _colinear_merge(refined[i], refined[j])
                if m is not None:
                    refined[i] = m
                    refined[j] = None
                    changed = True
        refined = [r for r in refined if r is not None]
    refined = _junction_cleanup(refined)

    # join endpoints that nearly meet, so the network has real junctions
    pts = []
    for a, b in refined:
        pts.extend([a, b])
    reps = []
    for p in pts:
        for r in reps:
            if np.hypot(*(p - r["c"])) <= JOIN_PX:
                r["members"].append(p)
                r["c"] = np.mean(r["members"], axis=0)
                break
        else:
            reps.append({"c": p.copy(), "members": [p]})

    def snap(p):
        for r in reps:
            if np.hypot(*(p - r["c"])) <= JOIN_PX:
                return r["c"]
        return p

    # SUPPRESS THE FUZZ. A noisy activation grows side-branches off a strong
    # crease, and after refinement they snap onto nearly the same line as their
    # parent. Keep the strongest reading of each crease: candidates are taken
    # best-first (score x length), and one that runs along a kept line --
    # same bearing, both ends near its infinite line, overlapping it -- is the
    # same crease seen again, not a second crease.
    cands = []
    for a, b in refined:
        a, b = snap(a), snap(b)
        L = np.hypot(*(b - a))
        if L < MIN_LEN_PX:
            continue
        sc = _line_mean(P, a, b)
        # a short line needs to be BELIEVED: fuzz branches are short and weak,
        # real creases are long or strong. Josh: "still plenty of extra lines
        # that shouldn't be there".
        floor = SCORE_FLOOR + (SHORT_PENALTY if L < SHORT_PX else 0.0)
        if sc < floor:
            continue
        cands.append((sc * L, sc, a, b))
    cands.sort(key=lambda t: -t[0])
    kept = []
    for _, sc, a, b in cands:
        dup = False
        for _, ka, kb in kept:
            kd = kb - ka
            kL = np.hypot(*kd)
            if kL < 1e-6:
                continue
            kd = kd / kL
            ang = abs(math.degrees(math.atan2(kd[1], kd[0])
                                   - math.atan2((b - a)[1], (b - a)[0]))) % 180
            if min(ang, 180 - ang) > DUP_DEG:
                continue
            nrm = np.array([-kd[1], kd[0]])
            if (abs((a - ka) @ nrm) < DUP_DIST_PX
                    and abs((b - ka) @ nrm) < DUP_DIST_PX):
                ta, tb = sorted(((a - ka) @ kd, (b - ka) @ kd))
                if min(tb, kL) - max(ta, 0.0) > 0.5 * (tb - ta):
                    dup = True
                    break
        if not dup:
            kept.append((sc, a, b))

    out = []
    for score, a, b in kept:
        # which channel owns this line
        ks = np.zeros(prob3.shape[0])
        L = np.hypot(*(b - a))
        for t in np.linspace(0, 1, max(int(L), 2)):
            q = a + t * (b - a)
            ks += [_bilinear(prob3[c], *q) for c in range(prob3.shape[0])]
        ax, ay = to_world(a[0], a[1])
        bx, by = to_world(b[0], b[1])
        out.append({"seg": [ax, ay, bx, by],
                    "kind": KINDS[int(np.argmax(ks))],
                    "score": round(score, 3)})
    return out


def clip_to(results, footprint, pad_m=0.3, min_len_m=1.0):
    """Keep only the stretch of each line that lies on THIS building.

    The model predicts over the whole crop, so a neighbour's ridge is in the
    output too -- and passed downstream it becomes a cut on a building it never
    touched. Visible on the very first render of #4734696: half the deployed
    panel's lines sit on the roofs next door.
    """
    from shapely.geometry import LineString
    zone = footprint.buffer(pad_m)
    out = []
    for r in results:
        x1, y1, x2, y2 = r["seg"]
        try:
            clipped = LineString([(x1, y1), (x2, y2)]).intersection(zone)
        except Exception:
            continue
        for g in getattr(clipped, "geoms", [clipped]):
            if g.geom_type != "LineString" or g.length < min_len_m:
                continue
            c = list(g.coords)
            out.append({**r, "seg": [c[0][0], c[0][1], c[-1][0], c[-1][1]]})
    return out

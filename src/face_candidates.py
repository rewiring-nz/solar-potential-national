"""Two independent readings of a roof's faces, and the evidence to choose one.

A day of Josh's verdicts established that no single front-end wins everywhere:

  LINE NETWORK  best on crease-textured hip houses -- "much better" on
                7 Anderson Heights -- because the detector fires on folds
  SAM           best where faces differ in tone or material -- "clearly
                better than any of your lines" on #4735106 -- because it
                segments surfaces, not folds

Every attempt to MERGE them degraded the winner: each stage's failure modes
multiplied. So they are not merged. Each generator produces a complete
candidate face-set, a scorer measures each against the imagery and LiDAR
evidence, and the better one is used -- per roof. Roofs where both score
poorly belong in Josh's markup queue, not in a guess.

The scorer's authority is not taken on faith: tools/select_faces.py measures,
on the benchmark roofs Josh has drawn, whether the scorer picks the candidate
that agrees better with HIS faces. Selection accuracy against his markup is
the only accepted validation here.
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAM_MAX_PROMPTS = 14
GRID_RES_M = 0.2
CLUTTER_MAX_M2 = 15.0
CLUTTER_STEP_M = 0.45
MERGE_SLOPE_DEG = 3.0
MERGE_ASPECT_DEG = 15.0


# ----------------------------------------------------------- shared helpers

def _grid(geom, bounds):
    import shapely
    gxs = np.arange(bounds[0], bounds[2] + GRID_RES_M, GRID_RES_M)
    gys = np.arange(bounds[1], bounds[3] + GRID_RES_M, GRID_RES_M)
    GX, GY = np.meshgrid(gxs, gys)
    inside = shapely.contains_xy(geom, GX.ravel(), GY.ravel()).reshape(GX.shape)
    return GX, GY, gys, inside


def _tile(faces, geom, bounds):
    """Assign a grid of the footprint to faces -> clean partition polygons."""
    import shapely
    import rasterio.features
    from rasterio.transform import from_origin
    from shapely.geometry import Polygon
    from scipy.spatial import cKDTree

    GX, GY, gys, inside = _grid(geom, bounds)
    lab = np.full(GX.shape, -1, dtype=int)
    for i, f in enumerate(faces):
        m = shapely.contains_xy(f, GX.ravel(), GY.ravel()).reshape(GX.shape)
        lab[m & inside & (lab < 0)] = i
    un = inside & (lab < 0)
    if un.any() and (lab >= 0).any():
        tree = cKDTree(np.c_[GX[lab >= 0], GY[lab >= 0]])
        _, nn = tree.query(np.c_[GX[un], GY[un]])
        lab[un] = lab[lab >= 0][nn]

    tr = from_origin(bounds[0] - GRID_RES_M / 2, gys[-1] + GRID_RES_M / 2,
                     GRID_RES_M, GRID_RES_M)
    out = []
    for rid in np.unique(lab):
        if rid < 0:
            continue
        mask = np.flipud(lab == rid).astype("uint8")
        best = None
        for geo, val in rasterio.features.shapes(mask, transform=tr):
            if val != 1:
                continue
            poly = Polygon(geo["coordinates"][0])
            if best is None or poly.area > best.area:
                best = poly
        if best is None:
            continue
        best = best.intersection(geom).simplify(0.3)
        for g in getattr(best, "geoms", [best]):
            if g.geom_type == "Polygon" and g.area >= 2.0:
                out.append(g)
    return out


def _plane(poly, pts):
    from src.roof_partition import _points_in, _fit_plane_robust
    sub = _points_in(poly, pts) if pts is not None and len(pts) else []
    if len(sub) < 10:
        return None, sub
    return _fit_plane_robust(sub), sub


# ------------------------------------------------------------- SAM faces

def sam_faces(predictor, rgb, geom, bounds, pts):
    """Coverage-completion SAM + LiDAR clutter/merge. Returns (faces, obs)."""
    import rasterio.features
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from src.roof_partition import _slope_aspect

    h, w = rgb.shape[:2]
    b = bounds
    predictor.set_image(rgb)

    def to_world(ring_px):
        return [(b[0] + x / w * (b[2] - b[0]),
                 b[1] + (1 - y / h) * (b[3] - b[1])) for x, y in ring_px]

    def px_of(pt):
        return np.array([[(pt.x - b[0]) / (b[2] - b[0]) * w,
                          (1 - (pt.y - b[1]) / (b[3] - b[1])) * h]],
                        dtype=np.float32)

    def mask_to_poly(mask):
        best = None
        for geo, val in rasterio.features.shapes(mask.astype("uint8")):
            if val != 1:
                continue
            poly = Polygon(to_world(geo["coordinates"][0]))
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty and (best is None or poly.area > best.area):
                best = poly
        return best

    faces = []
    uncovered = geom
    # a fixed prompt budget starves big buildings: #4726050's whole west wing
    # went unpanelled because fourteen prompts ran out before coverage did
    budget = max(SAM_MAX_PROMPTS, int(geom.area / 120))
    for _ in range(budget):
        if uncovered.is_empty or uncovered.area < 3.0:
            break
        probe = max(getattr(uncovered, "geoms", [uncovered]),
                    key=lambda g: g.area)
        if probe.area < 3.0:
            break
        seed = probe.representative_point()
        mk, sc, _ = predictor.predict(point_coords=px_of(seed),
                                      point_labels=np.array([1]),
                                      multimask_output=True)
        pick = None
        for i in np.argsort(-sc):
            poly = mask_to_poly(mk[i])
            if poly is None:
                continue
            clipped = poly.intersection(geom)
            if clipped.is_empty or clipped.area < 2.0:
                continue
            if clipped.area > 0.85 * geom.area:
                continue
            pick = clipped
            break
        if pick is None:
            uncovered = uncovered.difference(seed.buffer(0.8))
            continue
        fresh = pick.difference(unary_union(faces)) if faces else pick
        for g in getattr(fresh, "geoms", [fresh]):
            if g.geom_type == "Polygon" and g.area >= 2.0:
                faces.append(g)
        uncovered = uncovered.difference(pick.buffer(0.05))

    if not faces:
        return [], []
    polys = _tile(faces, geom, bounds)

    # clutter: small and sitting above what surrounds it
    obs = []
    kept = []
    for poly in polys:
        if poly.area <= CLUTTER_MAX_M2 and pts is not None and len(pts):
            pl, sub = _plane(poly, pts)
            ring = poly.buffer(1.2).difference(poly).intersection(geom)
            _, around = _plane(ring, pts)
            if (pl is not None and len(around) >= 10
                    and float(np.median(sub[:, 2]))
                    - float(np.median(around[:, 2])) > CLUTTER_STEP_M):
                obs.append(poly)
                continue
        kept.append(poly)
    polys = _tile(kept, geom, bounds) if obs else polys

    # merge neighbours whose planes agree
    from src.roof_partition import _slope_aspect as _sa
    changed = True
    while changed and len(polys) > 1:
        changed = False
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                pi, pj = polys[i], polys[j]
                if pi.buffer(0.3).intersection(pj).is_empty:
                    continue
                pl1, _ = _plane(pi, pts)
                pl2, _ = _plane(pj, pts)
                if pl1 is None or pl2 is None:
                    continue
                s1, a1 = _sa(pl1)
                s2, a2 = _sa(pl2)
                da = abs(a1 - a2) % 360
                da = min(da, 360 - da)
                if abs(s1 - s2) < MERGE_SLOPE_DEG and \
                        (da < MERGE_ASPECT_DEG or max(s1, s2) < 4):
                    merged = pi.union(pj).buffer(0.02).buffer(-0.02)
                    if merged.geom_type == "Polygon":
                        polys = ([p for k, p in enumerate(polys)
                                  if k not in (i, j)] + [merged])
                        changed = True
                        break
            if changed:
                break
    # a roof edge is straight; a wavy boundary is mask noise, and Josh reads
    # it as "fuzzy incorrect lines". Simplify harder, and collapse micro-jags
    # by closing/opening before the simplify so staircase pixels go first.
    out_faces = []
    for p2 in polys:
        if p2.area < 2.0:
            continue
        q = p2.buffer(0.15).buffer(-0.15).simplify(0.35)
        if q.geom_type != "Polygon" or q.is_empty:
            q = p2.simplify(0.3)
        out_faces.append(q)
    return out_faces, obs


# ------------------------------------------------------------ line faces

def obstruction_mask(rgb, geom, bounds, pts, h, w):
    """Pixel mask of probable rooftop obstructions, computed BEFORE any
    line reasoning. Josh, 14 Sep: "there can be really clear roof lines
    with a few obstructions and they seem to throw off your roof lines...
    define the simple roof lines first, then try to detect the
    obstructions." The detector cannot tell a duct edge from a ridge, so
    obstruction pixels are silenced in its activation before extraction.
    """
    import numpy as np
    mask = np.zeros((h, w), dtype=bool)
    try:
        from src.roof_partition import _fit_plane_robust, _points_in
        from scipy.spatial import cKDTree
        sub = _points_in(geom, pts) if pts is not None else None
        if sub is None or len(sub) < 60:
            return mask
        pl = _fit_plane_robust(sub)
        if pl is None:
            return mask
        resid = sub[:, 2] - (pl[0] * sub[:, 0] + pl[1] * sub[:, 1] + pl[2])
        high = sub[resid > 0.45]
        if len(high) < 8:
            return mask
        b = bounds
        px = ((high[:, 0] - b[0]) / (b[2] - b[0]) * w).astype(int)
        py = ((1 - (high[:, 1] - b[1]) / (b[3] - b[1])) * h).astype(int)
        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        # each above-plane return silences a small disc around itself
        R = max(3, int(0.6 / max((b[2] - b[0]) / w, 1e-6)))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        disc = (yy ** 2 + xx ** 2) <= R * R
        for x0, y0 in zip(px[ok], py[ok]):
            y1, y2 = max(0, y0 - R), min(h, y0 + R + 1)
            x1, x2 = max(0, x0 - R), min(w, x0 + R + 1)
            mask[y1:y2, x1:x2] |= disc[(y1 - y0 + R):(y2 - y0 + R),
                                       (x1 - x0 + R):(x2 - x0 + R)]
    except Exception:
        pass
    return mask


def lidar_step_channel(pts, bounds, h, w):
    """Cliff evidence measured, not learned. Josh: "It needs to see hips
    and edges/cliffs." A cliff is a height step, and the laser measures
    height steps directly (cliff F1 from imagery alone: 0.143) -- so the
    cliff channel gets the LiDAR's answer OR'd in at extraction, and the
    camera is only asked about what only the camera can see.
    """
    import numpy as np
    out = np.zeros((h, w), dtype=np.float32)
    if pts is None or len(pts) < 60:
        return out
    try:
        from scipy.spatial import cKDTree
        xy = pts[:, :2]
        tree = cKDTree(xy)
        rng = np.random.default_rng(3)
        idx = rng.permutation(len(pts))[:1500]
        b = bounds
        R = max(2, int(0.5 / max((b[2] - b[0]) / w, 1e-6)))
        for i in idx:
            nb = tree.query_ball_point(xy[i], 0.9)
            if len(nb) < 5:
                continue
            z = pts[nb, 2]
            if z.max() - z.min() > 0.8:
                px = int((xy[i, 0] - b[0]) / (b[2] - b[0]) * w)
                py = int((1 - (xy[i, 1] - b[1]) / (b[3] - b[1])) * h)
                if 0 <= px < w and 0 <= py < h:
                    out[max(0, py - R):py + R + 1,
                        max(0, px - R):px + R + 1] = 0.9
    except Exception:
        pass
    return out


def line_faces(line_model, device, rgb, geom, bounds, pts, building_id=0):
    """The line-network reading: detect -> extract -> polygonize -> faces."""
    import torch
    import os
    from src.line_extract import extract, clip_to
    from src.roof_partition import line_facets

    h, w = rgb.shape[:2]
    b = bounds
    ph, pw = (-h) % 16, (-w) % 16
    arr = np.pad(rgb, ((0, ph), (0, pw), (0, 0)))
    x = torch.from_numpy(arr).float().permute(2, 0, 1)[None] / 255.0
    with torch.no_grad():
        pr = torch.sigmoid(line_model(x.to(device)))[0].cpu().numpy()[:, :h, :w]
    # MEASURED DEAD END (14 Sep): silencing above-plane pixels before
    # extraction dropped LINE agreement 0.754 -> 0.578 on the drawn
    # benchmark -- a pitched roof's RIDGE is above its single robust
    # plane, so the mask killed real ridges along with ducts. Obstructions
    # are only definable relative to structure (see hypothesis_faces);
    # the flag stays for re-testing with per-face planes, default off.
    # (LiDAR height-steps were briefly OR'd into the cliff channel HERE --
    # measured: LINE agreement collapsed 0.754 -> 0.42, because the fat step
    # blobs feed the THINNING stage and skeletonise into junk lines along
    # every eave. Step evidence belongs to the scorer's height-step term
    # and the evidence map, never to the extractor's input.)
    if os.environ.get("SOLAR_MASK_OBSTRUCTIONS", "0") == "1":
        m = obstruction_mask(rgb, geom, bounds, pts, h, w)
        if m.any():
            pr = pr.copy()
            pr[:, m] = 0.0

    def tw(px, py):
        return (b[0] + px / w * (b[2] - b[0]),
                b[1] + (1 - py / h) * (b[3] - b[1]))

    segs = network_lines(pr, geom, b, w, h, pts)
    if not segs:
        return [], pr
    facets = line_facets(building_id, geom, pts, segs) or []
    return [f["geometry"] for f in facets], pr


# --------------------------------------------------------------- scorer

def lidar_faces(pts, geom):
    """LiDAR-first reading: normal-based region growing on the point cloud,
    regularised to the building's axes, tiled to the footprint.

    Why a third family: the two imagery families go blind exactly where Josh
    kept flagging failures -- tree shadow (#4735106) and cluttered flats
    (#4735244). Shadows and clutter do not exist in the point cloud. Greedy
    RANSAC was measured absorbing small raised faces into neighbouring big
    planes (2 Kent St: 4 faces where Josh counts 13-14); growing regions by
    NORMAL agreement is the fix the audit named.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from shapely.geometry import Point, Polygon, MultiPoint
    from shapely.ops import unary_union, voronoi_diagram

    if pts is None or len(pts) < 60:
        return []
    xy = pts[:, :2]
    tree = cKDTree(xy)
    n = len(pts)

    # per-point unit normals + curvature by local PCA (k nearest)
    k = min(12, n - 1)
    _, idx = tree.query(xy, k=k + 1)
    normals = np.zeros((n, 3))
    curv = np.full(n, 1.0)
    P3 = pts[:, :3]
    for i in range(n):
        nb = P3[idx[i]]
        c = nb - nb.mean(axis=0)
        cov = c.T @ c
        w, v = np.linalg.eigh(cov)
        nv = v[:, 0]
        if nv[2] < 0:
            nv = -nv
        normals[i] = nv
        tot = w.sum()
        curv[i] = w[0] / tot if tot > 0 else 1.0

    # region growing: flattest seeds first, admit a neighbour when its normal
    # agrees with the REGION's running mean and it sits on the region's plane
    ANG = np.cos(np.radians(12.0))
    DIST = 0.13
    label = np.full(n, -1)
    order = np.argsort(curv)
    region_id = 0
    for seed in order:
        if label[seed] >= 0:
            continue
        stack = [seed]
        label[seed] = region_id
        members = [seed]
        nsum = normals[seed].copy()
        csum = P3[seed].copy()
        while stack:
            i = stack.pop()
            for j in tree.query_ball_point(xy[i], 1.1):
                if label[j] >= 0:
                    continue
                m = nsum / np.linalg.norm(nsum)
                if normals[j] @ m < ANG:
                    continue
                cen = csum / len(members)
                if abs((P3[j] - cen) @ m) > DIST:
                    continue
                label[j] = region_id
                members.append(j)
                nsum += normals[j]
                csum += P3[j]
                stack.append(j)
        region_id += 1

    # regions -> footprint-tiled polygons: every point claims its cell of a
    # dense grid Voronoi (cheap: nearest-labelled-point lookup on a raster)
    minx, miny, maxx, maxy = geom.bounds
    step = 0.4
    gx = np.arange(minx - 0.2, maxx + 0.2, step)
    gy = np.arange(miny - 0.2, maxy + 0.2, step)
    GX, GY = np.meshgrid(gx, gy)
    q = np.c_[GX.ravel(), GY.ravel()]
    _, nearest = tree.query(q)
    cell_label = label[nearest].reshape(GY.shape)

    sizes = np.bincount(label[label >= 0])
    keep = {r for r in range(region_id) if sizes[r] >= 25}
    # EQUIPMENT IS NOT ROOF. Region growing is too good at ducting: the
    # reference roof (#5370338, 223 m2 of plant) came back as tidy duct-top
    # "faces" that won selection. Two dead ends are recorded here so they
    # are not retried: (a) "small region above the MAIN plane" -- a hip
    # face across the ridge is above the main face's extended plane too,
    # and it cost #4735106 two real faces; (b) border-cell z jumps -- the
    # point-Voronoi raster smooths borders to ~0.00 m (measured), walls
    # donate their z to both sides. What separates plant is REGION MEDIANS:
    # measured on the reference roof, ducts sit +0.65-0.8 m above the
    # membrane region that dominates their border, storeys sit 2.5 m+
    # away. Flag a region whose dominant neighbour is 0.35-1.8 m below it.
    equipment = set()
    if keep and len(keep) > 1:
        zmed = {r2: float(np.median(P3[label == r2, 2])) for r2 in keep}
        neigh = {r2: {} for r2 in keep}
        for dy2, dx2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = np.roll(cell_label, (dy2, dx2), axis=(0, 1))
            for r2 in keep:
                edge = (cell_label == r2) & (nb != r2) & (nb >= 0)
                if edge.any():
                    vals, cnts = np.unique(nb[edge], return_counts=True)
                    for v2, c2 in zip(vals, cnts):
                        if v2 in keep:
                            neigh[r2][v2] = neigh[r2].get(v2, 0) + int(c2)
        # ...and only between near-FLAT pairs: on a pitched roof two faces'
        # medians differ by slope geometry, not by a step (this test at its
        # first cut took #4735106's lidar reading from 0.48 to 0.08 by
        # flagging its hip faces). Plant sits on flat membrane.
        def _slope_of(r2):
            q3 = P3[label == r2]
            if len(q3) < 20:
                return 99.0
            A2 = np.c_[q3[:, 0], q3[:, 1], np.ones(len(q3))]
            try:
                coef, *_ = np.linalg.lstsq(A2, q3[:, 2], rcond=None)
            except Exception:
                return 99.0
            return float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
        slope_c = {r2: _slope_of(r2) for r2 in keep}
        for r2 in list(keep):
            if sizes[r2] > 700 or not neigh[r2]:
                continue
            dom = max(neigh[r2], key=neigh[r2].get)
            dz = zmed[r2] - zmed[dom]
            # the region's OWN slope says nothing (half-round duct tops fit
            # 8-20 degree planes and slipped through); what matters is what
            # it stands ON: plant stands on flat membrane, a hip triangle's
            # dominant neighbour is itself pitched.
            if 0.35 < dz < 1.8 and slope_c[dom] < 8.0:
                equipment.add(r2)
        keep -= equipment

    # small regions dissolve into their biggest neighbour at the raster level
    if keep:
        big = max(keep, key=lambda r: sizes[r])
        flat = cell_label.ravel()
        flat[np.isin(flat, list(equipment))] = -1   # holes stay holes
        flat[~np.isin(flat, list(keep) + [-1])] = -9
        # nearest kept label for dissolved cells
        for _ in range(3):
            m2 = flat.reshape(cell_label.shape)
            for dy in (-1, 1):
                roll = np.roll(m2, dy, axis=0)
                m2[(m2 == -9) & (roll != -9)] = roll[(m2 == -9) & (roll != -9)]
            for dx in (-1, 1):
                roll = np.roll(m2, dx, axis=1)
                m2[(m2 == -9) & (roll != -9)] = roll[(m2 == -9) & (roll != -9)]
        cell_label = m2
        cell_label[cell_label == -9] = big

    # dominant axes of the footprint for boundary regularisation
    mrr = geom.minimum_rotated_rectangle
    cc = list(mrr.exterior.coords)
    ax = np.arctan2(cc[1][1] - cc[0][1], cc[1][0] - cc[0][0])

    faces = []
    for r in sorted(set(cell_label.ravel())):
        if r < 0:
            continue
        mask = cell_label == r
        if mask.sum() < 12:
            continue
        boxes = []
        ys, xs = np.nonzero(mask)
        for yy, xx in zip(ys, xs):
            boxes.append(Polygon([
                (gx[xx] - step / 2, gy[yy] - step / 2),
                (gx[xx] + step / 2, gy[yy] - step / 2),
                (gx[xx] + step / 2, gy[yy] + step / 2),
                (gx[xx] - step / 2, gy[yy] + step / 2)]))
        poly = unary_union(boxes).buffer(step * 0.51).buffer(-step * 0.51)
        poly = poly.intersection(geom)
        if poly.is_empty:
            continue
        parts = list(getattr(poly, "geoms", [poly]))
        for part in parts:
            if part.geom_type != "Polygon" or part.area < 6.0:
                continue
            # regularise: rotate into the building frame, simplify with a
            # coarse tolerance there (axis-parallel jags collapse), rotate back
            import shapely.affinity as aff
            rot = aff.rotate(part, -np.degrees(ax), origin=(0, 0))
            rot = rot.simplify(0.55)
            # axis-snap: raster stair-steps survive simplify as short jogs;
            # in the building frame a nearly-axis-parallel edge IS axis
            # parallel, so collapse runs of nearly-equal x (or y) vertices
            if rot.geom_type == "Polygon":
                cs = list(rot.exterior.coords)[:-1]
                snapped = []
                for i2 in range(len(cs)):
                    x0, y0 = cs[i2]
                    xp, yp = cs[i2 - 1]
                    if abs(x0 - xp) < 0.5:
                        x0 = (x0 + xp) / 2
                        if snapped:
                            snapped[-1] = (x0, snapped[-1][1])
                    if abs(y0 - yp) < 0.5:
                        y0 = (y0 + yp) / 2
                        if snapped:
                            snapped[-1] = (snapped[-1][0], y0)
                    snapped.append((x0, y0))
                if len(snapped) >= 3:
                    cand2 = Polygon(snapped)
                    if cand2.is_valid and cand2.area > 0.7 * rot.area:
                        rot = cand2.simplify(0.35)
            reg = aff.rotate(rot, np.degrees(ax), origin=(0, 0))
            if not reg.is_valid or reg.is_empty or reg.geom_type != "Polygon":
                reg = part.simplify(0.3)
            if reg.geom_type == "Polygon" and reg.area >= 6.0:
                faces.append(reg)
    return faces




def _lidar_plateau(part, pts, cell=0.75, step=0.35, min_area=4.0,
                   want_step=False):
    """A raised flat region inside `part`, or None.

    Finds the thing a form vocabulary cannot guess: where a roof stops
    rising and goes flat. A plateau announces itself as a STEP -- cells
    sitting well above the local background with a sharp edge -- so that is
    what is measured, rather than "high" (a pyramid's centre is always
    high) or "flat" (a whole flat roof is).

    Written for #4735292, where the flat top is 4 m across and 0.9 m above
    the roof around it, and both the truncated form (which put it in the
    wrong place) and the plain pyramid (which denied it) were wrong.
    """
    import numpy as np
    from shapely.geometry import Polygon, MultiPoint
    from src.roof_partition import _points_in
    _none = (None, None) if want_step else None
    sub = _points_in(part, pts)
    if sub is None or len(sub) < 120:
        return _none
    minx, miny, maxx, maxy = part.bounds
    nx = int((maxx - minx) / cell) + 1
    ny = int((maxy - miny) / cell) + 1
    if nx < 5 or ny < 5 or nx * ny > 20000:
        return _none
    ix = ((sub[:, 0] - minx) / cell).astype(int).clip(0, nx - 1)
    iy = ((sub[:, 1] - miny) / cell).astype(int).clip(0, ny - 1)
    zg = np.full((ny, nx), np.nan)
    order = np.argsort(sub[:, 2])
    for k in order:                      # median-ish: last write is highest
        zg[iy[k], ix[k]] = sub[k, 2]
    # local background: the lower quartile of a generous neighbourhood, so
    # the plateau cannot raise its own baseline
    from scipy.ndimage import generic_filter, label
    rad = max(2, int(2.5 / cell))
    filled = np.where(np.isnan(zg), np.nanmin(zg), zg)
    bg = generic_filter(filled, lambda w: np.percentile(w, 25),
                        size=2 * rad + 1, mode="nearest")
    raised = (zg - bg > step) & ~np.isnan(zg)
    if raised.sum() < max(4, int(min_area / (cell * cell))):
        return _none
    lab, n = label(raised)
    if n == 0:
        return _none
    sizes = [(lab == i).sum() for i in range(1, n + 1)]
    best = int(np.argmax(sizes)) + 1
    if sizes[best - 1] * cell * cell < min_area:
        return _none
    ys, xs = np.nonzero(lab == best)
    pts_xy = [(minx + (x + 0.5) * cell, miny + (y + 0.5) * cell)
              for x, y in zip(xs, ys)]
    if len(pts_xy) < 3:
        return _none
    hull = MultiPoint(pts_xy).convex_hull
    if hull.geom_type != "Polygon" or hull.area < min_area:
        return _none
    if want_step:
        med_step = float(np.nanmedian((zg - bg)[lab == best]))
        return hull, med_step
    return hull


def hypothesis_faces(pts, geom, prob_max, to_px):
    """STRUCTURE FIRST: test simple parametric roof forms against the
    evidence, instead of assembling detections into shapes.

    Josh, 14 Sep, after weeks of bottom-up failures on visually obvious
    roofs: "Roofs are simpler shapes... define the simple roof lines
    first, then try to detect the obstructions." Bottom-up compounds
    errors -- fragments, guessed archetypes, obstruction edges read as
    ridges. Here the vocabulary is closed: per rectangular part, the roof
    is FLAT, a GABLE, a HIP, or a PYRAMID; each form's predicted seams
    and faces are scored jointly on detector activation and LiDAR plane
    fit, with complexity paying rent (a gable must beat flat by a real
    margin, a hip must beat a gable). Obstructions cannot create a false
    ridge because ridges exist only in the vocabulary.
    """
    import numpy as np
    from shapely.geometry import Polygon, LineString
    from shapely.ops import unary_union
    from src.roof_partition import _fit_plane_robust, _points_in
    from src.line_extract import _line_mean

    import os as _os
    if pts is None or len(pts) < 40:
        return []

    def seam_support(a, b):
        pa = np.array(to_px(*a)); pb = np.array(to_px(*b))
        return _line_mean(prob_max, pa, pb)

    def tilt_field(part):
        """Local downhill direction per ~2 m cell: the LiDAR's own tilt
        field, compared against each form's PREDICTION at every cell."""
        from scipy.spatial import cKDTree
        from shapely.geometry import Point
        sub = _points_in(part, pts)
        if len(sub) < 60:
            return []
        tree = cKDTree(sub[:, :2])
        minx, miny, maxx, maxy = part.bounds
        cells = []
        step = 2.0
        y = miny + step / 2
        while y < maxy:
            x = minx + step / 2
            while x < maxx:
                # (interior-only sampling was measured 14 Sep: it fixed
                # the mono-pitch roofs and re-broke #4735623 -- the referee
                # oscillates +-0.02 around 0.62 across ten variants. The
                # aggregate is knob-tuning noise at this point; the next
                # real gain is the pretraining arc, not an eleventh knob.)
                if part.contains(Point(x, y)):
                    nb = tree.query_ball_point([x, y], 1.6)
                    if len(nb) >= 10:
                        q = sub[nb]
                        Amat = np.c_[q[:, 0], q[:, 1], np.ones(len(q))]
                        try:
                            coef, *_ = np.linalg.lstsq(Amat, q[:, 2],
                                                       rcond=None)
                            gn = float(np.hypot(coef[0], coef[1]))
                            if gn > 0.05:
                                cells.append((x, y, -coef[0] / gn,
                                              -coef[1] / gn, gn))
                        except Exception:
                            pass
                x += step
            y += step
        return cells

    def face_plane(poly):
        """(inlier, downhill_unit_or_None, slope_deg) for one face."""
        sub = _points_in(poly, pts)
        if len(sub) < 15:
            return 0.5, None, 0.0
        pl = _fit_plane_robust(sub)
        if pl is None:
            return 0.0, None, 0.0
        r = sub[:, 2] - (pl[0] * sub[:, 0] + pl[1] * sub[:, 1] + pl[2])
        inl = float((np.abs(r) < 0.18).mean())
        gnorm = float(np.hypot(pl[0], pl[1]))
        slope = float(np.degrees(np.arctan(gnorm)))
        down = (np.array([-pl[0], -pl[1]]) / gnorm) if gnorm > 1e-6 else None
        return inl, down, slope

    def forms_for(part):
        mrr = part.minimum_rotated_rectangle
        cc = list(mrr.exterior.coords)[:4]
        e = [(np.hypot(cc[(k + 1) % 4][0] - cc[k][0],
                       cc[(k + 1) % 4][1] - cc[k][1]), k) for k in range(4)]
        e.sort(reverse=True)
        L, k0 = e[0]
        Wd = e[2][0] if len(e) > 2 else e[1][0]
        A = np.array(cc[k0]); B = np.array(cc[(k0 + 1) % 4])
        D = np.array(cc[(k0 + 3) % 4])
        u = (B - A) / max(np.linalg.norm(B - A), 1e-9)
        v = (D - A) / max(np.linalg.norm(D - A), 1e-9)
        out = {"flat": ([part], [])}
        # GABLE: ridge full length. Real gables are asymmetric, so the
        # ridge SLIDES across the width and keeps the offset where seam
        # activation peaks (v1 fixed it at the midline and scored 0.5-0.6
        # on his simple gables while the incumbents hit 0.85+).
        best_off, best_sup = 0.5, -1.0
        for t in (0.3, 0.38, 0.44, 0.5, 0.56, 0.62, 0.7):
            ra = A + v * (Wd * t); rb = ra + u * L
            sup = seam_support(tuple(ra), tuple(rb))
            if sup > best_sup:
                best_sup, best_off = sup, t
        mid_a = A + v * (Wd * best_off)
        mid_b = mid_a + u * L
        half1 = Polygon([A, A + u * L, mid_b, mid_a])
        half2 = Polygon([mid_a, mid_b, D + u * L, D])
        g1 = half1.intersection(part); g2 = half2.intersection(part)
        if g1.geom_type == "Polygon" and g2.geom_type == "Polygon" \
                and g1.area > 4 and g2.area > 4:
            out["gable"] = ([g1, g2], [(tuple(mid_a), tuple(mid_b))])
        # GABLE ACROSS: some parts pitch along the short axis
        sa = A + u * (L * 0.5)
        sb = sa + v * Wd
        c_half1 = Polygon([A, sa, sb, D])
        c_half2 = Polygon([sa, A + u * L, D + u * L, sb])
        x1 = c_half1.intersection(part); x2 = c_half2.intersection(part)
        if x1.geom_type == "Polygon" and x2.geom_type == "Polygon" \
                and x1.area > 4 and x2.area > 4:
            out["gable_x"] = ([x1, x2], [(tuple(sa), tuple(sb))])
        # HIP with a RIDGE BAND. Josh's #4735623 gold: hips at the ends
        # and a THIN flat band along the ridge (a clerestory strip). Pure
        # hip (band 0) and truncated hip are endpoints of one family; the
        # band width is chosen by evidence like the gable ridge offset.
        # "You are meant to be recognising this" -- the form existed only
        # at the endpoints, so the 1.5 m band matched neither and a plain
        # gable won the vote.
        if L > Wd * 1.15:
            c1, c2, c3, c4 = A, A + u * L, D + u * L, D
            for wb in (0.0, 1.6, 2.6):
                if wb >= Wd * 0.6:
                    continue
                half = (Wd - wb) / 2
                r1a = A + v * half + u * half
                r1b = A + v * half + u * (L - half)
                r2a = r1a + v * wb
                r2b = r1b + v * wb
                fA = Polygon([c1, c2, r1b, r1a])
                fB = Polygon([c4, c3, r2b, r2a])
                if wb > 0.1:
                    fC = Polygon([c1, r1a, r2a, c4])
                    fD = Polygon([c2, r1b, r2b, c3])
                    band = Polygon([r1a, r1b, r2b, r2a])
                    fam = [fA, fB, fC, fD, band]
                    seams = [(tuple(r1a), tuple(r1b)),
                             (tuple(r2a), tuple(r2b)),
                             (tuple(c1), tuple(r1a)), (tuple(c4), tuple(r2a)),
                             (tuple(c2), tuple(r1b)), (tuple(c3), tuple(r2b))]
                    name = f"hip_band{wb:g}"
                else:
                    fC = Polygon([c1, r1a, c4])
                    fD = Polygon([c2, r1b, c3])
                    fam = [fA, fB, fC, fD]
                    seams = [(tuple(r1a), tuple(r1b)),
                             (tuple(c1), tuple(r1a)), (tuple(c4), tuple(r1a)),
                             (tuple(c2), tuple(r1b)), (tuple(c3), tuple(r1b))]
                    name = "hip"
                faces = [f.intersection(part) for f in fam]
                if all(f.geom_type == "Polygon" and f.area > 1.5
                       for f in faces):
                    out[name] = (faces, seams)
        # TRUNCATED HIP: sloped skirt around a flat top -- the form both
        # #4735623 and #5371107 actually are (Josh's gold overlay made it
        # unmissable: perimeter hips, recessed centre). Inset slides like
        # the gable ridge does: the top edge sits where seam evidence
        # peaks.
        c1, c2, c3, c4 = A, A + u * L, D + u * L, D
        best_t, best_ts = None, -1.0
        for t in (0.22, 0.30, 0.38):
            d0 = min(Wd, L) * t
            i1 = c1 + u * d0 + v * d0
            i2 = c2 - u * d0 + v * d0
            i3 = c3 - u * d0 - v * d0
            i4 = c4 + u * d0 - v * d0
            sup4 = np.mean([seam_support(tuple(i1), tuple(i2)),
                            seam_support(tuple(i2), tuple(i3)),
                            seam_support(tuple(i3), tuple(i4)),
                            seam_support(tuple(i4), tuple(i1))])
            if sup4 > best_ts:
                best_ts, best_t = sup4, t
        d0 = min(Wd, L) * best_t
        if d0 > 1.2:
            i1 = c1 + u * d0 + v * d0
            i2 = c2 - u * d0 + v * d0
            i3 = c3 - u * d0 - v * d0
            i4 = c4 + u * d0 - v * d0
            top = Polygon([i1, i2, i3, i4])
            skirts = [Polygon([c1, c2, i2, i1]),
                      Polygon([c2, c3, i3, i2]),
                      Polygon([c3, c4, i4, i3]),
                      Polygon([c4, c1, i1, i4])]
            faces = [top.intersection(part)] + \
                [f.intersection(part) for f in skirts]
            if all(f.geom_type == "Polygon" and f.area > 3 for f in faces):
                seams = [(tuple(i1), tuple(i2)), (tuple(i2), tuple(i3)),
                         (tuple(i3), tuple(i4)), (tuple(i4), tuple(i1))]
                out["trunc_hip"] = (faces, seams)
        # THE FLAT TOP WHERE THE LIDAR ACTUALLY PUTS IT. The form above
        # insets the same distance from all four sides, so its top is
        # always centred, and snap_form then slides it on a guess-grid --
        # which is how #4735292 got a box carved into one slope. Josh, on
        # the fourth time he raised that roof: "you are missing the square
        # in the middle at the top. It ends in a square at the peak which
        # should be visible in both imagery and lidar." It is: a plateau of
        # cells 0.9 m above the surrounding roof with a sharp step at its
        # edge. So MEASURE the top rather than sweeping for it -- per-side
        # insets read off the plateau, which is the same family, placed by
        # evidence instead of by symmetry.
        plat = _lidar_plateau(part, pts)
        if plat is not None:
            pu = [np.dot(np.array(q) - A, u) for q in plat.exterior.coords]
            pv = [np.dot(np.array(q) - A, v) for q in plat.exterior.coords]
            lo_u, hi_u = max(0.6, min(pu)), min(L - 0.6, max(pu))
            lo_v, hi_v = max(0.6, min(pv)), min(Wd - 0.6, max(pv))
            if hi_u - lo_u > 1.2 and hi_v - lo_v > 1.2:
                j1 = A + u * lo_u + v * lo_v
                j2 = A + u * hi_u + v * lo_v
                j3 = A + u * hi_u + v * hi_v
                j4 = A + u * lo_u + v * hi_v
                top = Polygon([j1, j2, j3, j4])
                sk = [Polygon([c1, c2, j2, j1]), Polygon([c2, c3, j3, j2]),
                      Polygon([c3, c4, j4, j3]), Polygon([c4, c1, j1, j4])]
                faces = [top.intersection(part)] + \
                    [f.intersection(part) for f in sk]
                if all(f.geom_type == "Polygon" and f.area > 2 for f in faces):
                    out["trunc_lidar"] = (faces, [
                        (tuple(j1), tuple(j2)), (tuple(j2), tuple(j3)),
                        (tuple(j3), tuple(j4)), (tuple(j4), tuple(j1))])
        if not (L > Wd * 1.15):
            # PYRAMID on square-ish parts
            ctr = (A + u * L / 2 + v * Wd / 2)
            c1, c2, c3, c4 = A, A + u * L, D + u * L, D
            faces = [Polygon([c1, c2, ctr]).intersection(part),
                     Polygon([c2, c3, ctr]).intersection(part),
                     Polygon([c3, c4, ctr]).intersection(part),
                     Polygon([c4, c1, ctr]).intersection(part)]
            if all(f.geom_type == "Polygon" and f.area > 3 for f in faces):
                seams = [(tuple(c1), tuple(ctr)), (tuple(c2), tuple(ctr)),
                         (tuple(c3), tuple(ctr)), (tuple(c4), tuple(ctr))]
                out["pyramid"] = (faces, seams)
        return out

    try:
        parts = [q for q in rect_parts(geom) if q.area > 15]
    except Exception:
        parts = []
    # Josh on the residential panel: "overly simplified. Missing some
    # clearly defined faces." An attached wing or garage is 8-11% of the
    # footprint, so the substantial-parts filter absorbed it into the main
    # blob and its faces never existed. At residential scale every
    # rectangular part >= 16 m2 gets its own form.
    big = [q for q in parts if q.area >= 16] if geom.area <= 450 else \
        [q for q in parts if q.area > 0.12 * geom.area]
    blobs = big if len(big) >= 2 else \
        ([geom] if geom.geom_type == "Polygon" else list(geom.geoms))

    COMPLEXITY_RENT = {"flat": 0.0, "gable": 0.045, "gable_x": 0.045,
                      "hip": 0.09, "pyramid": 0.09, "skeleton": 0.07,
                      "trunc_hip": 0.10, "trunc_lidar": 0.10, "hip_band1.6": 0.10,
                      "hip_band2.6": 0.10}
    # THE SKELETON FORM. Josh's 3-6 face tier (his villas: hips with
    # valleys wrapping the wings) measured 0.14-0.40 because no simple
    # form has an L-hip. The constructive straight-skeleton builder from
    # the August arc generates exactly that network from the outline; it
    # competes on the WHOLE footprint against the per-part assembly.
    skel_faces = []
    try:
        from src.roof_skeleton import skeleton_roof
        # its internal envelope-fit gate (0.55) was tuned to refuse
        # borderline roofs outright; HERE the contest judges, so the gate
        # opens wide (0.10) and a bad skeleton simply loses on evidence
        sk = skeleton_roof(0, geom, pts, min_envelope_fit=0.10)
        skel_faces = [f["geometry"] for f in sk
                      if f.get("geometry") is not None
                      and f["geometry"].geom_type == "Polygon"
                      and f["geometry"].area > 3]
    except Exception as _e:
        import os as _os
        if _os.environ.get("SOLAR_DEBUG_HYP"):
            print("  skel gen failed:", repr(_e)[:100], flush=True)
        skel_faces = []
    import os as _os
    if _os.environ.get("SOLAR_DEBUG_HYP"):
        print(f"  skel_faces: {len(skel_faces)}", flush=True)
    out_faces = []
    confs = []
    for blob in blobs:
        best_name, best_sc, best_faces = None, -1e9, None
        cells = tilt_field(blob)
        from shapely.geometry import LineString as _LS, Point as _Pt
        from shapely.ops import unary_union as _uu2

        def snap_form(faces, seams):
            """Slide the form's INNER skeleton (every non-eave vertex) as a
            group to the evidence peak. The parametric seams sit at ideal
            MRR positions; the real ridge/hips are often offset a metre or
            two, so the hip evidence v6 finally sees went unclaimed (sup
            0.20 along the ideal lines vs 0.74 along Josh's drawn hips)."""
            if not seams:
                return faces, seams, 0.0
            best = (faces, seams, np.mean([seam_support(a, b)
                                           for a, b in seams]))
            eave_pts = set()
            for f in faces:
                for c in f.exterior.coords:
                    eave_pts.add((round(c[0], 2), round(c[1], 2)))
            inner = set()
            for a, b in seams:
                inner.add(a); inner.add(b)
            # inner nodes = seam endpoints not on the part boundary
            for du in (-1.2, -0.6, 0.0, 0.6, 1.2):
                for dv in (-1.2, -0.6, 0.0, 0.6, 1.2):
                    if du == 0 and dv == 0:
                        continue
                    off = np.array([du, dv])
                    def mv(pt):
                        p2 = np.array(pt)
                        if blob.exterior.distance(_Pt(*pt)) < 0.8:
                            return tuple(pt)   # eave-attached ends stay
                        return tuple(p2 + off)
                    s2 = [(mv(a), mv(b)) for a, b in seams]
                    sup2 = np.mean([seam_support(a, b) for a, b in s2])
                    if sup2 > best[2] + 0.02:
                        f2 = []
                        ok = True
                        for f in faces:
                            ring = [mv(c) for c in list(f.exterior.coords)[:-1]]
                            from shapely.geometry import Polygon as _Pg
                            g2 = _Pg(ring)
                            if not g2.is_valid or g2.is_empty:
                                ok = False
                                break
                            f2.append(g2)
                        if ok:
                            best = (f2, s2, sup2)
            return best

        _plat_ref = list(_lidar_plateau(blob, pts, want_step=True))
        for name, (faces, seams) in forms_for(blob).items():
            # DO NOT SNAP A MEASURED FORM. snap_form slides inner vertices
            # on a +-1.2 m grid looking for seam evidence, which is right
            # for a form whose position was guessed and wrong for one whose
            # position was read off the LiDAR: it slid the plateau-placed
            # top straight off the plateau it was built on, and the form
            # then lost to a pyramid for splitting the very feature it was
            # holding. This is the same slide that put the box on the wrong
            # side of #4735292 in the first place.
            if name != "trunc_lidar":
                faces, seams, _snapped_sup = snap_form(faces, seams)
            ridge = _uu2([_LS([a, b]) for a, b in seams]) if seams else None
            pl_num = pl_den = 0.0
            for f in faces:
                inl, down, slope = face_plane(f)
                pl_num += f.area * inl
                pl_den += f.area
            plane = pl_num / max(pl_den, 1e-9)
            # FIELD ASPECT -- per-face means hid the disagreement: on
            # #4735623 a gable scored aspect 1.00 over a hip-band roof
            # because its two big faces average away the end zones where
            # the LiDAR tilts ENDWAYS. Every pitched ~2 m cell votes: does
            # the local tilt match what this form predicts at that cell
            # (away from the ridge of whichever face holds it)?
            agree_sum = 0.0
            agree_n = 0
            # WATER RUNS TO THE EAVE. The prediction for a cell is the
            # direction toward its own face's EAVE -- the part of that
            # face's boundary lying on the building outline -- because
            # that is what every form in this vocabulary actually claims.
            #
            # It used to be "away from the nearest seam", which is a
            # geometric artifact, not a claim: a pyramid's seams are the
            # four corner diagonals, so "away from the seam" points 45
            # degrees off the true downhill and a PERFECTLY CORRECT
            # pyramid scored ~0.78, while a truncated hip -- whose flat
            # top's seams run parallel to the eaves -- scored 0.91 for
            # inventing a flat top that is not there. #4735292 (36 Stanley
            # St, the square pyramid Josh has flagged repeatedly) lost
            # 0.733 to 0.689 on exactly this and shipped with a box
            # carved into one slope. Josh: "you are inventing places to
            # put lines that are clearly not right in the visual imagery."
            eaves = []
            _rim = geom.exterior.buffer(0.6)
            for f in faces:
                try:
                    ev = f.exterior.intersection(_rim)
                    eaves.append(ev if (not ev.is_empty and ev.length > 0.8)
                                 else None)
                except Exception:
                    eaves.append(None)
            if cells:
                for (cx, cy, dx, dy, _gn) in cells:
                    pt = _Pt(cx, cy)
                    hi = None
                    for k, f in enumerate(faces):
                        if f.contains(pt):
                            hi = k
                            break
                    if hi is None:
                        continue
                    ev = eaves[hi]
                    if ev is None:
                        # An INTERIOR face claims flat (the flat top of a
                        # truncated hip, a hip band). Cells only exist
                        # where the LiDAR tilts at all, so every cell here
                        # is evidence against that claim -- and a claim
                        # that cannot be contradicted is not evidence.
                        agree_sum += -min(1.0, _gn / 0.09)
                        agree_n += 1
                        continue
                    npt = ev.interpolate(ev.project(pt))
                    d0 = np.array([npt.x - cx, npt.y - cy])
                    nd = np.linalg.norm(d0)
                    if nd < 0.3:
                        continue
                    # UNCLIPPED: a wrong prediction must cost. Clipped at
                    # zero, a gable claiming a hip's end zones paid nothing
                    # for predicting sideways where the roof tilts endways,
                    # and the complexity rent then decided AGAINST the true
                    # form on both #4735623 and #5371107.
                    agree_sum += float(dx * d0[0] / nd + dy * d0[1] / nd)
                    agree_n += 1
            aspect = ((agree_sum / agree_n + 1) / 2) if agree_n >= 8 else 0.5
            # seams CONFIRM; absence in a hip-blind detector must not
            # DENY (mean over all seams drowned true hips at 0.16-0.18
            # while a gable's one lucky ridge scored 0.61): mean of the
            # best half.
            if seams:
                ss = sorted((seam_support(a, b) for a, b in seams),
                            reverse=True)
                sup = float(np.mean(ss[:max(1, len(ss) // 2)]))
            else:
                sup = 0.0
            # A SEAM THE LIDAR MEASURED IS SUPPORTED, whatever the camera
            # can see. seam_support asks the imagery detector, and from
            # straight above a 0.9 m vertical step often shows as nothing
            # at all -- on #4735292 the plateau-placed form scored best of
            # every form on BOTH LiDAR terms (plane 0.99, aspect 0.96) and
            # was then held down to 0.25 on imagery support for a feature
            # the imagery cannot resolve. Exactly the trap that hid hips.
            # The step height IS the evidence, so it sets a floor.
            if name == "trunc_lidar" and _plat_ref[1] is not None:
                sup = max(sup, min(1.0, _plat_ref[1] / 0.6))
            # rent is charged on UNPROVEN complexity only ("complexity
            # earned, not assumed"): a hip whose seams sit on 0.86 evidence
            # has earned its faces; one with silent seams pays full price.
            # Flat constants were the deciding wrong vote on #4735623 and
            # #5371107 after snapping (true forms led pre-rent on both).
            sc = 0.30 * plane + 0.45 * aspect + 0.15 * sup \
                - COMPLEXITY_RENT[name] * (1.0 - min(1.0, sup))
            # A MEASURED PLATEAU IS ONE SURFACE. Where the LiDAR shows a
            # flat top with a step at its edge, a form that runs four
            # sloping faces through it is contradicted by the data, and
            # until now nothing said so: on #4735292 the plain pyramid
            # (which denies the square Josh kept pointing at) and the
            # plateau-placed form scored 0.761 against 0.759, a tie broken
            # by nothing. The penalty is proportional to how badly the form
            # splits it, so a form that holds the plateau in one face pays
            # nothing and only a form that carves it up is charged.
            if _plat_ref[0] is not None:
                pl_poly = _plat_ref[0]
                if pl_poly.area > 1e-6:
                    share = max((f.intersection(pl_poly).area
                                 for f in faces), default=0.0) / pl_poly.area
                    sc -= 0.18 * max(0.0, 0.85 - share)
            if name == "flat":
                # a ONE-FACE claim is "one tilt direction (or none)": score
                # it by tilt-field COHERENCE. The horizontal-only bonus
                # collapsed every mono-pitch single-face roof Josh drew
                # (three 1.00 -> 0.50 regressions in one measurement):
                # a shed roof is one face with a uniform tilt, not "flat".
                if len(cells) >= 8:
                    # magnitude-weighted resultant: near-threshold cells on
                    # a 3-degree roof have noise directions and were
                    # scattering the field (#5370377, one drawn face, lost
                    # to a gable_x on incoherence that was pure noise)
                    wsum = sum(c[4] for c in cells) or 1e-9
                    vx = sum(c[2] * c[4] for c in cells) / wsum
                    vy = sum(c[3] * c[4] for c in cells) / wsum
                    coherence = min(1.0, float(np.hypot(vx, vy)))
                else:
                    coherence = 1.0   # no measurable tilt: flat is right
                sc = 0.30 * plane + 0.45 * coherence + 0.15 * sup
            if _os.environ.get("SOLAR_DEBUG_HYP"):
                print(f"    form {name:11s} pl {plane:.2f} asp {aspect:.2f} "
                      f"sup {sup:.2f} -> {sc:.3f}", flush=True)
            if sc > best_sc:
                best_sc, best_name, best_faces = sc, name, faces
        if best_faces:
            out_faces.extend(best_faces)
            confs.append(best_sc)
    part_conf = min(confs) if confs else 0.0
    # whole-building skeleton vs per-part assembly -- judged with the SAME
    # terms (plane + aspect + seam - rent). The first cut skipped the
    # aspect term for skeletons and weighted seams 0.35, which punished
    # the skeleton exactly where it is right: the detector is blind on
    # hips (ridge F1 0.446), so hip seams score low. Aspect agreement is
    # the term the skeleton wins on -- every face tilts away from its
    # ridge by construction, and the LiDAR confirms it when the form fits.
    if skel_faces:
        from shapely.geometry import LineString as _LS
        from shapely.ops import unary_union as _uu3
        rim = geom.exterior.buffer(0.5)
        seams = []
        for f in skel_faces:
            cs = list(f.exterior.coords)
            for a2, b2 in zip(cs, cs[1:]):
                if _LS([a2, b2]).difference(rim).length > 0.8:
                    seams.append((a2, b2))
        ridge = _uu3([_LS([a2, b2]) for a2, b2 in seams]) if seams else None
        pl_num = pl_den = asp_num = asp_den = 0.0
        for f in skel_faces:
            inl, down, slope = face_plane(f)
            pl_num += f.area * inl
            pl_den += f.area
            if ridge is not None and down is not None and slope >= 4.0:
                c = f.centroid
                npt = ridge.interpolate(ridge.project(c))
                d = np.array([c.x - npt.x, c.y - npt.y])
                nd = np.linalg.norm(d)
                if nd > 0.4:
                    asp_num += f.area * max(0.0, float(np.dot(down, d / nd)))
                    asp_den += f.area
        plane = pl_num / max(pl_den, 1e-9)
        aspect = (asp_num / asp_den) if asp_den > 0 else 0.5
        sup = (np.mean([seam_support(a2, b2) for a2, b2 in seams[:40]])
               if seams else 0.0)
        sk_conf = 0.35 * plane + 0.30 * aspect + 0.25 * sup \
            - COMPLEXITY_RENT["skeleton"]
        if _os.environ.get("SOLAR_DEBUG_HYP"):
            print(f"  sk_conf {sk_conf:.3f} (pl {plane:.2f} asp {aspect:.2f} sup {sup:.2f}) vs part_conf {part_conf:.3f}", flush=True)
        if sk_conf > part_conf and len(skel_faces) >= 2:
            hypothesis_faces.last_confidence = sk_conf
            return skel_faces
    hypothesis_faces.last_confidence = part_conf
    return out_faces


def evidence_map(prob_max, rgb):
    """Support for a boundary: the model's activation OR a visible edge.

    Josh, on a run of roofs the selector fumbled: "It's pretty clear the model
    is not working very well at detecting lines from my markups." He is right
    -- and worse, the scorer judged every candidate BY that weak model's
    activation, so on #4735106 SAM's six faces at 99% coverage (his verdict:
    clearly better) scored 0.341 and lost to a two-face reading. The judge
    had the defendant's eyesight.

    Image gradient is model-free: his "clearly visible lines" are literally
    high gradient. Evidence = max(model activation, scaled gradient), so a
    correct boundary on a visible crease scores even where the model is blind.
    """
    g = rgb.astype(np.float32).mean(axis=2)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    hi = np.percentile(mag, 99) or 1.0
    grad = np.clip(mag / hi, 0, 1)
    return np.maximum(prob_max, 0.75 * grad)


def type_agreement(faces, geom, typed_probs, to_px, pts):
    """Does the candidate's GEOMETRY tell the same story as the detector's
    fold TYPES? Josh, setting the night's direction: "differentiating
    ridges, valleys, and cliffs to better detect geometry shapes of
    rooftops." A valley line means both faces drain toward it; a ridge
    means both drain away; a cliff means a height step. Every candidate
    claims types implicitly through its plane fits -- this term makes the
    claim answerable to the typed channels (v7: the pretrained model that
    lost the union contest but types folds twice as well as anything
    else; each model serves its measured strength).

    Returns (agreement 0..1, n_typed_edges); neutral (0.5, 0) when there
    is nothing to type.

    MEASURED 15 Sep, kept for diagnostics but NOT a selection term: with
    v7 typing it anti-correlates (v7's Dutch typing is random, 0.24, on
    real Queenstown folds); with v6 typing it is a wash between families
    (oracle-best 0.647 vs others 0.641) -- every family's faces sit on
    roughly the same folds, so their type stories match equally. Fold
    types may yet matter inside geometry CONSTRUCTION; they do not pick
    winners.
    """
    import numpy as np
    import shapely.geometry as sg
    from src.line_extract import _line_mean
    if typed_probs is None or len(faces) < 2 or pts is None:
        return 0.5, 0
    # fitted plane per face, once
    planes = []
    for f in faces:
        pl, _sub = _plane(f, pts)
        planes.append(pl)
    CH = {"ridge": 0, "valley": 1, "cliff": 2, "hip": 3}
    num = den = 0.0
    n_edges = 0
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            if planes[i] is None or planes[j] is None:
                continue
            try:
                shared = faces[i].buffer(0.12).intersection(
                    faces[j].buffer(0.12))
            except Exception:
                continue
            if shared.is_empty or shared.area < 0.15:
                continue
            mrr = shared.minimum_rotated_rectangle
            cc = list(mrr.exterior.coords)[:4]
            e = sorted(((np.hypot(cc[(k + 1) % 4][0] - cc[k][0],
                                  cc[(k + 1) % 4][1] - cc[k][1]), k)
                        for k in range(4)), reverse=True)
            L, k0 = e[0]
            if L < 1.2:
                continue
            a2 = (np.array(cc[k0]) + np.array(cc[(k0 + 3) % 4])) / 2
            b2 = (np.array(cc[(k0 + 1) % 4])
                  + np.array(cc[(k0 + 2) % 4])) / 2
            m2 = (a2 + b2) / 2
            pi, pj = planes[i], planes[j]
            zi = pi[0] * m2[0] + pi[1] * m2[1] + pi[2]
            zj = pj[0] * m2[0] + pj[1] * m2[1] + pj[2]
            ci = faces[i].centroid
            cj = faces[j].centroid
            zci = pi[0] * ci.x + pi[1] * ci.y + pi[2]
            zcj = pj[0] * cj.x + pj[1] * cj.y + pj[2]
            if abs(zi - zj) > 0.6:
                kind = "cliff"
            else:
                down_i = zi < zci - 0.05     # face i descends toward edge
                down_j = zj < zcj - 0.05
                if down_i and down_j:
                    kind = "valley"
                elif not down_i and not down_j:
                    # convex fold: ridge, or hip when it dives toward a
                    # footprint corner
                    kind = "ridge"
                    try:
                        corners = list(geom.exterior.coords)
                        for endpt in (a2, b2):
                            if any((endpt[0] - c[0]) ** 2
                                   + (endpt[1] - c[1]) ** 2 < 2.0 ** 2
                                   for c in corners):
                                kind = "hip"
                                break
                    except Exception:
                        pass
                else:
                    continue     # mixed drainage: geometrically untyped
            pa = np.array(to_px(*a2))
            pb = np.array(to_px(*b2))
            probs = [_line_mean(typed_probs[c], pa, pb)
                     for c in range(typed_probs.shape[0])]
            tot = sum(probs)
            if tot < 0.15:
                continue          # detector silent here: no vote
            ch = CH[kind]
            if ch >= typed_probs.shape[0]:
                continue
            num += L * (probs[ch] / tot)
            den += L
            n_edges += 1
    if den == 0:
        return 0.5, 0
    return num / den, n_edges


def score_candidate(faces, geom, prob_max, to_px, pts, inv_px=None):
    """How well a face-set fits the evidence. Higher is better.

    Two terms, one per instrument, both bounded:
      EDGES  interior boundaries should lie on imagery activation -- measured
             as mean activation along them. Exterior edges are the outline's
             business and score nothing either way.
      PLANES each face should be one plane -- mean LiDAR inlier fraction,
             area-weighted.
    A candidate with no interior edges (one big face) earns only its plane
    term, so a genuinely multi-face roof rewards the reading that found its
    folds.
    """
    from src.line_extract import _line_mean
    from src.roof_partition import _inlier_fraction

    if not faces:
        return 0.0
    rim = geom.exterior.buffer(0.5)
    edge_len = edge_sup = 0.0
    for f in faces:
        coords = list(f.exterior.coords)
        for a, bb in zip(coords, coords[1:]):
            import shapely.geometry as sg
            seg = sg.LineString([a, bb])
            inner = seg.difference(rim)
            L = inner.length
            if L < 0.5:
                continue
            pa = np.array(to_px(*a))
            pb = np.array(to_px(*bb))
            edge_len += L
            edge_sup += L * _line_mean(prob_max, pa, pb)
    edge_term = (edge_sup / edge_len) if edge_len > 3.0 else 0.35

    # RECALL of the activation: edge precision alone is biased toward the line
    # reading, whose edges lie on activation by construction -- the scorer
    # chose LINE on roofs where SAM agreed far better with Josh's faces
    # (#4735316: 0.90 vs 0.61, picked LINE). A reading that MISSES a fold the
    # imagery clearly shows must pay for it, whichever family it came from.
    ys, xs = np.nonzero(prob_max > 0.5)
    recall_term = 0.5
    if len(xs) > 30:
        import shapely.geometry as sg
        from shapely.ops import unary_union
        rings = [sg.LineString(list(f.exterior.coords)) for f in faces]
        # Tolerance is the teeth of this term. buffer(3.0) here was WORLD
        # metres -- any boundary within ~3.4 m "explained" an evidence pixel,
        # so recall read 0.62 vs 0.59 on #4735106 while edge precision (which
        # structurally favours the line family, whose edges are built ON the
        # activation) decided the contest alone. A fold you missed by 3 m is
        # a fold you missed.
        net = unary_union(rings)
        # (Tried interior-only recall -- excluding rim-adjacent evidence --
        # on the theory the outline gradient inflates under-segmented
        # readings. Measured 0.736 vs 0.769 on the drawn benchmark: worse.
        # Rim evidence stays in.)
        pxs = np.c_[xs, ys][np.random.default_rng(7).permutation(len(xs))[:400]]
        hit = 0
        for x2, y2 in pxs:
            wpt = sg.Point(inv_px(x2, y2))
            if net.distance(wpt) < 0.7:
                hit += 1
        recall_term = hit / len(pxs)

    # STEP RECALL. Josh, on 8 Isle Street: "This missed a roof plane" -- the
    # selector shipped the reading that ran one face across an annex sitting a
    # storey lower. A height step is the one boundary LiDAR sees decisively,
    # and no term looked at it: a reading that separates faces across a big
    # step earns credit, one that papers over it pays.
    step_term = 0.5
    if pts is not None and len(pts) > 80:
        from scipy.spatial import cKDTree
        import shapely.geometry as sg
        from shapely.ops import unary_union
        xy = pts[:, :2]
        tree = cKDTree(xy)
        rng = np.random.default_rng(11)
        idx = rng.permutation(len(pts))[:250]
        steps = []
        for i in idx:
            nb = tree.query_ball_point(xy[i], 1.0)
            if len(nb) < 4:
                continue
            z = pts[nb, 2]
            if z.max() - z.min() > 0.7:
                steps.append(xy[i])
        if len(steps) >= 8:
            net = unary_union(
                [sg.LineString(list(f.exterior.coords)) for f in faces])
            hit = sum(1 for q in steps
                      if net.distance(sg.Point(q)) < 0.8)
            step_term = hit / len(steps)

    plane_num = plane_den = 0.0
    for f in faces:
        pl, sub = _plane(f, pts)
        if pl is None:
            continue
        plane_num += f.area * _inlier_fraction(sub, pl)
        plane_den += f.area
    plane_term = (plane_num / plane_den) if plane_den else 0.0
    # (Tried gating coverage credit on per-face plane inlier to stop
    # equipment-tracing readings -- it crushed SAM harder than the junk:
    # big honest faces spanning ducts fail the floor while small duct-top
    # regions pass it. Reverted; the flat-defer owns that roof class.)
    # A reading is also answerable for the roof it left unexplained. Josh's
    # faces tile the footprint; a candidate of two clean faces covering 40%
    # scored best of all before this factor -- quality of what it kept, no
    # charge for what it dropped (#4734678: score 0.77, agreement 0.28).
    coverage = min(1.0, sum(f.area for f in faces) / max(geom.area, 1e-9))
    # (Weights 0.2 edge / 0.3 recall measured 0.734 vs 0.769 -- the edge
    # term earns its 0.3 on the drawn corpus; the bias it carries is real
    # but repricing it globally costs more than it buys.)
    return (0.3 * edge_term + 0.2 * recall_term + 0.2 * step_term
            + 0.3 * plane_term) * coverage


# ------------------------------------------------- line-network generator

def _medial_axis(poly):
    """Interior medial-axis segments of one polygon, straightened."""
    from scipy.spatial import Voronoi
    from shapely.geometry import Point, LineString, MultiLineString
    from shapely.ops import linemerge
    ring = list(poly.exterior.coords)[:-1]
    dense = []
    for a2, b2 in zip(ring, ring[1:] + ring[:1]):
        L = np.hypot(b2[0] - a2[0], b2[1] - a2[1])
        n2 = max(int(L / 0.4), 1)
        for t in range(n2):
            dense.append((a2[0] + (b2[0] - a2[0]) * t / n2,
                          a2[1] + (b2[1] - a2[1]) * t / n2))
    if len(dense) < 8:
        return []
    vor = Voronoi(np.array(dense))
    shrunk = poly.buffer(-0.25)
    if shrunk.is_empty:
        return []
    raw = []
    for v1, v2 in vor.ridge_vertices:
        if v1 < 0 or v2 < 0:
            continue
        p1, p2 = vor.vertices[v1], vor.vertices[v2]
        if shrunk.contains(Point(*p1)) and shrunk.contains(Point(*p2)):
            raw.append(LineString([p1, p2]))
    if not raw:
        return []
    merged = linemerge(MultiLineString(raw))
    segs = []
    for g2 in getattr(merged, "geoms", [merged]):
        c = list(g2.coords)

        def rec(i, j):
            a3 = np.array(c[i]); b3 = np.array(c[j])
            d = b3 - a3; L = np.hypot(*d)
            if L < 1e-9 or j - i < 2:
                segs.append((a3, b3)); return
            nv = np.array([-d[1], d[0]]) / L
            devs = [abs((np.array(c[k]) - a3) @ nv) for k in range(i, j + 1)]
            k = int(np.argmax(devs))
            if devs[k] > 0.35:
                rec(i, i + k); rec(i + k, j)
            else:
                segs.append((a3, b3))
        rec(0, len(c) - 1)
    return [(a3, b3) for a3, b3 in segs if np.hypot(*(b3 - a3)) >= 0.9]


def network_lines(prob3, geom, bounds, w, h, pts):
    """The line network Josh rated best: candidate nets scored whole, winner
    snapped to the activation and junction-cleaned. Ported from the preview
    where it lived while he judged it; the selector was still feeding
    line_facets the RAW extraction, which reads 7 Anderson Heights as one
    blob face -- "This is still broken", and it was.
    """
    from src.line_extract import (extract, clip_to, _line_mean,
                                  _junction_cleanup, _colinear_merge)
    import shapely.affinity as aff
    from shapely.ops import unary_union

    b = bounds
    P = prob3.max(axis=0)

    def w2p(x, y):
        return np.array([(x - b[0]) / (b[2] - b[0]) * w,
                         (1 - (y - b[1]) / (b[3] - b[1])) * h])

    def p2w(a):
        return (b[0] + a[0] / w * (b[2] - b[0]),
                b[1] + (1 - a[1] / h) * (b[3] - b[1]))

    def tw(px, py):
        return p2w(np.array([px, py]))

    ext = clip_to(extract(prob3, tw), geom)
    cand_A = [(w2p(r["seg"][0], r["seg"][1]), w2p(r["seg"][2], r["seg"][3]))
              for r in ext]
    # Josh, third flag on the same two-part building: "Still missing the
    # middle ridgeline here, I have told you this many times". The archetype
    # candidates were built from the WHOLE footprint -- a gable "ridge" for a
    # flat-block-plus-hall runs diagonally across both, scores nothing, and
    # the hall's own obvious ridge is never proposed. Decompose first: each
    # near-rectangular part proposes its own medial axis and its own gable.
    try:
        parts = [b2 for b2 in rect_parts(geom) if b2.area > 15]
    except Exception:
        parts = []
    # decompose only when the parts are SUBSTANTIAL: a garage notch on a
    # simple hip house is not a second building, and splitting there cost
    # 0.04 of benchmark agreement on the drawn houses that were already right
    big = [b2 for b2 in parts if b2.area > 0.12 * geom.area]
    if len(big) >= 2:
        blobs = big
    else:
        blobs = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    def snap(a2, b2):
        d2 = b2 - a2
        L = np.hypot(*d2)
        if L < 1e-6:
            return a2, b2
        u2 = d2 / L
        nrm = np.array([-u2[1], u2[0]])
        best, bo = -1.0, 0.0
        for o in np.arange(-5.0, 5.01, 0.5):
            m2 = _line_mean(P, a2 + o * nrm, b2 + o * nrm)
            if m2 > best:
                best, bo = m2, o
        return a2 + bo * nrm, b2 + bo * nrm

    def score_net(net):
        # Baseline is the price of drawing a line. At 0.25, any line above
        # faint activation ADDS score, so a long weakly-supported ring
        # out-totals a short strongly-supported skeleton: #4734914's
        # truncated-hip band (huge length, eave-shadow support) beat the
        # extracted hip skeleton that matched the visible ridge exactly.
        # Josh: "If you are not detecting clear lines you should not just
        # randomly draw them" -- a line must be clearly supported to pay
        # for itself.
        tot = 0.0
        for a2, b2 in net:
            a3, b3 = snap(a2, b2)
            tot += np.hypot(*(b3 - a3)) * (_line_mean(P, a3, b3) - 0.45)
        return tot

    # PER-PART SELECTION. One family rarely fits a compound building: the
    # hall wants its gable, the wing wants its hips, and a whole-building
    # winner forces one answer on both. Each part runs its own contest --
    # extraction segments assigned by midpoint, archetypes built per part --
    # and the winners union into the net.
    def seg_mid_in(a2, b2, blob):
        from shapely.geometry import Point
        m = (a2 + b2) / 2
        return blob.buffer(0.4).contains(Point(p2w(m)))

    best_net = []
    all_fams = []
    for blob in blobs:
        fam_A = [(a2, b2) for a2, b2 in cand_A if seg_mid_in(a2, b2, blob)]
        fam_B = [(w2p(*a3), w2p(*b3)) for a3, b3 in _medial_axis(blob)]
        mrr = blob.minimum_rotated_rectangle
        cc = list(mrr.exterior.coords)[:4]
        e = sorted(((np.hypot(cc[(k + 1) % 4][0] - cc[k][0],
                              cc[(k + 1) % 4][1] - cc[k][1]), k)
                    for k in range(4)), reverse=True)
        k0 = e[0][1]
        mid_a = (np.array(cc[k0]) + np.array(cc[(k0 + 3) % 4])) / 2
        mid_b = (np.array(cc[(k0 + 1) % 4]) + np.array(cc[(k0 + 2) % 4])) / 2
        fam_C = [(w2p(*mid_a), w2p(*mid_b))]
        fam_D = []
        if e[0][0] / max(e[2][0], 1e-6) < 1.5:
            import shapely.affinity as _aff2
            inner = _aff2.scale(mrr, 0.35, 0.35, origin="centroid")
            ic = list(inner.exterior.coords)[:4]
            for k in range(4):
                fam_D.append((w2p(*ic[k]), w2p(*ic[(k + 1) % 4])))
                fam_D.append((w2p(*ic[k]), w2p(*cc[k])))
        b_net, b_sc = [], -1e9
        import os as _os
        if _os.environ.get("SOLAR_DEBUG_NET"):
            for nm, fam in (("A-extract", fam_A), ("B-medial", fam_B),
                            ("C-gable", fam_C), ("D-trunc", fam_D)):
                print(f"    part {nm}: {len(fam)} segs "
                      f"score {score_net(fam) if fam else float('nan'):.1f}")
        for fam in (fam_A, fam_B, fam_C, fam_D):
            if not fam:
                continue
            sc = score_net(fam)
            if sc > b_sc:
                b_sc, b_net = sc, fam
        all_fams.extend([fam_B, fam_C, fam_D])
        best_net.extend(b_net)

    # THE TWO REGIMES COMPETE. Per-part selection fixed the compound
    # buildings Josh flagged three times ("Still missing the middle
    # ridgeline") and cost 0.03 on the drawn benchmark: an L-shaped house
    # that whole-building extraction read perfectly gets split and its parts
    # out-voted. Neither regime owns every building, so the per-part union
    # and the whole-building winner are both assembled and the higher-scoring
    # net ships -- the same selection principle one level up.
    whole_best, whole_sc = [], -1e9
    import os as _os
    if _os.environ.get("SOLAR_DEBUG_NET"):
        print(f"    whole A-extract: {len(cand_A)} segs score {score_net(cand_A):.1f}; "
              f"parts-union: {len(best_net)} segs score {score_net(best_net) if best_net else float('nan'):.1f}")
    for fam in [cand_A] + all_fams:
        if not fam:
            continue
        sc = score_net(fam)
        if sc > whole_sc:
            whole_sc, whole_best = sc, fam
    if not best_net or score_net(best_net) < whole_sc:
        best_net = whole_best

    cleaned = _junction_cleanup([snap(a2, b2) for a2, b2 in best_net])

    # COMPLETE THE WINNER'S JUNCTIONS. Josh, on the first Anderson reading he
    # called much better: "you are missing a few ridge lines and therefore
    # faces missing" -- the short links between the dormer pyramid and the end
    # hips. One candidate family rarely carries every line, but the missing
    # ones have a structural signature: both endpoints already ARE nodes of
    # the winning network. A line from any family may join the winner if the
    # activation supports it, it duplicates nothing, and it connects two
    # existing nodes -- nothing free-floating is ever added.
    from src.line_extract import _line_mean as _lm
    nodes = []
    for a2, b2 in cleaned:
        nodes.extend([a2, b2])
    extra = []
    for fam in [cand_A] + all_fams:
        for a2, b2 in fam:
            a3, b3 = snap(a2, b2)
            if np.hypot(*(b3 - a3)) < 12:
                continue
            if _lm(P, a3, b3) < 0.45:
                continue
            near_a = any(np.hypot(*(a3 - n)) < 12 for n in nodes)
            near_b = any(np.hypot(*(b3 - n)) < 12 for n in nodes)
            if not (near_a and near_b):
                continue
            d3 = (b3 - a3) / max(np.hypot(*(b3 - a3)), 1e-9)
            dup = False
            for ka, kb in cleaned + extra:
                kd = kb - ka
                kL = np.hypot(*kd)
                if kL < 1e-9:
                    continue
                kd = kd / kL
                ang = abs(np.degrees(np.arctan2(kd[1], kd[0])
                                     - np.arctan2(d3[1], d3[0]))) % 180
                if min(ang, 180 - ang) > 12:
                    continue
                nrm = np.array([-kd[1], kd[0]])
                if (abs((a3 - ka) @ nrm) < 6 and abs((b3 - ka) @ nrm) < 6):
                    dup = True
                    break
            if not dup:
                extra.append((a3, b3))
    # Node-pair SYNTHESIS -- inventing a connector between two existing
    # junctions -- was built and measured in four variants (activation-gated,
    # dual-evidence, crossing-guarded, plane-intersection-tested) and every
    # one lowered mean agreement with Josh's faces below this configuration
    # (0.737 / 0.711 / 0.731 against 0.744). It found the one ridge he named
    # on Anderson only in the variant that hurt most elsewhere. Not kept: a
    # connector invisible to both instruments routes its roof to the markup
    # queue instead of being guessed.
    if extra:
        cleaned = _junction_cleanup(cleaned + extra)

    # ROOF-GRAPH CLOSURE. Josh, three times on the same missing Anderson
    # ridge -- and topology says he is right to insist. Where two hips meet
    # and stop, the roof planes either side are still separated, so the fold
    # MUST continue until it meets another line: a junction whose edges all
    # leave on one side is not a place a fold can end. The continuation's
    # existence and direction are both forced -- down the open gap -- and it
    # terminates only on the network. No evidence gate is needed for what
    # geometry mandates; four evidence-gated syntheses of this same line each
    # cost the global mean, because they also admitted lines topology forbids.
    def closure(net):
        nodes2 = []
        for a2, b2 in net:
            for q in (a2, b2):
                if not any(np.hypot(*(q - m)) < 3 for m in nodes2):
                    nodes2.append(q)
        added = []
        for nd in nodes2:
            dirs = []
            for a2, b2 in net:
                for q, o in ((a2, b2), (b2, a2)):
                    if np.hypot(*(q - nd)) < 3:
                        v = o - q
                        L = np.hypot(*v)
                        if L > 1e-6:
                            dirs.append(np.arctan2(v[1], v[0]))
            if len(dirs) == 1:
                # A DANGLING FOLD END MID-ROOF IS THE CANONICAL MUST-CONTINUE
                # CASE -- a fold cannot end on a plane -- and this rule
                # originally skipped exactly those nodes. Anderson's missing
                # right ridge was a short stub off the pyramid corner whose
                # far end dangled here, invisible to a junction-gap test.
                # Continue along the stub's own direction; direction is never
                # invented, only length, and only to a network hit.
                bis = dirs[0] + np.pi
                u2 = np.array([np.cos(bis), np.sin(bis)])
                best_t = None
                for a2, b2 in net:
                    d2 = b2 - a2
                    den = d2[0] * u2[1] - d2[1] * u2[0]
                    if abs(den) < 1e-9:
                        continue
                    t = ((a2 - nd)[0] * d2[1] - (a2 - nd)[1] * d2[0]) / -den
                    r2 = ((a2 - nd)[0] * u2[1] - (a2 - nd)[1] * u2[0]) / -den
                    if t > 6 and -0.05 <= r2 <= 1.05:
                        if best_t is None or t < best_t:
                            best_t = t
                for m in nodes2:
                    v = m - nd
                    t = v @ u2
                    if t > 6 and np.hypot(*(v - t * u2)) < 5:
                        if best_t is None or t < best_t:
                            best_t = t
                if best_t is not None and best_t <= 140:
                    added.append((nd.copy(), nd + best_t * u2))
                continue
            dirs.sort()
            gaps = [(dirs[(i + 1) % len(dirs)] - dirs[i]) % (2 * np.pi)
                    for i in range(len(dirs))]
            gi = int(np.argmax(gaps))
            if gaps[gi] < np.radians(200):
                continue          # edges leave all around: a closed junction
            bis = dirs[gi] + gaps[gi] / 2
            u2 = np.array([np.cos(bis), np.sin(bis)])
            # march down the gap to the first thing the network offers
            best_t = None
            for a2, b2 in net:
                d2 = b2 - a2
                den = d2[0] * u2[1] - d2[1] * u2[0]
                if abs(den) < 1e-9:
                    continue
                t = ((a2 - nd)[0] * d2[1] - (a2 - nd)[1] * d2[0]) / -den
                r2 = ((a2 - nd)[0] * u2[1] - (a2 - nd)[1] * u2[0]) / -den
                if t > 6 and -0.05 <= r2 <= 1.05:
                    if best_t is None or t < best_t:
                        best_t = t
            for m in nodes2:
                v = m - nd
                t = v @ u2
                if t > 6 and np.hypot(*(v - t * u2)) < 5:
                    if best_t is None or t < best_t:
                        best_t = t
            if best_t is not None and best_t <= 100:
                added.append((nd.copy(), nd + best_t * u2))
        return added

    def cluster(net, tol=6.0):
        """Weld near-coincident junctions into one point.

        The probe on Anderson showed why closure misfired: four separate
        nodes within ~8 px around one pyramid corner, degree-1 fragments
        hanging off them. Topology rules cannot read a soup. Every endpoint
        joins the centroid of its cluster; zero-length remains are dropped.
        """
        reps = []
        def find(q):
            for r in reps:
                if np.hypot(*(q - r["c"])) <= tol:
                    return r
            return None
        for a2, b2 in net:
            for q in (a2, b2):
                r = find(q)
                if r is None:
                    reps.append({"c": q.copy(), "m": [q]})
                else:
                    r["m"].append(q)
                    r["c"] = np.mean(r["m"], axis=0)
        out2 = []
        for a2, b2 in net:
            ra, rb = find(a2), find(b2)
            na = ra["c"] if ra is not None else a2
            nb = rb["c"] if rb is not None else b2
            if np.hypot(*(nb - na)) >= 8:
                out2.append((na, nb))
        return out2

    cleaned = cluster(cleaned)
    grown = closure(cleaned)
    if grown:
        cleaned = cluster(cleaned + grown)

    # STRAIGHTEN JOGS. Josh, on the one line still wrong on Anderson: "A
    # misplaced line in the middle, which should be at the point of the
    # triangle" -- a 1.9 m skewed splice between two runs of one straight
    # ridge, turning it just short of the apex. When a short segment's two
    # neighbours run nearly collinear THROUGH it, the jog is an artefact of
    # tracing, not a fold: project its endpoints onto the neighbours' common
    # line so the collinear merge can absorb all three into one.
    def straighten_jogs(net):
        for i, (a2, b2) in enumerate(net):
            L = np.hypot(*(b2 - a2))
            if not (1.0 < L < 30.0):
                continue
            def longest_other_at(pt):
                best = None
                for j, (c2, e2) in enumerate(net):
                    if j == i:
                        continue
                    for q, o in ((c2, e2), (e2, c2)):
                        if np.hypot(*(q - pt)) < 3:
                            Ln2 = np.hypot(*(o - q))
                            if Ln2 > 2 * L and (best is None or Ln2 > best[0]):
                                best = (Ln2, q, o)
                return best
            na = longest_other_at(a2)
            nb = longest_other_at(b2)
            if na is None or nb is None:
                continue
            da = na[2] - na[1]
            db = nb[2] - nb[1]
            ang = abs(np.degrees(np.arctan2(da[1], da[0])
                                 - np.arctan2(db[1], db[0]))) % 180
            if min(ang, 180 - ang) > 12:
                continue
            u2 = da / max(np.hypot(*da), 1e-9)
            # both neighbours on one line? check B's anchor against A's line
            nrm = np.array([-u2[1], u2[0]])
            if abs((nb[1] - na[1]) @ nrm) > 3:
                continue
            base = na[1]
            net[i] = (base + ((a2 - base) @ u2) * u2,
                      base + ((b2 - base) @ u2) * u2)
        return net

    cleaned = straighten_jogs(list(cleaned))
    merged4 = True
    while merged4:
        merged4 = False
        for i in range(len(cleaned)):
            if cleaned[i] is None:
                continue
            for j in range(i + 1, len(cleaned)):
                if cleaned[j] is None:
                    continue
                m4 = _colinear_merge(cleaned[i], cleaned[j])
                if m4 is not None:
                    cleaned[i] = m4
                    cleaned[j] = None
                    merged4 = True
        cleaned = [c for c in cleaned if c is not None]

    out = []
    for a2, b2 in cleaned:
        x1, y1 = p2w(a2)
        x2, y2 = p2w(b2)
        out.append([x1, y1, x2, y2])

    # NEAR-DUPLICATE SEGMENTS SHATTER THE POLYGONIZATION. The Anderson autopsy:
    # 13 of Josh's 14 lines present, polygonize yields 9 cells matching his 9
    # faces -- and one is an 83 m2 leak beside a 2.6 m2 SLIVER. Two almost-
    # coincident lines survive the pixel weld, polygonize builds a thin
    # corridor between them, and the faces either side connect around its
    # ends. Keep the longer of any coincident pair, in metres, at the end.
    ded = []
    for seg in sorted(out, key=lambda s2: -np.hypot(s2[2] - s2[0],
                                                    s2[3] - s2[1])):
        a3 = np.array(seg[:2]); b3 = np.array(seg[2:])
        L = np.hypot(*(b3 - a3))
        if L < 1e-6:
            continue
        u3 = (b3 - a3) / L
        dup = False
        for k in ded:
            ka = np.array(k[:2]); kb = np.array(k[2:])
            kd = kb - ka
            kL = np.hypot(*kd)
            kd = kd / kL
            ang = abs(np.degrees(np.arctan2(kd[1], kd[0])
                                 - np.arctan2(u3[1], u3[0]))) % 180
            if min(ang, 180 - ang) > 10:
                continue
            nrm = np.array([-kd[1], kd[0]])
            if abs((a3 - ka) @ nrm) > 0.4 or abs((b3 - ka) @ nrm) > 0.4:
                continue
            t = sorted(((a3 - ka) @ kd, (b3 - ka) @ kd))
            if min(t[1], kL) - max(t[0], 0.0) > 0.5 * L:
                dup = True
                break
        if not dup:
            ded.append(seg)

    # AN ALMOST-CLOSED MOUTH LEAKS A WHOLE FACE. Two hips welded to each
    # other half a metre above the eave corner are both degree-2 there, so the
    # degree-based sealer never extends them, and polygonize leaks two of
    # Josh's faces into one 83 m2 cell through the gap. Snapping every
    # near-outline endpoint onto the ring was measured and REVERTED -- it
    # dragged legitimate interior ends sideways and cost 0.06 of mean
    # agreement. The mouth case is specific: an endpoint near a footprint
    # VERTEX belongs on that corner, and only that is done.
    corners = [np.array(c) for c in list(geom.exterior.coords)[:-1]]
    snapped = []
    for seg in ded:
        pts2 = [np.array(seg[:2]), np.array(seg[2:])]
        for i2 in range(2):
            for c2 in corners:
                if 1e-6 < np.hypot(*(pts2[i2] - c2)) < 0.8:
                    pts2[i2] = c2.copy()
                    break
        if np.hypot(*(pts2[1] - pts2[0])) >= 0.8:
            snapped.append([pts2[0][0], pts2[0][1], pts2[1][0], pts2[1][1]])
    return snapped


def rect_parts(geom, depth=0):
    """Split a footprint at reflex corners into near-rectangular parts.

    Public here because flatness is a PER-PART question: #5371128 is a flat
    block joined to a gabled hall, and a whole-building plane fit read the
    pair as flat, deferring a roof the selector should own.
    """
    from shapely.geometry import LineString
    from shapely.ops import split as shp_split
    ring = list(geom.exterior.coords)[:-1]
    n = len(ring)
    area2 = sum(ring[i][0] * ring[(i + 1) % n][1]
                - ring[(i + 1) % n][0] * ring[i][1] for i in range(n))
    if depth >= 6:
        return [geom]
    for i in range(n):
        ax, ay = ring[i - 1]
        bx, by = ring[i]
        cx, cy = ring[(i + 1) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if (cross < 0) != (area2 > 0):
            continue
        best = None
        for ox, oy in ((bx - ax, by - ay), (bx - cx, by - cy)):
            L = np.hypot(ox, oy)
            if L < 1e-9:
                continue
            u = (ox / L, oy / L)
            cut = LineString([(bx - u[0] * 0.05, by - u[1] * 0.05),
                              (bx + u[0] * 200, by + u[1] * 200)])
            inner = cut.intersection(geom)
            ln = sum(g2.length for g2 in getattr(inner, "geoms", [inner])
                     if g2.geom_type == "LineString")
            if ln > 0.5 and (best is None or ln < best[0]):
                best = (ln, cut)
        if best is None:
            continue
        try:
            pieces = [g2 for g2 in shp_split(geom, best[1]).geoms
                      if g2.geom_type == "Polygon" and g2.area > 4.0]
        except Exception:
            continue
        if len(pieces) >= 2:
            out = []
            for pc2 in pieces:
                out.extend(rect_parts(pc2, depth + 1))
            return out
    return [geom]

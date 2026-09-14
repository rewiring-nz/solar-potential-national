"""Show WHERE the detector puts its lines, so Josh can judge placement by eye.

Josh: "Visually present me your work as a way to check it is right, don't
measure arbitrary things like number of panels. I can visually check if it is
right or not."

So this renders no metrics at all. Per roof, side by side on the same crop:

  DEPLOYED   the v1 model through the old component-axis extraction -- what the
             live build currently sees
  REBUILT    the chosen model through the skeleton-traced, junction-split,
             peak-snapped extraction (src/line_extract)
  JOSH       the lines he drew, where the roof is labelled -- the standard the
             other two panels are aiming at

Line colours match his markup tool exactly (ridge green, valley blue, cliff
red), so a wrong KIND is as visible as a wrong position.

Usage:
    python tools/lines_preview.py --ids 4734914 4735341 --out lines_check.html
    python tools/lines_preview.py --flagged --out lines_check.html
"""

import argparse
import base64
import io
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

PAD_M = 4.0
SCALE = 3            # crops are ~200 px; lines need room to be judged
COL = {"ridge": (31, 255, 122), "valley": (53, 182, 255), "cliff": (255, 59, 48)}


def render(rgb, lines, bounds, outline):
    """One panel: the crop with lines over it, world coords -> pixels."""
    import numpy as np
    from PIL import Image, ImageDraw
    h, w = rgb.shape[:2]
    im = Image.fromarray(rgb.astype("uint8")).resize((w * SCALE, h * SCALE),
                                                     Image.LANCZOS)
    d = ImageDraw.Draw(im)
    minx, miny, maxx, maxy = bounds

    def px(x, y):
        return ((x - minx) / (maxx - minx) * w * SCALE,
                (1 - (y - miny) / (maxy - miny)) * h * SCALE)

    if outline is not None:
        d.line([px(x, y) for x, y in outline.exterior.coords],
               fill=(255, 200, 80), width=2)
    for x1, y1, x2, y2, kind in lines:
        d.line([px(x1, y1), px(x2, y2)], fill=COL.get(kind, (255, 255, 255)),
               width=3)
        for x, y in ((x1, y1), (x2, y2)):
            cx, cy = px(x, y)
            d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3],
                      fill=COL.get(kind, (255, 255, 255)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--flagged", action="store_true",
                    help="the roofs Josh has flagged (data/flagged_ids.txt)")
    ap.add_argument("--model-old", default="data/models/roof_lines_v1.pt")
    ap.add_argument("--model-new", default="data/models/roof_lines_v4.pt")
    ap.add_argument("--out", default="lines_check.html")
    a = ap.parse_args()

    import numpy as np
    import torch
    import geopandas as gpd
    import rasterio
    import rasterio.windows
    from src.region_build import area_paths, all_areas
    from src.pointcloud_source import PointCloudSource
    import train_line_model as T
    from predict_roof_lines import segments_from_mask
    from src.line_extract import (extract, clip_to, _line_mean, _bilinear,
                                  _junction_cleanup, _colinear_merge, _refine)

    # THE ARCHETYPE PANEL. Josh, on a clean hip roof the detector fumbled:
    # "This is a clear shaped roof and yet the failure is very bad so clearly
    # not the right process" -- and on the residual small errors: "These small
    # errors here might have big impacts across a large dataset."
    #
    # He is right about the process. Bottom-up extraction can only clean up
    # what the model fired on, and its errors -- a line stopping short, a spur
    # -- are exactly the kind that scale badly. A hip roof's line network is
    # already implied by its outline: the constructive skeleton gives one
    # full-length ridge, hips into the corners, exact junctions, and NOTHING
    # else, by construction. So the archetype proposes and the imagery model's
    # job shrinks to verifying and positioning, which is a far easier job than
    # inventing geometry.
    def _medial(poly):
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

    def _rect_parts(geom, depth=0):
        """Split at reflex corners into near-rectangular parts.

        The archetype was drawn for the whole outline at once, and Josh's
        verdict on the first compound building was "Super wrong" -- because it
        is not ONE roof: a flat section and two pitched wings share the
        footprint. A hypothesis per PART can be judged per part.
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
            ax, ay = ring[i - 1]; bx, by = ring[i]; cx, cy = ring[(i + 1) % n]
            cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
            if (cross < 0) != (area2 > 0):
                continue          # convex corner
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
                    out.extend(_rect_parts(pc2, depth + 1))
                return out
        return [geom]

    def skeleton_lines(bid, geom, pts):
        """Per-part archetype, each part judged on evidence.

        Split the footprint at reflex corners; LiDAR says which parts are flat
        (slope is its one job) -- those get NO lines; contiguous pitched parts
        are re-merged so an L-shaped pitched wing keeps its true valley, and
        each pitched blob gets its medial axis. Valleys leave reflex corners.
        """
        from shapely.ops import unary_union
        from src.roof_partition import _points_in, _fit_plane_robust, _slope_aspect
        parts = _rect_parts(geom)
        pitched = []
        for part in parts:
            sub = _points_in(part, pts) if len(pts) else []
            slope = None
            if len(sub) >= 25:
                pl = _fit_plane_robust(sub)
                if pl is not None:
                    slope, _ = _slope_aspect(pl)
            # a single plane fitted over a whole PITCHED part reads near-flat
            # (both sides average out), so also try the spread of residuals:
            # a hip part has points well above the eaves plane
            if slope is not None and slope < 4.0 and len(sub) >= 25:
                z = sub[:, 2]
                if (np.percentile(z, 95) - np.percentile(z, 5)) < 0.9:
                    continue          # genuinely flat: no lines here
            pitched.append(part)
        if not pitched:
            return []
        blobs = unary_union([p2.buffer(0.05) for p2 in pitched])
        out = []
        for blob in getattr(blobs, "geoms", [blobs]):
            blob = blob.buffer(-0.05)
            if blob.is_empty or blob.area < 8.0:
                continue
            ring = list(blob.exterior.coords)[:-1]
            n = len(ring)
            area2 = sum(ring[i][0] * ring[(i + 1) % n][1]
                        - ring[(i + 1) % n][0] * ring[i][1] for i in range(n))
            reflex = []
            for i in range(n):
                ax, ay = ring[i - 1]; bx, by = ring[i]; cx, cy = ring[(i + 1) % n]
                cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
                if (cross < 0) == (area2 > 0):
                    reflex.append(ring[i])
            for a3, b3 in _medial(blob):
                kind = "ridge"
                for rx, ry in reflex:
                    if (np.hypot(a3[0] - rx, a3[1] - ry) < 1.2
                            or np.hypot(b3[0] - rx, b3[1] - ry) < 1.2):
                        kind = "valley"; break
                out.append([a3[0], a3[1], b3[0], b3[1], kind])
        return out

    ids = a.ids
    if a.flagged and not ids:
        ids = [int(x) for x in
               (ROOT / "data" / "flagged_ids.txt").read_text().split()]
        ids = sorted(set(ids))
    if not ids:
        print("give --ids or --flagged")
        return 2

    def load(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = T.build_unet(ck.get("pretrained", False))
        m.load_state_dict(ck["state_dict"])
        m.eval()
        return m

    m_old = load(ROOT / a.model_old)
    m_new = load(ROOT / a.model_new)

    labels = {}
    lp = ROOT / "data" / "roof_labels.json"
    if lp.exists():
        labels = json.loads(lp.read_text()).get("buildings", {})

    rows = []
    ctxs = {}
    for bid in ids:
        placed = False
        for name in ["pilot"] + [x for x in all_areas() if x != "pilot"]:
            if name not in ctxs:
                p = area_paths(name)
                if not (p["outlines"].exists() and p["imagery"].exists()):
                    ctxs[name] = None
                    continue
                dd = p["dir"] / "building_outlines_dedup.geojson"
                ctxs[name] = {
                    "gdf": gpd.read_file(dd if dd.exists() else p["outlines"]
                                         ).set_index("building_id", drop=False),
                    "img": rasterio.open(p["imagery"]),
                    "pc": PointCloudSource(max_cached_tiles=3)}
            ctx = ctxs[name]
            if ctx is None or bid not in ctx["gdf"].index:
                continue
            geom = ctx["gdf"].loc[bid].geometry
            minx, miny, maxx, maxy = geom.bounds
            b = (minx - PAD_M, miny - PAD_M, maxx + PAD_M, maxy + PAD_M)
            win = rasterio.windows.from_bounds(*b, ctx["img"].transform)
            rgb = np.moveaxis(ctx["img"].read([1, 2, 3], window=win,
                                              boundless=True, fill_value=0),
                              0, -1)
            h, w = rgb.shape[:2]
            if h < 32 or w < 32:
                break
            arr = np.pad(rgb, ((0, (-h) % 16), (0, (-w) % 16), (0, 0)))
            x = torch.from_numpy(arr).float().permute(2, 0, 1)[None] / 255.0

            def tw(px, py):
                return (b[0] + px / w * (b[2] - b[0]),
                        b[1] + (1 - py / h) * (b[3] - b[1]))

            with torch.no_grad():
                pr_old = torch.sigmoid(m_old(x))[0].numpy()[:, :h, :w]
                pr_new = torch.sigmoid(m_new(x))[0].numpy()[:, :h, :w]

            old_lines = []
            for k in range(3):
                for seg, _ in segments_from_mask(pr_old[k], tw):
                    old_lines.append(seg + [["ridge", "valley", "cliff"][k]])
            new_lines = [r["seg"] + [r["kind"]]
                         for r in clip_to(extract(pr_new, tw), geom)]

            drawn = []
            lab = labels.get(str(bid)) or {}
            for l in lab.get("lines") or []:
                if l.get("a") and l.get("b"):
                    drawn.append([l["a"][0], l["a"][1], l["b"][0], l["b"][1],
                                  l.get("kind", "ridge")])

            arch = []
            cliff_cands = []
            pts_top = None
            try:
                from src.roof_partition import top_surface, _points_in
                pts_all = ctx["pc"].points_in_bbox(minx - 1, miny - 1,
                                                   maxx + 1, maxy + 1,
                                                   building_only=True)
                if len(pts_all) > 50:
                    pts_top = top_surface(pts_all)
                    arch = skeleton_lines(bid, geom, pts_top)
                    # CLIFF CANDIDATES: the boundaries between footprint parts.
                    # A cliff is a height BREAK, so its evidence is LiDAR, not
                    # imagery -- the model's weakest channel, and Josh's "misses
                    # cliff line here" was exactly a step the imagery barely
                    # shows. The geometry is already known: where the outline
                    # steps, the part cut runs along the break.
                    from shapely.ops import unary_union
                    parts = _rect_parts(geom)
                    for i2 in range(len(parts)):
                        for j2 in range(i2 + 1, len(parts)):
                            shared = parts[i2].buffer(0.08).intersection(
                                parts[j2].buffer(0.08))
                            if shared.is_empty or shared.area < 0.05:
                                continue
                            mrr = shared.minimum_rotated_rectangle
                            cc = list(mrr.exterior.coords)
                            edges2 = sorted(
                                ((np.hypot(cc[k + 1][0] - cc[k][0],
                                           cc[k + 1][1] - cc[k][1]),
                                  cc[k], cc[k + 1]) for k in range(4)),
                                reverse=True)
                            Lm, e1, e2 = edges2[0]
                            if Lm < 1.5:
                                continue
                            mid1 = ((e1[0] + edges2[1][1][0]) / 2,
                                    (e1[1] + edges2[1][1][1]) / 2)
                            cliff_cands.append((np.array(e1), np.array(e2)))
            except Exception:
                arch = []

            # FUSED, third design, after two Josh rejected. Line-level and
            # chunk-level surgery both failed the same way: evidence is judged
            # where the model happens to be loud, so a true ridge dies in the
            # shadowed half of a roof ("Fused makes this one worse") and a
            # square top the hypothesis never proposed cannot appear ("Still
            # no square in the middle ... obvious from imagery").
            #
            # So the unit of decision is now a WHOLE NETWORK. Candidates:
            #   A  the rebuilt extraction (what the model actually traced)
            #   B  hip archetype -- medial axis per pitched blob
            #   C  gable archetype -- one full-length ridge per pitched blob
            #   D  truncated pyramid -- inner rectangle + corner hips, for
            #      near-square blobs; the "square on top" Josh keeps seeing
            #      and no other candidate could say
            # Each is scored WHOLE: sum over lines of length x (support-0.25),
            # so a junk line subtracts and a true line in shadow merely fails
            # to add -- it no longer kills its network. The winner is kept
            # intact, snapped sideways to the activation, junctions cleaned.
            def w2p(x, y):
                return ((x - b[0]) / (b[2] - b[0]) * w,
                        (1 - (y - b[1]) / (b[3] - b[1])) * h)

            P = pr_new.max(axis=0)

            def sup_px(a2, b2):
                return _line_mean(P, a2, b2)

            def channel_kind(a2, b2, fallback):
                tot = np.zeros(3)
                L = np.hypot(*(b2 - a2))
                for t in np.linspace(0, 1, max(int(L), 2)):
                    q = a2 + t * (b2 - a2)
                    tot += [_bilinear(pr_new[c], *q) for c in range(3)]
                return (["ridge", "valley", "cliff"][int(np.argmax(tot))]
                        if tot.max() > 0.15 * max(int(np.hypot(*(b2 - a2))), 2)
                        else fallback)

            def to_px_net(lines_w):
                out2 = []
                for x1, y1, x2, y2, kind in lines_w:
                    out2.append((np.array(w2p(x1, y1)),
                                 np.array(w2p(x2, y2)), kind))
                return out2

            def snap_line(a2, b2):
                d2 = b2 - a2
                L = np.hypot(*d2)
                if L < 1e-6:
                    return a2, b2
                u2 = d2 / L
                nrm = np.array([-u2[1], u2[0]])
                best, best_o = -1.0, 0.0
                for o2 in np.arange(-5.0, 5.01, 0.5):
                    m2 = sup_px(a2 + o2 * nrm, b2 + o2 * nrm)
                    if m2 > best:
                        best, best_o = m2, o2
                return a2 + best_o * nrm, b2 + best_o * nrm

            def score_net(net):
                tot = 0.0
                for a2, b2, _ in net:
                    a3, b3 = snap_line(a2, b2)
                    L = np.hypot(*(b3 - a3))
                    tot += L * (sup_px(a3, b3) - 0.25)
                return tot

            # candidate constructions, all in world coords first
            from shapely.ops import unary_union as _uu
            import shapely.affinity as _aff
            pitched_blobs = []
            if arch:
                try:
                    parts = _rect_parts(geom)
                    keep2 = []
                    if pts_top is not None and len(pts_top):
                        from src.roof_partition import _points_in as _pin
                        for part in parts:
                            sub = _pin(part, pts_top)
                            if len(sub) >= 25:
                                z = sub[:, 2]
                                if (np.percentile(z, 95)
                                        - np.percentile(z, 5)) < 0.9:
                                    continue
                            keep2.append(part)
                    else:
                        keep2 = parts
                    blobs = _uu([p2.buffer(0.05) for p2 in keep2])                         if keep2 else None
                    if blobs is not None:
                        pitched_blobs = [g2.buffer(-0.05) for g2 in
                                         getattr(blobs, "geoms", [blobs])
                                         if g2.buffer(-0.05).area > 8]
                except Exception:
                    pitched_blobs = []

            cand_B = [(np.array(w2p(l[0], l[1])), np.array(w2p(l[2], l[3])),
                       l[4]) for l in arch]
            cand_A = to_px_net(new_lines)
            cand_C, cand_D = [], []
            for blob in pitched_blobs:
                mrr = blob.minimum_rotated_rectangle
                cc = list(mrr.exterior.coords)[:4]
                e = [(np.hypot(cc[(k + 1) % 4][0] - cc[k][0],
                               cc[(k + 1) % 4][1] - cc[k][1]), k)
                     for k in range(4)]
                e.sort(reverse=True)
                long_k = e[0][1]
                p1 = np.array(cc[long_k]); p2 = np.array(cc[(long_k + 1) % 4])
                p4 = np.array(cc[(long_k + 3) % 4])
                mid_a = (p1 + p4) / 2
                mid_b = (p2 + np.array(cc[(long_k + 2) % 4])) / 2
                cand_C.append((np.array(w2p(*mid_a)), np.array(w2p(*mid_b)),
                               "ridge"))
                aspect = e[0][0] / max(e[2][0], 1e-6)
                if aspect < 1.5:
                    inner = _aff.scale(mrr, 0.35, 0.35, origin="centroid")
                    ic = list(inner.exterior.coords)[:4]
                    for k in range(4):
                        cand_D.append((np.array(w2p(*ic[k])),
                                       np.array(w2p(*ic[(k + 1) % 4])),
                                       "ridge"))
                        cand_D.append((np.array(w2p(*ic[k])),
                                       np.array(w2p(*cc[k])), "ridge"))

            cands = [c for c in (cand_A, cand_B, cand_C, cand_D) if c]
            best_net, best_sc = [], -1e9
            for c in cands:
                sc = score_net(c)
                if sc > best_sc:
                    best_sc, best_net = sc, c

            fused_px, fused_kind = [], []
            for a2, b2, kind in best_net:
                a3, b3 = snap_line(a2, b2)
                fused_px.append((a3, b3))
                fused_kind.append(channel_kind(a3, b3, kind))

            # cliffs verified by the height step across them, LiDAR only
            if cliff_cands and pts_top is not None and len(pts_top):
                for ca_w, cb_w in cliff_cands:
                    d3 = cb_w - ca_w
                    L3 = np.hypot(*d3)
                    if L3 < 1e-6:
                        continue
                    u3 = d3 / L3
                    n3 = np.array([-u3[1], u3[0]])
                    za, zb = [], []
                    for t in np.linspace(0.15, 0.85, 9):
                        q = ca_w + t * d3
                        for side, acc in ((1.0, za), (-1.0, zb)):
                            sel = pts_top[
                                (np.abs((pts_top[:, :2] - q) @ u3) < L3 * 0.08)
                                & ((pts_top[:, :2] - q) @ n3 * side > 0.3)
                                & ((pts_top[:, :2] - q) @ n3 * side < 1.6)]
                            if len(sel):
                                acc.extend(sel[:, 2])
                    if len(za) > 6 and len(zb) > 6 and                             abs(np.median(za) - np.median(zb)) > 0.8:
                        fused_px.append((np.array(w2p(*ca_w)),
                                         np.array(w2p(*cb_w))))
                        fused_kind.append("cliff")
            cleaned = _junction_cleanup(fused_px)
            # cleanup can drop segments; re-pair kinds by nearest original
            fused = []
            for a2, b2 in cleaned:
                mid = (a2 + b2) / 2
                best = min(range(len(fused_px)),
                           key=lambda i2: np.hypot(
                               *(mid - (fused_px[i2][0] + fused_px[i2][1]) / 2)),
                           default=None)
                kind = fused_kind[best] if best is not None else "ridge"
                x1, y1 = tw(a2[0], a2[1])
                x2, y2 = tw(b2[0], b2[1])
                fused.append([x1, y1, x2, y2, kind])

            panels = [("DEPLOYED", render(rgb, old_lines, b, geom)),
                      ("REBUILT", render(rgb, new_lines, b, geom))]
            if arch:
                panels.append(("ARCHETYPE", render(rgb, arch, b, geom)))
            panels.append(("FUSED", render(rgb, fused, b, geom)))
            if drawn:
                panels.append(("JOSH", render(rgb, drawn, b, geom)))
            rows.append({"id": bid,
                         "addr": lab.get("address", ""),
                         "panels": panels})
            placed = True
            break
        if not placed:
            print(f"  skip #{bid}: no imagery here")

    cells = []
    for r in rows:
        imgs = "".join(
            f'<figure><img src="data:image/jpeg;base64,{jpg}">'
            f"<figcaption>{cap}</figcaption></figure>"
            for cap, jpg in r["panels"])
        cells.append(f'<section><h2>#{r["id"]}'
                     + (f' &middot; {r["addr"]}' if r["addr"] else "")
                     + f"</h2><div class=row>{imgs}</div></section>")

    html = f"""<title>Roof Line Placement Check</title>
<style>
 body{{background:#12161a;color:#e8edf2;font:15px/1.5 system-ui;margin:0;padding:20px}}
 h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#8b97a3;margin-bottom:18px}}
 h2{{font-size:15px;margin:18px 0 6px}}
 .row{{display:flex;gap:10px;flex-wrap:wrap}}
 figure{{margin:0}} img{{max-width:420px;height:auto;display:block;border-radius:4px}}
 figcaption{{color:#8b97a3;font-size:12px;letter-spacing:.08em;margin-top:3px}}
 .leg span{{display:inline-block;margin-right:14px}}
 .leg i{{display:inline-block;width:18px;height:4px;vertical-align:middle;margin-right:5px}}
</style>
<h1>Where the detector puts its lines</h1>
<div class=sub>DEPLOYED is what the live build sees. REBUILT is the retrained
model through the new extraction. JOSH is your drawing, where it exists.</div>
<div class="sub leg">
 <span><i style="background:#1fff7a"></i>ridge</span>
 <span><i style="background:#35b6ff"></i>valley</span>
 <span><i style="background:#ff3b30"></i>cliff</span>
 <span><i style="background:#ffc850"></i>outline</span></div>
{"".join(cells)}"""
    dest = ROOT / "data" / "preview" / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html)
    print(f"wrote {dest}  ({dest.stat().st_size/1e6:.1f} MB, {len(rows)} roofs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

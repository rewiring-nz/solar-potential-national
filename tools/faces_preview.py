"""What a foundation segmentation model sees on each roof, judged by eye.

Josh, after a day of watching a small custom detector fumble clear roofs:
"The images have all the pixel data to clearly show the shape of the roof. It
makes no sense you can't detect that when image recognition models can detect
things in far more detail. You should be able to clearly see the shape of the
roof, obstructions, what are just shadows from surrounding trees etc."

He is right, and this is the test of it. SAM (ViT-B, the checkpoint already in
data/) segments each roof crop with no training on our data at all. Every mask
that sits on the building becomes a candidate FACE -- no line extraction, no
archetypes, no fusion heuristics. The panel pairs it with the faces Josh drew,
which are the standard.

Runs under .venv-sam (torch + segment_anything live there):
    .venv-sam/bin/python tools/faces_preview.py --ids ... --out faces_check.html
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

PAD_M = 4.0
SCALE = 3
FILLS = [(31, 255, 122), (53, 182, 255), (255, 179, 60), (255, 99, 195),
         (170, 120, 255), (120, 235, 235), (255, 235, 90), (140, 255, 140),
         (255, 130, 90), (90, 160, 255)]


def render(rgb, faces, bounds, outline, grey=()):
    """Translucent face fills + boundaries over the crop."""
    import numpy as np
    from PIL import Image, ImageDraw
    h, w = rgb.shape[:2]
    im = Image.fromarray(rgb.astype("uint8")).convert("RGB").resize(
        (w * SCALE, h * SCALE), Image.LANCZOS)
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    minx, miny, maxx, maxy = bounds

    def px(x, y):
        return ((x - minx) / (maxx - minx) * w * SCALE,
                (1 - (y - miny) / (maxy - miny)) * h * SCALE)

    for i, ring in enumerate(faces):
        col = FILLS[i % len(FILLS)]
        pts = [px(x, y) for x, y in ring]
        if len(pts) >= 3:
            d.polygon(pts, fill=col + (70,), outline=col + (255,), width=3)
    for ring in grey:
        pts = [px(x, y) for x, y in ring]
        if len(pts) >= 3:
            d.polygon(pts, fill=(40, 40, 40, 150), outline=(230, 230, 230, 255),
                      width=2)
    im = Image.alpha_composite(im.convert("RGBA"), lay).convert("RGB")
    d2 = ImageDraw.Draw(im)
    if outline is not None:
        d2.line([px(x, y) for x, y in outline.exterior.coords],
                fill=(255, 200, 80), width=2)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, required=True)
    ap.add_argument("--checkpoint", default="data/sam_vit_b.pth")
    ap.add_argument("--out", default="faces_check.html")
    ap.add_argument("--points", type=int, default=24)
    a = ap.parse_args()

    import numpy as np
    import torch
    import geopandas as gpd
    import rasterio
    import rasterio.windows
    import rasterio.features
    from shapely.geometry import shape as shp_shape, Polygon
    from segment_anything import (sam_model_registry, SamAutomaticMaskGenerator,
                                  SamPredictor)
    from src.region_build import area_paths, all_areas

    # SAM's automatic generator hands float64 point grids to torch, which MPS
    # refuses; cast at the seam rather than falling back to CPU
    from segment_anything.utils.transforms import ResizeLongestSide as _RLS
    _oac = _RLS.apply_coords
    _RLS.apply_coords = (lambda self, c, sz:
                         _oac(self, c, sz).astype(np.float32))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(ROOT / a.checkpoint))
    sam.to(device)
    predictor = SamPredictor(sam)

    # THE SPLITTER. SAM glides over low-contrast creases on uniform roofs, so
    # its mask can span two true faces -- Anderson stayed "clearly worse" than
    # Josh's markup for exactly that reason, and merge/drop arbitration cannot
    # fix a face that needed CUTTING. The line detector is the one instrument
    # here that fires on those creases, so: split a face along a confident
    # detected line, and keep the split only if LiDAR says the two sides are
    # different planes. Image proposes the cut, LiDAR approves it.
    line_model = None
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        import train_line_model as _T
        _ck = torch.load(ROOT / "data/models/roof_lines_v4.pt",
                         map_location="cpu", weights_only=False)
        line_model = _T.build_unet(_ck.get("pretrained", False))
        line_model.load_state_dict(_ck["state_dict"])
        line_model.to(device).eval()
    except Exception as e:
        print(f"  (no line model for splitting: {e})")

    labels = {}
    lp = ROOT / "data" / "roof_labels.json"
    if lp.exists():
        labels = json.loads(lp.read_text()).get("buildings", {})

    rows = []
    ctxs = {}
    for bid in a.ids:
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
                    "img": rasterio.open(p["imagery"])}
            ctx = ctxs[name]
            if ctx is None or bid not in ctx["gdf"].index:
                continue
            geom = ctx["gdf"].loc[bid].geometry
            minx, miny, maxx, maxy = geom.bounds
            b = (minx - PAD_M, miny - PAD_M, maxx + PAD_M, maxy + PAD_M)
            win = rasterio.windows.from_bounds(*b, ctx["img"].transform)
            rgb = np.moveaxis(ctx["img"].read([1, 2, 3], window=win,
                                              boundless=True, fill_value=0),
                              0, -1).astype("uint8")
            h, w = rgb.shape[:2]
            if h < 32 or w < 32:
                break

            # COVERAGE-COMPLETION PROMPTING. Josh, on the automatic grid:
            # "A lot of faces are being missed, but it does seem better at
            # detecting where they are and their shape than the line method".
            # SAM answers where it is asked; the grid does not ask everywhere.
            # So ask deliberately: segment, subtract what came back, and ask
            # again at the biggest patch of roof still unaccounted for, until
            # the footprint is covered. Recall stops being luck.
            predictor.set_image(rgb)

            def to_world(ring_px):
                return [(b[0] + x / w * (b[2] - b[0]),
                         b[1] + (1 - y / h) * (b[3] - b[1]))
                        for x, y in ring_px]

            def px_of(pt):
                return np.array([[(pt.x - b[0]) / (b[2] - b[0]) * w,
                                  (1 - (pt.y - b[1]) / (b[3] - b[1])) * h]],
                                dtype=np.float32)

            def mask_to_poly(mask):
                best = None
                for geo, val in rasterio.features.shapes(
                        mask.astype("uint8")):
                    if val != 1:
                        continue
                    poly = Polygon(to_world(geo["coordinates"][0]))
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if poly.is_empty:
                        continue
                    if best is None or poly.area > best.area:
                        best = poly
                return best

            from shapely.ops import unary_union as _uu
            faces = []
            uncovered = geom
            for _ in range(14):
                if uncovered.is_empty or uncovered.area < 3.0:
                    break
                probe = max(getattr(uncovered, "geoms", [uncovered]),
                            key=lambda g2: g2.area)
                if probe.area < 3.0:
                    break
                seed = probe.representative_point()
                mk, sc, _ = predictor.predict(
                    point_coords=px_of(seed),
                    point_labels=np.array([1]), multimask_output=True)
                pick = None
                for i2 in np.argsort(-sc):
                    poly = mask_to_poly(mk[i2])
                    if poly is None:
                        continue
                    clipped = poly.intersection(geom)
                    if clipped.is_empty or clipped.area < 2.0:
                        continue
                    if clipped.area > 0.85 * geom.area:
                        continue      # the whole building, not a face
                    pick = clipped
                    break
                if pick is None:
                    uncovered = uncovered.difference(seed.buffer(0.8))
                    continue
                fresh = pick.difference(_uu(faces)) if faces else pick
                for g2 in getattr(fresh, "geoms", [fresh]):
                    if g2.geom_type == "Polygon" and g2.area >= 2.0:
                        faces.append(g2.simplify(0.15))
                uncovered = uncovered.difference(pick.buffer(0.05))

            # WHAT SAM CANNOT KNOW, LIDAR ARBITRATES. Josh flagged the two
            # big commercial roofs: rooftop plant segmented faithfully and then
            # wrongly promoted to faces, one mask spanning two true faces, and
            # ragged boundaries. SAM sees shape; only the point cloud knows
            # which regions share a plane and which sit ON the roof rather
            # than being roof.
            #
            #   RE-TILE   assign a 0.2 m grid of the footprint to the best
            #             covering mask, so faces PARTITION the roof -- shared
            #             edges become clean by construction
            #   CLUTTER   a small face elevated above its neighbour's plane is
            #             plant, not roof: dropped, cells rejoin the neighbour
            #   MERGE     adjacent faces whose fitted planes agree are one
            #             face SAM happened to see twice
            try:
                pts_r = None
                from src.pointcloud_source import PointCloudSource as _PCS
                if "pc" not in ctx:
                    ctx["pc"] = _PCS(max_cached_tiles=3)
                from src.roof_partition import (top_surface as _ts,
                                                _points_in as _pin,
                                                _fit_plane_robust as _fpr,
                                                _slope_aspect as _sa)
                pts_r = _ts(ctx["pc"].points_in_bbox(minx - 1, miny - 1,
                                                     maxx + 1, maxy + 1,
                                                     building_only=True))
            except Exception as e:
                print(f"  (LiDAR unavailable for #{bid}: {e})")
                pts_r = None

            import shapely as _shp
            from scipy.spatial import cKDTree as _KD
            res = 0.2
            gxs = np.arange(b[0], b[2] + res, res)
            gys = np.arange(b[1], b[3] + res, res)
            GX, GY = np.meshgrid(gxs, gys)
            inside = _shp.contains_xy(geom, GX.ravel(), GY.ravel()
                                      ).reshape(GX.shape)
            lab = np.full(GX.shape, -1, dtype=int)
            for i2, f in enumerate(faces):
                m2 = _shp.contains_xy(f, GX.ravel(), GY.ravel()
                                      ).reshape(GX.shape) & inside
                lab[m2 & (lab < 0)] = i2
            # unclaimed inside cells join the nearest claimed cell's face
            un = inside & (lab < 0)
            if un.any() and (lab >= 0).any():
                cl = np.argwhere(lab >= 0)
                tree = _KD(np.c_[GX[lab >= 0], GY[lab >= 0]])
                _, nn = tree.query(np.c_[GX[un], GY[un]])
                lab[un] = lab[lab >= 0][nn]

            def face_polys(lab_grid):
                from rasterio.transform import from_origin
                tr = from_origin(b[0] - res / 2, gys[-1] + res / 2, res, res)
                out2 = {}
                for rid in np.unique(lab_grid):
                    if rid < 0:
                        continue
                    mask2 = np.flipud(lab_grid == rid).astype("uint8")
                    best = None
                    for geo, val in rasterio.features.shapes(mask2,
                                                             transform=tr):
                        if val != 1:
                            continue
                        poly = Polygon(geo["coordinates"][0])
                        if best is None or poly.area > best.area:
                            best = poly
                    if best is not None and best.area >= 2.0:
                        out2[int(rid)] = best.intersection(geom)
                return out2

            def planes_of(polys):
                out2 = {}
                if pts_r is None or not len(pts_r):
                    return out2
                for rid, poly in polys.items():
                    sub = _pin(poly, pts_r)
                    if len(sub) >= 10:
                        pl = _fpr(sub)
                        if pl is not None:
                            out2[rid] = (pl, float(np.median(sub[:, 2])))
                return out2

            polys = face_polys(lab)
            planes = planes_of(polys)
            obstructions = []
            # clutter: small, sitting above the plane of what surrounds it
            for rid, poly in list(polys.items()):
                if poly.is_empty or poly.area > 15.0 or rid not in planes:
                    continue
                ring2 = poly.buffer(1.2).difference(poly)
                if pts_r is None:
                    continue
                around = _pin(ring2.intersection(geom), pts_r)
                if len(around) < 10:
                    continue
                zme = planes[rid][1]
                if zme - float(np.median(around[:, 2])) > 0.45:
                    obstructions.append(poly)
                    # cells rejoin whoever surrounds them
                    sel = lab == rid
                    lab[sel] = -1
                    cl2 = inside & (lab >= 0)
                    tree = _KD(np.c_[GX[cl2], GY[cl2]])
                    _, nn = tree.query(np.c_[GX[sel], GY[sel]])
                    lab[sel] = lab[cl2][nn]
            polys = face_polys(lab)
            planes = planes_of(polys)
            # merge neighbours whose planes agree
            changed = True
            while changed:
                changed = False
                rids = list(polys)
                for i3 in range(len(rids)):
                    for j3 in range(i3 + 1, len(rids)):
                        r1, r2 = rids[i3], rids[j3]
                        if r1 not in polys or r2 not in polys:
                            continue
                        if r1 not in planes or r2 not in planes:
                            continue
                        if polys[r1].buffer(0.3).intersection(
                                polys[r2]).is_empty:
                            continue
                        s1, a1 = _sa(planes[r1][0])
                        s2, a2 = _sa(planes[r2][0])
                        da = abs(a1 - a2) % 360
                        da = min(da, 360 - da)
                        if abs(s1 - s2) < 3.0 and (da < 15 or max(s1, s2) < 4):
                            lab[lab == r2] = r1
                            polys = face_polys(lab)
                            planes = planes_of(polys)
                            changed = True
                            break
                    if changed:
                        break

            # split faces along confident detected lines, LiDAR approving
            if line_model is not None and pts_r is not None and len(pts_r):
                try:
                    from src.line_extract import extract as _lex, \
                        clip_to as _lclip
                    from shapely.ops import split as _shsplit
                    from shapely.geometry import LineString as _LS2
                    ph, pw2 = (-h) % 16, (-w) % 16
                    arr2 = np.pad(rgb, ((0, ph), (0, pw2), (0, 0)))
                    x2 = torch.from_numpy(arr2).float().permute(2, 0, 1)[None] / 255.0
                    with torch.no_grad():
                        pr2 = torch.sigmoid(line_model(x2.to(device))
                                            )[0].cpu().numpy()[:, :h, :w]

                    def tw2(px2, py2):
                        return (b[0] + px2 / w * (b[2] - b[0]),
                                b[1] + (1 - py2 / h) * (b[3] - b[1]))

                    cut_lines = [r3 for r3 in _lclip(_lex(pr2, tw2), geom)
                                 if r3["score"] >= 0.5]
                    # the detector sees one crease in pieces (on the twin: a
                    # ridge as three 5 m segments), and each piece alone covers
                    # a quarter of the chord it proposes, failing the coverage
                    # gate. Merge collinear pieces in world space first.
                    merged3 = True
                    while merged3:
                        merged3 = False
                        for i5 in range(len(cut_lines)):
                            if cut_lines[i5] is None:
                                continue
                            for j5 in range(i5 + 1, len(cut_lines)):
                                if cut_lines[j5] is None:
                                    continue
                                s1_ = cut_lines[i5]["seg"]
                                s2_ = cut_lines[j5]["seg"]
                                a5 = np.array(s1_[:2]); b5 = np.array(s1_[2:])
                                c5 = np.array(s2_[:2]); e5 = np.array(s2_[2:])
                                d5 = b5 - a5
                                L5 = np.hypot(*d5)
                                L6 = np.hypot(*(e5 - c5))
                                if L5 < 1e-6 or L6 < 1e-6:
                                    continue
                                ref = (a5, d5 / L5) if L5 >= L6 else \
                                    (c5, (e5 - c5) / L6)
                                ang5 = np.degrees(
                                    np.arctan2(d5[1], d5[0])
                                    - np.arctan2((e5 - c5)[1], (e5 - c5)[0]))
                                ang5 = abs(ang5) % 180
                                if min(ang5, 180 - ang5) > 8:
                                    continue
                                nr5 = np.array([-ref[1][1], ref[1][0]])
                                if any(abs((q5 - ref[0]) @ nr5) > 0.35
                                       for q5 in (a5, b5, c5, e5)):
                                    continue
                                ts = sorted((q5 - ref[0]) @ ref[1]
                                            for q5 in (a5, b5, c5, e5))
                                t1s = sorted(((q5 - ref[0]) @ ref[1]
                                              for q5 in (a5, b5)))
                                t2s = sorted(((q5 - ref[0]) @ ref[1]
                                              for q5 in (c5, e5)))
                                if max(t1s[0], t2s[0]) - min(t1s[1], t2s[1]) \
                                        > 1.5:
                                    continue
                                na = ref[0] + ts[0] * ref[1]
                                nb = ref[0] + ts[-1] * ref[1]
                                cut_lines[i5] = {
                                    "seg": [na[0], na[1], nb[0], nb[1]],
                                    "score": max(cut_lines[i5]["score"],
                                                 cut_lines[j5]["score"]),
                                    "kind": cut_lines[i5]["kind"]}
                                cut_lines[j5] = None
                                merged3 = True
                        cut_lines = [c5 for c5 in cut_lines if c5 is not None]
                    cut_lines.sort(key=lambda r3: -r3["score"])
                    import os as _os
                    _dbg = _os.environ.get("SOLAR_DEBUG_SPLIT") == "1"
                    if _dbg:
                        print(f"    [split] {len(cut_lines)} merged lines, "
                              f"{len(polys)} faces")
                    for r3 in cut_lines:
                        x1c, y1c, x2c, y2c = r3["seg"]
                        d4 = np.array([x2c - x1c, y2c - y1c])
                        L4 = np.hypot(*d4)
                        if L4 < 2.5:
                            continue
                        u4 = d4 / L4
                        seg4 = _LS2([(x1c, y1c), (x2c, y2c)])
                        far = _LS2([(x1c - u4[0] * 300, y1c - u4[1] * 300),
                                    (x2c + u4[0] * 300, y2c + u4[1] * 300)])
                        for rid, poly in list(polys.items()):
                            if poly.is_empty or rid not in planes:
                                continue
                            # shapely only splits on a FULL crossing, and a
                            # hip ridge is interior by nature -- the first
                            # version of this pass never split anything, which
                            # is why identical twin roofs kept coming out
                            # completely different. So cut with the CHORD (the
                            # infinite extension clipped to this face), gated
                            # by the detection actually covering most of it --
                            # the _covers_cell rule, in its right home at last.
                            chord = far.intersection(poly)
                            clen = sum(g4.length for g4 in
                                       getattr(chord, "geoms", [chord])
                                       if g4.geom_type == "LineString")
                            if clen < 2.0:
                                continue
                            cov = sum(g4.length for g4 in
                                      getattr(chord, "geoms", [chord])
                                      if g4.geom_type == "LineString"
                                      for _ in [0]
                                      ) and seg4.buffer(0.7).intersection(
                                          chord).length
                            if _dbg:
                                print(f"    [split] face {rid} "
                                      f"a={polys[rid].area:.0f} "
                                      f"clen={clen:.1f} cov={cov:.1f}")
                            if cov < 0.5 * clen:
                                continue
                            try:
                                pieces = [g4 for g4 in getattr(
                                    poly.difference(far.buffer(0.02)),
                                    "geoms",
                                    [poly.difference(far.buffer(0.02))])
                                    if g4.geom_type == "Polygon"
                                    and g4.area >= 2.5]
                            except Exception:
                                continue
                            if _dbg:
                                print(f"    [split] pieces={len(pieces)}")
                            if len(pieces) < 2:
                                continue
                            fits = []
                            for g4 in pieces[:3]:
                                sub4 = _pin(g4, pts_r)
                                pl4 = _fpr(sub4) if len(sub4) >= 10 else None
                                if pl4 is None:
                                    break
                                fits.append(_sa(pl4))
                            if len(fits) < 2:
                                continue
                            (s1_, a1_), (s2_, a2_) = fits[0], fits[1]
                            da_ = abs(a1_ - a2_) % 360
                            da_ = min(da_, 360 - da_)
                            # different planes -> the cut was real
                            if abs(s1_ - s2_) >= 3.0 or \
                                    (min(s1_, s2_) >= 4 and da_ >= 20):
                                nid = max(list(polys) + [0]) + 1
                                polys.pop(rid)
                                for k4, g4 in enumerate(pieces):
                                    polys[nid + k4] = g4
                                planes = planes_of(polys)
                except Exception as e:
                    print(f"  (split pass failed on #{bid}: {e})")

            faces = [poly.simplify(0.3) for poly in polys.values()
                     if not poly.is_empty and poly.area >= 2.0]
            face_rings = [list(f.exterior.coords) for f in faces]
            obs_rings = [list(o.exterior.coords) for o in obstructions]

            drawn = []
            lab = labels.get(str(bid)) or {}
            for f in lab.get("faces") or []:
                try:
                    drawn.append([(q[0], q[1]) for q in f["ring"]])
                except Exception:
                    pass

            panels = [("SAM+LIDAR FACES",
                       render(rgb, face_rings, b, geom, grey=obs_rings))]
            if drawn:
                panels.append(("JOSH FACES", render(rgb, drawn, b, geom)))
            rows.append({"id": bid, "addr": lab.get("address", ""),
                         "n": len(face_rings), "panels": panels})
            print(f"  #{bid}: {len(face_rings)} faces, "
                  f"{100 * sum(f.area for f in faces) / geom.area:.0f}% covered")
            placed = True
            break
        if not placed:
            print(f"  skip #{bid}")

    cells = []
    for r in rows:
        imgs = "".join(
            f'<figure><img src="data:image/jpeg;base64,{jpg}">'
            f"<figcaption>{cap}</figcaption></figure>"
            for cap, jpg in r["panels"])
        cells.append(f'<section><h2>#{r["id"]}'
                     + (f' &middot; {r["addr"]}' if r["addr"] else "")
                     + f"</h2><div class=row>{imgs}</div></section>")

    html = f"""<title>Roof Faces From SAM</title>
<style>
 body{{background:#12161a;color:#e8edf2;font:15px/1.5 system-ui;margin:0;padding:20px}}
 h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#8b97a3;margin-bottom:18px}}
 h2{{font-size:15px;margin:18px 0 6px}}
 .row{{display:flex;gap:10px;flex-wrap:wrap}}
 figure{{margin:0}} img{{max-width:430px;height:auto;display:block;border-radius:4px}}
 figcaption{{color:#8b97a3;font-size:12px;letter-spacing:.08em;margin-top:3px}}
</style>
<h1>What a foundation segmentation model sees</h1>
<div class=sub>SAM, zero training on our data: every coloured region is a face
it found on the building. JOSH FACES is your markup, the standard.</div>
{"".join(cells)}"""
    dest = ROOT / "data" / "preview" / a.out
    dest.write_text(html)
    print(f"wrote {dest}  ({dest.stat().st_size/1e6:.1f} MB, {len(rows)} roofs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

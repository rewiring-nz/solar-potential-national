"""Precompute selected roof faces per building, for the build to read.

The selector (src/face_candidates + the scorer) needs SAM and torch, which
live in .venv-sam and have no business inside the build environment. So faces
are computed here, ahead of time, one JSON per building -- exactly the seam
vision_lines already uses -- and roof_partition reads them behind a flag, the
same way it reads Josh's drawn faces.

Written per building: the winning candidate's face rings (NZTM), which reading
won, its evidence score, and both candidates' scores, so the build can apply
its own shipping threshold without recomputing anything.

    .venv-sam/bin/python tools/predict_faces.py --region pilot --ids ...
    .venv-sam/bin/python tools/predict_faces.py --bench
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

OUT = ROOT / "data" / "selected_faces"
PAD_M = 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="pilot")
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--bench", action="store_true",
                    help="every roof in the benchmark set")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import numpy as np
    import torch
    import geopandas as gpd
    import rasterio
    import rasterio.windows
    from segment_anything import sam_model_registry, SamPredictor
    from segment_anything.utils.transforms import ResizeLongestSide as _RLS
    from src.region_build import area_paths
    from src.pointcloud_source import PointCloudSource
    from src.roof_partition import top_surface
    from src.face_candidates import (sam_faces, line_faces, score_candidate,
                                     evidence_map, lidar_faces,
                                     hypothesis_faces)
    import train_line_model as T

    _oac = _RLS.apply_coords
    _RLS.apply_coords = (lambda self, c, sz:
                         _oac(self, c, sz).astype(np.float32))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(ROOT / "data/sam_vit_b.pth"))
    sam.to(device)
    predictor = SamPredictor(sam)
    ck = torch.load(ROOT / "data/models/roof_lines_v5.pt",
                    map_location="cpu", weights_only=False)
    lm = T.build_unet(ck.get("pretrained", False), out_channels=3)
    lm.load_state_dict(ck["state_dict"])
    lm.to(device).eval()
    # ROLE SPLIT: v5 drives the line-extraction stack (its statistics are
    # what every threshold downstream was calibrated on -- v6 there
    # collapsed LINE agreement 0.754 -> 0.422); v6, which finally SEES
    # hips (combined crease F1 0.736 vs 0.446), is the EVIDENCE model:
    # the scorer's eyes and the form contest's seam support.
    ck6 = torch.load(ROOT / "data/models/roof_lines_v6.pt",
                     map_location="cpu", weights_only=False)
    lm6 = T.build_unet(ck6.get("pretrained", False), out_channels=4)
    try:
        lm6.load_state_dict(ck6["state_dict"])
        lm6.to(device).eval()
    except Exception:
        lm6 = None

    ids = a.ids
    if a.bench and not ids:
        ids = [int(x) for x in
               (ROOT / "data/bench_ids.txt").read_text().split()]

    p = area_paths(a.region)
    dd = p["dir"] / "building_outlines_dedup.geojson"
    gdf = gpd.read_file(dd if dd.exists() else p["outlines"]
                        ).set_index("building_id", drop=False)
    img = rasterio.open(p["imagery"])
    pc = PointCloudSource(max_cached_tiles=3)
    if not ids:
        ids = [int(x) for x in gdf["building_id"]]
    if a.limit:
        ids = ids[:a.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    done = 0
    resume = os.environ.get("SOLAR_PREDICT_RESUME", "0") == "1"
    for bid in ids:
        if resume and (OUT / f"{bid}.json").exists():
            done += 1
            continue
        if bid not in gdf.index:
            continue
        geom = gdf.loc[bid].geometry
        minx, miny, maxx, maxy = geom.bounds
        b = (minx - PAD_M, miny - PAD_M, maxx + PAD_M, maxy + PAD_M)
        win = rasterio.windows.from_bounds(*b, img.transform)
        try:
            rgb = np.moveaxis(img.read([1, 2, 3], window=win, boundless=True,
                                       fill_value=0), 0, -1).astype("uint8")
        except Exception:
            continue
        h, w = rgb.shape[:2]
        if h < 32 or w < 32:
            continue
        pts = top_surface(pc.points_in_bbox(minx - 1, miny - 1,
                                            maxx + 1, maxy + 1,
                                            building_only=True))

        def to_px(x, y):
            return ((x - b[0]) / (b[2] - b[0]) * w,
                    (1 - (y - b[1]) / (b[3] - b[1])) * h)

        def inv_px(px2, py2):
            return (b[0] + px2 / w * (b[2] - b[0]),
                    b[1] + (1 - py2 / h) * (b[3] - b[1]))

        # NEAR-FLAT ROOFS ARE NOT THE SELECTOR'S TO SHIP. Josh, on the first
        # region render: the flat, obstruction-heavy commercials came out as
        # arbitrary webs -- the line net polygonises plant edges, and the
        # scorer cannot tell, because a flat plane fits every partition of
        # itself. The proven LiDAR path already handles these acceptably on
        # the live map, so the selector writes nothing and the build falls
        # through to it.
        # THE IMAGERY OUTVOTES THE FLATNESS DEFER. #4734914 sits at 0.87 m of
        # LiDAR spread -- "flat" -- yet its hips are plainly visible and v4
        # fires ten lines at score 1.0 along them; deferring it handed the
        # roof to the old path, which invented diagonals ("Clearly wrong
        # building lines even though this is a very easy roof visually").
        # A roof only defers when LiDAR reads flat AND the detector sees
        # nothing worth drawing.
        strong_lines = 0
        try:
            ph0, pw0 = (-h) % 16, (-w) % 16
            arr0 = np.pad(rgb, ((0, ph0), (0, pw0), (0, 0)))
            x0 = torch.from_numpy(arr0).float().permute(2, 0, 1)[None] / 255.0
            with torch.no_grad():
                pr = torch.sigmoid(lm(x0.to(device)))[0].cpu().numpy()[:, :h, :w]
            from src.line_extract import extract as _lex0, clip_to as _lclip0
            strong_lines = sum(1 for r0 in _lclip0(_lex0(pr, lambda px0, py0: (
                b[0] + px0 / w * (b[2] - b[0]),
                b[1] + (1 - py0 / h) * (b[3] - b[1]))), geom)
                if r0["score"] >= 0.6)
        except Exception:
            pr = None

        # ...and flatness is judged PER PART: #5371128 is a flat block joined
        # to a gabled hall, and a whole-building fit read the pair as flat.
        # Defer only when every reflex-split part reads flat -- 118 of 152
        # deferred under the whole-building test, which was the test failing,
        # not the roofs.
        if pts is not None and len(pts) > 80:
            import numpy as _np
            from src.roof_partition import (_fit_plane_robust, _slope_aspect,
                                            _points_in)
            from src.face_candidates import rect_parts
            all_flat = True
            for part in rect_parts(geom):
                sub = _points_in(part, pts)
                if len(sub) < 40:
                    continue
                z = sub[:, 2]
                spread = float(_np.percentile(z, 95) - _np.percentile(z, 5))
                pl0 = _fit_plane_robust(sub)
                sl0 = _slope_aspect(pl0)[0] if pl0 is not None else 99.0
                # a flat membrane under heavy plant reads "not flat" by
                # spread alone -- the ducting IS the spread. #5370338 (223 m2
                # of ducting on a flat roof, the class this defer was built
                # for) sailed past it and the selector shipped duct-top
                # faces. A dominant flat plane with outliers is still flat.
                from src.roof_partition import _inlier_fraction as _inl0
                flat_inl = _inl0(sub, pl0) if pl0 is not None else 0.0
                if not (sl0 < 4.5 and (spread < 1.2 or flat_inl > 0.55)):
                    all_flat = False
                    break
            # the strong-lines exemption exists for #4734914, a 430 m2
            # house with visible hips that LiDAR reads flat. On an
            # industrial-scale flat, "strong lines" are duct edges: #4722059
            # (10,000 m2) got an 85-face line reading whose residual-fill
            # partition ate a 2-hour build budget. Houses keep the
            # exemption; big flats defer regardless.
            if all_flat and (strong_lines < 3 or geom.area > 800):
                # a stale file from an earlier chain must not outlive the
                # decision to defer -- delete it or the build keeps shipping
                # the very reading the defer just refused
                (OUT / f"{bid}.json").unlink(missing_ok=True)
                continue

        try:
            f_sam, _ = sam_faces(predictor, rgb, geom, b, pts)
        except Exception:
            f_sam = []
        try:
            f_line, pr = line_faces(lm, device, rgb, geom, b, pts, bid)
        except Exception:
            f_line, pr = [], None
        if pr is None:
            ph, pw = (-h) % 16, (-w) % 16
            arr = np.pad(rgb, ((0, ph), (0, pw), (0, 0)))
            x2 = torch.from_numpy(arr).float().permute(2, 0, 1)[None] / 255.0
            with torch.no_grad():
                pr = torch.sigmoid(lm(x2.to(device)))[0].cpu().numpy()[:, :h, :w]
        # evidence = the calibrated v5 landscape PLUS the one thing v5
        # cannot see: v6's dedicated hip channel (activation along Josh's
        # drawn hips 0.16-0.24 -> 0.74-0.79). Swapping the whole map to v6
        # cost 0.021 of picked agreement -- its hotter statistics
        # mis-calibrate the edge term -- so only the new signal joins.
        ev = pr.max(axis=0)
        if lm6 is not None:
            import torch as _t6
            _ph, _pw = (-h) % 16, (-w) % 16
            _arr = np.pad(rgb, ((0, _ph), (0, _pw), (0, 0)))
            _x6 = _t6.from_numpy(_arr).float().permute(2, 0, 1)[None] / 255.0
            with _t6.no_grad():
                _p6 = _t6.sigmoid(lm6(_x6.to(device)))[0].cpu().numpy()[:, :h, :w]
            if _p6.shape[0] >= 4:
                ev = np.maximum(ev, _p6[3])
        P = evidence_map(ev, rgb)
        sc_sam = score_candidate(f_sam, geom, P, to_px, pts, inv_px) \
            if f_sam else 0.0
        sc_line = score_candidate(f_line, geom, P, to_px, pts, inv_px) \
            if f_line else 0.0
        # LIDAR-FIRST reading: normals region-grown on the point cloud.
        # Where the imagery is blind -- tree shadow (#4735106), cluttered
        # flats -- both imagery families fail their gates and the roof used
        # to fall to the old RANSAC path. This family exists for exactly
        # those roofs. On roofs imagery reads fine it may win outright only
        # by a clear margin (+0.08, measured on the 32 drawn roofs as the
        # no-harm threshold: 3-way naive picking LOST 0.03 of agreement, so
        # deference to the imagery winner is the measured default).
        try:
            f_lid = lidar_faces(pts, geom)
        except Exception:
            f_lid = []
        sc_lid = score_candidate(f_lid, geom, P, to_px, pts, inv_px) \
            if f_lid else 0.0
        if not f_sam and not f_line and not f_lid:
            continue
        # Josh: "If you are not detecting clear lines you should not just
        # randomly draw them." A line-winner must stand on clear lines --
        # length-weighted activation along its interior edges >= 0.5.
        # Calibrated on his verdicts: the two webs he flagged sit at 0.43 and
        # 0.47, Anderson at 0.87. A roof that fails falls to SAM if SAM earned
        # a score, else to no file and the old pipeline -- deferring a decent
        # roof costs little, shipping a web costs a flag.
        line_ok = bool(f_line)
        if line_ok:
            from src.line_extract import _line_mean as _lm2
            import shapely.geometry as _sg
            rim2 = geom.exterior.buffer(0.5)
            sup2 = len2 = 0.0
            for f in f_line:
                cs = list(f.exterior.coords)
                for aa, bb in zip(cs, cs[1:]):
                    seg2 = _sg.LineString([aa, bb])
                    Li = seg2.difference(rim2).length
                    if Li < 0.5:
                        continue
                    pa2 = np.array(to_px(*aa))
                    pb2 = np.array(to_px(*bb))
                    sup2 += Li * _lm2(P, pa2, pb2)
                    len2 += Li
            line_ok = len2 > 2 and (sup2 / len2) >= 0.5
        # Imagery-degraded roofs judge themselves: #4735106 sits under tree
        # shadow at mean in-footprint luminance 96 with 15% deep-dark pixels,
        # where normal roofs read 130-145 with ~0%. Under shadow the imagery
        # candidates' scores are not trustworthy evidence, so LiDAR needs no
        # margin there -- near-parity suffices.
        try:
            from rasterio.features import geometry_mask as _gm
            import rasterio.transform as _rt
            _tr = _rt.from_bounds(*b, w, h)
            _m = ~_gm([geom], out_shape=(h, w), transform=_tr)
            _lum = rgb.astype(np.float32).mean(axis=2)[_m]
            dark = _lum.mean() < 110 or (_lum < 70).mean() > 0.08
        except Exception:
            dark = False
        # SIMPLE FORMS FOR SIMPLE ROOFS. Josh on the side-by-side panel
        # (14 Sep): "Neither are great. but hypothesis is slightly better...
        # at least the lines are simpler and there are less extra
        # unnecessary lines." Ships for the weak class only: residential
        # scale, no incumbent reading scoring >= 0.50, and the form itself
        # decisive (conf >= 0.55, with aspect agreement in the score so a
        # pyramid claim needs four tilt directions in the LiDAR).
        # Area cap raised 450 -> 2000 (17 Sep). The 450 fence made #4735292
        # (a ~550 m2 textbook pyramid Josh flagged) fall to a 6-fragment SAM
        # reading; hypothesis, once allowed, won the score contest and
        # produced the 4 correct faces. Bench A/B 450 vs 2000: every markup
        # metric identical, +147 panels. The conf/incumbent gates below are
        # what protect against over-calling pyramids; the area cap was not.
        f_hyp, conf_hyp = [], 0.0
        if geom.area <= float(os.environ.get("SOLAR_HYP_MAX_AREA", "2000")):
            try:
                f_hyp = hypothesis_faces(pts, geom, P, to_px)
                conf_hyp = getattr(hypothesis_faces, "last_confidence", 0.0)
            except Exception:
                f_hyp, conf_hyp = [], 0.0
        # Scored whenever a form exists: the weak-evidence branch below
        # ships forms under 0.55 confidence, and a 0.0 here would write a
        # file the build's own SELECTED_MIN_SCORE (0.30) then rejects --
        # the selection would silently fall through to the old path.
        sc_hyp = score_candidate(f_hyp, geom, P, to_px, pts, inv_px) \
            if f_hyp else 0.0
        # WHEN NOTHING READS CLEARLY, GUESS SIMPLE (Josh, 14 Sep). A
        # reading scoring 0.33 is not knowledge, and shipping it ships a
        # jagged 15-vertex guess: #4734994 (sam 0.33) and #4735106 (lidar
        # 0.37) are the two roofs on his flagged sheet whose boundaries
        # follow nothing visible. Below 0.45 no detector has earned the
        # roof, so a merely plausible simple form (0.40) is preferred to
        # a confident-looking mess. Above that the old bar stands.
        # DEFAULT OFF (SOLAR_HYP_WEAK=1 to try it). Measured 18 Sep: it
        # does make these roofs simpler -- #4734994's 15-vertex sam blob
        # becomes 3 straight faces -- but rendered against the imagery
        # they are simpler AND still wrong, and the bench cannot see the
        # change at all (its roofs are Josh's, where his markup governs).
        # Simplicity is his instruction; shipping an unmeasurable
        # behaviour change across 15k buildings is how five regressions
        # reached a deploy. It waits for evidence, not for agreement.
        _weak = (os.environ.get("SOLAR_HYP_WEAK") == "1"
                 and max(sc_sam, sc_line, sc_lid) < 0.45)
        if f_hyp and (conf_hyp >= 0.55
                      and (max(sc_sam, sc_line, sc_lid) < 0.50
                           or (os.environ.get("SOLAR_HYP_COMPETE") == "1"
                               and sc_hyp > max(sc_sam, sc_line, sc_lid)))
                      or (conf_hyp >= 0.40 and _weak)):
            # SOLAR_HYP_COMPETE: experimental -- a confident simple form may
            # also beat a >=0.50 incumbent on raw score, not just fill the
            # weak class. Benched before any default change.
            pick, faces, score = "hypothesis", f_hyp, sc_hyp
        elif f_lid and dark and sc_lid >= 0.30 \
                and sc_lid >= max(sc_sam, sc_line) - 0.10:
            pick, faces, score = "lidar", f_lid, sc_lid
        elif f_lid and sc_lid > max(sc_sam, sc_line) + 0.08:
            pick, faces, score = "lidar", f_lid, sc_lid
        elif line_ok and sc_line >= sc_sam:
            pick, faces, score = "line", f_line, sc_line
        elif f_sam and sc_sam >= 0.30:
            pick, faces, score = "sam", f_sam, sc_sam
        elif line_ok:
            pick, faces, score = "line", f_line, sc_line
        elif f_lid and sc_lid >= 0.30:
            # both imagery readings refused -- the roof the old path used to
            # inherit. The regularised LiDAR reading ships instead.
            pick, faces, score = "lidar", f_lid, sc_lid
        elif (_hyp := hypothesis_faces(pts, geom, P, to_px)) and \
                score_candidate(_hyp, geom, P, to_px, pts, inv_px) >= 0.28:
            # last resort before the old path's webs: the SIMPLEST roof form
            # consistent with the evidence. Josh, 14 Sep: "Roofs are simpler
            # shapes that you are seeming to guess" -- when nothing reads
            # clearly, guess simple, not elaborate.
            pick, faces = "hypothesis", _hyp
            score = score_candidate(_hyp, geom, P, to_px, pts, inv_px)
        elif f_sam:
            # a pitched building must not fall back to the old path's webs
            # (#5372567: both candidates dropped, the old pipeline drew "lots
            # of incorrect lines"). SAM's honest partial ships if its faces
            # are good QUALITY even at low coverage -- score is quality x
            # coverage, so divide coverage back out.
            cov = min(1.0, sum(f.area for f in f_sam) / max(geom.area, 1e-9))
            if cov > 0.15 and sc_sam / max(cov, 1e-9) >= 0.55:
                pick, faces, score = "sam", f_sam, sc_sam
            else:
                continue
        else:
            continue
        (OUT / f"{bid}.json").write_text(json.dumps({
            "source": pick, "score": round(score, 3),
            "score_sam": round(sc_sam, 3), "score_line": round(sc_line, 3),
            "score_lidar": round(sc_lid, 3),
            "faces": [[[round(v, 2) for v in xy]
                       for xy in f.exterior.coords] for f in faces]}))
        done += 1
        if done % 20 == 0:
            print(f"  {done}...")
    print(f"{done} buildings -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

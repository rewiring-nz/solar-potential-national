"""
Are the drawn lines where the roof actually is, or where the PHOTO puts it?

Josh traces on an orthophoto. An aerial photo is taken at an angle, so a tall
roof leans away from nadir and appears displaced from its true ground position
-- which is exactly what he noticed as "the building outline is not aligned with
the rooftop". The outline is not the thing that moved; the picture is.

That has a consequence for the labels themselves. If the roof in the photo sits
a metre north of where the roof really is, then a ridge traced on the photo is
recorded a metre north of the real ridge. It would look perfect on screen and
still be a metre wrong in NZTM -- and every score computed against LiDAR-derived
geometry would be understated by that much, uniformly, for reasons that have
nothing to do with the segmenter.

So: for each labelled building, find the translation of the drawn lines that
best matches the predicted ones, and look at the distribution.

  A consistent non-zero shift, in a direction that varies with position in the
  survey, means photo lean is in the labels. It is correctable at ingest and
  the baseline should be restated.

  Shifts scattered with no agreement mean the labels are fine and the
  disagreement is genuine -- the segmenter and the labeller really do differ.

The distinction matters enough to measure before drawing conclusions from a
52% F1, because only one of these two worlds needs the model fixed.

ANSWER, over Josh's first 41 labelled buildings (2 Sep 2026):

    mean shift              +0.04, -0.05 m
    directional coherence   0.10
    height vs shift         r = +0.066   (short 0.73 m, tall 0.71 m)

NO LEAN. Both tests are needed and only together are they conclusive: a mean
of zero rules out a uniform offset, but lean points away from whatever was
under the aircraft, so across a survey its DIRECTION varies and cancels while
its MAGNITUDE grows with height. No height relationship means no lean.

A first pass said the opposite -- r = +0.569, tall roofs wanting 1.27 m against
0.50 m -- and was wrong, resting on the 10 rows that survived a truncated log.
Rows are written to data/label_registration.json now rather than trusted from
stdout.

So the line F1 is a real measurement of the segmenter, and the median 0.71 m
residual is drawing precision plus genuine error.

Do NOT read the best-shifted F1 as achievable. It fits two free parameters per
building over 49 candidate positions on noisy data, so most of that gain is
overfitting, not a correction anyone could apply.

Usage:
    python tools/check_label_registration.py
    python tools/check_label_registration.py --max-shift 2.0 --step 0.25
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

LABELS = ROOT / "data" / "roof_labels.json"


def shift_lines(lines, dx, dy):
    return [[[p[0] + dx, p[1] + dy] for p in seg] for seg in lines]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--max-shift", type=float, default=1.5)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--tol", type=float, default=0.75)
    a = ap.parse_args()

    import numpy as np
    import geopandas as gpd
    import rasterio
    from src.region_build import area_paths
    from src.roof_segmentation import segment_building_best
    from src.pointcloud_source import PointCloudSource
    from score_geometry import (line_scores, predicted_lines_from_facets,
                                _line_points)

    if not LABELS.exists():
        print("no labels yet")
        return 2
    labels = json.loads(LABELS.read_text())["buildings"]
    ids = a.ids or sorted(int(k) for k in labels)

    pc = PointCloudSource()
    ctxs = {}
    steps = int(a.max_shift / a.step)
    rows = []

    print(f"best translation of the DRAWN lines onto the predicted ones\n")
    print(f"{'building':>10}{'F1 as-is':>10}{'F1 shifted':>12}"
          f"{'best shift':>14}{'gain':>8}")

    for bid in ids:
        lab = labels[str(bid)]
        area = lab.get("area")
        if area not in ctxs:
            p = area_paths(area)
            if not (p["outlines"].exists() and p["dsm"].exists()):
                ctxs[area] = None
            else:
                ctxs[area] = {
                    "gdf": gpd.read_file(p["outlines"]).set_index("building_id", drop=False),
                    "dsm": rasterio.open(p["dsm"]),
                    "img": rasterio.open(p["imagery"]) if p["imagery"].exists() else None,
                }
        ctx = ctxs[area]
        if not ctx or bid not in ctx["gdf"].index:
            continue
        geom = ctx["gdf"].loc[bid].geometry

        true_lines = [_line_points(l) for l in lab.get("lines", [])
                      if l.get("kind") != "outline"]
        true_lines = [t for t in true_lines if t]
        if not true_lines:
            continue
        try:
            facets = segment_building_best(ctx["dsm"], pc, geom, bid,
                                           imagery_ds=ctx["img"]) or []
        except Exception:
            continue
        pred = predicted_lines_from_facets(facets, geom)
        if not pred:
            continue

        base = line_scores(pred, true_lines, a.tol)
        if not base:
            continue
        best = (0.0, 0.0, base["f1"])
        for iy in range(-steps, steps + 1):
            for ix in range(-steps, steps + 1):
                dx, dy = ix * a.step, iy * a.step
                if dx == 0 and dy == 0:
                    continue
                s = line_scores(pred, shift_lines(true_lines, dx, dy), a.tol)
                if s and s["f1"] > best[2]:
                    best = (dx, dy, s["f1"])

        # Height, because lean displacement scales with it. This is the test
        # that separates photo lean from ordinary disagreement: a global mean
        # shift of zero rules out a uniform offset, but lean points away from
        # whatever was directly under the aircraft, so its DIRECTION varies
        # across a survey and cancels. Its MAGNITUDE does not -- it grows with
        # how far the roof stands above the ground.
        height = None
        try:
            c = geom.centroid
            gp = pc.ground_points_in_bbox(c.x - 25, c.y - 25, c.x + 25, c.y + 25)
            rp = pc.points_in_bbox(*geom.bounds)
            if gp is not None and len(gp) >= 20 and rp is not None and len(rp) >= 20:
                height = float(np.percentile(rp[:, 2], 75) -
                               np.percentile(gp[:, 2], 50))
        except Exception:
            pass
        rows.append({"id": bid, "f1": base["f1"], "bf1": best[2],
                     "dx": best[0], "dy": best[1], "height": height})
        print(f"{bid:>10}{base['f1']:>9.0%}{best[2]:>11.0%}"
              f"{f'{best[0]:+.2f},{best[1]:+.2f}':>14}"
              f"{best[2] - base['f1']:>+8.0%}")

    if not rows:
        print("nothing measured")
        return 1

    out = ROOT / "data" / "label_registration.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\n  wrote {out}")

    dxs = np.array([r["dx"] for r in rows])
    dys = np.array([r["dy"] for r in rows])
    gains = np.array([r["bf1"] - r["f1"] for r in rows])
    mags = np.hypot(dxs, dys)

    print(f"\nover {len(rows)} buildings:")
    print(f"  median shift        {np.median(dxs):+.2f}, {np.median(dys):+.2f} m")
    print(f"  mean shift          {dxs.mean():+.2f}, {dys.mean():+.2f} m")
    print(f"  median magnitude    {np.median(mags):.2f} m")
    print(f"  wanting >=0.5 m     {int((mags >= 0.5).sum())} of {len(rows)}")
    print(f"  mean F1 as drawn    {np.mean([r['f1'] for r in rows]):.1%}")
    print(f"  mean F1 best-shifted{np.mean([r['bf1'] for r in rows]):>7.1%}"
          f"   (+{gains.mean():.1%})")

    # A real lean is a COHERENT displacement: individual shifts agreeing in
    # direction. Scattered shifts of similar size are just each building finding
    # its own best fit in noise.
    coherence = np.hypot(dxs.mean(), dys.mean()) / (np.median(mags) or 1)
    print(f"\n  directional coherence {coherence:.2f}   "
          f"(1.0 = all shifts agree, 0 = cancel out)")
    hs = [(r["height"], np.hypot(r["dx"], r["dy"])) for r in rows
          if r.get("height") is not None]
    if len(hs) >= 12:
        H = np.array([h for h, _ in hs]); S = np.array([s for _, s in hs])
        r_hs = float(np.corrcoef(H, S)[0, 1])
        med = np.median(H)
        print(f"\n  height vs shift magnitude: r = {r_hs:+.3f} over {len(hs)} buildings")
        print(f"    shorter half (<{med:.1f} m): median shift "
              f"{np.median(S[H < med]):.2f} m")
        print(f"    taller  half (>={med:.1f} m): median shift "
              f"{np.median(S[H >= med]):.2f} m")
        if r_hs > 0.35:
            print("    -> shift grows with height, which is what lean does.")
        else:
            print("    -> no height relationship, so this is not lean: it is")
            print("       drawing precision plus genuine segmenter error.")

    if coherence < 0.35:
        print("  -> shifts do NOT agree on a direction. This is not photo lean;")
        print("     each building is finding its own best fit in noise, so the")
        print("     labels are registered correctly and the disagreement with")
        print("     the segmenter is genuine.")
    else:
        print("  -> shifts agree on a direction, which is what photo lean looks")
        print("     like. Worth correcting at ingest and restating the baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

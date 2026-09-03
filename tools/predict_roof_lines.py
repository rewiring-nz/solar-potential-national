"""
Run the trained line model over buildings and write what roof_line_source reads.

The seam already exists. src/roof_line_source.py looks for
data/vision_lines/<building_id>.json and, finding one, offers those lines to the
partition alongside the imagery-derived ones. The LiDAR gate downstream is
unchanged: a predicted line is only cut on if the surface actually turns or the
two sides sit at different heights. The model proposes, the LiDAR disposes.

CANDIDATES ONLY, DELIBERATELY. roof_line_source has a second path where model
lines become STRONG lines, acted on without LiDAR confirmation. That is not used
here. Held-out F1 for this model is 0.43 on ridges and 0.13 on cliffs, which is
nowhere near good enough to overrule the point cloud -- but it is quite good
enough to SUGGEST a crease the Hough detector missed and let the LiDAR decide.

FROM MASK TO SEGMENTS. The model emits a per-pixel probability per kind. Turning
that into line segments without pulling in OpenCV or scikit-image: threshold,
label connected components, and reduce each to its principal axis. A component
that is not elongated is not a line -- it is a smudge of activation -- so
elongation is required rather than assumed.

Usage:
    python tools/predict_roof_lines.py --region pilot
    python tools/predict_roof_lines.py --all
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

OUT = ROOT / "data" / "vision_lines"
MODEL = ROOT / "data" / "models" / "roof_lines_v1.pt"

THRESHOLD = 0.45          # probability above which a pixel is "on a line"
MIN_COMPONENT_PX = 25     # smaller than this is noise, not a crease
MIN_ELONGATION = 2.5      # a line is long and thin; a blob is not a line
MIN_LEN_M = 1.0
PAD_M = 4.0


def segments_from_mask(prob, transform_xy, thr=THRESHOLD):
    """Connected components of the thresholded mask, reduced to line segments.

    Each component becomes its principal axis, extended to the component's own
    extent. Anything insufficiently elongated is dropped: an unelongated blob of
    activation is the model being unsure over an area, not finding a line."""
    import numpy as np
    from scipy import ndimage
    mask = prob > thr
    if not mask.any():
        return []
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < MIN_COMPONENT_PX:
            continue
        pts = np.c_[xs.astype(float), ys.astype(float)]
        mean = pts.mean(axis=0)
        cen = pts - mean
        try:
            _, s, vt = np.linalg.svd(cen, full_matrices=False)
        except Exception:
            continue
        if s[1] < 1e-6:
            elong = 999.0
        else:
            elong = float(s[0] / s[1])
        if elong < MIN_ELONGATION:
            continue
        d = vt[0]
        t = cen @ d
        a_px = mean + d * t.min()
        b_px = mean + d * t.max()
        score = float(prob[ys, xs].mean())
        ax, ay = transform_xy(a_px[0], a_px[1])
        bx, by = transform_xy(b_px[0], b_px[1])
        if np.hypot(bx - ax, by - ay) < MIN_LEN_M:
            continue
        out.append(([ax, ay, bx, by], score))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    a = ap.parse_args()

    import numpy as np
    import torch
    import geopandas as gpd
    import rasterio
    import rasterio.windows
    import config
    from src.region_build import area_paths, all_areas
    import train_line_model as T

    if not MODEL.exists():
        print(f"no model at {MODEL} -- train one first")
        return 1
    ck = torch.load(MODEL, map_location="cpu", weights_only=False)
    model = T.build_unet(ck.get("pretrained", False))
    model.load_state_dict(ck["state_dict"])
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    regions = ([a.region] if a.region else
               (["pilot"] + [x for x in all_areas() if x != "pilot"]
                if a.all else None))
    if not regions:
        print("give --region NAME or --all")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    total_b = total_l = 0
    for region in regions:
        p = area_paths(region)
        if not (p["outlines"].exists() and p["imagery"].exists()):
            print(f"{region}: no imagery on this machine, skipping")
            continue
        dd = p["dir"] / "building_outlines_dedup.geojson"
        gdf = gpd.read_file(dd if dd.exists() else p["outlines"])
        img = rasterio.open(p["imagery"])
        n_b = n_l = 0
        for _, row in gdf.iterrows():
            bid = int(row["building_id"])
            if bid in getattr(config, "DEMOLISHED_BUILDING_IDS", ()):
                continue
            geom = row.geometry
            minx, miny, maxx, maxy = geom.bounds
            b = (minx - PAD_M, miny - PAD_M, maxx + PAD_M, maxy + PAD_M)
            win = rasterio.windows.from_bounds(*b, img.transform)
            try:
                rgb = np.moveaxis(img.read([1, 2, 3], window=win,
                                           boundless=True, fill_value=0), 0, -1)
            except Exception:
                continue
            h, w = rgb.shape[:2]
            if h < 32 or w < 32:
                continue
            # pad to a multiple of 16 so the U-Net's four downsamples line up
            ph, pw = (-h) % 16, (-w) % 16
            arr = np.pad(rgb, ((0, ph), (0, pw), (0, 0)))
            x = torch.from_numpy(arr).float().permute(2, 0, 1)[None] / 255.0
            with torch.no_grad():
                pr = torch.sigmoid(model(x.to(device)))[0].cpu().numpy()
            pr = pr[:, :h, :w]

            def to_world(px, py):
                return (b[0] + px / w * (b[2] - b[0]),
                        b[1] + (1 - py / h) * (b[3] - b[1]))

            lines, scores = [], []
            for k in range(pr.shape[0]):
                for seg, sc in segments_from_mask(pr[k], to_world, a.threshold):
                    lines.append([round(v, 3) for v in seg])
                    scores.append(round(sc, 3))
            if not lines:
                continue
            (OUT / f"{bid}.json").write_text(json.dumps({
                "lines": lines, "scores": scores,
                "model": "roof_lines_v1"}))
            n_b += 1
            n_l += len(lines)
            if a.limit and n_b >= a.limit:
                break
        print(f"{region}: {n_b} buildings with predictions, {n_l} lines")
        total_b += n_b
        total_l += n_l
    print(f"\n{total_b} buildings, {total_l} predicted lines -> {OUT}")
    print("roof_line_source picks these up automatically as CANDIDATES; the")
    print("LiDAR gate still decides whether any of them becomes a cut.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

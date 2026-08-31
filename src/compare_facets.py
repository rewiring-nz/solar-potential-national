"""
Two roof MODELS side by side over the imagery, for Josh to judge.

compare_layouts shows panels; this shows the facet boundaries that decide where
panels can go. When the question is whether a change improved the roof SHAPE --
as with imagery-guided cut lines -- panels are a lagging indicator and the
outlines are the thing to look at.

Usage:
  python src/compare_facets.py --area pilot --ids 5371128 4735247 ...
"""

import argparse
import base64
import io
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely
from matplotlib.patches import Polygon as MplPolygon

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.region_build import area_paths
import src.roof_partition as RP
from src.roof_lines import roof_line_candidates
from src.pointcloud_source import PointCloudSource

PAD_M = 3.0
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _with_lines(cands):
    """_best_cut that also considers imagery lines, scored identically."""
    orig = RP._best_cut

    def best(poly, pts, base):
        out = orig(poly, pts, base)
        for ang, off in cands:
            parts = RP._cut(poly, ang, off)
            if len(parts) < 2:
                continue
            tot = num = 0.0
            for part in parts:
                pl, sc = RP._score(part, pts)
                if pl is None:
                    tot = 0.0
                    break
                num += sc * part.area
                tot += part.area
            if tot <= 0:
                continue
            c = num / tot
            if c > base + RP.MIN_SPLIT_GAIN and (out is None or c > out[0]):
                out = (c, parts)
        return out
    return best


def draw(ax, imagery, facets, bounds, title, lines=None):
    minx, miny, maxx, maxy = bounds
    w = rasterio.windows.from_bounds(minx - PAD_M, miny - PAD_M, maxx + PAD_M,
                                     maxy + PAD_M, imagery.transform)
    rgb = np.moveaxis(imagery.read([1, 2, 3], window=w), 0, -1)
    ax.imshow(rgb, extent=[minx - PAD_M, maxx + PAD_M, miny - PAD_M, maxy + PAD_M])
    for f in facets:
        xs, ys = f["geometry"].exterior.xy
        ax.add_patch(MplPolygon(np.c_[xs, ys], closed=True, fill=False,
                                edgecolor="#ffd400", linewidth=1.5))
    if lines:
        for ang, off in lines:
            th = np.radians(ang)
            d = np.array([np.cos(th), np.sin(th)])
            n = np.array([-np.sin(th), np.cos(th)])
            c = np.array([(minx + maxx) / 2, (miny + maxy) / 2]) + n * off
            span = max(maxx - minx, maxy - miny)
            ax.plot(*zip(c - d * span, c + d * span), color="#39c0ff",
                    linewidth=0.7, alpha=0.55)
    ax.set_xlim(minx - PAD_M, maxx + PAD_M)
    ax.set_ylim(miny - PAD_M, maxy + PAD_M)
    ax.set_title(title, color="#eee", fontsize=9)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="pilot")
    ap.add_argument("--ids", nargs="+", type=int, required=True)
    a = ap.parse_args()

    import geopandas as gpd
    p = area_paths(a.area)
    dedup = p["dir"] / "building_outlines_dedup.geojson"
    gdf = gpd.read_file(dedup if dedup.exists() else p["outlines"]).set_index(
        "building_id", drop=False)
    pc = PointCloudSource()
    img = rasterio.open(p["imagery"])

    cards = []
    orig_best = RP._best_cut
    for i, bid in enumerate(a.ids, 1):
        if bid not in gdf.index:
            continue
        g = gdf.loc[bid].geometry
        mn, mi, mx, ma = g.bounds
        pts = pc.points_in_bbox(mn - 2, mi - 2, mx + 2, ma + 2, building_only=True)
        pts = pts[shapely.contains_xy(g, pts[:, 0], pts[:, 1])]
        if len(pts) < 50:
            continue
        RP._best_cut = orig_best
        a_f = RP.partition_roof(bid, g, pts)
        cands = roof_line_candidates(img, g)
        RP._best_cut = _with_lines(cands)
        b_f = RP.partition_roof(bid, g, pts)
        RP._best_cut = orig_best

        fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.4), facecolor="#15141a")
        draw(ax[0], img, a_f, g.bounds, f"SWEPT CUTS  {len(a_f)} facets")
        draw(ax[1], img, b_f, g.bounds,
             f"+ IMAGERY LINES  {len(b_f)} facets   (blue = lines detected)", cands)
        fig.tight_layout(pad=0.4)
        buf = io.BytesIO()
        fig.savefig(buf, format="jpg", dpi=105, bbox_inches="tight",
                    facecolor="#15141a", pil_kwargs={"quality": 82, "optimize": True})
        plt.close(fig)
        cards.append((i, bid, len(a_f), len(b_f), base64.b64encode(buf.getvalue()).decode()))
        print(f"  {i:2d}. {bid}  {len(a_f)} -> {len(b_f)} facets, {len(cands)} lines", flush=True)

    html = ["<title>Roof model: swept cuts vs imagery lines</title>",
            "<style>body{background:#15141a;color:#eee;font:14px system-ui;margin:0;padding:24px}"
            "h1{font-size:18px;font-weight:600}img{width:100%;border-radius:6px;margin:6px 0 22px}"
            "</style>",
            "<h1>Roof model &mdash; swept cuts (left) vs imagery-guided lines (right)</h1>",
            "<p style='opacity:.7'>Yellow = facet boundaries. Blue = straight lines found in the "
            "0.1&thinsp;m imagery and offered to the partition as candidate cuts.</p>"]
    for i, bid, na, nb, b64 in cards:
        html.append(f"<div><b>{i}. #{bid}</b> &nbsp; {na} &rarr; {nb} facets</div>"
                    f"<img src='data:image/jpeg;base64,{b64}'>")
    out = DATA_DIR / "compare_facets.html"
    out.write_text("\n".join(html))
    print(f"\nSaved {out} ({out.stat().st_size/1e6:.1f}MB, {len(cards)} pairs)")


if __name__ == "__main__":
    main()

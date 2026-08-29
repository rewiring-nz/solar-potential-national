"""
Clean, large roof renders for Josh to draw the true roof lines on.

Every automated signal tried so far -- plane fit, fold fraction, fold location,
acceptance threshold, plane counts -- fails to separate the roofs Josh calls
correct from the ones he calls wrong. On 7 Anderson Heights and 5 Isle St the
numbers are nearly identical and his verdicts are opposite, so there is nothing
left to tune against.

What is missing is ground truth for roof STRUCTURE, and the cheapest way to get
it is for him to draw it. He marks ridges, hips, valleys and level changes on
these images; those lines get encoded once and every future attempt can be
scored against them without asking him again.

Deliberately plain: imagery only, a metre grid for reference, no model output
overlaid. Showing the current faces would anchor the answer to what the pipeline
already believes, which is the thing under question.

Usage: python src/mark_roofs.py --ids 5371108 4735015 ...
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.region_build import area_paths

PAD_M = 2.0
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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
    img = rasterio.open(p["imagery"])

    for bid in a.ids:
        if bid not in gdf.index:
            print(f"  #{bid} not in {a.area}")
            continue
        g = gdf.loc[bid].geometry
        minx, miny, maxx, maxy = g.bounds
        w = rasterio.windows.from_bounds(minx - PAD_M, miny - PAD_M,
                                         maxx + PAD_M, maxy + PAD_M, img.transform)
        rgb = np.moveaxis(img.read([1, 2, 3], window=w), 0, -1)
        span = max(maxx - minx, maxy - miny)
        fig, ax = plt.subplots(figsize=(13, 13 * (maxy - miny + 2 * PAD_M) /
                                        (maxx - minx + 2 * PAD_M)), facecolor="white")
        ax.imshow(rgb, extent=[minx - PAD_M, maxx + PAD_M, miny - PAD_M, maxy + PAD_M])
        xs, ys = g.exterior.xy
        ax.plot(xs, ys, color="#ffd400", lw=1.6, alpha=0.95)
        # metre grid, so a line can be described in words if drawing is awkward
        step = 5 if span < 60 else 10
        for x in np.arange(np.ceil((minx) / step) * step, maxx, step):
            ax.axvline(x, color="w", lw=0.4, alpha=0.35)
            ax.text(x, miny - PAD_M + 0.4, f"{int(x % 1000)}", color="w",
                    fontsize=7, ha="center")
        for y in np.arange(np.ceil((miny) / step) * step, maxy, step):
            ax.axhline(y, color="w", lw=0.4, alpha=0.35)
            ax.text(minx - PAD_M + 0.4, y, f"{int(y % 1000)}", color="w", fontsize=7, va="center")
        ax.set_xlim(minx - PAD_M, maxx + PAD_M)
        ax.set_ylim(miny - PAD_M, maxy + PAD_M)
        ax.axis("off")
        ax.set_title(f"#{bid}  ({g.area:.0f} m2)  — mark every roof line: "
                     f"ridges, hips, valleys, level changes",
                     fontsize=11, color="#222")
        out = DATA_DIR / f"mark_{bid}.jpg"
        fig.tight_layout(pad=0.3)
        fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white",
                    pil_kwargs={"quality": 92})
        plt.close(fig)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()

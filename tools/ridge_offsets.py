"""How far each shared ridge sits from the crest the survey shows.

    python tools/ridge_offsets.py arrowtown_millbrook            # region's DSM
    python tools/ridge_offsets.py arrowtown_millbrook --pointcloud

Reads data/regions/<region>/panel_layouts.geojson (the emitted facets),
pairs every two pitched, opposite-facing facets that share a run of
boundary, and measures the crest with the same tent fit src/ridge_snap.py
uses. Prints the distribution and the share of ridges the fit could not
measure. Run before and after a build to see what the snap changed; a
rebuild that leaves 15% of ridges over a metre from the crest has not run it.

Baseline, 23 Sep, arrowtown_millbrook on the 1 m DSM, before the snap:
2,171 ridges, median 0.00 m, 36% beyond 0.5 m, 15% beyond 1 m.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region")
    ap.add_argument("--pointcloud", action="store_true", help="measure against the point cloud, not the DSM")
    ap.add_argument("--layout", default=None, help="a panel_layouts.geojson to read instead of the region's")
    a = ap.parse_args()
    import pyproj
    import rasterio
    from shapely.geometry import shape
    from shapely.ops import transform as st
    from src import ridge_snap
    from src.region_build import area_paths
    paths = area_paths(a.region)
    layout = Path(a.layout) if a.layout else paths["dir"] / "panel_layouts.geojson"
    to = pyproj.Transformer.from_crs(4326, 2193, always_xy=True).transform
    by = defaultdict(list)
    for f in json.load(open(layout))["features"]:
        p = f["properties"]
        if p.get("kind") == "facet":
            by[p["building_id"]].append(dict(p, geometry=st(to, shape(f["geometry"]))))
    pc, dsm = None, None
    if a.pointcloud:
        from src.pointcloud_source import PointCloudSource
        pc = PointCloudSource()
    else:
        ds = rasterio.open(paths["dsm"])
        dsm = (ds.read(1), ds.transform, ds.nodata)
    offs, pairs = [], 0
    for bid, fl in by.items():
        for i in range(len(fl)):
            for j in range(i + 1, len(fl)):
                fi, fj = fl[i], fl[j]
                if fi["slope_deg"] < ridge_snap.RIDGE_MIN_SLOPE_DEG or fj["slope_deg"] < ridge_snap.RIDGE_MIN_SLOPE_DEG:
                    continue
                da = abs(((fi["aspect_deg"] - fj["aspect_deg"]) + 180) % 360 - 180)
                if da < ridge_snap.RIDGE_OPPOSITE_DEG:
                    continue
                if ridge_snap._ridge_line(fi["geometry"], fj["geometry"]) is None:
                    continue
                pairs += 1
                r = ridge_snap.crest_offset(pc, dsm, fi["geometry"], fj["geometry"])
                if r is not None:
                    offs.append(abs(r[0]))
    offs = np.asarray(offs)
    print(f"{a.region}: {pairs} shared ridges >= {ridge_snap.RIDGE_MIN_LEN_M} m; crest measurable on {len(offs)} ({100 * len(offs) / max(pairs, 1):.0f}%)")
    if len(offs):
        print(f"  |offset|: median {np.median(offs):.2f} m, p75 {np.percentile(offs, 75):.2f}, p90 {np.percentile(offs, 90):.2f}; "
              f"beyond 0.5 m: {100 * (offs > 0.5).mean():.0f}%, beyond 1 m: {100 * (offs > 1).mean():.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

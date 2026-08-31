"""
Re-fit ONE building through the real pipeline, for any area, and report what
changed. The iteration loop for placement work.

src/live_server.py can already do this, but only for the pilot and only over
HTTP. Josh's reports land all over the district, and a full rebuild is hours,
so tuning placement without this means changing a constant and waiting.

Usage:
    python src/refit_one.py 4733121 [4735403 ...]      # compare to the shipped layout
    python src/refit_one.py --area pilot 5371121
"""

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
from shapely.ops import unary_union

warnings.filterwarnings("ignore")
# ...but never deprecations. A blanket ignore is exactly how 68 calls to
# shapely.vectorized -- an API documented for REMOVAL, under an unpinned
# shapely>=2.0 -- stayed invisible until 31 Aug. Third-party noise stays
# suppressed; a countdown to the pipeline breaking does not.
warnings.filterwarnings("default", category=DeprecationWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.region_build import all_areas, area_paths
from src.roof_segmentation import segment_building_best
from src.pointcloud_source import PointCloudSource
from src.obstruction_detection import detect_obstructions_combined
from src.panel_fitting import fit_panels_on_facet
from src.solar_model import SolarModel

TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True).transform
_CACHE = {}


def _area_of(building_id):
    """Which area owns this building? Scan the shipped layouts by raw text --
    far cheaper than parsing 24 files to find one id."""
    for area in all_areas():
        p = area_paths(area)["panel_layouts"]
        if not p.exists():
            continue
        t = p.read_text()
        if f'"building_id": {building_id}' in t or f'"building_id":{building_id}' in t:
            return area, t
    return None, None


def _load(area):
    if area in _CACHE:
        return _CACHE[area]
    paths = area_paths(area)
    dedup = paths["dir"] / "building_outlines_dedup.geojson"
    gdf = gpd.read_file(dedup if dedup.exists() else paths["outlines"]).set_index("building_id", drop=False)
    dsm = rasterio.open(paths["dsm"])
    img = None
    if paths["imagery"].exists():
        img = rasterio.open(paths["imagery"])   # rural areas have none (LINZ urban-only)
    ctx = {"gdf": gdf, "dsm": dsm, "dsm_band": dsm.read(1), "img": img,
           "pc": PointCloudSource(), "model": SolarModel()}
    _CACHE[area] = ctx
    return ctx


def refit(building_id, shipped_text=None, area=None):
    if area is None:
        area, shipped_text = _area_of(building_id)
        if area is None:
            print(f"#{building_id}: not found in any area's layouts")
            return
    ctx = _load(area)
    row = ctx["gdf"].loc[building_id]
    # imagery matters to segmentation now (roof_partition cuts on strong image
    # lines), so pass it or this tool silently measures a different pipeline
    # from the one that builds the map
    facets = segment_building_best(ctx["dsm"], ctx["pc"], row.geometry, building_id,
                                   imagery_ds=ctx["img"])

    total, per_facet = 0, []
    for f in facets:
        plane = (f["plane_a"], f["plane_b"], f["plane_c"])
        obst = detect_obstructions_combined(ctx["img"], ctx["pc"], f["geometry"], plane)
        sibs = [o for o in facets if o is not f]
        panels = fit_panels_on_facet(f, obstructions=obst, sibling_facets=sibs)
        total += len(panels)
        per_facet.append((f["geometry"].area, f["slope_deg"], f["aspect_deg"],
                          unary_union(obst).area if obst else 0.0, len(panels)))

    roof = unary_union([f["geometry"] for f in facets]).area if facets else 0.0
    plan = sum(2.0 * n * np.cos(np.radians(s)) for _, s, _, _, n in per_facet)
    print(f"\n#{building_id}  ({area})")
    print(f"  facets {len(facets)}  roof {roof:6.1f} m2   panels {total:4d}"
          f"   fill {100 * plan / max(roof, 1e-9):3.0f}% of roof")
    for a, s, asp, ob, n in per_facet:
        print(f"    facet {a:6.1f} m2  slope {s:4.1f}  aspect {asp:5.1f}  obstr {ob:5.1f} m2  panels {n:3d}")

    if shipped_text:
        import json
        d = json.loads(shipped_text)
        was = sum(1 for f in d["features"]
                  if f["properties"].get("building_id") == building_id
                  and f["properties"]["kind"] == "panel")
        print(f"  shipped layout had {was} panels  ->  {total} now  ({total - was:+d})")
    return total


def main():
    argv = sys.argv[1:]
    area = None
    if "--area" in argv:
        i = argv.index("--area")
        area = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    ids = [int(a) for a in argv if not a.startswith("--")]
    if not ids:
        raise SystemExit(__doc__)
    for bid in ids:
        if area:
            refit(bid, area_paths(area)["panel_layouts"].read_text(), area)
        else:
            refit(bid)


if __name__ == "__main__":
    main()

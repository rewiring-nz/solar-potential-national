"""
Bake per-building horizons onto a region's solar_potential features (and the
merged district file when present): `horizon_b64` (72-bin profile, see
building_horizon.py) and `horizon_beam_pct` (share of annual direct beam the
sky leaves this building, 0-100).

Patch-in-place, same shape as add_addresses.py / patch_roof_confidence.py:
idempotent, re-running overwrites the same two fields.

Usage: python src/bake_building_horizons.py <region> [...]
"""

import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preflight import preflight
from src.building_horizon import (compute_building_horizon, encode_horizon,
                                  beam_visible_fraction)
from src.region_build import area_paths, area_centroid_wgs84, write_json_atomic
from src.solar_model import SolarModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEM_WIDE = DATA_DIR / "dem_wide_mosaic.tif"


def bake(region):
    paths = area_paths(region)
    sp_path = paths["solar_potential"]
    if not sp_path.exists():
        print(f"[{region}] no solar_potential.geojson -- run derive first")
        return
    gdf = gpd.read_file(paths["outlines"]).set_index("building_id", drop=False)

    dem_ds = rasterio.open(DEM_WIDE)
    dem_band = dem_ds.read(1)
    dsm_ds = rasterio.open(paths["dsm"])
    dsm_band = dsm_ds.read(1)

    c = area_centroid_wgs84(region)
    model = SolarModel(*c) if c else SolarModel()

    sp = json.loads(sp_path.read_text())
    t0 = time.time()
    done = skipped = 0
    for f in sp["features"]:
        bid = f["properties"].get("building_id")
        if bid not in gdf.index:
            skipped += 1
            continue
        geom = gdf.loc[bid].geometry
        profile, far = compute_building_horizon(dem_band, dem_ds.transform, dem_ds.nodata,
                                                dsm_band, dsm_ds.transform, dsm_ds.nodata,
                                                geom)
        if profile is None:
            skipped += 1
            continue
        f["properties"]["horizon_b64"] = encode_horizon(profile)
        if far is not None:
            f["properties"]["horizon_far_b64"] = encode_horizon(far)
        f["properties"]["horizon_beam_pct"] = round(
            beam_visible_fraction(profile, model.hourly) * 100.0, 1)
        done += 1
        if done % 500 == 0:
            print(f"  {done} baked ({time.time() - t0:.0f}s)", flush=True)

    write_json_atomic(sp_path, sp)
    print(f"[{region}] horizons baked: {done} buildings, {skipped} skipped, "
          f"{time.time() - t0:.0f}s")

    # keep the deployed merged file in step when it exists (single-region
    # deploys serve data/solar_potential.geojson directly)
    merged = DATA_DIR / "solar_potential.geojson"
    if merged.exists():
        by_id = {f["properties"].get("building_id"): f["properties"] for f in sp["features"]}
        md = json.loads(merged.read_text())
        n = 0
        for f in md["features"]:
            src = by_id.get(f["properties"].get("building_id"))
            if src and "horizon_b64" in src:
                f["properties"]["horizon_b64"] = src["horizon_b64"]
                f["properties"]["horizon_beam_pct"] = src["horizon_beam_pct"]
                if "horizon_far_b64" in src:
                    f["properties"]["horizon_far_b64"] = src["horizon_far_b64"]
                n += 1
        write_json_atomic(merged, md)
        print(f"  merged file: {n} buildings patched")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        preflight("bake_building_horizons", r)
        bake(r)

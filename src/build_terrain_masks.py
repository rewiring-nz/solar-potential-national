"""
Per-building terrain sun-visibility masks for the generation curves.

Problem (user-verified, 22 Aug): the seasonal curves shared ONE terrain
horizon (town centre), so an Arrowtown valley roof showed the same winter
afternoon rolloff as downtown -- when in reality its sun drops behind the
ranges at ~13:00 vs 15:15 (measured from the DEM). Yields already use
region-centroid horizons; curves used the pilot's everywhere.

This bakes location-specific terrain into the buildings file:
- Building centroids are snapped to a CELL_M grid; one horizon profile is
  computed per unique cell from data/dem_wide_mosaic.tif (mountains don't
  change across 150m; per-building profiles would cost hours for no
  visible difference).
- For each cell, for each season (summer/autumn/winter/spring) and each
  hour 0-23: the fraction of that season's days on which the sun sits
  above the terrain horizon during that hour -> quantized 0-9.
- Written per building into solar_potential.geojson as "tshade": a
  4x24-digit string (season-major). '9' = always visible, '0' = always
  terrain-blocked. The frontend multiplies curve hours by max(digit/9,
  diffuse floor).

ORDERING: this stage writes ONLY the MERGED data/solar_potential.geojson, and
merge_regions REGENERATES that file from the per-region files. So it must run
AFTER the merge, always. Run before it and the masks are silently wiped -- no
error, just unshaded seasonal curves district-wide (cost us the Island Bay
rebuild on 31 Aug; addresses and horizons survived only because those stages
write the region file too). Same rule applies to bake_density_deciles.

Usage: python src/build_terrain_masks.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import pyproj
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.region_build import write_json_atomic
from src.terrain_horizon import compute_horizon_profile_from_array, horizon_angle_at

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEM = str(DATA_DIR / "dem_wide_mosaic.tif")
CELL_M = 150
SEASONS = {"summer": (12, 1, 2), "autumn": (3, 4, 5), "winter": (6, 7, 8), "spring": (9, 10, 11)}
TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)


def main():
    sp_path = DATA_DIR / "solar_potential.geojson"
    sp = json.loads(sp_path.read_text())

    # Fail loudly rather than quietly producing open-horizon masks for every
    # building: without the DEM this script still "succeeds", writes a full
    # tshade for each roof, and silently turns terrain shading into a no-op.
    if not Path(DEM).exists():
        raise SystemExit(f"missing {DEM} -- fetch the wide DEM mosaic first")

    # solar position series, one year, hourly (same convention as the model)
    loc = pvlib.location.Location(-45.03, 168.66, tz="Pacific/Auckland", altitude=310)
    times = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="1h", tz="Pacific/Auckland")
    solpos = loc.get_solarposition(times)
    sun_az = solpos["azimuth"].to_numpy()
    sun_el = solpos["apparent_elevation"].to_numpy()
    month = times.month.to_numpy()
    hour = times.hour.to_numpy()

    # building centroid -> grid cell
    cells = {}
    feats = []
    for f in sp["features"]:
        g = f["geometry"]
        ring = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0]
        lng = sum(c[0] for c in ring) / len(ring)
        lat = sum(c[1] for c in ring) / len(ring)
        x, y = TO_NZTM.transform(lng, lat)
        cell = (round(x / CELL_M), round(y / CELL_M))
        cells.setdefault(cell, []).append(f)
        feats.append(f)
    print(f"{len(feats)} buildings -> {len(cells)} horizon cells at {CELL_M}m")

    # Read the 65MB mosaic ONCE. compute_horizon_profile() reopens and
    # re-reads the whole raster per call, which across ~1.5k cells is the
    # dominant cost of this script for no benefit -- terrain_horizon exposes
    # the array-taking variant for exactly this reason.
    with rasterio.open(DEM) as ds:
        dem_band, dem_transform, dem_nodata = ds.read(1), ds.transform, ds.nodata

    done = 0
    fallbacks = 0
    for cell, members in cells.items():
        cx, cy = cell[0] * CELL_M, cell[1] * CELL_M
        try:
            prof = compute_horizon_profile_from_array(dem_band, dem_transform, dem_nodata, cx, cy)
            horiz = horizon_angle_at(prof, sun_az)
            visible = (sun_el > horiz) & (sun_el > 0)
        except Exception:
            visible = sun_el > 0  # outside DEM -- open horizon fallback
            fallbacks += 1
        mask = ""
        for months in SEASONS.values():
            in_season = np.isin(month, months)
            for h in range(24):
                sel = in_season & (hour == h)
                daylight = sel & (sun_el > 0)
                if daylight.sum() == 0:
                    mask += "9"  # night hours -- curve is zero anyway, don't dim
                else:
                    frac = visible[daylight].sum() / daylight.sum()
                    mask += str(min(9, int(round(frac * 9))))
        for f in members:
            f["properties"]["tshade"] = mask
        done += 1
        if done % 250 == 0:
            print(f"  {done}/{len(cells)} cells")

    if fallbacks:
        # A handful of edge cells is normal. Most/all of them means the DEM
        # does not actually cover the build area and the masks are fiction.
        print(f"  WARNING: {fallbacks}/{len(cells)} cells fell back to an open "
              f"horizon (outside the DEM extent)")
    write_json_atomic(sp_path, sp)
    print(f"Saved {sp_path} ({sp_path.stat().st_size / 1e6:.1f}MB) with tshade masks")


if __name__ == "__main__":
    main()

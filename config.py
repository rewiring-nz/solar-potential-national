"""Pilot run configuration."""

# WGS84 [min_lon, min_lat, max_lon, max_lat]. Central Queenstown town basin --
# confirmed against real LINZ data: 1270 buildings, full building-outline
# coverage (capture source "Queenstown 0.1m Urban Aerial Photos (2021)"),
# DSM available (layer 105855, "Otago - Queenstown LiDAR 1m DSM (2021)").
PILOT_BBOX = [168.655, -45.045, 168.675, -45.025]
PILOT_BBOX_NZTM2000 = [1257815.95, 5002860.10, 1259272.13, 5005166.35]  # EPSG:2193, same area

# --- Queenstown-wide expansion regions --------------------------------------
# WGS84 [min_lon, min_lat, max_lon, max_lat] per urban sub-region. Boxes are
# data-driven, not hand-drawn: candidate rectangles were validated against
# real LINZ building-outline counts (WFS), then tightened to the 2nd-98th
# percentile of actual building centroids +150m pad (with k-means splits for
# regions whose buildings cluster into separate settlements) -- fetching
# imagery for one giant rectangle spanning town-to-Arrowtown would be ~60GB
# of mostly lake and mountainside. ~9,650 buildings across these boxes
# (verified counts per box, Aug 2026) + 1,270 in the existing pilot bbox.
# The pilot itself stays on its original top-level paths; each region here
# gets its own data/regions/<name>/ tree.
REGIONS = {
    # Island Bay, Wellington. The first region outside Queenstown, chosen as
    # the portability test: different LiDAR survey (Wellington City 2019-20),
    # different imagery (0.075 m, 2021 -- sharper than Queenstown's 0.1 m),
    # different housing stock. Deployed as its OWN build and page, per Josh:
    # "I want the Queenstown build and the national build to be separate in
    # case of any problems."
    "island_bay": [174.7660, -41.3480, 174.7840, -41.3280],
}


# Buildings confirmed demolished/replaced since the 2021 capture (field
# reports) -- excluded from every build until LINZ data catches up.
# Real buildings whose top surface is not a usable roof (rooftop car decks,
# etc.) -- kept on the map, but no panels placed.
NON_ROOF_BUILDING_IDS = {
    4744271,  # 19 Industrial Pl -- rooftop parking deck (Josh, 23 Aug)
}

DEMOLISHED_BUILDING_IDS = {
    4735131,  # 61 Ballarat St -- now under the new road corridor (Josh, 23 Aug)
}
LINZ_LIDAR_TILE_INDEX_LAYER = 105025  # "Otago - Queenstown LiDAR Tile Index (2021)" -- maps a
# bbox to the CL2_*.copc.laz point-cloud tile names, which are then fetched from OpenTopography's
# public bulk store (LINZ hosts the derived DSM/DEM rasters but not the raw point cloud)

# LINZ layer IDs (confirmed to exist and cover the pilot bbox)
LINZ_BUILDING_OUTLINES_LAYER = 101290
LINZ_DSM_LAYER = 105024  # "Otago - Queenstown LiDAR 1m DSM (2021)"
LINZ_DEM_LAYER = 105023  # "Otago - Queenstown LiDAR 1m DEM (2021)" -- bare earth, for shading horizon
LINZ_IMAGERY_LAYER = 105744  # "Queenstown 0.1m Urban Aerial Photos (2026)" -- captured 12 Feb-3 Mar
# 2026, replacing the 2021 capture this pilot originally used. More current (new/changed rooftop
# equipment, growth) at the cost of no longer matching the DSM/building-outline capture year exactly
# (both still 2021) -- worth revisiting if that skew ever shows up as a real building-outline/roof
# misalignment, but the two are already independently-sourced datasets with their own tolerances.

PANEL_WIDTH_M = 1.0
PANEL_HEIGHT_M = 2.0
PANEL_EDGE_SETBACK_M = 0.3  # clearance from the roof's own outer edge (eave/verge) -- common
# fire-code convention. Lowered to 0.1 earlier per explicit request after it was found strangling
# narrow facets (a real ~1.4m-wide strip loses 0.6m total, under the panel's own 1m minimum
# dimension, so it fit zero panels despite real usable area, on #5371143) -- but that traded away
# realistic edge clearance on every *normal-width* facet just to rescue the rare narrow one, and
# was reported back as making edges "clearly wrong" on ordinary roofs. Restored to 0.3;
# PANEL_EDGE_SETBACK_FALLBACK_M below handles the narrow-facet case instead, per-facet.
PANEL_EDGE_SETBACK_FALLBACK_M = 0.1  # retried only for a facet that fits zero panels at the
# primary setback above -- keeps narrow facets panelable without loosening the default for
# everything else.
RIDGE_SETBACK_M = 0.25  # extra clearance specifically along a boundary shared with another real
# roof plane on the same building (a real ridge, hip, or valley) -- separate from, and on top of,
# PANEL_EDGE_SETBACK_M's outer-edge clearance. Two adjacent facets each erode this far back from
# their shared boundary, so the real join between two differently-angled roof sections reads as
# an actual visible gap (like real ridge cap flashing) instead of two panel grids butting flush
# against each other with no visual break between them.
# Was 45, which was cutting off real roofs rather than unusable ones. Josh, on
# 1/5 Sydney St -- a twelve-unit terrace whose facets are ALL 44-50 degrees, so
# the cap silently excluded the entire building and left it with 6 panels:
# "I also don't know why there is a panel cut off at 45 degree, even 90 degree
# panels can be economic if facing the right direction."
#
# He is right that steepness alone does not make a panel uneconomic -- the solar
# model already prices slope and aspect, and the per-panel ROI bands already show
# a badly-oriented panel as red. A hard slope cut is doing that job twice, and
# worse.
#
# It is not raised to 90 because past about 72 degrees a surface is a WALL, not a
# roof, and the facets here come from points inside the building footprint, so
# wall returns would start collecting panels. 72 is the same threshold
# roof_reconstruct already uses to tell roof from wall. That admits every real
# roof including steep mansards, and leaves the economics to decide the rest.
MAX_ROOF_SLOPE_DEG = 72

# --- PV system assumptions ---
# These are shown to the end user in the UI, not just baked into the model
# silently -- the brief calls for transparency here. Keep this block as the
# single source of truth so the frontend can render exactly these numbers
# next to every estimate.
PV_ASSUMPTIONS = {
    "panel_rated_power_w": 440,  # W per panel at STC, typical current residential panel
    "panel_area_m2": PANEL_WIDTH_M * PANEL_HEIGHT_M,
    "panel_efficiency_pct": 22.0,  # STC efficiency implied by 440W / 2m2 / 1000W/m2
    "inverter_efficiency_pct": 97.0,  # typical string/micro-inverter conversion efficiency
    # 19%, following the PVWatts convention with two deliberate departures.
    # PVWatts' 14% default EXCLUDES cell temperature (it models that from
    # weather) and EXCLUDES the inverter (a separate parameter, as here). We
    # cannot model cell temperature -- it needs hourly ambient temperature and
    # wind we do not fetch -- so it is folded in; and we drop PVWatts' 3%
    # shading allowance because shading is modelled explicitly, per panel, from
    # the LiDAR surface.
    #   cell temperature ~8   (temperate NZ; PVGIS implies 9.6% incl. spectral
    #                          and reflection losses)
    #   soiling           2
    #   mismatch          2
    #   wiring/connections 2.5
    #   light-induced degradation 1.5
    #   nameplate tolerance 1
    #   availability      2
    #   shading           0   -- modelled per panel instead
    # 14% was the old value and omitted temperature entirely, which the PVGIS
    # cross-check measured as ~5% optimistic on AC yield. 0.97 x 0.81 = 0.786
    # against PVGIS's effective 0.791.
    "system_derate_pct": 19.0,
    "notes": (
        "kWp = panel count x rated power at Standard Test Conditions "
        "(1000 W/m2, 25C cell temp) -- a nameplate figure, not a real-world "
        "output. Daily/annual kWh starts from pvlib clear-sky irradiance, "
        "scales it to NIWA's measured radiation record for this area, splits "
        "that into direct and diffuse light, and projects it onto each roof "
        "face's own slope and aspect using an anisotropic sky model. Each "
        "face and panel is then reduced for the terrain, trees and "
        "neighbouring buildings that actually shade it, before the inverter "
        "and system losses above -- 19% covering cell temperature, soiling, "
        "wiring, mismatch and downtime, which follows the industry "
        "(PVWatts) convention except that shading is not in that number: "
        "we model it per panel from the laser scan instead. Real output "
        "still varies with weather, "
        "panel brand and age, and anything that has changed since the LiDAR "
        "survey was flown -- tree growth in particular."
    ),
}

# Wellington City 2019-20 survey (the Island Bay build). Chosen over the 2025
# refresh because the raw point cloud is on OpenTopography today; upgrade path
# recorded in data/roof_truth notes.
POINTCLOUD_BULK_URL = "https://opentopography.s3.sdsc.edu/pc-bulk/NZ19_Wellington"
POINTCLOUD_TILE_YEAR = "2019"

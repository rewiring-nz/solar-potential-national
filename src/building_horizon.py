"""
Per-building horizon: ONE 72-bin profile per building, the single source of
truth for every number the site shows (Josh, 30 Aug: "make sure all
calculations, like generation profiles, economics and savings that show, and
heat maps, all take into account these horizons").

Two layers, combined by per-bin max:
  FAR  -- the wide bare-earth DEM (8 m), out to its full extent (~10-15 km
          here; distant ranges beyond that subtend well under a degree).
          Replaces the single per-AREA terrain profile: Wellington's hills
          give a valley-floor building a genuinely different sky from a
          hilltop one three streets away.
  NEAR -- the region's 1 m DSM out to 300 m: trees and neighbouring
          buildings. The building's own footprint (buffered 1 m) is excluded
          unconditionally -- see the comment at the near scan for why the
          per-facet overhang rule must NOT apply here.

Observer: footprint centroid at EAVE height (15th percentile of the DSM
inside the footprint -- the roof's low edge). Known caveats, kept honest:
tree heights are LiDAR-vintage; one centroid ray origin under-represents
horizon variation across a very large roof.

The profile is baked onto solar_potential building properties as
`horizon_b64`: 72 azimuth bins (0=N, 5 deg step), each uint8 = elevation
degrees x 2 (caps at 127.5 deg), base64-encoded -- ~100 bytes. The frontend
decodes it for the horizon tab and its seasonal sun arcs; the model side
uses the same profile for beam masking, so chart and economics can never
disagree about the sky.
"""

import base64
import sys
from pathlib import Path

import numpy as np
import rasterio
import shapely
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.terrain_horizon import compute_horizon_profile_from_array, horizon_angle_at

N_BINS = 72
AZ_STEP = 360.0 / N_BINS
FAR_MAX_KM = 20.0            # marcher stops at the DEM edge anyway
FAR_STEP_M = 24.0
NEAR_MAX_KM = 0.3
NEAR_STEP_M = 2.0
EAVE_PCTL = 15
OVERHANG_CLEARANCE_M = 2.5
OWN_FOOTPRINT_MARGIN_M = 1.0


def eave_height(dsm_band, dsm_transform, dsm_nodata, geom):
    """Eave z: a low percentile of the DSM inside the footprint."""
    minx, miny, maxx, maxy = geom.bounds
    r0, c0 = rasterio.transform.rowcol(dsm_transform, minx, maxy)
    r1, c1 = rasterio.transform.rowcol(dsm_transform, maxx, miny)
    r0, r1 = max(0, min(r0, r1)), min(dsm_band.shape[0], max(r0, r1) + 1)
    c0, c1 = max(0, min(c0, c1)), min(dsm_band.shape[1], max(c0, c1) + 1)
    if r1 <= r0 or c1 <= c0:
        return None
    window = dsm_band[r0:r1, c0:c1]
    xs = np.arange(c0, c1) * dsm_transform.a + dsm_transform.c + dsm_transform.a / 2
    ys = np.arange(r0, r1) * dsm_transform.e + dsm_transform.f + dsm_transform.e / 2
    gx, gy = np.meshgrid(xs, ys)
    inside = shapely.contains_xy(geom, gx, gy)
    vals = window[inside]
    if dsm_nodata is not None:
        vals = vals[vals != dsm_nodata]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 4:
        return None
    return float(np.percentile(vals, EAVE_PCTL))


def compute_building_horizon(dem_band, dem_transform, dem_nodata,
                             dsm_band, dsm_transform, dsm_nodata,
                             geom, eave_z=None):
    """(combined, far_terrain_only) 72-bin horizon profiles
    {azimuth_deg: elevation_deg} for one building -- the frontend draws the
    far terrain silhouette with the tree/building layer darker on top.
    (None, None) when the observer is outside both rasters."""
    c = geom.centroid
    if eave_z is None:
        eave_z = eave_height(dsm_band, dsm_transform, dsm_nodata, geom)

    far = near = None
    try:
        far = compute_horizon_profile_from_array(
            dem_band, dem_transform, dem_nodata, c.x, c.y,
            azimuth_step_deg=AZ_STEP, max_distance_km=FAR_MAX_KM,
            sample_step_m=FAR_STEP_M, observer_z=eave_z)
    except ValueError:
        pass
    try:
        # The own footprint is excluded UNCONDITIONALLY -- no exclude_max_z
        # overhang rule here. That rule is right for a per-facet scan (this
        # facet's taller sibling wing genuinely shades it) and wrong for a
        # BUILDING's sky profile: a two-storey building's own upper level
        # would read as a 50-degree wall of horizon in every direction
        # (measured on #5119630: 12% annual beam left). Trees overhanging the
        # footprint are the accepted cost; they still show up in obstruction
        # detection and the per-facet shading factor.
        near = compute_horizon_profile_from_array(
            dsm_band, dsm_transform, dsm_nodata, c.x, c.y,
            azimuth_step_deg=AZ_STEP, max_distance_km=NEAR_MAX_KM,
            sample_step_m=NEAR_STEP_M,
            exclude_geom=geom.buffer(OWN_FOOTPRINT_MARGIN_M),
            observer_z=eave_z)
    except ValueError:
        pass

    if far is None and near is None:
        return None, None
    if far is None:
        return near, None
    if near is None:
        return far, far
    return {az: max(far.get(az, 0.0), near.get(az, 0.0)) for az in far}, far


def encode_horizon(profile):
    """72 uint8s (elevation deg x 2, N-first, 5-deg steps) -> base64 str."""
    vals = np.array([profile.get(i * AZ_STEP, 0.0) for i in range(N_BINS)])
    q = np.clip(np.round(vals * 2.0), 0, 255).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode("ascii")


def decode_horizon(b64):
    q = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return {i * AZ_STEP: float(v) / 2.0 for i, v in enumerate(q)}


def beam_visible_fraction(profile, hourly):
    """Share of the year's DNI that clears this horizon vs a flat one --
    the quick per-building 'how much beam does the sky cost me' number."""
    sun_az = hourly["solar_azimuth"].to_numpy()
    sun_el = hourly["solar_elevation"].to_numpy()
    dni = hourly["dni"].to_numpy()
    open_sky = sun_el > 0
    base = dni[open_sky].sum()
    if base <= 0:
        return 1.0
    h = horizon_angle_at(profile, sun_az)
    return float(dni[open_sky & (sun_el > h)].sum() / base)


def facet_horizon_factor(profile, baseline_profile, slope_deg, aspect_deg, hourly):
    """Aspect-aware beam correction: the ratio of plane-of-array BEAM energy
    surviving `profile` vs `baseline_profile` for a facet at (slope, aspect).

    The POA lookup table already zeroes beam behind the AREA's shared terrain
    profile; this converts that to the BUILDING's own combined horizon without
    rebuilding the table per building. Cosine-of-incidence weighted, so an
    east-blocking hill discounts east-facing roofs hardest -- a flat scalar
    cannot represent that. Clamped to 1.0 max: a building seeing MORE sky
    than the area baseline gains a little, capped to stay conservative."""
    sun_az = np.radians(hourly["solar_azimuth"].to_numpy())
    sun_el = np.radians(hourly["solar_elevation"].to_numpy())
    dni = hourly["dni"].to_numpy()

    tilt = np.radians(slope_deg)
    az = np.radians(aspect_deg)
    cos_inc = (np.sin(sun_el) * np.cos(tilt)
               + np.cos(sun_el) * np.sin(tilt) * np.cos(sun_az - az))
    beam = dni * np.clip(cos_inc, 0, None)

    el_deg = hourly["solar_elevation"].to_numpy()
    az_deg = hourly["solar_azimuth"].to_numpy()
    base_h = horizon_angle_at(baseline_profile, az_deg) if baseline_profile is not None else 0.0
    base_vis = el_deg > np.maximum(base_h, 0.0)
    denom = beam[base_vis].sum()
    if denom <= 0:
        return 1.0
    own_h = horizon_angle_at(profile, az_deg)
    num = beam[base_vis & (el_deg > own_h)].sum()
    return float(min(num / denom, 1.0))


def far_profile(dem_band, dem_transform, dem_nodata, geom, eave_z):
    """FAR terrain profile only, for the model-side per-building correction.
    (The near layer must not enter the yield math here: the per-facet
    building_shading_factor already scans the same neighbours/trees, and
    counting them twice double-discounts. The tab still shows the combined
    profile -- display truth vs calculation partitioning.)"""
    c = geom.centroid
    try:
        return compute_horizon_profile_from_array(
            dem_band, dem_transform, dem_nodata, c.x, c.y,
            azimuth_step_deg=AZ_STEP, max_distance_km=FAR_MAX_KM,
            sample_step_m=FAR_STEP_M, observer_z=eave_z)
    except ValueError:
        return None


def far_beam_ratio(far, baseline, hourly):
    """DNI-weighted scalar: beam surviving the building's far horizon vs the
    area baseline the POA lookup was built with. Capped at 1.0 (a hilltop
    building never claims more than the area-calibrated table)."""
    if far is None:
        return 1.0
    el = hourly["solar_elevation"].to_numpy()
    az = hourly["solar_azimuth"].to_numpy()
    dni = hourly["dni"].to_numpy()
    base_h = horizon_angle_at(baseline, az) if baseline is not None else 0.0
    base_vis = el > np.maximum(base_h, 0.0)
    denom = dni[base_vis].sum()
    if denom <= 0:
        return 1.0
    num = dni[base_vis & (el > horizon_angle_at(far, az))].sum()
    return float(min(num / denom, 1.0))

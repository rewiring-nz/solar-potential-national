"""
Per-facet near-field shading from nearby taller buildings/structures --
the direct-beam sun a roof facet loses to something else in the way,
distinct from (and on top of) the far-field terrain horizon in
src/terrain_horizon.py.

Confirmed as a real, separate gap directly on a reported building
(#5370338): the DSM shows real structures up to ~4-5m taller within the
same block, and neither the slope/aspect POA lookup nor the shared
terrain-horizon correction (one profile for the whole pilot, tuned for
distant mountains -- see terrain_horizon.py's own module comment) has any
way to notice a taller neighbour a few metres away. A narrow low-rise roof
wedged between two taller sections reads as if it had the same open sky
as a rooftop with nothing around it.

Computed per facet, not per building: an early per-building version (one
scalar at the building's footprint centroid, applied uniformly to every
facet) directly reproduced a second thing the user reported -- "this very
variable angle and height roof space is a uniform colour" -- because a
single centroid almost always lands on the building's largest, most open
facet, masking real shading on the building's own smaller/lower wings.
Checked directly on #5370338: per-facet factors ranged 0.50-1.00 across
its 12 facets, while the building-centroid version gave 0.998 for all of
them. A facet-level factor also naturally lets one wing of a complex
building shade another wing of the *same* building, which a building-level
scalar structurally cannot represent.

Method: reuses terrain_horizon.py's ray-marching, pointed at the same 1m
DSM already loaded everywhere else in this project instead of the wide
bare-earth DEM, over a short radius (nearby buildings only) instead of a
continental one. Rather than re-running the full hourly clear-sky/POA
transposition per facet (the shared SLOPE_BIN x ASPECT_BIN lookup table's
whole reason to exist is *not* re-running that per facet), this computes a
single scalar shading_factor: the fraction of the year's direct-beam
irradiance (DNI, weighted -- not just an hour count, since a blocked hour
near solar noon in summer costs far more real energy than one at a low
winter sun angle) that survives the facet's own near-field horizon, given
the sun was already visible past the shared terrain horizon. Ray samples
falling on the facet's own roof are excluded per-ray (see
OWN_ROOF_MARGIN_M) rather than via one exclusion radius, so a real close
neighbour perpendicular to the facet isn't blanked out along with the
facet's own structure. That factor then scales the whole POA lookup value
for that facet (see SolarModel.facet_yield) -- an approximation, since it
applies a DNI-derived ratio to the diffuse and ground-reflected components
too, not just the direct beam they were computed from. Documented
simplification, same conservative direction as the DNI-only treatment
terrain_horizon.py already uses for the same reason: diffuse skylight
isn't blocked by one nearby building anywhere near as completely as direct
beam is, so this somewhat over-corrects rather than under-corrects, in the
same "cost a bit of headline kWh, not overstate a shaded roof" direction
this project's obstruction/shading logic already leans.
"""

import sys
from pathlib import Path

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.terrain_horizon import compute_horizon_profile_from_array, horizon_angle_at

BUILDING_MAX_DISTANCE_KM = 0.1  # 100m -- nearby buildings/structures only, not the same
# continental-scale search terrain_horizon.py does for distant mountains
BUILDING_SAMPLE_STEP_M = 2.0  # matches the 1m DSM's own resolution closely enough to catch a
# real building edge without the per-building ray-march cost of stepping every 1m out to 100m
BUILDING_AZIMUTH_STEP_DEG = 4.0  # coarser than terrain_horizon's 2 degrees -- nearby buildings
# present much larger angular width than a distant mountain ridge, so this loses little real
# detail while roughly halving the per-building ray count (1270 of these add up)
OVERHANG_CLEARANCE_M = 2.5  # inside the own-footprint mask, anything standing more than this
# above the observer is treated as a real blocker (overhanging canopy), not as the roof itself.
OWN_ROOF_MARGIN_M = 1.0  # buffer added around the building's own footprint before the near-field
# scan starts counting anything as an "obstruction" -- covers eaves overhang and the fact the
# observer (x, y) is the footprint centroid, not necessarily the roof's own highest point


def building_shading_factor(dsm_band, dsm_transform, dsm_nodata, x, y, hourly, own_geom=None,
                              terrain_horizon_profile=None):
    """Returns a [0, 1] scalar: fraction of hourly['dni'] (already filtered
    down to hours the sun clears the shared terrain horizon, if
    terrain_horizon_profile is given) that also clears this specific
    (x, y)'s own near-field horizon. 1.0 = fully open, no nearby obstruction
    found or none tall enough to matter. hourly: the dict SolarModel.hourly
    exposes (dni, solar_azimuth, solar_elevation Series, all the same
    length/index). own_geom: the facet's (or building's) own footprint
    polygon (in the DSM's CRS) -- ray samples falling inside it (buffered
    by OWN_ROOF_MARGIN_M) are excluded per-ray so the facet's own roof
    isn't mistaken for a neighbour; omitting it disables that exclusion."""
    exclude_geom = own_geom.buffer(OWN_ROOF_MARGIN_M) if own_geom is not None else None
    # Mask the observer's own roof, but NOT anything towering over it: tree
    # canopy overhanging a roof used to be skipped as "own building" and so
    # cast no shade at all, which is precisely the case where you should not
    # be putting panels. Anything inside the footprint more than
    # OVERHANG_CLEARANCE_M above the observer keeps blocking.
    exclude_max_z = None
    if exclude_geom is not None:
        try:
            r, c = rasterio.transform.rowcol(dsm_transform, x, y)
            if 0 <= r < dsm_band.shape[0] and 0 <= c < dsm_band.shape[1]:
                z0 = float(dsm_band[r, c])
                if dsm_nodata is None or z0 != dsm_nodata:
                    exclude_max_z = z0 + OVERHANG_CLEARANCE_M
        except Exception:
            exclude_max_z = None

    try:
        profile = compute_horizon_profile_from_array(
            dsm_band, dsm_transform, dsm_nodata, x, y,
            azimuth_step_deg=BUILDING_AZIMUTH_STEP_DEG,
            max_distance_km=BUILDING_MAX_DISTANCE_KM,
            sample_step_m=BUILDING_SAMPLE_STEP_M,
            exclude_geom=exclude_geom,
            exclude_max_z=exclude_max_z,
        )
    except ValueError:
        return 1.0  # observer point outside the DSM extent -- degrade to unshaded rather than crash

    if not any(v > 0.5 for v in profile.values()):
        return 1.0  # nothing nearby taller than half a degree in any direction -- not worth the rest of this

    sun_az = hourly["solar_azimuth"].to_numpy()
    sun_el = hourly["solar_elevation"].to_numpy()
    dni = hourly["dni"].to_numpy()

    building_horizon_at_sun_az = horizon_angle_at(profile, sun_az)
    if terrain_horizon_profile is not None:
        terrain_horizon_at_sun_az = horizon_angle_at(terrain_horizon_profile, sun_az)
        baseline_visible = sun_el > terrain_horizon_at_sun_az
    else:
        baseline_visible = sun_el > 0

    baseline_dni = dni[baseline_visible].sum()
    if baseline_dni <= 0:
        return 1.0

    still_visible = baseline_visible & (sun_el > building_horizon_at_sun_az)
    return float(dni[still_visible].sum() / baseline_dni)

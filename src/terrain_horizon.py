"""
Local terrain horizon profile -- how high the surrounding mountains sit
above flat-horizon, per compass direction, as seen from the pilot area.

This exists because the solar yield model (src/solar_model.py) previously
assumed a fully open horizon: pvlib's clear-sky model computes irradiance
as if the sun were visible the moment it's above 0 degrees elevation,
anywhere in the pilot. That's a bad assumption for Queenstown specifically
-- the town sits in a basin ringed by real mountains (the Remarkables to
the south/southeast reach over 2300m; Ben Lomond and Queenstown Hill sit
right against the town itself) that measurably delay sunrise and bring
forward sunset in large parts of the sky, especially in winter when the
sun's whole daily arc is lower. NASA POWER's own cloud-correction factor
(the other half of the yield model) can't rescue this either -- its solar
data is a 1x1 degree grid (~80-110km cells at this latitude), which
treats the entire Queenstown Lakes district as one averaged point with no
awareness of the town's own specific ring of peaks.

Method: ray-march outward from a reference observer point through a wide-
area DEM (LINZ's nationwide 8m grid -- coarser than the 1m LiDAR DSM used
elsewhere in this project, but that's the right tool here: horizon-
relevant terrain is many km away, so a wide 1m fetch would be enormous for
no real benefit over 8m at that distance) along many compass directions,
and for each direction keep the single highest elevation angle any
terrain cell along that ray reaches. That's the horizon: the sun is
geometrically blocked whenever its actual elevation is below the
horizon's own value at the sun's current azimuth, regardless of the
terrain's exact shape or distance beyond that.

One profile for the whole pilot rather than one per building: the terrain
that actually matters here (the Remarkables, the ranges beyond the lake)
sits many km away, where a building's few-hundred-metre position within
the ~2km pilot bbox shifts the angle to it only marginally. Terrain much
closer to town (Ben Lomond, Queenstown Hill) could plausibly differ more
building-to-building -- documented simplification, not hidden; a
per-building profile is a natural follow-up if that turns out to matter.
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
import shapely
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AZIMUTH_STEP_DEG = 2.0
MAX_DISTANCE_KM = 30.0
SAMPLE_STEP_M = 100.0  # coarser than the 8m DEM grid -- horizon angle changes slowly with
# distance once past a few km, and this keeps a 30km ray to a few hundred samples, not thousands


def compute_horizon_profile_from_array(band, transform, nodata, observer_x, observer_y,
                                        azimuth_step_deg=AZIMUTH_STEP_DEG, max_distance_km=MAX_DISTANCE_KM,
                                        sample_step_m=SAMPLE_STEP_M, exclude_geom=None,
                                        exclude_max_z=None, observer_z=None):
    """Same ray-marching as compute_horizon_profile, but against an
    already-loaded (band, transform, nodata) instead of opening a file --
    the shared entry point both the wide-area terrain profile and
    building_shading.py's per-building near-field profile use, so opening
    (or re-reading) a raster 1270 times isn't the cost of the latter.

    exclude_geom: ray samples falling inside this geometry are skipped
    entirely (not just capped) before taking each direction's max angle.
    The wide-area terrain profile leaves it None (the nearest DEM cell is
    always many metres from anything but real terrain). building_shading.py's
    near-field scan needs it -- without it, the observer's own roof (a
    ridge or opposite slope a few metres from an off-ridge centroid, often
    *taller* than the centroid pixel) reads as a false nearby "obstruction",
    which is self-shading-by-geometry, not a real neighbour blocking the
    sun. A per-sample polygon test (rather than a single exclusion radius)
    matters here: a uniform radius sized to the building's own longest
    extent would, in one direction, also blank out a real close neighbour
    that a per-ray test correctly still sees."""
    row0, col0 = rasterio.transform.rowcol(transform, observer_x, observer_y)
    if not (0 <= row0 < band.shape[0] and 0 <= col0 < band.shape[1]):
        raise ValueError("Observer point falls outside the raster extent")
    if observer_z is None:
        # default: the raster's own surface at the observer -- right for the
        # wide-area terrain profile. A BUILDING's horizon is seen from eave
        # height, which on a bare-earth DEM the cell cannot know; callers
        # doing per-building work pass observer_z explicitly.
        observer_z = float(band[row0, col0])

    distances = np.arange(sample_step_m, max_distance_km * 1000, sample_step_m)
    azimuths = np.arange(0, 360, azimuth_step_deg)

    # Fully vectorized over (azimuth x distance) rather than looping azimuths in Python --
    # building_shading.py now calls this once per roof facet rather than once per building
    # (~6x more calls across the pilot), so the per-call cost matters here in a way it didn't
    # when this was only the once-per-building or once-for-the-whole-pilot terrain profile.
    az_rad = np.radians(azimuths)
    dx, dy = np.sin(az_rad), np.cos(az_rad)  # compass bearing -> unit vector (east, north)
    xs = observer_x + np.outer(dx, distances)  # (n_az, n_dist)
    ys = observer_y + np.outer(dy, distances)

    rows, cols = rasterio.transform.rowcol(transform, xs.ravel(), ys.ravel())
    rows = np.asarray(rows).reshape(xs.shape)
    cols = np.asarray(cols).reshape(xs.shape)
    in_bounds = (rows >= 0) & (rows < band.shape[0]) & (cols >= 0) & (cols < band.shape[1])
    zs = np.where(in_bounds, band[np.clip(rows, 0, band.shape[0] - 1),
                                  np.clip(cols, 0, band.shape[1] - 1)], np.nan)
    if nodata is not None:
        zs = np.where(zs == nodata, np.nan, zs)
    valid = in_bounds & ~np.isnan(zs)
    if exclude_geom is not None:
        in_excl = shapely.contains_xy(exclude_geom, xs, ys).reshape(xs.shape)
        if exclude_max_z is not None:
            # Only mask what is plausibly the observer's OWN ROOF. Anything
            # inside the footprint standing well above it -- overhanging tree
            # canopy, a neighbour's crown reaching over -- is a real blocker
            # and must keep shading the panel beneath it.
            in_excl = in_excl & (zs <= exclude_max_z)
        valid = valid & ~in_excl

    rise = zs - observer_z
    run = np.broadcast_to(distances, xs.shape)
    angles = np.where(valid, np.degrees(np.arctan2(rise, run)), -np.inf)
    max_angle = angles.max(axis=1)
    max_angle = np.where(np.isfinite(max_angle), max_angle, 0.0)

    return {float(az): float(a) for az, a in zip(azimuths, max_angle)}


def compute_horizon_profile(dem_path, observer_x, observer_y,
                             azimuth_step_deg=AZIMUTH_STEP_DEG, max_distance_km=MAX_DISTANCE_KM,
                             sample_step_m=SAMPLE_STEP_M):
    """observer_x/y in the DEM's own CRS (metres -- NZTM2000 for the LINZ
    8m DEM). Returns dict {azimuth_deg: horizon_elevation_angle_deg},
    azimuth_deg compass bearing (0=N, 90=E). Positive angle = terrain
    sits above flat-horizon in that direction; 0 = fully open."""
    with rasterio.open(dem_path) as ds:
        return compute_horizon_profile_from_array(ds.read(1), ds.transform, ds.nodata,
                                                    observer_x, observer_y, azimuth_step_deg,
                                                    max_distance_km, sample_step_m)


def horizon_angle_at(profile, azimuth_deg):
    """Linear interpolation between the profile's own azimuth samples.
    azimuth_deg may be a scalar or a numpy array -- callers computing this
    across a full 8760-hour series pass the whole array in one call rather
    than looping per-hour, since re-sorting profile.keys() and re-building
    the lookup arrays per call is the actual cost at that scale, not the
    interpolation itself."""
    azs = np.array(sorted(profile.keys()))
    vals = np.array([profile[a] for a in azs])
    az = np.asarray(azimuth_deg) % 360
    result = np.interp(az, azs, vals, period=360)
    return float(result) if np.isscalar(azimuth_deg) or np.ndim(azimuth_deg) == 0 else result

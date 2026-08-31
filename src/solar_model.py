"""
Annual/daily kWh per roof facet, from slope + aspect + latitude via pvlib,
bias-corrected against NASA POWER's actual (cloud-adjusted) irradiance so
the estimate reflects real Queenstown weather, not idealised clear-sky
sun -- then further corrected for the surrounding mountains actually
blocking the sun, which neither of those two sources knows about on its
own (see the terrain-horizon paragraph below).

Method:
1. pvlib's Ineichen clear-sky model gives hourly GHI/DNI/DHI for the
   pilot's location over a representative year.
2. NASA POWER's climatology API gives real monthly-average GHI
   (ALLSKY_SFC_SW_DWN) alongside its own clear-sky reference
   (CLRSKY_SFC_SW_DWN) for the same point. The ratio of the two is an
   empirical "how much does cloud cut irradiance this month" factor,
   applied to every clear-sky hour in that month before anything else.
   Important caveat, not previously documented here: NASA POWER's solar
   data is a 1x1 degree grid (roughly 80-110km per cell at this
   latitude) -- confirmed directly against its own docs. That's the
   entire Queenstown Lakes district and a good part of surrounding Otago
   averaged into one number, with no awareness of Queenstown's own
   specific microclimate or terrain. It's the best broad cloud-climatology
   correction available without a paid data source, but it cannot see
   anything at town scale, which is exactly what the terrain horizon step
   below is for.
3. Terrain horizon: both of the above assume a fully open horizon --
   pvlib's clear-sky model treats the sun as visible the instant it's
   above 0 degrees geometric elevation, anywhere in the pilot. Queenstown
   sits in a real basin ringed by real mountains (the Remarkables to the
   south/southeast top 2300m; Ben Lomond and Queenstown Hill sit right
   against the town). Confirmed directly from LINZ's nationwide 8m DEM
   (src/terrain_horizon.py): the local horizon averages ~9 degrees above
   flat and peaks near 17 degrees in some directions from the pilot's own
   reference point -- large enough to matter, especially in winter when
   the sun's whole daily arc is already low. Whenever the sun's actual
   elevation is below the horizon angle at its current azimuth, the
   direct-beam component (DNI) is zeroed for that hour before anything
   is transposed onto a roof plane; diffuse skylight (DHI) is left as-is
   -- a real simplification (a deep basin also loses some sky view for
   diffuse light, not modelled here), the same direction PVGIS and other
   horizon-shading tools take as their own standard simplification.
4. For each (tilt, azimuth) combination, pvlib transposes the
   cloud- and horizon-corrected horizontal irradiance onto the tilted
   panel plane and the result is integrated over the year -> kWh/m2/year
   actually expected on a facet with that slope/aspect.
5. That's cached in a small lookup table (5 degree slope bins x 10 degree
   aspect bins = a few hundred pvlib runs total) rather than re-run per
   facet -- there are 3600+ facets in the pilot, but only ~350 distinct
   (slope, aspect) bins.
6. kWh -> panel output uses the PVWatts-style linear approximation:
   dc_kWh = POA_kWh_per_m2 * panel_rated_kW (since panel_rated_w is
   defined at the 1000 W/m2 STC reference, "kWh of POA irradiance per m2"
   converts directly to "equivalent full-rated-power hours"), then
   config.PV_ASSUMPTIONS' inverter efficiency and system derate are
   applied on top. This ignores temperature derating specifically (rolled
   into the flat system_derate_pct instead) -- a real design would model
   cell temperature separately, documented simplification for the pilot.

One shared terrain horizon for the whole pilot, not per-building: see
src/terrain_horizon.py's own module comment for why -- the terrain that
actually matters here sits many km away, where a building's position
within the ~2km pilot bbox barely changes the angle to it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import pyproj
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.terrain_horizon import compute_horizon_profile, horizon_angle_at

SLOPE_BIN_DEG = 5
ASPECT_BIN_DEG = 10
MONTH_NAMES = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def pilot_location():
    min_lon, min_lat, max_lon, max_lat = config.PILOT_BBOX
    lat, lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
    return lat, lon


def fetch_nasa_power_monthly_factors(lat, lon):
    """Returns dict {month_abbr: actual/clearsky GHI ratio}, e.g. {'JAN': 0.69, ...}.
    NASA POWER's solar data is a 1x1 degree grid (~80-110km cells at this
    latitude, confirmed directly against its own docs) -- the entire
    Queenstown Lakes district and a good part of surrounding Otago
    averaged into one point, no awareness of any town's own microclimate.
    Kept as the portable fallback (see NIWA_STATIONS below) for any region
    without a real nearby ground station on file."""
    url = "https://power.larc.nasa.gov/api/temporal/climatology/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "format": "JSON",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    param = resp.json()["properties"]["parameter"]
    allsky, clrsky = param["ALLSKY_SFC_SW_DWN"], param["CLRSKY_SFC_SW_DWN"]
    return {m: allsky[m] / clrsky[m] for m in MONTH_NAMES}


# --- NIWA ground-station-derived monthly factors (preferred over NASA POWER) --------
#
# NASA POWER's 100km-scale grid can't see Queenstown's own microclimate at
# all (see the docstring above) -- confirmed directly: cross-checked
# against real ground-truth sunshine-hours climatology (1991-2020 mean)
# from NIWA's Queenstown station (5446, right in the pilot bbox) and found
# NASA POWER overstates real local GHI by 8-22% in every single month,
# worst in winter (17-22%, May-Aug) -- same direction and same season as
# the separate terrain-horizon correction above, independent confirmation
# rather than a restatement of the same effect (this is a cloud/microclimate
# gap, horizon shading is a geometric one).
#
# Sunshine *hours* aren't irradiance directly, so converting real station
# data into a usable GHI estimate uses the Angstrom-Prescott relation
# (Glover & McCulloch 1958 coefficients, a standard, published method for
# exactly this conversion -- Duffie & Beckman, "Solar Engineering of
# Thermal Processes"):
#   H = H0 * (a + b * (n / N))
# H0 = daily extraterrestrial irradiation (pure astronomy, no data needed),
# N = maximum possible sunshine hours (day length), n = NIWA's actual mean
# sunshine hours, a = 0.29*cos(latitude), b = 0.52. The Glover-McCulloch
# coefficients are a well-established global default, not calibrated
# specifically to Queenstown's own station (no local regression was
# available for this pilot) -- a real residual source of imprecision,
# documented rather than hidden, though it's the standard, defensible
# choice for a location without its own published a/b fit.
#
# Pilot-specific by construction (hardcoded Queenstown station data) --
# the natural place a national rollout falls back to NASA POWER instead:
# a future region gets this treatment only once its own nearest NIWA
# station (or equivalent local ground truth) is looked up and added here.

NIWA_STATIONS = {
    "queenstown": {
        "lat": -45.03476, "lon": 168.66364,  # station 5446, right in the pilot bbox
        # Mean monthly total sunshine hours, 1991-2020 (Earth Sciences NZ / NIWA
        # "Mean monthly sunshine (hours)" climate normals).
        "sunshine_hours": {
            "JAN": 236.9, "FEB": 209.3, "MAR": 180.0, "APR": 136.2,
            "MAY": 83.8, "JUN": 69.3, "JUL": 85.2, "AUG": 119.5,
            "SEP": 151.0, "OCT": 197.0, "NOV": 217.9, "DEC": 212.6,
        },
    },
}
NIWA_MATCH_RADIUS_DEG = 0.5  # ~50km -- how close the requested lat/lon must be to a station
# for its ground-truth data to be trustworthy as a stand-in; a wider swap-in would reintroduce
# the same "distant point, unknown microclimate" problem this whole correction exists to avoid
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_REP_DAY = [17, 47, 75, 105, 135, 162, 198, 228, 258, 288, 318, 344]  # Duffie & Beckman
GLOVER_MCCULLOCH_B = 0.52


def _nearest_niwa_station(lat, lon):
    best, best_dist = None, None
    for station in NIWA_STATIONS.values():
        dist = np.hypot(station["lat"] - lat, station["lon"] - lon)
        if best_dist is None or dist < best_dist:
            best, best_dist = station, dist
    if best is not None and best_dist <= NIWA_MATCH_RADIUS_DEG:
        return best
    return None


def fetch_niwa_derived_monthly_factors(lat, lon, pvlib_clearsky_daily_mean):
    """Returns dict {month_abbr: actual/clearsky GHI ratio} derived from a
    nearby NIWA station's real sunshine-hours climatology via the
    Angstrom-Prescott relation (see module comment), or None if no
    station is close enough to trust. pvlib_clearsky_daily_mean: dict
    {month_abbr: mean daily clear-sky GHI in Wh/m2/day} from this same
    location's own pvlib Ineichen run, so the returned ratio is on
    exactly the same clear-sky reference the rest of the pipeline uses
    (not NASA POWER's own, different, clear-sky model)."""
    station = _nearest_niwa_station(lat, lon)
    if station is None:
        return None

    lat_rad = np.radians(station["lat"])
    factors = {}
    for i, month in enumerate(MONTH_NAMES):
        day = MONTH_REP_DAY[i]
        decl = np.radians(23.45 * np.sin(np.radians(360 * (284 + day) / 365)))
        ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(decl), -1, 1))  # sunset hour angle, rad
        daylight_hours = (2 / 15) * np.degrees(ws)
        n_avg = station["sunshine_hours"][month] / DAYS_IN_MONTH[i]

        h0_joules = ((24 * 3600 / np.pi) * 1367.0 * (1 + 0.033 * np.cos(np.radians(360 * day / 365)))
                     * (np.cos(lat_rad) * np.cos(decl) * np.sin(ws) + ws * np.sin(lat_rad) * np.sin(decl)))
        a_coef = 0.29 * np.cos(lat_rad)
        h_wh_per_m2 = h0_joules / 3600 * (a_coef + GLOVER_MCCULLOCH_B * (n_avg / daylight_hours))

        factors[month] = h_wh_per_m2 / pvlib_clearsky_daily_mean[month]

    return factors


# --- SolarView calibration (measured-radiation ground truth) -----------------
# Monthly mean daily global horizontal irradiance (kWh/m2/day) at the pilot
# center, exported from NIWA SolarView (custom location -45.03, 168.66, tilt 0,
# 21 Aug 2026 export): 36 years of MEASURED radiation from Queenstown Aero AWS
# (NIWA agent 5451), including NIWA's own local-terrain horizon.
#
# Exists because a direct comparison against this export showed the
# sunshine-hours-derived factors (fetch_niwa_derived_monthly_factors) run the
# whole model ~14% LOW at horizontal (1174 vs SolarView's 1365 kWh/m2/yr) and
# distort the seasonal shape (Angstrom-Prescott inference bottoms in June;
# measured data bottoms across Nov-May relative to clear sky). The clear-sky
# models themselves agree closely (~1800 kWh/m2/yr cloudless both sides), so
# calibrating the monthly factor directly against SolarView's measured monthly
# means transfers 36 years of real radiation data into the pipeline in one
# step. Sunshine-derived factors remain as the fallback outside the pilot.
# Attribution & licensing: these 12 monthly aggregates are derived from NIWA
# SolarView output (https://niwa.co.nz/renewable-energy/solarview), used here
# as a non-commercial calibration reference per NIWA's EULA. The raw SolarView
# export files are deliberately NOT committed to this repository -- NIWA's
# terms cover use of the tool's output, not republication of it.
SOLARVIEW_CAL_LOCATION = (-45.03, 168.66)
# ~30 km. Widened from 0.05 (5 km) on 31 Aug after the PVGIS cross-check showed
# what the 5 km gate was costing: it left 23 of 24 Queenstown regions on the
# sunshine-hours fallback that THIS FILE already documents as running ~14% low,
# and the external check measured the result at 10-19% low on north-facing
# roofs -- the roofs that matter.
#
# The measured mean monthly factor at the pilot is 0.729; the fallback gives
# 0.612 at Arrowtown, 20 km away in the same basin. A 16% difference in cloud
# climate over 20 km is not credible, and PVGIS puts Arrowtown slightly SUNNIER
# than the CBD, not darker.
#
# What travels is the FACTOR -- the ratio of real to clear-sky irradiance, i.e.
# how cloudy the place is. That is a basin-scale climate property. Everything
# genuinely local (latitude, elevation, air mass, and each building's own
# terrain horizon) is already handled by the clear-sky model and the horizon
# scan, so widening the gate does not smear site-specific geometry around.
#
# 0.30 covers the whole Queenstown district (furthest region 0.222 deg) with
# margin, and comes nowhere near Wellington at 7.1 deg -- that repo shares this
# file and must keep falling through to its own calibration.
SOLARVIEW_CAL_MAX_DIST_DEG = 0.30
SOLARVIEW_MEASURED_GHI_KWH_M2_DAY = {
    "JAN": 6.29, "FEB": 5.22, "MAR": 3.78, "APR": 2.31, "MAY": 1.38, "JUN": 1.09,
    "JUL": 1.41, "AUG": 2.26, "SEP": 3.56, "OCT": 4.98, "NOV": 6.11, "DEC": 6.49,
}


def solarview_calibrated_monthly_factors(lat, lon, clearsky_daily_mean_horizon_adj):
    """Returns dict {month_abbr: factor} such that clearsky * factor, with the
    terrain-horizon DNI zeroing already reflected in the denominator,
    reproduces SolarView's measured monthly GHI exactly -- or None when the
    requested location is too far from where the SolarView export was taken.

    clearsky_daily_mean_horizon_adj: mean daily clear-sky GHI (Wh/m2/day) per
    month WITH the local terrain-horizon blocking already applied. Using the
    horizon-adjusted series as the denominator matters: SolarView's measured
    values inherently include the real mountains (the station physically sits
    behind them), and our pipeline separately zeroes horizon-blocked DNI -- a
    factor computed against the unblocked clear sky would count the mountains
    twice, landing ~5-8% under the measured record it was calibrated to."""
    dist = np.hypot(lat - SOLARVIEW_CAL_LOCATION[0], lon - SOLARVIEW_CAL_LOCATION[1])
    if dist > SOLARVIEW_CAL_MAX_DIST_DEG:
        return None
    return {
        month: SOLARVIEW_MEASURED_GHI_KWH_M2_DAY[month] * 1000.0 / clearsky_daily_mean_horizon_adj[month]
        for month in MONTH_NAMES
    }


# --- NIWA measured station normals (national calibration) --------------------
# Mean daily global horizontal radiation, 1991-2020 normals, from NIWA's
# published "Mean daily global radiation (MJ/sq m)" dataset, converted to
# kWh/m2/day. 28 stations covering New Zealand end to end.
#
# This exists because the sunshine-hours inference below is MEASURABLY wrong --
# this file already recorded it running ~14% low at Queenstown, and the PVGIS
# cross-check found the same signature wherever it was in use. Sunshine hours
# are a proxy; these are radiation actually measured with a pyranometer.
#
# It also makes the national rollout honest by default: any future region gets
# calibrated against the nearest measured station instead of an inference known
# to be low.
#
# Attribution: derived from NIWA (Earth Sciences New Zealand) published climate
# normals, https://niwa.co.nz/climate-and-weather/mean-daily-global-radiation-mjsq-m
# Only these 12-value monthly aggregates are kept; the source workbook is not
# committed.
#
# Queenstown keeps its SolarView calibration in preference to the Queenstown
# Aero station row here, and the two are consistent once you account for what
# each describes: SolarView was exported AT the pilot centre and carries NIWA's
# terrain horizon for that point (1365 kWh/m2/yr), while the station normal is
# measured at the airport on open ground 357 m up (1466). The valley floor
# getting ~7% less than the airport is the mountains, not a disagreement.
NIWA_STATION_MAX_DIST_DEG = 1.5   # ~150 km; NZ stations are sparse and cloud
# climate travels further than terrain does. Beyond this, fall through.

NIWA_MEASURED_GHI_STATIONS = [
    ("Kaitaia", -35.1335, 173.2629, [6.504, 5.653, 4.61, 3.324, 2.447, 2.039, 2.206, 2.995, 3.989, 5.122, 5.858, 6.188]),
    ("Whangarei", -35.7706, 174.3629, [6.033, 5.149, 4.257, 3.104, 2.299, 1.913, 2.066, 2.774, 3.756, 4.752, 5.548, 5.775]),
    ("Auckland", -37.0081, 174.7887, [6.289, 5.488, 4.453, 3.224, 2.25, 1.827, 2.006, 2.761, 3.784, 4.872, 5.823, 6.164]),
    ("Tauranga", -37.6750, 176.1920, [6.637, 5.687, 4.676, 3.32, 2.378, 1.888, 2.018, 2.788, 3.805, 4.992, 6.025, 6.348]),
    ("Hamilton", -37.7739, 175.3052, [6.415, 5.498, 4.402, 3.206, 2.252, 1.813, 1.974, 2.656, 3.644, 4.728, 5.765, 5.917]),
    ("Rotorua", -38.1059, 176.3148, [6.395, 5.477, 4.51, 3.182, 2.233, 1.761, 1.931, 2.647, 3.602, 4.716, 5.759, 6.052]),
    ("Gisborne", -38.6275, 177.9218, [6.556, 5.532, 4.294, 2.972, 2.437, 1.877, 2.135, 2.767, 3.698, 5.122, 5.847, 6.321]),
    ("New Plymouth", -39.0079, 174.1838, [6.679, 5.907, 4.683, 3.228, 2.177, 1.713, 1.946, 2.741, 3.754, 4.901, 6.115, 6.344]),
    ("Napier", -39.4703, 176.8646, [6.576, 5.562, 4.444, 3.132, 2.261, 1.788, 1.94, 2.775, 3.921, 5.208, 6.136, 6.458]),
    ("Whanganui", -39.9370, 175.0451, [6.663, 5.715, 4.471, 3.043, 2.03, 1.594, 1.81, 2.547, 3.61, 4.742, 5.97, 6.404]),
    ("Palmerston North", -40.3820, 175.6091, [6.125, 5.573, 4.304, 2.986, 1.966, 1.501, 1.75, 2.453, 3.411, 4.322, 5.511, 5.889]),
    ("Masterton", -40.9570, 175.7070, [5.144, 4.508, 3.391, 2.217, 1.597, 1.167, 1.392, 1.947, 2.922, 4.088, 4.944, 4.987]),
    ("Wellington", -41.2844, 174.7679, [6.191, 5.656, 4.161, 2.799, 1.825, 1.322, 1.592, 2.394, 3.458, 4.646, 5.531, 6.044]),
    ("Nelson", -41.2990, 173.2260, [6.246, 5.742, 4.591, 3.181, 2.171, 1.603, 1.764, 2.455, 3.599, 4.671, 5.94, 6.21]),
    ("Blenheim", -41.4989, 173.9629, [6.681, 5.898, 4.664, 3.149, 2.145, 1.63, 1.843, 2.686, 3.884, 5.203, 6.294, 6.71]),
    ("Westport", -41.7407, 171.5798, [6.092, 5.422, 4.177, 2.746, 1.783, 1.36, 1.609, 2.33, 3.308, 4.36, 5.598, 5.848]),
    ("Kaikoura", -42.4204, 173.6963, [5.954, 5.082, 4.126, 2.815, 1.895, 1.472, 1.663, 2.476, 3.571, 4.81, 5.963, 6.101]),
    ("Hokitika", -42.7120, 170.9840, [5.936, 5.138, 4.057, 2.676, 1.736, 1.349, 1.607, 2.249, 3.196, 4.339, 5.442, 5.747]),
    ("Christchurch", -43.4930, 172.5370, [6.001, 5.187, 3.977, 2.659, 1.751, 1.32, 1.537, 2.283, 3.434, 4.755, 5.911, 6.079]),
    ("Mt Cook", -43.7363, 170.0977, [6.157, 5.554, 4.129, 2.798, 1.583, 1.242, 1.46, 2.288, 3.285, 4.612, 5.833, 6.089]),
    ("Lake Tekapo", -44.0017, 170.4432, [6.806, 6.039, 4.534, 3.072, 1.899, 1.544, 1.722, 2.592, 3.898, 5.204, 6.546, 6.833]),
    ("Timaru", -44.4088, 171.2543, [5.616, 4.798, 3.653, 2.389, 1.451, 1.049, 1.308, 2.03, 3.357, 4.588, 5.549, 5.791]),
    ("Queenstown", -45.0176, 168.7402, [6.562, 5.781, 4.33, 2.841, 1.691, 1.346, 1.574, 2.401, 3.648, 5.076, 6.189, 6.766]),
    ("Clyde", -45.2034, 169.3182, [6.315, 5.522, 4.012, 2.645, 1.559, 1.142, 1.386, 2.309, 3.619, 4.814, 5.91, 6.5]),
    ("Manapouri", -45.5329, 167.6416, [6.128, 5.329, 3.901, 2.539, 1.445, 1.091, 1.246, 2.081, 3.268, 4.619, 5.824, 6.352]),
    ("Dunedin", -45.9013, 170.5147, [5.37, 4.748, 3.521, 2.239, 1.365, 1.021, 1.221, 1.872, 3.024, 4.226, 5.122, 5.571]),
    ("Invercargill", -46.4110, 168.3178, [5.66, 4.832, 3.478, 2.249, 1.324, 1.015, 1.222, 1.936, 3.051, 4.316, 5.376, 5.926]),
    ("Chatham Islands", -43.8168, -176.4750, [6.071, 5.008, 3.753, 2.448, 1.568, 1.241, 1.472, 2.096, 3.21, 4.5, 5.753, 5.927]),
]


def niwa_station_measured_monthly_factors(lat, lon, clearsky_daily_mean):
    """Factors from the NEAREST measured-radiation station, or None if none is
    within NIWA_STATION_MAX_DIST_DEG.

    clearsky_daily_mean is the OPEN (not horizon-adjusted) clear-sky series, to
    match the sunshine-hours path: a station normal is measured on open ground,
    so the ratio it defines is cloud climate alone. The target site's own
    terrain is applied separately, per building, by the horizon scan -- which
    is why the adjusted series would double-count it here."""
    best, best_d = None, None
    for name, slat, slon, monthly in NIWA_MEASURED_GHI_STATIONS:
        d = np.hypot(lat - slat, lon - slon)
        if best_d is None or d < best_d:
            best, best_d = (name, monthly), d
    if best is None or best_d > NIWA_STATION_MAX_DIST_DEG:
        return None
    name, monthly = best
    return {MONTH_NAMES[i]: monthly[i] * 1000.0 / clearsky_daily_mean[MONTH_NAMES[i]]
            for i in range(12)}


def build_poa_lookup_table(lat, lon, tz="Pacific/Auckland", year=2023, horizon_profile=None):
    """Returns dict {(slope_bin_deg, aspect_bin_deg): annual_poa_kwh_per_m2}.
    horizon_profile: optional dict from terrain_horizon.compute_horizon_profile
    -- when given, hours where the sun's actual position sits below the
    local terrain horizon have their direct-beam (DNI) contribution
    zeroed before transposition (see module comment)."""
    location = pvlib.location.Location(lat, lon, tz=tz, altitude=310)  # ~Queenstown lake level
    times = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="1h", tz=tz)

    clearsky = location.get_clearsky(times, model="ineichen")
    solpos = location.get_solarposition(times)

    month_abbr_by_num = dict(enumerate(MONTH_NAMES, start=1))
    daily_ghi = clearsky["ghi"].resample("1D").sum()  # hourly W summed over a day = Wh/m2/day
    clearsky_daily_mean = {
        month_abbr_by_num[m]: daily_ghi[daily_ghi.index.month == m].mean()
        for m in range(1, 13)
    }

    # Horizon-adjusted clear-sky monthly means -- the denominator the SolarView
    # calibration needs (see solarview_calibrated_monthly_factors for why the
    # horizon must be inside the denominator, not counted a second time later).
    if horizon_profile is not None:
        horizon_at_az = horizon_angle_at(horizon_profile, solpos["azimuth"].to_numpy())
        blocked_cs = solpos["apparent_elevation"].to_numpy() < horizon_at_az
        ghi_cs_adj = clearsky["ghi"].where(~blocked_cs, clearsky["dhi"])
    else:
        ghi_cs_adj = clearsky["ghi"]
    daily_ghi_adj = ghi_cs_adj.resample("1D").sum()
    clearsky_daily_mean_adj = {
        month_abbr_by_num[m]: daily_ghi_adj[daily_ghi_adj.index.month == m].mean()
        for m in range(1, 13)
    }

    # Factor priority: measured-radiation calibration (pilot-local) >
    # sunshine-hours inference (any NZ location near a NIWA station) >
    # NASA POWER (portable global fallback).
    monthly_factor = solarview_calibrated_monthly_factors(lat, lon, clearsky_daily_mean_adj)
    if monthly_factor is None:
        # Measured station radiation before inferred-from-sunshine-hours: the
        # inference is documented above as ~14% low and the PVGIS check found
        # that signature everywhere it was used.
        monthly_factor = niwa_station_measured_monthly_factors(lat, lon, clearsky_daily_mean)
    if monthly_factor is None:
        monthly_factor = fetch_niwa_derived_monthly_factors(lat, lon, clearsky_daily_mean)
    if monthly_factor is None:  # no NIWA station close enough to trust -- portable fallback
        monthly_factor = fetch_nasa_power_monthly_factors(lat, lon)
    factor_series = times.month.map(lambda m: monthly_factor[month_abbr_by_num[m]])
    factor_series = pd.Series(factor_series, index=times)

    # Scale the GHI -- the quantity the monthly factors actually calibrate --
    # and then DECOMPOSE it into beam and diffuse, instead of scaling all three
    # clear-sky components by the same number.
    #
    # Scaling all three preserves the CLEAR-SKY beam:diffuse ratio at every
    # hour, which is wrong for a cloudy climate: real cloud converts beam into
    # diffuse rather than removing both in proportion. The consequence is
    # directional and was invisible to every internal check, because internal
    # numbers all derive from this same table. Measured against PVGIS
    # (src/validate_against_pvgis.py) at 35 degrees tilt:
    #     Queenstown   north +11.1%   south -18.3%
    #     Island Bay    north -4.8%   south -16.6%
    # Sun-facing pitched roofs overstated, surfaces living on diffuse light
    # (south-facing, shaded, winter) badly understated, flat roofs about right
    # because they see the total either way.
    #
    # Erbs is the standard empirical decomposition from the clearness index and
    # needs nothing we do not already have. DIRINT is more accurate hour by hour
    # but wants pressure and dew point; our series is a monthly-mean-scaled
    # clear-sky day, so the extra fidelity would be spurious.
    ghi = clearsky["ghi"] * factor_series
    _split = pvlib.irradiance.erbs(ghi, solpos["apparent_zenith"], times)
    dni = _split["dni"].fillna(0.0)
    dhi = _split["dhi"].fillna(0.0)

    if horizon_profile is not None:
        horizon_at_sun_az = horizon_angle_at(horizon_profile, solpos["azimuth"].to_numpy())
        blocked = solpos["apparent_elevation"].to_numpy() < horizon_at_sun_az
        # Direct beam is geometrically blocked by terrain; diffuse skylight from the rest of
        # the sky dome isn't modelled as blocked here (documented simplification, see module
        # comment) so GHI drops to just DHI on a blocked hour rather than to zero.
        dni = dni.where(~blocked, 0.0)
        ghi = ghi.where(~blocked, dhi)

    # Perez, not the default isotropic sky. An isotropic dome spreads diffuse
    # light evenly, which over-feeds any surface pointed AWAY from the sun --
    # real diffuse is strongly anisotropic, concentrated near the sun
    # (circumsolar) and near the horizon. With the beam:diffuse split fixed but
    # the sky still isotropic, PVGIS put us +12% on a 20 degree south face and
    # +20% at 35 degrees while north faces were already within a couple of
    # percent: the classic signature. Perez is the standard anisotropic model
    # and is the family PVGIS itself uses.
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    airmass = location.get_airmass(times, solar_position=solpos)["airmass_relative"]

    lookup = {}
    for slope_bin in range(0, config.MAX_ROOF_SLOPE_DEG + SLOPE_BIN_DEG, SLOPE_BIN_DEG):
        for aspect_bin in range(0, 360, ASPECT_BIN_DEG):
            poa = pvlib.irradiance.get_total_irradiance(
                surface_tilt=slope_bin,
                surface_azimuth=aspect_bin,
                dni=dni, ghi=ghi, dhi=dhi,
                solar_zenith=solpos["apparent_zenith"],
                solar_azimuth=solpos["azimuth"],
                dni_extra=dni_extra, airmass=airmass, model="perez",
            )
            annual_kwh_per_m2 = poa["poa_global"].sum() / 1000  # Wh -> kWh (hourly W values summed = Wh)
            lookup[(slope_bin, aspect_bin)] = annual_kwh_per_m2

    # dni/solpos (after the terrain-horizon and cloud corrections above) are also handed back for
    # src/building_shading.py to reuse directly -- computing this hourly series is most of this
    # function's own cost, and building-level near-field shading needs the exact same series to
    # test each building's own horizon against, not a re-derived approximation of it.
    hourly = {"dni": dni, "solar_azimuth": solpos["azimuth"], "solar_elevation": solpos["apparent_elevation"]}
    return lookup, monthly_factor, hourly


def _nearest_bin(value, bin_size, max_value=None):
    b = round(value / bin_size) * bin_size
    if max_value is not None:
        b = min(b, max_value)
    return int(b) % 360 if max_value is None else int(b)


DEM_WIDE_PATH = Path(__file__).resolve().parent.parent / "data" / "dem_wide_mosaic.tif"


def _pilot_horizon_profile(lat, lon):
    """Computes the terrain horizon profile from the wide-area DEM if it's
    present, else None (falls back to the old open-horizon assumption --
    e.g. a dev environment that hasn't fetched data/dem_wide_mosaic.tif
    yet, or a future region without one). Not cached to disk: this is one
    ~180-ray DEM scan per SolarModel construction, a couple seconds, not
    worth the staleness risk of a cache keyed on a DEM file that could
    change."""
    if not DEM_WIDE_PATH.exists():
        return None
    to_nztm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    x, y = to_nztm.transform(lon, lat)
    try:
        return compute_horizon_profile(str(DEM_WIDE_PATH), x, y)
    except ValueError:
        return None  # observer point outside the DEM extent -- degrade to open-horizon rather than crash


class SolarModel:
    def __init__(self, lat=None, lon=None):
        if lat is None or lon is None:
            lat, lon = pilot_location()
        self.lat, self.lon = lat, lon
        self.horizon_profile = _pilot_horizon_profile(lat, lon)
        self.lookup, self.monthly_factor, self.hourly = build_poa_lookup_table(
            lat, lon, horizon_profile=self.horizon_profile)

    def annual_poa_kwh_per_m2(self, slope_deg, aspect_deg):
        slope_bin = _nearest_bin(slope_deg, SLOPE_BIN_DEG, max_value=config.MAX_ROOF_SLOPE_DEG)
        aspect_bin = _nearest_bin(aspect_deg, ASPECT_BIN_DEG)
        return self.lookup[(slope_bin, aspect_bin)]

    def facet_yield(self, facet, n_panels, shading_factor=1.0):
        """Returns dict with kwp, dc_kwh_year, ac_kwh_year, ac_kwh_day_avg for
        n_panels sitting on this facet. shading_factor: optional [0, 1]
        multiplier from src/building_shading.py -- see its own module
        comment -- for direct-beam sun this facet's own building loses to
        nearby taller buildings/structures, on top of the slope/aspect and
        terrain-horizon effects already baked into the lookup table."""
        assumptions = config.PV_ASSUMPTIONS
        kwp = n_panels * assumptions["panel_rated_power_w"] / 1000
        poa = self.annual_poa_kwh_per_m2(facet["slope_deg"], facet["aspect_deg"]) * shading_factor
        dc_kwh_year = poa * kwp
        ac_kwh_year = (dc_kwh_year
                       * (assumptions["inverter_efficiency_pct"] / 100)
                       * (1 - assumptions["system_derate_pct"] / 100))
        return {
            "kwp": kwp,
            "dc_kwh_year": dc_kwh_year,
            "ac_kwh_year": ac_kwh_year,
            "ac_kwh_day_avg": ac_kwh_year / 365,
        }

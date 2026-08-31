"""
Export seasonal generation-curve shapes to data/seasonal_curves.json for the
frontend's per-building season charts.

For every (slope, aspect) bin: per season (NZ: summer=DJF, autumn=MAM,
winter=JJA, spring=SON), a 24-value hourly curve in kW of AC output per kWp
of installed panels --
- "avg":  mean hour-of-day output over the season's days, cloud-adjusted
          with the same calibrated monthly factors the yield model uses.
- "peak": the season's best clear-sky day (max daily clear-sky POA total),
          no cloud derate -- "a very sunny day in that season".
Terrain is deliberately NOT applied here. It used to be: the curves were
built with the pilot's horizon profile, direct beam zeroed when the sun sat
behind Queenstown's mountains. But the frontend then multiplies every curve
by the building's own "tshade" mask (src/build_terrain_masks.py, one horizon
per 150m cell), so terrain was being counted twice -- measured on 28 Rees St
in winter, 16:00 read 0.010 kW/kWp where diffuse-only physics gives ~0.058:
the curve had already collapsed 0.318 -> 0.058 from the horizon, then the
mask's diffuse floor cut it another 82%. Roughly 5x too low in exactly the
winter afternoon hours the chart exists to show.

So the curve now carries only what varies with SLOPE AND ASPECT -- clear-sky
POA, cloud-corrected -- and tshade carries everything that varies with
LOCATION. One mechanism each, which is also what makes an Arrowtown valley
roof differ from a downtown one rather than both inheriting the pilot's
mountains.

(The monthly cloud factors still come from the pilot's calibration, and that
calibration used a horizon-adjusted denominator -- see
solar_model.solarview_calibrated_monthly_factors. Applying them to an open
clear sky is mildly inconsistent, worth far less than the 5x above, and is
the same factor series the yield model uses.)

kW per kWp = POA W/m2 / 1000 (STC ratio) x inverter efficiency x (1 - system
losses) -- the same derates as facet_yield, so a building's curve scaled by
its placed kWp integrates to roughly its reported annual kWh.

Usage: python src/build_seasonal_curves.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.solar_model import SolarModel, MONTH_NAMES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SLOPE_STEP = 10   # coarser than the yield lookup on purpose: curve SHAPE
ASPECT_STEP = 30  # varies slowly with orientation, and this keeps the JSON small
SEASONS = {"summer": (12, 1, 2), "autumn": (3, 4, 5), "winter": (6, 7, 8), "spring": (9, 10, 11)}


def main():
    model = SolarModel()  # pilot location; calibrated factors + terrain horizon come along
    location = pvlib.location.Location(model.lat, model.lon, tz="Pacific/Auckland", altitude=310)
    times = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="1h", tz="Pacific/Auckland")
    clearsky = location.get_clearsky(times, model="ineichen")
    solpos = location.get_solarposition(times)

    # Open horizon on purpose -- see the module comment. Terrain belongs to
    # tshade, per building, and applying it here as well double-counted it.
    dni_cs, ghi_cs, dhi_cs = clearsky["dni"], clearsky["ghi"], clearsky["dhi"]

    month_abbr = dict(enumerate(MONTH_NAMES, start=1))
    factor = pd.Series(times.month.map(lambda m: model.monthly_factor[month_abbr[m]]), index=times)

    # Same physics as build_poa_lookup_table, so the chart and the kWh figure
    # above it cannot disagree: the cloud factor scales GHI, and Erbs splits
    # that into beam and diffuse. Scaling clear-sky POA directly (what this did)
    # preserves the clear-sky beam:diffuse ratio, which PVGIS showed costs up to
    # 18 percentage points on a south face. Transposition is Perez rather than
    # the isotropic default for the same reason.
    ghi_avg = ghi_cs * factor
    _split = pvlib.irradiance.erbs(ghi_avg, solpos["apparent_zenith"], times)
    dni_avg = _split["dni"].fillna(0.0)
    dhi_avg = _split["dhi"].fillna(0.0)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    airmass = location.get_airmass(times, solar_position=solpos)["airmass_relative"]
    _perez = dict(dni_extra=dni_extra, airmass=airmass, model="perez")

    a = config.PV_ASSUMPTIONS
    derate = (a["inverter_efficiency_pct"] / 100) * (1 - a["system_derate_pct"] / 100)

    curves = {}
    for slope in range(0, config.MAX_ROOF_SLOPE_DEG + SLOPE_STEP, SLOPE_STEP):
        for aspect in range(0, 360, ASPECT_STEP):
            poa_cs = pvlib.irradiance.get_total_irradiance(
                surface_tilt=slope, surface_azimuth=aspect,
                dni=dni_cs, ghi=ghi_cs, dhi=dhi_cs,
                solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"],
                **_perez,
            )["poa_global"].clip(lower=0)
            # ...and the same POA with the DIRECT beam removed: what the roof
            # still receives from sky diffuse + ground reflection when terrain
            # blocks the sun. Only the clear-day ("peak") curve gets this twin.
            # The frontend used one flat 0.18 multiplier for both lines, and on
            # a clear day that is far too generous -- Queenstown's winter 16:00
            # clear sky is DNI 699 W/m2 against DHI 14, so a north-facing 20
            # degree roof keeps 4.3%, not 18%.
            #
            # The season-MEAN curve deliberately does NOT get one. It is
            # clear-sky scaled by a monthly cloud factor, which preserves the
            # clear-sky beam:diffuse ratio -- wrong for a cloudy day, where
            # diffuse is most of the light and losing the beam to terrain costs
            # little. Applying the clear-sky fraction to a cloud-averaged mean
            # would understate typical winter afternoons. Decomposing the
            # cloudy fraction is not something monthly scalars can do, so the
            # frontend keeps a flat floor there and says so.
            poa_dif = pvlib.irradiance.get_total_irradiance(
                surface_tilt=slope, surface_azimuth=aspect,
                dni=dni_cs * 0, ghi=dhi_cs, dhi=dhi_cs,
                solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"],
                **_perez,
            )["poa_global"].clip(lower=0)

            poa_avg = pvlib.irradiance.get_total_irradiance(
                surface_tilt=slope, surface_azimuth=aspect,
                dni=dni_avg, ghi=ghi_avg, dhi=dhi_avg,
                solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"],
                **_perez,
            )["poa_global"].clip(lower=0)
            kw_avg = poa_avg / 1000 * derate   # kW per kWp
            kw_peak = poa_cs / 1000 * derate
            kw_peak_dif = poa_dif / 1000 * derate

            entry = {"avg": [], "peak": [], "peak_dif": []}
            for months in SEASONS.values():
                in_season = times.month.isin(months)
                def hourly_mean(series):
                    x = series[in_season]
                    return [round(v, 3) for v in x.groupby(x.index.hour).mean().reindex(range(24), fill_value=0)]
                entry["avg"].append(hourly_mean(kw_avg))
                sp = kw_peak[in_season]
                best_day = sp.resample("1D").sum().idxmax().date()
                # The diffuse twin must come from the SAME day, or the dotted
                # "sunny day" line would splice two different days together.
                for src, key in ((kw_peak, "peak"), (kw_peak_dif, "peak_dif")):
                    d = src[in_season]
                    d = d[d.index.date == best_day]
                    entry[key].append([round(v, 3) for v in d.groupby(d.index.hour).mean().reindex(range(24), fill_value=0)])
            curves[f"{slope}_{aspect}"] = entry

    out = {"slope_step": SLOPE_STEP, "aspect_step": ASPECT_STEP,
           "max_slope": config.MAX_ROOF_SLOPE_DEG,
           "seasons": list(SEASONS), "curves": curves}
    path = DATA_DIR / "seasonal_curves.json"
    path.write_text(json.dumps(out))
    print(f"Saved {path} ({path.stat().st_size / 1e3:.0f}KB, {len(curves)} orientation bins)")


if __name__ == "__main__":
    main()

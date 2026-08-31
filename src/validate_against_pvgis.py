"""
Cross-check our irradiance model against PVGIS, an independent public tool.

Every internal check this project has -- the truth scorecard, the layout
audits, compare_builds -- measures us against ourselves. None of them can see
a systematic bias in the irradiance model: if the whole POA table were 10%
high, every internal number would agree with every other internal number and
the map would be confidently wrong. Josh asked for an outside opinion.

PVGIS (European Commission JRC) is the reference free tool: global coverage
including New Zealand via ERA5 reanalysis, no API key, and -- unusually -- it
reports the plane-of-array IRRADIATION separately from the modelled PV output.
That separation is what makes it useful here, because in-plane irradiation is
exactly what our lookup table computes, while PV output additionally folds in
cell-temperature and spectral modelling that we deliberately approximate with
a flat derate. Comparing irradiation isolates our actual claim; comparing
energy conflates it with PV physics we never attempted.

What is compared, per site and per (tilt, aspect):

  H(i)_y  PVGIS annual in-plane irradiation, kWh/m2/yr
          vs  SolarModel.annual_poa_kwh_per_m2(slope, aspect)      <- the real test

  E_y     PVGIS annual AC yield per kWp
          vs  our POA x inverter efficiency x (1 - system derate)  <- indicative only

PVGIS is asked twice, with its DEM horizon off and on. Ours always carries the
area terrain horizon, so the horizon-on column is the like-for-like one; the
difference between the two PVGIS columns says how much of any gap is terrain
rather than climate.

Caveats worth keeping in view before believing a delta:
  - ERA5 is a ~30 km reanalysis. In alpine terrain like Queenstown a 30 km cell
    is a poor description of a valley floor; disagreement there is not
    automatically our error.
  - PVGIS averages 2005-2020. Our monthly factors come from NIWA/SolarView
    measurements or NASA POWER over different windows.
  - Their DEM horizon and ours are different products at different resolutions.

Usage:
    python src/validate_against_pvgis.py                 # default sites/angles
    python src/validate_against_pvgis.py --sites pilot island_bay
    python src/validate_against_pvgis.py --refresh       # ignore the cache
"""

import argparse
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.solar_model import SolarModel

CACHE = Path(__file__).resolve().parent.parent / "data" / "pvgis_cache.json"
PVGIS = "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc"
TIMEOUT_S = 90
PAUSE_S = 1.0          # be a polite client of a free public service

# Angles worth checking rather than a dense sweep: flat, the common residential
# pitches, and the four cardinal aspects at a typical pitch.
DEFAULT_ANGLES = [
    (0, 0), (10, 0), (20, 0), (30, 0),          # aspect irrelevant when flat-ish
    (20, 0), (20, 90), (20, 180), (20, 270),    # N / E / S / W at 20 deg
    (35, 0), (35, 180),
]


def _load_cache():
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(c):
    CACHE.write_text(json.dumps(c, indent=1, sort_keys=True))


def our_aspect_to_pvgis(aspect_deg):
    """Ours: compass bearing the roof faces, 0 = north, 90 = east.
    PVGIS: 0 = SOUTH, -90 = east, +90 = west, 180 = north (both hemispheres)."""
    a = (aspect_deg - 180.0) % 360.0
    return a - 360.0 if a > 180.0 else a


def query_pvgis(lat, lon, tilt, aspect_deg, loss_pct, use_horizon, cache, refresh=False):
    key = f"{lat:.4f},{lon:.4f},{tilt},{aspect_deg},{loss_pct},{int(use_horizon)}"
    if not refresh and key in cache:
        return cache[key]
    params = {
        "lat": f"{lat:.5f}", "lon": f"{lon:.5f}",
        "peakpower": 1, "loss": loss_pct,
        "angle": tilt, "aspect": our_aspect_to_pvgis(aspect_deg),
        "outputformat": "json", "pvtechchoice": "crystSi",
        "mountingplace": "building", "raddatabase": "PVGIS-ERA5",
        "usehorizon": 1 if use_horizon else 0,
    }
    url = PVGIS + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
        d = json.loads(r.read().decode())
    t = d["outputs"]["totals"]["fixed"]
    out = {"E_y": t["E_y"], "H_y": t["H(i)_y"], "elevation": d["inputs"]["location"]["elevation"]}
    cache[key] = out
    time.sleep(PAUSE_S)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    from src.region_build import area_centroid_wgs84, all_areas
    site_names = a.sites or [n for n in (["pilot"] + list(config.REGIONS))][:1] or ["pilot"]

    pv = config.PV_ASSUMPTIONS
    dc_to_ac = (pv["inverter_efficiency_pct"] / 100.0) * (1 - pv["system_derate_pct"] / 100.0)
    # PVGIS's single "loss" number is the closest thing it has to our two-stage
    # derate; matching them keeps the SECONDARY comparison honest even though
    # PVGIS also models cell temperature and we do not.
    loss_pct = round((1 - dc_to_ac) * 100, 1)

    cache = _load_cache()
    print(f"PVGIS cross-check  |  our derate {dc_to_ac:.3f} -> PVGIS loss {loss_pct}%")
    print(f"{'site':<12}{'tilt':>5}{'aspect':>7}"
          f"{'PVGIS H(i)':>12}{'ours POA':>10}{'delta':>8}"
          f"{'  |':>3}{'PVGIS no-horizon':>18}{'terrain effect':>16}")
    print("-" * 100)

    rows = []
    for site in site_names:
        c = area_centroid_wgs84(site)
        if c is None:
            # 'pilot' returns None by convention -- SolarModel defaults to it.
            from src.solar_model import pilot_location
            c = pilot_location()
        lat, lon = c
        model = SolarModel(lat, lon)
        seen = set()
        for tilt, aspect in DEFAULT_ANGLES:
            if (tilt, aspect) in seen:
                continue
            seen.add((tilt, aspect))
            try:
                on = query_pvgis(lat, lon, tilt, aspect, loss_pct, True, cache, a.refresh)
                off = query_pvgis(lat, lon, tilt, aspect, loss_pct, False, cache, a.refresh)
            except Exception as e:
                print(f"{site:<12}{tilt:>5}{aspect:>7}   PVGIS query failed: {type(e).__name__}: {e}")
                continue
            ours = model.annual_poa_kwh_per_m2(tilt, aspect)
            delta = 100.0 * (ours - on["H_y"]) / on["H_y"] if on["H_y"] else float("nan")
            terrain = 100.0 * (on["H_y"] - off["H_y"]) / off["H_y"] if off["H_y"] else 0.0
            rows.append((site, tilt, aspect, on["H_y"], ours, delta, terrain))
            print(f"{site:<12}{tilt:>5}{aspect:>7}{on['H_y']:>12.0f}{ours:>10.0f}"
                  f"{delta:>7.1f}%{'  |':>3}{off['H_y']:>18.0f}{terrain:>15.1f}%")
        _save_cache(cache)

    if rows:
        deltas = [r[5] for r in rows if r[5] == r[5]]
        mean = sum(deltas) / len(deltas)
        worst = max(deltas, key=abs)
        print("-" * 100)
        print(f"in-plane irradiation vs PVGIS: mean {mean:+.1f}%, worst {worst:+.1f}%, "
              f"n={len(deltas)}")
        print("A consistent sign across angles is a MODEL bias; scatter that changes "
              "sign with aspect is geometry or terrain, not calibration.")
    _save_cache(cache)


if __name__ == "__main__":
    main()

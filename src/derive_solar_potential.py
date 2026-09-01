"""Derive a region's solar_potential.geojson FROM its built panel layouts.

build_heatmap re-runs the full pipeline (segment, obstructions, panels) to
produce the building-summary layer -- hours of recomputation of numbers the
layout stage already wrote, and a standing source of disagreement between the
two layers (5 Beach St: 135 vs 203 panels, documented in preview.html). This
derives the same schema by aggregation in seconds, and the layers cannot
disagree because one IS the sum of the other.

Usage: python src/derive_solar_potential.py <region> [...]
"""
import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preflight import preflight
import geopandas as gpd
import pyproj
from shapely.geometry import shape
from shapely.ops import transform as _shtransform
import config
from src.region_build import area_paths

_TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True).transform


# Kept here rather than imported from build_layout_geojson so derive can run
# without pulling in the whole layout stack.
_REASON_TEXT = {
    "too_steep": f"Roof is steeper than we model "
                 f"(over {config.MAX_ROOF_SLOPE_DEG}\u00b0)",
    "no_lidar": "Not enough laser survey data over this roof",
    "no_surface": "No usable roof surface could be resolved",
    "low_confidence": "Roof shape could not be read confidently enough",
    "all_obstructed": "Roof is covered by vents, plant or other obstructions",
    "too_small": "No roof area large enough for a panel",
    "demolished": "Recorded as demolished or replaced since the survey",
    "timed_out": "Too complex to finish modelling",
    "failed": "Modelling failed on this building",
}


def _facet_area_m2(feature):
    """Plan area in m2 from the facet's own geometry. The layout emitter does
    not write an area_m2 property on facet features, so reading it out of
    properties silently produced 0 for every building -- which zeroed the heat
    map mode's whole estimate ladder (kWp = area x coverage x density)."""
    a = feature["properties"].get("area_m2")
    if a:
        return float(a)
    try:
        return _shtransform(_TO_NZTM, shape(feature["geometry"])).area
    except Exception:
        return 0.0


def derive(region):
    paths = area_paths(region)
    gdf = gpd.read_file(paths["outlines"]).set_index("building_id", drop=False)
    gdf_wgs = gdf.to_crs("EPSG:4326")
    d = json.loads(paths["panel_layouts"].read_text())
    agg = defaultdict(lambda: {"facet_count": 0, "obstruction_count": 0,
                               "panel_count": 0, "ac_kwh_year": 0.0,
                               "facet_area_m2": 0.0, "poa_w": 0.0,
                               "obstructed_m2": 0.0, "reason": None})
    for f in d["features"]:
        p = f["properties"]
        a = agg[p["building_id"]]
        k = p["kind"]
        if k == "facet":
            a["facet_count"] += 1
            area = _facet_area_m2(f)
            a["facet_area_m2"] += area
            a["poa_w"] += area * (p.get("poa_kwh_m2_yr") or 0.0)
        elif k == "obstruction":
            a["obstruction_count"] += 1
            a["obstructed_m2"] += _facet_area_m2(f)
        elif k == "no_estimate":
            # the layout stage could not model this roof and said why
            a["reason"] = p.get("no_estimate_reason")
        elif k == "panel":
            a["panel_count"] += 1
            a["ac_kwh_year"] += p.get("ac_kwh_year") or 0.0

    panel_kw = config.PV_ASSUMPTIONS["panel_rated_power_w"] / 1000.0
    features = []
    # EVERY building in the outlines gets a feature, including ones we could
    # not model. Josh: "The buildings should stay. We should have a marking on
    # the building saying why it is not estimated." A building that is simply
    # absent is indistinguishable, to someone searching their address, from a
    # broken site -- and it is the one case where they most want an answer.
    for bid, row in gdf_wgs.iterrows():
        a = agg.get(bid)
        if a is None:
            a = {"facet_count": 0, "obstruction_count": 0, "panel_count": 0,
                 "ac_kwh_year": 0.0, "facet_area_m2": 0.0, "poa_w": 0.0,
                 "obstructed_m2": 0.0, "reason": "no_surface"}
        kwp = a["panel_count"] * panel_kw
        # Only buildings with NO panels carry a reason. Where the layout stage
        # did not leave one, infer the most defensible from what the aggregate
        # shows, rather than reporting a bare zero with no explanation.
        reason = None
        if a["panel_count"] == 0:
            reason = a["reason"]
            if reason is None:
                if a["facet_count"] == 0:
                    reason = "no_surface"
                elif a["facet_area_m2"] and \
                        a["obstructed_m2"] / a["facet_area_m2"] > 0.6:
                    reason = "all_obstructed"
                elif a["facet_area_m2"] < 6:
                    reason = "too_small"
                else:
                    reason = "low_confidence"
        features.append({
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "building_id": int(bid),
                "facet_count": a["facet_count"],
                "obstruction_count": a["obstruction_count"],
                "panel_count": a["panel_count"],
                "kwp": round(kwp, 2),
                "ac_kwh_day_avg": round(a["ac_kwh_year"] / 365.0, 1),
                "ac_kwh_year": round(a["ac_kwh_year"], 0),
                "facet_area_m2": round(a["facet_area_m2"], 1),
                "avg_poa_kwh_m2": round(a["poa_w"] / a["facet_area_m2"], 0)
                                  if a["facet_area_m2"] > 0 else 0,
                **({"no_estimate_reason": reason,
                    "no_estimate_text": _REASON_TEXT.get(reason, reason)}
                   if reason else {}),
            },
        })
    out = {"type": "FeatureCollection", "assumptions": config.PV_ASSUMPTIONS,
           "features": features}
    paths["solar_potential"].write_text(json.dumps(out))
    no_est = [f for f in features if f["properties"].get("no_estimate_reason")]
    print(f"{region}: {len(features)} buildings, "
          f"{sum(1 for f in features if f['properties']['panel_count'])} with panels, "
          f"{len(no_est)} with no estimate")
    if no_est:
        from collections import Counter
        for r, c in Counter(f["properties"]["no_estimate_reason"]
                            for f in no_est).most_common():
            print(f"    {c:>5}  {r}")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        preflight("derive_solar_potential", r)
        derive(r)

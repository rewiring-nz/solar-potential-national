"""
Spatial-join street addresses onto the buildings in data/solar_potential.geojson,
from the LINZ NZ Addresses point layer (123113), so the frontend can title a
building's panel with its address instead of an opaque outline id.

Join rule: every address point is assigned to the building footprint that
contains it; buildings containing no address point take the nearest address
within NEAREST_MAX_M (an address point usually sits on the dwelling, but
garages/outbuildings/odd digitisation put some just outside). Buildings with
several address points (units, flats) take the lowest street number and note
the count. No address within range -> no "address" property; the frontend
falls back to the outline id.

Patches solar_potential.geojson in place -- re-run after any rebuild of that
file (build_heatmap.py doesn't know about addresses).

Usage: python src/add_addresses.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy.spatial import cKDTree
from shapely.geometry import shape
import shapely
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preflight import preflight
import config
from src.fetch_data import fetch_building_outlines

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NZ_ADDRESSES_LAYER = 123113
NEAREST_MAX_M = 40.0  # beyond this, an address point is someone else's


def fetch_addresses(bbox_nztm, api_key):
    """The WFS helper is layer-agnostic despite its name -- reuse it."""
    return fetch_building_outlines(bbox_nztm, api_key, layer_id=NZ_ADDRESSES_LAYER)


def main(area="pilot"):
    preflight("add_addresses", area)
    from src.region_build import area_paths, write_json_atomic
    load_dotenv()
    api_key = os.environ["LINZ_API_KEY"]

    print(f"Fetching NZ Addresses for {area} bbox...")
    # WGS84 bbox, not NZTM: unlike the outlines layer, this layer's WFS default
    # SRS is lon/lat -- an NZTM bbox silently matches zero features.
    bbox = config.PILOT_BBOX if area == "pilot" else config.REGIONS[area]
    addr = fetch_addresses(bbox, api_key)
    pts, labels = [], []
    for f in addr["features"]:
        g = f.get("geometry")
        if not g or g["type"] != "Point":
            continue
        props = f["properties"]
        label = props.get("full_address_ascii") or props.get("full_address")
        if not label:
            continue
        # Drop the ", Suburb, Town" tail for panel-title brevity; keep number + street.
        short = label.split(",")[0].strip()
        pts.append(g["coordinates"][:2])
        labels.append(short)
    pts = np.array(pts)
    print(f"{len(pts)} address points")

    sp_path = area_paths(area)["solar_potential"]
    sp = json.loads(sp_path.read_text())

    # Address coordinates come back in the WFS layer's CRS -- this layer serves
    # NZGD2000 lon/lat, and solar_potential geometries are WGS84 lon/lat too
    # (equivalent at this precision), so the join runs directly in degrees with
    # a metre-scaled KD-tree (lon compressed by cos(lat)).
    lat0 = np.radians(-45.03)
    scale = np.array([np.cos(lat0) * 111320.0, 111132.0])
    tree = cKDTree(pts * scale)

    n_contained = n_nearest = n_none = 0
    for feat in sp["features"]:
        geom = shape(feat["geometry"])
        inside = shapely.contains_xy(geom, pts[:, 0], pts[:, 1])
        idx = np.where(inside)[0]
        if len(idx) == 0:
            c = geom.centroid
            dist, i = tree.query(np.array([c.x, c.y]) * scale)
            if dist <= NEAREST_MAX_M:
                idx = np.array([i])
                n_nearest += 1
            else:
                n_none += 1
                continue
        else:
            n_contained += 1
        chosen = sorted(labels[i] for i in idx)[0]
        feat["properties"]["address"] = chosen
        if len(idx) > 1:
            feat["properties"]["address_count"] = int(len(idx))

    write_json_atomic(sp_path, sp)
    print(f"contained={n_contained} nearest={n_nearest} unmatched={n_none}")
    print(f"Patched {sp_path}")


if __name__ == "__main__":
    from src.region_build import areas_from_argv
    import sys
    for _area in areas_from_argv(sys.argv):
        main(_area)

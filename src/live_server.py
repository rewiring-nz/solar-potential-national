"""
Local server for the pilot: serves preview.html and data/ as static
files (same as `python3 -m http.server`), plus one API endpoint,
/api/refit, that recomputes segmentation/obstruction-detection/panel-
fitting for a single building on demand with overridable parameters --
so the sliders in preview.html can show what a parameter change actually
does to a real building's layout, live, instead of guessing and waiting
for a full ~90s pilot-wide rebuild.

All the heavy inputs (building outlines, DSM, imagery, the pvlib/NASA
POWER yield model) are loaded once at startup and reused across requests
-- a single-building refit is cheap (RANSAC + obstruction detection for
one footprint, tens of milliseconds), so this stays responsive as you
drag a slider.

Usage: python src/live_server.py [port]   (default port 8000)
"""

import http.server
import json
import sys
import urllib.parse
import warnings
from pathlib import Path

import geopandas as gpd
import pyproj
import rasterio
from shapely.ops import transform as shapely_transform

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.roof_segmentation import segment_building_best
from src.pointcloud_source import PointCloudSource
from src.obstruction_detection import detect_obstructions_combined
from src.panel_fitting import fit_panels_on_facet, apply_panel_density, drop_minor_arrays, assign_fill_ranks
from src.solar_model import SolarModel
from src.building_shading import building_shading_factor

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
# Serve from the repo root (one level up), not the project dir, so the URL
# stays http://localhost:8000/solar-map/preview.html -- consistent with
# every link already shared for this pilot rather than dropping the prefix.
SERVE_ROOT = PROJECT_DIR.parent

print("Loading shared data (buildings, DSM, imagery, solar model)...")
GDF = gpd.read_file(DATA_DIR / "building_outlines.geojson").set_index("building_id", drop=False)
DSM_DS = rasterio.open(DATA_DIR / "dsm_mosaic.tif")
IMAGERY_DS = rasterio.open(DATA_DIR / "imagery_mosaic.tif")
PC_SOURCE = PointCloudSource()
MODEL = SolarModel()
DSM_BAND = DSM_DS.read(1)  # loaded once, reused for every building's own near-field shading scan
TO_WGS84 = pyproj.Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True).transform
print("Ready.")


def refit_building(building_id, setback, ransac_threshold, z_threshold, density_pct=100):
    row = GDF.loc[building_id]
    # imagery_ds MUST be passed. This file opens IMAGERY_DS at startup and for a
    # long time did not hand it to the segmenter, so the live server silently ran
    # a different roof model from the build: no skylight detection and no
    # imagery-informed facets. That is the same defect that made anderson2.py
    # render pictures of a pipeline that was never shipped, and it is the sixth
    # time a tool here has diverged from the build it was meant to inspect.
    facets = segment_building_best(DSM_DS, PC_SOURCE, row.geometry, building_id,
                                    ransac_distance_threshold=ransac_threshold,
                                    imagery_ds=IMAGERY_DS)

    features = []
    facet_panels = []  # one list per facet, same order as `facets` -- filled in below, filtered after
    facet_shading = []  # same order as `facets` -- facet_yield below needs each facet's own factor
    for f in facets:
        facet_centroid = f["geometry"].centroid
        shading_factor = building_shading_factor(DSM_BAND, DSM_DS.transform, DSM_DS.nodata,
                                                   facet_centroid.x, facet_centroid.y, MODEL.hourly,
                                                   own_geom=f["geometry"], terrain_horizon_profile=MODEL.horizon_profile)
        facet_shading.append(shading_factor)
        plane = (f["plane_a"], f["plane_b"], f["plane_c"])
        obstructions = detect_obstructions_combined(IMAGERY_DS, PC_SOURCE, f["geometry"], plane,
                                                     z_threshold=z_threshold, boundary_erode_m=setback)
        siblings = [other for other in facets if other is not f]
        panels = fit_panels_on_facet(f, setback=setback, obstructions=obstructions, sibling_facets=siblings)
        poa = MODEL.annual_poa_kwh_per_m2(f["slope_deg"], f["aspect_deg"]) * shading_factor
        for p in panels:
            p["poa_kwh_m2_yr"] = poa
        facet_panels.append(panels)

        features.append({
            "type": "Feature",
            "geometry": shapely_transform(TO_WGS84, f["geometry"]).__geo_interface__,
            "properties": {"kind": "facet", "slope_deg": round(f["slope_deg"], 1),
                            "aspect_deg": round(f["aspect_deg"], 1), "poa_kwh_m2_yr": round(poa, 0)},
        })
        for o in obstructions:
            features.append({
                "type": "Feature",
                "geometry": shapely_transform(TO_WGS84, o).__geo_interface__,
                "properties": {"kind": "obstruction"},
            })

    # Same building-level post-processing as build_layout_geojson, so live-tuned
    # results can't drift from the static data's rules: straggler groups
    # dropped, then fill ranks assigned across the whole building
    # (sunniest-facet-first), then the density cut.
    facet_panels = drop_minor_arrays(facet_panels)
    all_panels = [p for panels in facet_panels for p in panels]
    assign_fill_ranks(all_panels)
    kept_panels = set(id(p) for p in apply_panel_density(all_panels, density_pct))

    total_panels = total_kwp = total_ac_kwh_year = 0
    for f, panels, shading_factor in zip(facets, facet_panels, facet_shading):
        kept = [p for p in panels if id(p) in kept_panels]
        if not kept:
            continue
        y = MODEL.facet_yield(f, len(kept), shading_factor=shading_factor)
        total_panels += len(kept)
        total_kwp += y["kwp"]
        total_ac_kwh_year += y["ac_kwh_year"]
        for p in kept:
            features.append({
                "type": "Feature",
                "geometry": shapely_transform(TO_WGS84, p["geometry"]).__geo_interface__,
                "properties": {"kind": "panel", "fill_rank": p["fill_rank"]},
            })

    return {
        "type": "FeatureCollection",
        "features": features,
        "summary": {
            "building_id": building_id,
            "facet_count": len(facets),
            "panel_count": total_panels,
            "kwp": round(total_kwp, 2),
            "ac_kwh_year": round(total_ac_kwh_year, 0),
        },
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SERVE_ROOT), **kwargs)

    def do_GET(self):
        if self.path.startswith("/api/refit"):
            self.handle_refit()
        elif "Range" in self.headers:
            self.handle_range()  # PMTiles fetches tile byte-ranges; stdlib handler lacks 206 support
        else:
            super().do_GET()

    def handle_range(self):
        import os
        path = self.translate_path(self.path.split("?")[0])
        if not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        m = self.headers["Range"].replace("bytes=", "").split("-")
        start = int(m[0]) if m[0] else 0
        end = int(m[1]) if len(m) > 1 and m[1] else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            self.send_error(416)
            return
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            self.wfile.write(f.read(end - start + 1))

    def handle_refit(self):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        try:
            building_id = int(params["building_id"][0])
            setback = float(params.get("setback", [config.PANEL_EDGE_SETBACK_M])[0])
            ransac_threshold = float(params.get("ransac_threshold", [0.15])[0])
            z_threshold = float(params.get("z_threshold", [2.75])[0])
            density_pct = float(params.get("density_pct", [100])[0])
            result = refit_building(building_id, setback, ransac_threshold, z_threshold, density_pct)
            body = json.dumps(result).encode()
            status = 200
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode()
            status = 400

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if "/api/refit" in (self.path or ""):
            print(f"{self.address_string()} - {fmt % args}")
        # stay quiet on ordinary static-file requests, same as before


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = http.server.ThreadingHTTPServer(("", port), Handler)
    print(f"Serving {SERVE_ROOT} on http://localhost:{port} (with /api/refit)")
    server.serve_forever()


if __name__ == "__main__":
    main()

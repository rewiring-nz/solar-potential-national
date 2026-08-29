"""
Pull building outlines (WFS) and the DSM raster (Exports API) for
config.PILOT_BBOX from the LINZ Data Service, and save both under data/.

Requires a LINZ_API_KEY with REST API scope enabled (Account -> API keys
-> edit the key -> enable "Search and Download"), not just the default
OGC web-services scope -- the export job creation 401s otherwise.

Usage: python src/fetch_data.py
"""

import json
import os
import sys
import time
import zipfile
from pathlib import Path

import rasterio
import requests
from dotenv import load_dotenv
from rasterio.merge import merge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_building_outlines(bbox_nztm2000, api_key, layer_id=config.LINZ_BUILDING_OUTLINES_LAYER):
    """
    bbox_nztm2000: [minx, miny, maxx, maxy] in EPSG:2193 metres.

    Uses WFS 1.0.0 (not 2.0.0) -- see README "Status" for why: 2.0.0's
    typeNames/count params silently returned 0 features for every bbox
    tried here, with no error, while 1.0.0's typeName/maxFeatures works.
    """
    type_name = f"data.linz.govt.nz:layer-{layer_id}"
    bbox_str = ",".join(str(v) for v in bbox_nztm2000)
    url = "https://data.linz.govt.nz/services"
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": type_name,
        "outputFormat": "json",
        "bbox": bbox_str,
        "maxFeatures": 10000,
    }
    resp = requests.get(f"{url};key={api_key}/wfs", params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "features" not in data:
        raise RuntimeError(f"Unexpected WFS response: {data}")
    return data


def fetch_raster(bbox_wgs84, api_key, layer_id, name, out_dir=DATA_DIR, format_key="grid"):
    """
    bbox_wgs84: [min_lon, min_lat, max_lon, max_lat]. Creates a Koordinates
    export job cropped to this extent, polls until complete, downloads and
    unzips it, then mosaics the (possibly several) 1km survey tiles into a
    single GeoTIFF at out_dir/{name}_mosaic.tif. Works for any LDS raster
    layer -- DSM/DEM are Koordinates data_type "Grid" (format_key="grid"),
    RGB aerial imagery is data_type "Raster" (format_key="raster"); the
    Exports API 400s if you send the wrong one for the layer's type.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    body = {
        "crs": "EPSG:2193",
        "formats": {format_key: "image/tiff;subtype=geotiff"},
        "items": [{"item": f"https://data.linz.govt.nz/services/api/v1/layers/{layer_id}/"}],
        "extent": {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat], [min_lon, max_lat],
                [max_lon, max_lat], [max_lon, min_lat], [min_lon, min_lat],
            ]],
        },
    }
    headers = {"Authorization": f"key {api_key}", "Content-Type": "application/json"}

    resp = requests.post("https://data.linz.govt.nz/services/api/v1/exports/", headers=headers, json=body, timeout=30)
    if resp.status_code == 401:
        raise SystemExit(
            "401 from Exports API -- LINZ_API_KEY needs REST API scope "
            "(Account -> API keys -> edit -> enable Search and Download), "
            "not just the default OGC web-services scope."
        )
    if resp.status_code >= 400:
        # LINZ returns a JSON body naming the exact problem (bad extent, wrong
        # format for the layer type, area outside coverage) -- surface it
        # instead of a bare 400.
        raise RuntimeError(f"Exports API {resp.status_code} for layer {layer_id} "
                            f"({name}): {resp.text[:300]}")
    job = resp.json()
    job_url = job["url"]
    print(f"Export job {job['id']} created, polling...")

    while job["state"] == "processing":
        time.sleep(5)
        job = requests.get(job_url, headers=headers, timeout=30).json()
        print(f"  state={job['state']} progress={job.get('progress')}")

    if job["state"] != "complete":
        raise RuntimeError(f"Export job did not complete: {job}")

    zip_path = out_dir / f"{name}_export.zip"
    with requests.get(job["download_url"], headers=headers, timeout=300, stream=True) as dl:
        dl.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)  # streamed -- region-scale imagery zips run to GBs, far past
                # what buffering the whole response in memory (the original approach) should carry

    extract_dir = out_dir / name
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    tifs = sorted(extract_dir.glob("*.tif"))
    srcs = [rasterio.open(f) for f in tifs]
    mosaic, transform = merge(srcs)
    profile = srcs[0].profile
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
    mosaic_path = out_dir / f"{name}_mosaic.tif"
    with rasterio.open(mosaic_path, "w", **profile) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()

    print(f"Mosaicked {len(tifs)} tiles -> {mosaic_path}")
    return mosaic_path


def main():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = os.environ.get("LINZ_API_KEY")
    if not api_key:
        raise SystemExit("LINZ_API_KEY not set -- copy .env.example to .env and fill it in")
    if config.PILOT_BBOX is None:
        raise SystemExit("config.PILOT_BBOX is not set")

    DATA_DIR.mkdir(exist_ok=True)

    print(f"Fetching building outlines for bbox {config.PILOT_BBOX_NZTM2000} (NZTM2000)...")
    buildings = fetch_building_outlines(config.PILOT_BBOX_NZTM2000, api_key)
    out_path = DATA_DIR / "building_outlines.geojson"
    out_path.write_text(json.dumps(buildings))
    print(f"Saved {len(buildings['features'])} buildings to {out_path}")

    print(f"Fetching DSM for bbox {config.PILOT_BBOX} (WGS84)...")
    fetch_raster(config.PILOT_BBOX, api_key, config.LINZ_DSM_LAYER, "dsm")

    print(f"Fetching aerial imagery for bbox {config.PILOT_BBOX} (WGS84)...")
    fetch_raster(config.PILOT_BBOX, api_key, config.LINZ_IMAGERY_LAYER, "imagery", format_key="raster")


if __name__ == "__main__":
    main()

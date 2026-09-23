# NZ Rooftop Solar Potential Map

Per-building rooftop solar potential for New Zealand, from LINZ building
outlines, LiDAR and aerial imagery: how much sun each roof face gets, how many
panels physically fit, what they would generate, and what that is worth.

This repository is the Wellington / national deployment (Island Bay). The
Queenstown Lakes deployment lives in the sibling `solar-map` repository, which
carries the shared code's backlog and tests; the two are kept in sync by hand
(`tools/check_repo_sync.py` there).

**Verify it yourself:** [docs/quickstart.md](docs/quickstart.md) runs the
identical methodology on any small NZ area you choose, in minutes, with a
per-building report to check against the aerial photo.
[method.html](https://rewiring-nz.github.io/nz-solar-potential/method.html)
explains every decision the pipeline makes in plain English.

## Data sources (verified August 2026)

| Source | What | License | Access |
|---|---|---|---|
| [LINZ LiDAR](https://www.linz.govt.nz/products-services/data/types-linz-data/elevation-data/lidar-data-coverage) | Point clouds (LAZ) + DSM/DEM GeoTIFFs, 20cm–1m res depending on region | CC-BY-4.0, free | [data.linz.govt.nz](https://data.linz.govt.nz/), needs free account + API key |
| [LINZ NZ Building Outlines](https://data.linz.govt.nz/layer/101290-nz-building-outlines/) | Building footprint polygons | CC-BY-4.0, free | Same LDS API key; **check coverage for the chosen pilot area first** — some regions (e.g. Bay of Plenty, Gisborne) weren't listed as covered as of this writing |
| [NASA POWER](https://power.larc.nasa.gov/docs/tutorials/service-data-request/api/) | Global solar irradiance, cloud-adjusted | Open, no key needed | REST API, but coarse (~1°×1° grid, ~100km) — use only as a broad calibration/sanity-check input, not the primary geometry-level model |
| [NIWA SolarView](https://niwa.co.nz/renewable-energy/solarview) | Per-address calculator using nearest climate station | **Results are non-commercial-use only per NIWA's EULA** | No bulk API — don't build on top of its output directly for a public tool; treat its methodology as a reference, not a data source |

**Irradiance approach:** rather than depending on NIWA's restricted
calculator, compute irradiance from first principles with
[pvlib](https://pvlib-python.readthedocs.io/) (sun position + clear-sky
model per roof facet's slope/aspect/lat), then bias-correct with NASA
POWER's actual-cloud-adjusted monthly averages for the pilot region. This
keeps the pipeline fully reproducible and licence-clean, and is the same
approach used in the published LiDAR-solar-potential literature (GRASS
`r.sun` / PVGIS-style models).

## How it works

1. **Fetch** -- building outlines, LiDAR DSM and point cloud, aerial imagery
   for a region (`src/fetch_regions.py`, `src/fetch_pointcloud_regions.py`).
2. **Roof geometry** -- each footprint is cut into planar faces
   (`src/roof_partition.py`, `src/roof_segmentation.py`). Hand-drawn markup
   (`mark_roofs.html`, `data/roof_labels.json`) always wins where it exists;
   otherwise a selected reading (SAM or line network) and the LiDAR
   partition compete. Shared ridges are snapped to the crest the LiDAR shows
   (`src/ridge_snap.py`); obstructions are detected from imagery and LiDAR.
3. **Panel layout** -- panels are racked on one grid frame per building
   (`src/panel_fitting.py`), clear of edges, ridges, obstructions and drawn
   fold lines; ordered sunniest-first so any density is the best subset.
4. **Yield** -- per-panel irradiance from slope, aspect, latitude and each
   building's own terrain horizon (`src/solar_model.py`,
   `src/building_horizon.py`), calibrated against measured data.
5. **Economics** -- cost, savings, payback and per-panel break-even, computed
   in the browser from `economics.js` with editable assumptions.
6. **Publish** -- each region emits vector tiles and summaries; a combine
   step joins them (`src/emit_region.py`, `src/combine_regions.py`) and
   `tools/deploy_from_vm.sh` gates the result against the live site before
   pushing.

Builds run on a cloud VM, region by region (`src/run_district_build.sh`;
architecture in [docs/scale-architecture.md](docs/scale-architecture.md)).
Nothing at district scale is computed on a laptop.

## Checks

```bash
bash tests/run_all.sh --fast    # unit, economics, deprecation, repo-sync and diagram checks
python tools/predeploy_check.py # what a new build changes against the live one
python tools/bench.py           # geometry against the hand-drawn markup
python tools/cases.py check     # every flagged roof, fixed or not
```

## Documentation

[docs/README.md](docs/README.md) maps the documentation: users, data
maintainers, developers, reviewers, the economics model and the theory.
[BACKLOG.md](BACKLOG.md) is the Wellington-specific open work.

## Local setup

```bash
cd solar-wellington
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To run the map (from the repo root, one level up from `solar-wellington/`), use
`live_server.py` rather than a plain static server -- it serves
`preview.html` and `data/` exactly the same way, plus the `/api/refit`
endpoint the tuning sliders need:

```bash
source solar-wellington/.venv/bin/activate
python solar-wellington/src/live_server.py 8000
```

Then open `http://localhost:8000/solar-wellington/preview.html`. A plain
`python3 -m http.server` still works for everything except the sliders
(they'll show "Refit failed -- is live_server.py running?").

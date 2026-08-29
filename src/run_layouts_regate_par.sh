#!/bin/bash
# Parallel layouts-only re-run. Same steps as run_layouts_regate.sh, but the
# per-area work runs several areas at a time.
#
# The areas are independent until merge_regions -- each reads its own region
# files and writes its own outputs -- so the serial loop in the original was
# leaving 11 of 12 cores idle. gate_panels peaks around 250MB per area, so
# JOBS is bounded by cores, not memory; it defaults to half the cores to leave
# the machine usable while a rebuild runs.
#
# Everything after the fan-in (merge, deciles, shrink, tippecanoe) is a single
# pass over the combined file and stays serial.
#
# JOBS fans out across AREAS; BUILD_JOBS is the per-area worker count inside
# build_layout_geojson. They multiply, so rebuilding ONE area wants JOBS=1 and a
# high BUILD_JOBS, while rebuilding all 24 wants the reverse. Getting this wrong
# is not slow, it is fatal: each build worker caches its own decoded LiDAR tiles,
# and 11 of them was enough to have the run killed outright.
#
# Usage: JOBS=6 bash src/run_layouts_regate_par.sh area1 area2 ...
#        JOBS=1 BUILD_JOBS=6 bash src/run_layouts_regate_par.sh pilot
#
# Weight JOBS over BUILD_JOBS for a district run. Measured on 28 Aug: the build
# stage is fast and parallel -- 182 buildings in 45s, 484 in 224s on 3 workers,
# and the whole district is only 3.8 core-hours of per-building work -- while
# gate_panels and rerank are SINGLE-THREADED per area and take 9-11 minutes
# each. With JOBS=2 that put 2 cores of 12 on the slow stage and made a 40-minute
# job take seven hours.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
JOBS="${JOBS:-$(( $(sysctl -n hw.ncpu 2>/dev/null || nproc) / 2 ))}"
[ "$JOBS" -lt 1 ] && JOBS=1
echo "running $# areas, $JOBS at a time"

# One area's three steps, as a unit -- exported so xargs can call it.
run_area() {
  r="$1"
  log="data/build_logs/${r}_regate.log"
  {
    .venv/bin/python src/build_layout_geojson.py "$r" --jobs "${BUILD_JOBS:-6}" &&
    .venv/bin/python src/gate_panels.py "$r" &&
    .venv/bin/python src/rerank_layouts.py "$r"
  } >"$log" 2>&1
  if [ $? -ne 0 ]; then echo "FAILED: $r (see $log)"; return 1; fi
  echo "done: $r"
}
export -f run_area

# printf %s\\n, not a bare list: area names go through xargs one per line so a
# missing quote cannot glue 24 names into a single argument. That exact bug
# ("File name too long") has bitten this pipeline before under zsh.
printf '%s\n' "$@" | xargs -P "$JOBS" -I{} bash -c 'run_area "$@"' _ {}
if [ $? -ne 0 ]; then echo "Skipping merge: at least one area failed"; exit 1; fi

$PY src/merge_regions.py && $PY src/bake_density_deciles.py &&
$PY src/shrink_panels_for_tiles.py &&
tippecanoe -o data/panel_layouts.pmtiles --force -l layout -Z13 -z16 \
  --drop-densest-as-needed --detect-shared-borders \
  -y kind -y building_id -y fill_rank -y fill_order -y array_id -y array_size \
  -y ac_kwh_year -y slope_deg -y aspect_deg \
  -y poa_kwh_m2_yr -y panel_count data/panel_layouts.geojson &&
# merge_regions rewrites solar_potential.geojson, which DROPS the per-building
# terrain masks -- without this the seasonal curves silently lose valley
# shading, a bug that was already found and fixed once. Caught on the 26 Aug
# rebuild: 0 of 15,353 buildings had tshade afterwards.
$PY src/build_terrain_masks.py &&
echo REGATE_COMPLETE

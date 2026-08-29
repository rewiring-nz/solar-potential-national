#!/bin/bash
# Layouts-only re-run: fit + ratio-based gate + rerank per area, then merge
# and post-steps. Exists for gate-rule changes -- solar model, heat rasters,
# and addresses are gate-independent and skipped (~2h instead of ~3.5h).
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
fail=0
for r in "$@"; do
  log="data/build_logs/${r}_regate.log"
  {
    $PY src/build_layout_geojson.py "$r" &&
    $PY src/gate_panels.py "$r" &&
    $PY src/rerank_layouts.py "$r"
  } >"$log" 2>&1
  if [ $? -ne 0 ]; then echo "FAILED: $r (see $log)"; fail=1; else echo "done: $r"; fi
done
if [ $fail -ne 0 ]; then echo "Skipping merge: at least one area failed"; exit 1; fi
$PY src/merge_regions.py && $PY src/bake_density_deciles.py &&
$PY src/shrink_panels_for_tiles.py &&
tippecanoe -o data/panel_layouts.pmtiles --force -l layout -Z13 -z16 \
  --drop-densest-as-needed --detect-shared-borders \
  -y kind -y building_id -y fill_rank -y fill_order -y array_id -y array_size \
  -y ac_kwh_year -y slope_deg -y aspect_deg \
  -y poa_kwh_m2_yr -y panel_count data/panel_layouts.geojson &&
echo REGATE_COMPLETE

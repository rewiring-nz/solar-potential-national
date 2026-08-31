#!/bin/bash
# Island Bay FULL rebuild -- the ib_full2 chain, rerun through run_stage.
#
# Carries everything ib_full2 did, plus what the 31 Aug code review added:
#   - panel_fitting sync: Island Bay was MISSING the gap-fill pass and the
#     straggler yield exception. This is the first IB build to have them, so
#     expect more panels at 100% density and a different removal order as the
#     slider comes down. That is the point, not a regression.
#   - preflight on every stage: a missing input stops the run in a second
#     instead of producing quietly-worse output over an hour.
#   - resume: re-running this script skips stages already done. Safe to
#     interrupt; safe to re-run after a failure.
#   - shapely.contains_xy migration (behaviour-neutral, verified).
#
# Prerequisite: the VM's solar-wellington and solar-map trees must be synced to
# today's code first. sync_vm.sh does that. Running this against the old tree
# silently rebuilds Island Bay WITHOUT the panel_fitting fix, which is the whole
# reason for the rebuild.
set -u
cd "$HOME/solar-wellington"
PY=$HOME/solar-map/.venv/bin/python
export RAYON_NUM_THREADS=1
export SOLAR_LAZ_SINGLE=1
MARK="$HOME/ib_full3_status.txt"
echo "IBFULL3 START $(date -u +%H:%M:%S)" > "$MARK"

# Refuse to build without imagery. Without it, obstruction detection loses its
# colour evidence and roof_partition loses its image lines -- SILENTLY. This
# guard predates preflight and stays because it can also FETCH the missing
# input rather than only complaining about it.
IMG=data/regions/island_bay/imagery_mosaic.tif
if [ ! -f "$IMG" ]; then
  echo "fetching imagery $(date -u +%H:%M:%S)" >> "$MARK"
  $PY src/fetch_regions.py island_bay > "$HOME/ib_fetch3.log" 2>&1 || true
fi
if [ ! -f "$IMG" ]; then
  echo "IBFULL3 ABORTED: no imagery -- refusing to build LiDAR-only" >> "$MARK"
  exit 1
fi
echo "imagery present $(date -u +%H:%M:%S)" >> "$MARK"

# Check the tree is actually today's BEFORE spending the compute, and check
# every prerequisite rather than only the first -- a script that aborts at
# stage four with "no such file" wastes the hour that got it there.
if ! grep -q "gap_fill" src/panel_fitting.py; then
  echo "IBFULL3 ABORTED: panel_fitting has no gap-fill pass." >> "$MARK"
  echo "  The VM tree is STALE. Rebuilding now would reproduce the exact bug" >> "$MARK"
  echo "  this rebuild exists to fix. Run tools/sync_vm.sh from the Mac." >> "$MARK"
  exit 1
fi
for f in src/run_stage.py src/preflight.py; do
  if [ ! -f "$f" ]; then
    echo "IBFULL3 ABORTED: missing $f -- VM tree predates the 31 Aug review work." >> "$MARK"
    echo "  Run tools/sync_vm.sh from the Mac (it refuses while a build runs)." >> "$MARK"
    exit 1
  fi
done

for s in build_layout_geojson gate_panels rerank_layouts derive_solar_potential \
         patch_roof_confidence bake_building_horizons build_heatmap_raster; do
  $PY src/run_stage.py --skip-done "$s" island_bay >> "$HOME/ib_full3.log" 2>&1 \
    || { echo "IBFULL3 FAILED at $s $(date -u +%H:%M:%S)" >> "$MARK"; exit 1; }
  echo "  $s done $(date -u +%H:%M:%S)" >> "$MARK"
done

# Addresses need the network and are a patch-in-place post-process: a failure
# must not discard the offline compute around it.
$PY src/run_stage.py --skip-done add_addresses island_bay >> "$HOME/ib_full3.log" 2>&1 \
  || echo "  WARN addresses failed -- patch later" >> "$MARK"

echo "IBFULL3 REGION DONE $(date -u +%H:%M:%S)" >> "$MARK"

# Fan-in. merge_regions FIRST: terrain masks, seasonal curves and density
# deciles write ONLY into the merged file, which merge regenerates from the
# region files -- running them first silently discards their work. preflight
# now refuses to run them against a stale merge, but the order is correct here
# so it never has to.
for s in merge_regions bake_density_deciles build_terrain_masks \
         build_seasonal_curves shrink_panels_for_tiles; do
  $PY src/run_stage.py --force "$s" island_bay >> "$HOME/ib_full3.log" 2>&1 \
    || { echo "IBFULL3 FAN-IN FAILED at $s" >> "$MARK"; exit 1; }
done

tippecanoe -o data/panel_layouts.pmtiles --force -l layout \
  -Z13 -z16 --drop-densest-as-needed --detect-shared-borders \
  -y kind -y building_id -y fill_rank -y fill_order -y array_id -y array_size \
  -y ac_kwh_year -y slope_deg -y aspect_deg -y roof_confidence \
  -y poa_kwh_m2_yr -y panel_count data/panel_layouts.geojson \
  >> "$HOME/ib_full3.log" 2>&1 || { echo "IBFULL3 TILES FAILED" >> "$MARK"; exit 1; }

echo "IBFULL3 COMPLETE $(date -u +%H:%M:%S)" >> "$MARK"

# Report for Josh to read before anything is pushed.
{
  echo "=== Island Bay rebuild, $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "FIRST IB build carrying the gap-fill pass and the straggler yield fix."
  echo
  echo "=== diff vs previous build ==="
  $PY src/compare_builds.py 2>&1 || echo "(compare_builds unavailable)"
} > "$HOME/ib_diff3.txt" 2>&1
echo "REPORT ~/ib_diff3.txt" >> "$MARK"

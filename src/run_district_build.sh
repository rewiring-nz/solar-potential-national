#!/bin/bash
# Full district build -- resumable, in the correct stage order.
#
# Replaces the pattern of hand-written one-off scripts scp'd to the VM. Those
# had no memory: the 31 Aug Queenstown rebuild was launched three times and
# each launch redid every completed region, because nothing on disk recorded
# what had already finished.
#
# Every stage goes through src/run_stage.py, which preflights the stage's
# inputs, records a completion marker on success, and (with --skip-done) skips
# work whose marker is newer than all of its inputs. So:
#
#   ./src/run_district_build.sh                 # resume: skip what is done
#   ./src/run_district_build.sh --force         # rebuild everything
#   ./src/run_district_build.sh --regions "a b"  # just these regions
#
# Interrupting this and re-running it continues where it stopped.
#
# ORDER MATTERS, and one ordering rule is not obvious: build_terrain_masks and
# build_seasonal_curves write ONLY into the merged data/solar_potential.geojson,
# and merge_regions REGENERATES that file from the region files. Running them
# before the merge silently discards their work -- it cost a full rebuild once.
# They run after the merge here, and run_stage's preflight independently
# refuses to run them against a stale merge.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOGDIR=data/build_logs
mkdir -p "$LOGDIR"

SKIP="--skip-done"
REGIONS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --force)   SKIP=""; shift ;;
    --regions) REGIONS="$2"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done

# The region list comes from config, not from a list pasted into this file --
# a hard-coded list is how a new region silently never gets built.
if [ -z "$REGIONS" ]; then
  REGIONS="pilot $($PY -c 'import config; print(" ".join(config.REGIONS))')"
fi

# Per-region stages, in dependency order.
STAGES="build_layout_geojson gate_panels rerank_layouts derive_solar_potential
        patch_roof_confidence bake_building_horizons build_heatmap_raster"

echo "=== district build $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "regions: $(echo $REGIONS | wc -w | tr -d ' ')   resume: ${SKIP:-off}"

fail=0
for r in $REGIONS; do
  echo "=== $r ($(date -u +%H:%M:%S)) ==="
  for s in $STAGES; do
    if ! $PY src/run_stage.py $SKIP "$s" "$r" >>"$LOGDIR/$r.log" 2>&1; then
      echo "  FAILED: $s for $r (see $LOGDIR/$r.log)"
      fail=1
      break
    fi
  done
  # Addresses need the network, and are a patch-in-place post-process. A
  # failure here must not discard the offline compute around it -- re-run
  # later with: python src/run_stage.py add_addresses <region>
  $PY src/run_stage.py $SKIP add_addresses "$r" >>"$LOGDIR/$r.log" 2>&1 \
    || echo "  WARN: addresses failed for $r -- patch later"
done

if [ $fail -ne 0 ]; then
  echo "=== stopping before the fan-in: at least one region failed ==="
  echo "Fix it, re-run this script, and completed regions will be skipped."
  exit 1
fi

echo "=== fan-in ($(date -u +%H:%M:%S)) ==="
# The merge and everything after it are district-wide, so they always run:
# any region rebuild invalidates them, and they are cheap next to the regions.
for s in merge_regions bake_density_deciles build_terrain_masks \
         build_seasonal_curves shrink_panels_for_tiles; do
  $PY src/run_stage.py --force "$s" || { echo "FAILED: $s"; exit 1; }
done

tippecanoe -o data/panel_layouts.pmtiles --force -l layout \
  -Z13 -z16 --drop-densest-as-needed --detect-shared-borders \
  -y kind -y building_id -y fill_rank -y fill_order -y array_id -y array_size \
  -y ac_kwh_year -y slope_deg -y aspect_deg -y roof_confidence \
  -y poa_kwh_m2_yr -y panel_count data/panel_layouts.geojson || exit 1

echo "=== complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

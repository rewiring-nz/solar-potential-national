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
#   ./src/run_district_build.sh                  # resume: skip what is done
#   ./src/run_district_build.sh --incremental    # only buildings whose
#                                                # reading changed (minutes)
#   ./src/run_district_build.sh --force          # rebuild everything
#   ./src/run_district_build.sh --regions "a b"  # just these regions
#
# Interrupting this and re-running it continues where it stopped.
#
# THERE IS NO MERGE ANY MORE. Each region ends by emitting its own tiles,
# cells, detail, heat-map tiles, addresses and a summary (src/emit_region.py),
# and src/combine_regions.py joins them into the served set. Nothing after the
# per-region stages reads the district into memory, which is what lets the
# same script build a town or a country (docs/scale-architecture.md). The
# terrain masks, deciles and panel shrink that used to run on the merged file
# run inside the emit, per region, at that region's own sun.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# The selected-faces chain LEADS district builds. Without this export the build
# silently ignores every data/selected_faces/*.json the precompute wrote --
# there is no error, the old path just answers instead. Set
# SOLAR_SELECTED_FACES=0 explicitly to build old-path only.
export SOLAR_SELECTED_FACES="${SOLAR_SELECTED_FACES:-1}"
LOGDIR=data/build_logs
mkdir -p "$LOGDIR"

SKIP="--skip-done"
REGIONS=""
INCREMENTAL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --force)       SKIP=""; shift ;;
    --incremental) INCREMENTAL=1; shift ;;
    --regions)     REGIONS="$2"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done

# The region list comes from all_areas(), which unions the config with what is
# actually on disk. A hard-coded list -- or the config alone -- is how a region
# silently never gets built: config.REGIONS held 23 entries while data/regions
# held 24, the missing one was `pilot`, and two district rebuilds skipped the
# town centre without erroring.
if [ -z "$REGIONS" ]; then
  REGIONS="$($PY -c 'from src.region_build import all_areas; print(" ".join(all_areas()))')"
fi

# Per-region stages, in dependency order.
STAGES="build_layout_geojson gate_panels rerank_layouts derive_solar_potential
        patch_roof_confidence bake_building_horizons build_heatmap_raster"
# After addresses: the region's own tiles, cells, detail and summary. This
# is what replaced the fan-in (docs/scale-architecture.md).
EMIT="emit_region"

echo "=== district build $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "regions: $(echo $REGIONS | wc -w | tr -d ' ')   resume: ${SKIP:-off}"

# SNAPSHOT THE BUILD WE ARE ABOUT TO REPLACE. The fan-in overwrites
# data/solar_potential.geojson, and once that is gone there is nothing left to
# compare the new build against -- every "did this help?" question becomes
# unanswerable. This was missing on 2 Sep: the only snapshot on the box predated
# the build that was actually deployed, so a comparison would have measured
# against the wrong baseline entirely, and it had to be taken by hand from the
# committed live file before the merge reached it.
#
# Deliberately non-fatal. A missing baseline is bad; losing eight hours of
# compute because the snapshot step tripped would be worse.
# The previous build's per-building ladders live in data/summaries/ now
# (written by combine); keep a copy so compare_builds can diff against them.
if [ -d data/summaries ]; then
  rm -rf data/summaries_prev && cp -r data/summaries data/summaries_prev \
    || echo "  WARN: could not snapshot the previous build -- comparison will be unavailable"
elif [ -f data/solar_potential.geojson ]; then
  $PY src/compare_builds.py --snapshot \
    || echo "  WARN: could not snapshot the previous build -- comparison will be unavailable"
else
  echo "  no existing build to snapshot (first run in this checkout)"
fi

# AUDIT THE INPUTS BEFORE SPENDING HOURS ON THEM. The 3 Sep run built 14
# regions LiDAR-only because their imagery mosaics were gone, and built
# arrowtown_hills as 50 buildings of zeros because its DSM described ground
# 340 m west of every building in it. Neither raised an error; both produced
# output indistinguishable from a real result, and both were found afterwards
# by hand. Ninety seconds of checking beforehand is the cheapest possible way
# to not repeat that.
#
# Non-fatal for the same reason as the snapshot above: a region with degraded
# inputs still builds, and refusing to start the district because one region is
# short of imagery would be a worse failure than the one being prevented. The
# point is that it is stated loudly at the top of the log rather than
# discovered days later.
if [ -f tools/audit_region_inputs.py ]; then
  echo "--- input audit ---"
  $PY tools/audit_region_inputs.py 2>/dev/null \
    | grep -E "PROBLEM|-> |only|regions with problems" \
    || echo "  (audit produced no findings)"
  echo "--- end input audit ---"
fi

# ---------------------------------------------------------------- incremental
#
# THE FAST PATH WAS ALREADY BUILT AND NOTHING CALLED IT, so fixes took a
# long time.
#
# It does, because the unit of work here is the DISTRICT. A one-line change in
# face_candidates invalidates build_layout_geojson for all 24 regions and costs
# four and a half hours, including every building that reading cannot have
# touched.
#
# tools/patch_stale_selected.py has done the right thing for weeks: it hashes
# each building's selected reading against data/built_from.json and rebuilds
# only the mismatches -- layouts, gate, merged file, solar_potential and all.
# Content-hashed rather than mtime-based, so it is correct however many times a
# preemptible VM kills it, and unlike mtimes it survives the patching that
# rewrites the layouts underneath it.
#
# So: --incremental does that and then the district tail, and a change touching
# forty roofs costs minutes. The full path is unchanged and is still what a new
# region, a new stage, or anything outside the selected-faces chain needs.
#
# NOT FOR A WHOLE-DISTRICT CHANGE. patch_buildings works in chunks of 60 and
# each chunk re-reads and rewrites the 394 MB merged layouts, which is cheap
# for a handful of roofs and ruinous for all of them: re-predicting every
# reading would be 223 chunks of that. When the change touches most buildings
# -- a new face_candidates, a new fitter -- the FULL path is the fast one.
# Rough line: under a thousand buildings, patch; above it, rebuild.
#
# data/built_from.json IS PER MACHINE and is not committed -- it records what
# THIS checkout has built. On a machine that has never run a full build
# everything hashes as stale and --incremental degrades to a full rebuild,
# which is correct but slow. Builds run on the VM, which has the state.
if [ $INCREMENTAL -eq 1 ]; then
  echo "=== incremental: rebuilding only buildings whose reading changed ==="
  $PY tools/patch_stale_selected.py --patch || exit 1
  # Re-emit every region whose files the patch touched, then combine. The
  # emit stage's marker is older than the patched region files, so
  # --skip-done re-emits exactly those and skips the rest.
  echo "=== re-emit ($(date -u +%H:%M:%S)) ==="
  for r in $REGIONS; do
    $PY src/run_stage.py --skip-done emit_region "$r" >>"$LOGDIR/$r.log" 2>&1 \
      || { echo "  FAILED: emit_region for $r (see $LOGDIR/$r.log)"; exit 1; }
  done
  echo "=== combine ($(date -u +%H:%M:%S)) ==="
  $PY src/combine_regions.py || { echo "FAILED: combine_regions"; exit 1; }
else

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
  # Where the photo sits relative to the LiDAR, per building; a failure here
  # only means the drawing stays where the LiDAR is.
  $PY src/run_stage.py $SKIP register_imagery "$r" >>"$LOGDIR/$r.log" 2>&1 \
    || echo "  WARN: image registration failed for $r -- drawing unshifted"
  if ! $PY src/run_stage.py $SKIP $EMIT "$r" >>"$LOGDIR/$r.log" 2>&1; then
    echo "  FAILED: $EMIT for $r (see $LOGDIR/$r.log)"
    fail=1
  fi
done

if [ $fail -ne 0 ]; then
  echo "=== stopping before the fan-in: at least one region failed ==="
  echo "Fix it, re-run this script, and completed regions will be skipped."
  exit 1
fi

echo "=== combine ($(date -u +%H:%M:%S)) ==="
# NO MERGE. Each region emitted its own tiles, cells, detail, heat-map tiles
# and addresses under data/out/<region>/; combine_regions joins them into the
# served set without ever reading the district into memory. The merged
# solar_potential.geojson and panel_layouts.geojson are no longer produced by
# the build -- src/merge_regions.py still exists for debugging, and nothing
# in the ship path reads its output.
$PY src/combine_regions.py || { echo "FAILED: combine_regions"; exit 1; }

fi   # end of the full-build branch

# DID THE BUILD ACTUALLY USE ITS INPUTS? On 10 Sep a resumed district run
# skipped every layout stage on stale markers and shipped the previous
# geometry with fresh mtimes -- zero errors, bit-identical totals. A green
# build that ignored its inputs must FAIL here, not deploy quietly. The count
# now comes from the region summaries the emit stage wrote, summed by combine.
if [ "${SOLAR_SELECTED_FACES}" = "1" ] && [ "$(ls data/selected_faces 2>/dev/null | wc -l)" -gt 100 ]; then
  n_sel=$($PY -c 'import json; print(int(json.load(open("data/build_summary.json"))["totals"].get("from_selected_facets", 0)))')
  if [ "${n_sel:-0}" -lt 50 ]; then
    echo "FAILED: selected-faces enabled but only ${n_sel} from_selected facets across the build -- it did not use its inputs"
    exit 1
  fi
  echo "guard: ${n_sel} from_selected facets across the build"
fi

echo "=== complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

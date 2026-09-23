#!/bin/bash
# One region, start to finish, on any machine: fetch -> predict -> build ->
# emit -> publish. This is the unit the worker loop runs and the unit the
# queue hands out (docs/scale-architecture.md).
#
#   ./src/build_region.sh <region> [--no-publish]
#
# Resumable at every stage: fetches skip what is on disk, face prediction is
# recorded per region, build stages use run_stage markers, and the emit is
# atomic. Re-running after a preemption continues from where it stopped.
#
# STOPS ON A FAILED FETCH, on purpose. The first expansion script made every
# stage non-fatal, so a 400 on the Wanaka DSM and 165 missing point-cloud
# tiles scrolled past and the build started on nothing.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
REGION="${1:?region name}"
PUBLISH=1
[ "${2:-}" = "--no-publish" ] && PUBLISH=0
[ -f .env ] && { set -a; . ./.env; set +a; }
export SOLAR_SELECTED_FACES="${SOLAR_SELECTED_FACES:-1}"
SHARDS="${SOLAR_PREDICT_SHARDS:-8}"
LOGDIR=data/build_logs; mkdir -p "$LOGDIR"
STATE=data/build_state; mkdir -p "$STATE"
log() { echo "[$REGION] $(date -u +%H:%M:%S) $*"; }

# The task file from the queue carries the region's bbox, so a worker whose
# config does not list this region can still build it.
mkdir -p "data/regions/$REGION"
gcloud storage cat "${SOLAR_BUCKET:-gs://rewiring-solar-data}/${SOLAR_BUILD_PREFIX:-build}/queue/$REGION.json" \
  > "data/regions/$REGION/task.json" 2>/dev/null || rm -f "data/regions/$REGION/task.json"

log "fetch"
$PY src/fetch_regions.py "$REGION" || { log "FATAL: fetch_regions rc=$?"; exit 1; }
$PY src/fetch_pointcloud_regions.py "$REGION" || { log "FATAL: pointcloud rc=$?"; exit 1; }

# Ownership. A region from the national planner is a grid cell that overlaps
# nothing, so this is a no-op there; hand-drawn regions overlap and need it.
log "dedupe outlines"
$PY src/region_build.py "$REGION" >>"$LOGDIR/$REGION.log" 2>&1 || log "WARN: dedupe rc=$? (continuing)"

if [ ! -f "$STATE/$REGION.predict.done" ]; then
  log "predict faces ($SHARDS shards)"
  rc=0
  for i in $(seq 0 $((SHARDS - 1))); do
    OMP_NUM_THREADS=2 $PY tools/predict_faces.py --region "$REGION" --shard "$i/$SHARDS" \
      >>"$LOGDIR/predict_$REGION.log" 2>&1 &
  done
  for job in $(jobs -p); do wait "$job" || rc=1; done
  [ $rc -eq 0 ] || { log "FATAL: a predict shard failed (see $LOGDIR/predict_$REGION.log)"; exit 1; }
  date -u +%FT%TZ > "$STATE/$REGION.predict.done"
else
  log "predict: already done"
fi

log "build"
for s in build_layout_geojson gate_panels rerank_layouts derive_solar_potential \
         patch_roof_confidence bake_building_horizons build_heatmap_raster; do
  $PY src/run_stage.py --skip-done "$s" "$REGION" >>"$LOGDIR/$REGION.log" 2>&1 \
    || { log "FATAL: $s failed (see $LOGDIR/$REGION.log)"; exit 1; }
done
$PY src/run_stage.py --skip-done add_addresses "$REGION" >>"$LOGDIR/$REGION.log" 2>&1 \
  || log "WARN: addresses failed -- patch later"
$PY src/run_stage.py --skip-done register_imagery "$REGION" >>"$LOGDIR/$REGION.log" 2>&1 \
  || log "WARN: image registration failed -- drawing unshifted"

log "emit"
$PY src/run_stage.py --skip-done emit_region "$REGION" >>"$LOGDIR/$REGION.log" 2>&1 \
  || { log "FATAL: emit_region failed (see $LOGDIR/$REGION.log)"; exit 1; }

if [ $PUBLISH -eq 1 ]; then
  log "publish"
  $PY src/publish_region.py "$REGION" || { log "FATAL: publish failed"; exit 1; }
fi
log "DONE"

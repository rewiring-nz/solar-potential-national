#!/bin/bash
# Full Queenstown build: every region through the whole pipeline, then merge.
#
# Each region runs in its own python process per stage on purpose:
# PointCloudSource caches every decoded LiDAR tile in memory for the life of
# the process, so one long-lived "all regions" process would accumulate the
# whole ~10GB+ tile set decoded in RAM. Per-region processes keep the cache
# bounded to one region's tiles.
#
# Resumable at region granularity: pass region names to rebuild just those,
# default is all of them. Logs land in data/build_logs/<region>.log.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOGDIR=data/build_logs
mkdir -p "$LOGDIR"

REGIONS="${@:-town_west_fernhill town_gorge_north frankton_flats frankton_quail_rise kelvin_heights jacks_point hanleys_farm shotover_lakehayes arthurs_point arrowtown_millbrook}"

fail=0
for r in $REGIONS; do
  log="$LOGDIR/$r.log"
  echo "=== $r ($(date +%H:%M:%S)) ==="
  {
    $PY src/build_heatmap.py "$r" &&
    # Addresses are a patch-in-place post-process on solar_potential.geojson,
    # and the only per-region step that needs the network (Josh's connection
    # is intermittent) -- a failure here must not throw away the ~15min of
    # offline-safe compute around it. Re-run later: python src/add_addresses.py <region>
    { $PY src/add_addresses.py "$r" || echo "WARN: addresses failed for $r -- patch later"; } &&
    $PY src/build_layout_geojson.py "$r" &&
    # placement gate: evidence-based per-panel veto (carparks/air/missed
    # structure), then re-band fill ranks on the gated set
    $PY src/gate_panels.py "$r" &&
    $PY src/rerank_layouts.py "$r" &&
    $PY src/build_heatmap_raster.py "$r"
  } >"$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "FAILED: $r (see $log)"
    fail=1
  else
    echo "done: $r"
  fi
done

if [ $fail -eq 0 ]; then
  $PY src/merge_regions.py && echo "MERGE COMPLETE"
else
  echo "Skipping merge: at least one region failed"
  exit 1
fi

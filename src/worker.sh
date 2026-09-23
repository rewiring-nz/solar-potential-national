#!/bin/bash
# The worker loop: claim a region from the bucket queue, build it, publish
# it, mark it done; repeat until the queue is empty; then, if asked, delete
# this machine so an idle fleet costs nothing.
#
#   ./src/worker.sh                 # run until the queue is empty
#   SOLAR_SELF_DELETE=1 ./src/worker.sh   # ...then delete this VM (fleet use)
#
# Everything about correctness lives in src/gcs_queue.py: the claim is an
# atomic create-if-absent, it is refreshed every five minutes from a
# background heartbeat here, and a claim that stops being refreshed is taken
# over by another worker after 45 minutes -- which is what a preempted spot
# VM looks like from the outside.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export SOLAR_WORKER="${SOLAR_WORKER:-$(hostname)}"
LOGDIR=data/build_logs; mkdir -p "$LOGDIR"
log() { echo "[worker $SOLAR_WORKER] $(date -u +%H:%M:%S) $*"; }

while true; do
  REGION=$($PY src/gcs_queue.py claim) || { log "queue empty"; break; }
  [ -n "$REGION" ] || { log "queue empty"; break; }
  log "claimed $REGION"

  # heartbeat while the build runs
  ( while true; do sleep 300; $PY -c "import sys; sys.path.insert(0,'.'); from src.gcs_queue import heartbeat; heartbeat('$REGION')" 2>/dev/null; done ) &
  HB=$!

  if ./src/build_region.sh "$REGION" >"$LOGDIR/worker_$REGION.log" 2>&1; then
    kill $HB 2>/dev/null; wait $HB 2>/dev/null
    $PY - <<PYEOF
import sys, json; sys.path.insert(0, '.')
from src.gcs_queue import mark_done
from pathlib import Path
s = json.loads(Path('data/out/$REGION/summary.json').read_text()); s.pop('ladder', None)
mark_done('$REGION', s)
PYEOF
    log "done $REGION"
  else
    kill $HB 2>/dev/null; wait $HB 2>/dev/null
    $PY - <<PYEOF
import sys; sys.path.insert(0, '.')
from src.gcs_queue import mark_failed
from pathlib import Path
mark_failed('$REGION', Path('$LOGDIR/worker_$REGION.log').read_text()[-4000:])
PYEOF
    log "FAILED $REGION (recorded; will be retried by any worker)"
  fi
done

if [ "${SOLAR_SELF_DELETE:-0}" = "1" ]; then
  ZONE=$(curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/zone" | rev | cut -d/ -f1 | rev)
  NAME=$(curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/name")
  log "queue empty -- deleting $NAME in $ZONE"
  gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet
fi

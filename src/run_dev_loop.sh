#!/bin/bash
# Fast iteration loop: rebuild ONE or TWO areas and measure, instead of the
# whole district.
#
# A full rebuild is 15,353 buildings across 24 areas and takes about 4.5
# hours, which is far too slow to iterate placement changes against -- Josh,
# after a day of it: "it's taking a lot of time now to implement fixes and we
# have a lot to fix". Pilot alone is 1,066 buildings (6.9% of the district)
# and rebuilds in roughly 15 minutes.
#
# Suggested pair, chosen to cover the failure modes that actually differ:
#   pilot           CBD -- flat commercial roofs, parapets, rooftop plant,
#                   sawtooth, plus dense complex residential
#   frankton_flats  large-format commercial/industrial, big simple roofs
#
# Not covered by those two, so re-check district-wide before shipping:
#   rural areas have NO aerial imagery (LINZ urban-only), so colour-based
#   obstruction detection is absent there entirely.
#
# This does NOT merge or tile -- deploying is a separate, slower step. The
# point here is to measure, using the same audits the district run uses.
#
# Usage: bash src/run_dev_loop.sh [area ...]      (default: pilot)
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/bin/python
AREAS=${@:-pilot}
START=$(date +%s)

for r in $AREAS; do
  echo "=== $r: rebuilding layouts ==="
  log="data/build_logs/${r}_dev.log"
  {
    $PY src/build_layout_geojson.py "$r" &&
    $PY src/gate_panels.py "$r" &&
    $PY src/rerank_layouts.py "$r"
  } >"$log" 2>&1 || { echo "FAILED: $r (see $log)"; exit 1; }
  grep -E "facets, .* panels|dropped .* panels" "$log" | tail -2
done

echo
echo "=== layout quality ==="
for r in $AREAS; do
  echo "--- $r"
  $PY src/audit_layout_quality.py --area "$r" --top 0 2>&1 | sed -n '3,9p'
done

echo
echo "=== obstruction validation (labelled set) ==="
$PY src/validate_obstructions.py 2>&1 | tail -20

echo
echo "dev loop finished in $(( ($(date +%s) - START) / 60 )) min"

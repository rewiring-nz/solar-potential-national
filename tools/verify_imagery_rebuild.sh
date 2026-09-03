#!/usr/bin/env bash
# Everything needed to answer "did the 3 Sep rebuild help?", in one pass.
#
# Written before the rebuild finished, so that the checks are fixed in advance
# rather than chosen once the numbers are visible. The prediction they test is
# recorded in BACKLOG.md under PRE-REGISTERED PREDICTION: restoring imagery
# should take the panel-crossing rate on Josh's labelled roofs from 23.8% to
# about 21.9%, leaving ~1.5 points unexplained.
#
# THE COMPARISON IS --no-model ON BOTH SIDES, and that is not a detail.
# predict_roof_lines has just regenerated the vision lines for regions that had
# none, and the model's lines are the yardstick behind the strongest triage
# signal. Comparing against a yardstick that moved between the two runs would
# measure the yardstick. triage_roofs.py now refuses such a comparison outright,
# but the flag is passed explicitly here so the intent is on the page.

set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
BASE=data/triage/baseline_prerebuild_vm.json

echo "=============================================================="
echo " 1. INPUTS -- is every region built on data that describes it?"
echo "=============================================================="
$PY tools/audit_region_inputs.py 2>/dev/null | tail -32

echo
echo "=============================================================="
echo " 2. DISTRICT TOTALS -- against the 09-02 snapshot"
echo "    (15,353 buildings, 682,433 panels, 368.2 GWh/yr)"
echo "=============================================================="
$PY src/compare_builds.py 2>/dev/null | head -40 \
  || echo "  compare_builds unavailable"

echo
echo "=============================================================="
echo " 3. THE PRE-REGISTERED NUMBER -- panels crossing Josh's lines"
echo "    30 Aug build 20.4%  |  1 Sep build 23.8%  |  predicted ~21.9%"
echo "=============================================================="
$PY tools/triage_roofs.py --validate --no-model 2>/dev/null \
  | sed -n '/SIGNAL VALIDATION/,/scoring on/p'

echo
echo "=============================================================="
echo " 4. GEOMETRY -- triage score against the pre-rebuild baseline"
echo "=============================================================="
$PY tools/triage_roofs.py --no-model --compare "$BASE" 2>/dev/null \
  | grep -E "AGAINST|REFUSING|then:|now :|FAILURE MODES" -A 6

echo
echo "=============================================================="
echo " 5. SANITY -- the standing rebuild checks"
echo "=============================================================="
$PY tools/verify_rebuild.py 2>/dev/null | tail -25 \
  || echo "  verify_rebuild unavailable"

echo
echo "Read BACKLOG.md 'PRE-REGISTERED PREDICTION' before interpreting section 3."
echo "A result near 21.9% confirms the imagery story AND leaves ~1.5 points"
echo "still unexplained. A result near 23.8% means imagery was not the cause"
echo "and the region split was confounded -- find the real cause, do not"
echo "re-explain this one."

#!/bin/bash
# Every check this repo has, in one command. Run before a push.
#
#   ./tests/run_all.sh          # everything
#   ./tests/run_all.sh --fast   # skip the golden tests (they segment real
#                               # buildings and take a few minutes)
#
# Exit status is non-zero if any check fails, so it works as a pre-push hook.
#
# What each one is for:
#   pure         arithmetic you cannot see -- aspect conventions, the horizon
#                codec, lookup binning, the derate. Mutation-checked.
#   deprecations the class that nearly removed shapely.vectorized from under
#                the geometry core on a routine dependency upgrade.
#   sync         the two repos have already diverged twice, silently, with
#                both bugs reaching the public site.
#   golden       pins what segmentation currently produces for the 28
#                ground-truth buildings, so a refactor cannot move geometry
#                unnoticed. Needs local region data; skips without it.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FAST=0
[ "${1:-}" = "--fast" ] && FAST=1

fail=0
run() {
  echo ""
  echo "=== $1 ==="
  shift
  "$@" || fail=1
}

run "pure functions"        $PY tests/test_pure.py
run "deprecated APIs"       $PY tests/test_no_deprecations.py
run "repo sync"             $PY tools/check_repo_sync.py
if [ $FAST -eq 0 ]; then
  run "golden buildings"    $PY -W ignore tests/test_golden.py
else
  echo ""
  echo "=== golden buildings: SKIPPED (--fast) ==="
fi

echo ""
if [ $fail -eq 0 ]; then
  echo "all checks passed"
else
  echo "SOME CHECKS FAILED -- see above"
fi
exit $fail

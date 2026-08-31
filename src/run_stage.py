"""
Run one pipeline stage with preflight, a completion marker, and resume.

Why: the build loop had no memory. A full district rebuild is ~25 regions at
~28 minutes each -- around twelve hours -- and every interruption started again
from region one. The Queenstown rebuild of 31 Aug was launched three times and
redid completed regions each time, which is most of a day of compute spent
recomputing work that was already correct on disk.

This is deliberately a THIN wrapper around exactly the command the build
scripts already run:

    python src/run_stage.py build_layout_geojson queenstown_hill
        ->  python src/build_layout_geojson.py queenstown_hill

Same interpreter, same arguments, same output. What it adds around that:

  PREFLIGHT   the stage's inputs are checked before it starts, so a missing
              input costs a second rather than an hour (see src/preflight.py).
  MARKER      on success, data/build_state/<region>.<stage>.done records when
              it finished and at which commit.
  RESUME      with --skip-done, a stage whose marker is newer than all of its
              declared inputs is skipped. Without the flag nothing is skipped,
              so existing behaviour is unchanged and a plain re-run still
              rebuilds.

Staleness reuses preflight's input table rather than a second list: if any
declared input is newer than the marker, the stage is NOT done, and it re-runs.
That means editing a region's outlines correctly invalidates everything
downstream of it without anyone maintaining a dependency graph by hand.

Usage in a build loop:

    for r in $REGIONS; do
      for s in build_layout_geojson gate_panels rerank_layouts; do
        $PY src/run_stage.py --skip-done "$s" "$r" || exit 1
      done
    done

Re-running that loop after a crash resumes where it stopped.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATE_DIR = DATA_DIR / "build_state"


def marker_path(stage, region):
    return STATE_DIR / f"{region or '_district'}.{stage}.done"


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _declared_inputs(stage, region):
    """The stage's inputs, from preflight's table -- one source of truth."""
    from src.preflight import REQUIRED, DATA_DIR as PF_DATA
    spec = REQUIRED.get(stage) or {}
    out = []
    if region is not None and (spec.get("region") or spec.get("optional_region")):
        from src.region_build import area_paths
        paths = area_paths(region)
        for key in list(spec.get("region", [])) + list(spec.get("optional_region", [])):
            p = paths.get(key)
            if p is not None:
                out.append(Path(p))
    for name in spec.get("root", []):
        out.append(PF_DATA / name)
    return out


def is_done(stage, region):
    """True only if the marker exists AND no declared input is newer."""
    m = marker_path(stage, region)
    if not m.exists():
        return False, "no marker"
    m_time = m.stat().st_mtime
    for p in _declared_inputs(stage, region):
        if p.exists() and p.stat().st_mtime > m_time + 1.0:
            return False, f"input newer than marker: {p.name}"
    return True, "done"


def write_marker(stage, region, seconds):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    marker_path(stage, region).write_text(json.dumps({
        "stage": stage,
        "region": region,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(seconds, 1),
        "commit": _git_sha(),
    }, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("region", nargs="?", default=None)
    ap.add_argument("--skip-done", action="store_true",
                    help="skip if the marker is newer than every declared input")
    ap.add_argument("--force", action="store_true",
                    help="run even with --skip-done set")
    ap.add_argument("--status", action="store_true",
                    help="report done/not-done and exit without running")
    a = ap.parse_args()

    script = ROOT / "src" / f"{a.stage}.py"
    if not script.exists():
        print(f"no such stage: {a.stage} ({script} does not exist)")
        return 2

    label = f"{a.stage}" + (f" [{a.region}]" if a.region else "")
    done, why = is_done(a.stage, a.region)

    if a.status:
        print(f"{'DONE ' if done else 'TODO '} {label}  ({why})")
        return 0

    if done and a.skip_done and not a.force:
        print(f"[skip] {label} already done -- {why}", flush=True)
        return 0

    # Preflight before spending the time. The stage checks its own inputs too;
    # doing it here as well means a resumable loop fails on region 1 rather
    # than at whatever hour region 14 would have started.
    try:
        from src.preflight import preflight
        preflight(a.stage, a.region)
    except SystemExit as e:
        print(str(e))
        return 1

    cmd = [sys.executable, str(script)] + ([a.region] if a.region else [])
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0

    if r.returncode != 0:
        print(f"[FAIL] {label} exited {r.returncode} after {elapsed:.0f}s "
              f"-- no marker written, so a resumed run will retry it",
              flush=True)
        return r.returncode

    write_marker(a.stage, a.region, elapsed)
    print(f"[done] {label} in {elapsed:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

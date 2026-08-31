"""
Fail when solar-map and solar-wellington drift apart.

Why this exists: the two repos are the same application pointed at different
regions, kept in step by remembering to copy files. On 31 Aug a review diffed
them for the first time and found TWO silent divergences, in opposite
directions:

  * panel_fitting.py -- Wellington was missing the gap-fill pass and the
    straggler yield exception. Island Bay had been placing fewer panels at
    100% density, and stripping good panels first, for days.
  * patch_buildings.py -- solar-map was missing the solar_potential splice
    that its own docstring claimed it performed. Patching a building there
    updated the map and left the dashboard quoting stale numbers.

Neither was detectable by any check that existed. Both shipped to the public
site. This script is the cheap stand-in until the real fix -- one codebase,
two configurations -- lands.

Run:  python tools/check_repo_sync.py
      python tools/check_repo_sync.py --other ../solar-wellington
      python tools/check_repo_sync.py --diff panel_fitting.py

Exit status is 1 on unexpected divergence, so it works as a pre-push hook or
a CI step unchanged.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OTHER = ROOT.parent / "solar-wellington"

# Files that legitimately differ, each with the reason. Anything NOT listed
# here is expected to be byte-identical. Add entries deliberately: every
# addition is a place where a fix can land in one repo and not the other.
ALLOWED = {
    "config.py":
        "per-deployment: regions, centre, tile sources, public assumptions text",
    "src/patch_buildings.py":
        "Wellington is a single-region deploy -- the region IS the district, so "
        "it rebuilds the merged file instead of patching a standing one",
    "site-config.js":
        "per-deployment FRONTEND settings: which town the map opens on, and the "
        "areas in the search box. Added 1 Sep because preview.html is shared "
        "byte-for-byte between the two deploys, so anything site-specific "
        "hardcoded in it is wrong for one of them -- which is exactly what "
        "happened: the Queenstown site opened on Island Bay for a day",
}


def shared_files(root):
    """Everything both repos are expected to hold in common."""
    out = [p.relative_to(root).as_posix()
           for p in sorted((root / "src").glob("*.py"))]
    # tools/ and tests/ count too. They were left out of the first version of
    # this script, which would have let the guard itself drift between repos --
    # the one file whose whole job is noticing drift.
    for sub in ("tools", "tests"):
        out += [p.relative_to(root).as_posix()
                for p in sorted((root / sub).glob("*.py"))]
    for extra in ("preview.html", "config.py", "requirements.txt",
                  "requirements.lock.txt", "site-config.js"):
        if (root / extra).exists():
            out.append(extra)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--other", default=str(DEFAULT_OTHER),
                    help="path to the other repo (default: ../solar-wellington)")
    ap.add_argument("--diff", metavar="FILE",
                    help="show the actual diff for one file and exit")
    a = ap.parse_args()

    other = Path(a.other).resolve()
    if not other.exists():
        print(f"other repo not found: {other}")
        return 2

    if a.diff:
        rel = a.diff if "/" in a.diff else f"src/{a.diff}"
        subprocess.run(["diff", "-u", str(ROOT / rel), str(other / rel)])
        return 0

    unexpected, missing, allowed_hits = [], [], []

    for rel in shared_files(ROOT):
        mine, theirs = ROOT / rel, other / rel
        if not theirs.exists():
            missing.append(rel)
            continue
        if mine.read_bytes() == theirs.read_bytes():
            continue
        d = subprocess.run(["diff", str(mine), str(theirs)],
                           capture_output=True, text=True).stdout
        here = sum(1 for l in d.splitlines() if l.startswith("<"))
        there = sum(1 for l in d.splitlines() if l.startswith(">"))
        (allowed_hits if rel in ALLOWED else unexpected).append((rel, here, there))

    # Files the other repo has and this one does not.
    for rel in shared_files(other):
        if not (ROOT / rel).exists():
            missing.append(f"{rel}  (only in {other.name})")

    if allowed_hits:
        print("expected divergence:")
        for rel, here, there in allowed_hits:
            print(f"  {rel}  (+{here}/-{there} lines)")
            print(f"      {ALLOWED[rel]}")
        print()

    if not unexpected and not missing:
        print(f"in sync: {len(shared_files(ROOT))} shared files, "
              f"{len(allowed_hits)} allowed divergences, 0 unexpected")
        return 0

    if missing:
        print("MISSING FILES:")
        for rel in missing:
            print(f"  {rel}")
        print()

    if unexpected:
        print("UNEXPECTED DIVERGENCE -- a fix has landed in one repo only:")
        for rel, here, there in unexpected:
            print(f"  {rel}")
            print(f"      {here} lines only in {ROOT.name}, "
                  f"{there} only in {other.name}")
        print()
        print("Inspect with:  python tools/check_repo_sync.py --diff <file>")
        print("Then either copy the fix across, or add the file to ALLOWED in")
        print("this script WITH the reason it is meant to differ.")

    return 1


if __name__ == "__main__":
    sys.exit(main())

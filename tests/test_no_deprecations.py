"""
Fail when a library module imports with a DeprecationWarning.

This exists because of a near-miss. 68 calls to `shapely.vectorized.contains`
were spread across 28 files. That API is deprecated in Shapely 2.x and
documented as "will be removed in a future version", and requirements.txt asked
for `shapely>=2.0` -- so an ordinary reinstall would eventually have removed
the function the geometry core is built on, and the pipeline would simply stop.

Nobody saw it, and the reason is uncomfortable: 28 modules call
`warnings.filterwarnings("ignore")` at import. The countdown to a hard breakage
was running in silence for as long as those lines have been there.

The test does three things, and the ORDER OF TRUST is the opposite of the order
you would guess. Each was checked by reintroducing a deprecated call and seeing
whether it actually failed:

  IMPORT   every library module (the ones OTHER modules import, not the one-shot
           scripts) in a subprocess with DeprecationWarning promoted to an error.
           A blanket filterwarnings inside a module cannot hide it, because -W is
           applied by the interpreter before the module runs.
           CAUGHT NOTHING on its own: `import shapely.vectorized` emits no
           warning at all. The warning fires when the function is CALLED.

  BANNED   a static scan for APIs we have already been bitten by. This is the
           one that works. It cannot find deprecations nobody knows about, but
           it catches a known one ANYWHERE, including on rare branches.

  EXERCISE run two real paths with deprecations fatal, to catch call-time
           deprecations the ban list does not know about:
             - the GEOMETRY core, by segmenting a real building;
             - the YIELD model, which is pandas/numpy time-series code and
               produces the kWh figures the public reads.
           Both promote FutureWarning as well as DeprecationWarning, because
           pandas signals upcoming breakage with FutureWarning -- a check that
           promoted only DeprecationWarning would sail straight past the
           warning class pandas actually uses.
           Honest limitation, measured rather than assumed: a deprecated call
           reintroduced on a branch the sample building did not take passed
           this step cleanly. It sees only what it runs, which is why the
           static ban list above carries the weight.

Scripts are excluded from the import pass on purpose: several do their whole job
at import (see score_all_marked, which used to rewrite its committed baseline
when merely imported), so importing them here would be slow and destructive.

The exercise step needs real region data, which is gitignored; without it that
half SKIPS rather than fails, so this stays useful on a build machine and
harmless anywhere else.

Run:  .venv/bin/python tests/test_no_deprecations.py
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def library_modules():
    """Modules that at least one other module imports -- the real dependencies.

    A stage script is invoked, never imported, so a deprecation inside one is a
    problem for that script alone. A deprecation in roof_segmentation stops the
    district."""
    names = {p.stem for p in SRC.glob("*.py")}
    imported = set()
    pattern = re.compile(r"^\s*(?:from\s+(?:src\.)?(\w+)\s+import|import\s+(?:src\.)?(\w+))",
                         re.M)
    for p in SRC.glob("*.py"):
        text = p.read_text()
        for m in pattern.finditer(text):
            mod = m.group(1) or m.group(2)
            if mod in names and mod != p.stem:
                imported.add(mod)
    return sorted(imported)


def check(mod):
    """Import one module with deprecations fatal. Returns None or the message."""
    r = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c",
         f"import sys; sys.path.insert(0, {str(ROOT)!r}); import src.{mod}"],
        capture_output=True, text=True, timeout=300)
    if r.returncode == 0:
        return None
    err = r.stderr.strip()
    if "DeprecationWarning" in err:
        for line in err.splitlines():
            if "DeprecationWarning" in line:
                return line.strip()
        return err.splitlines()[-1]
    # Some other import error -- real, and worth failing on too.
    return err.splitlines()[-1] if err else f"exit {r.returncode}"


# A real workload, run with deprecations fatal. Deliberately exercises the
# geometry core -- segmentation pulls in roof_partition, roof_skeleton,
# pointcloud_source and the point-in-polygon paths where the deprecated call
# actually lived.
EXERCISE = """
import sys, warnings
sys.path.insert(0, {root!r})
sys.argv = ["exercise"]
warnings.simplefilter("error", DeprecationWarning)
warnings.simplefilter("error", FutureWarning)
import geopandas as gpd, rasterio
from src.region_build import area_paths
from src.pointcloud_source import PointCloudSource
from src.roof_segmentation import segment_building_best

p = area_paths("pilot")
if not p["outlines"].exists() or not p["dsm"].exists():
    print("SKIP: no local region data")
    raise SystemExit(0)
gdf = gpd.read_file(p["outlines"]).set_index("building_id", drop=False)
dsm = rasterio.open(p["dsm"])
img = rasterio.open(p["imagery"]) if p["imagery"].exists() else None
pc = PointCloudSource()
bid = int(gdf.index[0])
facets = segment_building_best(dsm, pc, gdf.loc[bid, "geometry"], bid, imagery_ds=img)
print(f"OK: segmented building {{bid}} into {{len(facets or [])}} facets")
"""


# The second path worth exercising, and a different risk entirely: the yield
# model is pandas- and numpy-heavy time-series code, and it produces the kWh
# figures the public reads. pandas signals upcoming breakage with FutureWarning
# rather than DeprecationWarning, which is why both are fatal here -- a check
# that only promoted DeprecationWarning would sail past the warning class
# pandas actually uses.
SOLAR_EXERCISE = """
import sys, warnings
sys.path.insert(0, {root!r})
sys.argv = ["exercise"]
warnings.simplefilter("error", DeprecationWarning)
warnings.simplefilter("error", FutureWarning)
from src.solar_model import SolarModel
m = SolarModel()
poa = m.annual_poa_kwh_per_m2(20, 0)
assert poa > 0, poa
print(f"OK: yield model clean (20deg north = {{poa:.0f}} kWh/m2/yr)")
"""


def exercise(script=None, label="geometry core"):
    """Run a real path with deprecations fatal. None, or the message."""
    r = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c",
         (script or EXERCISE).format(root=str(ROOT))],
        capture_output=True, text=True, timeout=900)
    if r.returncode == 0:
        return None, (r.stdout.strip().splitlines() or ["ok"])[-1]
    err = (r.stderr or "").strip()
    for line in err.splitlines():
        if "DeprecationWarning" in line:
            return line.strip(), "failed"
    return (err.splitlines()[-1] if err else f"exit {r.returncode}"), "failed"


# APIs we have already been bitten by, banned by name across the whole tree.
#
# The dynamic checks below only see code they actually execute -- verified the
# hard way: reintroducing a deprecated call on a branch the sample building did
# not take passed cleanly. A static ban has the opposite trade-off. It cannot
# find deprecations we do not know about, but it catches a known one ANYWHERE,
# including on the rare branches that are the easiest place for one to hide.
BANNED = {
    "shapely.vectorized": (
        "deprecated in Shapely 2.x and documented for removal; 68 call sites "
        "across 28 files nearly broke the geometry core. Use shapely.contains_xy."),
}


def banned_symbols():
    """Every use of a known-deprecated API, anywhere in src/ or tools/."""
    hits = []
    for d in ("src", "tools"):
        for p in sorted((ROOT / d).glob("*.py")):
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                for sym in BANNED:
                    if sym in line:
                        hits.append((f"{d}/{p.name}:{i}", sym, line.strip()))
    return hits


def main():
    mods = library_modules()
    print(f"checking {len(mods)} library modules with DeprecationWarning as error\n")
    bad = []
    for m in mods:
        msg = check(m)
        if msg:
            bad.append((m, msg))
            print(f"  FAIL  {m}\n          {msg[:150]}")
        else:
            print(f"  pass  {m}")

    print("\nscanning for known-deprecated APIs (catches every branch, not just "
          "executed ones)")
    hits = banned_symbols()
    if hits:
        for where, sym, line in hits[:12]:
            bad.append((where, f"banned API {sym}"))
            print(f"  FAIL  {where}: {line[:90]}")
            print(f"          {BANNED[sym]}")
    else:
        print(f"  pass  none of {len(BANNED)} banned API(s) present")

    print("\nexercising real paths (where call-time deprecations live)")
    for script, label in ((EXERCISE, "geometry core"),
                          (SOLAR_EXERCISE, "yield model")):
        msg, note = exercise(script, label)
        if msg:
            bad.append((f"<{label}>", msg))
            print(f"  FAIL  {label}: {msg[:150]}")
        else:
            print(f"  pass  {note}")

    total = len(mods) + 3
    print(f"\n{total - len(bad)}/{total} checks clean")
    if bad:
        print("\nA deprecation here is a countdown to the pipeline breaking on a")
        print("routine dependency upgrade. Fix the call site rather than widening")
        print("a filterwarnings -- that is precisely how the last one stayed hidden.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

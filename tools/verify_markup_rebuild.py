"""
Did the rebuild actually ship the geometry Josh drew?

The agreement measurement (tools/measure_facet_agreement.py) runs the
segmentation live. This checks the OPPOSITE end: what is in the built
panel_layouts.geojson that is about to be deployed. A fix that measures well
in isolation and does not survive the build is not a fix, and that gap is
exactly what went unnoticed before -- his markup was computed correctly and
discarded downstream for weeks.

Reports per region and overall:
  MATCH      built facet count equals the usable faces he drew
  MEAN ERR   average absolute difference in facet count
  Anything not matching is listed, worst first, because a roof that ships 9
  facets where he drew 2 is a different failure from one that ships 3.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    from src.region_build import all_areas, area_paths
    from src.roof_line_source import drawn_faces
    labels = json.loads((ROOT / "data" / "roof_labels.json").read_text())["buildings"]
    SKIP = {"absent", "not_building", "unclear"}
    want = {int(k): len([f for f in drawn_faces(int(k)) if f["usable"]])
            for k, v in labels.items()
            if v.get("complete") and v.get("problem") not in SKIP}
    want = {k: v for k, v in want.items() if v}
    rows = []
    for a in all_areas():
        p = area_paths(a)["dir"] / "panel_layouts.geojson"
        if not p.exists():
            continue
        got = {}
        for f in json.loads(p.read_text())["features"]:
            pr = f["properties"]
            if pr.get("kind") == "facet":
                got[pr["building_id"]] = got.get(pr["building_id"], 0) + 1
        for bid, w in want.items():
            if bid in got:
                rows.append((a, bid, w, got[bid]))
    if not rows:
        print("no labelled roofs found in any built region")
        return 1
    match = sum(1 for _, _, w, g in rows if w == g)
    err = sum(abs(w - g) for _, _, w, g in rows) / len(rows)
    print(f"\nBUILT OUTPUT vs JOSH'S MARKUP, {len(rows)} labelled roofs\n")
    print(f"  exact facet-count match : {match}/{len(rows)} ({100*match/len(rows):.0f}%)")
    print(f"  mean absolute error     : {err:.2f} facets")
    bad = sorted((r for r in rows if r[2] != r[3]), key=lambda r: -abs(r[2] - r[3]))
    if bad:
        print(f"\n  worst mismatches:")
        for a, bid, w, g in bad[:12]:
            print(f"    #{bid}  {a:22s} he drew {w:3d}  built {g:3d}")
    print("\n  Measured on the BUILT file, not by re-running segmentation: a fix")
    print("  that scores well in isolation and does not survive the build is")
    print("  not a fix, and that is the gap his markup fell through for weeks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

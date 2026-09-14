"""One small area, rebuilt and scored in minutes, so a change can be judged now.

Josh: "Is it possible to just do one small section? And keep iterating till its
perfect? Rebuilding is taking a long time and there are still many bad rooftops
in every build."

Yes. A district rebuild is 4.5 hours and answers one question per day, which is
why five regressions reached a deploy before anyone saw them. This builds a
FIXED set of a hundred-odd neighbouring roofs -- the real pipeline, geometry and
panels -- scores them, and prints the delta against the last run. Minutes, not
hours, and the set never changes, so two runs are always comparable.

WHAT IT SCORES, worst-to-best-understood:

  PANELS ACROSS A LINE   panels overlapping a ridge or valley Josh drew. This is
                         the actual visible failure -- the thing he points at in
                         a screenshot -- and no aggregate above it can see it.
  FACES EXACT            for roofs he has drawn, built facets that equal his
                         rings (IoU > FACE_MATCH_IOU). Deliberately strict: a face
                         clipped by 9% scores as a miss, which is how
                         drop_roof_features was caught silently carving 1,312 m2
                         off his markup while every softer metric passed.
  INVENTED EDGES         facet boundaries he never drew.
  FACETS / ROOF          fragmentation, the complaint that started this.

THE TRAP THIS TOOL IS BUILT AROUND. "Iterate until perfect on a small area" is
also the recipe for a model that has memorised that area. So the benchmark
roofs are PINNED AS VALIDATION and must never train the detector:

    python tools/export_training_data.py --val-ids data/bench_ids.txt

Written out by --make. Score the model on roofs it has never seen or the number
is a fiction that only collapses on the next district build.

Usage:
    python tools/bench.py --make --near 5371108 --n 120   # define the set once
    python tools/bench.py                                 # run and score
    python tools/bench.py --note "drawn-face exemption"    # label the run
"""

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# HOW CLOSE COUNTS AS "the face Josh drew".
#
# This was 0.999 for its first three runs and that was measuring serialisation,
# not geometry: preview_sample rounds facet rings to 2 dp for file size, and on
# a roof whose faces sit at 45 degrees that shifts each face's area by a few
# tenths of a percent. #5371108 scored 8/8 exact when compared against unrounded
# geometry and 1/8 through the bundle, which is how it was caught. The reported
# score jumped from 44% to 81% purely on the threshold.
#
# 0.99 is still strict where it matters: drop_roof_features was carving 9-30%
# off faces Josh drew, which scores 0.71-0.89 and fails cleanly.
FACE_MATCH_IOU = 0.99

SET_PATH = ROOT / "data" / "bench_set.json"
IDS_PATH = ROOT / "data" / "bench_ids.txt"
HIST_PATH = ROOT / "data" / "bench_history.json"


def make_set(seed_bid, n, include=()):
    """Pick the n buildings nearest a seed building, within one region.

    Seeded at the densest cluster of drawn roofs rather than anywhere pretty:
    the benchmark is only as good as the truth inside it, and a hundred roofs
    with five drawn ones cannot tell a real improvement from noise. `include`
    forces specific buildings in -- the ones Josh has pointed at, which are the
    cases the loop exists to fix and must never fall out of the set.
    """
    import geopandas as gpd
    from src.region_build import area_paths, all_areas

    for region in ["pilot"] + [x for x in all_areas() if x != "pilot"]:
        p = area_paths(region)
        dd = p["dir"] / "building_outlines_dedup.geojson"
        src = dd if dd.exists() else p["outlines"]
        if not src.exists():
            continue
        gdf = gpd.read_file(src).set_index("building_id", drop=False)
        if seed_bid not in gdf.index:
            continue
        c = gdf.loc[seed_bid].geometry.centroid
        gdf = gdf.assign(_d=gdf.geometry.centroid.distance(c))
        # Skip slivers: a 12 m2 shed is not a roof anyone judges the build on.
        gdf = gdf[gdf.geometry.area >= 40.0].sort_values("_d")
        ids = [int(x) for x in gdf["building_id"][:n]]
        for b in include:
            if b not in ids and b in gdf.index:
                ids.append(int(b))
        return region, sorted(ids)
    return None, []


def _panels_across_lines(roof):
    """Panels overlapping a line Josh drew, with a hand's width of tolerance."""
    from shapely.geometry import LineString, Polygon
    drawn = roof.get("drawn") or []
    if not drawn:
        return None
    segs = []
    for s in drawn:
        try:
            segs.append(LineString([(s[0], s[1]), (s[2], s[3])]).buffer(0.15))
        except Exception:
            pass
    if not segs:
        return None
    n = 0
    for ring in roof.get("panels") or []:
        try:
            p = Polygon(ring)
        except Exception:
            continue
        if any(p.intersects(s) for s in segs):
            n += 1
    return n


def score(rows, labels):
    """Turn built roofs into the numbers worth watching."""
    from shapely.geometry import Polygon

    out = {"roofs": 0, "facets": 0, "panels": 0,
           "labelled_roofs": 0, "faces_exact": 0, "faces_total": 0,
           "panels_across": 0, "panels_on_labelled": 0, "errors": 0,
           # Josh: "How can you make sure you are not drawing any extra lines
           # compared to what I've now marked on those exact rooftops?"
           # faces_exact answers "did it reproduce what he drew"; it does not
           # answer "and nothing else". A roof can reproduce all 8 of his faces
           # and add a 9th, and score 8/8.
           "extra": 0, "missing": 0, "mismatched_roofs": []}
    for r in rows:
        if r.get("error"):
            out["errors"] += 1
            continue
        out["roofs"] += 1
        out["facets"] += len(r.get("facets") or [])
        out["panels"] += len(r.get("panels") or [])

        lab = labels.get(str(r["id"]))
        if not lab or not lab.get("complete"):
            continue
        drawn = []
        for f in lab.get("faces") or []:
            if not f.get("usable", True):
                continue
            try:
                drawn.append(Polygon([(p[0], p[1]) for p in f["ring"]]))
            except Exception:
                pass
        if not drawn:
            continue
        out["labelled_roofs"] += 1
        r_ex = 0
        for f in r.get("facets") or []:
            try:
                P = Polygon(f["ring"])
            except Exception:
                continue
            out["faces_total"] += 1
            best = 0.0
            for d in drawn:
                try:
                    u = P.union(d).area
                    if u > 0:
                        best = max(best, P.intersection(d).area / u)
                except Exception:
                    pass
            if best > FACE_MATCH_IOU:
                out["faces_exact"] += 1
                r_ex += 1
        built_n = len(r.get("facets") or [])
        d_extra = max(0, built_n - len(drawn))
        d_missing = max(0, len(drawn) - built_n)
        out["extra"] += d_extra
        out["missing"] += d_missing
        if d_extra or d_missing or r_ex < built_n:
            out["mismatched_roofs"].append(
                {"id": r["id"], "drew": len(drawn), "built": built_n,
                 "exact": r_ex})
        ac = _panels_across_lines(r)
        if ac is not None:
            out["panels_across"] += ac
            out["panels_on_labelled"] += len(r.get("panels") or [])
    return out


def show(cur, prev, note):
    def line(label, val, pv, fmt="{:.1f}", lower_better=True, suffix=""):
        s = fmt.format(val) + suffix
        if pv is None:
            print(f"  {label:26s} {s:>10s}")
            return
        d = val - pv
        if abs(d) < 1e-9:
            print(f"  {label:26s} {s:>10s}      --")
            return
        good = (d < 0) if lower_better else (d > 0)
        arrow = "better" if good else "WORSE"
        print(f"  {label:26s} {s:>10s}   {d:+.1f}{suffix}  {arrow}")

    print(f"\n  benchmark: {cur['roofs']} roofs"
          + (f"  ({note})" if note else ""))
    if cur["errors"]:
        print(f"  {cur['errors']} roofs failed to build")
    print()
    fpr = cur["facets"] / max(cur["roofs"], 1)
    pfpr = (prev["facets"] / max(prev["roofs"], 1)) if prev else None
    line("facets per roof", fpr, pfpr)
    line("panels", cur["panels"], prev["panels"] if prev else None,
         fmt="{:.0f}", lower_better=False)

    if cur["faces_total"]:
        pct = 100 * cur["faces_exact"] / cur["faces_total"]
        ppct = (100 * prev["faces_exact"] / prev["faces_total"]
                if prev and prev.get("faces_total") else None)
        line("faces matching markup", pct, ppct, lower_better=False, suffix="%")
        print(f"  {'':26s} {cur['faces_exact']}/{cur['faces_total']} on "
              f"{cur['labelled_roofs']} drawn roofs")
    if cur.get("extra") or cur.get("missing"):
        line("extra facets vs markup", cur["extra"],
             prev.get("extra") if prev else None, fmt="{:.0f}")
        line("missing facets vs markup", cur["missing"],
             prev.get("missing") if prev else None, fmt="{:.0f}")
    if cur["panels_on_labelled"]:
        pct = 100 * cur["panels_across"] / cur["panels_on_labelled"]
        ppct = (100 * prev["panels_across"] / prev["panels_on_labelled"]
                if prev and prev.get("panels_on_labelled") else None)
        line("panels across a line", pct, ppct, suffix="%")
        print(f"  {'':26s} {cur['panels_across']} of "
              f"{cur['panels_on_labelled']} panels on drawn roofs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true", help="define the set")
    ap.add_argument("--near", type=int, default=5371108,
                    help="seed building the set is grown around")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--include", nargs="*", type=int,
                    default=[5371107, 5371108],
                    help="buildings that must be in the set whatever the radius")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--note", default="", help="what this run is testing")
    ap.add_argument("--out", default="bench.html")
    a = ap.parse_args()

    if a.make:
        region, ids = make_set(a.near, a.n, a.include)
        if not ids:
            print(f"#{a.near} not found in any region")
            return 1
        SET_PATH.parent.mkdir(parents=True, exist_ok=True)
        SET_PATH.write_text(json.dumps(
            {"region": region, "seed": a.near, "ids": ids}, indent=1))
        IDS_PATH.write_text("\n".join(str(i) for i in ids) + "\n")
        labels = json.loads((ROOT / "data" / "roof_labels.json").read_text())["buildings"]
        n_lab = sum(1 for i in ids if labels.get(str(i), {}).get("complete"))
        print(f"benchmark set: {len(ids)} roofs in {region}, "
              f"{n_lab} of them drawn -> {SET_PATH}")
        print(f"ids for the training holdout -> {IDS_PATH}")
        print("\nPin them out of training before retraining the detector:")
        print("  python tools/export_training_data.py --val-ids data/bench_ids.txt")
        return 0

    if not SET_PATH.exists():
        print(f"no benchmark set. Make one:\n"
              f"  python tools/bench.py --make --near {a.near} --n {a.n}")
        return 1
    spec = json.loads(SET_PATH.read_text())
    region, ids = spec["region"], spec["ids"]
    labels = json.loads((ROOT / "data" / "roof_labels.json").read_text())["buildings"]

    import multiprocessing
    from preview_sample import _init, _one, PAGE, OUT_DIR
    jobs = a.jobs or max(1, min(8, (__import__("os").cpu_count() or 2) - 1))
    print(f"building {len(ids)} roofs from {region} on {jobs} workers...")
    t0 = time.time()
    rows = []
    ctxm = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init,
                             initargs=(region,), mp_context=ctxm) as ex:
        for r in ex.map(_one, ids):
            if r:
                rows.append(r)
    secs = time.time() - t0

    cur = score(rows, labels)
    hist = json.loads(HIST_PATH.read_text()) if HIST_PATH.exists() else []
    prev = hist[-1]["score"] if hist else None
    show(cur, prev, a.note)

    bad = cur.get("mismatched_roofs") or []
    if bad:
        print(f"\n  roofs not matching your markup ({len(bad)}):")
        for b in sorted(bad, key=lambda t: -(abs(t["built"] - t["drew"]))) [:15]:
            d = b["built"] - b["drew"]
            tag = (f"{d:+d} facets" if d else f"{b['exact']}/{b['built']} exact")
            print(f"    #{b['id']}  you drew {b['drew']}, built {b['built']}   {tag}")

    hist.append({"when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "note": a.note, "secs": round(secs, 1), "score": cur})
    HIST_PATH.write_text(json.dumps(hist, indent=1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / a.out
    sub = (f"{len(rows)} roofs from {region} &middot; benchmark set &middot; "
           f"built with the current working tree")
    dest.write_text(PAGE.replace("__ROOFS__", json.dumps(rows, separators=(",", ":")))
                        .replace("__SUB__", sub))
    print(f"\n  {secs/60:.1f} min   {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

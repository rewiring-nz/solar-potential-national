"""
Take label files back in, check them, merge them, and update what the tool shows
as already collected.

This is the other half of tools/label_template.html. Without it the round trip
is manual -- drop a file in data/, remember to hand-edit label_progress.json --
and the step everyone forgets is the progress file, which is exactly the one
that stops two people marking the same roof next week.

WHY IT VALIDATES RATHER THAN JUST COPYING. Labels are training and scoring
truth: a bad one is worse than a missing one, because it silently moves the
number a model is measured against. Every check here exists because the failure
it catches is silent -- coordinates in the wrong CRS still parse as JSON, a line
kind that was renamed still loads, a self-intersecting ring still has an area.

MERGING. One document per building, last writer wins by saved_utc. Two people
marking the same roof is not an error worth refusing over -- it is a duplicate,
and the newer one is kept and reported. Anything genuinely suspect is reported
and SKIPPED, never silently repaired.

Usage:
    python tools/ingest_labels.py roof_labels.json
    python tools/ingest_labels.py ~/Downloads/*.json
    python tools/ingest_labels.py new.json --dry-run
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
LABELS = DATA_DIR / "roof_labels.json"
PROGRESS = ROOT / "label_progress.json"          # served beside mark_roofs.html
BUNDLE = DATA_DIR / "label_set" / "mark_roofs.html"

VALID_KINDS = {"ridge", "valley", "cliff"}
# A roof can be wrong in ways geometry cannot express. "absent" feeds
# config.DEMOLISHED_BUILDING_IDS, which the build already honours; the other two
# are outline-quality findings that need LINZ to catch up, not a code change.
VALID_PROBLEMS = {"absent", "not_building", "bad_outline", "unclear"}
# A crop is the footprint plus a 4 m pad, so anything outside it by more than a
# little cannot be a mark on that roof -- most likely the wrong CRS or the wrong
# building.
OUTSIDE_TOLERANCE_M = 2.0


def bundle_roofs():
    """id -> {area, bounds} straight from the bundle people are marking, so the
    checks are against what they were actually shown."""
    if not BUNDLE.exists():
        return {}
    m = re.search(r"const ROOFS = (\[.*?\]);\n", BUNDLE.read_text(), re.S)
    if not m:
        return {}
    return {str(r["id"]): {"area": r["area"], "bounds": r["bounds"]}
            for r in json.loads(m.group(1))}


def ring_area(ring):
    s = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i][:2]
        x2, y2 = ring[(i + 1) % len(ring)][:2]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def check_building(bid, rec, known):
    """Everything wrong with one building's labels. Empty list means clean."""
    problems = []
    ref = known.get(str(bid))
    if ref is None:
        problems.append("not one of the roofs in the bundle")
        return problems
    minx, miny, maxx, maxy = ref["bounds"]
    t = OUTSIDE_TOLERANCE_M

    def outside(p):
        return not (minx - t <= p[0] <= maxx + t and miny - t <= p[1] <= maxy + t)

    if rec.get("area") and rec["area"] != ref["area"]:
        problems.append(f"area says {rec['area']}, bundle says {ref['area']}")

    lines = rec.get("lines") or []
    for i, l in enumerate(lines):
        if l.get("kind") not in VALID_KINDS:
            problems.append(f"line {i}: unknown kind {l.get('kind')!r}")
        a, b = l.get("a"), l.get("b")
        if not (a and b):
            problems.append(f"line {i}: missing an endpoint")
            continue
        if outside(a) or outside(b):
            problems.append(f"line {i}: endpoint outside this roof's crop "
                            "(wrong CRS, or the wrong building?)")
        if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < 0.09:      # 0.3 m
            problems.append(f"line {i}: zero length")

    for i, o in enumerate(rec.get("obstructions") or []):
        ring = o.get("ring")
        if not ring:
            problems.append(f"obstruction {i}: no ring "
                            "(exported by a version older than 1 Sep 2026?)")
            continue
        if len(ring) < 3:
            problems.append(f"obstruction {i}: ring has {len(ring)} points")
        elif ring_area(ring) <= 0.01:
            problems.append(f"obstruction {i}: zero area")
        if any(outside(p) for p in ring):
            problems.append(f"obstruction {i}: outside this roof's crop")

    for i, p in enumerate(rec.get("nopanel") or []):
        if outside(p):
            problems.append(f"no-panel tag {i}: outside this roof's crop")

    prob = rec.get("problem")
    if prob is not None and prob not in VALID_PROBLEMS:
        problems.append(f"unknown problem flag {prob!r}")
    if prob and lines:
        # not fatal, but worth saying: geometry drawn on something the labeller
        # then said is not a roof is geometry nobody should train on
        problems.append(f"flagged {prob} but also carries {len(lines)} drawn lines")
    if not prob and not (lines or rec.get("obstructions") or rec.get("nopanel")):
        problems.append("nothing marked on it")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="roof_labels.json files to take in")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="merge buildings that failed their checks anyway")
    a = ap.parse_args()

    known = bundle_roofs()
    if not known:
        print(f"cannot read the bundle at {BUNDLE} -- checks would be blind")
        return 2

    merged = {}
    if LABELS.exists():
        try:
            merged = json.loads(LABELS.read_text()).get("buildings", {})
        except json.JSONDecodeError:
            print(f"existing {LABELS.name} is not valid JSON; refusing to clobber it")
            return 2
    print(f"already held: {len(merged)} roofs\n")

    added = replaced = skipped = 0
    for f in a.files:
        p = Path(f)
        if not p.exists():
            print(f"{p}: no such file")
            continue
        try:
            doc = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"{p.name}: not valid JSON ({e})")
            continue
        buildings = doc.get("buildings", {})
        print(f"{p.name}: {len(buildings)} roofs, saved {doc.get('saved_utc', '?')}")

        for bid, rec in buildings.items():
            problems = check_building(bid, rec, known)
            if problems and not a.force:
                skipped += 1
                print(f"   SKIP #{bid}: " + "; ".join(problems[:3])
                      + (f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""))
                continue
            if problems:
                print(f"   forced #{bid} despite: {problems[0]}")
            rec = dict(rec)
            rec["source_file"] = p.name
            rec["saved_utc"] = doc.get("saved_utc", "")
            if bid in merged:
                # two people on the same roof: keep the newer, say so
                if rec["saved_utc"] >= merged[bid].get("saved_utc", ""):
                    merged[bid] = rec
                    replaced += 1
                    print(f"   #{bid} already held -- kept this newer one")
                else:
                    print(f"   #{bid} already held by a newer file -- kept that")
            else:
                merged[bid] = rec
                added += 1

    flagged = {}
    for bid, b in merged.items():
        if b.get("problem"):
            flagged.setdefault(b["problem"], []).append(bid)
    if flagged:
        print("\nROOFS FLAGGED AS NOT REALLY ROOFS:")
        for prob in sorted(flagged):
            ids = sorted(flagged[prob])
            print(f"  {prob:<14} {len(ids)}")
            print(f"      {', '.join(ids[:10])}" + (" ..." if len(ids) > 10 else ""))
        if flagged.get("unclear"):
            print("\n  'unclear' means the imagery was too poor to mark. Those roofs")
            print("  are excluded from scoring -- a guessed line would become truth.")
        absent = sorted(flagged.get("absent", []))
        if absent:
            print("\n  'absent' means nothing is there. config.py already excludes")
            print("  such buildings from every build -- add them:")
            print(f"    DEMOLISHED_BUILDING_IDS = {{{', '.join(absent)}}}")

    total_lines = sum(len(b.get("lines") or []) for b in merged.values())
    total_obs = sum(len(b.get("obstructions") or []) for b in merged.values())
    print(f"\n{added} new, {replaced} replaced, {skipped} skipped")
    print(f"holding {len(merged)} roofs / {total_lines} lines / {total_obs} obstructions"
          f"  ({len(merged) / len(known):.0%} of the {len(known)} in the bundle)")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    LABELS.parent.mkdir(parents=True, exist_ok=True)
    LABELS.write_text(json.dumps(
        {"tool": "ingest_labels", "buildings": merged}, indent=1))
    print(f"\nwrote {LABELS}")

    # The bit that is always forgotten by hand: the tool reads this to grey out
    # roofs already collected, so people stop re-marking them.
    from datetime import date
    PROGRESS.write_text(json.dumps({
        "updated": date.today().isoformat(),
        "done": sorted(merged.keys()),
        "note": "Roof ids already collected. Refreshed by tools/ingest_labels.py.",
    }, indent=1))
    print(f"wrote {PROGRESS} ({len(merged)} roofs will show as collected)")
    print("\nNext:  python tools/score_geometry.py")
    print("Then commit label_progress.json so the shared tool picks it up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

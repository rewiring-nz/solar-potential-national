"""Every flagged roof, and whether it is fixed RIGHT NOW.

THE PROBLEM THIS SOLVES. Flagged examples did not reliably get fully fixed;
there has to be a tracked way to make sure things are improving. The
failures were structural, not bad luck. In one day:

  * #4735292 was called fixed on the strength of a SCORE. Nobody looked at
    the roof. It was still wrong.
  * The fix was then real, but a district driver skipped its region in
    silence, so it never reached the map -- twice. The same roof came back
    unchanged, twice.
  * #4722059 re-acquired the exact pathological reading that had already
    been fixed once, and shipped dark.

Nothing in the system knew about any of it, because the examples lived only
as screenshots in a chat. This makes them first-class objects with a status
that is recomputed from the data every time.

WHAT A CASE IS. One flagged roof, the note that came with it, and a
fingerprint of the geometry it was flagged for. A case is never
deleted: a roof that gets fixed and then breaks again must surface as a
REGRESSION rather than quietly disappear.

THE LOOP, and note where the reviewer's time goes -- only the last step:

    1. A roof is flagged.            tools/cases.py add <id> "the note"
    2. Every build re-measures it.   tools/cases.py check
    3. Changed roofs get rendered.   -> data/preview/cases.html
    4. The reviewer rules on it.     tools/cases.py verdict <id> fixed|wrong

The reviewer is only ever shown roofs whose geometry CHANGED since the last
ruling, before and after, side by side. A roof already passed that has not
moved never appears again. That is the whole point: review attention is the
scarce input, so it is spent only where it can change a decision.

WHAT IT MEASURES per case, from the data rather than from an opinion:
    facets, coverage of the footprint, panels, and -- where the roof has been
    drawn -- agreement with the drawn faces and panels crossing drawn lines.
The fingerprint is the geometry itself, so "unchanged" is a fact.

    python tools/cases.py check
    python tools/cases.py check --render
    python tools/cases.py verdict 4735292 fixed
    python tools/cases.py add 4712345 "roof lines make no sense here"
    python tools/cases.py live        # is the fix actually on the map?
"""

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REG = ROOT / "data" / "roof_cases.json"
LIVE = "https://rewiring-nz.github.io/nz-solar-potential/data/solar_potential.geojson"


def load():
    return json.loads(REG.read_text())


def save(d):
    REG.write_text(json.dumps(d, indent=1))


def _find_region(bid):
    import geopandas as gpd
    from src.region_build import area_paths
    import config
    d = ROOT / "data/regions"
    names = sorted({p.name for p in d.iterdir() if p.is_dir()} | set(config.REGIONS)) \
        if d.exists() else list(config.REGIONS)
    for r in names:
        op = area_paths(r)["outlines"]
        if not Path(op).exists():
            continue
        try:
            if bid in set(int(x) for x in gpd.read_file(op)["building_id"]):
                return r
        except Exception:
            continue
    return None


def measure(case, ctx):
    """Rebuild this roof and measure it. Returns a dict, or an error dict."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from src.region_build import area_paths, area_centroid_wgs84
    from src.solar_model import SolarModel
    import src.build_layout_geojson as blg

    bid, region = case["id"], case.get("region")
    if not region:
        region = _find_region(bid)
        case["region"] = region
    if not region:
        return {"error": "building not in any region on this machine"}
    if region not in ctx:
        p = area_paths(region)
        dd = p["dir"] / "building_outlines_dedup.geojson"
        if not p["imagery"].exists():
            return {"error": f"no imagery for {region} on this machine"}
        gdf = gpd.read_file(dd if dd.exists() else p["outlines"]).set_index(
            "building_id", drop=False)
        c = area_centroid_wgs84(region)
        blg._init_worker(region, SolarModel(*c) if c else SolarModel())
        ctx[region] = gdf
    gdf = ctx[region]
    if bid not in gdf.index:
        return {"error": "not in region outlines"}
    geom = gdf.loc[bid].geometry
    try:
        feats = blg._build_one(bid)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    import pyproj
    from shapely.ops import transform as sht
    from shapely.geometry import shape as shp
    to_nztm = pyproj.Transformer.from_crs(4326, 2193, always_xy=True).transform
    facets = [sht(to_nztm, shp(f["geometry"])) for f in feats
              if f["properties"]["kind"] == "facet"]
    panels = [sht(to_nztm, shp(f["geometry"])) for f in feats
              if f["properties"]["kind"] == "panel"]
    cov = (unary_union(facets).intersection(geom).area / geom.area
           if facets else 0.0)
    m = {"facets": len(facets), "panels": len(panels),
         "coverage": round(cov, 3),
         "fingerprint": hashlib.md5(json.dumps(
             sorted([[(round(x, 2), round(y, 2))
                      for x, y in f.exterior.coords] for f in facets]),
             sort_keys=True).encode()).hexdigest()[:12]}

    # where the roof was drawn, measure against the DRAWN faces, not against ourselves
    labs = json.loads((ROOT / "data/roof_labels.json").read_text())["buildings"]
    lab = labs.get(str(bid))
    if lab and lab.get("faces"):
        drawn = [Polygon([(x, y) for x, y in f["ring"]])
                 for f in lab["faces"] if f.get("ring") and len(f["ring"]) >= 3]
        drawn = [d for d in drawn if d.is_valid and d.area > 1.0]
        if drawn and facets:
            tot = matched = 0.0
            for d in drawn:
                best = max((f.intersection(d).area / f.union(d).area
                            for f in facets), default=0.0)
                tot += best
                matched += 1.0 if best >= 0.5 else 0.0
            m["agree"] = round(tot / len(drawn), 3)
            m["matched"] = round(matched / len(drawn), 3)
            m["drew"] = len(drawn)
        segs = []
        for l in lab.get("lines") or []:
            pts = l.get("points") or ([l.get("a"), l.get("b")]
                                      if l.get("a") else None)
            if pts:
                from shapely.geometry import LineString
                for i in range(len(pts) - 1):
                    segs.append(LineString([tuple(pts[i]),
                                            tuple(pts[i + 1])]).buffer(0.15))
        if segs and panels:
            m["across"] = sum(1 for p in panels
                              if any(p.intersects(s) for s in segs))
        # THE DRAWN LINES, ONE BY ONE. Every verdict of 18 Sep was
        # counted in lines, not faces: "missing two valley lines and a
        # ridgeline that I clearly drew", "missing one valley line",
        # "missing two ridge lines that I drew". So the number tracked has
        # to be the same one the reviewer counts -- how many of the lines
        # drew exist as a facet boundary, and how much boundary we drew
        # that were not drawn. Face-level agreement averages both away.
        raw = []
        for l in lab.get("lines") or []:
            pts = l.get("points") or ([l.get("a"), l.get("b")]
                                      if l.get("a") else None)
            if not pts:
                continue
            from shapely.geometry import LineString
            for i in range(len(pts) - 1):
                seg = LineString([tuple(pts[i]), tuple(pts[i + 1])])
                if seg.length > 0.5:
                    raw.append((seg, l.get("kind")))
        # AN OPEN-ENDED LINE IS NOT THE SAME KIND OF MISS.
        #
        # The labelling tool derives faces from a planar subdivision, so a line with a
        # free end bounds no region and never becomes a facet boundary.
        # Measured across the three "missing lines" roofs that accounts for
        # every miss and nothing else. Counting those the same as a line we
        # simply failed to find leaves roofs sitting in the queue for
        # something deliberate -- a line is not always meant to go all the
        # way to an edge -- so they are counted apart.
        # They are still honoured: panels are kept off them, and the map's
        # markup layer draws them.
        whole = []
        for l in lab.get("lines") or []:
            pts = l.get("points") or ([l.get("a"), l.get("b")]
                                      if l.get("a") else None)
            if pts and len(pts) >= 2:
                from shapely.geometry import LineString
                whole.append(LineString([tuple(q) for q in pts]))
        open_ends = None
        if whole:
            from shapely.geometry import Point
            from shapely.ops import unary_union
            fp = geom.exterior
            open_ends = []
            for i, ls in enumerate(whole):
                rest = (unary_union([o for j, o in enumerate(whole) if j != i])
                        if len(whole) > 1 else None)
                for e in (Point(ls.coords[0]), Point(ls.coords[-1])):
                    d = fp.distance(e)
                    if rest is not None:
                        d = min(d, rest.distance(e))
                    if d > 0.6:
                        open_ends.append(ls)
                        break
            open_ends = unary_union(open_ends) if open_ends else None

        if raw and facets:
            from shapely.ops import unary_union
            edges = unary_union([f.exterior for f in facets])
            found = 0
            missed = []
            hanging = 0
            for seg, kind in raw:
                mid = seg.interpolate(0.5, normalized=True)
                q1 = seg.interpolate(0.25, normalized=True)
                q3 = seg.interpolate(0.75, normalized=True)
                if max(edges.distance(q1), edges.distance(mid),
                       edges.distance(q3)) < 1.0:
                    found += 1
                elif open_ends is not None and open_ends.distance(mid) < 0.2:
                    hanging += 1
                else:
                    missed.append(kind or "line")
            m["lines_drawn"] = len(raw)
            m["lines_found"] = found
            m["lines_open"] = hanging
            if missed:
                from collections import Counter
                m["missing"] = dict(Counter(missed))
    return m


def cmd_check(a):
    d = load()
    ctx = {}
    rows = []
    for c in d["cases"]:
        m = measure(c, ctx)
        hist = c.setdefault("history", [])
        prev = hist[-1] if hist else None
        changed = (prev or {}).get("fingerprint") != m.get("fingerprint")
        if m.get("error"):
            rows.append((c, m, "ERROR", False))
            continue
        # a roof PASSED that has since moved is a regression until the
        # reviewer says otherwise -- this is the check that was missing when
        # #4722059 quietly re-acquired its bad reading
        # A REGRESSION THAT UNDOES ITSELF CLEARS ITSELF. The fingerprint
        # approved is recorded with the verdict, so a roof that comes
        # back to exactly that geometry is fixed again and must not sit in
        # the queue asking to be re-judged -- review attention is the scarce
        # input. Anything else stays flagged until it is looked at.
        approved = next((v.get("fingerprint") for v in
                         reversed(c.get("verdicts") or [])
                         if v.get("verdict") == "fixed"), None)
        if c["status"] == "regressed" and m.get("fingerprint") == approved:
            c["status"] = "fixed"
            changed = False
        elif c["status"] == "fixed" and changed:
            c["status"] = "regressed"
        elif c["status"] in ("open", "wrong") and changed:
            c["status"] = "needs_verdict"
        if changed:
            m["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            hist.append(m)
        rows.append((c, m, c["status"], changed))
    save(d)

    order = {"regressed": 0, "needs_verdict": 1, "open": 2, "wrong": 2,
             "ERROR": 3, "fixed": 4}
    rows.sort(key=lambda r: order.get(r[2], 5))
    print(f"\n{'roof':>9} {'status':13s} {'facets':>6} {'cover':>6} "
          f"{'panels':>6} {'your lines':>11} {'open':>5} {'across':>6}  defect")
    for c, m, st, ch in rows:
        if m.get("error"):
            print(f"{c['id']:>9} {'ERROR':13s} {m['error'][:54]}")
            continue
        print(f"{c['id']:>9} {st:13s} {m['facets']:>6} "
              f"{m['coverage']*100:>5.0f}% {m['panels']:>6} "
              f"{((str(m['lines_found'])+'/'+str(m['lines_drawn'])) if 'lines_drawn' in m else '-'):>11} "
              f"{(str(m.get('lines_open', 0)) if 'lines_drawn' in m else '-'):>5} "
              f"{(str(m['across']) if 'across' in m else '-'):>6}"
              f"  {c['defect'][:44]}{'  <-- CHANGED' if ch else ''}")
    n = {k: sum(1 for c in d["cases"] if c["status"] == k)
         for k in ("fixed", "regressed", "needs_verdict", "open", "wrong")}
    total = len(d["cases"])
    print(f"\n  {n['fixed']}/{total} fixed and confirmed"
          f"   {n['regressed']} REGRESSED"
          f"   {n['needs_verdict']} awaiting a verdict"
          f"   {n['open'] + n['wrong']} still open")
    if a.render:
        render([r for r in rows if r[3] or r[2] in ("regressed", "open",
                                                    "wrong")], ctx)


def render(rows, ctx):
    """One page: only the roofs whose geometry moved, or still wrong."""
    if not rows:
        print("  nothing changed -- no page written")
        return
    import subprocess
    ids = ",".join(str(c["id"]) for c, _, _, _ in rows)
    out = ROOT / "data/preview/cases.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"  rendering {len(rows)} roofs -> {out}")
    subprocess.run([sys.executable, str(ROOT / "tools/preview_sample.py"),
                    "--ids", *[str(c["id"]) for c, _, _, _ in rows],
                    "--out", str(ROOT / "data/preview/cases.html")])


def cmd_verdict(a):
    d = load()
    for c in d["cases"]:
        if c["id"] == a.id:
            c["status"] = "fixed" if a.verdict == "fixed" else "wrong"
            c.setdefault("verdicts", []).append(
                {"at": time.strftime("%Y-%m-%d"), "verdict": a.verdict,
                 "fingerprint": (c.get("history") or [{}])[-1].get("fingerprint")})
            if a.note:
                c.setdefault("quotes", []).append(a.note)
            save(d)
            print(f"#{a.id}: {a.verdict}")
            return 0
    print(f"no case for #{a.id}")
    return 1


def cmd_add(a):
    d = load()
    if any(c["id"] == a.id for c in d["cases"]):
        for c in d["cases"]:
            if c["id"] == a.id:
                c.setdefault("quotes", []).append(a.quote)
                c["flagged_times"] = c.get("flagged_times", 1) + 1
                c["status"] = "wrong"
        save(d)
        print(f"#{a.id}: flagged again ({c['flagged_times']}x)")
        return 0
    d["cases"].append({"id": a.id, "region": _find_region(a.id),
                       "first_flagged": time.strftime("%Y-%m-%d"),
                       "defect": a.quote[:90], "quotes": [a.quote],
                       "status": "open", "flagged_times": 1})
    save(d)
    print(f"#{a.id}: case opened")
    return 0


def cmd_live(a):
    """Is the fix actually ON THE MAP? The pilot-skip class of failure."""
    import urllib.request
    d = load()
    print("fetching live build...")
    live = json.loads(urllib.request.urlopen(LIVE).read())
    lp = {str(f["properties"].get("building_id")): f["properties"].get("panel_count")
          for f in live["features"]}
    ctx = {}
    bad = 0
    for c in d["cases"]:
        m = measure(c, ctx)
        if m.get("error"):
            continue
        onmap = lp.get(str(c["id"]))
        same = onmap == m["panels"]
        if not same:
            bad += 1
        print(f"{c['id']:>9}  local {m['panels']:>5} panels   "
              f"live {str(onmap):>5}   {'ok' if same else 'NOT ON THE MAP YET'}")
    print(f"\n  {bad} case roofs differ from the live map")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--render", action="store_true")
    c.set_defaults(fn=cmd_check)
    v = sub.add_parser("verdict"); v.add_argument("id", type=int)
    v.add_argument("verdict", choices=["fixed", "wrong"])
    v.add_argument("--note", default=None); v.set_defaults(fn=cmd_verdict)
    ad = sub.add_parser("add"); ad.add_argument("id", type=int)
    ad.add_argument("quote"); ad.set_defaults(fn=cmd_add)
    lv = sub.add_parser("live"); lv.set_defaults(fn=cmd_live)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main() or 0)

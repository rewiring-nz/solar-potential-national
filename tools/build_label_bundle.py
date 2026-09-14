"""
Build ONE self-contained HTML file for marking up roofs. No server, no Python.

Josh: "You should make it possible to open the tool on a standard computer,
maybe as an HTML file? And then draw the lines on the buildings, then save the
file to be uploaded to you."

So this bakes the imagery, the building outline and the neighbouring outlines
into a single .html that opens by double-clicking it, anywhere, offline. Marking
up writes to the browser's local storage as you go, and a Download button saves
one JSON to send back. Nothing needs installing and nothing phones home.

WHY EVERYTHING IS EMBEDDED. A folder of images plus an HTML file is one careless
zip away from a tool that opens to blank squares. One file cannot lose its
images. The cost is size -- roughly 40-60 KB per roof -- so a 150-roof bundle
lands around 8 MB, which is fine to open locally and fine to email.

NEIGHBOURING OUTLINES matter more than they sound. Josh: "provide the building
outline, so it's clear on busy rooftops where they stop and where is the next
building." On a terrace or a dense commercial block the roof under the cursor
runs straight into its neighbour, and a line drawn across that join is a wrong
label that would teach a model the wrong thing. The target building is drawn
solid, every neighbour dashed and dimmed.

Usage:
    python tools/build_label_bundle.py                    # from the sampled queue
    python tools/build_label_bundle.py --ids 5371108 4735015
    python tools/build_label_bundle.py --max 40 --out roofs_batch1.html
"""

import argparse
import base64
import io
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "label_set"
PAD_M = 4.0
MAX_PX = 1400         # per-roof crop. Only a handful of roofs are this big;
                      # the cap exists to stop one warehouse dominating the
                      # bundle, not to downscale ordinary houses.


def crop(imagery, bounds):
    import numpy as np
    import rasterio.windows
    from PIL import Image, ImageFilter
    minx, miny, maxx, maxy = bounds
    w = rasterio.windows.from_bounds(minx, miny, maxx, maxy, imagery.transform)
    rgb = np.moveaxis(imagery.read([1, 2, 3], window=w,
                                   boundless=True, fill_value=0), 0, -1)
    im = Image.fromarray(rgb.astype("uint8"))
    # Source is 0.1 m/px and the tool always views it upscaled, so roof creases
    # sit right at the resolution limit. A mild unsharp mask does not invent
    # detail but it does make the edges that ARE there easier to trace against.
    # Applied before the resize below so it works on real pixels, not resampled
    # ones.
    im = im.filter(ImageFilter.UnsharpMask(radius=1.4, percent=115, threshold=3))
    if max(im.size) > MAX_PX:
        s = MAX_PX / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                       Image.LANCZOS)
    buf = io.BytesIO()
    # Crops are tiny (median 291 px) and get fitted up ~3-4x on screen, so
    # compression artefacts are magnified along with everything else. On images
    # this small the extra quality costs very little.
    im.save(buf, format="JPEG", quality=95, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--artifact", action="store_true",
                    help="strip the document wrapper so the file can be "
                         "published as an Artifact, which supplies its own")
    ap.add_argument("--bench-n", type=int, default=0, dest="bench_n",
                    help="how many unmarked roofs to include (0 = all). The "
                         "roofs already drawn always come too, for context")
    ap.add_argument("--bench", action="store_true",
                    help="the benchmark cluster: the roofs already marked plus "
                         "every unmarked one around them, opened as a work queue")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--out", default="mark_roofs.html")
    a = ap.parse_args()

    import geopandas as gpd
    import pyproj
    import rasterio
    from src.region_build import area_paths, all_areas

    # Josh: "some rooftops I need to check the 3D shape". The imagery is a
    # single orthophoto, so a dormer and a flat vent can look identical from
    # straight above. A lat/lon per roof lets the tool link straight out to
    # Google Earth's 3D mesh, which settles it in seconds.
    to_wgs = pyproj.Transformer.from_crs("EPSG:2193", "EPSG:4326",
                                         always_xy=True).transform

    ids = a.ids
    why = {}
    # THE BENCHMARK CLUSTER AS A WORK QUEUE. Josh: "only shows the marked
    # rooftops and then the ones in the area you want me to do. So I can just
    # press next unmarked".
    #
    # Marking scattered roofs across the district trains the detector well and
    # leaves the scoreboard thin -- tools/bench.py scores 152 roofs of which
    # only 11 are drawn, which can see a large regression and not a small one.
    # The same effort spent inside one cluster turns it into dense ground truth.
    # The marked roofs come along so their geometry is visible for context; the
    # bundle opens on the first unmarked one.
    if a.bench and ids is None:
        bp = DATA_DIR / "bench_ids.txt"
        if not bp.exists():
            print("no data/bench_ids.txt -- run tools/bench.py --make first")
            return 2
        ids = [int(x) for x in bp.read_text().split() if x.strip()]
        # RIGHT-SIZE THE ASK. Josh: "Does it need to be 152? That is a lot and I
        # imagine diminishing returns?" It does not, and the returns were
        # measured rather than guessed: bootstrapping the score over his 84
        # drawn roofs, the 95% interval is +/-6.2 points at 10 roofs, +/-4.3 at
        # 25 and +/-2.3 at 84. The knee is around 25-30; past that a day of
        # drawing buys well under a point.
        #
        # The other roofs stay in the BENCHMARK -- they are built and scored
        # automatically for fragmentation and panel counts, which costs him
        # nothing. Only the drawn subset needs to grow, so only it is bundled.
        #
        # SIMPLEST FIRST, because the scarce resource is Josh's time, not the
        # roof count. Josh: "Why don't we fix simple roofs first which are
        # faster for me to draw?" -- and the data agrees more strongly than the
        # argument did. Across his 84 drawn roofs:
        #
        #   simple (2-3 faces)   24 roofs   93.9% of faces exact    4.4 lines/roof
        #   medium (4-6)         31         95.9%                  10.6
        #   complex (7+)         29         96.7%                  33.1
        #
        # Simple roofs are the WORST band, so they are not already solved, and
        # they cost a fifth of the drawing. Per line he draws they return about
        # 1.1 faces of signal against 0.16 for a complex roof -- roughly seven
        # times the value of the same hour. Ordering by complexity descending,
        # which is what this did first, optimised faces per ROOF and ignored
        # that a complex roof is seven roofs' worth of work.
        if a.bench_n:
            _lab = json.loads((DATA_DIR / "roof_labels.json").read_text())["buildings"] \
                if (DATA_DIR / "roof_labels.json").exists() else {}
            done = [i for i in ids if _lab.get(str(i), {}).get("complete")]
            todo = [i for i in ids if i not in set(done)]

            def _weight(bid):
                vp = DATA_DIR / "vision_lines" / f"{bid}.json"
                nlines = 0
                if vp.exists():
                    try:
                        nlines = len(json.loads(vp.read_text()).get("lines") or [])
                    except Exception:
                        nlines = 0
                return (nlines if nlines else 999)
            todo.sort(key=_weight)
            # ROOFS JOSH HAS POINTED AT COME FIRST, whatever their complexity.
            # The ordering below is a proxy for his drawing effort and it is a
            # rough one: #4734696 is a plain hip roof that the detector fires 20
            # lines at, so it sorted to the back of a "simplest first" queue
            # while being both quick to draw and visibly wrong on the map. A
            # roof he has looked at and called wrong is worth more than any
            # proxy, so those are pulled to the front.
            flag = DATA_DIR / "flagged_ids.txt"
            if flag.exists():
                want = [int(x) for x in flag.read_text().split() if x.strip()]
                todo = ([b for b in want if b in todo]
                        + [b for b in todo if b not in set(want)])
            ids = done + todo[:a.bench_n]
    if ids is None:
        q = OUT_DIR / "queue.json"
        if not q.exists():
            print("no queue.json -- run tools/triage_roofs.py --bundle 40 "
                  "(or tools/sample_roofs_to_label.py) first")
            return 2
        qd = json.loads(q.read_text())
        ids = qd["ids"]
        # A triage-built queue knows why each roof is here. Showing that while
        # marking is worth real accuracy: "10 panels cross a predicted line"
        # tells the labeller where to look, rather than leaving them to rediscover
        # the fault on a roof that may look fine at first glance.
        why = {int(k): v for k, v in (qd.get("reasons") or {}).items()}
    if a.max:
        ids = ids[:a.max]

    _saved = {}
    lp = DATA_DIR / "roof_labels.json"
    if lp.exists():
        for k, v in json.loads(lp.read_text()).get("buildings", {}).items():
            if not (v.get("lines") or v.get("obstructions")
                    or v.get("nopanel") or v.get("problem")):
                continue
            _saved[int(k)] = {
                "lines": v.get("lines") or [],
                "obstructions": v.get("obstructions") or [],
                "nopanel": v.get("nopanel") or [],
                "complete": bool(v.get("complete")),
                "problem": v.get("problem"),
            }

    truth = {}
    tp = DATA_DIR / "roof_truth.json"
    if tp.exists():
        for r in json.loads(tp.read_text()).get("roofs", []):
            if r.get("building_id"):
                truth[int(r["building_id"])] = r

    # ADDRESSES COME FROM THE BUILD, NOT FROM roof_truth.json. Josh: "Give me
    # the address in the title for the building". roof_truth carries 28 roofs;
    # the built solar_potential.geojson carries an address for all 15,353,
    # because add_addresses runs over the whole district. Reading truth first
    # meant almost every roof showed its region name instead of a street.
    addr = {}
    sp = DATA_DIR / "solar_potential.geojson"
    if sp.exists():
        try:
            for f in json.loads(sp.read_text()).get("features", []):
                pr = f.get("properties") or {}
                if pr.get("building_id") is not None and pr.get("address"):
                    addr[int(pr["building_id"])] = pr["address"]
        except Exception:
            pass

    # WHAT THE PIPELINE CURRENTLY THINKS THE ROOF IS.
    #
    # Josh: "showing how you interpret rooftops and how I have drawn them, and
    # ones where you need help". Until now the tool showed him the imagery and
    # his own lines; it never showed what the build had made of the roof. So
    # the one person who can say "that is wrong" could not see what to correct,
    # and the disagreement only surfaced days later on the live map.
    #
    # Read straight from the built panel_layouts, reprojected to the tool's
    # frame. If a region has not been built the roof simply carries none and
    # the tool behaves as before.
    def _built_facets(region):
        import rasterio  # noqa: F401  (pyproj already imported above)
        pth = area_paths(region)["dir"] / "panel_layouts.geojson"
        if not pth.exists():
            return {}
        to_nztm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193",
                                              always_xy=True).transform
        out = {}
        try:
            doc = json.loads(pth.read_text())
        except Exception:
            return {}
        for f in doc.get("features", []):
            pr = f.get("properties") or {}
            if pr.get("kind") != "facet":
                continue
            b = pr.get("building_id")
            if b is None:
                continue
            try:
                ring = f["geometry"]["coordinates"][0]
                out.setdefault(int(b), []).append(
                    [[round(v, 2) for v in to_nztm(x, y)] for x, y in ring])
            except Exception:
                continue
        return out

    built_cache = {}
    ctxs = {}
    roofs = []
    print(f"building a bundle of {len(ids)} roofs...")
    for i, bid in enumerate(ids, 1):
        placed = False
        for name in ["pilot"] + [x for x in all_areas() if x != "pilot"]:
            if name not in ctxs:
                p = area_paths(name)
                if not p["outlines"].exists() or not p["imagery"].exists():
                    ctxs[name] = None
                    continue
                dd = p["dir"] / "building_outlines_dedup.geojson"
                ctxs[name] = {
                    "gdf": gpd.read_file(dd if dd.exists() else p["outlines"]
                                         ).set_index("building_id", drop=False),
                    "img": rasterio.open(p["imagery"]),
                }
            ctx = ctxs[name]
            if ctx is None or bid not in ctx["gdf"].index:
                continue
            g = ctx["gdf"].loc[bid].geometry
            minx, miny, maxx, maxy = g.bounds
            b = (minx - PAD_M, miny - PAD_M, maxx + PAD_M, maxy + PAD_M)
            jpg, size = crop(ctx["img"], b)
            # neighbours whose outline intersects the crop, so a busy block is
            # legible -- see the note in the module docstring
            from shapely.geometry import box
            win = box(*b)
            nb = []
            for oid, og in zip(ctx["gdf"]["building_id"], ctx["gdf"].geometry):
                if int(oid) == bid or not og.intersects(win):
                    continue
                nb.append([[round(x, 2), round(y, 2)]
                           for x, y in og.exterior.coords])
                if len(nb) >= 12:
                    break
            lon, lat = to_wgs((minx + maxx) / 2, (miny + maxy) / 2)
            t = truth.get(bid, {})
            roofs.append({
                "id": bid, "area": name,
                "address": t.get("address") or addr.get(bid, ""),
                "m2": round(g.area, 1),
                "ll": [round(lat, 6), round(lon, 6)],
                "bounds": [round(v, 2) for v in b],
                "px": list(size),
                "outline": [[round(x, 2), round(y, 2)] for x, y in g.exterior.coords],
                # courtyards. Rare (1 in 156 here) but a hole drawn as solid
                # roof invites a labeller to mark faces over open air, and on a
                # big commercial block that is a 1200 m2 mistake.
                "holes": [[[round(x, 2), round(y, 2)] for x, y in r.coords]
                          for r in g.interiors],
                "neighbours": nb,
                "why": why.get(bid, []),
                "built": built_cache.setdefault(
                    name, _built_facets(name)).get(bid, []),
                # WORK ALREADY DONE TRAVELS WITH THE BUNDLE. The tool keeps
                # marks in the browser's local storage, so a fresh bundle on a
                # fresh machine shows every roof as untouched -- including the
                # ones Josh has already drawn, which he would then draw again
                # and "next unmarked" would stop on. Seeding from
                # roof_labels.json makes the bundle carry its own history.
                "saved": _saved.get(bid),
                "jpg": jpg,
            })
            placed = True
            break
        if not placed:
            print(f"  skip #{bid} (no imagery on this machine)")
        if i % 25 == 0:
            print(f"  {i}/{len(ids)}")

    html = (ROOT / "tools" / "label_template.html").read_text()
    html = html.replace("/*__ROOFS__*/", json.dumps(roofs, separators=(",", ":")))
    # Replaces the placeholder AND the "{}" default after it, so the template
    # stays valid JavaScript when opened directly and valid once substituted.
    html = html.replace("/*__OPTS__*/{}",
                        json.dumps({"unmarkedOnly": bool(a.bench)}))
    # PUBLISHED ONLINE THE PAGE IS A FRAGMENT, NOT A DOCUMENT. The Artifact
    # host wraps whatever it is given in its own doctype/head/body, so shipping
    # ours nests a second document inside the first. Everything between the
    # tags is kept verbatim -- title, styles and script are all still there.
    #
    # Worth it because online the tool reaches claude.use("db") and saves each
    # roof server-side as it is drawn, instead of trusting one browser's local
    # storage with a day of markup.
    if a.artifact:
        import re
        html = re.sub(r"(?is)<!doctype[^>]*>", "", html)
        html = re.sub(r"(?is)</?html[^>]*>", "", html)
        html = re.sub(r"(?is)</?head[^>]*>", "", html)
        html = re.sub(r"(?is)</?body[^>]*>", "", html)
        html = re.sub(r"(?is)<meta[^>]*charset[^>]*>", "", html)
        html = html.strip()

    out = OUT_DIR / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    mb = out.stat().st_size / 1024 / 1024
    print(f"\n{len(roofs)} roofs -> {out}  ({mb:.1f} MB)")
    print("  Open it by double-clicking. It works offline and saves as you go.")
    print("  When done, click Download and send the JSON back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

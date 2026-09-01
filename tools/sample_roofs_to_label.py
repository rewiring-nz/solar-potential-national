"""
Pick a diverse set of roofs to label, and write them into one folder.

Labelling 300 random roofs would spend most of the effort on the same simple
gable over and over, and barely touch the shapes that actually fail. This sorts
every building in the district into roof types from what the pipeline already
measures -- facet count, slope spread, aspect clustering, footprint area,
flatness -- and then samples across the types instead of across the district.

The types are deliberately observable rather than architectural. Nothing here
knows what a "mansard" is; it knows "many facets, two aspect clusters, steep",
which is what the geometry has to get right anyway.

WHY THIS ORDER MATTERS: the failures already recorded in roof_truth.json are
concentrated in three types -- sawtooth commercial, flat commercial, and stepped
houses. Those are rare in the district and would be nearly absent from a random
sample, which is exactly how a model trained on random roofs would end up
excellent at gables and useless at the roofs we cannot currently do.

Output lands in data/label_set/:
    index.html      a contact sheet, every roof with its type, click to enlarge
    queue.json      the id list, fed straight to tools/label_roofs.py
    roofs/<id>.jpg  one render per roof
    summary.json    counts per type and why each roof was chosen

Usage:
    python tools/sample_roofs_to_label.py                  # ~120 roofs
    python tools/sample_roofs_to_label.py --n 300
    python tools/sample_roofs_to_label.py --types sawtooth flat_commercial
"""

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "label_set"

# How many of each type to aim for. Weighted towards what fails, not towards
# what is common: a simple gable is already handled well and each extra one
# teaches almost nothing.
TARGETS = {
    "sawtooth":         18,   # 24 Beach St's class -- repeated parallel ridges
    "flat_commercial":  18,   # 9 Henry St, 32 Frankton Rd -- barely modelled
    "stepped":          14,   # 26 Panorama Tce -- risers admitted as roof
    "complex_multi":    20,   # many facets, several aspect clusters
    "hip":              16,
    "gable":            14,
    "small_simple":     10,   # the easy majority, enough to keep it honest
    "steep":            10,   # where the 42-55 degree question lives
}


def _aspect_clusters(aspects, tol=35.0):
    """How many distinct directions this roof faces. A gable has 2, a hip 4."""
    if not aspects:
        return 0
    used, groups = [False] * len(aspects), 0
    for i, a in enumerate(aspects):
        if used[i]:
            continue
        groups += 1
        for j, b in enumerate(aspects):
            d = abs(a - b) % 360
            if min(d, 360 - d) <= tol:
                used[j] = True
    return groups


def classify(p, foot_area):
    """Roof type from what the pipeline already measured. Order matters --
    the first match wins, and the failure-prone types are tested first."""
    n = p.get("facet_count") or 0
    roof = p.get("facet_area_m2") or 0
    slopes = p.get("_slopes") or []
    aspects = p.get("_aspects") or []
    if not n:
        return None
    flat = [s for s in slopes if s < 8]
    steep = [s for s in slopes if s > 40]
    clusters = _aspect_clusters(aspects)

    # Sawtooth: many facets, only two aspect clusters (up-slope and down-slope
    # of every tooth), and a big roof. This is the shape the segmenter smears.
    if n >= 6 and clusters <= 2 and foot_area >= 150:
        return "sawtooth"
    # Flat commercial: big, and mostly level.
    if foot_area >= 300 and len(flat) >= max(1, 0.6 * n):
        return "flat_commercial"
    # Stepped: several facets at similar aspect but a wide slope spread --
    # level changes read as extra faces.
    if n >= 4 and slopes and (max(slopes) - min(slopes)) > 25 and clusters <= 3:
        return "stepped"
    if steep:
        return "steep"
    if n >= 8:
        return "complex_multi"
    if n >= 4 and clusters >= 4:
        return "hip"
    if 2 <= n <= 3 and clusters <= 2:
        return "gable"
    if n <= 3 and foot_area < 150:
        return "small_simple"
    return "complex_multi"


def load_buildings():
    """Every building with its facet stats, from the shipped layout + potential."""
    sp = json.loads((DATA_DIR / "solar_potential.geojson").read_text())
    props = {int(f["properties"]["building_id"]): dict(f["properties"])
             for f in sp["features"]}
    lay = json.loads((DATA_DIR / "panel_layouts.geojson").read_text())
    slopes, aspects = defaultdict(list), defaultdict(list)
    for f in lay["features"]:
        q = f["properties"]
        if q.get("kind") != "facet":
            continue
        b = int(q["building_id"])
        if q.get("slope_deg") is not None:
            slopes[b].append(float(q["slope_deg"]))
        if q.get("aspect_deg") is not None:
            aspects[b].append(float(q["aspect_deg"]))
    for b, p in props.items():
        p["_slopes"] = slopes.get(b, [])
        p["_aspects"] = aspects.get(b, [])
    return props


def render(bid, area, out_path, bundles):
    """One clean render per roof: imagery, footprint, no model output.

    No model overlay ON PURPOSE -- mark_roofs.py's reasoning still holds, that
    showing the current faces anchors the answer to what the pipeline already
    believes. The interactive tool can toggle the guess on; the contact sheet
    should not."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio.windows
    ctx = bundles.area(area)
    if ctx is None or ctx["imagery"] is None or bid not in ctx["gdf"].index:
        return False
    g = ctx["gdf"].loc[bid].geometry
    minx, miny, maxx, maxy = g.bounds
    pad = 3.0
    w = rasterio.windows.from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad,
                                     ctx["imagery"].transform)
    rgb = np.moveaxis(ctx["imagery"].read([1, 2, 3], window=w,
                                          boundless=True, fill_value=0), 0, -1)
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
    ax.imshow(rgb, extent=[minx - pad, maxx + pad, miny - pad, maxy + pad])
    xs, ys = g.exterior.xy
    ax.plot(xs, ys, color="#ffd400", lw=1.3, alpha=.9)
    ax.set_xlim(minx - pad, maxx + pad); ax.set_ylim(miny - pad, maxy + pad)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="override the total")
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--no-renders", action="store_true")
    a = ap.parse_args()

    from tools.label_roofs import Bundles, _truth_index
    from src.region_build import all_areas
    import geopandas as gpd
    from src.region_build import area_paths

    props = load_buildings()
    # footprint area and area name per building
    foot, area_of = {}, {}
    for name in ["pilot"] + [x for x in all_areas() if x != "pilot"]:
        p = area_paths(name)
        if not p["outlines"].exists():
            continue
        g = gpd.read_file(p["outlines"])
        for bid, geom in zip(g["building_id"], g.geometry):
            foot[int(bid)] = geom.area
            area_of[int(bid)] = name

    by_type = defaultdict(list)
    for bid, p in props.items():
        if bid not in foot:
            continue
        t = classify(p, foot[bid])
        if t:
            by_type[t].append((bid, p, foot[bid]))

    print("roof types found across the district:")
    for t, v in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        print(f"  {t:18s} {len(v):6,}")

    truth = _truth_index()
    targets = dict(TARGETS)
    if a.types:
        targets = {t: targets.get(t, 12) for t in a.types}
    if a.n:
        scale = a.n / sum(targets.values())
        targets = {t: max(2, round(v * scale)) for t, v in targets.items()}

    chosen, why = [], {}
    # Josh's already-marked roofs go first, whatever their type: turning his
    # prose into coordinates is the cheapest labelling available.
    for bid in sorted(truth):
        if bid in foot:
            chosen.append(bid)
            why[bid] = "already marked by Josh (prose -> coordinates)"

    for t, want in targets.items():
        pool = [x for x in by_type.get(t, []) if x[0] not in why]
        # biggest first within a type: more roof per label, and the big ones are
        # where the money and the errors both concentrate
        pool.sort(key=lambda x: -x[2])
        step = max(1, len(pool) // max(want, 1))
        for bid, p, fa in pool[::step][:want]:
            chosen.append(bid)
            why[bid] = f"{t} ({p.get('facet_count')} facets, {fa:.0f} m2)"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "roofs").mkdir(exist_ok=True)
    bundles = Bundles()
    rendered = 0
    if not a.no_renders:
        print(f"\nrendering {len(chosen)} roofs...")
        for i, bid in enumerate(chosen, 1):
            if render(bid, area_of[bid], OUT_DIR / "roofs" / f"{bid}.jpg", bundles):
                rendered += 1
            if i % 25 == 0:
                print(f"  {i}/{len(chosen)}")

    (OUT_DIR / "queue.json").write_text(json.dumps({"ids": chosen}, indent=1))
    counts = Counter(why[b].split(" (")[0] for b in chosen)
    (OUT_DIR / "summary.json").write_text(json.dumps({
        "total": len(chosen), "rendered": rendered,
        "by_reason": dict(counts),
        "buildings": {str(b): {"why": why[b], "area": area_of[b],
                               "address": (truth.get(b) or {}).get("address", "")}
                      for b in chosen}}, indent=1, sort_keys=True))
    _write_index(chosen, why, area_of, truth)
    print(f"\n{len(chosen)} roofs -> {OUT_DIR}")
    for k, v in counts.most_common():
        print(f"  {k:52s} {v}")
    print(f"\n  contact sheet: {OUT_DIR/'index.html'}")
    print(f"  label them:    python tools/label_roofs.py --ids $(python -c "
          f"\"import json;print(' '.join(map(str,json.load(open('{OUT_DIR}/queue.json'))['ids'])))\")")


def _write_index(chosen, why, area_of, truth):
    cards = []
    for b in chosen:
        addr = (truth.get(b) or {}).get("address", "")
        cards.append(
            f'<figure><img src="roofs/{b}.jpg" alt="#{b}" loading="lazy">'
            f'<figcaption><b>#{b}</b> {addr}<br><span>{why[b]}</span>'
            f'<br><span class="a">{area_of[b]}</span></figcaption></figure>')
    (OUT_DIR / "index.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>Roofs to label</title>
<style>
 body{{background:#14171a;color:#e8edf1;font:14px -apple-system,BlinkMacSystemFont,sans-serif;margin:0;padding:24px}}
 h1{{font-size:20px;margin:0 0 4px}} p.sub{{color:#94a3ad;margin:0 0 22px;max-width:70ch}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:16px}}
 figure{{margin:0;background:#1c2126;border:1px solid #2b333a;border-radius:4px;overflow:hidden}}
 img{{width:100%;display:block;background:#000;cursor:zoom-in}}
 figcaption{{padding:9px 11px;font-size:12.5px;line-height:1.45}}
 figcaption span{{color:#94a3ad}} figcaption .a{{font-family:ui-monospace,monospace;font-size:11px}}
 img:target,img.big{{position:fixed;inset:0;margin:auto;max-width:96vw;max-height:96vh;width:auto;z-index:9;cursor:zoom-out}}
</style>
<h1>{len(chosen)} roofs to label</h1>
<p class="sub">Sampled across roof TYPES rather than at random, weighted towards the shapes that
currently fail — sawtooth, flat commercial and stepped houses are rare in the district and would
barely appear in a random draw. Josh's already-marked roofs come first. Click a roof to enlarge.</p>
<div class="grid">{''.join(cards)}</div>
<script>document.querySelectorAll('img').forEach(i=>i.onclick=()=>i.classList.toggle('big'));</script>
""")


if __name__ == "__main__":
    main()

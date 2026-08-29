"""
Before/after panel layouts, side by side, for Josh to judge.

The point of this over every metric tried so far: he looks at a pair and says
which is better. Plane counts, off-plane residuals and fill percentages have
each been misleading in one direction or another today -- a residual always
improves when a roof is cut into more pieces, a plane count cannot see a
straight edge. A person looking at two layouts can see all of it at once.

Usage:
  python src/compare_layouts.py --old <saved.geojson> --area pilot --n 12
  python src/compare_layouts.py --old <saved.geojson> --area pilot --ids 4734907 ...
  python src/compare_layouts.py --refit --area pilot --ids 4734907 ...

--refit is the one that answers "is the code better than what shipped?". Both
sides came from saved files before, so comparing the shipped file against
itself produced eighteen identical pairs -- the fixes under test only change
what gets FITTED, and nothing had been re-fitted. With --refit the AFTER side
is re-run through the current pipeline, building by building, so a change can
be judged before spending a rebuild on it.
"""

import argparse
import base64
import io
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import shape
from shapely.ops import transform as shp_transform, unary_union

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.region_build import area_paths

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True).transform
PAD_M = 3.0
DENSITY = 100      # show everything; the density slider is a separate question


def load(path):
    d = json.loads(Path(path).read_text())
    by = defaultdict(lambda: {"panel": [], "facet": [], "obstruction": []})
    for f in d["features"]:
        pr = f.get("properties", {})
        k = pr.get("kind")
        if k not in ("panel", "facet", "obstruction"):
            continue
        if k == "panel" and pr.get("fill_rank") is not None and pr["fill_rank"] > DENSITY:
            continue
        by[int(pr["building_id"])][k].append(shp_transform(TO_NZTM, shape(f["geometry"])).buffer(0))
    return by


def draw(ax, imagery, g, bounds, title):
    minx, miny, maxx, maxy = bounds
    w = rasterio.windows.from_bounds(minx - PAD_M, miny - PAD_M, maxx + PAD_M, maxy + PAD_M,
                                     imagery.transform)
    img = np.moveaxis(imagery.read([1, 2, 3], window=w, boundless=True, fill_value=0), 0, -1)
    wt = imagery.window_transform(w)
    ax.imshow(img, extent=(wt.c, wt.c + img.shape[1] * wt.a,
                           wt.f + img.shape[0] * wt.e, wt.f), origin="upper")
    for fg in g["facet"]:
        for poly in (fg.geoms if fg.geom_type == "MultiPolygon" else [fg]):
            if not poly.is_empty:
                ax.plot(*poly.exterior.xy, color="white", lw=1.0, ls=(0, (2, 2)), alpha=0.9)
    for o in g["obstruction"]:
        for poly in (o.geoms if o.geom_type == "MultiPolygon" else [o]):
            if not poly.is_empty:
                ax.add_patch(MplPolygon(list(zip(*poly.exterior.xy)), closed=True,
                                        facecolor="#a855f7", edgecolor="#a855f7", alpha=0.8))
    for p in g["panel"]:
        if p.is_empty or p.geom_type != "Polygon":
            continue
        ax.add_patch(MplPolygon(list(zip(*p.exterior.xy)), closed=True, facecolor="#0b1c3a",
                                edgecolor="#7fd4ff", lw=0.6, alpha=0.75))
    ax.set_xlim(minx - PAD_M, maxx + PAD_M)
    ax.set_ylim(miny - PAD_M, maxy + PAD_M)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=8.5, color="#ddd")


def _refit_ids(area, ids, partition=False):
    """Run the real pipeline for these buildings and return the same structure
    load() produces, so the AFTER side reflects the code as it stands now."""
    import geopandas as gpd
    from src.roof_segmentation import segment_building_best
    from src.obstruction_detection import detect_obstructions_combined
    from src.panel_fitting import fit_panels_on_facet, drop_minor_arrays, assign_fill_ranks
    from src.pointcloud_source import PointCloudSource

    p = area_paths(area)
    dedup = p["dir"] / "building_outlines_dedup.geojson"
    gdf = gpd.read_file(dedup if dedup.exists() else p["outlines"]).set_index(
        "building_id", drop=False)
    dsm = rasterio.open(p["dsm"])
    img = rasterio.open(p["imagery"]) if p["imagery"].exists() else None
    pc = PointCloudSource()

    out = defaultdict(lambda: {"panel": [], "facet": [], "obstruction": []})
    for bid in ids:
        if bid not in gdf.index:
            continue
        geom = gdf.loc[bid].geometry
        if partition:
            import shapely.vectorized
            from src.roof_partition import partition_roof
            mn, mi, mx, ma = geom.bounds
            allp = pc.points_in_bbox(mn - 2, mi - 2, mx + 2, ma + 2, building_only=True)
            allp = allp[shapely.vectorized.contains(geom, allp[:, 0], allp[:, 1])]
            facets = partition_roof(bid, geom, allp)
            for f in facets:
                f["building_geometry"] = geom   # panel_fitting aligns rows to it
        else:
            facets = segment_building_best(dsm, pc, geom, bid, imagery_ds=img)
        per_facet = []
        for f in facets:
            plane = (f["plane_a"], f["plane_b"], f["plane_c"])
            ob = detect_obstructions_combined(img, pc, f["geometry"], plane)
            out[bid]["facet"].append(f["geometry"])
            out[bid]["obstruction"].extend(ob)
            per_facet.append(fit_panels_on_facet(
                f, obstructions=ob,
                sibling_facets=[o for o in facets if o is not f]))
        panels = [q for lst in drop_minor_arrays(per_facet) for q in lst]
        for i, q in enumerate(panels):
            q.setdefault("poa_kwh_m2_yr", 1000.0)
            q.setdefault("facet_key", 0)
            q.setdefault("order", i)
        if panels:
            assign_fill_ranks(panels)
        out[bid]["panel"] = [q["geometry"] for q in panels
                             if (q.get("fill_rank") or 0) <= DENSITY]
        print(f"    refit #{bid}: {len(facets)} facets, {len(out[bid]['panel'])} panels",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old")
    ap.add_argument("--area", default="pilot")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ids", nargs="*", type=int)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--refit", action="store_true",
                    help="re-run the pipeline for the chosen ids as the AFTER side")
    ap.add_argument("--partition", action="store_true",
                    help="AFTER side uses roof_partition instead of the segmenter")
    a = ap.parse_args()

    if not a.refit and not a.old:
        ap.error("--old is required unless --refit is given")
    old = load(a.old) if a.old else load(area_paths(a.area)["panel_layouts"])
    new = None if a.refit else load(area_paths(a.area)["panel_layouts"])
    sp = json.loads(area_paths(a.area)["solar_potential"].read_text())
    addr = {int(f["properties"]["building_id"]): f["properties"].get("address", "")
            for f in sp["features"]}

    if a.ids:
        ids = [i for i in a.ids if i in old or (new is not None and i in new)]
    elif a.refit:
        ap.error("--refit needs --ids")
    else:
        # Weighted to houses, and to buildings where the two actually differ --
        # a pair that looks identical teaches nothing.
        both = [b for b in set(old) & set(new)
                if old[b]["facet"] and new[b]["facet"]]
        def roof(b, src): return unary_union(src[b]["facet"]).area
        cand = [b for b in both if 60 <= roof(b, new) <= 400]
        cand.sort(key=lambda b: -abs(len(new[b]["panel"]) - len(old[b]["panel"])))
        rng = np.random.default_rng(a.seed)
        top = cand[:max(a.n * 3, 30)]
        ids = list(rng.choice(top, size=min(a.n, len(top)), replace=False)) if top else []

    if a.refit:
        new = _refit_ids(a.area, [int(i) for i in ids], partition=a.partition)

    imagery = rasterio.open(area_paths(a.area)["dir"] / "imagery_mosaic.tif")
    cards = []
    for i, bid in enumerate(ids, 1):
        bid = int(bid)
        go, gn = old.get(bid), new.get(bid)
        if not go or not gn:
            continue
        allg = go["facet"] + gn["facet"] + go["panel"] + gn["panel"]
        if not allg:
            continue
        xs = [c for geom in allg for c in geom.bounds[0::2]]
        ys = [c for geom in allg for c in geom.bounds[1::2]]
        bounds = (min(xs), min(ys), max(xs), max(ys))
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.4), facecolor="#15141a")
        draw(ax[0], imagery, go, bounds,
             f"BEFORE  {len(go['panel'])} panels, {len(go['facet'])} facets")
        draw(ax[1], imagery, gn, bounds,
             f"AFTER  {len(gn['panel'])} panels, {len(gn['facet'])} facets")
        fig.tight_layout(pad=0.4)
        buf = io.BytesIO()
        fig.savefig(buf, format="jpg", dpi=105, bbox_inches="tight", facecolor="#15141a",
                    pil_kwargs={"quality": 82, "optimize": True})
        plt.close(fig)
        cards.append({"i": i, "bid": bid, "addr": addr.get(bid, ""),
                      "before": len(go["panel"]), "after": len(gn["panel"]),
                      "img": base64.b64encode(buf.getvalue()).decode()})
        print(f"  {i:2d}. {bid} {addr.get(bid,''):<26} {len(go['panel']):4d} -> {len(gn['panel']):4d} panels")
    imagery.close()
    out = DATA_DIR / f"compare_layouts_{a.area}.html"
    out.write_text(build_html(cards))
    print(f"\nSaved {out} ({out.stat().st_size/1e6:.1f}MB, {len(cards)} pairs)")


def build_html(cards):
    def card(c):
        d = c["after"] - c["before"]
        return f"""<figure class="card">
  <figcaption class="head"><span class="n">{c['i']}</span>
    <span class="addr">{c['addr'] or ('#' + str(c['bid']))}</span>
    <span class="delta">{c['before']} &rarr; {c['after']} panels ({d:+d})</span></figcaption>
  <img src="data:image/jpeg;base64,{c['img']}" alt="{c['addr']}">
</figure>"""
    return f"""<title>Layouts before and after</title>
<style>
  :root {{ --bg:#f8f7f4; --panel:#fff; --ink:#16161a; --ink-2:#5d5d67; --line:#e4e1dc; --mark:#b45309; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
    --bg:#131217; --panel:#1d1c22; --ink:#f1efec; --ink-2:#a6a2aa; --line:#302e37; --mark:#fbbf24; }} }}
  :root[data-theme="dark"] {{ --bg:#131217; --panel:#1d1c22; --ink:#f1efec;
    --ink-2:#a6a2aa; --line:#302e37; --mark:#fbbf24; }}
  body {{ background:var(--bg); color:var(--ink); margin:0; padding:26px;
    font:14px/1.55 ui-sans-serif,-apple-system,"Helvetica Neue",sans-serif; }}
  h1 {{ font-size:21px; margin:0 0 6px; }}
  .sub {{ color:var(--ink-2); max-width:76ch; margin:0 0 22px; }}
  .grid {{ display:grid; gap:20px; }}
  .card {{ margin:0; background:var(--panel); border:1px solid var(--line);
    border-radius:9px; padding:11px 13px 13px; }}
  .head {{ display:flex; align-items:baseline; gap:9px; }}
  .n {{ background:var(--mark); color:#1a1a1a; font-weight:800; border-radius:4px;
    padding:0 7px; font-size:12px; }}
  .addr {{ font-weight:700; flex:1; }}
  .delta {{ color:var(--ink-2); font-size:12.5px; font-variant-numeric:tabular-nums; }}
  .card img {{ display:block; width:100%; height:auto; border-radius:5px; margin-top:9px; }}
</style>
<h1>Layouts before and after</h1>
<p class="sub">Left is the segmentation on the map today; right is the same roof rebuilt as
planes joined along their intersection lines and clipped to the surveyed outline. White dashes
are facet boundaries, purple is a detected obstruction, blue are panels at full density.
<strong>Reply with the numbers where BEFORE is better</strong> &mdash; silence means the new one
wins or they are level.</p>
<div class="grid">
{chr(10).join(card(c) for c in cards)}
</div>"""


if __name__ == "__main__":
    main()

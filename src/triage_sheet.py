"""
Render the buildings an audit flagged into one page, so a bad layout can be
confirmed or dismissed in a glance instead of hunted for on the map.

Josh has been finding bugs by clicking around the live map and screenshotting
them. Every report has been real, but it is slow and it only ever surfaces
what he happens to look at. audit_layouts.py already scores every building;
what was missing was the other half of the loop -- seeing the flagged ones.

The split this makes matters more than the rendering. audit_layouts flags a
panel "lumpy" when the LiDAR under it sits far off its facet plane, and the
FLAG RATE says which of two different bugs it is:

  obstruction   a minority of panels flagged, in a clump -- the roof is
                mostly planar and something on it (plant, chimney, dormer)
                was not detected. Fix belongs in obstruction_detection.
  plane         most of the panels on the building flagged -- a roof is not
                92% chimneys. The facet plane does not describe the surface:
                segmentation fitted one plane across a curved, stepped or
                multi-level roof. Fix belongs in roof_segmentation.

Both look identical on the map and they have been getting reported as one
thing, which is part of why fixes keep trading one against the other.

Renders the SHIPPED layout, not a re-fit, so what appears is what is on the
map today. Red outline = the panels the LiDAR objects to.

Usage:
  python src/triage_sheet.py --area pilot --category plane --n 12
  python src/triage_sheet.py --area pilot --category obstruction --n 12
  python src/triage_sheet.py --area pilot --ids 4733121 4735403
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
import shapely.vectorized
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.audit_layouts import (LUMPY_MEDIAN_M, LUMPY_MIN_PTS, ZSPLIT_RANGE_M,
                               plane_from_facet_points)
from src.pointcloud_source import PointCloudSource
from src.region_build import area_paths

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True).transform
# Above this share of panels flagged, the plane is wrong rather than the roof
# being covered in equipment. Chosen from the pilot distribution: genuine
# missed obstructions clustered under 0.30, the plane failures sat over 0.60.
PLANE_FAIL_RATE = 0.5
PAD_M = 4.0


def load_building_geoms(area):
    """Group the shipped layout by building, reprojected to NZTM."""
    layouts = json.loads(area_paths(area)["panel_layouts"].read_text())
    by = defaultdict(lambda: {"panel": [], "facet": [], "obstruction": []})
    for f in layouts["features"]:
        k = f["properties"].get("kind")
        if k in ("panel", "facet", "obstruction"):
            by[int(f["properties"]["building_id"])][k].append(
                shp_transform(TO_NZTM, shape(f["geometry"])).buffer(0))
    return by


def flag_panels(g, pc):
    """Per-panel lumpy/z_split, same test as audit_layouts -- but returning
    WHICH panels, which the audit throws away."""
    facets = g["facet"]
    if not facets:
        return set(), set()
    tree = STRtree(facets)
    planes = []
    for fg in facets:
        minx, miny, maxx, maxy = fg.bounds
        pts = pc.points_in_bbox(minx, miny, maxx, maxy, building_only=True)
        fp = pts[shapely.vectorized.contains(fg, pts[:, 0], pts[:, 1])] if len(pts) >= 12 else pts[:0]
        planes.append(plane_from_facet_points(fp) if len(fp) >= 12 else None)
    lumpy, zsplit = set(), set()
    for i, panel in enumerate(g["panel"]):
        if panel.is_empty:
            continue
        cand = tree.query(panel)
        fi = None
        for idx in cand:
            if facets[idx].contains(panel.centroid):
                fi = int(idx)
                break
        if fi is None and len(cand):
            fi = int(min(cand, key=lambda j: facets[j].distance(panel.centroid)))
        if fi is None or planes[fi] is None or planes[fi][2] is None:
            continue
        minx, miny, maxx, maxy = panel.bounds
        pts = pc.points_in_bbox(minx, miny, maxx, maxy, building_only=True)
        if len(pts) < LUMPY_MIN_PTS:
            continue
        pp = pts[shapely.vectorized.contains(panel, pts[:, 0], pts[:, 1])]
        if len(pp) < LUMPY_MIN_PTS:
            continue
        x0, y0, cf = planes[fi]
        res = cf[0] * (pp[:, 0] - x0) + cf[1] * (pp[:, 1] - y0) + cf[2] - pp[:, 2]
        if np.median(np.abs(res)) > LUMPY_MEDIAN_M:
            lumpy.add(i)
        if np.percentile(res, 95) - np.percentile(res, 5) > ZSPLIT_RANGE_M:
            zsplit.add(i)
    return lumpy, zsplit


def render(bid, g, lumpy, zsplit, imagery):
    """Live-map colours (see debug_render_check.py) plus the flag overlay."""
    allg = g["facet"] + g["panel"] + g["obstruction"]
    if not allg:
        return None
    xs = [c for geom in allg for c in geom.bounds[0::2]]
    ys = [c for geom in allg for c in geom.bounds[1::2]]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    window = rasterio.windows.from_bounds(minx - PAD_M, miny - PAD_M,
                                          maxx + PAD_M, maxy + PAD_M, imagery.transform)
    img = np.moveaxis(imagery.read([1, 2, 3], window=window, boundless=True, fill_value=0), 0, -1)
    wt = imagery.window_transform(window)
    extent = (wt.c, wt.c + img.shape[1] * wt.a, wt.f + img.shape[0] * wt.e, wt.f)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(img, extent=extent, origin="upper")
    for fg in g["facet"]:
        for poly in (fg.geoms if fg.geom_type == "MultiPolygon" else [fg]):
            ax.plot(*poly.exterior.xy, color="white", lw=1.1, ls=(0, (2, 2)))
    for o in g["obstruction"]:
        for poly in (o.geoms if o.geom_type == "MultiPolygon" else [o]):
            ax.add_patch(MplPolygon(list(zip(*poly.exterior.xy)), closed=True,
                                    facecolor="#a855f7", edgecolor="#a855f7", alpha=0.8))
    for i, p in enumerate(g["panel"]):
        if p.is_empty or p.geom_type != "Polygon":
            continue
        bad = i in lumpy or i in zsplit
        ax.add_patch(MplPolygon(list(zip(*p.exterior.xy)), closed=True, facecolor="#000000",
                                edgecolor="#ff4d4d" if bad else "#7fd4ff",
                                lw=1.3 if bad else 0.6, alpha=0.45))
    ax.set_xlim(minx - PAD_M, maxx + PAD_M)
    ax.set_ylim(miny - PAD_M, maxy + PAD_M)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    buf = io.BytesIO()
    # JPEG, not PNG: these are aerial photographs, and a 16-card PNG sheet came
    # out at 17.7MB -- over the limit for publishing it as one page.
    fig.savefig(buf, format="jpg", dpi=110, bbox_inches="tight",
                pil_kwargs={"quality": 82, "optimize": True})
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="pilot")
    ap.add_argument("--category", choices=["plane", "obstruction", "all"], default="all")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ids", nargs="*", type=int)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    audit_path = DATA_DIR / f"audit_{a.area}.json"
    if not audit_path.exists():
        raise SystemExit(f"no {audit_path} -- run: python src/audit_layouts.py {a.area} --sample 500")
    rows = json.loads(audit_path.read_text())

    if a.ids:
        picked = [r for r in rows if r["building_id"] in a.ids]
        found = {r["building_id"] for r in picked}
        picked += [{"building_id": i, "panels": 0, "lumpy": 0, "z_split": 0, "flag_rate": 0.0}
                   for i in a.ids if i not in found]
    else:
        def cat(r):
            return "plane" if r["flag_rate"] >= PLANE_FAIL_RATE else "obstruction"
        picked = [r for r in rows if a.category == "all" or cat(r) == a.category]
        # Worst first, but weight by how many panels are affected rather than
        # rate alone -- a 100% flagged two-panel shed is not the priority.
        picked.sort(key=lambda r: -(r["lumpy"] + r["z_split"]))
        picked = picked[:a.n]

    print(f"{len(picked)} buildings -> rendering")
    by = load_building_geoms(a.area)
    pc = PointCloudSource()   # tiles are district-wide, not per-area
    imagery = rasterio.open(area_paths(a.area)["dir"] / "imagery_mosaic.tif")

    cards = []
    for r in picked:
        bid = r["building_id"]
        g = by.get(bid)
        if not g:
            print(f"  {bid}: not in layout file, skipped")
            continue
        lumpy, zsplit = flag_panels(g, pc)
        b64 = render(bid, g, lumpy, zsplit, imagery)
        if not b64:
            continue
        kind = "plane" if r["flag_rate"] >= PLANE_FAIL_RATE else "obstruction"
        cards.append({"bid": bid, "img": b64, "kind": kind, "panels": len(g["panel"]),
                      "lumpy": len(lumpy), "zsplit": len(zsplit),
                      "obstructions": len(g["obstruction"]), "facets": len(g["facet"]),
                      "rate": r["flag_rate"]})
        print(f"  {bid}: {len(g['panel'])} panels, {len(lumpy)} lumpy, "
              f"{len(zsplit)} z-split, {len(g['facet'])} facets -> {kind}")
    imagery.close()

    out = Path(a.out) if a.out else DATA_DIR / f"triage_{a.area}_{a.category}.html"
    out.write_text(build_html(cards, a.area, a.category))
    print(f"\nSaved {out}  ({out.stat().st_size / 1e6:.1f}MB, {len(cards)} cards)")


def build_html(cards, area, category):
    def card(c):
        return f"""<figure class="card" id="b{c['bid']}">
  <img src="data:image/jpeg;base64,{c['img']}" alt="building {c['bid']}">
  <figcaption>
    <div class="cap-head"><span class="bid">#{c['bid']}</span>
      <span class="tag tag-{c['kind']}">{c['kind']}</span></div>
    <dl>
      <div><dt>panels</dt><dd>{c['panels']}</dd></div>
      <div><dt>flagged</dt><dd>{c['lumpy']} lumpy &middot; {c['zsplit']} split</dd></div>
      <div><dt>facets</dt><dd>{c['facets']}</dd></div>
      <div><dt>obstructions</dt><dd>{c['obstructions']}</dd></div>
    </dl>
  </figcaption>
</figure>"""
    n_plane = sum(1 for c in cards if c["kind"] == "plane")
    n_obs = len(cards) - n_plane
    return f"""<title>Layout triage &mdash; {area}</title>
<style>
  :root {{ --bg:#faf9f7; --panel:#fff; --ink:#1a1a1a; --ink-2:#5a5a5a; --line:#e3e0da;
           --plane:#b45309; --obs:#7c3aed; }}
  :root:not([data-theme="light"]) {{}}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
    --bg:#16151a; --panel:#1f1e25; --ink:#f0eeea; --ink-2:#a8a49c; --line:#333039;
    --plane:#fbbf24; --obs:#c4b5fd; }} }}
  :root[data-theme="dark"] {{ --bg:#16151a; --panel:#1f1e25; --ink:#f0eeea;
    --ink-2:#a8a49c; --line:#333039; --plane:#fbbf24; --obs:#c4b5fd; }}
  body {{ background:var(--bg); color:var(--ink); margin:0; padding:28px;
    font:14px/1.5 ui-sans-serif,-apple-system,"Helvetica Neue",sans-serif; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .sub {{ color:var(--ink-2); margin:0 0 22px; max-width:70ch; }}
  .grid {{ display:grid; gap:18px; grid-template-columns:repeat(auto-fill,minmax(310px,1fr)); }}
  .card {{ margin:0; background:var(--panel); border:1px solid var(--line);
    border-radius:8px; overflow:hidden; }}
  .card img {{ display:block; width:100%; height:auto; }}
  figcaption {{ padding:10px 12px 12px; border-top:1px solid var(--line); }}
  .cap-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }}
  .bid {{ font-weight:700; font-variant-numeric:tabular-nums; }}
  .tag {{ font-size:11px; text-transform:uppercase; letter-spacing:.06em; font-weight:700; }}
  .tag-plane {{ color:var(--plane); }} .tag-obstruction {{ color:var(--obs); }}
  dl {{ margin:0; display:grid; grid-template-columns:1fr 1fr; gap:2px 14px; }}
  dl div {{ display:flex; justify-content:space-between; gap:8px; }}
  dt {{ color:var(--ink-2); font-size:12px; }}
  dd {{ margin:0; font-size:12px; font-variant-numeric:tabular-nums; }}
</style>
<h1>Layout triage &mdash; {area}</h1>
<p class="sub">Red-outlined panels are the ones the LiDAR objects to: the points beneath
them sit more than {LUMPY_MEDIAN_M}m off their facet plane. <strong>{n_obs} obstruction</strong>
(a clump flagged on an otherwise planar roof &mdash; something on the roof went undetected)
and <strong>{n_plane} plane</strong> (most of the roof flagged &mdash; the facet plane does not
describe the surface, so segmentation is at fault, not detection).
White dashes are facet boundaries, purple is a detected obstruction. This renders the
shipped layout at full density, so it is what is on the map today.</p>
<div class="grid">
{chr(10).join(card(c) for c in cards)}
</div>"""


if __name__ == "__main__":
    main()

"""
Build and RENDER a handful of roofs, so a geometry change can be judged in
minutes instead of a district rebuild.

Josh: "is there a way to test on a smaller amount of buildings so we can have a
more efficient feedback loop". Yes, and the lack of one has been the real cost
this week. Every change today was judged either by an aggregate that cannot see
a building shipping zero panels, or by a 4.5 hour rebuild followed by opening
the live map. Both are too slow and too coarse, which is why five regressions
reached a deploy.

WHAT THIS DOES. Runs the real pipeline -- segment_building_best, obstruction
detection, panel fitting, the same calls build_layout_geojson makes -- over a
chosen set of buildings, then writes ONE self-contained HTML page showing each
roof's imagery with its facets and panels drawn over it. No server, no deploy,
no district merge. Open it and look.

WHAT IT DELIBERATELY DOES NOT DO. No solar model, no shading, no gate, no
horizon. Those need the district context and none of them change the GEOMETRY,
which is what a partition change alters and what a person can actually judge by
eye. The panel count here is what the fitter placed, before gating -- close to
but not identical with what ships.

CHOOSING THE SAMPLE MATTERS MORE THAN THE SIZE. --labelled draws from the roofs
Josh has drawn, where there is ground truth; --like takes buildings that
resemble a given one, which is how you check whether a fix generalises off the
roofs it was tuned on. That distinction is exactly what caught the sawtooth
twin: 7 Anderson Heights was right and 7 Duncan's Place, the same roof design
unlabelled, was wrong.

Usage:
    python tools/preview_sample.py --ids 4725721 5371110 5371108
    python tools/preview_sample.py --region pilot --n 40
    python tools/preview_sample.py --labelled --n 20 --out check.html
"""

import argparse
import base64
import io
import json
import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "preview"
PAD_M = 3.0
PX = 520                 # rendered size per roof; big enough to judge a crease


_CTX = {}


def _init(region):
    import rasterio
    import geopandas as gpd
    from src.region_build import area_paths
    from src.pointcloud_source import PointCloudSource
    p = area_paths(region)
    dd = p["dir"] / "building_outlines_dedup.geojson"
    _CTX["gdf"] = gpd.read_file(dd if dd.exists() else p["outlines"]).set_index(
        "building_id", drop=False)
    _CTX["dsm"] = rasterio.open(p["dsm"])
    _CTX["img"] = rasterio.open(p["imagery"]) if p["imagery"].exists() else None
    _CTX["pc"] = PointCloudSource(max_cached_tiles=3)


def _one(bid):
    """Segment + fit one building, and crop its imagery. Returns a dict."""
    import numpy as np
    import rasterio.windows
    from PIL import Image
    from src.roof_segmentation import segment_building_best
    from src.obstruction_detection import detect_obstructions_combined
    from src.panel_fitting import fit_panels_on_facet
    from src.roof_line_source import drawn_segments

    g = _CTX["gdf"]
    if bid not in g.index:
        return None
    geom = g.loc[bid].geometry
    try:
        facets = segment_building_best(_CTX["dsm"], _CTX["pc"], geom, bid,
                                       imagery_ds=_CTX["img"]) or []
    except Exception as exc:
        return {"id": bid, "error": f"{type(exc).__name__}: {exc}"}

    panels = []
    for f in facets:
        if f.get("plane_a") is None:
            continue
        try:
            obs = detect_obstructions_combined(
                _CTX["img"], _CTX["pc"], f["geometry"],
                (f["plane_a"], f["plane_b"], f["plane_c"]),
                roof_geom=f.get("building_geometry")) or []
        except Exception:
            obs = []
        sib = [o for o in facets if o is not f]
        try:
            for pnl in (fit_panels_on_facet(f, obstructions=obs,
                                            sibling_facets=sib) or []):
                panels.append(pnl["geometry"] if isinstance(pnl, dict) else pnl)
        except Exception:
            pass

    b = (geom.bounds[0] - PAD_M, geom.bounds[1] - PAD_M,
         geom.bounds[2] + PAD_M, geom.bounds[3] + PAD_M)
    jpg = ""
    if _CTX["img"] is not None:
        try:
            win = rasterio.windows.from_bounds(*b, _CTX["img"].transform)
            rgb = np.moveaxis(_CTX["img"].read([1, 2, 3], window=win,
                                               boundless=True, fill_value=0),
                              0, -1)
            im = Image.fromarray(rgb.astype("uint8")).resize((PX, PX))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=72)
            jpg = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            jpg = ""

    def ring(poly):
        try:
            return [[round(x, 2), round(y, 2)] for x, y in poly.exterior.coords]
        except Exception:
            return []

    return {
        "id": bid,
        "bounds": [round(v, 2) for v in b],
        "outline": ring(geom),
        "facets": [{"ring": ring(f["geometry"]),
                    "slope": round(f.get("slope_deg", 0), 1),
                    "labels": bool(f.get("from_labels")),
                    "m2": round(f.get("area_m2", 0), 1)} for f in facets],
        "panels": [ring(p) for p in panels],
        "drawn": [list(s) for s in (drawn_segments(bid) or [])],
        "jpg": jpg,
    }


PAGE = """<title>Roof preview</title>
<style>
 :root{--bg:#12161a;--fg:#e8edf2;--mut:#8b97a3;--fac:#ffffff;--pan:#4da3ff;
       --lab:#ffd24d;--drawn:#4dff88}
 body{background:var(--bg);color:var(--fg);font:13px/1.5 ui-sans-serif,system-ui;margin:0;padding:18px}
 h1{font-size:17px;margin:0 0 4px}
 .sub{color:var(--mut);margin-bottom:16px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
 .card{background:#1a1f25;border-radius:6px;padding:8px}
 .card h2{font-size:13px;margin:0 0 2px}
 .meta{color:var(--mut);font-size:11px;margin-bottom:6px}
 canvas{width:100%;height:auto;border-radius:4px;display:block;background:#000}
 .key{color:var(--mut);font-size:11px;margin:10px 0 14px}
 .k{display:inline-block;width:9px;height:9px;border-radius:2px;margin:0 3px 0 12px;vertical-align:middle}
</style>
<h1>Roof preview</h1>
<div class="sub">__SUB__</div>
<div class="key">
 <span class="k" style="background:var(--fac)"></span>facet edge
 <span class="k" style="background:var(--lab)"></span>facet from Josh's markup
 <span class="k" style="background:var(--pan)"></span>panel
 <span class="k" style="background:var(--drawn)"></span>line Josh drew
</div>
<div class="grid" id="g"></div>
<script>
const ROOFS = __ROOFS__;
const g = document.getElementById('g');
for (const r of ROOFS) {
  const d = document.createElement('div'); d.className = 'card';
  const nlab = (r.facets||[]).filter(f=>f.labels).length;
  d.innerHTML = `<h2>#${r.id}</h2><div class="meta">${(r.facets||[]).length} facets`
    + (nlab? ` (${nlab} from markup)`:'') + ` &middot; ${(r.panels||[]).length} panels`
    + (r.error? ` &middot; <span style="color:#ff8080">${r.error}</span>`:'') + `</div>`;
  const c = document.createElement('canvas'); c.width = c.height = 520;
  d.appendChild(c); g.appendChild(d);
  const ctx = c.getContext('2d');
  const [x0,y0,x1,y1] = r.bounds;
  const X = p => (p[0]-x0)/(x1-x0)*520, Y = p => (1-(p[1]-y0)/(y1-y0))*520;
  const draw = () => {
    const poly = (ring, stroke, w, fill) => {
      if (!ring || ring.length<2) return;
      ctx.beginPath(); ctx.moveTo(X(ring[0]), Y(ring[0]));
      for (let i=1;i<ring.length;i++) ctx.lineTo(X(ring[i]), Y(ring[i]));
      ctx.closePath();
      if (fill){ ctx.fillStyle=fill; ctx.fill(); }
      ctx.strokeStyle=stroke; ctx.lineWidth=w; ctx.stroke();
    };
    poly(r.outline, 'rgba(255,180,60,.85)', 2);
    for (const f of (r.facets||[]))
      poly(f.ring, f.labels? 'rgba(255,210,77,.95)':'rgba(255,255,255,.75)', 1.4);
    for (const p of (r.panels||[])) poly(p, 'rgba(77,163,255,.95)', 1, 'rgba(77,163,255,.35)');
    for (const s of (r.drawn||[])) {
      ctx.beginPath(); ctx.moveTo(X([s[0],s[1]]), Y([s[0],s[1]]));
      ctx.lineTo(X([s[2],s[3]]), Y([s[2],s[3]]));
      ctx.strokeStyle='rgba(77,255,136,.9)'; ctx.lineWidth=2; ctx.stroke();
    }
  };
  if (r.jpg) { const im=new Image(); im.onload=()=>{ctx.drawImage(im,0,0,520,520); draw();};
               im.src='data:image/jpeg;base64,'+r.jpg; }
  else draw();
}
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--region", default="pilot")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--labelled", action="store_true",
                    help="sample from roofs Josh has marked complete")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--out", default="preview.html")
    a = ap.parse_args()

    import geopandas as gpd
    from src.region_build import area_paths, all_areas

    labels = {}
    lp = ROOT / "data" / "roof_labels.json"
    if lp.exists():
        labels = json.loads(lp.read_text()).get("buildings", {})

    # IDS ARE GROUPED BY REGION, not matched to one. The first version took
    # whichever region held any of them and silently dropped the rest, which
    # cost the most useful comparison there is: 7 Anderson Heights is in pilot
    # and 107 Beach Street is in town_west_fernhill, and those two together are
    # what showed the detector emits stubs.
    region = a.region
    ids = a.ids
    by_region = {}
    if ids:
        want = set(ids)
        for r in ["pilot"] + [x for x in all_areas() if x != "pilot"]:
            if not want:
                break
            p = area_paths(r)
            dd = p["dir"] / "building_outlines_dedup.geojson"
            src = dd if dd.exists() else p["outlines"]
            if not src.exists():
                continue
            have = {int(x) for x in gpd.read_file(src)["building_id"]}
            hit = sorted(want & have)
            if hit:
                by_region[r] = hit
                want -= set(hit)
        if want:
            print(f"  not found in any region: {sorted(want)}")
    elif a.labelled:
        ids = sorted(int(k) for k, v in labels.items()
                     if v.get("area") == region and v.get("complete"))[:a.n]
    else:
        p = area_paths(region)
        dd = p["dir"] / "building_outlines_dedup.geojson"
        gdf = gpd.read_file(dd if dd.exists() else p["outlines"])
        gdf = gdf.assign(_a=gdf.geometry.area).sort_values("_a", ascending=False)
        ids = [int(x) for x in gdf["building_id"]][:a.n]
    if not ids:
        print("no buildings selected")
        return 1

    if not by_region:
        by_region = {region: ids}
    jobs = a.jobs or max(1, min(8, (__import__("os").cpu_count() or 2) - 1))
    total = sum(len(v) for v in by_region.values())
    print(f"building {total} roofs from {len(by_region)} region(s) on {jobs} workers...")
    import multiprocessing
    ctxm = multiprocessing.get_context("spawn")
    out = []
    for reg, rids in by_region.items():
        print(f"  {reg}: {len(rids)}")
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init,
                                 initargs=(reg,), mp_context=ctxm) as ex:
            for r in ex.map(_one, rids):
                if r:
                    out.append(r)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / a.out
    sub = (f"{len(out)} roofs from {region} &middot; built with the current "
           f"working tree, not the deployed build")
    dest.write_text(PAGE.replace("__ROOFS__", json.dumps(out, separators=(",", ":")))
                        .replace("__SUB__", sub))
    mb = dest.stat().st_size / 1024 / 1024
    nf = sum(len(r.get("facets") or []) for r in out)
    npn = sum(len(r.get("panels") or []) for r in out)
    print(f"  {len(out)} roofs, {nf} facets, {npn} panels")
    print(f"\nwrote {dest}  ({mb:.1f} MB) -- open it and look")
    return 0


if __name__ == "__main__":
    sys.exit(main())

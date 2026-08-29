"""
Run roof_reconstruct against the shipped segmentation on the same buildings
and put the two side by side.

The point is the comparison, not the pretty picture: this stage has twice been
"fixed" on one roof and quietly broken on others, so nothing goes near the
pipeline until a fixed set of known-bad buildings is visibly better and the
known-good ones are unchanged.

Left = what is on the map today. Right = reconstruction. Both over the same
imagery, same dashed-white facet convention.

Usage: python src/compare_reconstruct.py 4735260 4740503 ...
       python src/compare_reconstruct.py --set problem
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
from src.pointcloud_source import PointCloudSource
from src.region_build import area_paths
from src.roof_reconstruct import reconstruct

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True).transform
PAD_M = 4.0

# The set the prototype has to beat, with why each one is here.
PROBLEM_SET = [
    (4735260, "35 Brecon St -- barrel vault modelled as ONE flat facet"),
    (4740503, "7 Panorama Tce -- stepped terrace, planes span the levels"),
    (4734907, "6 Turner St -- hip roof over-segmented, slopes 34-70 deg"),
    (4734815, "111 Hallenstein St -- one side of the roof right, other wrong"),
    (5372608, "19 Camp St -- curved/fan roof"),
    (4725633, "93 Beach St -- 857 panels over 3 facets"),
    (4735015, "5 Isle St -- 228 panels on a single facet"),
    (4734769, "28 Rees St -- 361 panels on a single facet"),
    (5372565, "1 Memorial St"),
    (4735272, "Queenstown centre, 233 of 293 panels off-plane"),
]


def shipped_facets(area):
    lay = json.loads(area_paths(area)["panel_layouts"].read_text())
    by = defaultdict(list)
    for f in lay["features"]:
        if f["properties"].get("kind") == "facet":
            by[int(f["properties"]["building_id"])].append(
                shp_transform(TO_NZTM, shape(f["geometry"])).buffer(0))
    return by


def outlines_nztm(area):
    raw = json.loads(area_paths(area)["outlines"].read_text())
    out = {}
    for f in raw["features"]:
        g = shape(f["geometry"])
        # Outlines are stored NZTM already; detect by magnitude, as audit does.
        if abs(g.bounds[0]) <= 180:
            g = shp_transform(TO_NZTM, g)
        out[int(f["properties"]["building_id"])] = g.buffer(0)
    return out


def draw(ax, imagery, geoms, facets, title, colour_by_plane=True):
    xs = [c for g in geoms for c in g.bounds[0::2]]
    ys = [c for g in geoms for c in g.bounds[1::2]]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    w = rasterio.windows.from_bounds(minx - PAD_M, miny - PAD_M,
                                     maxx + PAD_M, maxy + PAD_M, imagery.transform)
    img = np.moveaxis(imagery.read([1, 2, 3], window=w, boundless=True, fill_value=0), 0, -1)
    wt = imagery.window_transform(w)
    ax.imshow(img, extent=(wt.c, wt.c + img.shape[1] * wt.a,
                           wt.f + img.shape[0] * wt.e, wt.f), origin="upper")
    cmap = plt.get_cmap("turbo")
    for i, f in enumerate(facets):
        g = f["geometry"] if isinstance(f, dict) else f
        col = cmap((i * 0.37) % 1.0) if colour_by_plane else "#7fd4ff"
        for poly in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
            if poly.is_empty:
                continue
            ax.add_patch(MplPolygon(list(zip(*poly.exterior.xy)), closed=True,
                                    facecolor=col, edgecolor="white", lw=1.1,
                                    linestyle=(0, (2, 2)), alpha=0.42))
    ax.set_xlim(minx - PAD_M, maxx + PAD_M)
    ax.set_ylim(miny - PAD_M, maxy + PAD_M)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9, color="#ddd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", type=int)
    ap.add_argument("--set", choices=["problem"], default=None)
    ap.add_argument("--area", default="pilot")
    a = ap.parse_args()
    todo = PROBLEM_SET if a.set == "problem" else [(i, "") for i in a.ids]
    if not todo:
        raise SystemExit("give building ids or --set problem")

    shipped = shipped_facets(a.area)
    outlines = outlines_nztm(a.area)
    pc = PointCloudSource()
    imagery = rasterio.open(area_paths(a.area)["dir"] / "imagery_mosaic.tif")

    cards, stats = [], []
    for bid, why in todo:
        outline = outlines.get(bid)
        old = shipped.get(bid, [])
        if outline is None:
            print(f"  {bid}: no outline, skipped")
            continue
        minx, miny, maxx, maxy = outline.bounds
        pts = pc.points_in_bbox(minx - 1, miny - 1, maxx + 1, maxy + 1, building_only=True)
        import shapely.vectorized
        pts = pts[shapely.vectorized.contains(outline.buffer(0.3), pts[:, 0], pts[:, 1])]
        new, new_obs = reconstruct(bid, outline, pts)

        def offplane(facets):
            """Share of roof points more than 0.35m off the plane they'd sit on
            -- the same test audit_layouts uses to call a panel lumpy."""
            if not facets:
                return None
            worst = []
            for f in facets:
                g = f["geometry"] if isinstance(f, dict) else f
                inside = shapely.vectorized.contains(g, pts[:, 0], pts[:, 1])
                pp = pts[inside]
                if len(pp) < 8:
                    continue
                if isinstance(f, dict):
                    r = f["plane_a"] * pp[:, 0] + f["plane_b"] * pp[:, 1] + f["plane_c"] - pp[:, 2]
                else:
                    from src.roof_reconstruct import fit_plane, residuals
                    r = residuals(fit_plane(pp), pp)
                worst.append((np.abs(r) > 0.35).sum() / len(pp) * len(pp))
            tot = sum(shapely.vectorized.contains(
                f["geometry"] if isinstance(f, dict) else f, pts[:, 0], pts[:, 1]).sum()
                for f in facets)
            return (sum(worst) / tot) if tot else None

        o_old, o_new = offplane(old), offplane(new)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), facecolor="#16151a")
        draw(axes[0], imagery, old or [outline], old,
             f"shipped -- {len(old)} facets" +
             (f", {o_old:.0%} of points off-plane" if o_old is not None else ""),
             colour_by_plane=True)
        draw(axes[1], imagery, [outline], new,
             f"reconstructed -- {len(new)} facets, {len(new_obs)} obstructions" +
             (f", {o_new:.0%} of points off-plane" if o_new is not None else ""))
        for ob in new_obs:
            axes[1].add_patch(MplPolygon(list(zip(*ob["geometry"].exterior.xy)), closed=True,
                                         facecolor="#a855f7", edgecolor="#e9d5ff",
                                         lw=0.8, alpha=0.75))
        fig.tight_layout(pad=0.4)
        buf = io.BytesIO()
        fig.savefig(buf, format="jpg", dpi=105, bbox_inches="tight",
                    facecolor="#16151a", pil_kwargs={"quality": 82, "optimize": True})
        plt.close(fig)
        cards.append({"bid": bid, "why": why, "img": base64.b64encode(buf.getvalue()).decode(),
                      "old": len(old), "new": len(new), "o_old": o_old, "o_new": o_new,
                      "pts": len(pts)})
        stats.append((bid, len(old), len(new), o_old, o_new))
        print(f"  {bid}: {len(old)} -> {len(new)} facets, {len(new_obs)} obs   off-plane "
              f"{'--' if o_old is None else f'{o_old:.0%}'} -> "
              f"{'--' if o_new is None else f'{o_new:.0%}'}   ({len(pts)} pts)")
    imagery.close()

    out = DATA_DIR / "reconstruct_compare.html"
    out.write_text(build_html(cards))
    print(f"\nSaved {out} ({out.stat().st_size / 1e6:.1f}MB)")


def build_html(cards):
    def pct(v):
        return "&mdash;" if v is None else f"{v:.0%}"

    def card(c):
        # Off-plane alone cannot decide this. Cutting a roof into more pieces
        # ALWAYS lowers the residual, so a meaningless diagonal across a flat
        # roof scores as an improvement (Josh, on 5 Isle St). A reconstruction
        # only counts as better if it also did not fragment the roof.
        o_old, o_new = c["o_old"], c["o_new"]
        if o_old is None or o_new is None:
            verdict = "level"
        else:
            fragmented = c["new"] > max(6, 2 * c["old"])
            if o_new < o_old - 0.02 and not fragmented:
                verdict = "better"
            elif o_new > o_old + 0.02:
                verdict = "worse"
            elif fragmented:
                verdict = "fragmented"
            else:
                verdict = "level"
        return f"""<figure class="card">
  <figcaption class="head">
    <span class="bid">#{c['bid']}</span>
    <span class="verdict v-{verdict}">{verdict}</span>
  </figcaption>
  <p class="why">{c['why']}</p>
  <img src="data:image/jpeg;base64,{c['img']}" alt="building {c['bid']}">
  <table>
    <tr><th></th><th>shipped</th><th>reconstructed</th></tr>
    <tr><td>facets</td><td>{c['old']}</td><td>{c['new']}</td></tr>
    <tr><td>points off-plane</td><td>{pct(c['o_old'])}</td><td>{pct(c['o_new'])}</td></tr>
  </table>
</figure>"""
    return f"""<title>Roof reconstruction trial</title>
<style>
  :root {{ --bg:#faf9f7; --panel:#fff; --ink:#17171a; --ink-2:#5c5c66; --line:#e2e0dc;
    --good:#0f766e; --bad:#b91c1c; --level:#78716c; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
    --bg:#141317; --panel:#1e1d23; --ink:#f2f0ec; --ink-2:#a5a1a9; --line:#312f38;
    --good:#5eead4; --bad:#fca5a5; --level:#a8a29e; }} }}
  :root[data-theme="dark"] {{ --bg:#141317; --panel:#1e1d23; --ink:#f2f0ec;
    --ink-2:#a5a1a9; --line:#312f38; --good:#5eead4; --bad:#fca5a5; --level:#a8a29e; }}
  body {{ background:var(--bg); color:var(--ink); margin:0; padding:26px;
    font:14px/1.55 ui-sans-serif,-apple-system,"Helvetica Neue",sans-serif; }}
  h1 {{ font-size:21px; margin:0 0 6px; }}
  .sub {{ color:var(--ink-2); max-width:76ch; margin:0 0 24px; }}
  .grid {{ display:grid; gap:20px; }}
  .card {{ margin:0; background:var(--panel); border:1px solid var(--line);
    border-radius:9px; padding:12px 14px 14px; }}
  .head {{ display:flex; justify-content:space-between; align-items:baseline; }}
  .bid {{ font-weight:700; font-variant-numeric:tabular-nums; }}
  .verdict {{ font-size:11px; font-weight:700; letter-spacing:.07em; text-transform:uppercase; }}
  .v-better {{ color:var(--good); }} .v-worse {{ color:var(--bad); }}
  .v-level {{ color:var(--level); }} .v-fragmented {{ color:var(--bad); }}
  .why {{ color:var(--ink-2); font-size:12.5px; margin:2px 0 10px; }}
  .card img {{ display:block; width:100%; height:auto; border-radius:5px; }}
  table {{ width:100%; margin-top:10px; border-collapse:collapse; font-size:12.5px;
    font-variant-numeric:tabular-nums; }}
  th, td {{ text-align:right; padding:3px 6px; border-bottom:1px solid var(--line); }}
  th:first-child, td:first-child {{ text-align:left; color:var(--ink-2); }}
</style>
<h1>Roof reconstruction trial</h1>
<p class="sub">Left is the roof model on the map today; right is the same building rebuilt as
planes joined along their intersection lines and clipped to the surveyed outline. Colours
separate facets. &ldquo;Points off-plane&rdquo; is the share of LiDAR returns sitting more than
0.35&thinsp;m from the plane they would be placed on &mdash; the same test that calls a panel
lumpy. Lower is better; it is the number that says whether the roof model actually describes
the roof.</p>
<div class="grid">
{chr(10).join(card(c) for c in cards)}
</div>"""


if __name__ == "__main__":
    main()

"""
Build a sheet of roofs to be LABELLED with the correct answer, so changes to
segmentation can be measured against what is right rather than against a proxy.

Why this exists: every metric available so far rewards cutting a roof into more
pieces. Off-plane residual always falls when you add a plane, so a meaningless
diagonal across a flat deck scores as an improvement -- which is what happened
on 5 Isle St, and Josh had to catch it by eye. Threshold tuning against that
metric trades one roof against another indefinitely.

A label here is deliberately small and checkable: how many distinct roof PLANES
the building should have, and what kind of roof it is. That is enough to score
the failure that actually matters -- over- and under-segmentation -- without
anyone having to trace polygons.

Each card shows the aerial photo and, beside it, a hillshade built from the
LiDAR alone. The hillshade is the point: ridges, hips and steps are obvious in
it, and it carries no output from either model, so the judgement is not primed
by the thing being judged.

Usage: python src/label_sheet.py --area pilot --n 20
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
from scipy import ndimage
from scipy.interpolate import griddata
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pointcloud_source import PointCloudSource
from src.region_build import area_paths
from src.roof_reconstruct import plane_slope_aspect, ransac_planes, reconstruct

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True).transform
PAD_M = 3.0
GRID = 0.25

# (id, note, answer-already-given, question-I-cannot-settle-myself)
SEEDS = [
    # Corrected by Josh from a 3D view. My first reading turned his "not that
    # diagonal" into "one plane", which was over-reading him: the roof is three
    # decks at three heights, separated by STEPS rather than by one arbitrary
    # diagonal. Shipped says 1, reconstruction says 4.
    (4735015, "three decks at three heights (Josh, from a 3D view)",
     (3, "three decks at three heights, one at a different pitch"), None),
    # My propose() heuristic called this "hip roof, four faces", which is
    # nonsense on a stepped commercial roof -- it fires on any building with
    # four dominant aspects. Josh counted seven from a 3D view.
    (5372565, "seven clear planes (Josh, from a 3D view)",
     (7, "stepped commercial roof, seven clear planes"), None),
    (4734815, "four faces, confirmed; ridges still sit off the imagery",
     (4, "hip roof, four faces"), None),
    (4735260, "barrel vault", None,
     "Curved, so no number of planes is strictly right. One plane for racking "
     "purposes, or strips following the curve?"),
    # I proposed 2 here and called it evidence that both models shred houses.
    # Josh says 9, which is what the reconstruction said. My propose() heuristic
    # is the thing that under-counts, not the models.
    (4734907, "nine planes (Josh)", (9, "nine planes"), None),
    (4750866, "two planes (Josh); the wing sits about 4m lower",
     (2, "two planes"), None),
    # Not every roof yields an honest count. Where it does not, a pairwise
    # preference between the two models is easier to give and still scores
    # them -- see roof_labels_*.json, kind: "preference".
    (5371149, "very complicated; Josh prefers the reconstruction here", None, None),
]


def hillshade(pts, bounds, az=315.0, alt=45.0):
    """Shaded relief from the roof points only -- no model output drawn on it."""
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx + GRID, GRID)
    ys = np.arange(miny, maxy + GRID, GRID)
    gx, gy = np.meshgrid(xs, ys)
    z = griddata(pts[:, :2], pts[:, 2], (gx, gy), method="linear")
    if np.isnan(z).all():
        return None
    z = np.where(np.isnan(z), np.nanmedian(z), z)
    z = ndimage.gaussian_filter(z, 0.8)
    dy, dx = np.gradient(z, GRID)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    a, A = np.radians(alt), np.radians(az)
    return np.clip(np.sin(a) * np.cos(slope)
                   + np.cos(a) * np.sin(slope) * np.cos(A - aspect), 0, 1)


def propose(planes, pts):
    """A first guess at the label, to be corrected rather than trusted."""
    if not planes:
        return "unknown", 1
    info = []
    for p in planes:
        s, asp = plane_slope_aspect(p)
        n = int((np.abs(p[0] * pts[:, 0] + p[1] * pts[:, 1] + p[2] - pts[:, 2]) < 0.15).sum())
        info.append((s, asp, n))
    info.sort(key=lambda t: -t[2])
    total = max(sum(i[2] for i in info), 1)
    big = [i for i in info if i[2] / total > 0.12]
    if not big:
        return "unknown", 1
    flat = [i for i in big if i[0] < 6]
    if len(big) == 1:
        return ("flat roof, one plane" if big[0][0] < 6 else "one pitched plane"), 1
    if len(flat) == len(big):
        return "flat roof (with falls), one plane", 1
    if len(big) == 2:
        d = abs(big[0][1] - big[1][1]) % 360
        if 150 < min(d, 360 - d) < 210:
            return "simple gable, two faces", 2
        return "two faces", 2
    if len(big) == 4:
        return "hip roof, four faces", 4
    return f"complex, {len(big)} faces", len(big)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="pilot")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    lay = json.loads(area_paths(a.area)["panel_layouts"].read_text())
    shipped = defaultdict(int)
    for f in lay["features"]:
        if f["properties"].get("kind") == "facet":
            shipped[int(f["properties"]["building_id"])] += 1

    raw = json.loads(area_paths(a.area)["outlines"].read_text())
    outl = {}
    for f in raw["features"]:
        g = shape(f["geometry"])
        outl[int(f["properties"]["building_id"])] = (
            shp_transform(TO_NZTM, g) if abs(g.bounds[0]) <= 180 else g).buffer(0)

    sp = json.loads(area_paths(a.area)["solar_potential"].read_text())
    addr = {int(f["properties"]["building_id"]): f["properties"].get("address", "")
            for f in sp["features"]}

    # Mostly houses -- Josh: "households are the more important check, they are
    # the rooftops you have failed on more often" -- plus two larger roofs so
    # the commercial failure mode stays represented.
    rng = np.random.default_rng(a.seed)
    seeded = [t[0] for t in SEEDS]
    houses = [b for b, o in outl.items()
              if 70 <= o.area <= 260 and b in shipped and b not in seeded]
    biggies = [b for b, o in outl.items()
               if 400 <= o.area <= 1500 and b in shipped and b not in seeded]
    n_house = max(a.n - len(SEEDS) - 2, 0)
    pick = (list(SEEDS)
            + [(int(b), "", None, None) for b in rng.choice(houses, size=min(n_house, len(houses)), replace=False)]
            + [(int(b), "", None, None) for b in rng.choice(biggies, size=min(2, len(biggies)), replace=False)])

    pc = PointCloudSource()
    imagery = rasterio.open(area_paths(a.area)["dir"] / "imagery_mosaic.tif")
    cards = []
    for idx, (bid, why, known, ask) in enumerate(pick, 1):
        o = outl.get(bid)
        if o is None:
            continue
        minx, miny, maxx, maxy = o.bounds
        pts = pc.points_in_bbox(minx - 1, miny - 1, maxx + 1, maxy + 1, building_only=True)
        pts = pts[shapely.vectorized.contains(o.buffer(0.3), pts[:, 0], pts[:, 1])]
        if len(pts) < 40:
            continue
        planes, _ = ransac_planes(pts, np.random.default_rng(0))
        label, n_expect = propose(planes, pts)
        if known is not None:
            n_expect, label = known
        try:
            rec, _obs = reconstruct(bid, o, pts)
        except Exception:
            rec = []

        w = rasterio.windows.from_bounds(minx - PAD_M, miny - PAD_M,
                                         maxx + PAD_M, maxy + PAD_M, imagery.transform)
        img = np.moveaxis(imagery.read([1, 2, 3], window=w, boundless=True, fill_value=0), 0, -1)
        wt = imagery.window_transform(w)
        ext = (wt.c, wt.c + img.shape[1] * wt.a, wt.f + img.shape[0] * wt.e, wt.f)
        hs = hillshade(pts, (minx - PAD_M, miny - PAD_M, maxx + PAD_M, maxy + PAD_M))

        fig, ax = plt.subplots(1, 2, figsize=(8.4, 4.2), facecolor="#15141a")
        ax[0].imshow(img, extent=ext, origin="upper")
        ax[0].set_title("aerial", fontsize=8, color="#ccc")
        if hs is not None:
            ax[1].imshow(hs, extent=(minx - PAD_M, maxx + PAD_M, miny - PAD_M, maxy + PAD_M),
                         origin="lower", cmap="gray")
        ax[1].set_title("LiDAR relief", fontsize=8, color="#ccc")
        for k in ax:
            k.plot(*o.exterior.xy, color="#ffd166", lw=1.0)
            k.set_xlim(minx - PAD_M, maxx + PAD_M)
            k.set_ylim(miny - PAD_M, maxy + PAD_M)
            k.set_aspect("equal")
            k.axis("off")
        fig.tight_layout(pad=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="jpg", dpi=105, bbox_inches="tight",
                    facecolor="#15141a", pil_kwargs={"quality": 82, "optimize": True})
        plt.close(fig)
        cards.append({"i": idx, "bid": bid, "addr": addr.get(bid, ""), "why": why,
                      "known": known is not None, "ask": ask,
                      "label": label, "expect": n_expect,
                      "shipped": shipped.get(bid, 0), "rec": len(rec),
                      "roof": round(o.area), "img": base64.b64encode(buf.getvalue()).decode()})
        print(f"  {idx:2d}. {bid} {addr.get(bid,''):<26} {round(o.area):>4} m2  "
              f"proposed: {label} ({n_expect})   shipped {shipped.get(bid,0)}  rec {len(rec)}")
    imagery.close()

    out = DATA_DIR / f"label_sheet_{a.area}.html"
    out.write_text(build_html(cards))
    (DATA_DIR / f"roof_labels_{a.area}.json").write_text(json.dumps(
        {str(c["bid"]): {"n": c["expect"], "label": c["label"], "confirmed": bool(c["known"])}
         for c in cards}, indent=1))
    print(f"\nSaved {out} ({out.stat().st_size/1e6:.1f}MB) and roof_labels_{a.area}.json")


def build_html(cards):
    def card(c):
        return f"""<figure class="card">
  <figcaption class="head">
    <span class="n">{c['i']}</span>
    <span class="addr">{c['addr'] or ('#' + str(c['bid']))}</span>
    <span class="roof">{c['roof']}&thinsp;m&sup2;</span>
  </figcaption>
  {f'<p class="why">{c["why"]}</p>' if c['why'] else ''}
  <img src="data:image/jpeg;base64,{c['img']}" alt="{c['addr']}">
  <p class="prop">{'Your call: ' if c['known'] else 'I say: '}<b>{c['label']}</b> &mdash; <b>{c['expect']}</b> plane{'' if c['expect']==1 else 's'}</p>
  {f'<p class="ask">{c["ask"]}</p>' if c.get("ask") else ''}
  <p class="models">shipped model: {c['shipped']} &middot; reconstruction: {c['rec']}</p>
</figure>"""
    return f"""<title>Roof labels to check</title>
<style>
  :root {{ --bg:#f8f7f4; --panel:#fff; --ink:#16161a; --ink-2:#5d5d67; --line:#e4e1dc; --mark:#b45309; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
    --bg:#131217; --panel:#1d1c22; --ink:#f1efec; --ink-2:#a6a2aa; --line:#302e37; --mark:#fbbf24; }} }}
  :root[data-theme="dark"] {{ --bg:#131217; --panel:#1d1c22; --ink:#f1efec;
    --ink-2:#a6a2aa; --line:#302e37; --mark:#fbbf24; }}
  body {{ background:var(--bg); color:var(--ink); margin:0; padding:26px;
    font:14px/1.55 ui-sans-serif,-apple-system,"Helvetica Neue",sans-serif; }}
  h1 {{ font-size:21px; margin:0 0 6px; }}
  .sub {{ color:var(--ink-2); max-width:74ch; margin:0 0 22px; }}
  .grid {{ display:grid; gap:18px; grid-template-columns:repeat(auto-fill,minmax(430px,1fr)); }}
  .card {{ margin:0; background:var(--panel); border:1px solid var(--line);
    border-radius:9px; padding:11px 13px 13px; }}
  .head {{ display:flex; align-items:baseline; gap:9px; }}
  .n {{ background:var(--mark); color:#1a1a1a; font-weight:800; border-radius:4px;
    padding:0 7px; font-size:12px; }}
  .addr {{ font-weight:700; flex:1; }}
  .roof {{ color:var(--ink-2); font-size:12px; font-variant-numeric:tabular-nums; }}
  .why {{ color:var(--mark); font-size:12.5px; margin:5px 0 0; }}
  .ask {{ margin:5px 0 0; font-size:12.5px; color:var(--mark); border-left:2px solid var(--mark);
    padding-left:8px; }}
  .card img {{ display:block; width:100%; height:auto; border-radius:5px; margin-top:9px; }}
  .prop {{ margin:9px 0 2px; font-size:13px; }}
  .models {{ margin:0; color:var(--ink-2); font-size:12px; font-variant-numeric:tabular-nums; }}
</style>
<h1>Roof labels to check</h1>
<p class="sub">For each roof: how many distinct planes should it have? The right-hand image is
shaded relief built from the LiDAR alone &mdash; ridges, hips and steps show up in it, and it
carries no output from either model, so it will not prime the answer. I have proposed a label
on each; <strong>reply with just the numbers I got wrong</strong> (e.g. &ldquo;3: 2 planes,
7: hip 4&rdquo;). Silence on a card means I had it right. A flat roof with drainage falls counts
as <em>one</em> plane &mdash; the call you already made on 5 Isle St.</p>
<div class="grid">
{chr(10).join(card(c) for c in cards)}
</div>"""


if __name__ == "__main__":
    main()

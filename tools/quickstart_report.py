"""Render the quickstart area as a self-contained verification report.

One card per building: aerial imagery with the built facet boundaries and
placed panels drawn over it, plus the numbers derived for that roof and
which geometry path produced it (markup / selected vision faces / LiDAR
partition). The point is that a stranger can hold the output against the
photograph and against docs/quickstart.md's checks.

    python tools/quickstart_report.py <area>
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import base64
    import io
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import geopandas as gpd
    import pyproj
    import rasterio
    import rasterio.windows
    from PIL import Image, ImageDraw
    from shapely.geometry import shape
    from shapely.ops import transform as shtr
    from src.region_build import area_paths

    area = sys.argv[1]
    p = area_paths(area)
    layouts = json.load(open(p["dir"] / "panel_layouts.geojson"))
    sp = json.load(open(p["dir"] / "solar_potential.geojson"))
    img = rasterio.open(p["imagery"]) if Path(p["imagery"]).exists() else None
    to_nztm = pyproj.Transformer.from_crs(4326, 2193, always_xy=True).transform

    by = {}
    for f in layouts["features"]:
        pr = f["properties"]
        by.setdefault(pr["building_id"], {"facet": [], "panel": [],
                                          "obstruction": []})
        if pr["kind"] in by[pr["building_id"]]:
            by[pr["building_id"]][pr["kind"]].append(f)
    stats = {f["properties"]["building_id"]: f["properties"]
             for f in sp["features"]}

    cards = []
    tot_panels = tot_kwh = 0
    for bid, d in sorted(by.items()):
        st = stats.get(bid, {})
        tot_panels += st.get("panel_count", 0) or 0
        tot_kwh += st.get("ac_kwh_year", 0) or 0
        if img is None or not d["facet"]:
            continue
        geoms = [shtr(to_nztm, shape(f["geometry"])) for f in d["facet"]]
        minx = min(g.bounds[0] for g in geoms) - 4
        miny = min(g.bounds[1] for g in geoms) - 4
        maxx = max(g.bounds[2] for g in geoms) + 4
        maxy = max(g.bounds[3] for g in geoms) + 4
        try:
            win = rasterio.windows.from_bounds(minx, miny, maxx, maxy,
                                               img.transform)
            rgb = np.moveaxis(img.read([1, 2, 3], window=win, boundless=True,
                                       fill_value=0), 0, -1).astype("uint8")
        except Exception:
            continue
        h, w = rgb.shape[:2]
        if h < 24 or w < 24:
            continue
        im = Image.fromarray(rgb).resize((w * 2, h * 2))
        dr = ImageDraw.Draw(im)

        def tp(x, y):
            return ((x - minx) / (maxx - minx) * w * 2,
                    (1 - (y - miny) / (maxy - miny)) * h * 2)

        srcs = set()
        for f, g in zip(d["facet"], geoms):
            pr = f["properties"]
            srcs.add("markup" if pr.get("from_labels")
                     else "vision" if pr.get("from_selected") else "lidar")
            dr.line([tp(*c) for c in g.exterior.coords],
                    fill=(255, 255, 255), width=2)
        for f in d["obstruction"]:
            g = shtr(to_nztm, shape(f["geometry"]))
            dr.line([tp(*c) for c in g.exterior.coords],
                    fill=(230, 90, 90), width=2)
        for f in d["panel"]:
            g = shtr(to_nztm, shape(f["geometry"]))
            dr.polygon([tp(*c) for c in g.exterior.coords],
                       outline=(120, 170, 255))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=70)
        cards.append({
            "id": bid, "img": base64.b64encode(buf.getvalue()).decode(),
            "facets": len(d["facet"]), "panels": st.get("panel_count", 0),
            "kwh": st.get("ac_kwh_year", 0), "src": "+".join(sorted(srcs)),
            "reason": st.get("no_estimate_reason") or st.get("reason") or "",
        })

    html = ["<title>Quickstart Verification Report</title>",
            "<style>body{background:#14161a;color:#e8e8e8;"
            "font:14px system-ui;margin:0;padding:18px}"
            ".card{margin-bottom:22px}img{max-width:92vw;border-radius:8px}"
            "h3{margin:8px 0 2px}.lbl{color:#9aa;font-size:12px}"
            ".sum{background:#1d2026;padding:12px 16px;border-radius:8px;"
            "margin-bottom:20px}</style>",
            f"<h2>Quickstart: {area}</h2>",
            f"<div class=sum>{len(by)} buildings &middot; "
            f"{tot_panels:,} potential panels &middot; "
            f"{tot_kwh/1000:,.0f} MWh/yr &middot; white = roof facets, "
            f"red = detected obstructions, blue = placed panels. Geometry "
            f"source per building is labelled (markup = a hand-drawn roof, "
            f"vision = the SAM/line/LiDAR selected reading, lidar = the "
            f"LiDAR partition fallback). Verify with docs/quickstart.md."
            f"</div>"]
    for c in cards:
        html.append(
            f"<div class=card><h3>#{c['id']}</h3>"
            f"<div class=lbl>{c['facets']} facets ({c['src']}) &middot; "
            f"{c['panels']} panels &middot; {c['kwh']:,.0f} kWh/yr"
            + (f" &middot; no estimate: {c['reason']}" if c['reason'] else "")
            + f"</div><img src='data:image/jpeg;base64,{c['img']}'></div>")
    out = p["dir"] / "quickstart_report.html"
    out.write_text("\n".join(html))
    print(f"wrote {out}  ({len(cards)} buildings rendered)")


if __name__ == "__main__":
    main()

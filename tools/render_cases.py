"""Contact sheet of the case roofs awaiting a verdict.

Red = the facets the build produces now. Green = the drawn faces, where
they were drawn. One tile per roof, captioned with its note, so a verdict
costs a glance rather than a navigation.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, rasterio, rasterio.windows, geopandas as gpd, pyproj
from PIL import Image, ImageDraw, ImageFont
from shapely.ops import transform as sht
from shapely.geometry import shape as shp
from src.region_build import area_paths, area_centroid_wgs84
from src.solar_model import SolarModel
import src.build_layout_geojson as blg

STATUSES = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else {"needs_verdict"}
OUT = sys.argv[1]
reg = json.loads((ROOT / "data/roof_cases.json").read_text())
labs = json.loads((ROOT / "data/roof_labels.json").read_text())["buildings"]
cases = [c for c in reg["cases"] if c.get("status") in STATUSES]
to_nztm = pyproj.Transformer.from_crs(4326, 2193, always_xy=True).transform

T, COLS, CAP = 430, 3, 46
rows = (len(cases) + COLS - 1) // COLS
sheet = Image.new("RGB", (COLS * T, rows * (T + CAP)), (16, 16, 18))
d0 = ImageDraw.Draw(sheet)
try:
    f_big = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    f_sm = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
except Exception:
    f_big = f_sm = ImageFont.load_default()

ctx = {}
for n, c in enumerate(cases):
    bid, region = c["id"], c.get("region")
    if region not in ctx:
        p = area_paths(region)
        dd = p["dir"] / "building_outlines_dedup.geojson"
        if not p["imagery"].exists():
            continue
        g = gpd.read_file(dd if dd.exists() else p["outlines"]).set_index("building_id", drop=False)
        cc = area_centroid_wgs84(region)
        blg._init_worker(region, SolarModel(*cc) if cc else SolarModel())
        ctx[region] = (g, rasterio.open(p["imagery"]))
    g, img = ctx[region]
    if bid not in g.index:
        continue
    geom = g.loc[bid].geometry
    try:
        feats = blg._build_one(bid)
    except Exception:
        feats = []
    mnx, mny, mxx, mxy = geom.bounds
    pad = max(mxx - mnx, mxy - mny) * 0.08 + 2
    span = max(mxx - mnx, mxy - mny) / 2 + pad
    cx, cy = (mnx + mxx) / 2, (mny + mxy) / 2
    b = (cx - span, cy - span, cx + span, cy + span)
    win = rasterio.windows.from_bounds(*b, img.transform)
    rgb = np.moveaxis(img.read([1, 2, 3], window=win, boundless=True,
                               fill_value=0, out_shape=(3, T, T)), 0, -1)
    im = Image.fromarray(rgb.astype("uint8")); dd2 = ImageDraw.Draw(im)
    px = lambda x, y: ((x - b[0]) / (b[2] - b[0]) * T,
                       (1 - (y - b[1]) / (b[3] - b[1])) * T)
    lab = labs.get(str(bid))
    if lab:
        for f in (lab.get("faces") or []):
            if f.get("ring"):
                r = [px(x, y) for x, y in f["ring"]]
                dd2.line(r + [r[0]], fill=(70, 255, 130), width=2)
    nf = 0
    for f in feats:
        if f["properties"]["kind"] != "facet":
            continue
        gg = sht(to_nztm, shp(f["geometry"]))
        try:
            r = [px(x, y) for x, y in gg.exterior.coords]
        except Exception:
            continue
        dd2.line(r + [r[0]], fill=(255, 65, 65), width=3); nf += 1
    npan = sum(1 for f in feats if f["properties"]["kind"] == "panel")
    ox, oy = (n % COLS) * T, (n // COLS) * (T + CAP)
    sheet.paste(im, (ox, oy))
    d0.rectangle([ox, oy + T, ox + T, oy + T + CAP], fill=(16, 16, 18))
    d0.text((ox + 6, oy + T + 4),
            f"#{bid}   {nf} facets, {npan} panels"
            + ("   (green = your markup)" if lab else ""),
            fill=(255, 230, 120), font=f_big)
    q = (c.get("quotes") or [c["defect"]])[0]
    d0.text((ox + 6, oy + T + 24), '"' + q[:62] + ('…"' if len(q) > 62 else '"'),
            fill=(165, 172, 182), font=f_sm)
sheet.save(OUT)
print(f"{len(cases)} cases -> {OUT}")

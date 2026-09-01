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
MAX_PX = 820          # per-roof crop; bigger is nicer to draw on but heavier


def crop(imagery, bounds):
    import numpy as np
    import rasterio.windows
    from PIL import Image
    minx, miny, maxx, maxy = bounds
    w = rasterio.windows.from_bounds(minx, miny, maxx, maxy, imagery.transform)
    rgb = np.moveaxis(imagery.read([1, 2, 3], window=w,
                                   boundless=True, fill_value=0), 0, -1)
    im = Image.fromarray(rgb.astype("uint8"))
    if max(im.size) > MAX_PX:
        s = MAX_PX / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--out", default="mark_roofs.html")
    a = ap.parse_args()

    import geopandas as gpd
    import rasterio
    from src.region_build import area_paths, all_areas

    ids = a.ids
    if ids is None:
        q = OUT_DIR / "queue.json"
        if not q.exists():
            print("no queue.json -- run tools/sample_roofs_to_label.py first")
            return 2
        ids = json.loads(q.read_text())["ids"]
    if a.max:
        ids = ids[:a.max]

    truth = {}
    tp = DATA_DIR / "roof_truth.json"
    if tp.exists():
        for r in json.loads(tp.read_text()).get("roofs", []):
            if r.get("building_id"):
                truth[int(r["building_id"])] = r

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
            t = truth.get(bid, {})
            roofs.append({
                "id": bid, "area": name,
                "address": t.get("address", ""),
                "m2": round(g.area, 1),
                "bounds": [round(v, 2) for v in b],
                "px": list(size),
                "outline": [[round(x, 2), round(y, 2)] for x, y in g.exterior.coords],
                "neighbours": nb,
                "jpg": jpg,
                "note": (f"You said {t['faces']} faces. " if t.get("faces") else "")
                        + (t.get("structure", "")[:180] if t.get("structure") else ""),
            })
            placed = True
            break
        if not placed:
            print(f"  skip #{bid} (no imagery on this machine)")
        if i % 25 == 0:
            print(f"  {i}/{len(ids)}")

    html = (ROOT / "tools" / "label_template.html").read_text()
    html = html.replace("/*__ROOFS__*/", json.dumps(roofs, separators=(",", ":")))
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

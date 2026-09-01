"""
Roof labelling server: draw true roof geometry in the browser, save it as vectors.

Track A of the vision pathway. mark_roofs.py renders a roof for Josh to draw on
with a pen; this replaces the pen with a canvas and, crucially, saves what he
draws as COORDINATES rather than as a picture and a sentence. Today's ground
truth is 28 roofs of which only 4 carry traced lines -- the rest are prose like
"8 faces on this one", which can score a face COUNT and nothing else. No model
can be trained on that, and no boundary error can be measured against it.

The point is throughput. A vision model wants a few hundred roofs, and the only
irreplaceable cost in the whole pathway is Josh's time drawing them, so
everything here exists to make a roof take a minute instead of five.

ANCHORING, deliberately handled rather than ignored. mark_roofs.py shows imagery
ONLY, on purpose: "showing the current faces would anchor the answer to what the
pipeline already believes, which is the thing under question." That reasoning is
still right. But drawing every line from scratch is what makes labelling slow.
So the model's guess can be loaded as a starting point OR the roof can be
started empty, it is one keystroke to clear the lot, and every saved label
records which mode it came from -- so if pre-labelled roofs later look
suspiciously like the segmenter, that is measurable instead of invisible.

Coordinates are NZTM (EPSG:2193) metres, the same frame as the outlines and the
point cloud, so a label can be compared to a facet without transforming.

Usage:
    python tools/label_roofs.py                 # serve on :8020, all marked roofs
    python tools/label_roofs.py --area pilot --port 8020
    python tools/label_roofs.py --ids 5371108 4735015

Labels are written to data/roof_labels.json after every save, so killing the
server never loses work.
"""

import argparse
import base64
import io
import json
import sys
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
LABELS = DATA_DIR / "roof_labels.json"
PAD_M = 3.0
MAX_PX = 1100          # longest edge of the served crop


def _load_labels():
    if LABELS.exists():
        try:
            return json.loads(LABELS.read_text())
        except Exception:
            pass
    return {"_note": ("Roof geometry drawn by Josh in tools/label_roofs.py. "
                      "Coordinates are NZTM (EPSG:2193) metres. `lines` are "
                      "roof lines by kind; `obstructions` are closed rings. "
                      "`seeded` records whether the model's guess was loaded "
                      "as a starting point, so anchoring can be checked."),
            "buildings": {}}


def _save_labels(d):
    tmp = LABELS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True))
    tmp.replace(LABELS)


def _truth_index():
    """What Josh has already said about these roofs, in prose or coordinates."""
    p = DATA_DIR / "roof_truth.json"
    if not p.exists():
        return {}
    out = {}
    for r in json.loads(p.read_text()).get("roofs", []):
        if r.get("building_id"):
            out[int(r["building_id"])] = r
    return out


class Bundles:
    """Per-area rasters and outlines, opened once and reused."""

    def __init__(self):
        self._areas = {}

    def area(self, name):
        if name in self._areas:
            return self._areas[name]
        import geopandas as gpd
        import rasterio
        from src.region_build import area_paths
        p = area_paths(name)
        if not p["outlines"].exists():
            return None
        dedup = p["dir"] / "building_outlines_dedup.geojson"
        ctx = {
            "gdf": gpd.read_file(dedup if dedup.exists() else p["outlines"]
                                 ).set_index("building_id", drop=False),
            "imagery": rasterio.open(p["imagery"]) if p["imagery"].exists() else None,
            "paths": p,
        }
        self._areas[name] = ctx
        return ctx

    def find(self, bid, areas):
        for name in areas:
            ctx = self.area(name)
            if ctx is not None and bid in ctx["gdf"].index:
                return name, ctx
        return None, None


def _crop_png(imagery, bounds):
    """Base64 PNG of the imagery over `bounds`, plus its exact extent."""
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
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), im.size


def _model_guess(area, bid, ctx):
    """The current segmenter's facet boundaries, as NZTM rings.

    Offered as a STARTING POINT only. See the anchoring note in the module
    docstring -- this is the thing the labels are supposed to be judging.
    """
    try:
        import rasterio
        from src.roof_segmentation import segment_building_best
        from src.pointcloud_source import PointCloudSource
        p = ctx["paths"]
        dsm = rasterio.open(p["dsm"])
        facets = segment_building_best(
            dsm, PointCloudSource(), ctx["gdf"].loc[bid].geometry, bid,
            imagery_ds=ctx["imagery"]) or []
        return [[[round(x, 3), round(y, 3)] for x, y in f["geometry"].exterior.coords]
                for f in facets]
    except Exception as exc:
        print(f"  (no model guess for {bid}: {type(exc).__name__}: {exc})")
        return []


def build_bundle(bid, bundles, areas, truth, want_guess=True):
    area, ctx = bundles.find(bid, areas)
    if ctx is None:
        return {"error": f"building {bid} not found in {', '.join(areas)}"}
    g = ctx["gdf"].loc[bid].geometry
    minx, miny, maxx, maxy = g.bounds
    bounds = (minx - PAD_M, miny - PAD_M, maxx + PAD_M, maxy + PAD_M)
    png, size = ("", (0, 0))
    if ctx["imagery"] is not None:
        png, size = _crop_png(ctx["imagery"], bounds)
    t = truth.get(bid, {})
    return {
        "building_id": bid,
        "area": area,
        "address": t.get("address") or "",
        "bounds": [round(v, 3) for v in bounds],
        "size_px": list(size),
        "footprint": [[round(x, 3), round(y, 3)] for x, y in g.exterior.coords],
        "area_m2": round(g.area, 1),
        "imagery_png": png,
        "model_guess": _model_guess(area, bid, ctx) if want_guess else [],
        # What Josh already said about this roof, shown as a hint beside it.
        "known": {k: t[k] for k in ("faces", "obstructions_marked", "structure",
                                    "key_finding") if k in t},
    }


class Handler(BaseHTTPRequestHandler):
    bundles = None
    areas = ()
    truth = {}
    queue = []

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            html = (ROOT / "tools" / "label_roofs.html").read_bytes()
            return self._send(200, html, "text/html; charset=utf-8")
        if u.path == "/api/queue":
            done = set(_load_labels()["buildings"])
            return self._send(200, json.dumps({
                "ids": self.queue,
                "done": sorted(int(k) for k in done),
            }))
        if u.path.startswith("/api/building/"):
            bid = int(u.path.rsplit("/", 1)[1])
            want = "noguess" not in (u.query or "")
            b = build_bundle(bid, self.bundles, self.areas, self.truth, want)
            existing = _load_labels()["buildings"].get(str(bid))
            if existing:
                b["existing"] = existing
            return self._send(200, json.dumps(b))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        u = urlparse(self.path)
        if not u.path.startswith("/api/labels/"):
            return self._send(404, json.dumps({"error": "not found"}))
        bid = u.path.rsplit("/", 1)[1]
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        d = _load_labels()
        d["buildings"][bid] = payload
        _save_labels(d)
        n_lines = len(payload.get("lines", []))
        n_obs = len(payload.get("obstructions", []))
        print(f"  saved #{bid}: {n_lines} lines, {n_obs} obstructions"
              f"{' (seeded from model)' if payload.get('seeded') else ''}")
        return self._send(200, json.dumps({"ok": True, "saved": bid,
                                           "total": len(d["buildings"])}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--area", nargs="*", default=None,
                    help="areas to search; default is every area with data")
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    a = ap.parse_args()

    from src.region_build import all_areas
    areas = a.area or (["pilot"] + [x for x in all_areas() if x != "pilot"])
    truth = _truth_index()
    # Default queue: the roofs Josh has already marked. They are the hardest and
    # most informative buildings in the district, and turning their prose into
    # coordinates is the cheapest labelling in the whole set.
    ids = a.ids or sorted(truth)

    Handler.bundles = Bundles()
    Handler.areas = areas
    Handler.truth = truth
    Handler.queue = ids

    done = len(_load_labels()["buildings"])
    print(f"labelling {len(ids)} roofs ({done} already saved)")
    print(f"  open  http://localhost:{a.port}/")
    print(f"  saves to {LABELS}")
    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

"""Join every region's emitted output into the set of files the map serves.

The other half of emit_region. Each region emitted its own tiles, cells,
detail, heat-map tiles and addresses under data/out/<region>/; this joins them
into data/ without ever holding more than one region's worth of anything in
memory that is not itself a sum. What each join is and why it is safe is the
table in docs/scale-architecture.md; in short:

    pmtiles      tile-join, hierarchically above 200 inputs
    cells        partial sums added, then tiled ONCE
    detail       per-tile dicts merged (keyed by building id)
    heat map     copied; a seam tile touched by two regions is composited
    addresses    concatenated, sharded by prefix when large
    curves       one per degree of latitude the build spans
    summaries    copied per region, totals added

Everything is built in a temporary folder and swapped in at the end, so the
served set is either the old complete one or the new complete one.

Usage: python src/combine_regions.py [--regions a b ...] [--out-root data/out]
                                     [--dest data]
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.region_build import DATA_DIR, write_json_atomic
from src.build_building_tiles import tile_bounds, CELL_BANDS, COVERAGE_STEPS
from src.building_types import BTYPES

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = DATA_DIR / "out"
JOIN_BATCH = 200          # tile-join inputs per process
ADDR_SHARD_AT = 150_000   # above this, addresses are sharded by prefix
HEATMAP_TILE = 256


def _run(cmd, cwd=ROOT):
    subprocess.run(cmd, check=True, cwd=cwd)


def _tile_join(inputs, out, workdir):
    """tile-join over any number of pmtiles, batching so no single process
    holds hundreds of files open. Two levels is enough for 40,000 regions."""
    inputs = [Path(p) for p in inputs if Path(p).exists()]
    if not inputs:
        raise SystemExit(f"nothing to join into {out}")
    if len(inputs) == 1:
        shutil.copy2(inputs[0], out)
        return
    level = inputs
    round_no = 0
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), JOIN_BATCH):
            batch = level[i:i + JOIN_BATCH]
            target = out if (len(level) <= JOIN_BATCH) else \
                workdir / f"join_{round_no}_{i // JOIN_BATCH}.pmtiles"
            listing = workdir / f"join_{round_no}_{i // JOIN_BATCH}.txt"
            listing.write_text("\n".join(str(p) for p in batch) + "\n")
            # -pk: no tile size limit. Regions were already cut to their own
            # budgets; the join must not re-drop. (tile-join has no feature
            # limit of its own -- -pf is a tippecanoe flag and it rejects it.)
            _run(["tile-join", "-q", "-pk", "-f", "-o", str(target),
                  "-r", str(listing)])
            nxt.append(target)
        level = nxt
        round_no += 1


def _combine_cells(regions, out_root, dest_tmp):
    sums = defaultdict(lambda: defaultdict(float))
    for r in regions:
        p = out_root / r / "cells.json"
        if not p.exists():
            continue
        for key, v in json.loads(p.read_text()).items():
            acc = sums[key]
            for k, x in v.items():
                acc[k] += x
    feats = []
    for key in sorted(sums):
        z, x, y, btype = key.split("/")
        z, x, y = int(z), int(x), int(y)
        v = sums[key]
        w, s, e, n = tile_bounds(x, y, z)
        props = {"cell": f"{z}/{x}/{y}", "lod": z, "btype": btype,
                 "n": int(v["n"]), "n_est": int(v["n_est"]),
                 "panel_count": int(v["panel_count"]),
                 "fitted_kwp": round(v["fitted_kwp"], 1),
                 "fitted_kwh": round(v["fitted_kwh"], 0),
                 "facet_area_m2": round(v["area"], 0),
                 "km2": round(abs(e - w) * abs(n - s) * 111.32 * 111.32
                              * math.cos(math.radians((n + s) / 2)), 3)}
        for pct in COVERAGE_STEPS:
            props[f"kwp_{pct}"] = round(v.get(f"kwp_{pct}", 0.0), 1)
            props[f"kwh_{pct}"] = round(v.get(f"kwh_{pct}", 0.0), 0)
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Polygon", "coordinates": [[
                          [w, s], [e, s], [e, n], [w, n], [w, s]]]}})
    src = dest_tmp / "_cells.geojson"
    src.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                              separators=(",", ":")))
    # A SECOND LAYER OF CELL-CENTRE POINTS: the zoomed-out view is a
    # traditional heat map rather than blocks, based
    # on generation density across areas." MapLibre's heat-map rendering takes
    # points, not polygons, so each cell also ships as a point at its centre
    # carrying the same sums plus its density; the page draws the smooth
    # surface from these and keeps the (invisible) cells for exact sums.
    pts = []
    for f in feats:
        p = dict(f["properties"])
        w, s_, e, n = tile_bounds(*[int(v) for v in p["cell"].split("/")[1:]], p["lod"])
        km2 = max(p.get("km2") or 0.001, 0.001)
        for pct in COVERAGE_STEPS:
            p[f"dens_{pct}"] = round((p.get(f"kwh_{pct}") or 0.0) / km2)   # kWh/yr per km2
        p["dens_fit"] = round((p.get("fitted_kwh") or 0.0) / km2)
        pts.append({"type": "Feature", "properties": p,
                    "geometry": {"type": "Point", "coordinates": [(w + e) / 2, (s_ + n) / 2]}})
    src_pts = dest_tmp / "_cellpts.geojson"
    src_pts.write_text(json.dumps({"type": "FeatureCollection", "features": pts},
                                  separators=(",", ":")))
    _run(["tippecanoe", "-q", "-o", str(dest_tmp / "building_cells.pmtiles"), "--force",
          "-Z", "0", "-z", str(max(hi for _, _, hi in CELL_BANDS)),
          "--simplification=2", "--no-feature-limit", "--no-tile-size-limit",
          "-r1", "-L", f"cells:{src}", "-L", f"cellpts:{src_pts}"])
    src.unlink(); src_pts.unlink()
    # the density scale the page normalises against: the 95th percentile of
    # the finest band's cells at full coverage, so the top of the ramp means
    # "among the densest 5% of built land in this build", whatever the build
    # -- and one per band, because a coarse cell averages its roofs over the
    # empty land around them: lod-12 densities run ~10x below lod-16, lod-9
    # ~30x below that, and one scale would leave the zoomed-out view blank.
    import numpy as np
    by_lod = {}
    for lod, _, _ in CELL_BANDS:
        d = [q["properties"]["dens_100"] for q in pts if q["properties"]["lod"] == lod and q["properties"]["n"] >= 5]
        by_lod[str(lod)] = float(np.percentile(d, 95)) if d else 0.0
    fine = str(max(z for z, _, _ in CELL_BANDS))
    _combine_cells.density_p95 = by_lod[fine]
    _combine_cells.density_p95_by_lod = by_lod
    return len(feats)


def _combine_detail(regions, out_root, dest_tmp):
    tiles = defaultdict(dict)
    z = None
    for r in regions:
        d = out_root / r / "detail"
        if not d.exists():
            continue
        for p in d.rglob("*.json"):
            z = int(p.parents[1].name)
            tiles[(p.parent.name, p.stem)].update(json.loads(p.read_text()))
    out = dest_tmp / "building_detail"
    index = {}
    for (x, y), payload in tiles.items():
        p = out / str(z) / x / f"{y}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, separators=(",", ":")))
        index[f"{x}/{y}"] = len(payload)
    out.mkdir(parents=True, exist_ok=True)
    from src.split_building_detail import DETAIL_KEYS, DETAIL_Z
    (out / "index.json").write_text(json.dumps(
        {"z": z or DETAIL_Z, "keys": DETAIL_KEYS, "tiles": index}, separators=(",", ":")))
    return len(tiles)


def _combine_heatmap(regions, out_root, dest_tmp):
    from PIL import Image
    out = dest_tmp / "heatmap_tiles"
    n = seams = 0
    zmin, zmax = 99, 0
    for r in regions:
        d = out_root / r / "heatmap"
        if not d.exists():
            continue
        for p in d.rglob("*.png"):
            z = int(p.parents[1].name)
            zmin, zmax = min(zmin, z), max(zmax, z)
            q = out / str(z) / p.parent.name / p.name
            q.parent.mkdir(parents=True, exist_ok=True)
            if q.exists():
                # a seam: both regions painted this tile; keep both
                Image.alpha_composite(Image.open(q).convert("RGBA"),
                                      Image.open(p).convert("RGBA")).save(q, optimize=True)
                seams += 1
            else:
                shutil.copy2(p, q)
            n += 1
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(
        {"minzoom": zmin if n else 13, "maxzoom": zmax if n else 17, "tileSize": HEATMAP_TILE}))
    return n, seams


def _addr_key(addr):
    s = "".join(ch for ch in addr.lower() if ch.isalnum())
    return (s[:2] if len(s) >= 2 else (s + "_")[:2]) or "__"


def _combine_addresses(regions, out_root, dest_tmp):
    rows = []
    for r in regions:
        p = out_root / r / "addresses.json"
        if p.exists():
            rows.extend(json.loads(p.read_text()))
    rows.sort()
    if len(rows) <= ADDR_SHARD_AT:
        (dest_tmp / "addresses.json").write_text(json.dumps(rows, separators=(",", ":")))
        return len(rows), 0
    # Sharded: the search box fetches the shard for the first two characters
    # typed. A flat 2-million-row file is 34 MB nobody asked for.
    shards = defaultdict(list)
    for row in rows:
        shards[_addr_key(row[0])].append(row)
    out = dest_tmp / "addresses"
    out.mkdir(parents=True, exist_ok=True)
    for k, v in shards.items():
        (out / f"{k}.json").write_text(json.dumps(v, separators=(",", ":")))
    (out / "index.json").write_text(json.dumps(
        {"prefix_chars": 2, "shards": sorted(shards), "total": len(rows)}))
    return len(rows), len(shards)


def _combine_curves(summaries, dest_tmp):
    """One curve set per degree of latitude the build spans."""
    from src.build_seasonal_curves import curves_for
    bands = {}
    for s in summaries:
        lon, lat = s["centroid"]
        band = int(math.floor(lat))
        bands.setdefault(band, []).append((lat, lon))
    out = dest_tmp / "seasonal_curves"
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    mean_lat = sum(lat for v in bands.values() for lat, _ in v) / max(
        1, sum(len(v) for v in bands.values()))
    nearest = None
    for band, pts in sorted(bands.items()):
        lat = sum(p[0] for p in pts) / len(pts)
        lon = sum(p[1] for p in pts) / len(pts)
        doc = curves_for(lat, lon)
        key = f"m{abs(band)}"
        (out / f"{key}.json").write_text(json.dumps(doc))
        index[key] = {"lat_min": band, "lat_max": band + 1}
        if nearest is None or abs(lat - mean_lat) < abs(nearest[0] - mean_lat):
            nearest = (lat, doc)
    (out / "index.json").write_text(json.dumps({"bands": index}))
    # The single file stays for anything that still reads it: the band
    # nearest the middle of the build.
    if nearest:
        (dest_tmp / "seasonal_curves.json").write_text(json.dumps(nearest[1]))
    return len(bands)


def combine(regions=None, out_root=OUT_ROOT, dest=DATA_DIR):
    t0 = time.time()
    out_root, dest = Path(out_root), Path(dest)
    if regions is None:
        regions = sorted(p.name for p in out_root.iterdir()
                         if (p / "summary.json").exists())
    if not regions:
        raise SystemExit(f"no emitted regions under {out_root}")
    summaries = [json.loads((out_root / r / "summary.json").read_text()) for r in regions]
    print(f"combining {len(regions)} regions")

    work = Path(tempfile.mkdtemp(prefix="combine_", dir=dest))
    try:
        _tile_join([out_root / r / "panel_layouts.pmtiles" for r in regions],
                   work / "panel_layouts.pmtiles", work)
        _tile_join([out_root / r / "buildings.pmtiles" for r in regions],
                   work / "buildings.pmtiles", work)
        n_cells = _combine_cells(regions, out_root, work)
        n_detail = _combine_detail(regions, out_root, work)
        n_heat, seams = _combine_heatmap(regions, out_root, work)
        n_addr, n_shards = _combine_addresses(regions, out_root, work)
        n_bands = _combine_curves(summaries, work)

        # assumptions.json is the one small file the page already fetches at
        # load, so the building-type list rides in it and the toggle needs no
        # extra request.
        assumptions = dict(next((s["assumptions"] for s in summaries if s.get("assumptions")), {}))
        assumptions["btypes"] = BTYPES
        assumptions["density_p95_kwh_km2"] = getattr(_combine_cells, "density_p95", 0.0)
        assumptions["density_p95_kwh_km2_by_lod"] = getattr(_combine_cells, "density_p95_by_lod", {})
        (work / "assumptions.json").write_text(json.dumps(assumptions))

        # per-region summaries for the gate, and the totals
        sdir = work / "summaries"
        sdir.mkdir()
        totals = defaultdict(float)
        by_type = defaultdict(lambda: defaultdict(float))
        for s in summaries:
            (sdir / f"{s['region']}.json").write_text(json.dumps(s, separators=(",", ":")))
            for k in ("n", "n_est", "panel_count", "kwp", "kwh", "from_selected_facets"):
                totals[k] += s.get(k, 0) or 0
            for t, v in s.get("by_type", {}).items():
                for k, x in v.items():
                    by_type[t][k] += x
        (work / "build_summary.json").write_text(json.dumps({
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "regions": regions,
            "totals": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in totals.items()},
            "by_type": {t: {k: round(x, 1) for k, x in v.items()} for t, v in by_type.items()},
            "btypes": BTYPES,
            "cells": n_cells, "detail_tiles": n_detail, "heatmap_tiles": n_heat,
            "addresses": n_addr, "address_shards": n_shards, "curve_bands": n_bands,
        }, indent=1))

        # swap in: files first, then folders
        for name in ("panel_layouts.pmtiles", "buildings.pmtiles", "building_cells.pmtiles",
                     "assumptions.json", "seasonal_curves.json", "build_summary.json"):
            if (work / name).exists():
                shutil.move(str(work / name), str(dest / name))
        if (work / "addresses.json").exists():
            shutil.move(str(work / "addresses.json"), str(dest / "addresses.json"))
            if (dest / "addresses").exists():
                shutil.rmtree(dest / "addresses")
        for folder in ("building_detail", "heatmap_tiles", "addresses", "seasonal_curves", "summaries"):
            if (work / folder).exists():
                if (dest / folder).exists():
                    shutil.rmtree(dest / folder)
                shutil.move(str(work / folder), str(dest / folder))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # The drawn lines, as the overlay that shows them -- global and tiny.
    ml = ROOT / "tools" / "build_markup_lines.py"
    if ml.exists():
        subprocess.run([sys.executable, str(ml)], cwd=ROOT)

    print(f"combined in {time.time() - t0:.0f}s: "
          f"{int(totals['n']):,} buildings, {int(totals['panel_count']):,} panels, "
          f"{totals['kwh'] / 1e6:,.1f} GWh; {n_cells:,} cells, {n_detail:,} detail tiles, "
          f"{n_heat:,} heat-map tiles ({seams} seams), {n_addr:,} addresses"
          f"{' in ' + str(n_shards) + ' shards' if n_shards else ''}, {n_bands} curve bands")
    for name in ("panel_layouts.pmtiles", "buildings.pmtiles", "building_cells.pmtiles"):
        print(f"  {name}: {(dest / name).stat().st_size / 1e6:.1f} MB")
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=None)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--dest", default=str(DATA_DIR))
    a = ap.parse_args()
    combine(a.regions, a.out_root, a.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())

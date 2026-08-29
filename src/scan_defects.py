"""
Rank EVERY building in an area by how wrong its layout is, using the real
pipeline, without a rebuild.

Why this exists
---------------
The loop was: run a 20-hour rebuild, have Josh click around the map until he
finds something obviously wrong, fix that one thing, rebuild again. Every
rebuild surfaced a fresh batch, because his sampling is necessarily thin -- he
sees a few dozen roofs out of 15,125 -- and because the fixes could only be
validated by another rebuild.

Two measurements say that loop can be replaced:

1. Every building Josh has reported is in `pilot` -- 1,066 of 15,125 buildings,
   7% of the district. It is central Queenstown, which is where anyone zooms.
   The other 23 areas have never once produced a bug report.

2. The defects he finds by eye are detectable by machine. He reported 93 Beach
   St; an independent planarity scan had already ranked it the 2nd worst facet
   in the area. He is finding the top of a list that can be computed.

So: compute the list. Rank all of pilot, fix down the ranking, verify each fix
with refit_one in seconds, and spend a rebuild only once the list is clean.

Signals
-------
nonplanar   the worst facet's INLIER FRACTION: the share of its points lying
            within 30 cm of its own plane, area-weighted by facet. A genuine
            roof facet measures 85-99%; a tilted sheet through a stepped
            building measures 22-36%.

            This started as a standard deviation and that was wrong. sd is not
            robust: a good flat facet holding a few wall or vegetation returns
            15 m away reads sd 0.7-1.3 while 85% of its points sit inside 10 cm
            of its plane. It flagged 406 of 1,066 pilot buildings, most of them
            fine, and it made the ranking useless -- the top of the list was
            dominated by facets with nothing wrong with them.
raised      panels sitting on structure proud of their own facet's plane:
            ducting, plant, a higher roof section. Josh's "panels clearly
            overlapping all sorts of obstructions".
carved      fraction of roof removed as obstruction. High means the opposite
            error -- "this one clearly could have more panels".
sparse      panel plan area vs usable roof area. Catches 10 Brecon St, "more
            panels could easily fit here", which no other signal sees.
spill       panels whose footprint escapes their own facet. Kept for safety,
            but it is structurally always zero: fitting clips panels to the
            facet, so a panel cannot leave it.

bent        panels that do not sit on a PLANE: fit a plane to the LiDAR under
            one panel and the residual RMS exceeds 12 cm. This is the signal
            spill cannot provide. Josh: "panel arrays have been placed
            overlapping roof ridges as if it's a flat plane that continues
            when it is not". When segmentation merges two roof faces into one
            facet, a panel bridging the real ridge is legitimately INSIDE its
            facet -- nothing about the facet is violated, so every
            facet-relative check passes while the panel is physically resting
            on a fold. Measured on the shipped pilot layout, 7.7% of panels
            fail this, with the worst buildings at 24-37%.

Usage:
    python src/scan_defects.py pilot                 # rank the area
    python src/scan_defects.py pilot --jobs 6
    python src/scan_defects.py pilot --top 40
"""

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# A real roof plane is DSM noise. These are the bars above which a signal is
# worth a human looking at, not pass/fail thresholds for the pipeline.
NONPLANAR_INLIER = 0.70       # below this share of points near its own plane
PLANARITY_BAND_M = 0.30       # ...measured within this band
CARVED_FRACTION = 0.30
SPARSE_FILL = 0.45
SPARSE_MIN_ROOF_M2 = 60.0     # below this, low fill is just a small awkward roof
SPILL_TOLERANCE_M2 = 0.15     # a panel corner this far out is rounding, not a real spill
BENT_RMS_M = 0.12             # a panel's own points should lie on a plane to within DSM noise
BENT_MIN_POINTS = 6
BENT_FRACTION = 0.10          # flag the building once this share of its panels are bent

_CTX = {}


def _ctx(area):
    """One heavy context per worker process. PointCloudSource caches decoded
    tiles for the life of the process, which is why this scan forks per area
    rather than holding every area open at once."""
    import geopandas as gpd
    import rasterio
    from src.region_build import area_paths
    from src.pointcloud_source import PointCloudSource

    if area in _CTX:
        return _CTX[area]
    p = area_paths(area)
    dedup = p["dir"] / "building_outlines_dedup.geojson"
    gdf = gpd.read_file(dedup if dedup.exists() else p["outlines"]).set_index(
        "building_id", drop=False)
    img = None
    if p["imagery"].exists():
        img = rasterio.open(p["imagery"])
    _CTX[area] = {"gdf": gdf, "dsm": rasterio.open(p["dsm"]), "img": img,
                  "pc": PointCloudSource()}
    return _CTX[area]


def _scan_one(args):
    area, bid = args
    import shapely.vectorized
    from shapely.ops import unary_union
    from src.roof_segmentation import segment_building_best
    from src.obstruction_detection import detect_obstructions_combined
    from src.panel_fitting import fit_panels_on_facet

    try:
        c = _ctx(area)
        geom = c["gdf"].loc[bid].geometry
        facets = segment_building_best(c["dsm"], c["pc"], geom, bid,
                                       imagery_ds=c["img"])
        if not facets:
            return {"building_id": bid, "area": area, "empty": True}

        worst_inlier, worst_area = 1.0, 0.0
        panels, obst, spill, bent = [], [], 0, 0
        plan_area = 0.0
        for f in facets:
            plane = (f["plane_a"], f["plane_b"], f["plane_c"])
            ob = detect_obstructions_combined(c["img"], c["pc"], f["geometry"], plane)
            obst.extend(ob)
            got = fit_panels_on_facet(
                f, obstructions=ob,
                sibling_facets=[o for o in facets if o is not f])
            panels.extend(got)
            plan_area += sum(p["geometry"].area for p in got)
            fminx, fminy, fmaxx, fmaxy = f["geometry"].bounds
            fpts_all = c["pc"].points_in_bbox(fminx - 1, fminy - 1, fmaxx + 1, fmaxy + 1,
                                              building_only=True)
            for p in got:
                if p["geometry"].difference(f["geometry"]).area > SPILL_TOLERANCE_M2:
                    spill += 1
                if len(fpts_all) == 0:
                    continue
                m = shapely.vectorized.contains(p["geometry"], fpts_all[:, 0], fpts_all[:, 1])
                P = fpts_all[m]
                if len(P) < BENT_MIN_POINTS:
                    continue
                A = np.column_stack([P[:, 0] - P[:, 0].mean(),
                                     P[:, 1] - P[:, 1].mean(), np.ones(len(P))])
                try:
                    coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
                except np.linalg.LinAlgError:
                    continue
                if float(np.sqrt(np.mean((A @ coef - P[:, 2]) ** 2))) > BENT_RMS_M:
                    bent += 1

            minx, miny, maxx, maxy = f["geometry"].bounds
            pts = c["pc"].points_in_bbox(minx, miny, maxx, maxy, building_only=True)
            if len(pts) >= 12:
                inside = shapely.vectorized.contains(f["geometry"], pts[:, 0], pts[:, 1])
                fp = pts[inside]
                if len(fp) >= 12:
                    r = fp[:, 2] - (plane[0] * fp[:, 0] + plane[1] * fp[:, 1] + plane[2])
                    inl = float((np.abs(r - np.median(r)) < PLANARITY_BAND_M).mean())
                    # weight by area: one bad 5 m2 sliver should not outrank a
                    # bad 2,000 m2 roof, which is what an unweighted min did.
                    if f["geometry"].area > worst_area * 0.25:
                        if inl < worst_inlier or f["geometry"].area > worst_area:
                            worst_inlier = min(worst_inlier, inl)
                            worst_area = max(worst_area, f["geometry"].area)

        roof = unary_union([f["geometry"] for f in facets])
        ob_area = (unary_union([o.buffer(0) for o in obst]).intersection(roof).area
                   if obst else 0.0)
        usable = max(roof.area - ob_area, 1e-9)
        return {"building_id": bid, "area": area, "empty": False,
                "roof_m2": float(roof.area), "panels": len(panels),
                "inlier": worst_inlier,
                "carved": float(ob_area / max(roof.area, 1e-9)),
                "fill": float(plan_area / usable),
                "spill": spill,
                "bent": bent,
                "bent_frac": float(bent / max(len(panels), 1))}
    except Exception as exc:
        return {"building_id": bid, "area": area, "error": repr(exc)[:200]}


def _flags(r):
    """Which signals fired, worst first. Empty means the building looks fine."""
    if r.get("error") or r.get("empty"):
        return ["FAILED"]
    out = []
    if r["inlier"] < NONPLANAR_INLIER:
        out.append(f"nonplanar {r['inlier']:.0%} on-plane")
    if r["spill"]:
        out.append(f"spill {r['spill']}")
    if r.get("bent_frac", 0) > BENT_FRACTION:
        out.append(f"bent {r['bent_frac']:.0%}")
    if r["carved"] > CARVED_FRACTION:
        out.append(f"carved {r['carved']:.0%}")
    if r["roof_m2"] >= SPARSE_MIN_ROOF_M2 and r["fill"] < SPARSE_FILL:
        out.append(f"sparse {r['fill']:.0%}")
    return out


def _severity(r):
    """Rank so the roofs a human would call obviously wrong come first."""
    if r.get("error") or r.get("empty"):
        return 1e6
    s = 0.0
    s += 60.0 * max(0.0, NONPLANAR_INLIER - r["inlier"])
    s += 6.0 * r["spill"]
    s += 45.0 * max(0.0, r.get("bent_frac", 0.0) - BENT_FRACTION)
    s += 60.0 * max(0.0, r["carved"] - CARVED_FRACTION)
    if r["roof_m2"] >= SPARSE_MIN_ROOF_M2:
        s += 40.0 * max(0.0, SPARSE_FILL - r["fill"])
    # a big roof getting it wrong matters more than a shed getting it wrong
    return s * (1.0 + min(r["roof_m2"], 2000.0) / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("area")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0, help="scan only the first N buildings")
    a = ap.parse_args()

    from src.region_build import area_paths
    p = area_paths(a.area)
    d = json.loads(p["panel_layouts"].read_text())
    ids = sorted({f["properties"]["building_id"] for f in d["features"]})
    if a.limit:
        ids = ids[:a.limit]
    print(f"scanning {len(ids)} buildings in {a.area} on {a.jobs} workers", flush=True)

    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(_scan_one, (a.area, b)) for b in ids]
        for fu in as_completed(futs):
            rows.append(fu.result())
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(ids)}", flush=True)

    rows.sort(key=_severity, reverse=True)
    out = Path("data") / f"defects_{a.area}.json"
    out.write_text(json.dumps(rows, indent=1))

    ok = [r for r in rows if not _flags(r)]
    bad = [r for r in rows if _flags(r)]
    print(f"\n{len(ok)} clean, {len(bad)} flagged of {len(rows)}")
    counts = {}
    for r in bad:
        for f in _flags(r):
            counts[f.split()[0]] = counts.get(f.split()[0], 0) + 1
    print("  by signal: " + "  ".join(f"{k} {v}" for k, v in
                                      sorted(counts.items(), key=lambda kv: -kv[1])))
    print(f"\nworst {a.top}:")
    for r in rows[:a.top]:
        fl = ", ".join(_flags(r)) or "-"
        rm = r.get("roof_m2", 0.0)
        print(f"  #{r['building_id']}  roof {rm:7.0f} m2  "
              f"panels {r.get('panels', 0):4d}   {fl}")
    print(f"\nfull ranking -> {out}")


if __name__ == "__main__":
    main()

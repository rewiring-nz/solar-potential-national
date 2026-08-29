"""
Self-audit of placed panels against the source data, per area.

For every building in an area's panel_layouts.geojson (NZTM reprojected),
each panel is checked for the failure modes reported in the field:

- edge_overlap: panel not (within tolerance) inside the building outline --
  panels visibly hanging over real roof edges.
- facet_escape: panel not inside its own facet (buffered) -- crossing a
  hip/ridge into a differently-oriented face.
- lumpy:  LiDAR building-class points under the panel sit far off the
  facet plane (median |residual| beyond threshold) -- the panel covers a
  chimney/vent/plant/structure that obstruction detection missed.
- z_split: the point-cloud z under one panel spans a big range -- the
  panel bridges two roof levels.

Output: data/audit_<area>.json with per-building counts + the worst
offenders ranked, and a summary line per area. Sampling flags rather than
perfection: the goal is finding SYSTEMIC failure patterns to fix in the
pipeline, then re-checking with the same audit.

Usage: python src/audit_layouts.py [region ...] [--sample N]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyproj
import shapely.vectorized
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pointcloud_source import PointCloudSource
from src.region_build import area_paths

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True).transform

EDGE_TOL_M = 0.15          # panel may exceed the outline by this much before flagged
FACET_TOL_M = 0.25         # ...and its facet by this much (facet borders are fuzzier)
LUMPY_MEDIAN_M = 0.35      # median |residual| under a panel beyond this = covering something
LUMPY_MIN_PTS = 6
ZSPLIT_RANGE_M = 1.2       # p95-p5 z-range under one 2m panel beyond this = bridges levels


def plane_from_facet_points(pts):
    """Trimmed least-squares: a facet polygon contains, in 2D, points from
    other roof levels above/below it -- fit, drop outliers, refit, so the
    plane follows the DOMINANT surface instead of averaging levels."""
    x0, y0 = pts[:, 0].mean(), pts[:, 1].mean()
    keep = np.ones(len(pts), bool)
    coeffs = None
    for _ in range(3):
        sub = pts[keep]
        if len(sub) < 12:
            break
        A = np.column_stack([sub[:, 0] - x0, sub[:, 1] - y0, np.ones(len(sub))])
        coeffs, *_ = np.linalg.lstsq(A, sub[:, 2], rcond=None)
        res_all = (coeffs[0] * (pts[:, 0] - x0) + coeffs[1] * (pts[:, 1] - y0)
                   + coeffs[2] - pts[:, 2])
        new_keep = np.abs(res_all) < 0.35
        if new_keep.sum() < 12 or (new_keep == keep).all():
            break
        keep = new_keep
    return x0, y0, coeffs




def _audit_building(bid, b, outline_by_id, pc):
    outline = outline_by_id.get(bid)
    facet_geoms = [g.buffer(0) for g, _ in b["facets"]]
    facet_tree = STRtree(facet_geoms) if facet_geoms else None
    counts = {"panels": len(b["panels"]), "edge_overlap": 0, "facet_escape": 0,
              "lumpy": 0, "z_split": 0}
    facet_planes = []
    for g in facet_geoms:
        minx, miny, maxx, maxy = g.bounds
        pts = pc.points_in_bbox(minx, miny, maxx, maxy, building_only=True)
        if len(pts) >= 12:
            inside = shapely.vectorized.contains(g, pts[:, 0], pts[:, 1])
            fp = pts[inside]
            facet_planes.append(plane_from_facet_points(fp) if len(fp) >= 12 else None)
        else:
            facet_planes.append(None)
    outline_buf = outline.buffer(EDGE_TOL_M) if outline is not None and not outline.is_empty else None
    for panel in b["panels"]:
        panel = panel.buffer(0)
        if panel.is_empty:
            continue
        if outline_buf is not None and not outline_buf.contains(panel):
            counts["edge_overlap"] += 1
        fi = -1
        if facet_tree is not None:
            c = panel.centroid
            cand = facet_tree.query(panel)
            best = None
            for idx in cand:
                if facet_geoms[idx].contains(c):
                    best = idx
                    break
            if best is None and len(cand):
                best = min(cand, key=lambda i: facet_geoms[i].distance(c))
            if best is not None:
                fi = int(best)
                if not facet_geoms[fi].buffer(FACET_TOL_M).contains(panel):
                    counts["facet_escape"] += 1
        minx, miny, maxx, maxy = panel.bounds
        pts = pc.points_in_bbox(minx, miny, maxx, maxy, building_only=True)
        if len(pts) >= LUMPY_MIN_PTS:
            inside = shapely.vectorized.contains(panel, pts[:, 0], pts[:, 1])
            pp = pts[inside]
            if len(pp) >= LUMPY_MIN_PTS:
                if fi >= 0 and facet_planes[fi] is not None and facet_planes[fi][2] is not None:
                    x0, y0, cf = facet_planes[fi]
                    res = cf[0] * (pp[:, 0] - x0) + cf[1] * (pp[:, 1] - y0) + cf[2] - pp[:, 2]
                    if np.median(np.abs(res)) > LUMPY_MEDIAN_M:
                        counts["lumpy"] += 1
                    # spread measured against the plane, not raw z -- raw z
                    # spans ~1.3m over a single panel on a 30-degree pitch.
                    if np.percentile(res, 95) - np.percentile(res, 5) > ZSPLIT_RANGE_M:
                        counts["z_split"] += 1
    flagged = counts["edge_overlap"] + counts["facet_escape"] + counts["lumpy"] + counts["z_split"]
    if not flagged:
        return None
    counts["building_id"] = int(bid)
    counts["flag_rate"] = round(flagged / counts["panels"], 3)
    return counts

def audit_area(name, sample=None, pc=None):
    paths = area_paths(name)
    layouts = json.loads(paths["panel_layouts"].read_text())
    outlines = json.loads(paths["outlines"].read_text())
    # Outlines are stored in NZTM already (WFS native CRS); panel_layouts is
    # WGS84. Reproject only what needs it -- detect by coordinate magnitude.
    def to_nztm(geom_dict):
        g = shape(geom_dict)
        x = g.bounds[0]
        return g if abs(x) > 180 else shp_transform(TO_NZTM, g)
    outline_by_id = {f["properties"]["building_id"]: to_nztm(f["geometry"])
                     for f in outlines["features"]}

    buildings = {}
    for f in layouts["features"]:
        p = f["properties"]
        b = buildings.setdefault(p["building_id"], {"facets": [], "panels": []})
        if p["kind"] == "facet":
            b["facets"].append((shp_transform(TO_NZTM, shape(f["geometry"])), p))
        elif p["kind"] == "panel":
            b["panels"].append(shp_transform(TO_NZTM, shape(f["geometry"])))

    ids = list(buildings)
    if sample:
        rng = np.random.default_rng(42)
        ids = list(rng.choice(ids, size=min(sample, len(ids)), replace=False))

    results = []
    skipped = 0
    for bid in ids:
        b = buildings[bid]
        if not b["panels"]:
            continue
        try:
            counts = _audit_building(bid, b, outline_by_id, pc)
        except Exception:
            skipped += 1
            continue
        if counts:
            results.append(counts)

    results.sort(key=lambda r: -(r["edge_overlap"] + r["facet_escape"] + r["lumpy"] + r["z_split"]))
    out = DATA_DIR / f"audit_{name}.json"
    out.write_text(json.dumps(results))
    tot = {k: sum(r[k] for r in results) for k in ("edge_overlap", "facet_escape", "lumpy", "z_split")}
    n_panels = sum(b["panels"] and len(b["panels"]) or 0 for b in buildings.values())
    if skipped:
        print(f"  NOTE: {skipped} buildings skipped (errored during audit)")
    print(f"{name}: {len(ids)} buildings audited, {n_panels} panels | "
          f"edge {tot['edge_overlap']} | facet {tot['facet_escape']} | "
          f"lumpy {tot['lumpy']} | z-split {tot['z_split']} | "
          f"{len(results)} buildings flagged -> {out.name}")


def main():
    argv = sys.argv[1:]
    sample = None
    if "--sample" in argv:
        i = argv.index("--sample")
        sample = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    args = [a for a in argv if not a.startswith("--")]
    pc = PointCloudSource()
    for name in (args or ["pilot"]):
        audit_area(name, sample=sample, pc=pc)


if __name__ == "__main__":
    main()

"""
Verify that the architecture diagram still describes the code.

Josh: "the diagram should always accurately match the code." A diagram that can
drift is worse than no diagram, because a reviewer trusts it and then reads the
code expecting to find what it promised. This file is what stops that.

Everything the diagram asserts is listed below as either a SYMBOL (a function,
class or constant that must exist at a named path) or a VALUE (a constant whose
number the diagram actually prints, which must still be that number). The check
fails if any symbol has moved or any quoted figure has changed.

That second kind matters more than it looks. The system derate moved three times
in one day -- 19% to 15% to 11.34% -- and each move silently invalidated any
document quoting the old figure. This turns that into a failing check instead of
a stale page nobody notices.

When it fails, the fix is to update the diagram AND the entry here together, in
the same commit. Do not just change the number here to make it pass; that is the
drift this exists to prevent.

Run:  python tools/check_diagram.py
      python tools/check_diagram.py --list     # what the diagram claims
"""

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DIAGRAM_URL = "https://claude.ai/code/artifact/4f676147-2042-4779-968a-75fe38ffe2c2"

# Every function/class the diagram names, as module -> names. If the diagram
# says "segment_building_best picks the best strategy", that symbol must exist.
SYMBOLS = {
    "src/fetch_data.py":          ["fetch_building_outlines", "fetch_raster",
                                   "reclaim_export_intermediates"],
    "src/fetch_regions.py":       ["bbox_nztm", "split_bbox", "fetch_raster_chunked"],
    "src/fetch_pointcloud_regions.py": ["tilename_to_filename", "main"],
    "src/pointcloud_source.py":   ["PointCloudSource"],
    "src/region_build.py":        ["area_paths", "area_bbox_nztm", "all_areas"],

    "src/solar_model.py":         ["build_poa_lookup_table", "SolarModel",
                                   "solarview_calibrated_monthly_factors",
                                   "niwa_station_measured_monthly_factors",
                                   "fetch_niwa_derived_monthly_factors",
                                   "fetch_nasa_power_monthly_factors"],
    "src/build_seasonal_curves.py": ["main"],

    "src/terrain_horizon.py":     ["compute_horizon_profile_from_array", "horizon_angle_at"],
    "src/building_horizon.py":    ["compute_building_horizon", "encode_horizon",
                                   "decode_horizon", "facet_horizon_factor", "far_profile"],
    "src/building_shading.py":    ["building_shading_factor"],
    "src/build_terrain_masks.py": ["main"],

    "src/roof_segmentation.py":   ["segment_building_best", "_area_weighted_inlier",
                                   "_maybe_reconstruct", "_attach_building_geometry"],
    "src/roof_partition.py":      ["partition_roof", "top_surface"],
    "src/roof_skeleton.py":       ["skeleton_roof"],
    "src/roof_lines.py":          ["roof_line_candidates", "strong_roof_lines"],
    "src/obstruction_detection.py": ["detect_obstructions_combined"],

    "src/panel_fitting.py":       ["fit_panels_on_facet", "assign_fill_ranks",
                                   "drop_minor_arrays"],
    "src/gate_panels.py":         ["gate_area", "gate_area_parallel"],
    "src/build_layout_geojson.py": ["_build_one", "_init_worker", "_facet_fit",
                                    "_memory_bounded_jobs", "main"],
    "src/rerank_layouts.py":      ["rerank_area"],

    "src/derive_solar_potential.py": ["derive"],
    "src/build_heatmap_raster.py":   ["shading_grid", "render_building"],
    "src/bake_density_deciles.py":   ["main"],
    "src/merge_regions.py":          ["merge_geojson"],

    "src/preflight.py":           ["preflight"],
    "src/run_stage.py":           ["is_done", "write_marker"],
    "src/invariants.py":          ["check"],
}

# Numbers the diagram prints. Path is dotted from the repo root; a config entry
# is written as config.PV_ASSUMPTIONS.system_derate_pct.
VALUES = {
    "config.MAX_ROOF_SLOPE_DEG": 55,
    "config.PANEL_WIDTH_M": 1.134,
    "config.PANEL_HEIGHT_M": 1.961,
    "config.PANEL_EDGE_SETBACK_M": 0.3,
    "config.RIDGE_SETBACK_M": 0.25,
    "config.PV_ASSUMPTIONS.panel_rated_power_w": 500,
    "config.PV_ASSUMPTIONS.panel_efficiency_pct": 22.5,
    "config.PV_ASSUMPTIONS.inverter_efficiency_pct": 97.0,
    "config.PV_ASSUMPTIONS.system_derate_pct": 11.34,
    "src.build_layout_geojson.MIN_ROOF_CONFIDENCE": 0.45,
    "src.build_layout_geojson.DEEP_SHADE_FACTOR": 0.45,
    "src.build_layout_geojson.BIG_ROOF_M2": 1000.0,
    "src.build_layout_geojson.BIG_ROOF_FACET_MIN_FIT": 0.60,
    "src.panel_fitting.SHALLOW_SEAM_DEG": 9.0,
    "src.building_horizon.N_BINS": 72,
    "src.building_horizon.FAR_MAX_KM": 20.0,
    "src.building_horizon.NEAR_MAX_KM": 0.3,
}

# Derived figures the diagram states in prose, each with how it is computed.
DERIVED = {
    "DC-to-AC factor 0.860": lambda c: round(
        (c.PV_ASSUMPTIONS["inverter_efficiency_pct"] / 100)
        * (1 - c.PV_ASSUMPTIONS["system_derate_pct"] / 100), 3) == 0.860,
    "total system loss 14.0%": lambda c: abs(
        100 * (1 - (c.PV_ASSUMPTIONS["inverter_efficiency_pct"] / 100)
               * (1 - c.PV_ASSUMPTIONS["system_derate_pct"] / 100)) - 14.0) < 0.05,
    "panel area 2.22 m2": lambda c: abs(
        c.PANEL_WIDTH_M * c.PANEL_HEIGHT_M - 2.224) < 0.005,
}


def _symbols_in(path):
    src = (ROOT / path).read_text()
    tree = ast.parse(src)
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
    return out


def _resolve(dotted):
    import importlib
    parts = dotted.split(".")
    # config.PV_ASSUMPTIONS.system_derate_pct  ->  module config, then attrs
    for split in range(len(parts) - 1, 0, -1):
        mod_name = ".".join(parts[:split])
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        obj = mod
        for p in parts[split:]:
            if isinstance(obj, dict):
                if p not in obj:
                    return None, f"key {p!r} missing"
                obj = obj[p]
            elif hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                return None, f"attribute {p!r} missing"
        return obj, None
    return None, "module not importable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        n = sum(len(v) for v in SYMBOLS.values())
        print(f"the diagram names {n} symbols across {len(SYMBOLS)} files,")
        print(f"quotes {len(VALUES)} constants and {len(DERIVED)} derived figures.")
        print(f"\n{DIAGRAM_URL}\n")
        for path, names in SYMBOLS.items():
            print(f"  {path}\n      {', '.join(names)}")
        return 0

    problems = []

    for path, names in SYMBOLS.items():
        p = ROOT / path
        if not p.exists():
            problems.append(f"FILE GONE: {path} — the diagram points at it")
            continue
        have = _symbols_in(path)
        for n in names:
            if n not in have:
                problems.append(f"SYMBOL GONE: {path}::{n} — the diagram names it")

    import config  # noqa: F401  (imported for DERIVED)
    for dotted, expected in VALUES.items():
        got, err = _resolve(dotted)
        if err:
            problems.append(f"VALUE MISSING: {dotted} — {err}")
        elif isinstance(expected, float) or isinstance(got, float):
            if abs(float(got) - float(expected)) > 1e-6:
                problems.append(
                    f"VALUE CHANGED: {dotted} is {got}, the diagram says {expected}")
        elif got != expected:
            problems.append(
                f"VALUE CHANGED: {dotted} is {got}, the diagram says {expected}")

    for label, fn in DERIVED.items():
        try:
            if not fn(config):
                problems.append(f"DERIVED FIGURE WRONG: the diagram states '{label}'")
        except Exception as exc:
            problems.append(f"DERIVED FIGURE UNCHECKABLE: {label} — {exc}")

    total = sum(len(v) for v in SYMBOLS.values()) + len(VALUES) + len(DERIVED)
    if not problems:
        print(f"diagram matches the code: {total} claims checked, all hold")
        print(f"  {DIAGRAM_URL}")
        return 0

    print(f"DIAGRAM IS OUT OF DATE — {len(problems)} of {total} claims no longer hold:\n")
    for p in problems:
        print(f"  {p}")
    print(f"\nUpdate the diagram AND the entry in this file, in the same commit.")
    print(f"Do not simply change the number here to make this pass -- that is the")
    print(f"drift this check exists to prevent.")
    print(f"\n  {DIAGRAM_URL}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

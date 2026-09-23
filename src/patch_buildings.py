"""Rebuild NAMED buildings with the current code and push just them live.

The iterate-by-full-rebuild loop takes hours; this takes minutes. It runs the
same per-building path as build_layout_geojson (so what you see is what a full
rebuild would produce for those buildings), swaps their features into the
region file, the merged district file and solar_potential, re-runs the panel
shrink + tippecanoe over the district, and optionally commits.

Usage:
  python src/patch_buildings.py 5371108 4734850 ... [--area pilot] [--push]

The area flag is only needed for buildings outside the pilot region; ids from
several areas need one invocation per area. Never run while a district rebuild
is writing the same files.
"""
import argparse, json, subprocess, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+", type=int)
    ap.add_argument("--area", default="pilot")
    ap.add_argument("--push", action="store_true")
    # See the solar-map copy: a district-wide post-process does not belong
    # inside a 60-building chunk. patch_stale_selected bakes once at the end.
    ap.add_argument("--skip-bake", action="store_true",
                    help="do not re-bake density deciles (caller will)")
    ap.add_argument("--skip-tiles", action="store_true",
                    help="patch the geojson only (for chained invocations; run tiles once at the end)")
    a = ap.parse_args()

    from src.region_build import area_paths, area_centroid_wgs84
    from src.solar_model import SolarModel
    import src.build_layout_geojson as blg

    t0 = time.time()
    c = area_centroid_wgs84(a.area)
    blg._init_worker(a.area, SolarModel(*c) if c else SolarModel())
    new_feats = {}
    for bid in a.ids:
        feats = blg._build_one(bid)
        if not feats:
            # A rebuild yielding NOTHING is a crash wearing a quiet face --
            # _build_one converts any exception into an empty list. Splicing
            # that in would DELETE the building from the live map. Refuse the
            # whole run instead: a patch must never ship less than it replaces
            # by accident. (A genuinely empty building would have been empty
            # in the district file already.)
            raise SystemExit(
                f"ABORT: #{bid} rebuilt to 0 features -- almost certainly a "
                f"pipeline exception. Run _build_one_inner({bid}) directly for "
                f"the traceback. Nothing was patched.")
        new_feats[bid] = feats
        n = sum(1 for f in feats if f["properties"]["kind"] == "panel")
        print(f"  #{bid}: {len(feats)} features, {n} panels  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # run the same post-stages those buildings would get in a full build
    ids = set(a.ids)

    def patch(path):
        d = json.load(open(path))
        before = len(d["features"])
        d["features"] = [f for f in d["features"]
                         if f["properties"].get("building_id") not in ids]
        for bid in a.ids:
            d["features"].extend(new_feats[bid])
        json.dump(d, open(path, "w"))
        print(f"  patched {path.name}: {before} -> {len(d['features'])} features", flush=True)

    region = area_paths(a.area)["panel_layouts"]
    patch(region)
    # gate just this area's new panels (in place, cheap for a handful of ids)
    from src.gate_panels import gate_area
    from src.pointcloud_source import PointCloudSource
    gate_area(a.area, PointCloudSource(), only_ids=ids)
    # re-copy region layouts into the merged district file
    patch(DATA / "panel_layouts.geojson")

    # solar_potential must tell the same story as the layouts it summarises.
    # Until 31 Aug this file's docstring claimed it patched solar_potential and
    # the code never did: a patched building got new panels on the map while
    # the dashboard beside it kept quoting the old count, kW and generation.
    # Ported from the Wellington copy, which had the implementation all along
    # -- the two repos are hand-synced, so each had a piece the other lacked.
    #
    # Splice ONLY the patched buildings' aggregates, preserving every other
    # building untouched (roof_confidence etc. live on these features).
    sp_path = DATA / "solar_potential.geojson"
    if sp_path.exists():
        import config
        from src.derive_solar_potential import _facet_area_m2
        reg = json.load(open(region))
        agg = {}
        for f in reg["features"]:
            p = f["properties"]
            if p.get("building_id") not in ids:
                continue
            b = agg.setdefault(p["building_id"], {"facet_count": 0, "obstruction_count": 0,
                                                  "panel_count": 0, "ac_kwh_year": 0.0,
                                                  "facet_area_m2": 0.0, "poa_w": 0.0})
            k = p["kind"]
            if k == "facet":
                b["facet_count"] += 1
                # Same fix as the solar-map copy: the layout emitter does
                # not write area_m2 on a facet, so reading it directly gave
                # every patched building facet_area_m2 = 0 and Heat Map mode
                # showed the roof as 0.0 kW.
                area = _facet_area_m2(f)
                b["facet_area_m2"] += area
                b["poa_w"] += area * (p.get("poa_kwh_m2_yr") or 0.0)
            elif k == "obstruction":
                b["obstruction_count"] += 1
            elif k == "panel":
                b["panel_count"] += 1
                b["ac_kwh_year"] += p.get("ac_kwh_year") or 0.0
        sp = json.load(open(sp_path))
        panel_kw = config.PV_ASSUMPTIONS["panel_rated_power_w"] / 1000.0
        n_upd = 0
        for f in sp["features"]:
            bid = f["properties"].get("building_id")
            if bid not in agg:
                continue
            b = agg[bid]
            # a rebuilt building with panels must not keep a stale
            # no-estimate reason from the run it is replacing
            if b["panel_count"] > 0:
                f["properties"].pop("no_estimate_reason", None)
                f["properties"].pop("reason", None)
            f["properties"].update({
                "facet_count": b["facet_count"],
                "obstruction_count": b["obstruction_count"],
                "panel_count": b["panel_count"],
                "kwp": round(b["panel_count"] * panel_kw, 2),
                "ac_kwh_day_avg": round(b["ac_kwh_year"] / 365.0, 1),
                "ac_kwh_year": round(b["ac_kwh_year"], 0),
                "facet_area_m2": round(b["facet_area_m2"], 1),
                "avg_poa_kwh_m2": round(b["poa_w"] / b["facet_area_m2"], 0)
                                  if b["facet_area_m2"] > 0 else 0,
            })
            n_upd += 1
        json.dump(sp, open(sp_path, "w"))
        print(f"  solar_potential: updated {n_upd} buildings", flush=True)
        # density deciles (fill_*) for the patched buildings come from the
        # merged layouts; bake refreshes them (writes solar_potential in place)
        if not a.skip_bake:
            subprocess.run([sys.executable, "src/bake_density_deciles.py"],
                           check=True, cwd=ROOT)

    if not a.skip_tiles:
        # Re-emit this region and recombine (docs/scale-architecture.md): the
        # merged file is no longer what the map serves.
        subprocess.run([sys.executable, "src/emit_region.py", a.area], check=True, cwd=ROOT)
        subprocess.run([sys.executable, "src/combine_regions.py"], check=True, cwd=ROOT)
        print(f"  region re-emitted and tiles recombined ({time.time()-t0:.0f}s total)", flush=True)

    if a.push:
        subprocess.run(["git", "add", "data/panel_layouts.pmtiles", "data/buildings.pmtiles",
                        "data/building_cells.pmtiles", "data/summaries", "data/build_summary.json",
                        "data/building_detail"], cwd=ROOT, check=True)
        subprocess.run(["git", "-c", "user.name=Josh", "-c", "user.email=josh@ideatious.com",
                        "commit", "-q", "-m",
                        f"Patch buildings {' '.join(map(str, a.ids))} with current code"],
                       cwd=ROOT, check=True)
        subprocess.run(["git", "push", "-q", "origin", "main"], cwd=ROOT, check=True)
        print("  pushed live", flush=True)

if __name__ == "__main__":
    main()

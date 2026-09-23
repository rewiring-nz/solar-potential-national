"""Put a region's outputs in the bucket, then delete its inputs.

Inputs are deleted after the build. They are 200 GB of the VM's 217 GB
today and terabytes nationally, and every byte is re-fetchable from LINZ.
They go the moment the outputs are safe -- and not a moment before.

ORDER, AND WHY IT IS THIS ORDER.
  1. upload  data/out/<region>/       tiles, cells, detail, summary, the
                                      enriched region file
             data/regions/<region>/*.geojson   layouts and solar_potential
             data/selected_faces/<id>.json     this region's face readings,
                                      so a re-run with a new fitter does not
                                      re-predict
  2. verify  list the bucket back and compare every size to the local file.
             A silently short upload followed by a delete is the one failure
             this stage must never produce.
  3. manifest  data/regions/<region>/manifest.json: what was built, from
             which survey, at which git sha, and where it went.
  4. delete  the rasters (dsm*, imagery*, dem*), and the point-cloud tiles
             that NO OTHER region still on this disk lists. Tiles are shared
             at borders; a worker holding two regions at once must not pull
             the ground from under the second.

The outlines, the tile list and the manifest stay: they are kilobytes and
they are what makes the region re-fetchable and auditable later.

Usage: python src/publish_region.py <region> [--no-delete] [--dry-run]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.region_build import DATA_DIR, REGIONS_DIR, area_paths, area_bbox_wgs84
from src.gcs_queue import url, _run

OUT_ROOT = DATA_DIR / "out"
SEL = DATA_DIR / "selected_faces"
PC = DATA_DIR / "pointcloud"
INPUT_GLOBS = ("dsm*.tif", "imagery*.tif", "dem*.tif", "*_part*_mosaic.tif")


def _sizes_local(root):
    return {p.relative_to(root).as_posix(): p.stat().st_size
            for p in root.rglob("*") if p.is_file()}


def _sizes_remote(prefix_url):
    r = _run(["ls", "-l", "-r", prefix_url + "/**"], check=False)
    out = {}
    if r.returncode != 0:
        return out
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1].startswith("gs://"):
            try:
                out[parts[-1][len(prefix_url) + 1:]] = int(parts[0])
            except ValueError:
                pass
    return out


def _rsync(local, remote, dry):
    if dry:
        print(f"  would upload {local} -> {remote}")
        return
    _run(["rsync", "-r", str(local), remote], capture=False)


def publish(region, delete=True, dry=False):
    t0 = time.time()
    out = OUT_ROOT / region
    if not (out / "summary.json").exists():
        raise SystemExit(f"[{region}] nothing emitted under {out} -- run emit_region first")
    paths = area_paths(region)
    rdir = paths["dir"]
    base = url("regions", region)

    # 1. upload
    _rsync(out, base + "/out", dry)
    geo = rdir / "_publish_geojson"
    geo.mkdir(exist_ok=True)
    for name in ("solar_potential.geojson", "panel_layouts.geojson",
                 "building_outlines_dedup.geojson", "building_outlines.geojson"):
        src = rdir / name
        if src.exists():
            (geo / name).unlink(missing_ok=True)
            (geo / name).hardlink_to(src) if hasattr(Path, "hardlink_to") else \
                subprocess.run(["ln", str(src), str(geo / name)], check=True)
    _rsync(geo, base + "/region", dry)
    ids = []
    try:
        ids = [str(f["properties"]["building_id"]) for f in
               json.loads((out / "solar_potential.geojson").read_text())["features"]]
    except Exception:
        pass
    readings = rdir / "_publish_readings"
    readings.mkdir(exist_ok=True)
    n_readings = 0
    for b in ids:
        src = SEL / f"{b}.json"
        if src.exists():
            (readings / src.name).unlink(missing_ok=True)
            subprocess.run(["ln", str(src), str(readings / src.name)], check=True)
            n_readings += 1
    if n_readings:
        _rsync(readings, base + "/readings", dry)

    # 2. verify, size for size
    if not dry:
        for local, remote in ((out, base + "/out"), (geo, base + "/region"),
                              (readings, base + "/readings")):
            want = _sizes_local(local)
            if not want:
                continue
            have = _sizes_remote(remote)
            short = {k: (v, have.get(k)) for k, v in want.items() if have.get(k) != v}
            if short:
                sample = list(short.items())[:5]
                raise SystemExit(f"[{region}] upload verification FAILED for {remote}: "
                                 f"{len(short)} files differ, e.g. {sample}. Nothing deleted.")
    for tmp in (geo, readings):
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()

    # 3. manifest
    summary = json.loads((out / "summary.json").read_text())
    summary.pop("ladder", None)
    manifest = {
        "region": region, "bbox": area_bbox_wgs84(region),
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bucket": base, "git": summary.get("git"), "buildings": summary.get("n"),
        "panel_count": summary.get("panel_count"), "kwh": summary.get("kwh"),
        "readings_uploaded": n_readings,
        "inputs_deleted": bool(delete and not dry),
    }
    try:
        from src.surveys import survey_for
        manifest["survey"] = survey_for(manifest["bbox"], region).get("name")
    except Exception:
        pass
    if not dry:
        (rdir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # 4. delete inputs
    freed = 0
    if delete:
        for pat in INPUT_GLOBS:
            for p in rdir.glob(pat):
                freed += p.stat().st_size
                if not dry:
                    p.unlink()
        # point-cloud tiles nobody else on this disk still lists
        mine = set()
        lst = rdir / "pointcloud_tiles.txt"
        if lst.exists():
            mine = set(lst.read_text().split())
        others = set()
        for other in REGIONS_DIR.iterdir():
            if other.name == region or not other.is_dir():
                continue
            if (other / "manifest.json").exists():
                continue          # already published: its inputs are gone too
            ol = other / "pointcloud_tiles.txt"
            if ol.exists():
                others |= set(ol.read_text().split())
        for fn in sorted(mine - others):
            p = PC / fn
            if p.exists():
                freed += p.stat().st_size
                if not dry:
                    p.unlink()
        # this region's own emitted output is in the bucket now too, but it
        # stays: combine_regions reads it, and it is a few MB
    print(f"[{region}] published to {base} in {time.time() - t0:.0f}s"
          f"{' (dry run)' if dry else ''}: {n_readings} readings, "
          f"{'freed' if not dry else 'would free'} {freed / 1e9:.1f} GB of inputs")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region")
    ap.add_argument("--no-delete", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    publish(a.region, delete=not a.no_delete, dry=a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Put regions on the bucket queue.

    python tools/enqueue_regions.py --config              # config.REGIONS
    python tools/enqueue_regions.py --plan data/national_regions.json \
        --bbox 174.4 -37.4 175.3 -36.4                    # e.g. Auckland
    python tools/enqueue_regions.py --plan ... --limit 20

Writes queue/<region>.json for each region not already queued or done.
Enqueuing is the funded act: nothing here fetches or builds. Each task
carries its bbox, so a worker whose config does not list the region can still
build it -- region_build.area_bbox_wgs84 reads a bbox from the task file
placed at data/regions/<region>/task.json before it consults the config.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gcs_queue import ls, write_json, url

MAX_BUILDINGS = 3000


def _split(region, n):
    """Split a planner cell into `n` equal-ish sub-boxes by longitude."""
    w, s, e, nn = region["bbox"]
    step = (e - w) / n
    return [{"name": f"{region['name']}_{i}",
             "bbox": [w + i * step, s, w + (i + 1) * step, nn],
             "buildings": region["buildings"] // n}
            for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", action="store_true")
    g.add_argument("--plan")
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    help="only planner cells whose centre falls inside")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tasks = []
    if a.config:
        import config
        for name, bbox in config.REGIONS.items():
            tasks.append({"name": name, "bbox": list(bbox)})
    else:
        plan = json.loads(Path(a.plan).read_text())
        for r in plan["regions"]:
            if a.bbox:
                cx = (r["bbox"][0] + r["bbox"][2]) / 2
                cy = (r["bbox"][1] + r["bbox"][3]) / 2
                if not (a.bbox[0] <= cx <= a.bbox[2] and a.bbox[1] <= cy <= a.bbox[3]):
                    continue
            n = max(1, math.ceil(r["buildings"] / MAX_BUILDINGS))
            tasks.extend(_split(r, n) if n > 1 else [r])
    if a.limit:
        tasks = tasks[:a.limit]

    queued = {n[:-5] for n in ls("queue")}
    done = {n[:-5] for n in ls("done")}
    n_new = 0
    for t in tasks:
        if t["name"] in queued or t["name"] in done:
            continue
        if a.dry_run:
            print(f"  would enqueue {t['name']}  {t['bbox']}  {t.get('buildings', '?')} buildings")
        else:
            write_json(t, "queue", t["name"] + ".json")
        n_new += 1
    print(f"{n_new} regions {'would be ' if a.dry_run else ''}enqueued "
          f"({len(tasks) - n_new} already queued or done) -> {url('queue')}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

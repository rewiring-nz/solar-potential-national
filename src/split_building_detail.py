"""Split the per-building detail out of the map source, into tiles fetched on click.

WHY. The solar potential data must not download whole for the map to work
for all of NZ? Maybe we need an approach that will be fast for all NZ, then
increase in accuracy as we get closer?"

The browser fetches data/solar_potential.geojson whole at load. That is 26 MB
for Queenstown's 15,353 buildings. New Zealand has roughly 2.1 million in the
LINZ outlines -- about 137x -- so the same file would be 3.4 GB and the map
would simply never open. It is the one hard blocker on a national rollout that
has nothing to do with roof geometry.

WHAT IS ACTUALLY IN THE FILE, measured rather than guessed:

    geometry              38.5%
    tshade                12.6%
    horizon_b64           12.5%
    horizon_far_b64       12.5%
    everything else       23.9%   (56 properties, none above 2.1%)

So 37.6% of the download is three fields -- a building's shading horizon and
its hour-by-hour shade mask -- and every one of them is read by exactly one
thing: the detail panel for the ONE building someone clicked. Nothing on the
map is coloured by them, nothing is summed over them, and 15,352 of the 15,353
copies are thrown away unread.

They also scale worst. At 2.1 million buildings that is about 600 MB of base64
nobody asked for.

HOW. The detail is written into tiles, keyed by the z13 web-mercator tile the
building's centroid falls in, so neighbours share a file and browsing a suburb
pulls one or two of them. A click fetches its tile once and the browser caches
it. Queenstown's detail lands in a few dozen files of a few hundred kB each --
at national scale that is roughly the same file size, just more of them, which
is the property that makes it a tiled format rather than one big download.

The slimmed solar_potential.geojson keeps everything the map itself needs:
geometry, the estimate, the coverage ladder, the fill/sys ladders, addresses
and the no-estimate reasons.

Usage: python src/split_building_detail.py
"""

import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.region_build import write_json_atomic

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = DATA / "solar_potential.geojson"
OUT_DIR = DATA / "building_detail"

# z13 is about 4.9 km a tile at this latitude, which puts a few hundred
# buildings in each -- small enough that one fetch is quick, large enough that
# panning around a suburb does not start a new fetch every block.
DETAIL_Z = 13

# Read by the detail panel only: the horizon ring it draws, the far-terrain
# ring behind it, the 4 x 24 hourly shade mask behind the season curves, and
# the beam fraction the panel quotes. Nothing on the map is drawn from these.
DETAIL_KEYS = ("horizon_b64", "horizon_far_b64", "tshade", "horizon_beam_pct")


def tile_of(lon, lat, z=DETAIL_Z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(max(-85.05, min(85.05, lat)))
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def _centroid(geom):
    """Ring average -- good enough to pick a tile, and avoids importing shapely
    into a stage whose only job is moving strings between files."""
    try:
        ring = geom["coordinates"][0]
        if ring and isinstance(ring[0][0], (list, tuple)):
            ring = ring[0]
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    except Exception:
        return None


def main():
    if not SRC.exists():
        print(f"no {SRC}")
        return 1
    doc = json.loads(SRC.read_text())
    feats = doc["features"]

    # IDEMPOTENT, OR IT DESTROYS THE THING IT BUILT. The first version cleared
    # the output directory and then wrote whatever it found inline -- so the
    # second run, against an already-split file, found nothing to move and left
    # an empty directory and a map with no horizons. A stage that is run twice
    # by a resumed build must not be destructive on the second pass.
    if not any(k in f["properties"] for f in feats for k in DETAIL_KEYS):
        have = len(list(OUT_DIR.rglob("*.json"))) - 1 if OUT_DIR.exists() else 0
        if have > 0:
            print(f"already split: {have} detail tiles, nothing inline to move")
            return 0
        print("WARNING: nothing inline to move and no detail tiles on disk -- "
              "the source has been split and the tiles are gone. Re-run "
              "derive_solar_potential/merge_regions to regenerate them.")
        return 1

    tiles = defaultdict(dict)
    moved = kept = 0
    for f in feats:
        p = f["properties"]
        detail = {k: p.pop(k) for k in DETAIL_KEYS if k in p}
        if not detail:
            kept += 1
            continue
        c = _centroid(f["geometry"])
        if c is None:
            # cannot place it in a tile, so it stays inline rather than being
            # silently lost -- a building with no detail panel is a bug report
            p.update(detail)
            kept += 1
            continue
        x, y = tile_of(*c)
        tiles[(x, y)][str(p["building_id"])] = detail
        moved += 1

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    total = 0
    index = {}
    for (x, y), payload in sorted(tiles.items()):
        d = OUT_DIR / str(DETAIL_Z) / str(x)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{y}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        total += path.stat().st_size
        index[f"{x}/{y}"] = len(payload)

    # An index so the frontend can skip a fetch it knows will 404. Cheap
    # (one line a tile) and it keeps the console clean, which matters because
    # a real failure hides in a page full of expected 404s.
    (OUT_DIR / "index.json").write_text(json.dumps(
        {"z": DETAIL_Z, "keys": DETAIL_KEYS, "tiles": index},
        separators=(",", ":")))

    before = SRC.stat().st_size
    write_json_atomic(SRC, doc)
    after = SRC.stat().st_size
    print(f"{moved} buildings -> {len(tiles)} detail tiles "
          f"({total/1e6:.1f} MB, {total/max(len(tiles),1)/1e3:.0f} kB each)")
    if kept:
        print(f"{kept} buildings had no detail to move")
    print(f"solar_potential.geojson {before/1e6:.1f} MB -> {after/1e6:.1f} MB "
          f"({100*(before-after)/before:.0f}% smaller)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

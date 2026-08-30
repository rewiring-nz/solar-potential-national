"""
Render the biggest build-over-build movers for review BEFORE a push.

Josh (30 Aug): "isn't there a risk there are different negative impacts on
the rest of the buildings?" -- unknown impacts must surface, not hide in 15k
buildings. compare_builds ranks the movers; this draws them.

Usage: python src/render_top_movers.py [--top 10] [--out DIR]

Reads data/build_snapshot_prev.json + data/solar_potential.geojson, takes the
`--top` largest absolute fill_panels_100 deltas, and renders each through
render_building_debug (current code, real pipeline). Buildings in regions
whose imagery/point clouds aren't on this machine render LiDAR-only or are
skipped with a note -- the report says which.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    top = 10
    out_dir = DATA_DIR / "mover_renders"
    argv = sys.argv[1:]
    if "--top" in argv:
        i = argv.index("--top"); top = int(argv[i + 1])
    if "--out" in argv:
        i = argv.index("--out"); out_dir = Path(argv[i + 1])
    out_dir.mkdir(exist_ok=True)

    prev = json.loads((DATA_DIR / "build_snapshot_prev.json").read_text())
    cur = {}
    d = json.loads((DATA_DIR / "solar_potential.geojson").read_text())
    for f in d["features"]:
        p = f["properties"]
        cur[str(p["building_id"])] = (p.get("fill_panels_100", 0), p.get("address", ""))

    deltas = []
    for b, (n, addr) in cur.items():
        pn = prev.get(b)
        if pn is None:
            continue
        deltas.append((abs(n - pn[0]), n - pn[0], b, pn[0], n, addr))
    deltas.sort(reverse=True)

    from src.render_building_debug import render
    print(f"{'building':>10} {'was':>5} {'now':>5} {'delta':>6}  address")
    for _, dlt, b, was, now, addr in deltas[:top]:
        print(f"{b:>10} {was:>5} {now:>5} {dlt:>+6}  {addr[:40]}")
        try:
            render(int(b), str(out_dir / f"mover_{b}.png"))
        except SystemExit as e:
            print(f"    (render skipped: {e})")
        except Exception as e:
            print(f"    (render failed: {type(e).__name__}: {e})")
    print(f"renders in {out_dir}")


if __name__ == "__main__":
    main()

"""The drawn lines as a GeoJSON overlay for the map.

A drawn line is not always meant to reach an edge, so it
cannot be forced into the face geometry -- cutting faces on those lines was
measured worse against the markup (fidelity 94.8% -> 77.2%). What was
actually missing is that the map never DREW them, so a drawn line looked
absent.

This extracts just the lines from data/roof_labels.json -- 2,380 of them
across 144 roofs -- as a small file the frontend can load on demand, rather
than shipping the 1.9 MB label file with its faces, obstructions and notes
to every visitor.

Usage: python tools/build_markup_lines.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VOID = {"absent", "not_building", "unclear"}
OUT = ROOT / "data" / "markup_lines.geojson"


def main():
    import pyproj
    B = json.loads((ROOT / "data/roof_labels.json").read_text())["buildings"]
    to_wgs = pyproj.Transformer.from_crs(2193, 4326, always_xy=True).transform
    feats = []
    for k, v in B.items():
        if v.get("problem") in VOID:
            continue
        for l in v.get("lines") or []:
            pts = l.get("points") or ([l.get("a"), l.get("b")]
                                      if l.get("a") and l.get("b") else None)
            if not pts or len(pts) < 2:
                continue
            coords = [list(to_wgs(float(p[0]), float(p[1]))) for p in pts]
            coords = [[round(x, 6), round(y, 6)] for x, y in coords]
            feats.append({
                "type": "Feature",
                "properties": {"building_id": int(k),
                               "kind": l.get("kind") or "line"},
                "geometry": {"type": "LineString", "coordinates": coords},
            })
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                              separators=(",", ":")))
    print(f"{len(feats)} drawn lines -> {OUT} "
          f"({OUT.stat().st_size/1e3:.0f} kB)")


if __name__ == "__main__":
    sys.exit(main() or 0)

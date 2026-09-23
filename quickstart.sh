#!/bin/bash
# QUICKSTART: run the full solar-potential methodology on YOUR OWN small
# area of New Zealand, end to end, and inspect every step of the result.
#
#   1. cp my_area.example.json my_area.json   (edit name + bbox)
#   2. export LINZ_API_KEY=...                (free key from data.linz.govt.nz,
#                                              REST API scope enabled)
#   3. bash quickstart.sh <name>
#
# This is NOT a simplified re-implementation. Your area becomes a
# first-class region and runs through the IDENTICAL stages the deployed
# Queenstown map was built with -- same code paths, same thresholds, same
# gates -- so anything you verify here is a verification of the real
# methodology. See docs/quickstart.md for how to check each stage.
set -u
cd "$(dirname "$0")"
# Windows (Git Bash / MSYS) puts the interpreter somewhere else.
if [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
else PY=.venv/bin/python
fi
AREA="${1:-}"
if [ -z "$AREA" ]; then
  echo "usage: bash quickstart.sh <area-name-from-my_area.json>"; exit 2
fi
if [ ! -f my_area.json ]; then
  echo "my_area.json not found -- copy my_area.example.json and edit it"; exit 2
fi
if [ -z "${LINZ_API_KEY:-}" ] && ! grep -q LINZ_API_KEY .env 2>/dev/null; then
  echo "LINZ_API_KEY not set (env or .env) -- free key at data.linz.govt.nz"; exit 2
fi
if [ ! -x "$PY" ]; then
  echo "no .venv -- see docs/data-maintainers/local-setup.md first"; exit 2
fi

echo "=== 1/5 fetch: outlines, DSM/DEM, imagery, LiDAR tiles (LINZ + OpenTopography) ==="
$PY src/fetch_regions.py "$AREA" || exit 1
$PY src/fetch_pointcloud_regions.py "$AREA" || \
  echo "  (point cloud unavailable for this survey -- continuing; LiDAR-dependent
   stages degrade and docs/quickstart.md explains exactly which)"

echo "=== 2/5 vision models (optional but part of the shipped methodology) ==="
VISION=1
if [ ! -f data/sam_vit_b.pth ]; then
  echo "  downloading SAM ViT-B checkpoint (358 MB, Meta AI's public release)"
  curl -L -o data/sam_vit_b.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
    || VISION=0
fi
[ -f data/models/roof_lines_v5.pt ] || VISION=0
if [ "$VISION" = "1" ] && $PY -c "import torch, segment_anything" 2>/dev/null; then
  echo "=== 3/5 vision precompute: SAM + line detector + LiDAR candidates per roof ==="
  $PY tools/predict_faces.py --region "$AREA" || VISION=0
else
  VISION=0
fi
if [ "$VISION" = "0" ]; then
  echo "  vision chain skipped (torch/segment-anything or checkpoints missing)."
  echo "  The build falls back to the LiDAR partition for every roof -- the"
  echo "  same fallback the production map uses where the vision chain defers."
fi

echo "=== 4/5 build: the exact production stages ==="
export SOLAR_SELECTED_FACES=1
for s in build_layout_geojson gate_panels rerank_layouts derive_solar_potential; do
  $PY src/run_stage.py "$s" "$AREA" || exit 1
done

echo "=== 5/5 report ==="
$PY tools/quickstart_report.py "$AREA" || exit 1
echo ""
echo "open data/regions/$AREA/quickstart_report.html and follow"
echo "docs/quickstart.md to verify each stage against what you can see."

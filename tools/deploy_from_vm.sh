#!/bin/bash
# Pull the VM's combined served set, gate it against live, and push.
#
#   tools/deploy_from_vm.sh            # pull + compare + gate, then stop and show
#   tools/deploy_from_vm.sh --push     # ...and commit + push if the gate passes
#
# The VM cannot publish (scp'd payload, no git). This is the one place the
# laptop is in the ship path, and it does nothing but relay files the VM
# built: nothing here is computed on the laptop.
set -u
cd "$(dirname "$0")/.."
VM=claude-doing-things; ZONE=australia-southeast1-b
PY=.venv/bin/python
PUSH=0; [ "${1:-}" = "--push" ] && PUSH=1

echo "=== pull the served set ==="
for f in panel_layouts.pmtiles buildings.pmtiles building_cells.pmtiles addresses.json \
         assumptions.json seasonal_curves.json build_summary.json markup_lines.geojson; do
  gcloud compute scp --zone $ZONE --quiet $VM:"~/solar-map/data/$f" "data/$f" || { echo "FATAL: could not pull $f"; exit 1; }
done
for d in building_detail heatmap_tiles summaries seasonal_curves addresses; do
  rm -rf "data/$d.pull" && mkdir -p "data/$d.pull"
  if gcloud compute scp --zone $ZONE --quiet --recurse $VM:"~/solar-map/data/$d" "data/$d.pull/" 2>/dev/null; then
    rm -rf "data/$d" && mv "data/$d.pull/$d" "data/$d"
  fi
  rm -rf "data/$d.pull"
done
[ -f data/addresses.json ] && [ -d data/addresses ] && rm -rf data/addresses   # flat file wins when both exist

echo "=== what the VM built ==="
$PY -c "import json; s=json.load(open('data/build_summary.json')); t=s['totals']; print(len(s['regions']), 'regions,', int(t['n']), 'buildings,', int(t['panel_count']), 'panels,', round(t['kwh']/1e6,1), 'GWh; by type:', {k:int(v['n']) for k,v in s['by_type'].items()})"

echo "=== gate against live ==="
$PY tools/predeploy_check.py || { echo "GATE FAILED -- not pushing"; exit 1; }

if [ $PUSH -eq 1 ]; then
  # bump the data version so browsers refetch every tile
  v=$(grep -oE 'dataVersion: "[0-9]+"' site-config.js | grep -oE '[0-9]+'); nv=$((v+1))
  sed -i '' "s/dataVersion: \"$v\"/dataVersion: \"$nv\"/" site-config.js
  git add -A data site-config.js
  git commit -q -m "District build: $($PY -c 'import json; s=json.load(open("data/build_summary.json")); t=s["totals"]; print(f"{len(s[\"regions\"])} regions, {int(t[\"panel_count\"]):,} panels, {t[\"kwh\"]/1e6:,.1f} GWh")')

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && git push -q origin main && echo "PUSHED (data v$nv)"
else
  echo "dry: re-run with --push to publish"
fi

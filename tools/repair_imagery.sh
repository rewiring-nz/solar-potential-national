#!/usr/bin/env bash
# Rebuild the missing imagery mosaics, one at a time.
#
# 14 of 24 regions built LiDAR-only in the 3 Sep district run because their
# imagery_mosaic.tif was absent. Nine of them already had every downloaded part
# sitting on disk -- only the merge was missing -- so this is mostly a merge job,
# not a re-download.
#
# WHY STRICTLY SEQUENTIAL. rasterio's merge() decompresses the whole mosaic into
# RAM: a measured 15.3 GB for a two-part region. Nine of those in parallel would
# take the 62 GB box out, and that is the likeliest reason the merges failed in
# the first place -- they ran while a build had ten workers resident. So one at a
# time, and each one waits until the machine actually has room.
#
# Safe to re-run: fetch_regions.py skips any mosaic that already exists and any
# part already downloaded.

set -u
cd "$(dirname "$0")/.." || exit 1

MIN_FREE_GB=${MIN_FREE_GB:-24}     # a merge needs ~15-20 GB; leave the build room
LOG=data/build_logs/imagery_repair.log
mkdir -p data/build_logs

# Regions with parts on disk needing only a merge, cheapest first.
MERGE="arrowtown_east arthurs_point arthurs_point_east dalefield frankton_arm
       jacks_point tucker_beach arrowtown_millbrook shotover_lakehayes"
# Regions with no imagery at all -- these must actually download.
FETCH="arrowtown_hills frankton_east_lake kelvin_heights kelvin_south town_south_lake"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

wait_for_ram() {
  local waited=0
  while :; do
    local avail
    avail=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$avail" -ge "$MIN_FREE_GB" ]; then return 0; fi
    if [ $((waited % 300)) -eq 0 ]; then
      say "  waiting for RAM: ${avail}GB available, need ${MIN_FREE_GB}GB"
    fi
    sleep 60
    waited=$((waited + 60))
  done
}

say "=== imagery repair starting ==="
for r in $MERGE $FETCH; do
  if [ -f "data/regions/$r/imagery_mosaic.tif" ]; then
    say "$r: mosaic already present, skipping"
    continue
  fi
  wait_for_ram
  say "$r: building mosaic..."
  if .venv/bin/python src/fetch_regions.py "$r" >> "$LOG" 2>&1; then
    if [ -f "data/regions/$r/imagery_mosaic.tif" ]; then
      sz=$(ls -la "data/regions/$r/imagery_mosaic.tif" | awk '{printf "%.2f", $5/1073741824}')
      say "$r: OK (${sz} GB)"
    else
      say "$r: FINISHED BUT NO MOSAIC -- check the log"
    fi
  else
    say "$r: FAILED (exit $?) -- continuing with the rest"
  fi
done

say "=== imagery repair done ==="
say "regions still without imagery:"
for d in data/regions/*/; do
  n=$(basename "$d")
  [ -f "$d/imagery_mosaic.tif" ] || say "  $n"
done

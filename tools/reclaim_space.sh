#!/bin/bash
# Reclaim disk by deleting REDUNDANT Earth Engine export archives.
#
# Why this exists: on 31 Aug the Queenstown rebuild was running with 20 GB free
# and 11 regions still to write, while 45.5 GB of *_export.zip archives sat on
# disk. Those zips are the raw downloads; once they have been unpacked into the
# region's *_mosaic.tif they are redundant, and they are re-fetchable. Nothing
# deletes them as the build goes, so they accumulate until the disk fills and
# the build dies hours in.
#
# SAFETY -- a zip is only touched when BOTH are true:
#   1. its region already has a build log (so nothing in flight needs it), and
#   2. the derived *_mosaic.tif exists (so the unpack demonstrably succeeded).
# Mosaics themselves are NEVER touched: the truth scorecard reads imagery, and
# deleting it out from under a later stage has broken a run before.
#
#   bash ~/reclaim_space.sh          # DRY RUN -- lists what it would remove
#   bash ~/reclaim_space.sh --yes    # actually delete
set -u
cd "$HOME/solar-map/data/regions" || exit 1
GO=0
[ "${1:-}" = "--yes" ] && GO=1

total=0; count=0
for z in $(find . -name "*export*.zip" -type f 2>/dev/null); do
  r=$(echo "$z" | cut -d/ -f2)
  kind=$(basename "$z" | sed "s/_part[0-9]*//; s/_export.zip//")
  [ -f "$HOME/solar-map/data/build_logs/$r.full2.log" ] || continue
  ls "$r/${kind}"*mosaic.tif >/dev/null 2>&1 || continue
  sz=$(stat -c%s "$z")
  total=$((total + sz)); count=$((count + 1))
  if [ $GO -eq 1 ]; then rm -f "$z" && echo "  removed $z"; else echo "  would remove $z"; fi
done

echo ""
awk -v t="$total" -v n="$count" 'BEGIN {printf "%d files, %.1f GB\n", n, t/1073741824}'
[ $GO -eq 0 ] && echo "DRY RUN -- nothing deleted. Re-run with --yes to do it."
df -h "$HOME" | tail -1

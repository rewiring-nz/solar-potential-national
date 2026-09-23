# Kingston and Wanaka: what is running, and how to finish it

Started 22 September. Everything below runs on the build VM
(`claude-doing-things`, australia-southeast1-b) and does **not** need Josh's
laptop. Closing the laptop does not stop it. The only step that needs a
person is the deploy at the end.

## What is running

`~/solar-map/expand.sh`, logging to `~/solar-map/expand.out`. Five new
regions, 12,284 buildings:

| region | buildings | survey | point cloud |
| --- | --- | --- | --- |
| `kingston` | 380 | Otago – Kingston LiDAR (2025) | **none published** → 1 m DSM |
| `wanaka_town` | 8,945 | Otago – Wanaka LiDAR (2022-23) | `NZ22_Wanaka` |
| `wanaka_albert_town` | 718 | ″ | ″ |
| `wanaka_west` | ~415 | ″ | ″ |
| `hawea` | ~1,826 | ″ | ″ |

Stages, in order: fetch rasters → fetch point cloud → dedupe outlines →
predict faces (8 shards a region) → `run_district_build.sh --regions`.

## How long

Scaled from Queenstown's measured rates (15,353 buildings, 6.6 h predict,
6.1 h build) plus an unmeasured fetch:

- fetch, ~93 GB from LINZ: **2–4 h**, and bounded by LINZ, not by the VM
- predict: **~5.3 h**
- build: **~4.9 h**

**Roughly 12–14 hours total.** It will not finish inside a 10-hour window.

## If it stops

The VM is preemptible. `expand.sh` is resumable — `predict_done_new.txt`
records a region only after all eight shards finish, and `run_stage` markers
cover the build. After a preemption:

```bash
gcloud compute instances start claude-doing-things --zone australia-southeast1-b
gcloud compute ssh claude-doing-things --zone australia-southeast1-b \
  --command "cd ~/solar-map && nohup ./expand.sh >> expand.out 2>&1 &"
```

Check progress with:

```bash
gcloud compute ssh claude-doing-things --zone australia-southeast1-b \
  --command "cd ~/solar-map && tail -5 expand.out; grep -c . predict_done_new.txt"
```

Done when `expand.out` ends with `EXPAND_DONE`.

## Finishing it: the deploy

The VM is an scp'd payload copy, not a git clone, so it cannot publish. Pull
the artefacts to the laptop and push from there:

```bash
cd ~/Desktop/J/Website/solar-map
for f in panel_layouts.pmtiles buildings.pmtiles building_cells.pmtiles \
         addresses.json assumptions.json solar_potential.geojson \
         seasonal_curves.json; do
  gcloud compute scp --zone australia-southeast1-b --quiet \
    claude-doing-things:"~/solar-map/data/$f" "data/$f"
done
rm -rf data/building_detail
gcloud compute scp --zone australia-southeast1-b --quiet --recurse \
  claude-doing-things:"~/solar-map/data/building_detail" data/
python tools/build_heatmap_tiles.py          # new regions need their tiles
python tools/predeploy_check.py              # compare against live BEFORE pushing
git add -A data && git commit && git push
```

`predeploy_check.py` is not optional. It compares every building against the
live site and reports zeroed roofs, big drops and any stale density ladder.
A district gaining two towns should show +12,000 buildings and no losses
among the existing 15,353; anything else needs looking at before it ships.

## Then stop the VM

```bash
gcloud compute instances stop claude-doing-things --zone australia-southeast1-b
```

The 400 GB disk keeps billing while stopped. Deleting it means re-fetching
~140 GB of LiDAR and imagery next time, so it is a trade, not an obvious win.

## The wide DEM was too small, and is now right

Found on 22 September while rebasing onto a colleague's new coverage test.
`data/dem_wide_mosaic.tif` -- the 8 m bare-earth model every far-horizon
calculation reads -- covered only 168.47-168.87 lon, -45.20 to -44.87, on the
laptop and on the VM. Kingston sits 20 km south of its bottom edge and Wanaka
is off it entirely, so both towns would have been modelled with **open sky in
the directions that are mountains**. Nothing would have said so:
`building_horizon.far_profile` marches to `FAR_MAX_KM` and stops at the DEM
edge, so a short DEM does not error, it just returns a sunnier roof.

Refetched on the VM: 15,555 x 18,251 at 8 m, 1.12 GB, covering
[168.086, -45.662, 169.754, -44.295]. What it is worth, measured on 25
buildings a region:

| region | direct beam lost to terrain |
| --- | --- |
| Kingston | **-3.86%** |
| Wanaka town | -0.85% |
| Hawea | -0.51% |

Queenstown was measured too, old mosaic against new, seven regions and 175
buildings: **-0.00% to -0.01%**. The terrain that matters to Queenstown was
already inside the small mosaic, so the existing 24 regions do NOT need
rebuilding, and about six VM-hours were not spent finding that out the
expensive way.

The extent is no longer a constant. `src/fetch_dem_wide.py` derives it from
`config.REGIONS` at call time and refuses a mosaic whose own bounds do not
contain it, so adding a region can no longer silently outgrow the terrain
model. The builders read only the window a region's rays reach (about 127 MB,
not 1.12 GB, and the pool spawns so that cost was per worker).

## Known limits, on purpose

**Kingston has no raw point cloud.** Its LiDAR is a 2025 survey and
OpenTopography has not published it — six store/year combinations were probed
against a real tile name (`CD11_1000_0326`) and all returned 404. Kingston
therefore builds from the 1 m DSM of that same survey: real LiDAR, gridded,
against the ~4.9 returns/m² Queenstown enjoys. **Expect its roofs to be read
less finely than Queenstown's**, and revisit when OpenTopography catches up.

**Kingston's imagery is 0.3 m, not 0.1 m.** Otago Rural Aerial Photos
(2019-2021) is the only LINZ layer whose extent contains it. Imagery-derived
roof lines will be weaker; the LiDAR does the load-bearing work either way.

**Nothing has been verified against a real roof yet.** When it lands, look at
a few Wanaka and Kingston roofs on the map before believing the totals — the
same discipline as every other release here.

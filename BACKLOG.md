# Backlog

The working queue. **Read this at the start of a session and after every
rebuild or deploy**, and update it as items move — it is the only place the
plan survives; anything that lives only in chat is lost when context is
compacted, which is why Josh kept having to re-state the list.

Ordered by evidence, not by appeal. Every item names what it is based on.

## CODE REVIEW — 31 Aug (Josh asked for one; findings ranked)

Full write-up published as an artifact. The measured findings, in the order
they should be worked:

STATUS 31 Aug (second pass): items 1, 2, 4, 5 DONE. Item 3 mostly done
(pure-function tests + golden segmentation tests). Item 6 partly done. Item 7
turned out to be two-thirds WRONG — see the corrections below, which matter
more than the original items.

TWO THINGS THIS REVIEW GOT WRONG, both corrected in code:
  * "roof_reconstruct.py is 860 lines of dead code behind USE_RECONSTRUCTION
    = False — delete it." NO. That flag governs only the UNCONDITIONAL path.
    `_attach_building_geometry` calls `_maybe_reconstruct` on every building,
    which runs reconstruction whenever the segmenter fits a roof below 0.70
    inlier and keeps it only on a clear gain that does not shatter the roof.
    It is LIVE guarded fallback code. Deleting it would have removed real
    handling for exactly the awkward roofs that need it. Consider renaming the
    flag to USE_RECONSTRUCTION_UNCONDITIONALLY so nobody repeats this.
  * "Prepared geometries missing from the point-in-polygon hot paths." The hot
    paths already use array-at-a-time contains, which is better than prep. BUT
    chasing it found something real — see the shapely item below.

1. **The two repos are a hand-maintained fork, and had already diverged.**
   66 of 68 source files were byte-identical; `panel_fitting.py` differed by
   70 lines, ALL of them Queenstown-only. Wellington was missing the gap-fill
   pass AND the straggler yield exception — both things Josh reported, both
   live in Queenstown for days. Island Bay was placing fewer panels at 100%
   and stripping good panels first. Synced + pushed + shipped to VM 31 Aug,
   but nothing prevents the next divergence. Wellington should be a CONFIG of
   one codebase, not a copy of it.
   DONE 31 Aug: synced + pushed, AND `tools/check_repo_sync.py` added to both
   repos — byte-compares the 72 shared files, fails on anything not in an
   explicit ALLOWED list (currently `config.py` and Wellington's single-region
   `patch_buildings.py`, each with its reason). Verified against an injected
   drift. **Run it before any push that touches shared code.** The real fix
   (one codebase, two configs) is still open.

1b. **SECOND divergence, opposite direction — solar-map's `patch_buildings.py`
   never updated solar_potential**, though its docstring always claimed it
   patched "the region file, the merged district file and solar_potential".
   The only occurrence of solar_potential in the file was that sentence. So
   patching a Queenstown building gave it new panels on the map while the
   dashboard kept quoting the old panel count, kW, generation and savings.
   Wellington had the implementation all along. Ported + pushed 31 Aug.
2. **Preflight assertions on stage inputs.** DONE 31 Aug — `src/preflight.py`,
   wired into all twelve stages so it cannot be forgotten. Existence + non-empty
   checks, plus the merge-before-masks ordering as an mtime invariant. Verified
   against all three incidents below, with no false alarm on a healthy tree.
   `python src/preflight.py --all-stages <region>` reports everything at once.
   Was: three incidents this month share
   one shape — missing input, no error, plausible-but-wrong output: the absent
   `dem_wide_mosaic.tif` shipped UNGATED panels; `build_terrain_masks` before
   `merge_regions` silently wiped the masks; JIT imagery cleanup broke the
   scorecard. Assert inputs exist before any stage runs; record in a manifest
   which inputs produced each output.
3. **Zero tests** (17,923 LOC, 68 modules, 0 test files).
   HALF DONE 31 Aug: `tests/test_pure.py` — 15 tests over the pure
   functions, mutation-checked (a flipped aspect sign, a factor-of-two in the
   horizon quantiser and an absurd derate each fail it). Run with
   `.venv/bin/python tests/test_pure.py`; no pytest in the venv.
   STILL OPEN: golden tests over the roof_truth buildings, which are the half
   that makes big refactors safe. Was: start with pure
   functions — `our_aspect_to_pvgis`, horizon encode/decode round-trip, seam
   classification, derate arithmetic — then golden tests over the
   `roof_truth.json` buildings so a refactor that moves 15% fails loudly.
4. **Pin dependencies and containerise.**
   PINNING DONE 31 Aug: `requirements.lock.txt` frozen from the VM venv that
   builds the published maps. Found real drift doing it — Mac scipy 1.18.0 vs
   VM 1.18.1; lockfile records the VM as truth, Josh's venv left alone.
   CONTAINER still open. Was: `requirements.txt` has 12 deps and
   ZERO pinned versions. An unpinned pvlib/shapely minor bump can move
   published kWh figures with no commit to explain it.
5. **Per-region resume markers.** DONE 31 Aug — `src/run_stage.py` wraps each
   stage with preflight + a completion marker + `--skip-done`; staleness reuses
   preflight's input table, so a marker only counts while every declared input
   is older than it. `src/run_district_build.sh` is the resumable loop, with the
   merge-before-masks order written down and enforced. Was: the
   Queenstown rebuild has been relaunched 3x and each restart redoes region 1.
   ~25 regions x ~28 min = ~12 h all-or-nothing. Same change makes regions
   distributable.
6. **Split the big modules; move dev scripts out of `src/`.**
   `roof_segmentation.py` 2,979 lines holds fit + merge + repair + attach +
   an off-by-default reconstruction path. 10 analysis scripts in `src/` are
   imported by nothing. `roof_reconstruct.py` is 860 lines behind
   `USE_RECONSTRUCTION = False` yet imported by 6 modules — delete or promote.
7. **GeoParquet intermediates + prepared geometries.** 53 whole-file
   `json.load` sites; `solar_potential.geojson` read by 10 modules;
   `panel_layouts.geojson` >80 MB. `shapely.prepared.prep` used in 0 files
   despite repeated point-in-polygon being the panel-fitting inner loop.

8. **DEPRECATED SHAPELY API — found while chasing item 7, and the most
   dangerous thing in this whole review.** 68 calls to
   `shapely.vectorized.contains` across 28 files. That API is deprecated in
   Shapely 2.x and documented as "will be removed in a future version", and
   `requirements.txt` asked for `shapely>=2.0` — so a routine reinstall would
   have broken the geometry core outright. It was INVISIBLE because 28 modules
   call `warnings.filterwarnings("ignore")`.
   FIXED 31 Aug: migrated to `shapely.contains_xy`. Verified behaviour-neutral
   rather than assumed — bit-identical over 200,000 points on 400 real
   footprints including on-boundary vertices, and identical facet counts AND
   areas through the full segmentation path, A/B against the pre-migration tree.
   LESSON: those blanket `filterwarnings("ignore")` calls hid a countdown to a
   hard breakage. Worth narrowing them to the specific warnings they were added
   for.

9. **`score_all_marked.py` rewrote its own committed baseline ON IMPORT.**
   It has no `if __name__ == "__main__"` guard, so its whole body runs when the
   module is merely imported — which happened during this review, from a loop
   that only checked that modules import. The write is now guarded.
   Still open: `check_marked.py` and `score_markup.py` have the same missing
   guard (they only print, so they are noisy rather than destructive).

Also found: **`DEFAULT_MAX_JOBS = 10` contradicts the comment directly above
it**, which records that 11 workers got a run OOM-killed and says "six is what
has actually been measured working". FIXED 31 Aug, and the contradiction had a
reason: 11 workers died on the 18GB Mac and 10 runs fine on the 62GB VM, so no
single literal could be right for both. Now derived from total RAM, calibrated
to reproduce both measured-good numbers exactly (6 on the Mac, 10 on the VM).
NOTE for whoever revisits this: there is a SECOND comment at the definition site
saying the cap was raised from 6 because the real OOM cause (unbounded LiDAR
tile caching) was fixed with an LRU, and that steady RSS is "well under a
gigabyte" per worker. If that is still true, PER_WORKER_GB = 1.75 is
conservative and the Mac could run more than 6. Worth measuring rather than
guessing.

Genuinely good and worth not breaking: 25-27% comment density explaining WHY
with measured evidence, zero hardcoded absolute paths, resource limits derived
from real incidents, and the PVGIS external-validation harness.

## Planarity repair — landed 27 Aug (29bafa0)

Nothing ever checked that a returned "plane" was planar. Pilot scan: 9% of
facets over 1 m residual sd carrying **22% of all panels**; 31% over 0.5 m
carrying 50%. A real roof plane is 0.1–0.2 m.

Now split at density valleys in **raw height** (a step is a vertical
discontinuity; residuals against a plane you already distrust are meaningless).
Parts must survive erosion by half a panel width, or they are ducting, not a
storey. 59 of 70 random buildings untouched; buildings with a facet over 1 m
sd: 18 → 10.

**Not yet in any shipped data — needs a rebuild.** The wave-7 rebuild finished
before this landed.

### Where it stands after the 27 Aug session

`python src/scan_defects.py pilot` — **750 clean, 316 flagged of 1,066**
(nonplanar 253, sparse 104, carved 3). Re-run this after any pipeline change;
it is the list, and it takes ~25 min on 7 workers rather than 20 hours.

Landed today: planarity repair, a robust trigger to replace the sd one that was
measuring outliers, the defect scanner itself, and a deck-seeking plane fit for
obstruction height residuals.

### Array ranking — landed 27 Aug (6eb9c16, 53ba189)

29 Park St: "lonely panels and small arrays surrounding a large array", and the
density slider stripping a whole dim side before touching them.

Root cause: the pipeline computed contiguous array membership but ran it *after*
ranking, as frontend metadata only — so nothing in the ordering knew what an
array was. Grouping was by facet, which on a curved roof split into three
sections sees three big groups and no fragments.

Found while fixing it: `_assign_arrays` buffered by 0.35 **degrees** on a 4326
geojson — a ~39 km probe. **0 of 1,033 pilot buildings had more than one array**;
`array_id`/`array_size` have been meaningless in the tiles since they were added.
Now 833 of 1,033. Applies to shipped data with no re-fit.

**Defect worth fixing properly:** `rerank_layouts.py` re-implements
`panel_fitting.assign_fill_ranks` instead of calling it. The same facet-grouping
bug existed in both, and fixing one would not have fixed the other. One of them
should go.

### Josh's 20-pair verdict, 27 Aug — `data/roof_verdicts_20aug27.json`

**10 better, 0 worse, 5 both-still-bad, 4 similar, 1 both-good.** First batch
judgement rather than one screenshot at a time. Themes, by frequency:

| n | theme |
|---|---|
| 10 | roof shape not understood |
| 4 | scattered fragments / standalone panels |
| 3 | asks directly for a clean 3D planar roof model |
| 2 | obstructions missed · panels over ridges · fuzzy outlines |

His exact plane counts are ground truth worth keeping: 5 Isle St is **3 planes**,
2/8 Wakatipu Heights is **8** (4 sloping each way).

### Reconstruction, now selected per building (commit above)

Only considered where the segmenter measures under 70% on-plane, kept only on a
≥10 point gain that costs less than 10% of usable area. Fragmentation is
guarded by **area, not facet count** — 5 Isle comes back as one facet, so any
ratio bound rejects the three planes it actually has.

Landed: 5 Isle 1 → 3 facets · 47 Stanley 4 → 10 (mitre joints) · 53 Hallenstein
75% → 98% on-plane.

### Planar partition — the step-change prototype (`src/roof_partition.py`)

Josh's design constraint, verbatim: roofs are "straight lines, generally a few
different angles", complex ones are "the same principles… just more of them on
the same building footprint", and "will almost never be some type of organic
shape". So: partition the surveyed footprint with straight cuts, recursively,
stopping when one plane explains a region.

**Structure is solved.** Worst facet outline is 7–19 vertices against the
shipped segmenter's 12–1035. Matches Josh's own plane counts on 2/8 Wakatipu (8)
and 29 Edinburgh (4). Fuzzy outlines are now structurally impossible.

**Fit is not.** Coverage-adjusted, it beats the segmenter on 1 of 5 test roofs.

Three things learned the hard way, all worth not repeating:
- **Diagonal cuts are mandatory** — a hip bisects a corner, so it runs at 45° to
  both walls. Without diagonals a hip roof cannot be cut at all.
- **A cut must not have to pay for itself immediately.** 1/5 Sydney fits one
  plane at 13%; its best single cut reaches 16%. It needs ~10 cuts before fit
  improves. One-step lookahead is blind to exactly the roofs this is for.
- **The merge needs a height-step test** — third time this bug has appeared here.
  Near-parallel faces at different levels read as 0° apart. Compare planes at
  their *shared boundary*, never normals alone.

**Failed hypothesis, do not retry as-is:** letting sparse parts score at the
parent's fit instead of vetoing a cut. Sounds right, measured worse — coverage
fell on 3 of 5 (2/8 Wakatipu 100→76%, 29 Edinburgh 96→79%). Reverted.

**Not a partition bug:** 1/5 Sydney's "67% coverage" is `MAX_ROOF_SLOPE_DEG = 45`
dropping six faces at 45–47° that fit *well* (78–100%). Whether 45° is the right
cap is a product question, not a geometry one.

Next: the one real defect there is a 138 m² face at 12° sitting at 15% on-plane
that never gets split further. Find why `_best_cut` gives up on it.

### Partition validated on random buildings — 27 Aug (99b25bd)

Not just the five hand-picked roofs. On **21 random pilot buildings**, coverage-
adjusted on-plane fit: **partition better on 19, segmenter on 2.** Worst facet
outline: segmenter median 12 vertices / max **2,594**; partition median 9 / max
**15**.

The five roofs used during development were Josh's hardest cases, which
understated it — there the partition won 3 of 5.

**This is now a candidate to replace segmentation, not a prototype.** Before
wiring in: judge it on *layouts* (panels placed), not geometry, and get Josh's
verdict on a before/after sheet. Geometry metrics have misled twice.

Known and not a geometry defect: 1/5 Sydney covers only 46% because
`MAX_ROOF_SLOPE_DEG = 45` drops 45–47° faces the partition finds and fits *well*
(78–100%). Whether 45° is the right cap is Josh's call.

### Open, in priority order

0. **55 Arrowtown (#4729642) needs Josh's eye.** Reconstruction fires (46%
   inlier) and it moves a lot: obstructions 35% → 0%, panels 87 → 186,
   on-raised 0 → 12. Filed under over-carve, which would make turning that
   175 m² into roof correct — but 12 panels on raised structure disagrees.
   Not resolvable from the validation set as annotated.

0b. **Curved roofs are still wrong.** 29 Park St comes back as 17 reconstructed
   facets; Josh says the light curve should be one face. The bridge angle is
   held at 5° on purpose — raising it re-creates the ridge-spanning defect he
   reported on #14 and #18. Needs a curve-vs-fold discriminator (a crease shows
   in the residuals at a real fold; a curve does not), not a looser threshold.

1. **253 buildings whose worst facet is still not a plane.** The top of the
   list is severe — #4734915 has **1%** of its points within 30 cm of its own
   plane, #5371119 31%, #4725584 (3,499 m²) 36%. The planarity repair either
   does not fire on these or fires and fails. This is the single biggest
   remaining defect class and it is what Josh keeps finding by eye.
2. **Regression on the equipment reference #5370338**: 6 panels on raised
   structure against a baseline of 1 (was 10 before the deck fit). Its ducting
   is now excluded by the planarity repair rather than flagged as an
   obstruction, and a few panels still find raised ground.
3. **The deck-trust gate at 0.50 is set on eight buildings** and the two
   populations nearly touch (51%/48% want it, 49%/46% must not have it). Widen
   the validation set before trusting it further.
4. **104 buildings flagged sparse** — fill under 45% of usable roof. Nothing
   has been done here yet; 10 Brecon St is the example Josh reported.
5. **1 Earl St** is the one under-detect case unmoved by anything: 12 panels on
   structure up to 0.95 m proud, through every change this session.

## Realism merge: correct but nearly inert (27 Aug)

Constrained to a 4 deg cap -- the steepest join a rigid panel can lie across --
it changes almost nothing: pilot panels 71,852 -> 71,868, facets 4,443 -> 4,437.

The large gain it showed before the cap (+1,596 panels, -33% facets) came from
merging across REAL ridges, median accepted angle 19.6 deg, which Josh caught
on the map as panels crossing roof sections.

**Conclusion: slivers cannot be merged away after the fact.** They are not
spurious subdivisions of one plane; they are genuinely different planes the
segmenter found. Getting "few large blocky faces" requires the segmenter to
produce them, which makes imagery-first boundaries the route, not an optional
extra. The merge stays in (it is correct and costs nothing) but is not the fix.

## APPLY AS SOON AS THE REBUILD CLEARS (27 Aug)

**Colour corroboration keeps the whole blob.** In detect_obstructions_combined,
a colour blob is kept entire when >=15% of it overlaps height evidence. On a
flat commercial roof the colour path flags membrane tone and shadow -- 7
Shotover St (#4734932): 242.7 m2 flagged on a 462 m2 roof -- so a 73 m2 tonal
region containing one small real vent is carved whole. That is the big pink
region Josh reported.

Fix: keep the corroborated PORTION, not the blob. Colour says where a tonal
region is; height says what is actually raised. Take the intersection with the
height evidence plus a small margin, and keep the whole blob only when it is
small enough to be a plausible single object anyway.

Validate both directions afterwards -- the equipment reference (#5370338) is
the canary, and the colour path is what finds pale plant the height path
misses.

## IN PROGRESS -- resume here

**Panel fitting is the maximum priority (Josh).**

1. **Realism merge** -- DONE and committed (`0222213`). Stops splitting roofs
   where the split costs more usable area than the yield it buys. Pilot
   rebuilding now with it; baseline for comparison is
   `scratchpad/pilot_old_layouts.geojson`.
   **Next step:** when the build finishes, run
   `python src/compare_layouts.py --old <that file> --area pilot --n 12`,
   publish it, and have Josh judge. Layouts are the judge, not counts.
2. **Imagery-first roof shapes** -- not started. Josh's proposal: derive roof
   boundaries from the 0.1 m imagery, where ridgelines are usually a clear
   tonal break, and use LiDAR only for the slope of each face and where the
   image is unreadable (bright roofs, flat commercial with no internal edges).
   Three separate failures today all wanted imagery: deck detection,
   obstruction footprints, ridge placement.
3. **"Lifetime ROI" reads like a rate** and is not -- 201% annualises to about
   3.8%/yr. Label it or show an annualised figure.

## Next up

| # | Item | Why | Blocked by |
|---|------|-----|------------|
| 1 | **Strong-path cap rework** | The size cap rejects large real plant *before* the above-plane test that identifies it. 76 panels sit on real structure across the validation set. Patch is written: `scratchpad/apply_cap_rework.py` | Must not be applied while a rebuild is running — each area is a fresh Python process, so a mid-run edit gives some areas old code and some new |
| 2 | **Business layouts more conservative** | Josh: flat commercial roofs have space to spare, so bigger setbacks around obstructions are cheap there | Touches `panel_fitting.py`; wait for rebuild |
| 3 | **Marginal-rate variant of panel economics** | Current highlight uses the building's blended rate. Strictly, a marginal panel *exports*, so a stricter test would use the export rate alone | Nothing — frontend only |
| 4 | **440 W / 550 W dual panel sizing** | Not started | — |

## Known-real, not yet measurable

- **Decks/balconies counted as roof.** Confirmed by Josh on 1/49 Belfast Terrace
  (facets over 99% of a 228 m² outline, about half of it open deck). Inflates
  capacity and puts panels on balconies. **Three detection attempts failed** —
  see `src/audit_decks.py` for what and why. Needs imagery, not LiDAR.
- **2 Kent St class: extraction misses small faces.** Both the shipped model and
  the reconstruction find 4 faces where Josh counts 13–14. Greedy RANSAC at a
  0.15 m tolerance absorbs small raised faces into the large planes near them.
  Fix would be normal-based region growing.
- **Under-detection.** Panels still on real structure at 1 Earl St, 17 Marine
  Pde, 35 Shotover St. Item 1 is the first attempt at this.

## Reconstruction: rejected in this form (26 Aug)

Josh reviewed ten before/after layouts and called it worse on all ten. The
cause is a design error, not tuning: `panel_fitting` erodes every facet by
`RIDGE_SETBACK_M` and panels cannot span facets, so splitting a roof costs
usable area every time (a 6 m² facet keeps 57% of itself, a 400 m² one keeps
94%).

**Rule for any next attempt: few large blocky faces.** A split must earn back
the setback area it costs. Judge on layouts, never on plane count or
off-plane residual -- both scored this version as fine.

## Bigger bets

- **Imagery-guided boundaries.** Imagery is 0.1 m, LiDAR ~0.42 m — a 4×
  advantage on edges that is currently unused. Three separate problems have now
  pointed here: deck detection, obstruction footprints, and ridge placement.
- **Optimise layout quality directly, not plane counts.** Today showed that
  optimising a proxy leads astray. `audit_layouts.py` already measures the hard
  violations; that should be the objective.

## Blocked on Josh

- **Permanent LINZ developer key** (`basemaps@linz.govt.nz`). Gates imagery for
  four areas — **1,901 buildings** that cannot be built at all.

## Done today (26 Aug)

- Self-consumption modelled as a daytime load, not a share of output
- Heat-map resolution swaps cross-fade instead of blinking
- Parallel area driver (`run_layouts_regate_par.sh`) — 8× on the rebuild
- Tariff assumptions surfaced on the building panel; ROI added
- Heat-map economics follow the coverage choice
- Obstruction blobs closed rather than only dilated (+3.2% panels on the
  validation set) — currently rebuilding
- 20 roofs labelled by Josh; scorer in `src/score_labels.py`
- Per-panel economics highlight with three ROI bands and filtering

## Standing rules

- Never edit pipeline files while a rebuild is running.
- A sweep must charge a crash as a failure, not skip it — a skipped case
  contributes zero error and makes the settings that crash most look best.
- Validate obstruction changes in BOTH directions (`validate_obstructions.py`);
  the equipment reference is the canary.
- Read the warnings already in a file before using it.
- Anything derived from state goes through `refreshDerived()`. Do not hand-pick
  dependents at a mutation site -- that is how the generation curve ended up
  describing panels that had been filtered away.
- One rule, one definition. `panelVisible` / `panelBandOf` decide which panels
  count, everywhere. The band rule had been written out four times and two
  copies were stale.
- A patch script that asserts and then writes at the end is all-or-nothing: a
  failed assertion late in the script silently discards the edits before it.
  Write after each edit, or verify the file actually changed.

## Horizon program — SHIPPED 31 Aug (per-building horizons + tab live on national site)

Landed overnight 30-31 Aug (details in the repos' commit log):
- src/building_horizon.py: two-layer 72-bin per-building profile (far 8m DEM
  at eave height, near 1m DSM to 300m, own footprint excluded), base64-baked
  onto solar_potential (horizon_b64 / horizon_far_b64 / horizon_beam_pct).
- Horizon tab on the building panel (terrain silhouette + darker trees/
  buildings + 3 seasonal sun arcs, client-side). Hidden until an area bakes.
- Yield wiring: aspect-aware facet_horizon_factor (far layer vs area
  baseline) multiplies facet+panel shading; far_beam_ratio scales heatmap
  rasters. Near-field stays with building_shading_factor (no double count).
- Validated internally on Island Bay: median beam 92.5%, p5 64%; the 11-29%
  extremes are tree-buried buildings that already carry zero panels.

STILL OPEN on horizons:
- SolarView cross-check: NIWA moved SolarView behind DataHub (API key) --
  needs Josh or a key. Winter-curve comparison for ~5 addresses.
- Queenstown: needs data/dem_wide_mosaic.tif + a bake before its tab lights
  up; yield wiring is a no-op there until then. Bake with next build cycle.
- tshade (seasonal-curve mask) still comes from build_terrain_masks' own DEM
  scan, not the baked horizon -- consolidate so one profile feeds both.
- Island Bay relayout carrying horizon yields staged on the VM (ib_relayout.sh),
  runs after the Queenstown chain.

## Horizon program original spec (kept for reference)

Josh: build the SolarView-style horizon tab, and "make sure all calculations,
like generation profiles, economics and savings that show, and heat maps, all
take into account these horizons to make sure you are confident in their
accuracy." One per-building horizon as the single source of truth, everywhere.

1. **Precompute per building** (VM build stage): 72-bin azimuth horizon in two
   layers — far terrain (8 m wide DEM, ~20 km) and near-field DSM (~300 m,
   trees + neighbours, own footprint excluded, ray origin roof-centre at eave
   height). ~200 bytes/building into solar_potential properties.
2. **Feed the model**: per-building hourly beam mask (sun el < horizon(az) →
   blocked) replacing the per-AREA terrain profile in SolarModel; flows into
   panel ac_kwh_year, so economics/savings inherit automatically. Heatmap
   stage uses the same factors. Frontend seasonal curves must derive from the
   same numbers.
3. **Horizon tab** on the generation chart: azimuth E→N→W, terrain silhouette,
   darker DSM overlay, four seasonal sun arcs computed client-side.
4. **Validation**: cross-check horizons + winter curves against NIWA SolarView
   for ~5 addresses (external ground truth), and record deltas in
   data/roof_truth.json.
   Caveats to keep honest: LiDAR-vintage tree heights; ray-origin choice.

## Multi-level roofs — top-surface filter landed 30 Aug (Josh's #5119630 report)

Island Bay #5119630 (complex hip, two roof levels) exposed a foundational bug:
under the eaves of a multi-level building LiDAR records BOTH surfaces at the
same plan location, and every segmentation strategy was fitting planes through
the mixture -- region-grow facets explaining 0% of their own polygons, the
partition smearing 12 wedges across a clean hip network. Fix:
`roof_partition.top_surface()` keeps only each 0.5 m plan cell's top points
(exposed lower-level roof keeps its points -- there it IS the top). Partition
explained-fraction on the report building went 0.66 -> 0.76 and the render
now follows the visible ridge network; #3528763 (textbook hip) comes out
with crisp hip diagonals.

Also landed: `explained_fraction` metric + a 0.85 gate in `_partition_facets`;
below it a label-based plane-arrangement rebuild (`partition_with_labels`:
region-grow labels for assignment, plane-intersection + reflex-corner cuts
for boundaries) competes and strictly-better wins. On current buildings the
top-surface partition usually wins; the arrangement is the safety net.

NEW OPEN ITEM from the same sweep: two-level flat commercial (#3371280) --
the lower band, now cleanly separated by top_surface, gets carved as an
OBSTRUCTION (height-strong, below main plane) instead of becoming its own
facet with panels. Conservative, not wrong, but it's free roof area. Wants:
when a large height-coherent "obstruction" is itself planar and sub-level,
re-run segmentation on its points and emit facets.

Sample regression (15 Island Bay buildings): 622 -> 540 panels (-13%), the
big movers verified by render as geometry corrections (panels were on wedges
crossing ridges). District rebuild will carry this; watch the diff report.

## EXTERNAL VALIDATION vs PVGIS -- two real biases found 31 Aug

src/validate_against_pvgis.py cross-checks our in-plane irradiation against
PVGIS (EU JRC, free, no key, global via ERA5). It compares H(i)_y -- annual
in-plane IRRADIATION -- not PV yield, so it tests what our lookup table
actually computes rather than PV physics we never attempted. Cached in
data/pvgis_cache.json; re-run costs nothing.

FINDING 1 -- beam:diffuse partition is wrong, both sites.
Our clear-sky irradiance is scaled by a monthly cloud factor, which PRESERVES
the clear-sky beam:diffuse ratio. Real cloudy hours are mostly diffuse, so we
carry too much beam and too little diffuse. Signature, at 35 deg tilt:
    Queenstown   north +11.1%   south -18.3%   (spread 29 pts)
    Island Bay   north  -4.8%   south -16.6%   (spread 12 pts)
Surfaces that live on diffuse light (south-facing, shaded, winter) are
understated; sun-facing pitched roofs are overstated. Flat planes are close,
because they see the total. The frontend already documents this limitation in
renderSeasonCurves -- it is now quantified from outside.
FIXED 31 Aug: GHI is scaled alone and split with Erbs, and transposition
moved from the isotropic default to Perez. Queenstown angular spread went from
29 points to 3 (worst case -18.3% -> +6.6%). Seasonal curves carry the same
change so the chart cannot contradict the figure above it, and the frontend's
flat 0.18 diffuse floor was replaced by a real beam-removed curve.

FINDING 2 -- SUPERSEDED, and it was backwards. RESOLVED 31 Aug.
The original entry said Wellington ran ~5-8% LOW, on the strength of PVGIS
alone. NIWA's published 1991-2020 measured normals say the opposite: against
Kelburn station (6 km from Island Bay) Wellington was ~8% HIGH.
    Island Bay flat plane   ours 1498 -> 1384   NIWA Kelburn 1387   PVGIS 1609
PVGIS/ERA5 overstates Wellington by ~16% versus the ground station and
understates Queenstown -- a 30 km reanalysis cell is not ground truth in
coastal or alpine terrain. LESSON: do not treat one external source as truth.
PVGIS remains excellent for the SHAPE of an error across angles (it found the
beam:diffuse bug); measured data settles absolute level.
FIXED by adding NIWA's 28-station measured radiation normals to solar_model as
the calibration source ahead of the sunshine-hours inference. Any future region
now calibrates from the nearest measured station automatically.

## FOR JOSH: the system derate looks ~4-5 points optimistic (31 Aug)

Not changed, because it is an ASSUMPTION you set and it is editable in the
UI -- changing it silently would move every payback figure on the site. But
the evidence is consistent and worth a decision.

Our AC yield sits 11.7-13.3% above PVGIS across angles at Queenstown. It
decomposes cleanly:
    irradiance  ours 1559 vs PVGIS 1471 at 20 deg north   +6.0%
    derate      ours 0.834 vs PVGIS effective 0.791       +5.4%

The irradiance half is defensible: NIWA's MEASURED record supports our level
for Queenstown, and PVGIS/ERA5 reads low in that alpine valley (it reads high
in Wellington -- see the corrected Finding 2).

The derate half is the question. PVGIS at loss=0 gives E/H = 0.904, so cell
temperature + spectral + reflection cost ~9.6% BEFORE any system losses. The
industry convention (PVWatts) is 14% system losses with temperature modelled
separately on top, landing near 21% total in a temperate climate. We apply a
single 16.6% (97% inverter x 14% "soiling, wiring, temperature, mismatch"),
which only works if temperature is genuinely inside that 14% -- and if it is,
it leaves ~4% for everything else, which is tight.

OPTIONS
  a) raise system_derate_pct 14 -> ~18-19, one-line change, matches the
     PVWatts-style total. Cheapest and defensible.
  b) model cell temperature properly (pvlib pvwatts_dc + a temperature model),
     which needs hourly ambient temperature and wind we do not currently fetch.
  c) leave it and say so in the assumptions text.
Effect of (a): headline kWh and savings fall ~5%, paybacks lengthen slightly.

## Build-order rule learned 31 Aug (cost a rebuild)

Stages that write ONLY the merged data/solar_potential.geojson must run AFTER
merge_regions, because the merge regenerates that file from the per-region
files and silently discards anything the regions do not carry:
  - build_terrain_masks (tshade)   <- lost this on the Island Bay rebuild
  - bake_density_deciles (fill_*)
Stages that write the REGION file survive the merge (add_addresses,
patch_roof_confidence, bake_building_horizons -- the last writes both).
There is no error when this goes wrong; the curves just come out unshaded.

## Also queued from the 29-30 Aug live-testing sessions

- District rebuild carrying: fill-order/mean-yield, confetti demotion,
  gap-fill (Anderson 24→35 at 100%), twin-portrait rule, fusion + sunken
  obstructions, eviction fix. Run on the VM after Island Bay ships.
- Region-growing segmentation for multi-level commercial (32 Frankton at 15%
  fit; 40 Camp). The big-roof panel gate hides the symptom, not the cause.
- Imagery classifier (SAM-proposer + LiDAR/colour verification): Panorama's
  recessed deck (geometric routes exhausted, all measured), existing rooftop
  solar (7 Coronation), Robins-style patio boundaries, Duke's residual speckle.
- Array regularity: "clean rectangular blocks, equal rows" (45 Camp north
  array gap).
- Bucket uploads rejected (gs://rewiring-solar-data empty; SA/composite issue)
  — blocks the Mac-offload and VM disk relief.
- VM GitHub deploy key so cloud builds push their own results.
- Full review tables (bugs fixed / simple done / big for Josh) — still owed.
- Wellington: swap to 2025 survey when its point cloud is downloadable.

## Curved-roof strip-planes — from 19 Camp St (30 Aug)

The model is plane-only; a curved roof gets approximated by a few big sheets
and panels run across the crest (Josh: "panels overlapping ridgelines...
have you built in a way to do curves?"). Plan: detect curvature (quadratic
surface fit decisively beating the plane fit, distinct from folds which are
creases), then partition the curved face into narrow parallel strips along
the curvature direction -- one panel row per strip, racked the way installers
actually follow gentle arcs. Exclude tight radii as un-rackable.

## Rebuild diff report — from Josh's regression question (30 Aug)

Nothing diffs the whole district build-over-build; the scorecard covers ~20
known roofs and change-evals cover ~120 samples. Add to the rebuild fan-in:
per-building panel-count and kWh diff vs the previous build, ranked movers
both directions, auto-render the top ~10 outliers for Josh's review BEFORE
the push. Unknown impacts must surface, not hide in 15k buildings.

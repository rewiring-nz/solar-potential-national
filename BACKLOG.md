# Backlog (Wellington deployment)

This repository is the Wellington / national deployment of the rooftop solar
pipeline. The code is shared with `solar-map` (Queenstown) by hand, and the
shared work -- geometry, layout, model, scale -- is tracked in
`solar-map/BACKLOG.md`. Only Wellington-specific items live here.

## Open

- **Island Bay rebuild.** The live Island Bay layouts were built with
  imagery on a workstation; the VM has never had
  `data/regions/island_bay/imagery_mosaic.tif`, and `fetch_regions.py`
  times out fetching it there. Ship the 2.0 GB file to the VM before any
  rebuild (`src/run_island_bay.sh` refuses to build LiDAR-only), then
  rebuild to pick up every change since 31 Aug (gap-fill, straggler yield
  order, the frame, the ridge snap, the 23 Sep segmentation fixes) and the
  density heat map (`cellpts` layer; the page keeps the block fills until
  the data is recombined).
- **Data version** is 33 here; bump with the rebuild.

## Standing rules

- `tools/check_repo_sync.py` (in `solar-map`) must report no unexpected
  divergence before pushing shared files. Per-deployment files: `config.py`,
  `site-config.js`, `src/patch_buildings.py`.
- Hand-drawn markup always wins over fitted geometry.
- Nothing at district scale is built on a laptop.

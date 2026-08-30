"""
Read the raw LiDAR point cloud tiles (data/pointcloud/*.copc.laz) as a
higher-resolution stand-in for the 1m gridded DSM.

Confirmed directly on a real tile before building this: ~8.6-8.8 points/m2
overall, ~5.8 points/m2 restricted to LAS classification 6 ("building")
alone -- roughly 6-9x the density the 1m DSM grid implies (1 point/m2 by
construction), plus classification lets us throw out tree/ground returns
the DSM has no way to distinguish from roof.

Rather than rewrite roof_segmentation.py's RANSAC/shape/dedupe pipeline to
work on an irregular point set directly (a bigger, riskier rewrite), this
rasterizes the point cloud onto a finer regular grid (median z per cell)
and hands back the exact (window_array, window_transform) shape a DSM read
already produces -- everything downstream of points_from_window is reused
completely unchanged, exactly as this project's own earlier documentation
anticipated ("a drop-in upgrade to points_from_window without touching the
RANSAC/vectorize code").
"""
import sys
from pathlib import Path

import laspy
import numpy as np
from collections import OrderedDict
from affine import Affine
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POINTCLOUD_DIR = Path(__file__).resolve().parent.parent / "data" / "pointcloud"
BUILDING_CLASSIFICATION = 6  # LAS standard classification code
MIN_BUILDING_POINTS = 10  # below this, the classification filter isn't trustworthy -- use all points instead


MAX_CACHED_TILES = 8  # decoded LiDAR tiles held per process. See __init__.
# Callers that FAN OUT must pass something smaller: 8 workers x 8 cached tiles
# was survivable on Queenstown's ~2026 survey but Wellington's 2019 tiles
# decode several times larger, and the parallel panel gate reached >50 GB and
# took Josh's machine down with it. Total memory is workers x tiles x decoded
# size -- budget it explicitly wherever both multipliers are in play.


class PointCloudSource:
    """Tile bounds are read from LAZ headers only (cheap) at construction;
    a tile's actual points are decoded from disk on first use and cached
    in memory after that."""

    def __init__(self, directory=POINTCLOUD_DIR, max_cached_tiles=None):
        # *.laz matches both the pilot's original *.copc.laz tiles and the
        # plain .laz tiles fetch_pointcloud_regions.py pulls from
        # OpenTopography's bulk store -- laspy reads either identically.
        self.tile_paths = sorted(
            q for q in Path(directory).glob("*.laz")
            # macOS tarballs ship AppleDouble "._foo.laz" metadata siblings;
            # laspy reads one and dies with 'Invalid file signature' -- inside
            # a pool initialiser that surfaces only as BrokenProcessPool, which
            # cost an hour of debugging on the VM. Nothing hidden is a tile.
            if not q.name.startswith("._") and not q.name.startswith("."))
        if not self.tile_paths:
            raise FileNotFoundError(f"No .laz tiles found in {directory}")
        self._bounds = {}
        for path in self.tile_paths:
            # Single-threaded decompression when fanned out. The default lazrs
            # backend spawns its own thread pool per read; 12 gate workers each
            # doing that put load average 141 on a 16-core VM and turned a
            # ~30-CPU-minute job into 21 CPU-hours of context switching. One
            # decode thread per worker process is the efficient shape.
            import os as _os
            _backend = ([laspy.LazBackend.Lazrs]
                        if _os.environ.get("SOLAR_LAZ_SINGLE") else None)
            with laspy.open(path, laz_backend=_backend) as f:
                h = f.header
                self._bounds[path] = (h.mins[0], h.mins[1], h.maxs[0], h.maxs[1])
        # Bounded, and it has to be. The cache was unbounded, which is fine for
        # one process working through an area but fatal in parallel: a single
        # build worker is already at 1.6 GB partway through the pilot and keeps
        # growing as it touches more tiles, so four of them late in a run
        # exhausted a 19 GB machine and every worker was killed -- leaving the
        # parent hung on a pool with nothing left in it, twice, before the
        # cause was clear. An LRU of a few tiles keeps the working set flat
        # while still holding whatever a run of nearby buildings needs, because
        # buildings are processed in roughly spatial order.
        self._cache = OrderedDict()
        self._max_cached = max_cached_tiles or MAX_CACHED_TILES

    def _tiles_overlapping(self, minx, miny, maxx, maxy):
        return [
            path for path, (tminx, tminy, tmaxx, tmaxy) in self._bounds.items()
            if tminx <= maxx and tmaxx >= minx and tminy <= maxy and tmaxy >= miny
        ]

    def _load_tile(self, path):
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached
        las = laspy.read(path)
        decoded = (
            np.asarray(las.x, dtype=np.float64), np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64), np.asarray(las.classification),
        )
        self._cache[path] = decoded
        # STRICTLY greater: '>=' made a capacity-2 cache hold ONE tile, so a
        # building straddling a tile border re-decoded a 7.3M-point tile for
        # EVERY panel -- >13 s each on the VM, an entire gate run spent
        # decompressing. Off-by-ones in eviction are invisible until capacity
        # is small and the data is dense, which is exactly when they ruin you.
        while len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
        return decoded

    GROUND_CLASSIFICATION = 2  # LAS standard: bare earth

    def ground_points_in_bbox(self, minx, miny, maxx, maxy):
        """Ground-class returns only. Ground classification is the most
        reliable field in a LAS file (it is what the whole survey is graded
        on), which is why the placement gate measures roof height against
        these rather than against building-class flags or a smoothed DEM."""
        xs, ys, zs = [], [], []
        for path in self._tiles_overlapping(minx, miny, maxx, maxy):
            tx, ty, tz, tc = self._load_tile(path)
            m = ((tx >= minx) & (tx <= maxx) & (ty >= miny) & (ty <= maxy)
                 & (tc == self.GROUND_CLASSIFICATION))
            if not m.any():
                continue
            xs.append(tx[m]); ys.append(ty[m]); zs.append(tz[m])
        if not xs:
            return np.empty((0, 3))
        import numpy as _np
        return _np.column_stack([_np.concatenate(xs), _np.concatenate(ys), _np.concatenate(zs)])

    def points_in_bbox(self, minx, miny, maxx, maxy, building_only=True):
        """Returns Nx3 (x, y, z) array."""
        xs, ys, zs, classes = [], [], [], []
        for path in self._tiles_overlapping(minx, miny, maxx, maxy):
            tx, ty, tz, tc = self._load_tile(path)
            mask = (tx >= minx) & (tx <= maxx) & (ty >= miny) & (ty <= maxy)
            if not mask.any():
                continue
            xs.append(tx[mask]); ys.append(ty[mask]); zs.append(tz[mask]); classes.append(tc[mask])
        if not xs:
            return np.empty((0, 3))
        x, y, z, c = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs), np.concatenate(classes)
        if building_only:
            building_mask = c == BUILDING_CLASSIFICATION
            if building_mask.sum() >= MIN_BUILDING_POINTS:
                return np.column_stack([x[building_mask], y[building_mask], z[building_mask]])
        return np.column_stack([x, y, z])


def rasterize_pointcloud_window(pc_source, building_geom, resolution, pad_m=2.0):
    """Bins point-cloud points inside (a small pad around) building_geom's
    bounds onto a regular grid at `resolution` -- median z per cell, same
    "one representative height per cell" idea a DSM already embodies, just
    at a finer grid and from real classified building points instead of
    whatever a coarser cell's highest return happened to be. Returns
    (window_array[1,H,W], window_transform, nodata) matching
    rasterio.mask.mask's return shape, so points_from_window can consume
    it exactly as it already does for a real DSM read. Returns
    (None, None, None) if too few points fall in the window to be useful."""
    minx, miny, maxx, maxy = building_geom.bounds
    minx, miny, maxx, maxy = minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m
    pts = pc_source.points_in_bbox(minx, miny, maxx, maxy, building_only=True)
    if len(pts) < MIN_BUILDING_POINTS:
        return None, None, None

    width = max(1, int(np.ceil((maxx - minx) / resolution)))
    height = max(1, int(np.ceil((maxy - miny) / resolution)))
    transform = Affine(resolution, 0, minx, 0, -resolution, maxy)

    col = np.clip(((pts[:, 0] - minx) / resolution).astype(np.int64), 0, width - 1)
    row = np.clip(((maxy - pts[:, 1]) / resolution).astype(np.int64), 0, height - 1)
    flat = row * width + col

    order = np.argsort(flat)
    flat_sorted, z_sorted = flat[order], pts[order, 2]
    unique_idx, start = np.unique(flat_sorted, return_index=True)
    bounds_idx = np.append(start, len(flat_sorted))
    medians = np.array([
        np.median(z_sorted[bounds_idx[i]:bounds_idx[i + 1]]) for i in range(len(unique_idx))
    ])

    nodata = -9999.0
    grid = np.full(height * width, nodata, dtype=np.float32).reshape(height, width)
    grid.flat[unique_idx] = medians

    # A real DSM is interpolated to be gapless; a naive per-cell bin of raw
    # points isn't -- confirmed directly: even at a resolution matched to
    # the point cloud's average density, individual cells routinely have
    # zero points (scan-pattern gaps, occlusion), so an unfilled grid comes
    # out mostly *empty* (measured 15% filled at 0.3m for a real building),
    # breaking the connected-component step downstream (ndimage.label sees
    # a field of disconnected single-cell islands instead of one
    # contiguous roof) -- which is why an earlier, unfilled version of this
    # function made segmentation dramatically *worse* despite denser input
    # data. Nearest-neighbour fill closes the gaps the same way DSM
    # production already does; cells that end up filled from far outside
    # the true roof get clipped away later when the facet is intersected
    # against the (accurate, imagery-derived) building outline anyway.
    invalid = grid == nodata
    if invalid.any() and not invalid.all():
        nearest_idx = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
        grid = grid[tuple(nearest_idx)]

    window_array = grid.reshape(1, height, width)
    return window_array, transform, nodata

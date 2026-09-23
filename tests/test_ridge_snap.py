"""Ridge snap on a synthetic gable: the crest is where the returns say."""
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import ridge_snap
from src.ridge_snap import snap_ridges_to_crest


class _Cloud:
    def __init__(self, pts):
        self.pts = pts

    def points_in_bbox(self, minx, miny, maxx, maxy, building_only=False):
        p = self.pts
        m = (p[:, 0] >= minx) & (p[:, 0] <= maxx) & (p[:, 1] >= miny) & (p[:, 1] <= maxy)
        return p[m]


def _gable(crest_x, width=8.4, length=16.0, pitch_deg=24.0, density=4.0, seed=0, noise=0.05):
    """A wing from x=0..width, y=0..length, ridge along y at x=crest_x."""
    rng = np.random.default_rng(seed)
    n = int(width * length * density)
    x = rng.uniform(0, width, n)
    y = rng.uniform(0, length, n)
    s = np.tan(np.radians(pitch_deg))
    z = 10.0 - s * np.abs(x - crest_x) + rng.normal(0, noise, n)
    return np.column_stack([x, y, z])


def _facets(boundary_x, width=8.4, length=16.0, hip=False):
    west = {"geometry": Polygon([(0, 0), (boundary_x, 0), (boundary_x, length), (0, length)]),
            "slope_deg": 24.0, "aspect_deg": 270.0}
    east = {"geometry": Polygon([(boundary_x, 0), (width, 0), (width, length), (boundary_x, length)]),
            "slope_deg": 24.0, "aspect_deg": 90.0}
    fs = [west, east]
    if hip:
        # a hip triangle whose apex sits on the ridge end must move with it
        fs.append({"geometry": Polygon([(0, length), (width, length), (boundary_x, length + 0.01)]).buffer(0),
                   "slope_deg": 24.0, "aspect_deg": 0.0})
    return fs


def _boundary_x(f):
    xs = np.asarray(f["geometry"].exterior.coords)[:, 0]
    return xs.max() if f["aspect_deg"] == 270 else xs.min()


def test_snaps_to_crest():
    cloud = _Cloud(_gable(crest_x=4.4))
    out = snap_ridges_to_crest(_facets(3.66), cloud)
    bx = _boundary_x(out[0])
    assert abs(bx - 4.4) < 0.12, bx
    assert abs(_boundary_x(out[1]) - bx) < 1e-6
    assert all(f["geometry"].is_valid for f in out)


def test_leaves_a_good_boundary():
    cloud = _Cloud(_gable(crest_x=4.2))
    out = snap_ridges_to_crest(_facets(4.1), cloud)
    assert abs(_boundary_x(out[0]) - 4.1) < 1e-9


def test_hip_apex_moves_with_ridge():
    cloud = _Cloud(_gable(crest_x=4.4, length=16.0))
    out = snap_ridges_to_crest(_facets(3.66, hip=True), cloud)
    apex = max(np.asarray(out[2]["geometry"].exterior.coords), key=lambda c: c[1])
    assert abs(apex[0] - _boundary_x(out[0])) < 1e-6, (apex, _boundary_x(out[0]))


def test_labelled_roofs_untouched():
    cloud = _Cloud(_gable(crest_x=4.4))
    fs = _facets(3.66)
    fs[0]["from_labels"] = True
    out = snap_ridges_to_crest(fs, cloud)
    assert abs(_boundary_x(out[0]) - 3.66) < 1e-9


def test_flat_roof_is_not_a_ridge():
    rng = np.random.default_rng(1)
    n = 500
    pts = np.column_stack([rng.uniform(0, 8.4, n), rng.uniform(0, 16, n), 10 + rng.normal(0, 0.05, n)])
    out = snap_ridges_to_crest(_facets(3.66), _Cloud(pts))
    assert abs(_boundary_x(out[0]) - 3.66) < 1e-9


def test_wandering_crest_is_refused():
    # crest drifts 2 m along the wing: not a line, so nothing moves
    pts = _gable(crest_x=4.4)
    pts[:, 2] = 10.0 - np.tan(np.radians(24)) * np.abs(pts[:, 0] - (3.0 + pts[:, 1] / 8.0))
    out = snap_ridges_to_crest(_facets(3.66), _Cloud(pts))
    assert abs(_boundary_x(out[0]) - 3.66) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            ridge_snap.SNAP_STATS.update({"pairs": 0, "measured": 0, "snapped": 0, "moved_m": []})
            fn()
            print("ok", name)

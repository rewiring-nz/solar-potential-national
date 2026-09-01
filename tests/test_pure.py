"""
Unit tests for the pure functions -- the arithmetic you cannot see.

Why these functions and not others: every regression this project has caught
so far was caught by LOOKING at something. That works for geometry, which is
visible, and fails completely for coordinate conventions and unit conversions,
which are not. The irradiance bias found on 31 Aug lived undetected for as
long as it existed because every internal check agreed with every other
internal check. The functions below are the ones where a silent sign flip or
a factor-of-two would change published kWh figures with nothing on screen
looking wrong.

They are all pure -- no I/O, no rasters, no network -- so they are fast and
deterministic, which is the whole reason to start here.

Run:  .venv/bin/python tests/test_pure.py
      (or `pytest tests/` if pytest is ever added to the venv -- the
      test_* naming and bare asserts work under both.)
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.building_horizon import N_BINS, AZ_STEP, decode_horizon, encode_horizon
from src.solar_model import _nearest_bin
from src.terrain_horizon import horizon_angle_at
from src.validate_against_pvgis import our_aspect_to_pvgis


# --------------------------------------------------------------------------
# Aspect convention: ours is a compass bearing (0 = north, 90 = east).
# PVGIS uses 0 = SOUTH, negative = east, positive = west, in both hemispheres.
# Getting this backwards would make the whole external validation agree with
# itself while comparing north-facing roofs against south-facing ones -- which
# in New Zealand is the difference between the best and worst roof on a house.
# --------------------------------------------------------------------------

def test_aspect_cardinals_map_to_pvgis():
    assert our_aspect_to_pvgis(180) == 0.0      # south -> PVGIS zero
    assert our_aspect_to_pvgis(90) == -90.0     # east  -> negative
    assert our_aspect_to_pvgis(270) == 90.0     # west  -> positive
    assert abs(our_aspect_to_pvgis(0)) == 180.0 # north -> the far side


def test_aspect_always_within_pvgis_range():
    for a in range(0, 360, 5):
        v = our_aspect_to_pvgis(a)
        assert -180.0 <= v <= 180.0, f"aspect {a} -> {v} outside PVGIS range"


def test_aspect_is_a_rotation_not_a_reflection():
    """A 10 deg step east of north must stay a 10 deg step in PVGIS terms.
    A reflection would preserve the cardinals above but silently mirror
    everything between them."""
    for a in range(0, 350, 10):
        d = our_aspect_to_pvgis(a + 10) - our_aspect_to_pvgis(a)
        d = (d + 180) % 360 - 180          # shortest way round
        assert abs(d - 10.0) < 1e-9, f"{a} -> {a+10} moved {d}, not 10"


# --------------------------------------------------------------------------
# Horizon encoding: 72 bins, uint8 = elevation degrees x 2, base64.
# This is baked onto every building and decoded by BOTH the frontend chart
# and the model's beam masking. If the two ever disagreed about the sky, the
# horizon tab would draw one thing and the economics would use another.
# --------------------------------------------------------------------------

def test_horizon_round_trip_preserves_half_degrees():
    profile = {i * AZ_STEP: (i % 40) * 0.5 for i in range(N_BINS)}
    back = decode_horizon(encode_horizon(profile))
    assert len(back) == N_BINS
    for az, v in profile.items():
        assert abs(back[az] - v) < 1e-9, f"bin {az}: {v} -> {back[az]}"


def test_horizon_quantises_to_the_nearest_half_degree():
    """0.5 deg is the storage resolution; a value between steps must land on
    the nearer one, not truncate downward."""
    back = decode_horizon(encode_horizon({0.0: 10.24, AZ_STEP: 10.26}))
    assert back[0.0] == 10.0
    assert back[AZ_STEP] == 10.5


def test_horizon_missing_bins_are_open_sky_not_dropped():
    """A sparse profile must encode as 72 bins with the gaps meaning 'no
    obstruction'. Dropping them would shift every later bin's azimuth."""
    back = decode_horizon(encode_horizon({0.0: 12.0, 180.0: 30.0}))
    assert len(back) == N_BINS
    assert back[0.0] == 12.0 and back[180.0] == 30.0
    assert back[90.0] == 0.0


def test_horizon_clamps_instead_of_wrapping():
    """uint8 caps at 127.5 deg. Wrapping would turn a bad value into a
    plausible small one -- a vertical wall becoming open sky."""
    back = decode_horizon(encode_horizon({0.0: 200.0, AZ_STEP: -5.0}))
    assert back[0.0] == 127.5
    assert back[AZ_STEP] == 0.0


# --------------------------------------------------------------------------
# Horizon lookup, called with the full 8760-hour array.
# --------------------------------------------------------------------------

def test_horizon_angle_interpolates_between_samples():
    profile = {0.0: 0.0, 90.0: 10.0, 180.0: 0.0, 270.0: 0.0}
    assert abs(horizon_angle_at(profile, 45.0) - 5.0) < 1e-9


def test_horizon_angle_wraps_past_north():
    """359 deg must interpolate across the 360/0 seam, not clamp to an end."""
    profile = {0.0: 10.0, 90.0: 0.0, 180.0: 0.0, 270.0: 10.0}
    assert horizon_angle_at(profile, 359.0) > 9.0
    assert abs(horizon_angle_at(profile, 720.0) - horizon_angle_at(profile, 0.0)) < 1e-9


def test_horizon_angle_accepts_arrays():
    """The hourly path passes all 8760 at once; scalar and array must agree."""
    profile = {i * AZ_STEP: float(i % 10) for i in range(N_BINS)}
    azs = np.array([0.0, 37.5, 123.0, 359.9])
    got = horizon_angle_at(profile, azs)
    assert isinstance(got, np.ndarray) and got.shape == azs.shape
    for i, a in enumerate(azs):
        assert abs(got[i] - horizon_angle_at(profile, float(a))) < 1e-9


# --------------------------------------------------------------------------
# Lookup-table binning.
# --------------------------------------------------------------------------

def test_nearest_bin_wraps_azimuth():
    assert _nearest_bin(359.0, 5.0) == 0        # 360 wraps to 0, not 360
    assert _nearest_bin(87.0, 5.0) == 85
    assert _nearest_bin(88.0, 5.0) == 90


def test_nearest_bin_clamps_slope():
    assert _nearest_bin(88.0, 5.0, max_value=60) == 60
    assert _nearest_bin(22.0, 5.0, max_value=60) == 20


# --------------------------------------------------------------------------
# Published assumptions. These are numbers the public reads.
# --------------------------------------------------------------------------

def test_dc_to_ac_matches_the_documented_derate():
    """Pins the published DC-to-AC factor. Updated 1 Sep from 0.97 x 0.81 when
    the derate was recalibrated 19% -> 15% for n-type panels; this test is what
    forced that change to be acknowledged rather than slipping through."""
    pv = config.PV_ASSUMPTIONS
    factor = (pv["inverter_efficiency_pct"] / 100.0) * (1 - pv["system_derate_pct"] / 100.0)
    assert abs(factor - 0.97 * 0.85) < 1e-9, factor


def test_derate_stays_in_a_defensible_band():
    """Cross-checked against PVGIS (~0.791) on 31 Aug. A change that puts this
    outside 0.72-0.85 is either a typo or a decision that needs the public
    assumptions text updated with it -- either way it should stop the build."""
    pv = config.PV_ASSUMPTIONS
    factor = (pv["inverter_efficiency_pct"] / 100.0) * (1 - pv["system_derate_pct"] / 100.0)
    assert 0.72 <= factor <= 0.85, f"DC->AC factor {factor:.3f} outside sane band"


def test_inverter_loss_is_not_double_counted():
    """PVGIS folds inverter losses into its single loss number; we keep them
    separate. The comparison in validate_against_pvgis only holds if the
    system derate excludes them -- see the config breakdown."""
    pv = config.PV_ASSUMPTIONS
    assert pv["inverter_efficiency_pct"] < 100.0
    assert pv["system_derate_pct"] < 30.0, "derate looks like it swallowed the inverter"


# --------------------------------------------------------------------------
# Export cleanup. This one deletes files, so its refusal cases matter more
# than its happy path: a bug here either fills the disk (what happened) or
# throws away the only copy of a half-fetched multi-GB download.
# --------------------------------------------------------------------------

def _export_fixture(tmp, mosaic_bytes=b"x" * 1000):
    from pathlib import Path
    d = Path(tmp)
    z = d / "imagery_export.zip"
    z.write_bytes(b"z" * 5000)
    ex = d / "imagery"
    ex.mkdir()
    (ex / "a.tif").write_bytes(b"t" * 3000)
    m = d / "imagery_mosaic.tif"
    if mosaic_bytes is not None:
        m.write_bytes(mosaic_bytes)
    return z, ex, m


def test_export_cleanup_removes_intermediates_but_keeps_the_mosaic():
    import tempfile
    from src.fetch_data import reclaim_export_intermediates as reclaim
    with tempfile.TemporaryDirectory() as tmp:
        z, ex, m = _export_fixture(tmp)
        freed = reclaim(z, ex, m)
        assert freed == 8000, freed
        assert not z.exists() and not ex.exists()
        assert m.exists(), "the mosaic must never be deleted"


def test_export_cleanup_keeps_everything_when_the_mosaic_is_missing():
    """A failed merge must leave the multi-GB download in place to retry from."""
    import tempfile
    from src.fetch_data import reclaim_export_intermediates as reclaim
    with tempfile.TemporaryDirectory() as tmp:
        z, ex, m = _export_fixture(tmp, mosaic_bytes=None)
        assert reclaim(z, ex, m) == 0
        assert z.exists() and ex.exists()


def test_export_cleanup_keeps_everything_when_the_mosaic_is_empty():
    """A zero-byte mosaic means the merge failed, even though the file exists."""
    import tempfile
    from src.fetch_data import reclaim_export_intermediates as reclaim
    with tempfile.TemporaryDirectory() as tmp:
        z, ex, m = _export_fixture(tmp, mosaic_bytes=b"")
        assert reclaim(z, ex, m) == 0
        assert z.exists() and ex.exists()


def test_export_cleanup_can_be_opted_out_for_debugging():
    import tempfile
    from src.fetch_data import reclaim_export_intermediates as reclaim
    with tempfile.TemporaryDirectory() as tmp:
        z, ex, m = _export_fixture(tmp)
        assert reclaim(z, ex, m, keep=True) == 0
        assert z.exists() and ex.exists()


def test_export_cleanup_never_raises():
    """Cleanup must not fail a fetch: a full disk is recoverable, a
    half-fetched region is not."""
    import tempfile
    from pathlib import Path
    from src.fetch_data import reclaim_export_intermediates as reclaim
    with tempfile.TemporaryDirectory() as tmp:
        _, _, m = _export_fixture(tmp)
        assert reclaim(Path("/nonexistent/x.zip"), Path("/nonexistent/d"), m) == 0


# --------------------------------------------------------------------------

def _main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  pass  {name}")
        except AssertionError as e:
            failed.append((name, str(e) or "assertion failed"))
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())

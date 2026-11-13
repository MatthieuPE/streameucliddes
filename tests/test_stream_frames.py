"""Tests of stream_frames.py, with gala great-circle frames and astropy frames."""

import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc
import numpy as np
import pytest

from streameucliddes.stream_frames import phi12_bin_centres, phi12_to_radec, phi12_window_around, radec_to_phi12

pytestmark = pytest.mark.stream_frames

STREAM_ENDS = ((42.0, -20.0), (58.0, -30.0))  # (ra, dec) of the ends of gc_frame, conftest.py


def _random_sky(rng, n):
    """n positions uniform on the sphere (deg)."""
    return rng.uniform(0.0, 360.0, n), np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))


def _angle_difference(a, b):
    """a - b, wrapped to [-180, 180) deg."""
    return (np.asarray(a) - b + 180.0) % 360.0 - 180.0


@pytest.fixture(scope="module")
def equatorial_frame():
    """gala great circle along the celestial equator: phi1 = ra, phi2 = dec."""
    north_pole = coord.SkyCoord(ra=0.0 * u.deg, dec=90.0 * u.deg)
    return gc.GreatCircleICRSFrame.from_pole_ra0(pole=north_pole, ra0=0.0 * u.deg)


# ===================================================
# Test radec_to_phi12 and phi12_to_radec
# ===================================================


# Along the celestial equator: phi1 = ra wrapped to [-180, 180), phi2 = dec
def test_equatorial_frame(equatorial_frame):
    phi1, phi2 = radec_to_phi12([0.0, 90.0, 200.0, 270.0], [10.0, -10.0, 45.0, 0.0], equatorial_frame)
    np.testing.assert_allclose(phi1, [0.0, 90.0, -160.0, -90.0], atol=1e-10)
    np.testing.assert_allclose(phi2, [10.0, -10.0, 45.0, 0.0], atol=1e-10)


# gala's frame from two endpoints (as in conftest.py): the endpoints lie on phi2 = 0,
# symmetric about phi1 = 0, the first at negative phi1
def test_endpoints_frame(gc_frame):
    (ra_1, dec_1), (ra_2, dec_2) = STREAM_ENDS
    phi1, phi2 = radec_to_phi12([ra_1, ra_2], [dec_1, dec_2], gc_frame)
    separation = coord.SkyCoord(ra_1 * u.deg, dec_1 * u.deg).separation(
        coord.SkyCoord(ra_2 * u.deg, dec_2 * u.deg)
    )
    np.testing.assert_allclose(phi1, [-separation.deg / 2, separation.deg / 2], atol=1e-10)
    np.testing.assert_allclose(phi2, 0.0, atol=1e-10)


# Any astropy frame works, whatever the names of its components: Galactic (l, b)
def test_other_astropy_frame(rng):
    ra, dec = _random_sky(rng, 100)
    l, b = radec_to_phi12(ra, dec, coord.Galactic())
    galactic = coord.SkyCoord(ra=ra * u.deg, dec=dec * u.deg).galactic
    np.testing.assert_allclose(_angle_difference(l, galactic.l.deg), 0.0, atol=1e-10)
    np.testing.assert_allclose(b, galactic.b.deg, atol=1e-10)


# phi12_to_radec inverts radec_to_phi12; phi1 in [-180, 180), ra in [0, 360)
def test_round_trip(gc_frame, rng):
    ra, dec = _random_sky(rng, 1000)
    phi1, phi2 = radec_to_phi12(ra, dec, gc_frame)
    assert np.all((phi1 >= -180.0) & (phi1 < 180.0))
    assert np.all(np.abs(phi2) <= 90.0)

    ra_back, dec_back = phi12_to_radec(phi1, phi2, gc_frame)
    assert np.all((ra_back >= 0.0) & (ra_back < 360.0))
    np.testing.assert_allclose(_angle_difference(ra_back, ra), 0.0, atol=1e-9)
    np.testing.assert_allclose(dec_back, dec, atol=1e-9)


# The full loop from the stream frame: (phi1, phi2) -> (ra, dec) -> (phi1, phi2) gives
# the same positions back, in each kind of frame, also at the edges phi1 = -180 and
# 180 (where phi1 wraps) and near the poles
@pytest.mark.parametrize("frame_name", ["gala from endpoints", "gala equatorial", "Galactic"])
def test_loop_from_phi12(frame_name, gc_frame, equatorial_frame, rng):
    frame = {
        "gala from endpoints": gc_frame,
        "gala equatorial": equatorial_frame,
        "Galactic": coord.Galactic(),
    }[frame_name]
    phi1, phi2 = _random_sky(rng, 1000)
    phi1 = np.concatenate([phi1 - 180.0, [-180.0, 179.999, 0.0, 0.0]])  # in [-180, 180)
    phi2 = np.concatenate([phi2, [0.0, 0.0, 89.0, -89.0]])

    ra, dec = phi12_to_radec(phi1, phi2, frame)
    phi1_back, phi2_back = radec_to_phi12(ra, dec, frame)
    assert np.all((phi1_back >= -180.0) & (phi1_back < 180.0))
    np.testing.assert_allclose(_angle_difference(phi1_back, phi1), 0.0, atol=1e-9)
    np.testing.assert_allclose(phi2_back, phi2, atol=1e-9)


# Scalars stay scalars; arrays are broadcast together and keep their shape
def test_shapes(gc_frame):
    phi1, phi2 = radec_to_phi12(10.0, 20.0, gc_frame)
    assert np.shape(phi1) == np.shape(phi2) == ()
    ra, dec = phi12_to_radec(np.zeros((4, 5)), 1.0, gc_frame)
    assert ra.shape == dec.shape == (4, 5)


# ===================================================
# Test phi12_bin_centres
# ===================================================


# Bins of bin_size_deg from the minimum of the window, covering it
def test_phi12_bin_centres():
    phi1, phi2 = phi12_bin_centres((10.0, 30.0, -5.0, 5.0), 1.0)
    np.testing.assert_allclose(phi1, np.linspace(10.5, 29.5, 20))
    np.testing.assert_allclose(phi2, np.linspace(-4.5, 4.5, 10))


# A bin size that does not divide the window: the bins whose centre is inside.
# A bin size that divides it: no extra bin from rounding (0.1 is not exact in binary)
@pytest.mark.parametrize(
    "window, n_phi1, n_phi2",
    [
        ((0.0, 1.04, 0.0, 1.06), 10, 11),
        ((0.0, 1.0, -0.15, 0.15), 10, 3),
        ((-20.0, 20.0, -5.0, 5.0), 400, 100),
    ],
)
def test_phi12_bin_centres_number_of_bins(window, n_phi1, n_phi2):
    phi1, phi2 = phi12_bin_centres(window, 0.1)
    assert (phi1.size, phi2.size) == (n_phi1, n_phi2)
    np.testing.assert_allclose(np.diff(phi1), 0.1)
    assert phi1[0] == pytest.approx(window[0] + 0.05)


# A window narrower than half a bin has no bin centre inside: an error
def test_phi12_bin_centres_narrow_window():
    with pytest.raises(ValueError, match="narrower than half a bin"):
        phi12_bin_centres((0.0, 0.04, 0.0, 1.0), 0.1)


# Test phi12_window_around


def test_window_around_points():
    """phi1: extent + padding; phi2: centred, extent + padding."""
    window = phi12_window_around([-3.0, 1.0, 4.0], [0.5, 1.5, 9.5], padding_phi1_deg=2.0, padding_phi2_deg=1.0,
                                 min_phi2_half_width_deg=0.5)
    assert window == pytest.approx((-5.0, 6.0, -0.5, 10.5))


def test_window_around_minimum_phi2_half_width():
    """Points on phi2 = 0 (two endpoints): phi2 half-width min_phi2_half_width_deg."""
    window = phi12_window_around([-3.0, 3.0], [0.0, 0.0], padding_phi1_deg=5.0, padding_phi2_deg=2.0,
                                 min_phi2_half_width_deg=4.0)
    assert window == pytest.approx((-8.0, 8.0, -4.0, 4.0))


# Stubs of the candidate investigation (docs/dev/candidate_investigation_plan.md)


@pytest.mark.skip(reason="stub")
def test_in_phi12_window():
    """Points inside, on the edges (kept) and outside a window."""


@pytest.mark.skip(reason="stub")
def test_window_outline_closed():
    """phi12_window_outline: closed, on the window edges."""


@pytest.mark.skip(reason="stub")
def test_radec_box_contains_window():
    """phi12_window_radec_box contains the window (round trip with phi12_to_radec)."""


@pytest.mark.skip(reason="stub")
def test_radec_box_ra_wrap_warns():
    """A window across ra = 0: ra_min > ra_max and a warning."""


@pytest.mark.skip(reason="stub")
def test_endpoints_on_phi2_zero():
    """stream_frame_from_endpoints: both endpoints at phi2 = 0 (1e-8 deg)."""


@pytest.mark.skip(reason="stub")
def test_endpoints_phi1_symmetric():
    """Endpoints at phi1 symmetric around 0, endpoint_1 negative."""

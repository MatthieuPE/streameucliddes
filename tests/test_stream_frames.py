"""Tests of stream_frames.py, against astropy's transforms and gala's great-circle frame."""

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

from streameucliddes.stream_frames import (
    phi12_bin_centres,
    phi12_to_radec,
    radec_to_phi12,
    rotation_matrix_from_astropy_frame,
    rotation_matrix_from_endpoints,
)

pytestmark = pytest.mark.stream_frames

STREAM_ENDS = (42.0, -20.0, 58.0, -30.0)  # (ra_1, dec_1, ra_2, dec_2) of gc_frame, conftest.py


def _random_sky(rng, n):
    """n positions uniform on the sphere (deg)."""
    return rng.uniform(0.0, 360.0, n), np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))


def _angle_difference(a, b):
    """a - b, wrapped to [-180, 180) deg."""
    return (np.asarray(a) - b + 180.0) % 360.0 - 180.0


def _assert_rotation(rotation_matrix):
    np.testing.assert_allclose(rotation_matrix @ rotation_matrix.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation_matrix) == pytest.approx(1.0)


# ===================================================
# Test rotation_matrix_from_astropy_frame
# ===================================================


# phi1/phi2 through the rotation matrix are those of astropy's transform to the frame:
# a gala great circle (as the galstreams frames), and the Galactic frame
@pytest.mark.parametrize("frame_name", ["gala great circle", "Galactic"])
def test_rotation_matrix_from_astropy_frame(frame_name, gc_frame, rng):
    frame = gc_frame if frame_name == "gala great circle" else coord.Galactic()
    rotation_matrix = rotation_matrix_from_astropy_frame(frame)
    _assert_rotation(rotation_matrix)

    ra, dec = _random_sky(rng, 1000)
    phi1, phi2 = radec_to_phi12(ra, dec, rotation_matrix)
    expected = (
        coord.SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        .transform_to(frame)
        .represent_as(coord.UnitSphericalRepresentation)
    )
    np.testing.assert_allclose(_angle_difference(phi1, expected.lon.deg), 0.0, atol=1e-8)
    np.testing.assert_allclose(phi2, expected.lat.deg, atol=1e-8)


# ===================================================
# Test rotation_matrix_from_endpoints
# ===================================================


# The two points lie on phi2 = 0, symmetric about phi1 = 0, the first at negative phi1
def test_rotation_matrix_from_endpoints():
    rotation_matrix = rotation_matrix_from_endpoints(*STREAM_ENDS)
    _assert_rotation(rotation_matrix)

    ra_1, dec_1, ra_2, dec_2 = STREAM_ENDS
    phi1, phi2 = radec_to_phi12([ra_1, ra_2], [dec_1, dec_2], rotation_matrix)
    separation = coord.SkyCoord(ra_1 * u.deg, dec_1 * u.deg).separation(
        coord.SkyCoord(ra_2 * u.deg, dec_2 * u.deg)
    )
    np.testing.assert_allclose(phi1, [-separation.deg / 2, separation.deg / 2], atol=1e-10)
    np.testing.assert_allclose(phi2, 0.0, atol=1e-10)


# Same frame as gala's great circle through the same two points
def test_rotation_matrix_from_endpoints_matches_gala(gc_frame):
    np.testing.assert_allclose(
        rotation_matrix_from_endpoints(*STREAM_ENDS),
        rotation_matrix_from_astropy_frame(gc_frame),
        atol=1e-12,
    )


# The same point twice, or two antipodal points, define no unique great circle
@pytest.mark.parametrize("points", [(10.0, 20.0, 10.0, 20.0), (10.0, 20.0, 190.0, -20.0)])
def test_rotation_matrix_from_endpoints_degenerate(points):
    with pytest.raises(ValueError, match="no unique great circle"):
        rotation_matrix_from_endpoints(*points)


# ===================================================
# Test radec_to_phi12 and phi12_to_radec
# ===================================================


# phi12_to_radec inverts radec_to_phi12; phi1 in [-180, 180), ra in [0, 360)
def test_round_trip(rng):
    rotation_matrix = rotation_matrix_from_endpoints(*STREAM_ENDS)
    ra, dec = _random_sky(rng, 1000)
    phi1, phi2 = radec_to_phi12(ra, dec, rotation_matrix)
    assert np.all((phi1 >= -180.0) & (phi1 < 180.0))
    assert np.all(np.abs(phi2) <= 90.0)

    ra_back, dec_back = phi12_to_radec(phi1, phi2, rotation_matrix)
    assert np.all((ra_back >= 0.0) & (ra_back < 360.0))
    np.testing.assert_allclose(_angle_difference(ra_back, ra), 0.0, atol=1e-9)
    np.testing.assert_allclose(dec_back, dec, atol=1e-9)


# The identity matrix is ICRS: phi1 = ra wrapped to [-180, 180), phi2 = dec
def test_identity_frame():
    phi1, phi2 = radec_to_phi12([0.0, 90.0, 200.0, 270.0], [10.0, -10.0, 45.0, 0.0], np.eye(3))
    np.testing.assert_allclose(phi1, [0.0, 90.0, -160.0, -90.0], atol=1e-12)
    np.testing.assert_allclose(phi2, [10.0, -10.0, 45.0, 0.0], atol=1e-12)


# Scalars stay scalars; arrays are broadcast together and keep their shape
def test_shapes():
    phi1, phi2 = radec_to_phi12(10.0, 20.0, np.eye(3))
    assert np.shape(phi1) == np.shape(phi2) == ()
    ra, dec = phi12_to_radec(np.zeros((4, 5)), 1.0, np.eye(3))
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

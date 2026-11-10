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


# Phase 2 stubs: the module imports, with its functions
def test_import():
    stubs = (
        rotation_matrix_from_astropy_frame,
        rotation_matrix_from_endpoints,
        radec_to_phi12,
        phi12_to_radec,
    )
    assert all(callable(stub) for stub in stubs)


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

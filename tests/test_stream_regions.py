"""Tests of analysis/stream_regions.py, on synthetic stars."""

import pytest

pytestmark = pytest.mark.stream_regions


@pytest.mark.skip(reason="stub")
def test_stream_regions_keys():
    """stream_regions returns the plain dict of its inputs."""


@pytest.mark.skip(reason="stub")
def test_stream_regions_rejects_overlap():
    """off_n_sigma[0] < on_n_sigma, or off_n_sigma not increasing, or sigma <= 0: ValueError."""


@pytest.mark.skip(reason="stub")
def test_on_off_masks_disjoint():
    """No star is both on and off; both need phi1 in phi1_range."""


@pytest.mark.skip(reason="stub")
def test_on_off_masks_limits():
    """Stars just inside / outside the on and off limits are kept / left out."""


@pytest.mark.skip(reason="stub")
def test_unmasked_area_matches_analytic():
    """No mask: Delta phi1 (sin phi2_max - sin phi2_min) (180/pi)^2 within 1 %."""


@pytest.mark.skip(reason="stub")
def test_unmasked_area_half_masked():
    """Half of the sky excluded: about half the area."""


@pytest.mark.skip(reason="stub")
def test_region_area_matches_analytic():
    """No mask: on and off areas of the regions within 1 % of the analytic ones."""


@pytest.mark.skip(reason="stub")
def test_phi2_density_profile_flat_with_half_masked():
    """Uniform stars, half the window excluded: flat density, consistent with the true one."""


@pytest.mark.skip(reason="stub")
def test_fit_recovers_centre_and_sigma():
    """Uniform stars + Gaussian stream of known centre and sigma: recovered within 3 errors."""


@pytest.mark.skip(reason="stub")
def test_phi1_density_profile_zero_without_stream():
    """Uniform stars: on - off density consistent with 0 in every phi1 bin."""


@pytest.mark.skip(reason="stub")
def test_phi1_density_profile_shows_stream_extent():
    """Stream over part of phi1: positive difference there only."""


@pytest.mark.skip(reason="stub")
def test_stream_track_phi12_constant():
    """Track at phi2 = phi2_centre over phi1_range."""


@pytest.mark.skip(reason="stub")
def test_stream_track_radec_round_trip():
    """stream_track_radec back to phi1/phi2 gives stream_track_phi12."""

"""Tests of analysis/candidate_investigation.py, on a synthetic catalog
(uniform stars + a Gaussian stream along the arc between the endpoints)."""

import pytest

pytestmark = pytest.mark.candidate_investigation


@pytest.mark.skip(reason="stub")
def test_stream_regions_keys():
    """stream_regions returns the plain dict of its inputs."""


@pytest.mark.skip(reason="stub")
def test_stream_regions_rejects_overlap():
    """off_n_sigma[0] < on_n_sigma, or off_n_sigma not increasing, or sigma <= 0: ValueError."""


@pytest.mark.skip(reason="stub")
def test_region_windows():
    """On window |phi2 - c| <= on_n_sigma sigma; off windows on each side; none overlap."""


@pytest.mark.skip(reason="stub")
def test_region_area_matches_analytic():
    """No mask: on and off areas within 1 % of the analytic ones."""


@pytest.mark.skip(reason="stub")
def test_phi2_profile_flat_with_half_masked():
    """Uniform stars, half the window excluded: flat phi2 density, consistent with the true one."""


@pytest.mark.skip(reason="stub")
def test_profile_over_two_windows():
    """Two windows: counts and areas add up."""


@pytest.mark.skip(reason="stub")
def test_fit_recovers_centre_and_sigma():
    """Uniform stars + Gaussian stream of known centre and sigma: recovered within 3 errors."""


@pytest.mark.skip(reason="stub")
def test_fit_rejects_phi1_profile():
    """A profile along phi1: ValueError."""


@pytest.mark.skip(reason="stub")
def test_on_off_phi1_profile_zero_without_stream():
    """Uniform stars: on - off consistent with 0 in every phi1 bin."""


@pytest.mark.skip(reason="stub")
def test_on_off_phi1_profile_shows_stream_extent():
    """Stream over part of phi1: positive difference there only."""


@pytest.mark.skip(reason="stub")
def test_track_phi12_straight():
    """track_phi12: phi2 = phi2_centre over phi1_range; phi2 = 0 between the endpoints before define_regions."""


@pytest.mark.skip(reason="stub")
def test_post_init_frame_window_box():
    """Construction sets the frame, a window containing the endpoints and an (ra, dec) box."""


@pytest.mark.skip(reason="stub")
def test_post_init_metallicity_or_feh():
    """Neither or both of metallicity and feh: ValueError."""


@pytest.mark.skip(reason="stub")
def test_steps_out_of_order_raise():
    """Each step before the previous one raises a ValueError naming the missing step (e.g. compute_hess before define_regions)."""


@pytest.mark.skip(reason="stub")
def test_fit_width_without_distance_modulus_raises():
    """fit_width without distance_modulus: ValueError."""


@pytest.mark.skip(reason="stub")
def test_set_endpoints_clears_later_results():
    """set_endpoints: new frame and window, later results cleared, sky_map kept."""


@pytest.mark.skip(reason="stub")
def test_set_endpoints_outside_catalog_raises():
    """A window leaving the (ra, dec) range of the catalog: ValueError saying to load a larger box."""


@pytest.mark.skip(reason="stub")
def test_run_all_steps_recovers_stream():
    """run_all_steps on the synthetic catalog: sigma and phi2_centre recovered; on - off excess positive."""


@pytest.mark.skip(reason="stub")
def test_summary_keys():
    """summary keys; None for the steps not done."""


@pytest.mark.skip(reason="stub")
def test_catalog_not_copied():
    """The kept catalog is the object given (no copy)."""

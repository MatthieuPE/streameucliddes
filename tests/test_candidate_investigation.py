"""Tests of analysis/candidate_investigation.py, on a synthetic catalog
(uniform stars + a Gaussian stream along the arc between the endpoints)."""

import pytest

pytestmark = pytest.mark.candidate_investigation


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

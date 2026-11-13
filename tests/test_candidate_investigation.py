"""Tests of analysis/candidate_investigation.py, on a synthetic catalog
(uniform stars + a Gaussian stream along the arc between the endpoints)."""

import numpy as np
import pytest

from streameucliddes.analysis.candidate_investigation import CandidateInvestigation
from streameucliddes.catalogs.isochrone_parameters import DEFAULT_METALLICITIES
from streameucliddes.stream_frames import phi12_to_radec, radec_to_phi12
from streameucliddes.unit_conversions import convert_FeH_to_z

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


ENDPOINTS = ((77.0, -40.0), (70.0, -43.0))  # (ra, dec) deg, the Euclid DR1 candidate of the example notebook


# Construction: frame through the endpoints, window around them (padding given), box containing the window
def test_post_init_frame_window_box(rng):
    cand = CandidateInvestigation("test", *ENDPOINTS, padding_phi1_deg=2.0, padding_phi2_deg=1.0,
                                  min_phi2_half_width_deg=3.0)
    phi1, phi2 = radec_to_phi12([ENDPOINTS[0][0], ENDPOINTS[1][0]], [ENDPOINTS[0][1], ENDPOINTS[1][1]],
                                cand.stream_frame)
    np.testing.assert_allclose(phi2, 0.0, atol=1e-8)
    assert cand.window == pytest.approx((phi1[0] - 2.0, phi1[1] + 2.0, -3.0, 3.0))
    (ra_min, ra_max), (dec_min, dec_max) = cand.radec_box
    ra, dec = phi12_to_radec(rng.uniform(*cand.window[:2], 10_000), rng.uniform(*cand.window[2:], 10_000),
                             cand.stream_frame)
    assert ((ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max)).all()
    assert cand.endpoint_1 == ENDPOINTS[0] and all(isinstance(x, float) for x in cand.endpoint_1)


# Isochrone: Z given, Z from [Fe/H], the default Z; both given: ValueError, also when set later
def test_isochrone_from_metallicity_or_feh():
    assert CandidateInvestigation("test", *ENDPOINTS, age=12.0, metallicity=0.001).isochrone == {
        "age": 12.0, "metallicity": 0.001}
    assert CandidateInvestigation("test", *ENDPOINTS, feh=-1.5).isochrone["metallicity"] == pytest.approx(
        convert_FeH_to_z(-1.5))
    assert CandidateInvestigation("test", *ENDPOINTS).isochrone["metallicity"] == DEFAULT_METALLICITIES[0]
    with pytest.raises(ValueError, match="not both"):
        CandidateInvestigation("test", *ENDPOINTS, metallicity=0.001, feh=-1.5)
    cand = CandidateInvestigation("test", *ENDPOINTS, feh=-1.5)
    cand.metallicity = 0.001
    with pytest.raises(ValueError, match="not both"):
        cand.isochrone


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

"""Tests of analysis/candidate_investigation.py, on a synthetic catalog
(uniform stars + a Gaussian stream along the arc between the endpoints)."""

import numpy as np
import pandas as pd
import pytest

from streameucliddes.analysis.candidate_investigation import CandidateInvestigation
from streameucliddes.analysis.generic_stream_search import generic_search_single_dm
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


NSIDE = 64  # coarse maps: fast tests
NO_OBJECT_MASKS = {"mask_lmc": False, "mask_smc": False, "mask_dwarfs": False, "mask_gc": False, "mask_galaxies": False}


@pytest.fixture(scope="module")
def catalog():
    """Uniform stars in a box around the endpoints, with the columns of the default match filter."""
    rng = np.random.default_rng(3)
    n = 50_000
    vis = rng.uniform(18.0, 26.0, n)
    return pd.DataFrame({
        "ra": rng.uniform(58.0, 90.0, n),
        "dec": np.degrees(np.arcsin(rng.uniform(*np.sin(np.radians([-52.0, -30.0])), n))),
        "euclid_q1_VIS_obs": vis, "euclid_q1_H_obs": vis - rng.uniform(-1.0, 1.5, n),
        "euclid_q1_VIS_err": np.full(n, 0.02), "euclid_q1_H_err": np.full(n, 0.02),
    })


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


# ===================================================
# Test compute_sky_map
# ===================================================


# The map of generic_search_single_dm with the isochrone and match filter of the object; the
# catalog kept, not copied; a given distance modulus over the field
def test_compute_sky_map(catalog):
    kwargs = {"nside": NSIDE, "smoothing_deg": 1.0, **NO_OBJECT_MASKS}
    cand = CandidateInvestigation("test", *ENDPOINTS, feh=-1.5, distance_modulus=17.0)
    assert cand.compute_sky_map(catalog, **kwargs) is cand
    expected = generic_search_single_dm(catalog, 17.0, cand.isochrone, **kwargs)
    for key in ("density", "residual"):
        np.testing.assert_array_equal(np.ma.filled(cand.sky_map[key], np.nan), np.ma.filled(expected[key], np.nan))
    assert cand.sky_map["isochrone"] == cand.isochrone and cand.sky_map["n_selected"] == expected["n_selected"]
    assert cand.catalog is catalog
    cand.compute_sky_map(catalog, 18.0, **kwargs)
    assert cand.sky_map["distance_modulus"] == 18.0 and cand.distance_modulus == 17.0


# No distance modulus given nor set: ValueError
def test_compute_sky_map_needs_distance_modulus(catalog):
    with pytest.raises(ValueError, match="distance_modulus"):
        CandidateInvestigation("test", *ENDPOINTS).compute_sky_map(catalog, nside=NSIDE, **NO_OBJECT_MASKS)

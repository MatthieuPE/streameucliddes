"""Tests of analysis/candidate_investigation.py, on a synthetic catalog
(conftest.py: uniform stars around the endpoints of the example candidate)."""

import healpy as hp
import numpy as np
import pytest

from streameucliddes.analysis import stream_frame_search as stream_frame_search_module
from streameucliddes.analysis.build_masks import object_mask_nside
from streameucliddes.analysis.candidate_investigation import DEFAULT_FOOTPRINT_NSIDE, CandidateInvestigation
from streameucliddes.analysis.generic_stream_search import generic_search_single_dm
from streameucliddes.analysis.stream_frame_search import prepare_search, stream_frame_search_single_dm
from streameucliddes.catalogs.isochrone_parameters import DEFAULT_METALLICITIES
from streameucliddes.catalogs.survey_footprint import convert_footprint_healpix_mask, survey_footprint_healpix_mask
from streameucliddes.stream_frames import phi12_to_radec, radec_to_phi12
from streameucliddes.unit_conversions import convert_FeH_to_z

pytestmark = pytest.mark.candidate_investigation

NSIDE = 64  # coarse sky maps: fast tests
BIN = 0.2  # bins of the maps in the frame (deg)


# ===================================================
# Regions, profiles and width
# ===================================================


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


# ===================================================
# Construction: frame, window, (ra, dec) box, isochrone
# ===================================================


# Frame through the endpoints, window around them (padding given), box containing the window
def test_post_init_frame_window_box(rng, candidate_endpoints):
    end_1, end_2 = candidate_endpoints
    cand = CandidateInvestigation("test", end_1, end_2, padding_phi1_deg=2.0, padding_phi2_deg=1.0,
                                  min_phi2_half_width_deg=3.0)
    phi1, phi2 = radec_to_phi12([end_1[0], end_2[0]], [end_1[1], end_2[1]], cand.stream_frame)
    np.testing.assert_allclose(phi2, 0.0, atol=1e-8)
    assert cand.window == pytest.approx((phi1[0] - 2.0, phi1[1] + 2.0, -3.0, 3.0))
    (ra_min, ra_max), (dec_min, dec_max) = cand.radec_box
    ra, dec = phi12_to_radec(rng.uniform(*cand.window[:2], 10_000), rng.uniform(*cand.window[2:], 10_000),
                             cand.stream_frame)
    assert ((ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max)).all()
    assert cand.endpoint_1 == end_1 and all(isinstance(x, float) for x in cand.endpoint_1)


# Isochrone: Z given, Z from [Fe/H], the default Z; both given: ValueError, also when set later
def test_isochrone_from_metallicity_or_feh(candidate_endpoints):
    assert CandidateInvestigation("test", *candidate_endpoints, age=12.0, metallicity=0.001).isochrone == {
        "age": 12.0, "metallicity": 0.001}
    assert CandidateInvestigation("test", *candidate_endpoints, feh=-1.5).isochrone["metallicity"] == pytest.approx(
        convert_FeH_to_z(-1.5))
    assert CandidateInvestigation("test", *candidate_endpoints).isochrone["metallicity"] == DEFAULT_METALLICITIES[0]
    with pytest.raises(ValueError, match="not both"):
        CandidateInvestigation("test", *candidate_endpoints, metallicity=0.001, feh=-1.5)
    cand = CandidateInvestigation("test", *candidate_endpoints, feh=-1.5)
    cand.metallicity = 0.001
    with pytest.raises(ValueError, match="not both"):
        cand.isochrone


# ===================================================
# compute_sky_map
# ===================================================


# The map of generic_search_single_dm with the isochrone and match filter of the object; the
# catalog kept, not copied; a given distance modulus over the field
def test_compute_sky_map(candidate_endpoints, candidate_catalog, no_object_masks):
    kwargs = {"nside": NSIDE, "smoothing_deg": 1.0, **no_object_masks}
    cand = CandidateInvestigation("test", *candidate_endpoints, feh=-1.5, distance_modulus=17.0)
    assert cand.compute_sky_map(candidate_catalog, **kwargs) is cand
    expected = generic_search_single_dm(candidate_catalog, 17.0, cand.isochrone, **kwargs)
    for key in ("density", "residual"):
        np.testing.assert_array_equal(np.ma.filled(cand.sky_map[key], np.nan), np.ma.filled(expected[key], np.nan))
    assert cand.sky_map["isochrone"] == cand.isochrone and cand.sky_map["n_selected"] == expected["n_selected"]
    assert cand.catalog is candidate_catalog
    cand.compute_sky_map(candidate_catalog, 18.0, **kwargs)
    assert cand.sky_map["distance_modulus"] == 18.0 and cand.distance_modulus == 17.0


# No distance modulus given nor set: ValueError
def test_compute_sky_map_needs_distance_modulus(candidate_endpoints, candidate_catalog, no_object_masks):
    with pytest.raises(ValueError, match="distance_modulus"):
        CandidateInvestigation("test", *candidate_endpoints).compute_sky_map(
            candidate_catalog, nside=NSIDE, **no_object_masks)


# ===================================================
# prepare, compute_zoom_map, set_endpoints
# ===================================================


# prepare_search with the frame, window, isochrone, distance moduli and prepare_kwargs of the object, and
# the footprint of the stars; no object masked: the excluded mask is that footprint, RING
def test_prepare(candidate_endpoints, candidate_catalog, no_object_masks):
    prepare_kwargs = {"bin_size_deg": BIN, **no_object_masks}
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_moduli=[16.0, 17.0],
                                  prepare_kwargs=prepare_kwargs)
    assert cand.prepare(candidate_catalog) is cand
    footprint = survey_footprint_healpix_mask(candidate_catalog["ra"], candidate_catalog["dec"],
                                              DEFAULT_FOOTPRINT_NSIDE, nest=False)
    expected = prepare_search(candidate_catalog, cand.stream_frame, cand.window, isochrone=cand.isochrone,
                              distance_moduli=[16.0, 17.0], footprint_mask=footprint, nest=False, **prepare_kwargs)
    assert cand.search.stars.index.equals(expected.stars.index)
    np.testing.assert_array_equal(cand.search.footprint_mask, expected.footprint_mask)
    assert cand.search.distance_moduli == [16.0, 17.0] and cand.search.isochrones == [cand.isochrone]
    np.testing.assert_array_equal(cand.excluded_healpix_mask, footprint)
    assert cand.catalog is candidate_catalog


# Objects masked (the empty mock catalogs of conftest.py): the excluded mask is at the object nside, the
# footprint brought to it OR the object mask of the search
def test_prepare_excluded_mask_with_objects(candidate_endpoints, candidate_catalog):
    cand = CandidateInvestigation("test", *candidate_endpoints, prepare_kwargs={"bin_size_deg": BIN})
    cand.prepare(candidate_catalog)
    nside = object_mask_nside(BIN)
    assert hp.npix2nside(cand.excluded_healpix_mask.size) == nside
    expected = convert_footprint_healpix_mask(cand.footprint_mask, nside, nest=False, nest_out=False)
    np.testing.assert_array_equal(cand.excluded_healpix_mask, expected | cand.search.object_healpix_mask)


# No catalog given nor kept, or a window beyond the stars: ValueError
def test_prepare_errors(candidate_endpoints, candidate_catalog):
    with pytest.raises(ValueError, match="compute_sky_map"):
        CandidateInvestigation("test", *candidate_endpoints).prepare()
    east = candidate_catalog[candidate_catalog["ra"] > 75.0]
    with pytest.raises(ValueError, match="load a larger box"):
        CandidateInvestigation("test", *candidate_endpoints).prepare(east)


# The map of stream_frame_search_single_dm on the prepared stars, at the distance modulus of the sky
# map, or the one given; before prepare: ValueError
def test_compute_zoom_map(candidate_with_maps, candidate_endpoints):
    cand = candidate_with_maps
    assert cand.zoom_map["distance_modulus"] == cand.sky_map["distance_modulus"]
    expected = stream_frame_search_single_dm(search=cand.search, distance_modulus=cand.sky_map["distance_modulus"],
                                             isochrone=cand.isochrone)
    np.testing.assert_array_equal(np.ma.filled(cand.zoom_map["density"], np.nan),
                                  np.ma.filled(expected["density"], np.nan))
    assert cand.compute_zoom_map(18.0).zoom_map["distance_modulus"] == 18.0
    with pytest.raises(ValueError, match="prepare"):
        CandidateInvestigation("test", *candidate_endpoints).compute_zoom_map(17.0)


# New endpoints: new frame, window, prepared stars; the sky map and the footprint kept; the later
# results cleared; window arguments set; another argument: TypeError
def test_set_endpoints(candidate_with_maps):
    cand = candidate_with_maps
    sky_map, footprint, old_window = cand.sky_map, cand.footprint_mask, cand.window
    new_ends = ((76.0, -41.0), (69.0, -44.0))
    assert cand.set_endpoints(*new_ends, padding_phi1_deg=3.0) is cand
    assert cand.endpoint_1 == new_ends[0] and cand.padding_phi1_deg == 3.0
    phi1, phi2 = radec_to_phi12([new_ends[0][0], new_ends[1][0]], [new_ends[0][1], new_ends[1][1]], cand.stream_frame)
    np.testing.assert_allclose(phi2, 0.0, atol=1e-8)
    assert cand.window[0] == pytest.approx(phi1[0] - 3.0) and cand.window != old_window
    assert cand.search.window == cand.window and cand.search.stream_frame is cand.stream_frame
    assert cand.sky_map is sky_map and cand.footprint_mask is footprint
    assert cand.zoom_map is None
    with pytest.raises(TypeError, match="set_endpoints takes"):
        cand.set_endpoints(*new_ends, bin_size_deg=0.1)


# set_endpoints keeps the object mask of the last prepare: not built again (the slow part)
def test_set_endpoints_keeps_object_mask(candidate_endpoints, candidate_catalog, monkeypatch):
    cand = CandidateInvestigation("test", *candidate_endpoints, prepare_kwargs={"bin_size_deg": BIN})
    cand.prepare(candidate_catalog)
    object_mask = cand.search.object_healpix_mask

    def not_again(*args, **kwargs):
        raise AssertionError("the object mask is built again")

    monkeypatch.setattr(stream_frame_search_module, "overlapped_objects_healpix_mask", not_again)
    cand.set_endpoints((76.0, -41.0), (69.0, -44.0))
    assert cand.search.object_healpix_mask is object_mask and cand.search.window == cand.window


# New endpoints beyond the stars: ValueError, the object left as it was
def test_set_endpoints_outside_catalog(candidate_with_maps):
    cand = candidate_with_maps
    before = (cand.endpoint_1, cand.endpoint_2, cand.stream_frame, cand.window, cand.search, cand.zoom_map)
    with pytest.raises(ValueError, match="load a larger box"):
        cand.set_endpoints((30.0, -40.0), (25.0, -43.0))
    after = (cand.endpoint_1, cand.endpoint_2, cand.stream_frame, cand.window, cand.search, cand.zoom_map)
    assert all(a is b for a, b in zip(before, after))


# Without a catalog: only the frame, window and box
def test_set_endpoints_without_catalog(candidate_endpoints):
    cand = CandidateInvestigation("test", *candidate_endpoints)
    cand.set_endpoints((76.0, -41.0), (69.0, -44.0))
    assert cand.search is None and cand.window != CandidateInvestigation("test", *candidate_endpoints).window


# ===================================================
# Later steps
# ===================================================


@pytest.mark.skip(reason="stub")
def test_steps_out_of_order_raise():
    """Each step before the previous one raises a ValueError naming the missing step (e.g. compute_hess before define_regions)."""


@pytest.mark.skip(reason="stub")
def test_fit_width_without_distance_modulus_raises():
    """fit_width without distance_modulus: ValueError."""


@pytest.mark.skip(reason="stub")
def test_run_all_steps_recovers_stream():
    """run_all_steps on the synthetic catalog: sigma and phi2_centre recovered; on - off excess positive."""


@pytest.mark.skip(reason="stub")
def test_summary_keys():
    """summary keys; None for the steps not done."""

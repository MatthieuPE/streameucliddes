"""Tests of analysis/candidate_investigation.py, on a synthetic catalog
(conftest.py: uniform stars around the endpoints of the example candidate)."""

import healpy as hp
import numpy as np
import pytest

from streameucliddes.analysis import stream_frame_search as stream_frame_search_module
from streameucliddes.analysis.build_masks import object_mask_nside
from streameucliddes.analysis.candidate_investigation import (
    DEFAULT_FIT_HALF_WIDTH_DEG,
    DEFAULT_FOOTPRINT_NSIDE,
    CandidateInvestigation,
    fit_stream_width,
    on_off_phi1_profile,
    phi12_density_map,
    phi12_density_profile,
    region_area_deg2,
    region_windows,
    stream_regions,
    stream_width_model,
)
from streameucliddes.analysis.generic_stream_search import generic_search_single_dm
from streameucliddes.analysis.stream_frame_search import prepare_search, stream_frame_search_single_dm
from streameucliddes.catalogs.isochrone_parameters import DEFAULT_METALLICITIES
from streameucliddes.catalogs.survey_footprint import (
    convert_footprint_healpix_mask,
    survey_footprint_healpix_mask,
    unmasked_area_deg2,
)
from streameucliddes.stream_frames import phi12_to_radec, radec_to_phi12, stream_frame_from_endpoints
from streameucliddes.unit_conversions import convert_FeH_to_z

pytestmark = pytest.mark.candidate_investigation

NSIDE = 64  # coarse sky maps: fast tests
BIN = 0.2  # bins of the maps in the frame (deg)


# ===================================================
# Regions, profiles and width
# ===================================================


# The plain dict of the inputs, as floats
def test_stream_regions_keys():
    regions = stream_regions((-3, 2), 0.2, 0.3, on_n_sigma=1.5, off_n_sigma=(2.5, 4.0))
    assert regions == {"phi1_range": (-3.0, 2.0), "phi2_centre": 0.2, "sigma": 0.3, "on_n_sigma": 1.5, "off_n_sigma": (2.5, 4.0)}


# Off bands overlapping the on region, off_n_sigma not increasing, sigma <= 0, phi1_range not increasing
@pytest.mark.parametrize("kwargs", [{"on_n_sigma": 2.6}, {"off_n_sigma": (3.0, 2.5)}, {"sigma": 0.0},
                                    {"phi1_range": (2.0, -3.0)}])
def test_stream_regions_rejects_overlap(kwargs):
    arguments = {"phi1_range": (-3.0, 2.0), "phi2_centre": 0.0, "sigma": 0.3, **kwargs}
    with pytest.raises(ValueError):
        stream_regions(**arguments)


# On window |phi2 - c| <= on_n_sigma sigma, off windows on each side; none overlap
def test_region_windows():
    windows = region_windows({"phi1_range": (-3.0, 2.0), "phi2_centre": 0.2, "sigma": 0.3, "on_n_sigma": 1.5, "off_n_sigma": (2.5, 4.0)})
    assert windows["on"] == pytest.approx((-3.0, 2.0, 0.2 - 0.45, 0.2 + 0.45))
    below, above = windows["off"]
    assert below == pytest.approx((-3.0, 2.0, 0.2 - 1.2, 0.2 - 0.75))
    assert above == pytest.approx((-3.0, 2.0, 0.2 + 0.75, 0.2 + 1.2))
    assert below[3] <= windows["on"][2] and windows["on"][3] <= above[2]


# No mask: on and off areas of the band formula; half of the phi1 range masked: half the areas
def test_region_area_matches_analytic(candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    regions = {"phi1_range": (-3.0, 2.0), "phi2_centre": 0.2, "sigma": 0.3, "on_n_sigma": 1.5, "off_n_sigma": (2.5, 4.0)}
    windows = region_windows(regions)
    areas = region_area_deg2(regions, frame, None)
    band = lambda window: unmasked_area_deg2(frame, window[:2], window[2:], None)[0, 0]  # noqa: E731
    assert areas["on"] == pytest.approx(band(windows["on"]))
    assert areas["off"] == pytest.approx(band(windows["off"][0]) + band(windows["off"][1]))
    half = region_area_deg2(regions, frame, _mask_phi1_below(frame, 512, -0.5))
    assert half["on"] == pytest.approx(areas["on"] / 2, rel=0.03) and half["off"] == pytest.approx(areas["off"] / 2, rel=0.03)


def _uniform_in_window(rng, window, n):
    """n points uniform on the sphere in a phi1/phi2 window."""
    phi1 = rng.uniform(window[0], window[1], n)
    phi2 = np.degrees(np.arcsin(rng.uniform(*np.sin(np.radians(window[2:])), n)))
    return phi1, phi2


def _mask_phi1_below(frame, nside, phi1_limit):
    """HEALPix mask (RING) of the sky at phi1 < phi1_limit in the frame."""
    ra, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    phi1, _ = radec_to_phi12(ra, dec, frame)
    return phi1 < phi1_limit


# Uniform stars, half of the window excluded (and its stars dropped, as prepare does): flat density in
# phi2, consistent with the true one; no dip where the mask is
def test_phi2_profile_flat_with_half_masked(rng, candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    window = (-4.0, 4.0, -3.0, 3.0)
    n = 40_000
    phi1, phi2 = _uniform_in_window(rng, window, n)
    true_density = n / unmasked_area_deg2(frame, window[:2], window[2:], None)[0, 0]
    mask = _mask_phi1_below(frame, 256, 0.0)
    ra, dec = phi12_to_radec(phi1, phi2, frame)
    kept = ~mask[hp.ang2pix(256, ra, dec, lonlat=True)]
    profile = phi12_density_profile(phi1[kept], phi2[kept], [window], frame, mask, axis="phi2", bin_deg=0.25)
    assert profile["centres"].size == 24 and profile["bin_deg"] == pytest.approx(0.25)
    assert profile["area_deg2"].sum() == pytest.approx(
        unmasked_area_deg2(frame, window[:2], window[2:], None)[0, 0] / 2, rel=0.02)
    pulls = (profile["density"] - true_density) / profile["error"]
    assert abs(pulls.mean()) < 3 / np.sqrt(pulls.size) and 0.6 < pulls.std() < 1.4


# Two windows: their counts and areas add up; bins outside both windows have no area
def test_profile_over_two_windows(rng, candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    below, above = (-2.0, 2.0, -3.0, -1.0), (-2.0, 2.0, 1.0, 3.0)
    phi1, phi2 = _uniform_in_window(rng, (-2.0, 2.0, -3.0, 3.0), 5000)
    kwargs = {"axis": "phi1", "bin_deg": 0.5}
    both = phi12_density_profile(phi1, phi2, [below, above], frame, None, **kwargs)
    one, other = (phi12_density_profile(phi1, phi2, [window], frame, None, **kwargs) for window in (below, above))
    np.testing.assert_array_equal(both["counts"], one["counts"] + other["counts"])
    np.testing.assert_allclose(both["area_deg2"], one["area_deg2"] + other["area_deg2"])
    wide = phi12_density_profile(phi1, phi2, [below], frame, None, axis="phi1", bin_deg=0.5, axis_range=(-4.0, 4.0))
    outside = (wide["centres"] < -2.0) | (wide["centres"] > 2.0)
    assert np.all(wide["area_deg2"][outside] == 0.0) and np.all(np.isnan(wide["density"][outside]))


# Uniform stars + a Gaussian stream of known centre and sigma: both recovered within 3 errors
def test_fit_recovers_centre_and_sigma(rng, candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    window = (-3.0, 3.0, -4.0, 4.0)
    phi1, phi2 = _uniform_in_window(rng, window, 20_000)
    centre, sigma = 0.4, 0.3
    phi1 = np.append(phi1, rng.uniform(*window[:2], 3000))
    phi2 = np.append(phi2, rng.normal(centre, sigma, 3000))
    profile = phi12_density_profile(phi1, phi2, [window], frame, None, axis="phi2", bin_deg=0.1)
    fit = fit_stream_width(profile, sigma_guess=0.5)
    assert abs(fit["phi2_centre"] - centre) < 3 * fit["errors"]["phi2_centre"]
    assert abs(fit["sigma"] - sigma) < 3 * fit["errors"]["sigma"]
    assert fit["model"].shape == profile["centres"].shape and 0.5 < fit["chi2"] / fit["n_dof"] < 1.5


# A field density with a gradient in phi2 (40 % over the 10 deg window) and a stream, at the statistics of
# the Euclid DR1 candidate (about 8000 field stars, a stream about as dense as the field on its track): the fit
# near the track recovers the stream; the default range is the first guess +- DEFAULT_FIT_HALF_WIDTH_DEG.
# (With far more stars, the slope left in the fitted range biases the centre beyond the errors.)
def test_fit_near_the_track_with_a_gradient(rng, candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    window = (-3.0, 3.0, -5.0, 5.0)
    phi1, phi2 = _uniform_in_window(rng, window, 10_000)
    keep = rng.uniform(0.0, 1.0, phi2.size) < 0.8 + 0.032 * phi2  # density rising with phi2: 0.64 to 0.96
    phi1, phi2 = phi1[keep], phi2[keep]
    centre, sigma = 0.2, 0.3
    phi1 = np.append(phi1, rng.uniform(*window[:2], 400))
    phi2 = np.append(phi2, rng.normal(centre, sigma, 400))
    profile = phi12_density_profile(phi1, phi2, [window], frame, None, axis="phi2", bin_deg=0.1)
    fit = fit_stream_width(profile)
    assert fit["phi2_fit_range"] == (-DEFAULT_FIT_HALF_WIDTH_DEG, DEFAULT_FIT_HALF_WIDTH_DEG)
    assert abs(fit["phi2_centre"] - centre) < 3 * fit["errors"]["phi2_centre"]
    assert abs(fit["sigma"] - sigma) < 3 * fit["errors"]["sigma"]
    narrow = fit_stream_width(profile, phi2_fit_range=(-1.0, 1.5))
    assert narrow["phi2_fit_range"] == (-1.0, 1.5) and narrow["n_dof"] == 25 - 4
    np.testing.assert_allclose(stream_width_model(fit, profile["centres"]), fit["model"])


# Many stars, a field gradient left in the fitted range: a linear background recovers the stream (a
# constant one shifts its centre beyond the errors, see fit_stream_width); the background coefficients
# are those of a polynomial in phi2 - background_reference, constant first
def test_fit_linear_background(rng, candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    window = (-3.0, 3.0, -5.0, 5.0)
    phi1, phi2 = _uniform_in_window(rng, window, 60_000)
    keep = rng.uniform(0.0, 1.0, phi2.size) < 0.8 + 0.032 * phi2
    phi1, phi2 = phi1[keep], phi2[keep]
    centre, sigma = 0.2, 0.3
    phi1 = np.append(phi1, rng.uniform(*window[:2], 3000))
    phi2 = np.append(phi2, rng.normal(centre, sigma, 3000))
    profile = phi12_density_profile(phi1, phi2, [window], frame, None, axis="phi2", bin_deg=0.1)
    fit = fit_stream_width(profile, background_degree=1)
    assert abs(fit["phi2_centre"] - centre) < 3 * fit["errors"]["phi2_centre"]
    assert abs(fit["sigma"] - sigma) < 3 * fit["errors"]["sigma"]
    assert fit["background_degree"] == 1 and len(fit["background_coefficients"]) == 2
    assert fit["background_reference"] == 0.0 and fit["background"] == fit["background_coefficients"][0]
    field = 60_000 / unmasked_area_deg2(frame, window[:2], window[2:], None)[0, 0]  # stars/deg² before thinning
    assert fit["background_coefficients"][1] == pytest.approx(0.032 * field, rel=0.3)
    assert len(fit_stream_width(profile)["background_coefficients"]) == 1
    with pytest.raises(ValueError, match="background_degree"):
        fit_stream_width(profile, background_degree=-1)


# 2D density of uniform stars, no mask: counts of every star, areas adding up to the window, density
# consistent with the true one
def test_phi12_density_map(rng, candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    window = (-3.0, 3.0, -2.0, 2.0)
    n = 30_000
    phi1, phi2 = _uniform_in_window(rng, window, n)
    density_map = phi12_density_map(phi1, phi2, np.linspace(-3.0, 3.0, 13), np.linspace(-2.0, 2.0, 9), frame, None)
    assert density_map["density"].shape == (12, 8) and density_map["counts"].sum() == n
    total_area = unmasked_area_deg2(frame, window[:2], window[2:], None)[0, 0]
    assert density_map["area_deg2"].sum() == pytest.approx(total_area)
    pulls = (density_map["density"] - n / total_area) / density_map["error"]
    assert abs(pulls.mean()) < 3 / np.sqrt(pulls.size) and 0.6 < pulls.std() < 1.4


# A profile along phi1: ValueError
def test_fit_rejects_phi1_profile(rng, candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    phi1, phi2 = _uniform_in_window(rng, (-3.0, 3.0, -1.0, 1.0), 1000)
    profile = phi12_density_profile(phi1, phi2, [(-3.0, 3.0, -1.0, 1.0)], frame, None, axis="phi1", bin_deg=0.5)
    with pytest.raises(ValueError, match="along phi2"):
        fit_stream_width(profile)


# Uniform stars: on - off consistent with 0 in every phi1 bin
def test_on_off_phi1_profile_zero_without_stream(rng, candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    phi1, phi2 = _uniform_in_window(rng, (-5.0, 5.0, -3.0, 3.0), 100_000)
    profile = on_off_phi1_profile(phi1, phi2, {"phi1_range": (-3.0, 2.0), "phi2_centre": 0.2, "sigma": 0.3, "on_n_sigma": 1.5, "off_n_sigma": (2.5, 4.0)}, frame, None, phi1_range=(-5.0, 5.0), bin_deg=0.5)
    assert profile["centres"].size == 20
    pulls = profile["difference"] / profile["error"]
    assert abs(pulls.mean()) < 3 / np.sqrt(pulls.size) and 0.5 < pulls.std() < 1.5


# A stream over phi1 from -3 to 2: positive difference there, consistent with 0 beyond
def test_on_off_phi1_profile_shows_stream_extent(rng, candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    phi1, phi2 = _uniform_in_window(rng, (-5.0, 5.0, -3.0, 3.0), 50_000)
    phi1 = np.append(phi1, rng.uniform(-3.0, 2.0, 3000))
    phi2 = np.append(phi2, rng.normal(0.2, 0.3, 3000))
    profile = on_off_phi1_profile(phi1, phi2, {"phi1_range": (-3.0, 2.0), "phi2_centre": 0.2, "sigma": 0.3, "on_n_sigma": 1.5, "off_n_sigma": (2.5, 4.0)}, frame, None, phi1_range=(-5.0, 5.0), bin_deg=0.5)
    pulls = profile["difference"] / profile["error"]
    inside = (profile["centres"] > -3.0) & (profile["centres"] < 2.0)
    assert np.all(pulls[inside] > 3.0) and np.all(np.abs(pulls[~inside]) < 4.0)


# Before the regions: the arc between the endpoints, at phi2 = 0, from endpoint_1 to endpoint_2 (also
# in ra/dec); with regions: phi2 = phi2_centre over phi1_range
def test_track_phi12_straight(candidate_endpoints):
    cand = CandidateInvestigation("test", *candidate_endpoints)
    phi1, phi2 = cand.track_phi12(n_points=11)
    assert phi1.size == 11 and np.all(phi2 == 0.0) and phi1[0] < 0.0 < phi1[-1]
    ra, dec = cand.track_radec(n_points=11)
    assert (ra[0], dec[0]) == pytest.approx(candidate_endpoints[0], abs=1e-8)
    assert (ra[-1], dec[-1]) == pytest.approx(candidate_endpoints[1], abs=1e-8)
    cand.regions = {"phi1_range": (-2.0, 1.0), "phi2_centre": 0.3, "sigma": 0.2, "on_n_sigma": 1.4,
                    "off_n_sigma": (2.5, 4.5)}
    phi1, phi2 = cand.track_phi12(n_points=7)
    assert phi1 == pytest.approx(np.linspace(-2.0, 1.0, 7)) and np.all(phi2 == 0.3)


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


# define_regions: the regions of the width fit (or the values given) over the phi1 of the endpoints, their
# areas, the phi1 profile over the window; earlier Hess diagrams cleared; errors: steps not done, off bands
# beyond the window
def test_define_regions(candidate_endpoints, candidate_stream_catalog, candidate_stream, no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, on_n_sigma=1.5, off_n_sigma=(2.5, 4.0),
                                  prepare_kwargs={"bin_size_deg": BIN, **no_object_masks})
    with pytest.raises(ValueError, match="prepare"):
        cand.define_regions()
    cand.prepare(candidate_stream_catalog)
    with pytest.raises(ValueError, match="fit_width"):
        cand.define_regions()
    cand.distance_modulus = candidate_stream["distance_modulus"]
    cand.fit_width()
    cand.on_hess = {"stale": True}
    assert cand.define_regions() is cand
    fit, regions = cand.width_fit, cand.regions
    assert regions == stream_regions(cand._endpoints_phi1_range(), fit["phi2_centre"], fit["sigma"], on_n_sigma=1.5,
                                     off_n_sigma=(2.5, 4.0))
    assert cand.areas == region_area_deg2(regions, cand.stream_frame, cand.excluded_healpix_mask)
    assert cand.phi1_profile["edges"][[0, -1]] == pytest.approx(cand.window[:2]) and cand.on_hess is None
    inside = np.abs(cand.phi1_profile["centres"]) < 2.5
    assert np.all(cand.phi1_profile["difference"][inside] > 0)
    cand.define_regions(phi1_range=(-1.0, 1.0), phi2_centre=0.0, sigma=0.4)
    assert cand.regions["phi1_range"] == (-1.0, 1.0) and cand.regions["sigma"] == 0.4
    with pytest.raises(ValueError, match="leave the window"):
        cand.define_regions(sigma=2.0)  # off bands to 8 deg


# ===================================================
# scan_distance_moduli
# ===================================================


# One map per distance modulus given, each that of stream_frame_search_single_dm with the isochrone of the
# object; default: the distance moduli of the object; before prepare: ValueError
def test_scan_distance_moduli(candidate_endpoints, candidate_catalog, no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, feh=-1.5, distance_moduli=[16.0, 17.0],
                                  prepare_kwargs={"bin_size_deg": BIN, **no_object_masks})
    with pytest.raises(ValueError, match="prepare"):
        cand.scan_distance_moduli()
    cand.prepare(candidate_catalog)
    assert list(cand.scan_distance_moduli().maps) == [16.0, 17.0]
    assert list(cand.scan_distance_moduli([18.0, 16.5]).maps) == [18.0, 16.5]
    expected = stream_frame_search_single_dm(search=cand.search, distance_modulus=16.5, isochrone=cand.isochrone)
    np.testing.assert_array_equal(np.ma.filled(cand.maps[16.5]["density"], np.nan),
                                  np.ma.filled(expected["density"], np.nan))
    assert cand.maps[16.5]["isochrone"] == cand.isochrone


# ===================================================
# Later steps
# ===================================================


@pytest.mark.skip(reason="stub")
def test_steps_out_of_order_raise():
    """Each step before the previous one raises a ValueError naming the missing step (e.g. compute_hess before define_regions)."""


# fit_width: the stream of the synthetic catalog (conftest.py) recovered, over the phi1 of the endpoints;
# before prepare, or without distance_modulus: ValueError
def test_fit_width(candidate_endpoints, candidate_stream_catalog, candidate_stream, no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, prepare_kwargs={"bin_size_deg": BIN, **no_object_masks})
    with pytest.raises(ValueError, match="prepare"):
        cand.fit_width()
    cand.prepare(candidate_stream_catalog)
    with pytest.raises(ValueError, match="distance_modulus"):
        cand.fit_width()
    cand.distance_modulus = candidate_stream["distance_modulus"]
    assert cand.fit_width() is cand
    fit = cand.width_fit
    assert abs(fit["phi2_centre"] - candidate_stream["phi2_centre"]) < 3 * fit["errors"]["phi2_centre"]
    assert abs(fit["sigma"] - candidate_stream["sigma"]) < 3 * fit["errors"]["sigma"]
    phi1_ends, _ = radec_to_phi12([e[0] for e in candidate_endpoints], [e[1] for e in candidate_endpoints],
                                  cand.stream_frame)
    assert cand.phi2_profile["windows"][0] == pytest.approx((*phi1_ends, *cand.window[2:]))
    width_map = cand.width_map  # the same stars in 2D, over the same window
    assert width_map["phi1_edges"][[0, -1]] == pytest.approx(phi1_ends)
    assert width_map["phi2_edges"][[0, -1]] == pytest.approx(cand.window[2:])
    assert width_map["counts"].sum() == cand.phi2_profile["counts"].sum()


@pytest.mark.skip(reason="stub")
def test_run_all_steps_recovers_stream():
    """run_all_steps on the synthetic catalog: sigma and phi2_centre recovered; on - off excess positive."""


@pytest.mark.skip(reason="stub")
def test_summary_keys():
    """summary keys; None for the steps not done."""

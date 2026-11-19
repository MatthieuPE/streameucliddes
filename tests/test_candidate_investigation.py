"""Tests of analysis/candidate_investigation.py, on a synthetic catalog
(conftest.py: uniform stars around the endpoints of the example candidate)."""

import healpy as hp
import numpy as np
import pandas as pd
import pytest

from streameucliddes.injection.inject import add_vis_templfit_magnitudes
from streameucliddes.analysis import candidate_investigation as candidate_investigation_module
from streameucliddes.analysis import stream_frame_search as stream_frame_search_module
from streameucliddes.analysis.build_masks import (
    CANDIDATE_MASK_HALF_WIDTH_DEG,
    STREAM_MASK_N_SIGMA,
    object_mask_nside,
    streams_healpix_mask,
)
from streameucliddes.analysis.candidate_investigation import (
    DEFAULT_FOOTPRINT_NSIDE,
    CANDIDATE_FIGURES_DIR,
    CandidateInvestigation,
    hess_error_table,
    on_off_phi1_profile,
    phi12_density_map,
    phi12_density_profile,
    region_areas,
    region_windows,
    stream_band_regions,
    stream_regions,
    update_candidate_best_dm,
)
from streameucliddes.analysis.cmd_match_filter import (
    difference_hess,
    hess_diagram,
    isochrone_cmd_track,
    match_filter_cmd_columns,
    photometric_errors_by_magnitude,
)
from streameucliddes.catalogs.stream_candidates import add_candidate, load_candidates
from streameucliddes.analysis.generic_stream_search import generic_search_single_dm
from streameucliddes.analysis.stream_frame_search import prepare_search, stream_frame_search_single_dm
from streameucliddes.catalogs.isochrone_parameters import (
    DEFAULT_FEHS,
    isochrone_from_feh,
)
from streameucliddes.catalogs.survey_footprint import (
    convert_footprint_healpix_mask,
    in_healpix_mask,
    star_count_nside,
    survey_footprint_healpix_mask,
    trim_footprint_edge,
    unmasked_area_deg2,
)
from streameucliddes.stream_frames import (
    in_phi12_window,
    phi12_to_radec,
    radec_to_phi12,
    stream_frame_from_endpoints,
)
from streameucliddes.unit_conversions import convert_FeH_to_z, convert_metallicity, get_isochrone_model

pytestmark = pytest.mark.candidate_investigation

NSIDE = 64  # coarse sky maps: fast tests
BIN = 0.2  # bins of the maps in the frame (deg)


# ===================================================
# Regions, profiles and width
# ===================================================


# The plain dict of the inputs, as floats
def test_stream_regions_keys():
    regions = stream_regions((-3, 2), 0.2, 0.3, on_n_sigma=1.5, off_n_sigma=(2.5, 4.0))
    assert regions == {"phi1_range": (-3.0, 2.0), "phi2_centre": 0.2, "sigma": 0.3, "on_n_sigma": 1.5, "off_n_sigma": (2.5, 4.0),
                       "on_phi2_offset_deg": 0.0, "off_phi2_offsets_deg": (0.0, 0.0)}


# Off regions overlapping the on region (wider, or shifted), off_n_sigma not increasing, sigma <= 0, phi1_range
# not increasing
@pytest.mark.parametrize("kwargs", [{"on_n_sigma": 2.6}, {"on_phi2_offset_deg": 0.5}, {"off_phi2_offsets_deg": (0.5, 0.0)},
                                    {"off_n_sigma": (3.0, 2.5)}, {"sigma": 0.0}, {"phi1_range": (2.0, -3.0)}])
def test_stream_regions_rejects_overlap(kwargs):
    arguments = {"phi1_range": (-3.0, 2.0), "phi2_centre": 0.0, "sigma": 0.3, "on_n_sigma": 1.4,
                 "off_n_sigma": (2.5, 4.5), **kwargs}
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


# Offsets shift each window in phi2, phi1 unchanged
def test_region_windows_offsets():
    regions = stream_regions((-3.0, 2.0), 0.2, 0.3, on_n_sigma=1.5, off_n_sigma=(2.5, 4.0), on_phi2_offset_deg=1.5,
                             off_phi2_offsets_deg=(-0.1, 1.5))
    shifted, nominal = region_windows(regions), region_windows({**regions, "on_phi2_offset_deg": 0.0,
                                                                 "off_phi2_offsets_deg": (0.0, 0.0)})
    for window, reference, offset in [(shifted["on"], nominal["on"], 1.5), (shifted["off"][0], nominal["off"][0], -0.1),
                                      (shifted["off"][1], nominal["off"][1], 1.5)]:
        assert window == pytest.approx((reference[0], reference[1], reference[2] + offset, reference[3] + offset))


def _window_areas(regions, frame, excluded_healpix_mask=None):
    """Areas of the windows of region_windows outside a mask (unmasked_area_deg2), as region_areas."""
    windows = region_windows(regions)
    on, below, above = (unmasked_area_deg2(frame, window[:2], window[2:], excluded_healpix_mask)[0, 0]
                        for window in (windows["on"], *windows["off"]))
    return {"on": on, "off_below": below, "off_above": above, "off": below + above}


# The bands of a straight track are the windows of region_windows: their stars, and the areas of the strip
# formula; half of the phi1 range masked: half the areas
def test_stream_band_regions(candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    regions = {"phi1_range": (-3.0, 2.0), "phi2_centre": 0.2, "sigma": 0.3, "on_n_sigma": 1.5, "off_n_sigma": (2.5, 4.0),
               "on_phi2_offset_deg": 0.1, "off_phi2_offsets_deg": (-0.2, 0.3)}
    rng = np.random.default_rng(4)
    ra, dec = phi12_to_radec(rng.uniform(-4.0, 3.0, 20_000), rng.uniform(-2.0, 2.5, 20_000), frame)
    stars = pd.DataFrame({"ra": ra, "dec": dec})
    band_regions = stream_band_regions(stars, regions, frame, None)
    assert band_regions["regions"] == regions
    windows = region_windows(regions)
    phi1, phi2 = radec_to_phi12(ra, dec, frame)
    for name, window in (("on", windows["on"]), ("off_below", windows["off"][0]), ("off_above", windows["off"][1])):
        np.testing.assert_array_equal(band_regions[name], in_phi12_window(phi1, phi2, window))
    areas = region_areas(band_regions)
    for name, area in _window_areas(regions, frame).items():
        assert areas[name] == pytest.approx(area, rel=1e-3)
    half = region_areas(stream_band_regions(stars, regions, frame, _mask_phi1_below(frame, 512, -0.5)))
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
    assert CandidateInvestigation("test", *candidate_endpoints).isochrone["metallicity"] == pytest.approx(
        convert_FeH_to_z(DEFAULT_FEHS[0], get_isochrone_model()))
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
# the footprint of the stars (at the nside of star_count_nside, its edge trimmed); no object masked: the excluded
# mask is that footprint, RING
def test_prepare(candidate_endpoints, candidate_catalog, no_object_masks):
    prepare_kwargs = {"bin_size_deg": BIN, **no_object_masks}
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_moduli=[16.0, 17.0],
                                  prepare_kwargs=prepare_kwargs)
    assert cand.prepare(candidate_catalog) is cand
    ra, dec = candidate_catalog["ra"], candidate_catalog["dec"]
    nside = star_count_nside(ra, dec, DEFAULT_FOOTPRINT_NSIDE)
    assert nside < DEFAULT_FOOTPRINT_NSIDE  # the synthetic catalog is sparse
    footprint = trim_footprint_edge(survey_footprint_healpix_mask(ra, dec, nside, nest=False), nest=False)
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


# Stream masks: a galstreams track crossing the candidate and another listed candidate, 3 deg north, excluded over
# their half-widths (STREAM_MASK_N_SIGMA times the width; that of the candidates without width), not the candidate
# itself;
# their stars not counted; kept by set_endpoints; mask_streams off and on: applied at once, the tracks not built
# again
def test_prepare_masks_other_streams(tmp_path, candidate_endpoints, candidate_catalog, monkeypatch):
    path = tmp_path / "candidates.csv"
    add_candidate(*candidate_endpoints, 17.0, path=path)  # id 1: the candidate
    add_candidate((77.0, -37.0), (70.0, -40.0), 17.0, path=path)  # id 2
    crossing_frame = stream_frame_from_endpoints((73.5, -48.0), (73.5, -35.0))
    crossing = {"name": "Cross-T00", "stream_name": "Cross", "ra": np.array([73.5, 73.5]),
                "dec": np.array([-48.0, -35.0]), "stream_frame": crossing_frame, "width_deg": 0.2, "dm_min": 17.0,
                "dm_max": 17.0}
    monkeypatch.setattr(candidate_investigation_module, "galstreams_mask_tracks", lambda *args, **kwargs: [crossing])
    inputs = {"path": path, "prepare_kwargs": {"bin_size_deg": BIN}}
    cand = CandidateInvestigation.from_candidate_id(1, **inputs).prepare(candidate_catalog)
    unmasked = CandidateInvestigation.from_candidate_id(1, mask_streams=False, **inputs).prepare(candidate_catalog)
    assert cand.summary()["masked_streams"] == ["Cross-T00", "candidate_2"]
    assert unmasked.summary()["masked_streams"] is None and unmasked.stream_healpix_mask is None
    nside = hp.npix2nside(cand.excluded_healpix_mask.size)
    assert nside == object_mask_nside(BIN)
    np.testing.assert_array_equal(cand.stream_healpix_mask, streams_healpix_mask(cand.stream_mask_tracks, nside,
                                                                                 nest=False))
    np.testing.assert_array_equal(cand.excluded_healpix_mask,
                                  unmasked.excluded_healpix_mask | cand.stream_healpix_mask)

    def masked(phi1, phi2, frame):
        return in_healpix_mask(*phi12_to_radec(phi1, phi2, frame), cand.stream_healpix_mask, nest=False)

    pixel = np.degrees(hp.nside2resol(nside))  # the band edge is known to a pixel
    other_frame = stream_frame_from_endpoints((77.0, -37.0), (70.0, -40.0))
    # across each band, away from the other one (candidate 2 at phi1 = 2 deg: ra 71, east of the crossing track)
    for half_width, frame, phi1 in ((STREAM_MASK_N_SIGMA * 0.2, crossing_frame, 0.0),
                                    (CANDIDATE_MASK_HALF_WIDTH_DEG, other_frame, 2.0)):
        inside, outside = half_width - pixel, half_width + pixel
        assert masked([phi1, phi1], [inside, -inside], frame).all()
        assert not masked([phi1, phi1], [outside, -outside], frame).any()
    phi1 = np.linspace(*cand._endpoints_phi1_range(), 200)
    own = masked(phi1, np.zeros(phi1.size), cand.stream_frame)  # the candidate: only where the crossing track is
    assert 0 < own.mean() < 0.2
    stars = cand.search.stars
    on_streams = in_healpix_mask(stars["ra"], stars["dec"], cand.stream_healpix_mask, nest=False)
    assert on_streams.sum() > 100
    np.testing.assert_array_equal(cand.stars_not_excluded, unmasked.stars_not_excluded & ~on_streams)

    def not_again(*args, **kwargs):
        raise AssertionError("the stream tracks are built again")

    monkeypatch.setattr(candidate_investigation_module, "galstreams_mask_tracks", not_again)
    stream_mask = cand.stream_healpix_mask
    cand.set_endpoints((76.0, -41.0), (69.0, -44.0))
    assert cand.stream_healpix_mask is stream_mask and cand.search is not None
    cand.mask_streams = False
    np.testing.assert_array_equal(cand.excluded_healpix_mask, cand.footprint_and_objects_mask)
    assert cand.stars_not_excluded.sum() > (~on_streams).sum() - 1000 and cand.summary()["masked_streams"] is None
    cand.mask_streams = True
    np.testing.assert_array_equal(cand.excluded_healpix_mask, cand.footprint_and_objects_mask | stream_mask)
    assert cand.masked_streams == ["Cross-T00", "candidate_2"]


# Only the streams within stream_mask_max_dm_difference of distance_modulus are masked: a new distance modulus
# masks the crossing track (dm 19) and clears the width; a cleared result names the change in the error
def test_stream_mask_follows_distance_modulus(tmp_path, candidate_endpoints, candidate_stream_catalog,
                                              candidate_stream, monkeypatch):
    frame = stream_frame_from_endpoints((73.5, -48.0), (73.5, -35.0))
    crossing = {"name": "Cross-T00", "stream_name": "Cross", "ra": np.array([73.5, 73.5]),
                "dec": np.array([-48.0, -35.0]), "stream_frame": frame, "width_deg": 0.2, "dm_min": 19.0,
                "dm_max": 19.0}
    monkeypatch.setattr(candidate_investigation_module, "galstreams_mask_tracks", lambda *args, **kwargs: [crossing])
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=17.0, distance_moduli=[17.0],
                                  stream_mask_max_dm_difference=1.5, candidates_path=tmp_path / "none.csv",
                                  prepare_kwargs={"bin_size_deg": BIN})
    cand.prepare(candidate_stream_catalog).fit_width().define_regions()
    assert cand.masked_streams == [] and not cand.stream_healpix_mask.any()
    np.testing.assert_array_equal(cand.excluded_healpix_mask, cand.footprint_and_objects_mask)
    cand.distance_modulus = 18.0
    assert cand.masked_streams == ["Cross-T00"] and cand.width_fit is None and cand.search is not None
    np.testing.assert_array_equal(cand.excluded_healpix_mask,
                                  cand.footprint_and_objects_mask | cand.stream_healpix_mask)
    stars = cand.search.stars
    np.testing.assert_array_equal(cand.stars_not_excluded, ~in_healpix_mask(stars["ra"], stars["dec"],
                                                                            cand.excluded_healpix_mask, nest=False))
    with pytest.raises(ValueError, match=r"cleared because cand.distance_modulus was changed \(17.0 -> 18.0\)"):
        cand.compute_hess()
    with pytest.raises(ValueError, match="width_fit \\(cleared because cand.distance_modulus"):
        cand.define_regions()
    cand.stream_mask_max_width_deg = 0.1  # the crossing track (0.2 deg) is too wide: not masked
    assert cand.masked_streams == []
    cand.stream_mask_max_width_deg = 0.3
    assert cand.masked_streams == ["Cross-T00"]
    cand.stream_mask_max_dm_difference = 0.5
    assert cand.masked_streams == []
    cand.fit_width().define_regions().compute_hess()
    assert cand.diff_hess is not None


# A candidate given by hand that is also in the list (its own band covers its track): not masked, warned, in
# overlapping_streams; left out by unmasked_streams: no warning
def test_prepare_leaves_out_streams_covering_the_candidate(tmp_path, candidate_endpoints, candidate_catalog, caplog):
    path = tmp_path / "candidates.csv"
    add_candidate(*candidate_endpoints, 17.0, path=path)
    inputs = {"candidates_path": path, "prepare_kwargs": {"bin_size_deg": BIN}}
    with caplog.at_level("WARNING", logger=candidate_investigation_module.__name__):
        cand = CandidateInvestigation("by hand", *candidate_endpoints, **inputs).prepare(candidate_catalog)
    assert "not masked" in caplog.text and "candidate_1" in caplog.text
    assert cand.masked_streams == [] and cand.overlapping_streams == ["candidate_1"]
    np.testing.assert_array_equal(cand.excluded_healpix_mask, cand.footprint_and_objects_mask)
    caplog.clear()
    with caplog.at_level("WARNING", logger=candidate_investigation_module.__name__):
        cand = CandidateInvestigation("by hand", *candidate_endpoints, unmasked_streams=["candidate_1"], **inputs)
        cand.prepare(candidate_catalog)
    assert not caplog.text and cand.summary()["masked_streams"] == [] and cand.overlapping_streams == []


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
# areas, the phi1 profile over the window; earlier Hess diagrams cleared; errors: steps not done, off regions
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
    assert cand.band_regions["regions"] == regions and cand.areas == region_areas(cand.band_regions)
    for name, area in _window_areas(regions, cand.stream_frame, cand.excluded_healpix_mask).items():
        assert cand.areas[name] == pytest.approx(area, rel=5e-3)
    assert cand.phi1_profile["edges"][[0, -1]] == pytest.approx(cand.window[:2]) and cand.on_hess is None
    inside = np.abs(cand.phi1_profile["centres"]) < 2.5
    assert np.all(cand.phi1_profile["difference"][inside] > 0)
    cand.define_regions(phi1_range=(-1.0, 1.0), phi2_centre=0.0, sigma=0.4)
    assert cand.regions["phi1_range"] == (-1.0, 1.0) and cand.regions["sigma"] == 0.4
    with pytest.raises(ValueError, match="leave the window"):
        cand.define_regions(sigma=2.0)  # off regions to 8 deg
    cand.define_regions(on_phi2_offset_deg=1.0, off_phi2_offsets_deg=(0.0, 1.0))
    assert cand.regions["on_phi2_offset_deg"] == 1.0 and cand.regions["off_phi2_offsets_deg"] == (0.0, 1.0)
    with pytest.raises(ValueError, match="leave the window"):
        cand.define_regions(on_phi2_offset_deg=cand.window[3])  # on region beyond the top of the window


# A footprint hole across half of the bottom off region, no stream: define_regions leaves the hole out of the
# off area (the on area untouched), so the off density of the uniform field stays that of the on region (on - off
# consistent with 0); divided by the full off area, it would be too low
def test_compute_hess_areas_outside_footprint_hole(candidate_endpoints, candidate_catalog, no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=17.0,
                                  prepare_kwargs={"bin_size_deg": BIN, **no_object_masks})
    nside = 256
    ra, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    phi1, phi2 = radec_to_phi12(ra, dec, cand.stream_frame)
    sigma = 0.4
    hole = (phi1 < 0.0) & (phi2 < -0.5 * (cand.on_n_sigma + cand.off_n_sigma[0]) * sigma)  # between on and off
    cand.prepare(candidate_catalog, footprint_mask=hole, nest=False)
    cand.define_regions(phi2_centre=0.0, sigma=sigma).compute_hess()
    full = _window_areas(cand.regions, cand.stream_frame)
    below = region_windows(cand.regions)["off"][0]
    full_below = unmasked_area_deg2(cand.stream_frame, below[:2], below[2:], None)[0, 0]
    phi1_min, phi1_max = cand.regions["phi1_range"]
    hole_fraction = (0.0 - phi1_min) / (phi1_max - phi1_min)
    assert cand.areas["on"] == pytest.approx(full["on"], rel=1e-3)
    assert cand.areas["off"] == pytest.approx(full["off"] - hole_fraction * full_below, rel=0.02)
    n_on, n_off = cand.on_hess["n_stars"], cand.off_hess["n_stars"]
    error = np.sqrt(n_on / cand.areas["on"] ** 2 + n_off / cand.areas["off"] ** 2)
    assert abs(n_on / cand.areas["on"] - n_off / cand.areas["off"]) < 3 * error
    assert n_on / cand.areas["on"] - n_off / full["off"] > 3 * error


# A field with a linear gradient in phi2 (no stream) and a footprint hole across half of the bottom off region: the
# off density, the mean of the two off regions, matches that of the on region (Hess diagrams, summary, and the
# phi1 profile where both off regions have an area; in the hole the off region above alone cannot cancel the
# gradient); the stars of both over their summed area would be too high, the denser region above weighing more
def test_off_density_cancels_gradient_with_hole(candidate_endpoints, no_object_masks):
    rng = np.random.default_rng(11)
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    gradient, density, phi1_half, phi2_half = 0.15, 1200.0, 12.0, 5.5  # density (1 + gradient phi2) per deg²
    n = int(density * 2 * phi1_half * 2 * phi2_half)
    phi1 = rng.uniform(-phi1_half, phi1_half, 2 * n)
    phi2 = rng.uniform(-phi2_half, phi2_half, 2 * n)
    keep = rng.uniform(0.0, 1.0 + gradient * phi2_half, 2 * n) < 1.0 + gradient * phi2
    ra, dec = phi12_to_radec(phi1[keep], phi2[keep], frame)
    vis = rng.uniform(18.0, 26.0, ra.size)
    catalog = add_vis_templfit_magnitudes(pd.DataFrame({"ra": ra, "dec": dec, "euclid_dr1_VIS_obs": vis,
                            "euclid_dr1_H_obs": vis - rng.uniform(-1.0, 1.5, ra.size),
                            "euclid_dr1_VIS_err": np.full(ra.size, 0.02), "euclid_dr1_H_err": np.full(ra.size, 0.02)}))
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=17.0,
                                  prepare_kwargs={"bin_size_deg": BIN, **no_object_masks})
    nside, sigma = 256, 0.4
    pixel_ra, pixel_dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    pixel_phi1, pixel_phi2 = radec_to_phi12(pixel_ra, pixel_dec, cand.stream_frame)
    hole = (pixel_phi1 < 0.0) & (pixel_phi2 < -0.5 * (cand.on_n_sigma + cand.off_n_sigma[0]) * sigma)
    cand.prepare(catalog, footprint_mask=hole, nest=False)
    cand.define_regions(phi2_centre=0.0, sigma=sigma)
    cand.compute_hess(color_bins=np.linspace(-1.0, 1.5, 11), mag_bins=np.linspace(18.0, 26.0, 9))
    assert cand.areas["off_below"] < 0.6 * cand.areas["off_above"]  # the hole
    on, off = cand.on_hess["density"].sum(), cand.off_hess["density"].sum()
    error = np.sqrt((cand.on_hess["error"] ** 2).sum() + (cand.off_hess["error"] ** 2).sum())
    assert abs(on - off) < 3 * error
    assert cand.off_hess["counts"].sum() / cand.off_hess["area_deg2"] - on > 3 * error
    profile = cand.phi1_profile
    both = np.isfinite(profile["difference"]) & (profile["n_off_regions"] == 2)
    assert both.sum() > 10 and np.any(profile["n_off_regions"] == 1)  # the hole: the region above alone
    pulls = profile["difference"][both] / profile["error"][both]
    assert abs(pulls.mean()) < 3 / np.sqrt(pulls.size)
    summary = cand.summary()
    assert abs(summary["n_excess"]) < 3 * summary["n_excess_error"]


# compute_hess: every prepared star of the on region and of the off regions, per deg² with the region areas,
# on - off; the stream's stars in the match filter make the difference positive there
def test_compute_hess(candidate_endpoints, candidate_stream_catalog, candidate_stream, no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=candidate_stream["distance_modulus"],
                                  prepare_kwargs={"bin_size_deg": BIN, **no_object_masks})
    cand.prepare(candidate_stream_catalog).fit_width().define_regions()
    assert cand.compute_hess(color_bins=np.linspace(-1.0, 1.5, 11), mag_bins=np.linspace(18.0, 26.0, 9)) is cand
    windows = region_windows(cand.regions)
    phi1, phi2 = cand.search.stars["phi1"], cand.search.stars["phi2"]
    assert cand.on_hess["n_stars"] == (cand.stars_not_excluded & in_phi12_window(phi1, phi2, windows["on"])).sum()
    assert cand.on_hess["area_deg2"] == cand.areas["on"] and cand.off_hess["area_deg2"] == cand.areas["off"]
    np.testing.assert_allclose(cand.diff_hess["density"], cand.on_hess["density"] - cand.off_hess["density"])
    assert cand.diff_hess["density"].sum() > 5 * np.sqrt((cand.diff_hess["error"] ** 2).sum())


# Null test: the on region and the off region on its side shifted off the stream by (on_n_sigma + off_n_sigma[0])
# sigma, the stream in none of the three regions: no excess, while the candidate's regions have one; regions and Hess
# diagrams of the candidate unchanged; a given direction; a shift leaving the window, an unknown direction:
# ValueError; cleared by new regions
def test_compute_null_test(candidate_endpoints, candidate_stream_catalog, candidate_stream, no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=candidate_stream["distance_modulus"],
                                  prepare_kwargs={"bin_size_deg": BIN, **no_object_masks})
    cand.prepare(candidate_stream_catalog)
    with pytest.raises(ValueError, match="define_regions"):
        cand.compute_null_test()
    centre, sigma = candidate_stream["phi2_centre"], candidate_stream["sigma"]
    bins = {"color_bins": np.linspace(-1.0, 1.5, 11), "mag_bins": np.linspace(18.0, 26.0, 9)}
    cand.define_regions(phi2_centre=centre, sigma=sigma).compute_hess(**bins)
    regions, diff_hess = cand.regions, cand.diff_hess
    assert cand.compute_null_test(**bins) is cand
    null_test = cand.null_test
    assert cand.regions is regions and cand.diff_hess is diff_hess
    assert null_test["direction"] in ("above", "below") and null_test["geometry"] == "standard"
    assert null_test["shift_deg"] == pytest.approx((cand.on_n_sigma + cand.off_n_sigma[0]) * sigma)
    stream_edge = cand.off_n_sigma[0] * sigma  # the stream taken as absent beyond it, as for the off regions
    windows = region_windows(null_test["regions"])
    for window in (windows["on"], *windows["off"]):
        assert window[2] >= centre + stream_edge - 1e-9 or window[3] <= centre - stream_edge + 1e-9
    assert cand.summary()["snr"] > 5.0 and abs(null_test["snr"]) < 3.0
    assert null_test["diff_hess"]["area_deg2"] == null_test["areas"]["on"]

    below = cand.compute_null_test(direction="below", **bins).null_test
    assert below["direction"] == "below" and region_windows(below["regions"])["on"][3] <= centre - stream_edge + 1e-9
    with pytest.raises(ValueError, match="leave the window"):
        cand.compute_null_test(shift_n_sigma=100.0)
    with pytest.raises(ValueError, match="direction"):
        cand.compute_null_test(direction="left")
    cand.define_regions(phi2_centre=centre, sigma=sigma)
    assert cand.null_test is None


# A wide stream in a narrow window: the shifted regions with their gap leave the window on both sides; the off region
# next to the shifted on region fits ("compact"), the stream still in none of the regions
def test_compute_null_test_compact(candidate_endpoints, candidate_catalog, no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=17.0,
                                  prepare_kwargs={"bin_size_deg": BIN, **no_object_masks})
    cand.prepare(candidate_catalog)
    half_width = cand.window[3]
    on_n, (off_0, off_1) = cand.on_n_sigma, cand.off_n_sigma
    # extent of the shifted side: (on + off_0 + off_1) sigma with the gap, (2 on + off_1) sigma without: in between
    sigma = half_width / (0.5 * ((on_n + off_0 + off_1) + (2 * on_n + off_1)))
    cand.define_regions(phi2_centre=0.0, sigma=sigma).compute_null_test()
    null_test = cand.null_test
    assert null_test["geometry"] == "compact"
    windows = region_windows(null_test["regions"])
    shifted_off = windows["off"][1] if null_test["direction"] == "above" else windows["off"][0]
    if null_test["direction"] == "above":
        assert shifted_off[2] == pytest.approx(windows["on"][3])  # next to the shifted on region
    else:
        assert shifted_off[3] == pytest.approx(windows["on"][2])
    for window in (windows["on"], *windows["off"]):
        assert window[2] >= off_0 * sigma - 1e-6 or window[3] <= -off_0 * sigma + 1e-6
    with pytest.raises(ValueError, match="also with the off region next to"):
        cand.compute_null_test(shift_n_sigma=2 * (on_n + off_0))


# "auto": the side with the larger unmasked on region, here above (the footprint has a hole below the stream)
def test_compute_null_test_auto_side(candidate_endpoints, candidate_catalog, no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=17.0,
                                  prepare_kwargs={"bin_size_deg": BIN, **no_object_masks})
    nside = 256
    ra, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    phi1, phi2 = radec_to_phi12(ra, dec, cand.stream_frame)
    sigma = 0.3
    hole = (phi1 < 0.0) & (phi2 < -cand.off_n_sigma[0] * sigma)  # half of the regions below the stream
    cand.prepare(candidate_catalog, footprint_mask=hole, nest=False)
    cand.define_regions(phi2_centre=0.0, sigma=sigma).compute_null_test()
    assert cand.null_test["direction"] == "above"
    below = cand.compute_null_test(direction="below").null_test
    assert below["areas"]["on"] < cand.compute_null_test(direction="above").null_test["areas"]["on"]


# best_dm of the listed candidates: the distance modulus of the largest SNR (the regions at best_dm_by_eye), written
# to the file; a candidate with best_dm skipped unless replace_force; only the ids given
def test_update_candidate_best_dm(candidate_endpoints, candidate_stream_catalog, candidate_stream, no_object_masks,
                                  tmp_path, monkeypatch):
    path = tmp_path / "candidates.csv"
    add_candidate(*candidate_endpoints, candidate_stream["distance_modulus"] - 0.2, path=path)  # by eye, a bit off
    add_candidate(*candidate_endpoints, candidate_stream["distance_modulus"], path=path)
    distance_moduli = np.round(np.arange(16.4, 17.61, 0.2), 2)
    kwargs = {"distance_moduli": distance_moduli, "path": path,
              "prepare_kwargs": {"bin_size_deg": BIN, **no_object_masks}, "mask_streams": False}
    candidates = update_candidate_best_dm(candidate_stream_catalog, [1], **kwargs)
    assert candidates.loc[0, "best_dm"] == pytest.approx(candidate_stream["distance_modulus"])
    assert np.isnan(candidates.loc[1, "best_dm"])  # not asked
    pd.testing.assert_frame_equal(load_candidates(path), candidates)  # written

    def no_scan(self, distance_moduli=None):
        raise AssertionError("SNR computed again")

    monkeypatch.setattr(CandidateInvestigation, "snr_against_distance_modulus", no_scan)
    update_candidate_best_dm(candidate_stream_catalog, [1], **kwargs)  # already done: skipped
    with pytest.raises(AssertionError, match="computed again"):
        update_candidate_best_dm(candidate_stream_catalog, [1], replace_force=True, **kwargs)
    with pytest.raises(ValueError, match="ids"):
        update_candidate_best_dm(candidate_stream_catalog, [3], **kwargs)


# compute_isochrone: the track of the object's age, Z or [Fe/H] and distance modulus, in its match filter; the
# values given only for this track; no distance modulus, or both Z and [Fe/H]: ValueError
def test_compute_isochrone(candidate_endpoints):
    cand = CandidateInvestigation("test", *candidate_endpoints, age=12.0, feh=-1.5)
    with pytest.raises(ValueError, match="distance_modulus"):
        cand.compute_isochrone()
    cand.distance_modulus = 17.0
    track = cand.compute_isochrone().isochrone_track
    expected = isochrone_cmd_track(17.0, 12.0, feh=-1.5)
    np.testing.assert_array_equal(track["mag"], expected["mag"])
    assert track["feh"] == -1.5 and track["age"] == 12.0
    track = cand.compute_isochrone(age=10.0, metallicity=0.001, distance_modulus=18.0).isochrone_track
    assert (track["age"], track["metallicity"], track["distance_modulus"], track["feh"]) == (10.0, 0.001, 18.0, None)
    assert (cand.age, cand.feh, cand.distance_modulus) == (12.0, -1.5, 17.0)  # inputs not changed
    with pytest.raises(ValueError, match="exactly one"):  # the check of isochrone_cmd_track
        cand.compute_isochrone(metallicity=0.001, feh=-1.0)


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


# A changed input clears the results computed from it, only those: distance modulus -> width, regions, Hess,
# isochrone; on/off_n_sigma -> regions and Hess, not the width; isochrone -> also the maps; match filter -> the
# prepared stars; the same value clears nothing; compute_isochrone uses the match filter of the search
def test_changed_input_clears_its_results(candidate_endpoints, candidate_stream_catalog, candidate_stream,
                                         no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=candidate_stream["distance_modulus"],
                                  distance_moduli=[17.0], prepare_kwargs={"bin_size_deg": BIN, **no_object_masks})

    def run_steps():
        cand.prepare(candidate_stream_catalog).scan_distance_moduli().fit_width().define_regions().compute_hess()
        cand.compute_isochrone()

    run_steps()
    cand.distance_modulus = cand.distance_modulus  # the same value
    assert cand.diff_hess is not None and cand.isochrone_track is not None
    cand.distance_modulus = 17.5
    assert all(getattr(cand, name) is None for name in ("width_fit", "regions", "areas", "diff_hess", "isochrone_track"))
    assert cand.maps is not None and cand.search is not None
    cand.distance_modulus = candidate_stream["distance_modulus"]
    run_steps()
    cand.off_n_sigma = (3.0, 5.0)
    assert cand.regions is None and cand.diff_hess is None and cand.width_fit is not None
    cand.define_regions()
    cand.age = 11.0
    assert cand.maps is None and cand.width_fit is None and cand.search is not None
    assert cand.isochrone_track is None
    run_steps()
    assert cand.isochrone_track["color"].size and cand.search.match_filter is not None
    cand.match_filter = cand.match_filter  # the same value: nothing cleared
    assert cand.search is not None
    cand.match_filter = {**cand.search.match_filter, "name": "renamed"}
    assert cand.search is None and cand.width_fit is None


# Each step before the one it needs: ValueError naming the missing step
def test_steps_out_of_order_raise(candidate_endpoints):
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=17.0)
    for call, step in [(cand.compute_zoom_map, "prepare"), (cand.scan_distance_moduli, "prepare"),
                       (cand.fit_width, "prepare"), (cand.define_regions, "prepare"),
                       (cand.compute_hess, "define_regions")]:
        with pytest.raises(ValueError, match=step):
            call()


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


# run_all_steps on the synthetic catalog: the results of the steps called one by one (with the same step_kwargs),
# the stream recovered, a positive on - off excess; no distance_modulus or an unknown step: ValueError
def test_run_all_steps(candidate_endpoints, candidate_stream_catalog, candidate_stream, no_object_masks):
    inputs = {"distance_modulus": candidate_stream["distance_modulus"], "distance_moduli": [16.5, 17.0],
              "prepare_kwargs": {"bin_size_deg": BIN, **no_object_masks}}
    sky_kwargs = {"nside": NSIDE, **no_object_masks}
    hess_kwargs = {"color_bins": np.linspace(-1.0, 1.5, 11), "mag_bins": np.linspace(18.0, 26.0, 9)}
    cand = CandidateInvestigation("test", *candidate_endpoints, **inputs)
    assert cand.run_all_steps(candidate_stream_catalog, step_kwargs={
        "compute_sky_map": sky_kwargs, "fit_width": {"background_degree": 1}, "compute_hess": hess_kwargs}) is cand
    by_step = CandidateInvestigation("test", *candidate_endpoints, **inputs)
    by_step.compute_sky_map(candidate_stream_catalog, **sky_kwargs).prepare().compute_zoom_map()
    by_step.scan_distance_moduli().fit_width(background_degree=1).define_regions().compute_hess(**hess_kwargs)
    by_step.compute_isochrone()
    assert list(cand.maps) == [16.5, 17.0] and cand.sky_map["distance_modulus"] == cand.distance_modulus
    assert cand.width_fit["phi2_centre"] == by_step.width_fit["phi2_centre"]
    assert len(cand.width_fit["background_coefficients"]) == 2
    assert cand.regions == by_step.regions and cand.areas == by_step.areas
    np.testing.assert_array_equal(cand.diff_hess["density"], by_step.diff_hess["density"])
    np.testing.assert_array_equal(cand.isochrone_track["color"], by_step.isochrone_track["color"])
    fit = cand.width_fit
    assert abs(fit["phi2_centre"] - candidate_stream["phi2_centre"]) < 3 * fit["errors"]["phi2_centre"]
    assert abs(fit["sigma"] - candidate_stream["sigma"]) < 3 * fit["errors"]["sigma"]
    summary = cand.summary()
    assert summary["n_excess"] > 5 * summary["n_excess_error"]
    with pytest.raises(ValueError, match="distance_modulus"):
        CandidateInvestigation("test", *candidate_endpoints).run_all_steps(candidate_stream_catalog)
    with pytest.raises(ValueError, match="prepare"):
        cand.run_all_steps(candidate_stream_catalog, step_kwargs={"prepare": {}})


# The SNR of fixed regions over distance moduli: that of summary at the distance modulus of the object, the stream at
# its distance modulus the highest; the object unchanged (distance_modulus, regions)
def test_snr_against_distance_modulus(candidate_endpoints, candidate_stream_catalog, candidate_stream, no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=candidate_stream["distance_modulus"],
                                  prepare_kwargs={"bin_size_deg": BIN, **no_object_masks})
    cand.prepare(candidate_stream_catalog)
    with pytest.raises(ValueError, match="define_regions"):
        cand.snr_against_distance_modulus([17.0])
    cand.define_regions(phi2_centre=candidate_stream["phi2_centre"], sigma=candidate_stream["sigma"])
    regions = cand.regions
    distance_moduli = [16.0, 16.5, 17.0, 17.5, 18.0]
    table = cand.snr_against_distance_modulus(distance_moduli)
    assert list(table.columns) == ["distance_modulus", "n_on", "n_off", "n_excess", "n_excess_error", "snr"]
    assert table["distance_modulus"].tolist() == distance_moduli
    at_stream = table.set_index("distance_modulus").loc[candidate_stream["distance_modulus"]]
    assert at_stream["snr"] == pytest.approx(cand.summary()["snr"])
    assert table["snr"].idxmax() == distance_moduli.index(candidate_stream["distance_modulus"])
    assert cand.distance_modulus == candidate_stream["distance_modulus"] and cand.regions is regions
    assert len(cand.snr_against_distance_modulus()) == len(cand.search.distance_moduli)  # default: the search grid


# summary: the same keys at every step, None for the steps not done; after define_regions, the counts of the stars
# in the match filter outside the excluded mask, in the on region and in the off regions, and the excess
def test_summary(candidate_endpoints, candidate_stream_catalog, candidate_stream, no_object_masks):
    keys = {"name", "endpoint_1", "endpoint_2", "window", "age", "metallicity", "feh", "distance_modulus",
            "distance_kpc", "masked_streams", "overlapping_streams", "phi2_centre", "sigma", "phi1_range", "area_on_deg2", "area_off_deg2", "n_on", "n_off",
            "n_excess", "n_excess_error", "snr"}
    cand = CandidateInvestigation("test", *candidate_endpoints, age=12.0, feh=-2.0,
                                  prepare_kwargs={"bin_size_deg": BIN, **no_object_masks})
    summary = cand.summary()
    assert set(summary) == keys
    assert summary["age"] == 12.0 and summary["feh"] == -2.0 and summary["metallicity"] == cand.isochrone["metallicity"]
    assert summary["window"] == pytest.approx(cand.window)
    assert all(summary[key] is None for key in keys - {"name", "endpoint_1", "endpoint_2", "window", "age",
                                                        "metallicity", "feh"})
    cand.distance_modulus = candidate_stream["distance_modulus"]
    cand.prepare(candidate_stream_catalog).fit_width()
    summary = cand.summary()
    assert summary["distance_kpc"] == pytest.approx(10 ** (cand.distance_modulus / 5 + 1) / 1e3)
    assert summary["sigma"] == cand.width_fit["sigma"] and summary["n_on"] is None
    cand.define_regions(phi2_centre=0.2, sigma=0.3)
    summary = cand.summary()
    assert set(summary) == keys and summary["sigma"] == 0.3 and summary["phi2_centre"] == 0.2
    assert summary["phi1_range"] == cand.regions["phi1_range"] and summary["area_on_deg2"] == cand.areas["on"]
    windows = region_windows(cand.regions)
    phi1, phi2 = cand.search.stars["phi1"], cand.search.stars["phi2"]
    selected = cand.stars_in_match_filter()
    n_on = (selected & in_phi12_window(phi1, phi2, windows["on"])).sum()
    n_below, n_above = ((selected & in_phi12_window(phi1, phi2, window)).sum() for window in windows["off"])
    areas = cand.areas
    off_density = 0.5 * (n_below / areas["off_below"] + n_above / areas["off_above"])  # both off regions with an area
    off_variance = 0.25 * (n_below / areas["off_below"] ** 2 + n_above / areas["off_above"] ** 2)
    assert (summary["n_on"], summary["n_off"]) == (n_on, n_below + n_above)
    assert summary["n_excess"] == pytest.approx(n_on - areas["on"] * off_density)
    assert summary["n_excess_error"] == pytest.approx(np.sqrt(n_on + areas["on"] ** 2 * off_variance))
    assert summary["snr"] == pytest.approx(summary["n_excess"] / np.sqrt(areas["on"] * off_density))

    # Regions edited by hand: their stars and areas built again
    cand.regions = stream_regions(cand.regions["phi1_range"], 0.2, 0.5, on_n_sigma=1.5, off_n_sigma=(2.5, 4.0))
    summary = cand.summary()
    assert cand.band_regions["regions"] == cand.regions and summary["area_on_deg2"] == pytest.approx(
        _window_areas(cand.regions, cand.stream_frame, cand.excluded_healpix_mask)["on"], rel=5e-3)


# From the id of a listed candidate: its endpoints, best_dm as distance modulus, a default name and its figure
# folder (that of an unlisted one: unnamed); the other inputs as given, a distance_modulus or figure_folder given
# replacing those of the list; the same frame as from the endpoints
def test_from_candidate_id(tmp_path, candidate_endpoints):
    path = tmp_path / "candidates.csv"
    add_candidate((45.5, -45.0), (44.0, -51.0), 17.0, path=path)
    add_candidate(*candidate_endpoints, 17.5, path=path)
    cand = CandidateInvestigation.from_candidate_id(2, path=path, age=12.0)
    assert cand.name == "candidate_2" and cand.distance_modulus == 17.5 and cand.age == 12.0
    assert (cand.endpoint_1, cand.endpoint_2) == candidate_endpoints
    by_hand = CandidateInvestigation("by hand", *candidate_endpoints)
    assert cand.window == pytest.approx(by_hand.window)
    assert cand.figure_folder == CANDIDATE_FIGURES_DIR / "2" and by_hand.figure_folder == CANDIDATE_FIGURES_DIR / "unnamed"
    other = CandidateInvestigation.from_candidate_id(2, path=path, name="stream 2", distance_modulus=18.0,
                                                     figure_folder=tmp_path / "figures")
    assert other.name == "stream 2" and other.distance_modulus == 18.0 and other.figure_folder == tmp_path / "figures"
    assert not (tmp_path / "figures").exists()  # not created
    with pytest.raises(ValueError, match="No candidate with id 5"):
        CandidateInvestigation.from_candidate_id(5, path=path)


# The isochrone model: in the isochrone and the track; changing it clears the maps
def test_isochrone_model(candidate_endpoints):
    cand = CandidateInvestigation("test", *candidate_endpoints, feh=-1.5, isochrone_model="Dotter2016")
    assert cand.isochrone == isochrone_from_feh(feh=-1.5, isochrone_model="Dotter2016")
    assert cand.compute_isochrone(distance_modulus=17.0).isochrone_track["isochrone_model"] == "Dotter2016"
    cand.sky_map = {"stub": True}
    cand.isochrone_model = None
    assert cand.sky_map is None and "isochrone_model" not in cand.isochrone


# The photometric errors of the stars of the on region (outside the masks), all or in the match filter; cleared with
# the regions; define_regions needed
def test_compute_photometric_errors(candidate_endpoints, candidate_stream_catalog, candidate_stream, no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=candidate_stream["distance_modulus"],
                                  prepare_kwargs={"bin_size_deg": BIN, **no_object_masks})
    with pytest.raises(ValueError, match="define_regions"):
        cand.compute_photometric_errors()
    cand.prepare(candidate_stream_catalog).fit_width().define_regions()
    assert cand.compute_photometric_errors(mag_bins=np.arange(18.0, 26.5, 0.5)) is cand
    mag_cols = match_filter_cmd_columns(cand.search.match_filter)[0]
    on = np.asarray(cand.band_regions["on"], dtype=bool)
    expected = photometric_errors_by_magnitude(cand.search.stars, mag_cols, mask=on, mag_bins=np.arange(18.0, 26.5, 0.5))
    for band in expected["bands"]:
        np.testing.assert_array_equal(cand.photometric_errors["n_stars"][band], expected["n_stars"][band])
        np.testing.assert_array_equal(cand.photometric_errors["error"][band], expected["error"][band])
    reference = expected["bands"][0]  # VIS 18 to 26: every star of the on region in the bins
    assert cand.photometric_errors["in_match_filter"] is False
    assert cand.photometric_errors["n_stars"][reference].sum() == on.sum()
    cand.compute_photometric_errors(in_match_filter=True, statistic="mean")
    assert cand.photometric_errors["statistic"] == "mean"
    n_in_filter = (on & cand.stars_in_match_filter()).sum()
    assert cand.photometric_errors["in_match_filter"]
    assert cand.photometric_errors["n_stars"][reference].sum() == n_in_filter
    assert 0 < n_in_filter < on.sum()
    cand.on_n_sigma = 1.0
    assert cand.photometric_errors is None


# One row per bin, colour by colour then magnitude by magnitude: the on - off, on and off densities, the areas on
# every row (density times area: the stars of the region); the
# errors of each band against its own magnitude beside, one row per magnitude bin of their own (other bins than the
# diagrams), for the median and the mean;
# NaN below the shorter block
def test_hess_error_table(stars_with_errors):
    stars = stars_with_errors
    mag_cols = ("euclid_dr1_VIS_obs", "euclid_dr1_H_obs")
    bins = {"color_bins": [0.0, 0.4, 0.8], "mag_bins": [20.0, 21.0, 22.0, 23.0]}
    on = hess_diagram(stars, mag_cols, area_deg2=2.0, **bins)
    off = hess_diagram(stars, mag_cols, mask=np.arange(7) < 2, area_deg2=1.0, **bins)
    table = hess_error_table(on, off)
    assert list(table.columns) == ["color_centre", "mag_centre", "difference_density", "on_density", "off_density",
                                   "area_on_deg2", "area_off_deg2"]
    assert len(table) == 6
    np.testing.assert_allclose(table["color_centre"], [0.2, 0.2, 0.2, 0.6, 0.6, 0.6])
    np.testing.assert_allclose(table["mag_centre"], [20.5, 21.5, 22.5] * 2)
    np.testing.assert_allclose(table["difference_density"], difference_hess(on, off)["density"].ravel())
    np.testing.assert_allclose(table["on_density"], on["density"].ravel())
    np.testing.assert_allclose(table["off_density"], off["density"].ravel())
    assert (table["area_on_deg2"] == 2.0).all() and (table["area_off_deg2"] == 1.0).all()
    np.testing.assert_allclose(table["on_density"] * table["area_on_deg2"], on["counts"].ravel())
    for statistic in ("median", "mean"):
        errors = photometric_errors_by_magnitude(stars, mag_cols, mag_bins=[20.0, 22.0, 23.0], statistic=statistic)
        table = hess_error_table(on, off, errors)
        assert list(table.columns)[7:] == ["error_mag_centre_VIS", f"{statistic}_error_VIS",
                                           "error_mag_centre_H", f"{statistic}_error_H"]
        assert len(table) == 6 and table["on_density"].notna().all()
        for band in ("VIS", "H"):
            np.testing.assert_allclose(table[f"error_mag_centre_{band}"][:2], [21.0, 22.5])
            assert table[f"error_mag_centre_{band}"][2:].isna().all()
            np.testing.assert_allclose(table[f"{statistic}_error_{band}"][:2], errors["error"][band])
    many_bins = photometric_errors_by_magnitude(stars, mag_cols, mag_bins=np.arange(20.0, 23.01, 0.25))  # 12 bins
    table = hess_error_table(on, off, many_bins)
    assert len(table) == 12 and table["mag_centre"][6:].isna().all() and table["error_mag_centre_H"].notna().all()

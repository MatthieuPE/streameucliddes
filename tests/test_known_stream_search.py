"""Tests of analysis/known_stream_search.py, with a mock stream on a uniform background.

The background is uniform in the window and in colour-magnitude space; the stream
stars are at phi2 = 0, inside the match filter of the default isochrone at their
distance modulus (drawn in its polygon).
"""

from types import SimpleNamespace

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from streamobs.match_filter import is_in_match_filter

from streameucliddes.analysis.catalog_preparation import prepare_match_filter_catalog
from streameucliddes.analysis.cmd_match_filter import (
    DEFAULT_MATCH_FILTER,
    build_match_filter_polygon,
    select_stars_in_match_filter,
)
from streameucliddes.analysis.known_stream_search import (
    known_stream_search_dm_along_track,
    known_stream_search_dm_scan,
    known_stream_search_single_dm,
)
from streameucliddes.analysis.star_density_maps import star_density_phi12
from streameucliddes.catalogs.galstreams_catalog import galstreams_dm_along_track
from streameucliddes.catalogs.isochrone_parameters import default_isochrone_grid
from streameucliddes.stream_frames import phi12_bin_centres, phi12_to_radec
from streameucliddes.unit_conversions import convert_DM_to_kpc

pytestmark = pytest.mark.known_stream_search

WINDOW = (-10.0, 10.0, -3.0, 3.0)
BIN = 0.2
SMOOTHING = 0.4
STREAM_DM = 17.0
ISOCHRONE = default_isochrone_grid()[0]


def _cmd_in_polygon(rng, n, dm):
    """n (g, r) inside the match filter of the default isochrone at dm."""
    polygon = build_match_filter_polygon(dm, ISOCHRONE["age"], ISOCHRONE["metallicity"], "g", "r", "des", "des")
    g, r = np.empty(0), np.empty(0)
    while g.size < n:
        g_try = rng.uniform(17.0, 25.0, 20 * n)
        r_try = g_try - rng.uniform(-0.5, 1.5, 20 * n)
        inside = is_in_match_filter(g_try, r_try, polygon_vertices=polygon)
        g, r = np.append(g, g_try[inside]), np.append(r, r_try[inside])
    return g[:n], r[:n]


def _catalog(rng, frame, stream_dm_of_phi1, n_background=60_000, n_stream=3000):
    """Background + stream stars, with ra, dec and DES g, r (and errors)."""
    phi1 = rng.uniform(WINDOW[0] - 1, WINDOW[1] + 1, n_background)
    phi2 = rng.uniform(WINDOW[2] - 1, WINDOW[3] + 1, n_background)
    g = rng.uniform(17.0, 25.0, n_background)
    r = g - rng.uniform(-0.5, 1.5, n_background)

    # Stream: 12 groups along phi1, each at the distance modulus of its centre
    group_edges = np.linspace(-8.0, 8.0, 13)
    stream_phi1, stream_g, stream_r = [], [], []
    for low, high in zip(group_edges[:-1], group_edges[1:]):
        n = n_stream // 12
        stream_phi1.append(rng.uniform(low, high, n))
        group_g, group_r = _cmd_in_polygon(rng, n, stream_dm_of_phi1((low + high) / 2))
        stream_g.append(group_g)
        stream_r.append(group_r)
    stream_phi1 = np.concatenate(stream_phi1)
    phi1 = np.concatenate([phi1, stream_phi1])
    phi2 = np.concatenate([phi2, rng.normal(0.0, 0.2, stream_phi1.size)])
    g = np.concatenate([g, *stream_g])
    r = np.concatenate([r, *stream_r])

    ra, dec = phi12_to_radec(phi1, phi2, frame)
    return pd.DataFrame({
        "ra": ra, "dec": dec, "des_yr6_g_obs": g, "des_yr6_r_obs": r,
        "des_yr6_g_err": np.full(g.size, 0.02), "des_yr6_r_err": np.full(g.size, 0.02),
        "is_stream": np.arange(g.size) >= n_background,
    })


@pytest.fixture(scope="module")
def stream_catalog(gc_frame):
    return _catalog(np.random.default_rng(1), gc_frame, lambda phi1: STREAM_DM)


def _track(frame, dm_of_phi1, phi1_range=(-6.0, 6.0), ref="synthetic"):
    """galstreams-like track along phi2 = 0 of frame, with the distance of dm_of_phi1."""
    phi1 = np.linspace(*phi1_range, 121)
    distance = convert_DM_to_kpc(dm_of_phi1(phi1))
    points = coord.SkyCoord(
        phi1=phi1 * u.deg, phi2=np.zeros(phi1.size) * u.deg, distance=distance, frame=frame
    ).transform_to(coord.ICRS())
    return SimpleNamespace(track=points, stream_frame=frame, InfoFlags="1111", ref=ref, track_name="Mock-T00")


def _stream_contrast(result):
    """Mean density in the stream (|phi2| < 0.3, |phi1| < 7) over the background (|phi2| > 1.5)."""
    density = result["density"]
    phi1, phi2 = np.meshgrid(result["phi1_centres"], result["phi2_centres"], indexing="ij")
    in_stream = (np.abs(phi2) < 0.3) & (np.abs(phi1) < 7.0)
    return density[in_stream].mean() / density[np.abs(phi2) > 1.5].mean()


# ===================================================
# Test known_stream_search_single_dm
# ===================================================


# The map is the shared preparation, selection and density map, with its metadata
def test_single_dm_is_the_pipeline(stream_catalog, gc_frame):
    result = known_stream_search_single_dm(
        stream_catalog, gc_frame, WINDOW, STREAM_DM, bin_size_deg=BIN, smoothing_deg=SMOOTHING
    )
    stars = prepare_match_filter_catalog(stream_catalog, stream_frame=gc_frame, window=WINDOW)
    selected = select_stars_in_match_filter(stars, STREAM_DM, ISOCHRONE)
    expected = star_density_phi12(stars["phi1"][selected], stars["phi2"][selected], WINDOW, BIN, smoothing_deg=SMOOTHING)
    np.testing.assert_array_equal(result["density"], expected)

    assert result["n_stars"] == len(stars) and result["n_selected"] == selected.sum()
    for key, centres in zip(("phi1_centres", "phi2_centres"), phi12_bin_centres(WINDOW, BIN)):
        np.testing.assert_array_equal(result[key], centres)
    assert result["distance_modulus"] == STREAM_DM
    assert result["distance_kpc"] == pytest.approx(25.12, abs=0.01)
    assert result["isochrone"] == ISOCHRONE and result["match_filter"] == DEFAULT_MATCH_FILTER
    assert (result["window"], result["bin_size_deg"], result["smoothing_deg"]) == (WINDOW, BIN, SMOOTHING)


# The stream stands out at its distance modulus
def test_single_dm_finds_the_stream(stream_catalog, gc_frame):
    result = known_stream_search_single_dm(stream_catalog, gc_frame, WINDOW, STREAM_DM, bin_size_deg=BIN)
    assert _stream_contrast(result) > 3.0


# Prepared stars give the same map as the catalog; one of them is needed, and a frame
# with the catalog
def test_single_dm_prepared_stars(stream_catalog, gc_frame):
    stars = prepare_match_filter_catalog(stream_catalog, stream_frame=gc_frame, window=WINDOW)
    from_catalog = known_stream_search_single_dm(stream_catalog, gc_frame, WINDOW, STREAM_DM)
    from_stars = known_stream_search_single_dm(None, None, WINDOW, STREAM_DM, stars=stars)
    np.testing.assert_array_equal(from_stars["density"], from_catalog["density"])
    with pytest.raises(ValueError, match="prepared stars"):
        known_stream_search_single_dm(None, gc_frame, WINDOW, STREAM_DM)
    with pytest.raises(ValueError, match="stream_frame"):
        known_stream_search_single_dm(stream_catalog, None, WINDOW, STREAM_DM)


# Bins outside the footprint are masked
def test_single_dm_footprint(stream_catalog, gc_frame):
    phi1_centres, phi2_centres = phi12_bin_centres(WINDOW, BIN)
    footprint_mask = np.broadcast_to(phi1_centres[:, None] > 5.0, (phi1_centres.size, phi2_centres.size))
    result = known_stream_search_single_dm(
        stream_catalog, gc_frame, WINDOW, STREAM_DM, bin_size_deg=BIN, footprint_mask=footprint_mask
    )
    np.testing.assert_array_equal(result["density"].mask, footprint_mask)


# ===================================================
# Test known_stream_search_dm_scan
# ===================================================


# One map per distance modulus, in order, each that of single_dm; the stream stands out
# most at its distance modulus
def test_dm_scan(stream_catalog, gc_frame):
    distance_moduli = [14.5, 17.0, 20.0]
    results = list(known_stream_search_dm_scan(stream_catalog, gc_frame, WINDOW, distance_moduli, bin_size_deg=BIN))
    assert [dm for dm, _ in results] == distance_moduli
    for dm, result in results:
        single = known_stream_search_single_dm(stream_catalog, gc_frame, WINDOW, dm, bin_size_deg=BIN)
        np.testing.assert_array_equal(result["density"], single["density"])
    contrasts = [_stream_contrast(result) for _, result in results]
    assert contrasts[1] > max(contrasts[0], contrasts[2]) + 1.0


# Processes give the same maps, in the same order
def test_dm_scan_workers(stream_catalog, gc_frame):
    distance_moduli = [17.0, 15.0, 19.0]
    serial = list(known_stream_search_dm_scan(stream_catalog, gc_frame, WINDOW, distance_moduli))
    parallel = list(known_stream_search_dm_scan(stream_catalog, gc_frame, WINDOW, distance_moduli, n_workers=2))
    assert [dm for dm, _ in parallel] == distance_moduli
    for (_, expected), (_, result) in zip(serial, parallel):
        np.testing.assert_array_equal(result["density"], expected["density"])
        assert result["n_selected"] == expected["n_selected"]


# ===================================================
# Test known_stream_search_dm_along_track
# ===================================================


# A track at one distance: the map of single_dm at that distance (offset 0)
def test_dm_along_track_constant(stream_catalog, gc_frame):
    track = _track(gc_frame, lambda phi1: np.full(np.shape(phi1), STREAM_DM))
    (dm_offset, result), = known_stream_search_dm_along_track(stream_catalog, track, None, WINDOW, bin_size_deg=BIN)
    single = known_stream_search_single_dm(stream_catalog, gc_frame, WINDOW, STREAM_DM, bin_size_deg=BIN)
    assert dm_offset == 0.0
    np.testing.assert_allclose(result["dm_along_phi1"], STREAM_DM)
    np.testing.assert_array_equal(result["density"], single["density"])
    assert result["n_selected"] == single["n_selected"]


# Distance moduli along phi1: those of the track, rounded, held at the ends of the
# track, plus each offset
def test_dm_along_track_distance_moduli(stream_catalog, gc_frame):
    track = _track(gc_frame, lambda phi1: 17.0 + 0.1 * phi1)  # 16.4 to 17.6 from phi1 -6 to 6
    offsets = (-0.5, 0.0, 0.5)
    results = list(known_stream_search_dm_along_track(
        stream_catalog, track, None, WINDOW, dm_offsets=offsets, dm_step=0.1, bin_size_deg=BIN
    ))
    assert [dm_offset for dm_offset, _ in results] == list(offsets)
    phi1_centres = results[0][1]["phi1_centres"]
    # Held at the ends (just inside: the end points come back from the frame within float rounding)
    held_phi1 = np.clip(phi1_centres, -5.99, 5.99)
    track_dm = galstreams_dm_along_track(track, held_phi1)
    np.testing.assert_allclose(track_dm, 17.0 + 0.1 * held_phi1, atol=1e-6)
    expected = np.round(track_dm / 0.1) * 0.1
    for dm_offset, result in results:
        np.testing.assert_allclose(result["dm_along_phi1"], expected + dm_offset)


# A stream whose distance changes along phi1 is found better by following its track
# than at one distance modulus
def test_dm_along_track_follows_the_stream(gc_frame):
    dm_of_phi1 = lambda phi1: 17.0 + 0.2 * np.asarray(phi1)  # 15.4 to 18.6 along the stream
    catalog = _catalog(np.random.default_rng(2), gc_frame, dm_of_phi1)
    track = _track(gc_frame, dm_of_phi1, phi1_range=(-8.0, 8.0))
    (_, along), = known_stream_search_dm_along_track(catalog, track, None, WINDOW, bin_size_deg=BIN)
    fixed = known_stream_search_single_dm(catalog, gc_frame, WINDOW, 17.0, bin_size_deg=BIN)

    stars = prepare_match_filter_catalog(catalog, stream_frame=gc_frame, window=WINDOW)
    n_stream = catalog.loc[stars.index, "is_stream"].sum()
    assert _stream_contrast(along) > _stream_contrast(fixed) + 0.5
    assert along["n_selected"] - fixed["n_selected"] > 0.2 * n_stream


# A track without distance cannot be followed
def test_dm_along_track_no_distance(stream_catalog, gc_frame):
    track = _track(gc_frame, lambda phi1: np.full(np.shape(phi1), STREAM_DM), ref="ibata2024")
    with pytest.raises(ValueError, match="no distance"):
        next(known_stream_search_dm_along_track(stream_catalog, track, None, WINDOW))

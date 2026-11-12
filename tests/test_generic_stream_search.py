"""Tests of analysis/generic_stream_search.py, with mock stars across ra = 0.

The stars are uniform on a patch of the sky (ra -15 to 15 deg, dec -30 to -10 deg), in
colour-magnitude space too (Euclid VIS and H, the bands of the default match filter).
"""

import tracemalloc

import healpy as hp
import numpy as np
import pandas as pd
import pytest
from streamobs.match_filter import is_in_match_filter

from streameucliddes.analysis import catalog_preparation, generic_stream_search
from streameucliddes.analysis.catalog_preparation import default_error_cuts
from streameucliddes.analysis.cmd_match_filter import build_match_filter_polygon, select_stars_in_match_filter
from streameucliddes.analysis.generic_stream_search import (
    PreparedGenericSearch,
    generic_search_cmd_stars,
    generic_search_dm_scan,
    generic_search_single_dm,
    prepare_generic_search,
)
from streameucliddes.catalogs.catalog_cuts import get_cut_mask
from streameucliddes.catalogs.isochrone_parameters import default_isochrone_grid
from streameucliddes.catalogs.survey_footprint import survey_footprint_healpix_mask

pytestmark = pytest.mark.generic_stream_search

NSIDE = 64  # pixels of 0.9 deg
SMOOTHING = 1.0
VIS, H = "euclid_q1_VIS_obs", "euclid_q1_H_obs"
ISOCHRONE = default_isochrone_grid()[0]


def _uniform_stars(rng, n, ra_range=(-15.0, 15.0), dec_range=(-30.0, -10.0)):
    """ra, dec uniform on the sphere in the ranges (ra wrapped to [0, 360))."""
    ra = rng.uniform(*ra_range, n) % 360.0
    dec = np.degrees(np.arcsin(rng.uniform(*np.sin(np.radians(dec_range)), n)))
    return ra, dec


def _catalog(rng, n, ra=None, dec=None):
    """Stars with ra, dec, Euclid VIS, H and errors; one in 20 with errors above the default cut."""
    if ra is None:
        ra, dec = _uniform_stars(rng, n)
    n = ra.size
    vis = rng.uniform(18.0, 26.0, n)
    errors = np.where(np.arange(n) % 20 == 0, 0.5, 0.02)
    return pd.DataFrame({
        "ra": ra, "dec": dec, VIS: vis, H: vis - rng.uniform(-1.0, 1.5, n),
        "euclid_q1_VIS_err": errors, "euclid_q1_H_err": errors,
    })


def _in_filter(n, rng, dm):
    """n (VIS, H) inside the default match filter of the default isochrone at dm."""
    polygon = build_match_filter_polygon(dm, ISOCHRONE["age"], ISOCHRONE["metallicity"], "VIS", "H", "euclid", "euclid")
    vis, h = np.empty(0), np.empty(0)
    while vis.size < n:
        vis_try = rng.uniform(18.0, 26.0, 20 * n)
        h_try = vis_try - rng.uniform(-1.0, 1.5, 20 * n)
        inside = is_in_match_filter(vis_try, h_try, polygon_vertices=polygon)
        vis, h = np.append(vis, vis_try[inside]), np.append(h, h_try[inside])
    return vis[:n], h[:n]


@pytest.fixture(scope="module")
def catalog():
    rng = np.random.default_rng(2)
    catalog = _catalog(rng, 300_000)
    catalog.loc[[3, 7], "ra"] = np.nan  # no coordinates: never counted
    return catalog


@pytest.fixture(scope="module")
def search(catalog):
    return prepare_generic_search(catalog, nside=NSIDE)


# ===================================================
# Test prepare_generic_search
# ===================================================


# Per star: kept (cuts, coordinates) and its pixel; the footprint from the star counts, as
# survey_footprint_healpix_mask; defaults: match filter, isochrones, distance moduli; the
# catalog is not modified. The same by chunks
def test_prepare(catalog, monkeypatch):
    before = catalog.copy()
    search = prepare_generic_search(catalog, nside=NSIDE)
    finite = np.isfinite(catalog["ra"]) & np.isfinite(catalog["dec"])
    errors_ok = (catalog["euclid_q1_VIS_err"] <= 0.2) & (catalog["euclid_q1_H_err"] <= 0.2)
    np.testing.assert_array_equal(search.keep, finite & errors_ok)
    expected_pixels = hp.ang2pix(NSIDE, catalog["ra"][finite], catalog["dec"][finite], lonlat=True, nest=False)
    np.testing.assert_array_equal(search.pixels[finite.to_numpy()], expected_pixels)
    np.testing.assert_array_equal(search.footprint_mask, survey_footprint_healpix_mask(
        catalog["ra"][finite].to_numpy(), catalog["dec"][finite].to_numpy(), NSIDE, nest=False,
    ))
    assert search.catalog is catalog
    assert search.match_filter == generic_stream_search.DEFAULT_MATCH_FILTER
    assert search.isochrones == default_isochrone_grid()
    assert search.distance_moduli == list(generic_stream_search.DEFAULT_DISTANCE_MODULI)
    assert min(abs(search.footprint_centre_lon), abs(search.footprint_centre_lon - 360.0)) < 1.0  # ra 0
    pd.testing.assert_frame_equal(catalog, before)

    monkeypatch.setattr(generic_stream_search, "CHUNK_SIZE", 7_000)
    by_chunks = prepare_generic_search(catalog, nside=NSIDE)
    np.testing.assert_array_equal(by_chunks.keep, search.keep)
    np.testing.assert_array_equal(by_chunks.pixels, search.pixels)
    np.testing.assert_array_equal(by_chunks.footprint_mask, search.footprint_mask)


# Given values: kept; a NESTED footprint, or at another nside, is put in RING at nside
def test_prepare_given_values(catalog):
    grid = [{"age": 11.0, "metallicity": 0.0002}, {"age": 13.0, "metallicity": 0.0004}]
    search = prepare_generic_search(catalog, nside=NSIDE, default_grid=grid, distance_moduli=[16.0, 17.0], cuts={})
    assert search.isochrones == grid and search.distance_moduli == [16.0, 17.0]
    assert search.keep.sum() == len(catalog) - 2  # no cut: all but the two without coordinates
    assert prepare_generic_search(catalog, nside=NSIDE, isochrone=grid[1], default_grid=grid).isochrones == [grid[1]]

    ring = survey_footprint_healpix_mask(*_uniform_stars(np.random.default_rng(0), 100_000), NSIDE, nest=False)
    nested = hp.reorder(ring, r2n=True)
    np.testing.assert_array_equal(
        prepare_generic_search(catalog, nside=NSIDE, footprint_mask=nested, nest=True).footprint_mask, ring
    )
    coarse = prepare_generic_search(catalog, nside=NSIDE, footprint_mask=hp.ud_grade(ring, NSIDE // 2), nest=False)
    assert coarse.footprint_mask.size == ring.size
    with pytest.raises(ValueError, match="nest"):
        prepare_generic_search(catalog, footprint_mask=ring)
    with pytest.raises(ValueError, match="Empty footprint"):
        prepare_generic_search(catalog, footprint_mask=np.ones_like(ring), nest=False)
    with pytest.raises(KeyError, match=H):
        prepare_generic_search(catalog.drop(columns=H))


# ===================================================
# Test generic_search_single_dm
# ===================================================


# Without smoothing, the density is the count of the stars kept and in the match filter, per
# pixel, masked outside the footprint; the same by chunks
def test_single_dm_counts(catalog, search, monkeypatch):
    result = generic_search_single_dm(distance_modulus=17.0, isochrone=ISOCHRONE, search=search, smoothing_deg=0.0)
    stars = catalog.dropna(subset=["ra"])
    cuts = get_cut_mask(stars, default_error_cuts())
    selected = select_stars_in_match_filter(stars, 17.0, ISOCHRONE, mask=cuts)
    pixels = hp.ang2pix(NSIDE, stars["ra"][selected], stars["dec"][selected], lonlat=True, nest=False)
    expected = np.bincount(pixels, minlength=hp.nside2npix(NSIDE))
    assert result["n_selected"] == selected.sum() > 0
    np.testing.assert_array_equal(result["density"].mask, search.footprint_mask)
    np.testing.assert_array_equal(result["density"].compressed(), expected[~search.footprint_mask])
    assert (result["nside"], result["nest"], result["distance_modulus"]) == (NSIDE, False, 17.0)
    assert result["isochrone"] == ISOCHRONE and result["distance_kpc"] == pytest.approx(25.12, abs=0.01)

    monkeypatch.setattr(generic_stream_search, "CHUNK_SIZE", 7_000)
    by_chunks = generic_search_single_dm(distance_modulus=17.0, isochrone=ISOCHRONE, search=search, smoothing_deg=0.0)
    np.testing.assert_array_equal(by_chunks["density"], result["density"])


# Smoothed counts divided by the smoothed footprint: uniform stars stay uniform up to the
# footprint edges (no drop there, unlike plain smoothing). The footprint: whole pixels (those
# with their centre in the patch), the stars uniform in them
def test_single_dm_edges():
    rng = np.random.default_rng(3)
    ra, dec = _uniform_stars(rng, 600_000, ra_range=(-20.0, 20.0), dec_range=(-35.0, -5.0))
    lon, lat = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    footprint = ~((((lon + 15.0) % 360.0) < 30.0) & (lat > -30.0) & (lat < -10.0))
    inside_stars = ~footprint[hp.ang2pix(NSIDE, ra, dec, lonlat=True)]
    catalog = _catalog(rng, 0, ra[inside_stars], dec[inside_stars])
    search = prepare_generic_search(catalog, nside=NSIDE, footprint_mask=footprint, nest=False)
    result = generic_search_single_dm(distance_modulus=17.0, search=search, smoothing_deg=SMOOTHING)
    inside = ~search.footprint_mask
    neighbours = hp.get_all_neighbours(NSIDE, np.flatnonzero(inside))
    edge = np.zeros_like(inside)
    edge[np.flatnonzero(inside)[(search.footprint_mask[neighbours] & (neighbours >= 0)).any(axis=0)]] = True
    interior_mean = result["density"][inside & ~edge].mean()
    assert result["density"][edge].mean() / interior_mean == pytest.approx(1.0, abs=0.1)

    counts = generic_search_single_dm(distance_modulus=17.0, search=search, smoothing_deg=0.0)["density"]
    plain = hp.smoothing(counts.filled(0.0), sigma=np.radians(SMOOTHING))
    assert plain[edge].mean() / plain[inside & ~edge].mean() < 0.8


# The background follows a smooth gradient across ra = 0, without a jump there: the
# residual is small, the background the same on each side of ra = 0
def test_single_dm_background_across_ra_zero():
    rng = np.random.default_rng(4)
    ra, dec = _uniform_stars(rng, 600_000)
    keep = rng.uniform(size=ra.size) < (1.0 + (dec + 30.0) / 20.0) / 2.0  # twice denser at dec -10 than -30
    search = prepare_generic_search(_catalog(rng, 0, ra[keep], dec[keep]), nside=NSIDE)
    result = generic_search_single_dm(distance_modulus=17.0, search=search, smoothing_deg=SMOOTHING)
    inside = ~search.footprint_mask
    assert abs(result["residual"][inside].mean()) < 0.02 * result["density"][inside].mean()

    lon, lat = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    band = inside & (np.abs(lat + 20.0) < 3.0)
    east, west = band & (lon > 0.0) & (lon < 3.0), band & (lon > 357.0)
    background = result["background"]
    assert background[east].mean() == pytest.approx(background[west].mean(), rel=0.05)
    assert background[lat > -15.0].mean() > 1.3 * background[lat < -25.0].mean()  # the gradient


# A stream in the match filter at dm 17 stands out in the residual at dm 17, less at dm 20
def test_single_dm_finds_a_stream():
    rng = np.random.default_rng(5)
    background = _catalog(rng, 300_000)
    n_stream = 4000
    vis, h = _in_filter(n_stream, rng, 17.0)
    stream = pd.DataFrame({
        "ra": rng.uniform(-10.0, 10.0, n_stream) % 360.0, "dec": rng.normal(-20.0, 0.3, n_stream),
        VIS: vis, H: h, "euclid_q1_VIS_err": 0.02, "euclid_q1_H_err": 0.02,
    })
    search = prepare_generic_search(pd.concat([background, stream], ignore_index=True), nside=NSIDE)
    lon, lat = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    inside = ~search.footprint_mask
    on_stream = inside & (np.abs(lat + 20.0) < 1.0) & ((lon < 8.0) | (lon > 352.0))
    off_stream = inside & (np.abs(lat + 20.0) > 4.0)

    def contrast(dm):
        residual = generic_search_single_dm(distance_modulus=dm, search=search, smoothing_deg=SMOOTHING)["residual"]
        return residual[on_stream].mean() / residual[off_stream].std()

    assert contrast(17.0) > 3.0
    assert contrast(17.0) > 2 * contrast(20.0)


def test_single_dm_errors(catalog, search):
    with pytest.raises(ValueError, match="distance_modulus"):
        generic_search_single_dm(search=search)
    with pytest.raises(ValueError, match="catalog"):
        generic_search_single_dm(catalog, 17.0, search=search)
    with pytest.raises(ValueError, match="nside"):
        generic_search_single_dm(distance_modulus=17.0, search=search, nside=32)
    with pytest.raises(ValueError, match="Give a catalog"):
        generic_search_single_dm(distance_modulus=17.0)


# ===================================================
# Test generic_search_dm_scan
# ===================================================


# Every isochrone, every distance modulus for the first, then the next: the maps of
# single_dm; the same with workers
def test_dm_scan(catalog):
    grid = [{"age": 12.0, "metallicity": 0.0002}, {"age": 13.0, "metallicity": 0.0003}]
    search = prepare_generic_search(catalog, nside=NSIDE, default_grid=grid, distance_moduli=[16.0, 17.5])
    results = list(generic_search_dm_scan(search=search, smoothing_deg=SMOOTHING))
    assert [(dm, result["isochrone"]) for dm, result in results] == [
        (16.0, grid[0]), (17.5, grid[0]), (16.0, grid[1]), (17.5, grid[1])
    ]
    single = generic_search_single_dm(distance_modulus=17.5, isochrone=grid[1], search=search, smoothing_deg=SMOOTHING)
    np.testing.assert_array_equal(results[3][1]["residual"], single["residual"])

    parallel = list(generic_search_dm_scan(search=search, smoothing_deg=SMOOTHING, n_workers=2))
    assert [dm for dm, _ in parallel] == [dm for dm, _ in results]
    for (_, serial_result), (_, parallel_result) in zip(results, parallel):
        np.testing.assert_array_equal(parallel_result["residual"], serial_result["residual"])

    one = list(generic_search_dm_scan(catalog, [17.0], ISOCHRONE, nside=NSIDE, smoothing_deg=0.0))
    assert len(one) == 1 and isinstance(search, PreparedGenericSearch)


# ===================================================
# Memory: the catalog is neither copied nor kept per map
# ===================================================


# Preparation: 5 bytes per star (cut mask and pixels) plus the temporaries of a chunk
# (healpy.ang2pix: about 35 bytes per star of the chunk); a map: the chunks, and the maps
# and their smoothing and background fit (fit_background_model: about 35 bytes per HEALPix
# pixel of the sky). A copy of one float column of the catalog would be 8 bytes per star
def test_memory(monkeypatch):
    chunk_size = 50_000
    monkeypatch.setattr(generic_stream_search, "CHUNK_SIZE", chunk_size)
    monkeypatch.setattr(catalog_preparation, "CHUNK_SIZE", chunk_size)  # cuts and pixels
    n = 3_000_000
    catalog = _catalog(np.random.default_rng(6), n)
    tracemalloc.start()
    try:
        search = prepare_generic_search(catalog, nside=NSIDE)
        _, prepare_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        start, _ = tracemalloc.get_traced_memory()
        generic_search_single_dm(distance_modulus=17.0, search=search, smoothing_deg=SMOOTHING)
        _, map_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert prepare_peak < 5 * n + 60 * chunk_size < 8 * n
    assert map_peak - start < 60 * chunk_size + 150 * hp.nside2npix(NSIDE) < 8 * n


# ===================================================
# Test generic_search_cmd_stars
# ===================================================


# About max_stars stars (at most), all kept by the search, spread over the catalog, the same
# at each call; the magnitude columns of the match filter, values and index of the catalog.
# All the kept stars when there are fewer
def test_cmd_stars(catalog, search):
    stars = generic_search_cmd_stars(search, max_stars=1000)
    assert 850 <= len(stars) <= 1000
    pd.testing.assert_frame_equal(generic_search_cmd_stars(search, max_stars=1000), stars)
    assert list(stars.columns) == [VIS, H]
    rows = catalog.index.get_indexer(stars.index)
    assert search.keep[rows].all()
    assert rows.max() > 0.9 * len(catalog)  # spread over the catalog, not its first rows
    pd.testing.assert_frame_equal(stars, catalog.loc[stars.index, [VIS, H]])
    assert len(generic_search_cmd_stars(search, max_stars=10 * len(catalog))) == search.keep.sum()

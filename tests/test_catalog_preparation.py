"""Tests of analysis/catalog_preparation.py, with a mock catalog."""

import tracemalloc

import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc
import healpy as hp
import numpy as np
import pandas as pd
import pytest

from streameucliddes.analysis import catalog_preparation
from streameucliddes.analysis.catalog_preparation import (
    DEFAULT_ERR_CUT,
    check_columns,
    default_error_cuts,
    prepare_match_filter_catalog,
    resolve_mag_columns,
    star_healpix_pixels,
)
from streameucliddes.analysis.cmd_match_filter import DEFAULT_MATCH_FILTER, DES_GR_MATCH_FILTER
from streameucliddes.stream_frames import radec_to_phi12

pytestmark = pytest.mark.catalog_preparation

MATCH_FILTER_GRI = {
    "name": "des_gri",
    "reference_band": ("des_yr6", "g"),
    "color_bands": [("des_yr6", "r"), ("des_yr6", "i")],
}
# No cut at all: the default error cuts of the default match filter removed
NO_ERROR_CUTS = {column: (None, None) for column in default_error_cuts()}


@pytest.fixture
def catalog(rng):
    """Stars around the conftest stream (dec -20 to -30), DES g, r, i and Euclid VIS, H, with errors."""
    n = 5000
    columns = {"ra": rng.uniform(35.0, 65.0, n), "dec": rng.uniform(-40.0, -10.0, n), "other": np.arange(n)}
    for band in ["des_yr6_g", "des_yr6_r", "des_yr6_i", "euclid_q1_VIS", "euclid_q1_H"]:
        columns[f"{band}_obs"] = rng.uniform(18.0, 26.0, n)
        columns[f"{band}_err"] = rng.uniform(0.0, 0.4, n)
    return pd.DataFrame(columns, index=np.arange(n) + 100)


def _full_sky_catalog(rng, n, n_extra_columns=0):
    """Stars uniform on the whole sky, Euclid VIS and H with errors, and extra columns."""
    columns = {"ra": rng.uniform(0.0, 360.0, n), "dec": np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))}
    for band in ["euclid_q1_VIS", "euclid_q1_H"]:
        columns[f"{band}_obs"] = rng.uniform(18.0, 26.0, n)
        columns[f"{band}_err"] = rng.uniform(0.0, 0.4, n)
    for i in range(n_extra_columns):
        columns[f"extra_{i}"] = rng.normal(size=n)
    return pd.DataFrame(columns)


def _frame(end1, end2):
    return gc.GreatCircleICRSFrame.from_endpoints(
        coord.SkyCoord(ra=end1[0] * u.deg, dec=end1[1] * u.deg),
        coord.SkyCoord(ra=end2[0] * u.deg, dec=end2[1] * u.deg),
    )


# Streams of the window tests: one crossing ra = 0, one near the south pole
FRAMES = {
    "across ra = 0": _frame((350.0, -30.0), (15.0, -40.0)),
    "near the pole": _frame((0.0, -80.0), (120.0, -85.0)),
}


# ===================================================
# Test resolve_mag_columns
# ===================================================


# The '<col>_corrected' column is used when it exists and is requested
def test_resolve_mag_columns():
    data = pd.DataFrame({"mag_g": [20.0], "mag_g_corrected": [19.9], "mag_r": [19.5]})
    assert resolve_mag_columns(data, ["mag_g", "mag_r"], True) == ["mag_g_corrected", "mag_r"]
    assert resolve_mag_columns(data, ["mag_g", "mag_r"], False) == ["mag_g", "mag_r"]


# ===================================================
# Test default_error_cuts and check_columns
# ===================================================


# Errors <= 0.2 on every band of the match filter
def test_default_error_cuts():
    assert default_error_cuts() == {"euclid_q1_VIS_err": DEFAULT_ERR_CUT, "euclid_q1_H_err": DEFAULT_ERR_CUT}
    assert list(default_error_cuts(DES_GR_MATCH_FILTER)) == ["des_yr6_g_err", "des_yr6_r_err"]
    assert DEFAULT_ERR_CUT == (None, 0.2)
    assert list(default_error_cuts(MATCH_FILTER_GRI)) == ["des_yr6_g_err", "des_yr6_r_err", "des_yr6_i_err"]


# Every missing column is named in the error
def test_check_columns(catalog):
    check_columns(catalog, ["ra", "des_yr6_g_obs"])
    with pytest.raises(KeyError, match=r"\['euclid_q1_Y_obs', 'phi1'\]"):
        check_columns(catalog, ["ra", "euclid_q1_Y_obs", "phi1"])


# ===================================================
# Test prepare_match_filter_catalog
# ===================================================


# Default: the error cuts of the default match filter, and only the columns needed
def test_prepare_default_cuts(catalog):
    stars = prepare_match_filter_catalog(catalog)
    passing = (catalog["euclid_q1_VIS_err"] <= 0.2) & (catalog["euclid_q1_H_err"] <= 0.2)
    assert stars.index.equals(catalog.index[passing])  # index of the catalog kept
    assert list(stars.columns) == ["euclid_q1_VIS_obs", "euclid_q1_H_obs", "euclid_q1_VIS_err", "euclid_q1_H_err"]
    des = prepare_match_filter_catalog(catalog, DES_GR_MATCH_FILTER)
    assert list(des.columns) == ["des_yr6_g_obs", "des_yr6_r_obs", "des_yr6_g_err", "des_yr6_r_err"]
    assert prepare_match_filter_catalog(catalog, cuts={}).index.equals(stars.index)  # {}: the defaults
    assert len(prepare_match_filter_catalog(catalog, cuts=NO_ERROR_CUTS)) == len(catalog)  # no cut


# The given cuts are added to the default error cuts; on an error column, they replace its default
def test_prepare_given_cuts(catalog):
    error_cuts = default_error_cuts(MATCH_FILTER_GRI)
    stars = prepare_match_filter_catalog(catalog, MATCH_FILTER_GRI, cuts={"des_yr6_i_obs": (None, 22.0)})
    passing = catalog["des_yr6_i_obs"] <= 22.0
    for column, (_, err_max) in error_cuts.items():
        passing &= catalog[column] <= err_max
    assert 0 < len(stars) and stars.index.equals(catalog.index[passing])

    replaced = prepare_match_filter_catalog(catalog, MATCH_FILTER_GRI, cuts={"des_yr6_g_err": (None, 0.05)})
    passing = catalog["des_yr6_g_err"] <= 0.05
    for column in ("des_yr6_r_err", "des_yr6_i_err"):  # still the default
        passing &= catalog[column] <= error_cuts[column][1]
    assert 0 < len(replaced) and replaced.index.equals(catalog.index[passing])

    removed = prepare_match_filter_catalog(catalog, MATCH_FILTER_GRI, cuts={"des_yr6_g_err": (None, None)})
    assert catalog.loc[removed.index, "des_yr6_g_err"].max() > error_cuts["des_yr6_g_err"][1]


# With a frame: phi1/phi2 of radec_to_phi12, and only the stars inside the window
def test_prepare_stream_frame_window(catalog, gc_frame):
    window = (-5.0, 5.0, -2.0, 2.0)
    stars = prepare_match_filter_catalog(catalog, cuts=NO_ERROR_CUTS, stream_frame=gc_frame, window=window)
    phi1, phi2 = radec_to_phi12(catalog["ra"].to_numpy(), catalog["dec"].to_numpy(), gc_frame)
    inside = (phi1 >= -5) & (phi1 <= 5) & (phi2 >= -2) & (phi2 <= 2)
    assert 0 < len(stars) == inside.sum()
    np.testing.assert_allclose(stars["phi1"], phi1[inside])
    np.testing.assert_allclose(stars["phi2"], phi2[inside])
    assert {"ra", "dec"} <= set(stars.columns) and "other" not in stars.columns

    all_stars = prepare_match_filter_catalog(catalog, cuts=NO_ERROR_CUTS, stream_frame=gc_frame)  # no window
    assert len(all_stars) == len(catalog) and "phi1" in all_stars.columns


# as_mask: the mask of the rows kept, the same rows as the stars; with or without a window
def test_prepare_as_mask(catalog, gc_frame):
    for kwargs in ({}, {"stream_frame": gc_frame, "window": (-5.0, 5.0, -2.0, 2.0)}):
        keep = prepare_match_filter_catalog(catalog, as_mask=True, **kwargs)
        assert keep.dtype == bool and keep.shape == (len(catalog),)
        assert prepare_match_filter_catalog(catalog, **kwargs).index.equals(catalog.index[keep])


# Errors: missing columns (all named), a window without frame
def test_prepare_errors(catalog):
    with pytest.raises(KeyError, match="euclid_q1_Y_obs"):
        prepare_match_filter_catalog(catalog, {**DEFAULT_MATCH_FILTER, "color_bands": [("euclid_q1", "Y")]})
    with pytest.raises(KeyError, match="SPURIOUS"):
        prepare_match_filter_catalog(catalog, cuts={"SPURIOUS_FLAG_euclid": 0})
    with pytest.raises(KeyError, match=r"\['ra', 'dec'\]"):
        prepare_match_filter_catalog(catalog.drop(columns=["ra", "dec"]), stream_frame=object())
    with pytest.raises(ValueError, match="stream_frame"):
        prepare_match_filter_catalog(catalog, window=(-5.0, 5.0, -2.0, 2.0))


# ===================================================
# Test the window: same stars as a transformation of the whole catalog
# ===================================================


# Every star of the window is kept, and only them, wherever the window is on the sky;
# with chunks that do not divide the catalog
@pytest.mark.parametrize("frame_name", FRAMES)
@pytest.mark.parametrize("window", [(-20.0, 20.0, -3.0, 3.0), (100.0, 160.0, -10.0, 5.0)])
def test_prepare_window_brute_force(rng, monkeypatch, frame_name, window):
    monkeypatch.setattr(catalog_preparation, "CHUNK_SIZE", 7_777)
    catalog = _full_sky_catalog(rng, 200_000)
    frame = FRAMES[frame_name]
    stars = prepare_match_filter_catalog(catalog, cuts=NO_ERROR_CUTS, stream_frame=frame, window=window)

    phi1, phi2 = radec_to_phi12(catalog["ra"].to_numpy(), catalog["dec"].to_numpy(), frame)
    inside = (phi1 >= window[0]) & (phi1 <= window[1]) & (phi2 >= window[2]) & (phi2 <= window[3])
    assert 0 < len(stars) == inside.sum()
    np.testing.assert_array_equal(stars.index, catalog.index[inside])
    np.testing.assert_allclose(stars["phi1"], phi1[inside])
    np.testing.assert_allclose(stars["phi2"], phi2[inside])
    np.testing.assert_array_equal(stars["euclid_q1_VIS_obs"], catalog["euclid_q1_VIS_obs"][inside])


# The window pre-selection: only the stars near the window are transformed to phi1/phi2
def test_prepare_window_transforms_few_stars(rng, monkeypatch):
    catalog = _full_sky_catalog(rng, 200_000)
    transformed = []

    def counting_radec_to_phi12(ra, dec, stream_frame):
        transformed.append(len(ra))
        return radec_to_phi12(ra, dec, stream_frame)

    monkeypatch.setattr(catalog_preparation, "radec_to_phi12", counting_radec_to_phi12)
    stars = prepare_match_filter_catalog(
        catalog, cuts=NO_ERROR_CUTS, stream_frame=FRAMES["across ra = 0"], window=(-20.0, 20.0, -3.0, 3.0)
    )
    # Window of 240 deg^2 (0.6 % of the sky): a margin of about a HEALPix pixel around it
    assert len(stars) <= sum(transformed) < 2 * len(stars)
    assert sum(transformed) < 0.02 * len(catalog)


# Stars with NaN coordinates are never in a window
def test_prepare_window_nan(catalog, gc_frame):
    catalog.loc[catalog.index[:100], "ra"] = np.nan
    stars = prepare_match_filter_catalog(catalog, cuts=NO_ERROR_CUTS, stream_frame=gc_frame, window=(-5.0, 5.0, -2.0, 2.0))
    assert not stars.index.isin(catalog.index[:100]).any()
    assert np.isfinite(stars[["phi1", "phi2"]].to_numpy()).all()


# ===================================================
# Test memory: the catalog is neither modified nor copied
# ===================================================


# The catalog is unchanged, and the stars are a copy of it, not a view
def test_prepare_does_not_modify(catalog, gc_frame):
    before = catalog.copy()
    stars = prepare_match_filter_catalog(catalog, stream_frame=gc_frame, window=(-5.0, 5.0, -2.0, 2.0))
    pd.testing.assert_frame_equal(catalog, before)
    assert not np.shares_memory(stars["euclid_q1_VIS_obs"].to_numpy(), catalog["euclid_q1_VIS_obs"].to_numpy())


# Peak memory: the stars kept, plus a few bytes per star of the catalog (the masks, 3
# bytes measured) and the temporary arrays of a chunk; below a copy of one of its columns
@pytest.mark.parametrize("with_window", [False, True])
def test_prepare_memory(rng, monkeypatch, with_window):
    chunk_size = 50_000
    monkeypatch.setattr(catalog_preparation, "CHUNK_SIZE", chunk_size)
    n = 1_000_000
    catalog = _full_sky_catalog(rng, n, n_extra_columns=6)  # 12 columns of floats: 96 MB
    kwargs = {"stream_frame": FRAMES["across ra = 0"], "window": (-20.0, 20.0, -3.0, 3.0)} if with_window else {}

    tracemalloc.start()
    try:
        stars = prepare_match_filter_catalog(catalog, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    overhead = peak - stars.memory_usage(deep=True).sum()
    allowed = 4 * n + 40 * chunk_size
    assert overhead < allowed < 8 * n  # 8 * n: a copy of one column of floats


# Peak memory with as_mask: the mask (1 byte per star) and the temporaries of a chunk
def test_prepare_as_mask_memory(rng, monkeypatch):
    chunk_size = 50_000
    monkeypatch.setattr(catalog_preparation, "CHUNK_SIZE", chunk_size)
    n = 1_000_000
    catalog = _full_sky_catalog(rng, n, n_extra_columns=6)
    tracemalloc.start()
    try:
        prepare_match_filter_catalog(catalog, as_mask=True)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < n + 40 * chunk_size < 8 * n


# ===================================================
# Test star_healpix_pixels
# ===================================================


# The pixels of healpy.ang2pix, in the ordering asked, by chunks; -1 without coordinates
@pytest.mark.parametrize("nest", [False, True])
def test_star_healpix_pixels(rng, monkeypatch, nest):
    monkeypatch.setattr(catalog_preparation, "CHUNK_SIZE", 300)
    catalog = _full_sky_catalog(rng, 1000)
    catalog.loc[[3, 500], "ra"] = np.nan
    catalog.loc[999, "dec"] = np.inf
    pixels = star_healpix_pixels(catalog, 16, nest=nest)
    assert pixels.dtype == np.int32
    bad = np.zeros(len(catalog), dtype=bool)
    bad[[3, 500, 999]] = True
    assert (pixels[bad] == -1).all()
    expected = hp.ang2pix(16, catalog["ra"][~bad], catalog["dec"][~bad], lonlat=True, nest=nest)
    np.testing.assert_array_equal(pixels[~bad], expected)


def test_star_healpix_pixels_errors(rng):
    with pytest.raises(KeyError, match="ra"):
        star_healpix_pixels(_full_sky_catalog(rng, 10).drop(columns="ra"), 16, nest=False)


# Peak memory: the pixels (4 bytes per star) and the temporaries of a chunk
def test_star_healpix_pixels_memory(rng, monkeypatch):
    chunk_size = 50_000
    monkeypatch.setattr(catalog_preparation, "CHUNK_SIZE", chunk_size)
    n = 1_000_000
    catalog = _full_sky_catalog(rng, n)
    tracemalloc.start()
    try:
        star_healpix_pixels(catalog, 512, nest=False)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * n + 60 * chunk_size < 8 * n

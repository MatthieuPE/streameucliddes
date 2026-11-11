"""Tests of analysis/catalog_preparation.py, with a mock catalog."""

import numpy as np
import pandas as pd
import pytest

from streameucliddes.analysis.catalog_preparation import (
    DEFAULT_ERR_CUT,
    check_columns,
    default_error_cuts,
    prepare_match_filter_catalog,
    resolve_mag_columns,
)
from streameucliddes.analysis.cmd_match_filter import DEFAULT_MATCH_FILTER
from streameucliddes.stream_frames import radec_to_phi12

pytestmark = pytest.mark.catalog_preparation

MATCH_FILTER_GRI = {
    "name": "des_gri",
    "reference_band": ("des_yr6", "g"),
    "color_bands": [("des_yr6", "r"), ("des_yr6", "i")],
}


@pytest.fixture
def catalog(rng):
    """Stars around the conftest stream (dec -20 to -30), DES g, r, i with errors."""
    n = 5000
    columns = {"ra": rng.uniform(35.0, 65.0, n), "dec": rng.uniform(-40.0, -10.0, n), "other": np.arange(n)}
    for band in "gri":
        columns[f"des_yr6_{band}_obs"] = rng.uniform(18.0, 26.0, n)
        columns[f"des_yr6_{band}_err"] = rng.uniform(0.0, 0.4, n)
    return pd.DataFrame(columns, index=np.arange(n) + 100)


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
    assert default_error_cuts() == {"des_yr6_g_err": DEFAULT_ERR_CUT, "des_yr6_r_err": DEFAULT_ERR_CUT}
    assert DEFAULT_ERR_CUT == (None, 0.2)
    assert list(default_error_cuts(MATCH_FILTER_GRI)) == ["des_yr6_g_err", "des_yr6_r_err", "des_yr6_i_err"]


# Every missing column is named in the error
def test_check_columns(catalog):
    check_columns(catalog, ["ra", "des_yr6_g_obs"])
    with pytest.raises(KeyError, match=r"\['euclid_q1_VIS_obs', 'phi1'\]"):
        check_columns(catalog, ["ra", "euclid_q1_VIS_obs", "phi1"])


# ===================================================
# Test prepare_match_filter_catalog
# ===================================================


# Default: the error cuts of the default match filter, and only the columns needed
def test_prepare_default_cuts(catalog):
    stars = prepare_match_filter_catalog(catalog)
    passing = (catalog["des_yr6_g_err"] <= 0.2) & (catalog["des_yr6_r_err"] <= 0.2)
    assert stars.index.equals(catalog.index[passing])  # index of the catalog kept
    assert list(stars.columns) == ["des_yr6_g_obs", "des_yr6_r_obs", "des_yr6_g_err", "des_yr6_r_err"]
    assert len(prepare_match_filter_catalog(catalog, cuts={})) == len(catalog)  # no cut


# The given cuts replace the default ones
def test_prepare_given_cuts(catalog):
    stars = prepare_match_filter_catalog(catalog, MATCH_FILTER_GRI, cuts={"des_yr6_i_obs": (None, 22.0)})
    assert stars.index.equals(catalog.index[catalog["des_yr6_i_obs"] <= 22.0])
    assert catalog.loc[stars.index, "des_yr6_g_err"].max() > 0.2  # no error cut
    assert "des_yr6_g_err" not in stars.columns  # not a cut column any more


# With a frame: phi1/phi2 of radec_to_phi12, and only the stars inside the window
def test_prepare_stream_frame_window(catalog, gc_frame):
    window = (-5.0, 5.0, -2.0, 2.0)
    stars = prepare_match_filter_catalog(catalog, cuts={}, stream_frame=gc_frame, window=window)
    phi1, phi2 = radec_to_phi12(catalog["ra"].to_numpy(), catalog["dec"].to_numpy(), gc_frame)
    inside = (phi1 >= -5) & (phi1 <= 5) & (phi2 >= -2) & (phi2 <= 2)
    assert 0 < len(stars) == inside.sum()
    np.testing.assert_allclose(stars["phi1"], phi1[inside])
    np.testing.assert_allclose(stars["phi2"], phi2[inside])
    assert {"ra", "dec"} <= set(stars.columns) and "other" not in stars.columns

    all_stars = prepare_match_filter_catalog(catalog, cuts={}, stream_frame=gc_frame)  # no window
    assert len(all_stars) == len(catalog) and "phi1" in all_stars.columns


# Errors: missing columns (all named), a window without frame
def test_prepare_errors(catalog):
    with pytest.raises(KeyError, match="euclid_q1_VIS_obs"):
        prepare_match_filter_catalog(catalog, {**DEFAULT_MATCH_FILTER, "color_bands": [("euclid_q1", "VIS")]})
    with pytest.raises(KeyError, match="SPURIOUS"):
        prepare_match_filter_catalog(catalog, cuts={"SPURIOUS_FLAG_euclid": 0})
    with pytest.raises(KeyError, match=r"\['ra', 'dec'\]"):
        prepare_match_filter_catalog(catalog.drop(columns=["ra", "dec"]), stream_frame=object())
    with pytest.raises(ValueError, match="stream_frame"):
        prepare_match_filter_catalog(catalog, window=(-5.0, 5.0, -2.0, 2.0))

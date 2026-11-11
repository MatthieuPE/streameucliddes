import numpy as np
import pandas as pd
import pytest
from streamobs.match_filter import build_match_filter, is_in_match_filter

from streameucliddes.analysis import cmd_match_filter

pytestmark = pytest.mark.cmd_match_filter


# ===================================================
# Test build_match_filter_polygon
# ===================================================


# The polygon is build_match_filter's, built once for the same parameters
def test_match_filter_polygon_cache():
    params = {
        "distance_modulus": 16.8,
        "age": 12.0,
        "metallicity": 0.0006,
        "band_1": "g",
        "band_2": "r",
        "survey_1": "des",
        "survey_2": "des",
    }
    polygon = cmd_match_filter.build_match_filter_polygon(*params.values())
    np.testing.assert_array_equal(polygon, build_match_filter(**params))

    assert cmd_match_filter.build_match_filter_polygon(*params.values()) is polygon
    assert not polygon.flags.writeable  # shared between calls


# ===================================================
# Test match_filter_columns and select_stars_in_match_filter
# ===================================================

ISOCHRONE = {"age": 12.0, "metallicity": 0.0006}
DES_GR = cmd_match_filter.DES_GR_MATCH_FILTER
MATCH_FILTER_GRI = {
    "name": "des_gri",
    "reference_band": ("des_yr6", "g"),
    "color_bands": [("des_yr6", "r"), ("des_yr6", "i")],
}


@pytest.fixture
def stars(rng):
    """Stars uniform in colour-magnitude space, DES g, r, i and Euclid VIS, H."""
    n = 20_000
    g = rng.uniform(17.0, 25.0, n)
    r = g - rng.uniform(-0.5, 1.5, n)
    vis = rng.uniform(18.0, 26.0, n)
    return pd.DataFrame({
        "des_yr6_g_obs": g, "des_yr6_r_obs": r, "des_yr6_i_obs": r - rng.uniform(-0.5, 1.0, n),
        "euclid_q1_VIS_obs": vis, "euclid_q1_H_obs": vis - rng.uniform(-1.0, 1.5, n),
    })


def _polygon(dm, isochrone, band_2):
    return build_match_filter(
        distance_modulus=dm, age=isochrone["age"], metallicity=isochrone["metallicity"],
        band_1="g", band_2=band_2, survey_1="des", survey_2="des",
    )


# Magnitude columns: the reference band, then the colour bands
def test_match_filter_columns():
    assert cmd_match_filter.match_filter_columns() == ["euclid_q1_VIS_obs", "euclid_q1_H_obs"]
    assert cmd_match_filter.match_filter_columns(DES_GR) == ["des_yr6_g_obs", "des_yr6_r_obs"]
    assert cmd_match_filter.match_filter_columns(MATCH_FILTER_GRI) == ["des_yr6_g_obs", "des_yr6_r_obs", "des_yr6_i_obs"]


# Same stars as streamobs is_in_match_filter, in every colour of the filter
def test_select_stars_in_match_filter(stars):
    g, r, i = (stars[f"des_yr6_{band}_obs"] for band in "gri")
    in_gr = is_in_match_filter(g, r, polygon_vertices=_polygon(16.8, ISOCHRONE, "r"))
    in_gi = is_in_match_filter(g, i, polygon_vertices=_polygon(16.8, ISOCHRONE, "i"))
    assert 0 < in_gr.sum() < len(stars)

    selected = cmd_match_filter.select_stars_in_match_filter(stars, 16.8, ISOCHRONE, DES_GR)
    np.testing.assert_array_equal(selected, in_gr)
    selected_gri = cmd_match_filter.select_stars_in_match_filter(stars, 16.8, ISOCHRONE, MATCH_FILTER_GRI)
    np.testing.assert_array_equal(selected_gri, in_gr & in_gi)


# A star outside the mask is never selected
def test_select_stars_mask(stars):
    mask = np.arange(len(stars)) % 2 == 0
    selected = cmd_match_filter.select_stars_in_match_filter(stars, 16.8, ISOCHRONE, DES_GR, mask=mask)
    full = cmd_match_filter.select_stars_in_match_filter(stars, 16.8, ISOCHRONE, DES_GR)
    np.testing.assert_array_equal(selected, full & mask)
    assert mask.sum() == len(stars) // 2  # the mask given is not modified


# Defaults: the default match filter and the first isochrone of the default grid
def test_select_stars_defaults(stars):
    from streameucliddes.catalogs.isochrone_parameters import default_isochrone_grid

    default = cmd_match_filter.select_stars_in_match_filter(stars, 16.8)
    explicit = cmd_match_filter.select_stars_in_match_filter(
        stars, 16.8, default_isochrone_grid()[0], cmd_match_filter.DEFAULT_MATCH_FILTER
    )
    np.testing.assert_array_equal(default, explicit)
    assert 0 < default.sum() < len(stars)
    # The default is Euclid VIS vs VIS - H: the polygon of streamobs for those bands
    vis_h_polygon = build_match_filter(
        distance_modulus=16.8, age=default_isochrone_grid()[0]["age"],
        metallicity=default_isochrone_grid()[0]["metallicity"],
        band_1="VIS", band_2="H", survey_1="euclid", survey_2="euclid",
    )
    expected = is_in_match_filter(stars["euclid_q1_VIS_obs"], stars["euclid_q1_H_obs"], polygon_vertices=vis_h_polygon)
    np.testing.assert_array_equal(default, expected)


# Another age or Z gives another selection: the one of streamobs for that isochrone
@pytest.mark.parametrize(
    "isochrone",
    [
        {"age": 8.0, "metallicity": 0.0006},  # younger
        {"age": 13.5, "metallicity": 0.0006},  # older
        {"age": 12.0, "metallicity": 0.0001},  # more metal-poor
        {"age": 12.0, "metallicity": 0.002},  # more metal-rich
    ],
)
def test_select_stars_isochrone_changes(stars, isochrone):
    g, r = stars["des_yr6_g_obs"], stars["des_yr6_r_obs"]
    reference = cmd_match_filter.select_stars_in_match_filter(stars, 16.8, ISOCHRONE, DES_GR)
    selected = cmd_match_filter.select_stars_in_match_filter(stars, 16.8, isochrone, DES_GR)
    np.testing.assert_array_equal(selected, is_in_match_filter(g, r, polygon_vertices=_polygon(16.8, isochrone, "r")))
    changed = np.sum(selected != reference)
    assert changed > 0.05 * reference.sum()  # not a few stars at the polygon edge
    assert (selected & reference).sum() > 0  # the same old population: the filters overlap


# A fainter distance modulus moves the filter to fainter stars: its bright end by the
# difference of distance modulus (its width also grows with the photometric errors)
def test_select_stars_distance_modulus(stars):
    g = stars["des_yr6_g_obs"].to_numpy()
    near = cmd_match_filter.select_stars_in_match_filter(stars, 16.0, ISOCHRONE, DES_GR)
    far = cmd_match_filter.select_stars_in_match_filter(stars, 18.0, ISOCHRONE, DES_GR)
    assert g[far].min() - g[near].min() == pytest.approx(2.0, abs=0.1)
    assert (near != far).sum() > 0.5 * near.sum()

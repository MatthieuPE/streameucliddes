import warnings

import numpy as np
import pandas as pd
import pytest
from streamobs.match_filter import build_match_filter, is_in_match_filter

from streameucliddes.analysis import cmd_match_filter
from streameucliddes.catalogs.isochrone_parameters import DEFAULT_ISOCHRONE_MODEL, feh_to_z
from streameucliddes.analysis.cmd_match_filter import (
    difference_hess,
    hess_diagram,
    isochrone_cmd_track,
    match_filter_cmd_columns,
    match_filter_cmd_limits,
    match_filter_polygons,
    mean_off_density,
    mean_off_hess,
    on_off_hess,
)

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


# One streamobs polygon per colour band, in their order; default: the default isochrone
# and match filter (Euclid VIS - H)
def test_match_filter_polygons():
    polygons = cmd_match_filter.match_filter_polygons(17.0, ISOCHRONE, MATCH_FILTER_GRI)
    assert len(polygons) == 2
    np.testing.assert_allclose(polygons[0], _polygon(17.0, ISOCHRONE, "r"))
    np.testing.assert_allclose(polygons[1], _polygon(17.0, ISOCHRONE, "i"))

    default_isochrone = cmd_match_filter.default_isochrone_grid()[0]
    default_polygon, = cmd_match_filter.match_filter_polygons(17.0)
    np.testing.assert_allclose(default_polygon, build_match_filter(
        distance_modulus=17.0, age=default_isochrone["age"], metallicity=default_isochrone["metallicity"],
        band_1="VIS", band_2="H", survey_1="euclid", survey_2="euclid",
    ))


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
        {"age": 12.0, "metallicity": 0.001},  # more metal-rich (the top of the Marigo et al. 2017 grid)
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


# Stubs of the candidate investigation (docs/dev/candidate_investigation_plan.md)


# The track at dm 20 is the track at dm 15, 5 mag fainter, same colour
def test_isochrone_track_shifts_with_distance_modulus():
    near = isochrone_cmd_track(15.0, 13.0, 0.0002)
    far = isochrone_cmd_track(20.0, 13.0, 0.0002)
    np.testing.assert_allclose(far["mag"], near["mag"] + 5.0)
    np.testing.assert_array_equal(far["color"], near["color"])
    assert far["distance_modulus"] == 20.0 and far["feh"] is None


# [Fe/H] = -1.9: Z of convert_FeH_to_z (~0.0002), the same track as with that Z
def test_isochrone_track_feh_gives_z():
    track = isochrone_cmd_track(17.0, 13.0, feh=-1.9)
    assert track["metallicity"] == pytest.approx(0.0002, rel=0.05) and track["feh"] == -1.9
    with_z = isochrone_cmd_track(17.0, 13.0, track["metallicity"])
    np.testing.assert_array_equal(track["mag"], with_z["mag"])


# Neither or both of metallicity and feh: ValueError
def test_isochrone_track_metallicity_or_feh():
    with pytest.raises(ValueError, match="exactly one"):
        isochrone_cmd_track(17.0, 13.0)
    with pytest.raises(ValueError, match="exactly one"):
        isochrone_cmd_track(17.0, 13.0, 0.0002, feh=-1.9)


# The main sequence of the track (fainter than the turn-off, to VIS 26) is inside the match-filter polygon
# of the same isochrone and distance modulus: both come from the same model
def test_isochrone_track_inside_polygon():
    isochrone = {"age": 13.0, "metallicity": 0.0002}
    track = isochrone_cmd_track(17.0, **isochrone)
    (polygon,) = cmd_match_filter.match_filter_polygons(17.0, isochrone)
    main_sequence = (track["mag"] > 17.0 + 3.5) & (track["mag"] < 26.0)  # turn-off at absolute VIS ~ 3.5
    inside = is_in_match_filter(track["mag"][main_sequence], track["mag"][main_sequence] - track["color"][main_sequence],
                                polygon_vertices=polygon)
    assert main_sequence.sum() > 20 and inside.mean() > 0.9


# No area: density = counts, error = sqrt(counts); the bins and labels of the columns; NaN magnitudes left out
def test_hess_counts_without_area():
    stars = pd.DataFrame({"euclid_dr1_VIS_obs": [20.05, 20.05, 21.05, np.nan],
                          "euclid_dr1_H_obs": [19.95, 19.95, 20.55, 20.0]})
    hess = hess_diagram(stars, ("euclid_dr1_VIS_obs", "euclid_dr1_H_obs"), color_bins=[0.0, 0.2, 0.6],
                        mag_bins=[20.0, 21.0, 22.0])
    np.testing.assert_array_equal(hess["counts"], [[2, 0], [0, 1]])
    np.testing.assert_array_equal(hess["density"], hess["counts"])
    np.testing.assert_allclose(hess["error"], np.sqrt(hess["counts"]))
    assert hess["area_deg2"] is None and hess["n_stars"] == 3
    assert (hess["color_label"], hess["mag_label"]) == ("VIS - H", "VIS")


# One star in a 2 deg^2 region: density 0.5 in one bin, 0 elsewhere; default bins
def test_hess_single_star_density():
    stars = pd.DataFrame({"euclid_q1_VIS_obs": [22.0], "euclid_q1_H_obs": [21.7]})
    hess = hess_diagram(stars, ("euclid_q1_VIS_obs", "euclid_q1_H_obs"), area_deg2=2.0)
    assert hess["density"].sum() == 0.5 and np.count_nonzero(hess["density"]) == 1
    assert hess["error"].max() == pytest.approx(0.5) and hess["area_deg2"] == 2.0
    assert hess["density"].shape == (hess["color_edges"].size - 1, hess["mag_edges"].size - 1)


# Only the stars of mask are counted
def test_hess_mask_selects_stars():
    stars = pd.DataFrame({"euclid_q1_VIS_obs": [20.0, 21.0, 22.0], "euclid_q1_H_obs": [19.5, 20.5, 21.5]})
    hess = hess_diagram(stars, ("euclid_q1_VIS_obs", "euclid_q1_H_obs"), mask=[True, False, True])
    assert hess["n_stars"] == 2


# area_deg2 <= 0: ValueError
def test_hess_rejects_non_positive_area():
    stars = pd.DataFrame({"euclid_q1_VIS_obs": [20.0], "euclid_q1_H_obs": [19.5]})
    with pytest.raises(ValueError, match="area_deg2"):
        hess_diagram(stars, ("euclid_q1_VIS_obs", "euclid_q1_H_obs"), area_deg2=0.0)


# On and off from the same uniform population, areas 1 and 3: on - off consistent with 0 (pulls ~ N(0, 1));
# the errors in quadrature, the area of the on region
def test_difference_hess_consistent_with_zero():
    rng = np.random.default_rng(4)
    columns = ("euclid_q1_VIS_obs", "euclid_q1_H_obs")
    bins = {"color_bins": np.linspace(-1.0, 1.5, 11), "mag_bins": np.linspace(18.0, 26.0, 9)}

    def population(n):
        vis = rng.uniform(18.0, 26.0, n)
        return pd.DataFrame({columns[0]: vis, columns[1]: vis - rng.uniform(-1.0, 1.5, n)})

    on = hess_diagram(population(40_000), columns, area_deg2=1.0, **bins)
    off = hess_diagram(population(120_000), columns, area_deg2=3.0, **bins)
    difference = difference_hess(on, off)
    pulls = difference["density"] / difference["error"]
    assert abs(pulls.mean()) < 3 / np.sqrt(pulls.size) and 0.7 < pulls.std() < 1.3
    np.testing.assert_allclose(difference["error"], np.hypot(on["error"], off["error"]))
    assert difference["area_deg2"] == 1.0
    assert difference["n_stars"] == pytest.approx(on["n_stars"] - off["n_stars"] / 3.0)  # excess of the on region
    np.testing.assert_allclose(difference["counts"], difference["density"] * 1.0)


# Different bin edges, or a diagram without area: ValueError
def test_difference_hess_rejects_different_bins():
    stars = pd.DataFrame({"euclid_q1_VIS_obs": [20.0], "euclid_q1_H_obs": [19.5]})
    columns = ("euclid_q1_VIS_obs", "euclid_q1_H_obs")
    on = hess_diagram(stars, columns, area_deg2=1.0)
    with pytest.raises(ValueError, match="mag_edges"):
        difference_hess(on, hess_diagram(stars, columns, area_deg2=1.0, mag_bins=np.linspace(17.0, 27.0, 11)))
    with pytest.raises(ValueError, match="area_deg2"):
        difference_hess(on, hess_diagram(stars, columns))


# The mean of the two densities and its error; where one off region has no area (NaN), the other alone; where
# neither has, NaN; the number of off regions in each mean
def test_mean_off_density():
    density, error, n_regions = mean_off_density([2.0, np.nan, 3.0, np.nan], [0.4, np.nan, 0.6, np.nan],
                                                 [1.0, 4.0, np.nan, np.nan], [0.3, 0.8, np.nan, np.nan])
    np.testing.assert_allclose(density[:3], [1.5, 4.0, 3.0])
    np.testing.assert_allclose(error[:3], [0.5 * np.hypot(0.4, 0.3), 0.8, 0.6])
    assert np.isnan(density[3]) and np.isnan(error[3])
    np.testing.assert_array_equal(n_regions, [2, 1, 1, 0])


# The off Hess diagram: the mean of the densities of the two off regions (2 and 1 stars/deg²: 1.5, not 6 stars
# / 5 deg²), their errors combined, the counts, stars and areas summed; an off region without area: the other
# alone; different bins or no area at all: ValueError
def test_mean_off_hess():
    columns = ("euclid_q1_VIS_obs", "euclid_q1_H_obs")
    bins = {"color_bins": [0.0, 1.0], "mag_bins": [20.0, 21.0]}

    def stars(n):
        return pd.DataFrame({columns[0]: np.full(n, 20.5), columns[1]: np.full(n, 20.0)})

    below = hess_diagram(stars(2), columns, area_deg2=1.0, **bins)
    above = hess_diagram(stars(4), columns, area_deg2=4.0, **bins)
    off = mean_off_hess(below, above)
    assert off["density"][0, 0] == pytest.approx(1.5) and off["n_off_regions"] == 2
    assert off["error"][0, 0] == pytest.approx(0.5 * np.hypot(np.sqrt(2) / 1.0, np.sqrt(4) / 4.0))
    assert off["counts"][0, 0] == 6 and off["n_stars"] == 6 and off["area_deg2"] == 5.0
    alone = mean_off_hess(hess_diagram(stars(2), columns, **bins), above)
    assert alone["density"][0, 0] == pytest.approx(1.0) and alone["n_off_regions"] == 1
    with pytest.raises(ValueError, match="mag_edges"):
        mean_off_hess(below, hess_diagram(stars(4), columns, area_deg2=4.0, color_bins=[0.0, 1.0], mag_bins=[19.0, 21.0]))
    with pytest.raises(ValueError, match="Neither"):
        mean_off_hess(hess_diagram(stars(2), columns, **bins), hess_diagram(stars(4), columns, **bins))


# ===================================================
# Test match_filter_cmd_columns, match_filter_cmd_limits and on_off_hess
# ===================================================


# One CMD per colour band: (reference, colour band); its limits: the polygon plus the margin
def test_match_filter_cmd_columns_and_limits():
    assert match_filter_cmd_columns(MATCH_FILTER_GRI) == [("des_yr6_g_obs", "des_yr6_r_obs"),
                                                          ("des_yr6_g_obs", "des_yr6_i_obs")]
    limits = match_filter_cmd_limits(18.0, ISOCHRONE, MATCH_FILTER_GRI, margin_mag=0.3)
    for (color_lim, mag_lim), polygon in zip(limits, match_filter_polygons(18.0, ISOCHRONE, MATCH_FILTER_GRI)):
        assert color_lim == pytest.approx((polygon[:, 0].min() - 0.3, polygon[:, 0].max() + 0.3))
        assert mag_lim == pytest.approx((polygon[:, 1].min() - 0.3, polygon[:, 1].max() + 0.3))


# One off region: its own diagram; two: their mean (mean_off_hess); then the difference
def test_on_off_hess(stars, rng):
    mag_cols = ("euclid_q1_VIS_obs", "euclid_q1_H_obs")
    region = rng.integers(0, 3, len(stars))
    on, below, above = (region == i for i in range(3))
    on_hess, off_hess, difference = on_off_hess(stars, mag_cols, on, 1.0, [below], [2.0])
    np.testing.assert_array_equal(on_hess["density"], hess_diagram(stars, mag_cols, mask=on, area_deg2=1.0)["density"])
    below_hess = hess_diagram(stars, mag_cols, mask=below, area_deg2=2.0)
    np.testing.assert_array_equal(off_hess["density"], below_hess["density"])
    np.testing.assert_array_equal(difference["density"], difference_hess(on_hess, off_hess)["density"])

    _, off_hess, _ = on_off_hess(stars, mag_cols, on, 1.0, [below, above], [2.0, 0.0])  # one without area: left out
    below_hess = hess_diagram(stars, mag_cols, mask=below, area_deg2=2.0)
    np.testing.assert_array_equal(off_hess["density"], below_hess["density"])
    _, off_hess, _ = on_off_hess(stars, mag_cols, on, 1.0, [below, above], [2.0, 3.0])
    expected = mean_off_hess(hess_diagram(stars, mag_cols, mask=below, area_deg2=2.0),
                             hess_diagram(stars, mag_cols, mask=above, area_deg2=3.0))
    np.testing.assert_array_equal(off_hess["density"], expected["density"])
    with pytest.raises(ValueError, match="One or two"):
        on_off_hess(stars, mag_cols, on, 1.0, [below, above, on], [1.0, 1.0, 1.0])


# ===================================================
# Test the isochrone model
# ===================================================

DOTTER = {"age": 13.0, "metallicity": 0.0002, "isochrone_model": "Dotter2016"}


# The "isochrone_model" of the isochrone reaches streamobs: the polygon of build_match_filter with that isochrone_model,
# another than the default one, cached apart
def test_match_filter_polygons_model():
    polygon, = match_filter_polygons(17.0, DOTTER)
    expected = build_match_filter(distance_modulus=17.0, age=13.0, metallicity=0.0002, band_1="VIS", band_2="H",
                                  survey_1="euclid", survey_2="euclid", isochrone_model="Dotter2016")
    np.testing.assert_array_equal(polygon, expected)
    default, = match_filter_polygons(17.0, {"age": 13.0, "metallicity": 0.0002})
    assert not np.array_equal(polygon, default)
    explicit, = match_filter_polygons(17.0, {"age": 13.0, "metallicity": 0.0002, "isochrone_model": "Marigo2017"})
    assert explicit is default  # the default model, the same cached polygon


# The track of the model, with [Fe/H] on its scale; the model in the track
def test_isochrone_cmd_track_model():
    track = isochrone_cmd_track(17.0, 13.0, feh=-1.9, isochrone_model="Dotter2016")
    assert track["isochrone_model"] == "Dotter2016"
    assert track["metallicity"] == pytest.approx(feh_to_z(-1.9, "Dotter2016"))
    default = isochrone_cmd_track(17.0, 13.0, feh=-1.9)
    assert default["isochrone_model"] == DEFAULT_ISOCHRONE_MODEL
    assert not np.array_equal(track["color"], default["color"])


# A warning when the grid of the model has no isochrone near the one asked (the DES Dotter2016 grid: 10 and 12 Gyr,
# Z 1e-4 and 2e-4; Z below 1e-4 for Marigo2017); none inside a full grid
def test_isochrone_grid_warning():
    cmd_match_filter._check_isochrone_grid.cache_clear()
    with pytest.warns(UserWarning, match="No Dotter2016 isochrone near 13 Gyr, Z = 0.0006 for des"):
        match_filter_polygons(17.0, {"age": 13.0, "metallicity": 0.0006, "isochrone_model": "Dotter2016"}, DES_GR)
    with pytest.warns(UserWarning, match="Z = 3.5e-05 for euclid: ugali uses the nearest of its grid, 13 Gyr, Z = "):
        isochrone_cmd_track(17.0, 13.0, metallicity=3.5e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        match_filter_polygons(17.0, DOTTER)  # the Euclid Dotter2016 grid: full
        isochrone_cmd_track(17.0, 12.5, metallicity=0.0004)

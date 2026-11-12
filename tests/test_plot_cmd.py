"""Tests of plotting/plot_cmd.py (plot_match_filter), with a synthetic Euclid VIS/H catalog."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from streameucliddes.analysis.catalog_preparation import default_error_cuts, prepare_match_filter_catalog
from streameucliddes.analysis.cmd_match_filter import (
    DEFAULT_MATCH_FILTER,
    match_filter_polygons,
    select_stars_in_match_filter,
)
from streameucliddes.catalogs.isochrone_parameters import default_isochrone_grid
from streameucliddes.plotting import plot_cmd
from streameucliddes.plotting.plot_cmd import plot_match_filter
from streameucliddes.stream_frames import phi12_to_radec

pytestmark = pytest.mark.plot_cmd

DM = 17.0
ISOCHRONE = {"age": 12.0, "metallicity": 0.0003}
VIS, H, J = "euclid_q1_VIS", "euclid_q1_H", "euclid_q1_J"
COLOR_LIM = (-0.6, 0.9)  # VIS - H
MAG_LIM = (19.0, 26.0)  # VIS
VIS_H_J_MATCH_FILTER = {
    "name": "vis_h_j",
    "reference_band": ("euclid_q1", "VIS"),
    "color_bands": [("euclid_q1", "H"), ("euclid_q1", "J")],
}


@pytest.fixture(scope="module")
def catalog(gc_frame):
    """Stars with ra, dec, Euclid VIS, H, J and errors; a tenth with errors above the default cut."""
    rng = np.random.default_rng(3)
    n = 20_000
    phi1, phi2 = rng.uniform(-10.0, 10.0, n), rng.uniform(-3.0, 3.0, n)
    ra, dec = phi12_to_radec(phi1, phi2, gc_frame)
    vis = rng.uniform(18.0, 27.0, n)
    errors = np.where(np.arange(n) % 10 == 0, 0.5, 0.02)
    return pd.DataFrame({
        "ra": ra, "dec": dec,
        f"{VIS}_obs": vis, f"{H}_obs": vis - rng.uniform(-1.0, 1.5, n), f"{J}_obs": vis - rng.uniform(-1.0, 1.5, n),
        f"{VIS}_err": errors, f"{H}_err": errors, f"{J}_err": errors,
    })


def _expected_counts(stars, color_column, color_lim, mag_lim):
    """2D histogram of (VIS - colour band, VIS), with the bins of plot_CMD."""
    vis = stars[f"{VIS}_obs"].to_numpy()
    counts, _, _ = np.histogram2d(vis - stars[color_column].to_numpy(), vis,
                                  bins=[np.arange(*color_lim, 0.01), np.arange(*mag_lim, 0.1)])
    return counts.T


def _polygon_line(ax):
    line, = [line for line in ax.lines if line.get_label() == "Match filter"]
    return line.get_xydata()


# ===================================================
# Test plot_match_filter
# ===================================================


# A catalog is prepared with the default cuts: the CMD is the histogram of colour
# VIS - H against VIS of the stars kept, the polygon is the one of the selection, closed,
# and the title gives the number of stars selected
def test_plot_match_filter_catalog(catalog):
    fig, axes = plot_match_filter(DM, ISOCHRONE, catalog=catalog, color_lim=COLOR_LIM, mag_lim=MAG_LIM)
    ax, = axes
    stars = prepare_match_filter_catalog(catalog)
    assert len(stars) < len(catalog)  # the default cuts removed stars
    np.testing.assert_array_equal(ax.images[0].get_array(), _expected_counts(stars, f"{H}_obs", COLOR_LIM, MAG_LIM))

    polygon, = match_filter_polygons(DM, ISOCHRONE)
    np.testing.assert_array_equal(_polygon_line(ax), np.vstack([polygon, polygon[:1]]))
    n_selected = select_stars_in_match_filter(stars, DM, ISOCHRONE).sum()
    assert n_selected > 0
    assert ax.get_title() == f"dm 17.0 (25.1 kpc), 12 Gyr, Z = 0.0003\n{n_selected} stars in the filter"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("VIS - H", "VIS")
    assert ax.get_xlim() == pytest.approx(COLOR_LIM)
    assert ax.get_ylim() == pytest.approx(MAG_LIM[::-1])  # bright at the top
    plt.close(fig)


# Prepared stars are used as they are (no cut), and give the same figure as the catalog
# prepared without cut
def test_plot_match_filter_stars(catalog):
    no_cut = {column: (None, None) for column in default_error_cuts()}  # default error cuts removed
    stars = prepare_match_filter_catalog(catalog, cuts=no_cut)
    fig, (ax,) = plot_match_filter(DM, ISOCHRONE, stars=stars, color_lim=COLOR_LIM, mag_lim=MAG_LIM)
    fig_catalog, (ax_catalog,) = plot_match_filter(DM, ISOCHRONE, catalog=catalog, cuts=no_cut,
                                                   color_lim=COLOR_LIM, mag_lim=MAG_LIM)
    assert len(stars) == len(catalog)
    np.testing.assert_array_equal(ax.images[0].get_array(), ax_catalog.images[0].get_array())
    assert ax.get_title() == ax_catalog.get_title()
    plt.close(fig)
    plt.close(fig_catalog)


# The arguments of prepare_match_filter_catalog reach it: only the stars of the window
def test_plot_match_filter_window(catalog, gc_frame):
    window = (-2.0, 2.0, -1.0, 1.0)
    fig, (ax,) = plot_match_filter(DM, ISOCHRONE, catalog=catalog, stream_frame=gc_frame, window=window,
                                   color_lim=COLOR_LIM, mag_lim=MAG_LIM)
    stars = prepare_match_filter_catalog(catalog, stream_frame=gc_frame, window=window)
    assert len(stars) < len(prepare_match_filter_catalog(catalog))
    np.testing.assert_array_equal(ax.images[0].get_array(), _expected_counts(stars, f"{H}_obs", COLOR_LIM, MAG_LIM))
    plt.close(fig)


# The polygon follows the distance modulus and the isochrone
def test_plot_match_filter_changes(catalog):
    fig, (ax,) = plot_match_filter(DM, ISOCHRONE)
    fig_far, (ax_far,) = plot_match_filter(DM + 1.0, ISOCHRONE)
    fig_young, (ax_young,) = plot_match_filter(DM, {"age": 3.0, "metallicity": 0.0003})
    polygon_far, = match_filter_polygons(DM + 1.0, ISOCHRONE)
    polygon_young, = match_filter_polygons(DM, {"age": 3.0, "metallicity": 0.0003})
    np.testing.assert_array_equal(_polygon_line(ax_far)[:-1], polygon_far)
    np.testing.assert_array_equal(_polygon_line(ax_young)[:-1], polygon_young)
    assert _polygon_line(ax_far)[:, 1].min() > _polygon_line(ax)[:, 1].min()  # fainter turn-off
    assert not np.array_equal(_polygon_line(ax_young), _polygon_line(ax))
    for figure in (fig, fig_far, fig_young):
        plt.close(figure)


# Without stars: the polygon only, default isochrone and match filter, limits from the
# polygon plus the margin
def test_plot_match_filter_no_stars():
    fig, (ax,) = plot_match_filter(DM)
    polygon, = match_filter_polygons(DM, default_isochrone_grid()[0], DEFAULT_MATCH_FILTER)
    np.testing.assert_array_equal(_polygon_line(ax)[:-1], polygon)
    assert not ax.images
    margin = plot_cmd.MATCH_FILTER_MARGIN_MAG
    assert ax.get_xlim() == pytest.approx((polygon[:, 0].min() - margin, polygon[:, 0].max() + margin))
    assert ax.get_ylim() == pytest.approx((polygon[:, 1].max() + margin, polygon[:, 1].min() - margin))
    assert "stars in the filter" not in ax.get_title()
    plt.close(fig)


# One panel per colour band, with its polygon and the histogram of its colour; the title
# counts the stars selected by all the colour bands
def test_plot_match_filter_two_colour_bands(catalog):
    fig, axes = plot_match_filter(DM, ISOCHRONE, VIS_H_J_MATCH_FILTER, catalog=catalog,
                                  color_lim=COLOR_LIM, mag_lim=MAG_LIM)
    stars = prepare_match_filter_catalog(catalog, VIS_H_J_MATCH_FILTER)
    polygons = match_filter_polygons(DM, ISOCHRONE, VIS_H_J_MATCH_FILTER)
    assert len(axes) == 2
    for ax, polygon, column, band in zip(axes, polygons, [f"{H}_obs", f"{J}_obs"], ["H", "J"]):
        np.testing.assert_array_equal(_polygon_line(ax)[:-1], polygon)
        np.testing.assert_array_equal(ax.images[0].get_array(), _expected_counts(stars, column, COLOR_LIM, MAG_LIM))
        assert ax.get_xlabel() == f"VIS - {band}"
    n_selected = select_stars_in_match_filter(stars, DM, ISOCHRONE, VIS_H_J_MATCH_FILTER).sum()
    assert fig.get_suptitle().endswith(f"\n{n_selected} stars in the filter")
    plt.close(fig)


# Given axes and title are used; no star in the ranges is not an error (linear scale)
def test_plot_match_filter_given_axes(catalog):
    fig, ax = plt.subplots()
    returned_fig, (returned_ax,) = plot_match_filter(DM, ISOCHRONE, catalog=catalog, axes=ax, title="My filter",
                                                     color_lim=(5.0, 6.0), mag_lim=MAG_LIM)
    assert returned_fig is fig and returned_ax is ax
    assert ax.get_title() == "My filter"
    assert ax.images[0].get_array().sum() == 0
    plt.close(fig)


def test_plot_match_filter_errors(catalog):
    stars = prepare_match_filter_catalog(catalog)
    with pytest.raises(ValueError, match="not both"):
        plot_match_filter(DM, catalog=catalog, stars=stars)
    with pytest.raises(ValueError, match="cuts"):
        plot_match_filter(DM, stars=stars, cuts={})
    with pytest.raises(ValueError, match="window"):
        plot_match_filter(DM, window=(-1.0, 1.0, -1.0, 1.0))
    with pytest.raises(KeyError, match=f"{H}_obs"):
        plot_match_filter(DM, stars=stars.drop(columns=f"{H}_obs"))
    with pytest.raises(KeyError, match=f"{J}_obs"):
        plot_match_filter(DM, match_filter=VIS_H_J_MATCH_FILTER, catalog=catalog.drop(columns=f"{J}_obs"))
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="1 axes for 2 colour bands"):
        plot_match_filter(DM, match_filter=VIS_H_J_MATCH_FILTER, axes=ax)
    plt.close("all")

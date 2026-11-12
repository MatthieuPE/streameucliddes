"""Tests of plotting/plot_generic_search.py, with uniform mock stars across ra = 0."""

import matplotlib

matplotlib.use("Agg")

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from streameucliddes.analysis.cmd_match_filter import match_filter_polygons
from streameucliddes.analysis.generic_stream_search import (
    generic_search_cmd_stars,
    generic_search_single_dm,
    prepare_generic_search,
)
from streameucliddes.plotting import plot_generic_search
from streameucliddes.plotting.plot_generic_search import (
    plot_generic_search_residual_map,
    plot_generic_search_result,
)

pytestmark = pytest.mark.plot_generic_search

NSIDE = 32


@pytest.fixture(scope="module")
def search():
    rng = np.random.default_rng(8)
    n = 100_000
    vis = rng.uniform(18.0, 26.0, n)
    catalog = pd.DataFrame({
        "ra": rng.uniform(-15.0, 15.0, n) % 360.0, "dec": rng.uniform(-30.0, -10.0, n),
        "euclid_q1_VIS_obs": vis, "euclid_q1_H_obs": vis - rng.uniform(-1.0, 1.5, n),
        "euclid_q1_VIS_err": np.full(n, 0.02), "euclid_q1_H_err": np.full(n, 0.02),
    })
    return prepare_generic_search(catalog, nside=NSIDE)


@pytest.fixture(scope="module")
def result(search):
    return generic_search_single_dm(distance_modulus=17.0, search=search, smoothing_deg=2.0)


def _clim(ax):
    """Colour limits of the HEALPix map drawn in a skyproj panel."""
    mappable, = [artist for artist in ax.collections + ax.images if artist.get_array() is not None]
    return mappable.get_clim()


# Four panels: the match filter of the map on the CMD of the stars, with the number of stars
# the map selects; the map, the background (same colour scale: percentiles of the map) and
# the corrected map (its own); the default title
def test_plot_result(search, result):
    cmd_stars = generic_search_cmd_stars(search, max_stars=5000)
    fig, axes = plot_generic_search_result(result, cmd_stars=cmd_stars)
    assert len(axes) == 4
    polygon, = match_filter_polygons(17.0, result["isochrone"], result["match_filter"])
    line, = [line for line in axes[0].lines if line.get_label() == "Match filter"]
    np.testing.assert_array_equal(line.get_xydata()[:-1], polygon)
    assert axes[0].images  # the CMD of the stars
    assert axes[0].get_title() == f"Match filter: {result['n_selected']:,} stars selected"

    low, high = plot_generic_search.DENSITY_PERCENTILES
    density_scale = np.percentile(result["density"].compressed(), [low, high])
    np.testing.assert_allclose(_clim(axes[1]), density_scale)
    np.testing.assert_allclose(_clim(axes[2]), density_scale)
    np.testing.assert_allclose(_clim(axes[3]), np.percentile(result["residual"].compressed(), [low, high]))
    assert [ax.get_title() for ax in axes[1:]] == [title for _, title in plot_generic_search.MAP_PANELS]
    assert fig.get_suptitle().startswith("dm 17.0 (25.1 kpc), ")
    plt.close(fig)


# Without stars: the match filter alone; the given title
def test_plot_result_no_stars(result):
    fig, axes = plot_generic_search_result(result, title="My map")
    assert not axes[0].images and fig.get_suptitle() == "My map"
    plt.close(fig)


# The footprint ranges, around its centre: across ra = 0, -15 to 15, not 0 to 360
def test_footprint_ranges(search):
    lon_range, lat_range = plot_generic_search._footprint_lonlat_ranges(search.footprint_mask, nest=False,
                                                                       margin_deg=1.0)
    resolution = np.degrees(hp.nside2resol(NSIDE))
    np.testing.assert_allclose(lon_range, (-16.0, 16.0), atol=2 * resolution)
    np.testing.assert_allclose(lat_range, (-31.0, -9.0), atol=2 * resolution)


# Frames of a gif: the panels stay in place from one distance modulus to the next
def test_plot_result_layout(search):
    positions = []
    for dm in (15.0, 21.0):
        result = generic_search_single_dm(distance_modulus=dm, search=search, smoothing_deg=2.0)
        fig, axes = plot_generic_search_result(result, color_lim=(-1.0, 1.5), mag_lim=(15.0, 27.0))
        fig.canvas.draw()
        positions.append([ax.get_position().bounds for ax in axes])
        plt.close(fig)
    np.testing.assert_allclose(positions[0], positions[1])


# ===================================================
# Test plot_generic_search_residual_map
# ===================================================


def _track_lines(ax):
    return [line for line in ax.lines if line.get_linestyle() == "--"]


# The corrected map, with the colour scale of the fourth panel; each track a faint dashed line
# (one line per side of the map edge at ra = 180), labelled only if it crosses the footprint
def test_plot_residual_map(result):
    dec = np.linspace(-35.0, -5.0, 50)
    tracks = [
        ("Inside", np.full(50, 5.0), dec),  # across the footprint
        ("Outside", np.full(50, 100.0), dec),  # outside
        ("Edge", np.linspace(170.0, 190.0, 50), np.full(50, -20.0)),  # across the map edge
    ]
    fig, ax = plot_generic_search_residual_map(result, tracks=tracks)
    low, high = plot_generic_search.DENSITY_PERCENTILES
    np.testing.assert_allclose(_clim(ax), np.percentile(result["residual"].compressed(), [low, high]))
    lines = _track_lines(ax)
    assert len(lines) == 4  # Edge: two parts
    style = plot_generic_search.KNOWN_TRACK_STYLE
    assert all(line.get_alpha() == style["alpha"] < 1 and line.get_linewidth() == style["lw"] for line in lines)
    names = {text.get_text() for text in ax.texts}
    assert "Inside" in names and not names & {"Outside", "Edge"}
    assert ax.get_title() == plot_generic_search.MAP_PANELS[2][1]
    assert fig.get_suptitle().startswith("dm 17.0 (25.1 kpc), ")
    plt.close(fig)


# Without tracks: the map alone; the given title
def test_plot_residual_map_no_tracks(result):
    fig, ax = plot_generic_search_residual_map(result, title="My map")
    assert not _track_lines(ax) and not ax.texts and fig.get_suptitle() == "My map"
    plt.close(fig)


# Frames of a gif: the map stays in place from one distance modulus to the next
def test_plot_residual_map_layout(search):
    tracks = [("Inside", np.full(50, 5.0), np.linspace(-35.0, -5.0, 50))]
    positions = []
    for dm in (15.0, 21.0):
        result = generic_search_single_dm(distance_modulus=dm, search=search, smoothing_deg=2.0)
        fig, ax = plot_generic_search_residual_map(result, tracks=tracks)
        fig.canvas.draw()
        positions.append(ax.get_position().bounds)
        plt.close(fig)
    np.testing.assert_allclose(positions[0], positions[1])

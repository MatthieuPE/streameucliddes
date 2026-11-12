"""Tests of plotting/plot_sky_density.py, with uniform mock stars across ra = 0."""

import matplotlib

matplotlib.use("Agg")

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import pytest

from streameucliddes.analysis.star_density_maps import star_density_healpix
from streameucliddes.plotting.plot_sky_density import plot_2D_density, plot_2D_density_from_hp

pytestmark = pytest.mark.plot_sky_density

NSIDE = 32


@pytest.fixture(scope="module")
def stars():
    rng = np.random.default_rng(3)
    n = 20_000
    return pd.DataFrame({"ra": rng.uniform(-15.0, 15.0, n) % 360.0, "dec": rng.uniform(-30.0, -10.0, n)})


def _drawn_values(ax):
    """Values of the map drawn by skyproj (one mesh per call)."""
    meshes = [c for c in ax.collections if c.get_array() is not None]
    assert len(meshes) == 1
    return ma.masked_invalid(meshes[0].get_array())


@pytest.mark.parametrize("smoothing", [False, True])
def test_same_figure_as_from_hp(stars, smoothing):
    fig, ax = plot_2D_density(stars, nside=NSIDE, smoothing=smoothing, smoothing_scale_degrees=2.0)
    hp_map, *_ = star_density_healpix(
        stars, nside=NSIDE, smoothing=smoothing, smoothing_scale_degrees=2.0
    )
    fig_hp, ax_hp = plot_2D_density_from_hp(hp_map.filled(hp.UNSEEN), nest=False)
    drawn, drawn_hp = _drawn_values(ax), _drawn_values(ax_hp)
    np.testing.assert_array_equal(ma.getmaskarray(drawn), ma.getmaskarray(drawn_hp))
    np.testing.assert_array_equal(drawn.compressed(), drawn_hp.compressed())
    assert ax.get_xlim() == ax_hp.get_xlim() and ax.get_ylim() == ax_hp.get_ylim()
    plt.close("all")


def test_empty_pixels_blank(stars):
    fig, ax = plot_2D_density(stars, nside=NSIDE, lon_range=(-60, 60), lat_range=(-60, 30))
    drawn = _drawn_values(ax)
    assert drawn.mask.any()  # no star outside the mock patch: blank, not 0
    assert drawn.min() >= 1
    plt.close("all")

"""Tests of plotting/plot_gc_dwarf_optimization.py, with uniform mock stars and a cluster on the isochrone."""

import matplotlib

matplotlib.use("Agg")

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from matplotlib.patches import Ellipse

from streameucliddes.analysis.cmd_match_filter import isochrone_cmd_track
from streameucliddes.analysis.gc_dwarf_optimization import (
    compare_selections,
    object_regions,
    stars_in_match_filter,
)
from streameucliddes.plotting.plot_gc_dwarf_optimization import (
    plot_object_cmd,
    plot_object_hess,
    plot_object_map,
    plot_selection_comparison,
)

pytestmark = pytest.mark.plot_gc_dwarf_optimization

DM = 17.0
ISOCHRONE = {"age": 12.0, "metallicity": 0.0003}
VIS, H, J = "euclid_q1_VIS", "euclid_q1_H", "euclid_q1_J"
VIS_H_J = {"name": "vis_h_j", "reference_band": ("euclid_q1", "VIS"),
           "color_bands": [("euclid_q1", "H"), ("euclid_q1", "J")]}
OBJECT = pd.Series({"key": "dwarf_a", "name": "Dwarf A", "ra": 50.0, "dec": -30.0, "radius_arcmin": 6.0,
                    "ellipticity": 0.4, "position_angle": 30.0, "distance_modulus": DM, "feh": -2.0})


@pytest.fixture(scope="module")
def catalog():
    """Uniform field stars, and cluster stars near the centre on the VIS - H and VIS - J isochrones."""
    rng = np.random.default_rng(2)
    n_field, n_cluster = 60_000, 1_500
    separation = np.concatenate([np.degrees(np.arccos(1 - rng.random(n_field) * (1 - np.cos(np.radians(1.2))))),
                                 np.abs(rng.normal(0.0, 0.08, n_cluster))])
    angle = rng.uniform(0.0, 360.0, n_field + n_cluster)
    sky = SkyCoord(OBJECT["ra"] * u.deg, OBJECT["dec"] * u.deg).directional_offset_by(angle * u.deg,
                                                                                       separation * u.deg)
    vis = rng.uniform(18.0, 27.0, n_field + n_cluster)
    h, j = vis - rng.uniform(-1.0, 1.5, vis.size), vis - rng.uniform(-1.0, 1.5, vis.size)
    for color_index, band in enumerate((h, j)):
        track = isochrone_cmd_track(DM, ISOCHRONE["age"], ISOCHRONE["metallicity"], match_filter=VIS_H_J,
                                    color_index=color_index)
        order = np.argsort(track["mag"])
        band[n_field:] = vis[n_field:] - np.interp(vis[n_field:], track["mag"][order], track["color"][order])
    return pd.DataFrame({"ra": sky.ra.deg, "dec": sky.dec.deg, f"{VIS}_obs": vis, f"{H}_obs": h, f"{J}_obs": j,
                         **{f"{band}_err": np.full(vis.size, 0.02) for band in (VIS, H, J)}})


@pytest.fixture(scope="module")
def regions(catalog):
    return object_regions(catalog, OBJECT, on_radii=(0.0, 2.0), off_radii=(5.0, 10.0), masked_object_types=(),
                          footprint_mask=np.zeros(hp.nside2npix(64), dtype=bool))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# The counts of the selected stars; the edges of the on and off regions; the SNR in the title
def test_plot_object_map(catalog, regions):
    selected = stars_in_match_filter(catalog, DM, ISOCHRONE, VIS_H_J)
    fig, ax = plot_object_map(regions, selected, bin_arcmin=2.0)
    image = ax.get_images()[0]
    left, right, bottom, top = image.get_extent()
    east, north = regions["offset_east_deg"] * 60.0, regions["offset_north_deg"] * 60.0
    in_map = (east >= left) & (east < right) & (north >= bottom) & (north < top)
    assert image.get_array().sum() == (selected & ~regions["excluded"] & in_map).sum()
    ellipses = [patch for patch in ax.patches if isinstance(patch, Ellipse)]
    assert len(ellipses) == 3  # on outer edge (inner 0 not drawn), off inner and outer
    assert ax.get_xlim()[0] > ax.get_xlim()[1]  # east to the left
    assert "SNR" in ax.get_title()


# One panel per colour of the match filter, with the polygon; or one per given pair, without
def test_plot_object_cmd(catalog, regions):
    fig, axes = plot_object_cmd(catalog, regions, DM, ISOCHRONE, VIS_H_J)
    assert len(axes) == 2
    assert [ax.get_xlabel() for ax in axes] == ["VIS - H", "VIS - J"]
    assert all(any(line.get_label() == "Match filter" for line in ax.get_lines()) for ax in axes)
    on_points = axes[0].collections[1].get_offsets()
    assert len(on_points) == regions["on"].sum()

    fig, axes = plot_object_cmd(catalog, regions, mag_cols=[(f"{H}_obs", f"{J}_obs")])
    assert len(axes) == 1 and axes[0].get_xlabel() == "H - J"
    assert not axes[0].get_lines()


# One row of on, off, on - off (and smoothed) per colour
def test_plot_object_hess(catalog, regions):
    fig, axes = plot_object_hess(catalog, regions, DM, ISOCHRONE, VIS_H_J)
    assert [len(row) for row in axes] == [4, 4]
    assert axes[1][0].get_xlabel() == "VIS - J"
    fig, axes = plot_object_hess(catalog, regions, mag_cols=[(f"{VIS}_obs", f"{H}_obs")], smoothed_difference=False)
    assert [len(row) for row in axes] == [3]
    with pytest.raises(ValueError, match="rows"):
        plot_object_hess(catalog, regions, DM, ISOCHRONE, VIS_H_J, axes=[plt.subplots(1, 4)[1]])


def test_plot_selection_comparison(catalog, regions):
    selections = {"all": np.ones(len(catalog), dtype=bool),
                  "VIS-H-J": stars_in_match_filter(catalog, DM, ISOCHRONE, VIS_H_J)}
    table = compare_selections(regions, selections)
    fig, ax = plot_selection_comparison(table)
    widths = [patch.get_width() for patch in ax.patches]
    np.testing.assert_allclose(widths, table["snr"].to_numpy())
    assert [label.get_text() for label in ax.get_yticklabels()] == ["all", "VIS-H-J"]
    _, ax = plot_selection_comparison(table.rename(index={"all": "VIS < 24_b"}))
    assert ax.get_yticklabels()[0].get_text() == r"VIS $<$ 24$\_$b"  # no such glyphs in the font

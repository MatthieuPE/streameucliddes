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

from streameucliddes.analysis.build_masks import elliptical_radius_deg
from streameucliddes.analysis.catalog_preparation import catalog_match_filter_mask
from streameucliddes.analysis.cmd_match_filter import isochrone_cmd_track
from streameucliddes.analysis.gc_dwarf_optimization import compare_selections, object_regions
from streameucliddes.plotting.plot_gc_dwarf_optimization import (
    plot_object_cmd,
    plot_object_hess,
    plot_object_map,
    plot_selection_comparison,
)
from streameucliddes.stream_frames import phi12_to_radec

pytestmark = pytest.mark.plot_gc_dwarf_optimization

DM = 17.0
ISOCHRONE = {"age": 12.0, "metallicity": 0.0003}
VIS, H, J = "euclid_q1_VIS", "euclid_q1_H", "euclid_q1_J"
VIS_H_J = {"name": "vis_h_j", "reference_band": ("euclid_q1", "VIS"),
           "color_bands": [("euclid_q1", "H"), ("euclid_q1", "J")]}
OBJECT = pd.Series({"key": "dwarf_a", "name": "Dwarf_A", "ra": 50.0, "dec": -30.0, "radius_arcmin": 6.0,
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


# The map of the frame (phi1 east, to the left); the outlines of the regions on their ellipses; the SNR in the title
def test_plot_object_map(catalog, regions):
    selected = catalog_match_filter_mask(catalog, DM, ISOCHRONE, VIS_H_J)
    fig, ax = plot_object_map(regions, selected, bin_size_deg=0.05)
    assert ax.get_xlim()[0] > ax.get_xlim()[1]  # east to the left
    outlines = {line.get_label(): line.get_xydata() for line in ax.get_lines() if line.get_label() in ("on", "off")}
    assert set(outlines) == {"on", "off"}
    for label, radius in (("on", 2.0), ("off", 10.0)):
        ra, dec = phi12_to_radec(*outlines[label].T, regions["stream_frame"])
        np.testing.assert_allclose(elliptical_radius_deg(ra, dec, 50.0, -30.0, 0.4, 30.0), radius * 0.1, rtol=1e-6)
    assert len(ax.get_lines()) == 3  # on outer edge (inner 0 not drawn), off inner and outer
    assert "SNR" in ax.get_title() and "Dwarf" in ax.get_title()


# One panel per colour of the match filter; the on stars of the regions
def test_plot_object_cmd(catalog, regions):
    fig, axes = plot_object_cmd(catalog, regions, DM, ISOCHRONE, VIS_H_J)
    assert [ax.get_xlabel() for ax in axes] == ["VIS - H", "VIS - J"]
    assert len(axes[0].collections[1].get_offsets()) == regions["on"].sum()
    shown = (catalog[f"{VIS}_obs"] < 22.0).to_numpy()
    fig, axes = plot_object_cmd(catalog, regions, mag_cols=[(f"{H}_obs", f"{J}_obs")], stars=shown)
    assert len(axes[0].collections[1].get_offsets()) == (regions["on"] & shown).sum()


# One row of on, off, on - off and smoothed per colour
def test_plot_object_hess(catalog, regions):
    fig, axes = plot_object_hess(catalog, regions, DM, ISOCHRONE, VIS_H_J)
    assert [len(row) for row in axes] == [4, 4]
    assert axes[1][0].get_xlabel() == "VIS - J"


# One bar per selection, its length the SNR; labels with < drawn by mathtext (no such glyph in the font)
def test_plot_selection_comparison(catalog, regions):
    selections = {"all": np.ones(len(catalog), dtype=bool),
                  "VIS < 24_b": catalog_match_filter_mask(catalog, DM, ISOCHRONE, VIS_H_J)}
    table = compare_selections(regions, selections)
    fig, ax = plot_selection_comparison(table)
    np.testing.assert_allclose([patch.get_width() for patch in ax.patches], table["snr"].to_numpy())
    assert [label.get_text() for label in ax.get_yticklabels()] == ["all", r"VIS $<$ 24$\_$b"]

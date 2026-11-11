"""Tests of plotting/plot_known_stream_search.py, with synthetic galstreams tracks."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import astropy.coordinates as coord
import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from streameucliddes.catalogs.galstreams_catalog import select_galstreams_in_footprint
from streameucliddes.plotting import plot_known_stream_search
from streameucliddes.plotting.plot_known_stream_search import (
    plot_galstreams_selection,
    plot_known_stream_search_single_dm,
)

pytestmark = pytest.mark.plot_known_stream_search

NSIDE = 128


def _track(frame, name, phi1_range=(-10.0, 10.0), phi2=0.0):
    """galstreams-like track along ``frame`` at 20 kpc, a point every 0.01 deg."""
    phi1 = np.linspace(*phi1_range, round((phi1_range[1] - phi1_range[0]) / 0.01) + 1)
    points = coord.SkyCoord(
        phi1=phi1 * u.deg, phi2=np.full(phi1.size, phi2) * u.deg,
        distance=np.full(phi1.size, 20.0) * u.kpc, frame=frame,
    ).transform_to(coord.ICRS())
    return SimpleNamespace(
        track=points, stream_frame=frame, InfoFlags="1111", ref="synthetic",
        track_name=name, stream_name=name.rsplit("-", 1)[0],
    )


@pytest.fixture
def southern_footprint():
    """Footprint dec < 0 (mask True outside)."""
    _, dec = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    return dec >= 0


def _label_texts(ax, names):
    return [text for text in ax.texts if text.get_text() in names]


# Phase 2 stub: the module imports, with its function still to write
def test_import():
    assert callable(plot_known_stream_search_single_dm)


# ===================================================
# Test plot_galstreams_selection
# ===================================================


# One line and one label (the stream name) per selected track; a track left out of the
# selection is neither drawn nor labelled. Legend: footprint and tracks
def test_plot_galstreams_selection(gc_frame, southern_footprint):
    mws = {
        "Alpha-T00": _track(gc_frame, "Alpha-T00"),  # conftest frame: dec -20 to -30
        "Beta-T00": _track(gc_frame, "Beta-T00", phi1_range=(-5.0, 5.0), phi2=3.0),
        "Gamma-T00": _track(gc_frame, "Gamma-T00", phi1_range=(150.0, 160.0)),  # north
    }
    selection = select_galstreams_in_footprint(mws, southern_footprint, nest=False)
    assert set(selection["track_name"]) == {"Alpha-T00", "Beta-T00"}

    fig, ax = plot_galstreams_selection(selection, mws, southern_footprint, nest=False)
    assert ax.figure is fig
    track_lines = [line for line in ax.lines if line.get_color() == plot_known_stream_search.TRACK_COLOR]
    assert len(track_lines) == 2
    assert {text.get_text() for text in _label_texts(ax, {"Alpha", "Beta", "Gamma"})} == {"Alpha", "Beta"}
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["Footprint", "Selected tracks"]
    assert ax.get_title().endswith(": 2")
    plt.close(fig)


# Two tracks along nearly the same path: their labels do not overlap
def test_plot_galstreams_selection_labels_do_not_overlap(gc_frame, southern_footprint):
    mws = {
        "Alpha-T00": _track(gc_frame, "Alpha-T00"),
        "Twin-T00": _track(gc_frame, "Twin-T00", phi2=0.2),
    }
    selection = select_galstreams_in_footprint(mws, southern_footprint, nest=False)
    fig, ax = plot_galstreams_selection(selection, mws, southern_footprint, nest=False)
    renderer = fig.canvas.get_renderer()
    first, second = (text.get_window_extent(renderer) for text in _label_texts(ax, {"Alpha", "Twin"}))
    assert not first.overlaps(second)
    plt.close(fig)


# No track selected: the footprint alone
def test_plot_galstreams_selection_empty(gc_frame, southern_footprint):
    mws = {"Gamma-T00": _track(gc_frame, "Gamma-T00", phi1_range=(150.0, 160.0))}
    selection = select_galstreams_in_footprint(mws, southern_footprint, nest=False)
    fig, ax = plot_galstreams_selection(selection, mws, southern_footprint, nest=False)
    assert not [line for line in ax.lines if line.get_color() == plot_known_stream_search.TRACK_COLOR]
    assert ax.get_title().endswith(": 0")
    plt.close(fig)


# A line crossing the edge of the map (lon_0 + 180) is cut there; others are not
def test_split_at_map_edge():
    ra = np.array([170.0, 175.0, 179.0, 181.0, 185.0])
    dec = np.zeros(5)
    parts = list(plot_known_stream_search._split_at_map_edge(ra, dec, lon_0=0.0))
    assert [list(part_ra) for part_ra, _ in parts] == [[170.0, 175.0, 179.0], [181.0, 185.0]]
    assert len(list(plot_known_stream_search._split_at_map_edge(ra, dec, lon_0=180.0))) == 1


# The centre of the map is the mean right ascension on the circle: across ra = 0 too
def test_central_ra():
    ra, dec = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    ra_wrapped = (ra + 180.0) % 360.0 - 180.0
    footprint_mask = ~((np.abs(ra_wrapped - (-10.0)) < 30.0) & (np.abs(dec) < 20.0))  # ra 320 to 20
    central_ra = plot_known_stream_search._central_ra(footprint_mask, nest=False)
    assert (central_ra - 350.0 + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=0.5)

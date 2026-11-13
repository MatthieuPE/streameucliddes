"""Tests of plotting/plot_stream_frame_search.py, with synthetic galstreams tracks."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import astropy.coordinates as coord
import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import skyproj

from streameucliddes.analysis.stream_frame_search import (
    stream_frame_search_dm_along_track,
    stream_frame_search_single_dm,
    prepare_search,
)
from streameucliddes.catalogs.galstreams_catalog import select_galstreams_in_footprint
from streameucliddes.stream_frames import phi12_bin_centres, phi12_to_radec
from streameucliddes.plotting import plot_stream_frame_search
from streameucliddes.plotting.plot_stream_frame_search import (
    TRACK_STYLE,
    draw_stream_track,
    draw_stream_track_on_sky,
    plot_galstreams_selection,
    plot_stream_frame_search_single_dm,
)

pytestmark = pytest.mark.plot_stream_frame_search

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


WINDOW = (-10.0, 10.0, -3.0, 3.0)
BIN = 0.2


@pytest.fixture
def search(gc_frame):
    """Prepared search on uniform mock stars (Euclid VIS, H), the footprint at phi1 < 8."""
    rng = np.random.default_rng(4)
    n = 30_000
    ra, dec = phi12_to_radec(rng.uniform(-11.0, 11.0, n), rng.uniform(-4.0, 4.0, n), gc_frame)
    vis = rng.uniform(18.0, 26.0, n)
    catalog = pd.DataFrame({
        "ra": ra, "dec": dec, "euclid_q1_VIS_obs": vis, "euclid_q1_H_obs": vis - rng.uniform(-1.0, 1.5, n),
        "euclid_q1_VIS_err": np.full(n, 0.02), "euclid_q1_H_err": np.full(n, 0.02),
    })
    phi1_centres, phi2_centres = phi12_bin_centres(WINDOW, BIN)
    footprint = np.broadcast_to(phi1_centres[:, None] > 8.0, (phi1_centres.size, phi2_centres.size))
    return prepare_search(catalog, gc_frame, WINDOW, bin_size_deg=BIN, footprint_mask=footprint)


# ===================================================
# Test plot_stream_frame_search_single_dm
# ===================================================


# The density on the grid: phi1 increasing to the left, bins outside the footprint grey,
# colour scale from the 1st to the 99.5th percentile, title from the result
def test_plot_single_dm(search):
    result = stream_frame_search_single_dm(distance_modulus=17.0, search=search)
    fig, ax = plot_stream_frame_search_single_dm(result)
    image = ax.images[0]
    np.testing.assert_array_equal(image.get_array(), result["density"].T)
    np.testing.assert_array_equal(image.get_array().mask, result["density"].mask.T)  # outside: masked
    assert matplotlib.colors.same_color(image.get_cmap().get_bad(), plot_stream_frame_search.FOOTPRINT_OUTSIDE_GREY)

    half = BIN / 2  # the extent: the edges of the bins
    phi1_centres, phi2_centres = result["phi1_centres"], result["phi2_centres"]
    np.testing.assert_allclose(image.get_extent(), [phi1_centres[0] - half, phi1_centres[-1] + half,
                                                    phi2_centres[0] - half, phi2_centres[-1] + half])
    assert ax.get_xlim()[0] > ax.get_xlim()[1]  # phi1 increasing to the left
    values = result["density"].compressed()
    low, high = plot_stream_frame_search.DENSITY_PERCENTILES
    np.testing.assert_allclose(image.get_clim(), (np.percentile(values, low), np.percentile(values, high)))
    assert ax.get_title() == "dm 17.0 (25.1 kpc), 13 Gyr, Z = 0.0002"
    assert ax.get_legend() is None  # no track
    plt.close(fig)


# The colour bar, with its ticks and label, inside the saved figure (constrained layout, as
# in plot_style), and the height of the map
def test_plot_single_dm_colorbar_inside_figure(search):
    result = stream_frame_search_single_dm(distance_modulus=17.0, search=search)
    with plt.rc_context({"figure.constrained_layout.use": True}):
        fig, ax = plot_stream_frame_search_single_dm(result)
        fig.canvas.draw()
    colorbar_ax, = ax.child_axes
    assert colorbar_ax.get_ylabel()  # the colour bar, with its label
    renderer = fig.canvas.get_renderer()
    box, figure_box = colorbar_ax.get_tightbbox(renderer), fig.bbox
    assert figure_box.x0 <= box.x0 and box.x1 <= figure_box.x1
    assert figure_box.y0 <= box.y0 and box.y1 <= figure_box.y1
    map_position, colorbar_position = ax.get_position(), colorbar_ax.get_position()
    np.testing.assert_allclose([colorbar_position.y0, colorbar_position.y1],
                               [map_position.y0, map_position.y1])
    plt.close(fig)


# The colour-bar labels have COLORBAR_DECIMALS decimals: maps of other colour scales (other
# ticks) keep the same size, for regular gifs
def test_plot_single_dm_colorbar_decimals(search):
    result = stream_frame_search_single_dm(distance_modulus=17.0, search=search)
    positions = []
    for scale in (1.0, 1.7, 3.1):
        scaled = {**result, "density": result["density"] * scale}
        with plt.rc_context({"figure.constrained_layout.use": True}):
            fig, ax = plot_stream_frame_search_single_dm(scaled)
            fig.canvas.draw()
        labels = [label.get_text() for label in ax.child_axes[0].get_yticklabels() if label.get_text()]
        decimals = plot_stream_frame_search.COLORBAR_DECIMALS
        assert labels and all(len(label.split(".")[1]) == decimals for label in labels)
        positions.append(ax.get_position().bounds)
        plt.close(fig)
    np.testing.assert_allclose(positions[1:], positions[:1] * 2)


# Tracks: the stream's dashed, the others in their colours, all in the legend; the given
# vmax, title and axes are used
def test_plot_single_dm_tracks(search):
    result = stream_frame_search_single_dm(distance_modulus=17.0, search=search)
    fig, ax = plt.subplots()
    phi1 = np.linspace(-6.0, 6.0, 50)
    returned_fig, returned_ax = plot_stream_frame_search_single_dm(
        result, track_phi12=(phi1, np.zeros(50)), other_tracks=[("Other", phi1, np.full(50, 1.5))],
        ax=ax, vmax=50.0, title="My map",
    )
    assert returned_ax is ax and returned_fig is fig
    stream_line, other_line = ax.lines
    assert stream_line.get_linestyle() == "--"
    assert matplotlib.colors.same_color(stream_line.get_color(), plot_stream_frame_search.TRACK_COLOR)
    assert not matplotlib.colors.same_color(other_line.get_color(), plot_stream_frame_search.TRACK_COLOR)
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["Track", "Other"]
    assert ax.images[0].get_clim()[1] == 50.0 and ax.get_title() == "My map"
    plt.close(fig)


# A map following the track: the offset in the title
def test_plot_single_dm_along_track(search, gc_frame):
    track = _track(gc_frame, "Mock-T00")
    (_, result), = stream_frame_search_dm_along_track(
        stars=search.stars, track=track, window=WINDOW, bin_size_deg=BIN, dm_offsets=(0.5,),
    )
    fig, ax = plot_stream_frame_search_single_dm(result)
    assert ax.get_title().startswith("Mock-T00, dm along the track +0.5, ")
    plt.close(fig)


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
    track_lines = [line for line in ax.lines if line.get_color() == plot_stream_frame_search.TRACK_COLOR]
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
    assert not [line for line in ax.lines if line.get_color() == plot_stream_frame_search.TRACK_COLOR]
    assert ax.get_title().endswith(": 0")
    plt.close(fig)


# A line crossing the edge of the map (lon_0 + 180) is cut there; others are not
def test_split_at_map_edge():
    ra = np.array([170.0, 175.0, 179.0, 181.0, 185.0])
    dec = np.zeros(5)
    parts = list(plot_stream_frame_search._split_at_map_edge(ra, dec, lon_0=0.0))
    assert [list(part_ra) for part_ra, _ in parts] == [[170.0, 175.0, 179.0], [181.0, 185.0]]
    assert len(list(plot_stream_frame_search._split_at_map_edge(ra, dec, lon_0=180.0))) == 1


# The centre of the map is the mean right ascension on the circle: across ra = 0 too
def test_central_ra():
    ra, dec = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    ra_wrapped = (ra + 180.0) % 360.0 - 180.0
    footprint_mask = ~((np.abs(ra_wrapped - (-10.0)) < 30.0) & (np.abs(dec) < 20.0))  # ra 320 to 20
    central_ra = plot_stream_frame_search._central_ra(footprint_mask, nest=False)
    assert (central_ra - 350.0 + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=0.5)


# ===================================================
# Test draw_stream_track and draw_stream_track_on_sky
# ===================================================


# One dashed line in TRACK_STYLE, shifted by phi2_offset_deg; a given style over it
def test_draw_stream_track():
    fig, ax = plt.subplots()
    phi1, phi2 = np.linspace(-5.0, 5.0, 20), np.zeros(20)
    (line,) = draw_stream_track(ax, phi1, phi2, label="track", phi2_offset_deg=0.5)
    np.testing.assert_allclose(line.get_ydata(), 0.5)
    assert line.get_linestyle() == "--" and line.get_alpha() == TRACK_STYLE["alpha"] and line.get_label() == "track"
    (line,) = draw_stream_track(ax, phi1, phi2, color="k", lw=2.0)
    assert line.get_color() == "k" and line.get_linewidth() == 2.0
    plt.close(fig)


# A track across the map edge (ra = lon_0 + 180): one line per side, labelled once
def test_draw_stream_track_on_sky_split():
    fig, ax = plt.subplots()
    sky_ax = skyproj.McBrydeSkyproj(ax=ax, lon_0=0.0).ax
    lines = draw_stream_track_on_sky(sky_ax, np.linspace(170.0, 190.0, 50), np.full(50, -20.0), label="edge")
    assert len(lines) == 2 and [line.get_label() for line in lines][0] == "edge"
    assert not lines[1].get_label() or lines[1].get_label().startswith("_")
    assert len(draw_stream_track_on_sky(sky_ax, np.full(50, 10.0), np.linspace(-30.0, -10.0, 50))) == 1
    plt.close(fig)

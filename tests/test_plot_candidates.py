"""Tests of plotting/plot_candidates.py: candidates and picked endpoints on a skyproj map (Agg backend)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import skyproj
from matplotlib.backend_bases import MouseEvent

from streameucliddes.catalogs.stream_candidates import add_candidate
from streameucliddes.plotting.plot_candidates import (
    EndpointPicker,
    endpoints_arc_radec,
    plot_candidates,
    sky_axes_to_radec,
)
from streameucliddes.stream_frames import radec_to_phi12, stream_frame_from_endpoints

pytestmark = pytest.mark.plot_candidates


@pytest.fixture
def sky_ax():
    """A skyproj map around the example candidate, as plot_generic_search_residual_map makes."""
    fig, ax = plt.subplots()
    sky = skyproj.McBrydeSkyproj(ax=ax, lon_0=60.0)
    sky.ax.set_extent([30.0, 90.0, -60.0, -30.0])
    yield sky.ax
    plt.close(fig)


def _pixels(ax, ra, dec):
    """Display (pixel) coordinates of (ra, dec) on the map."""
    x, y = ax.projection.transform_points(np.array([ra]), np.array([dec]))[0, :2]
    return ax.transData.transform((x, y))


def _click(ax, ra, dec, button, *, release_at=None):
    """Press of the mouse at (ra, dec) and release at the same place (or at ``release_at``, a drag)."""
    press, release = _pixels(ax, ra, dec), _pixels(ax, *(release_at or (ra, dec)))
    for name, (pixel_x, pixel_y) in (("button_press_event", press), ("button_release_event", release)):
        event = MouseEvent(name, ax.figure.canvas, pixel_x, pixel_y, button=button)
        ax.figure.canvas.callbacks.process(name, event)


# The arc goes from one endpoint to the other, on phi2 = 0 of their frame
def test_endpoints_arc_radec():
    endpoint_1, endpoint_2 = (45.5, -45.0), (44.0, -51.0)
    ra, dec = endpoints_arc_radec(endpoint_1, endpoint_2, n_points=50)
    assert (ra[0], dec[0]) == pytest.approx(endpoint_1) and (ra[-1], dec[-1]) == pytest.approx(endpoint_2)
    _, phi2 = radec_to_phi12(ra, dec, stream_frame_from_endpoints(endpoint_1, endpoint_2))
    assert np.abs(phi2).max() < 1e-8


# Projected points come back to their (ra, dec); points beyond the edge of the sky are NaN
def test_sky_axes_to_radec(sky_ax):
    ra, dec = np.array([45.5, 70.0, 359.0]), np.array([-45.0, -44.5, -50.0])
    xy = sky_ax.projection.transform_points(ra, dec)
    back_ra, back_dec = sky_axes_to_radec(sky_ax, xy[:, 0], xy[:, 1])
    np.testing.assert_allclose(back_ra, ra, atol=1e-8)
    np.testing.assert_allclose(back_dec, dec, atol=1e-8)
    far = 10 * np.abs(xy).max()
    outside_ra, _ = sky_axes_to_radec(sky_ax, far, far)
    assert np.isnan(outside_ra[0])


# One arc, two endpoints and one label (at the first endpoint) per candidate of the file; nothing for an
# empty file
def test_plot_candidates(sky_ax, tmp_path):
    path = tmp_path / "candidates.csv"
    assert plot_candidates(sky_ax, path=path) == []
    add_candidate((45.5, -45.0), (44.0, -51.0), 17.0, path=path)
    add_candidate((77.0, -40.5), (70.0, -44.5), 17.5, path=path)
    artists = plot_candidates(sky_ax, path=path)
    texts = [artist for artist in artists if isinstance(artist, plt.Text)]
    assert [text.get_text().strip() for text in texts] == ["1", "2"]
    assert len(artists) == 2 * 3  # arc (in one part, far from the map edge), endpoints, label
    sky_ax.figure.canvas.draw()
    endpoints = artists[1]
    first_endpoint = endpoints.get_transform().transform(np.column_stack(endpoints.get_data()))[0]
    assert (texts[0].get_window_extent().x0, texts[0].get_window_extent().y0) == pytest.approx(first_endpoint)
    assert [artist.get_label() for artist in artists].count("candidates") == 1
    assert len(plot_candidates(sky_ax, path=path, label_ids=False)) == 2 * 2


# Left clicks: the endpoints at the clicked (ra, dec), their arc; a third click starts a new pair;
# right click: removes the last endpoint
def test_endpoint_picker_clicks(sky_ax):
    picker = EndpointPicker(sky_ax)
    with pytest.raises(ValueError, match="click two"):
        _ = picker.endpoints
    _click(sky_ax, 45.5, -45.0, button=1)
    _click(sky_ax, 44.0, -51.0, button=1)
    (ra_1, dec_1), (ra_2, dec_2) = picker.endpoints
    assert (ra_1, dec_1, ra_2, dec_2) == pytest.approx((45.5, -45.0, 44.0, -51.0), abs=0.05)  # pixel rounding
    assert len(picker._artists) == 2  # the marks and the arc
    _click(sky_ax, 60.0, -40.0, button=1)
    assert len(picker.points) == 1 and picker.points[0] == pytest.approx((60.0, -40.0), abs=0.05)
    _click(sky_ax, 60.0, -40.0, button=3)
    assert picker.points == [] and picker._artists == []
    picker.disconnect()
    _click(sky_ax, 45.5, -45.0, button=1)
    assert picker.points == []


# A drag (press and release apart: zoom or pan) adds no endpoint; a click after it does
def test_endpoint_picker_ignores_drags(sky_ax):
    picker = EndpointPicker(sky_ax)
    _click(sky_ax, 45.5, -45.0, button=1, release_at=(50.0, -47.0))
    assert picker.points == []
    _click(sky_ax, 45.5, -45.0, button=1)
    assert len(picker.points) == 1


# A new picker on a new map (the cell run again) takes the clicks, the old one no longer does
def test_endpoint_picker_on_a_new_map(sky_ax):
    first = EndpointPicker(sky_ax)
    _click(sky_ax, 45.5, -45.0, button=1)
    _click(sky_ax, 44.0, -51.0, button=1)
    fig, ax = plt.subplots()
    sky = skyproj.McBrydeSkyproj(ax=ax, lon_0=60.0)
    sky.ax.set_extent([30.0, 90.0, -60.0, -30.0])
    second = EndpointPicker(sky.ax)
    _click(sky.ax, 60.0, -40.0, button=1)
    _click(sky.ax, 70.0, -44.5, button=1)
    assert np.ravel(second.endpoints) == pytest.approx([60.0, -40.0, 70.0, -44.5], abs=0.05)
    assert np.ravel(first.endpoints) == pytest.approx([45.5, -45.0, 44.0, -51.0], abs=0.05)
    plt.close(fig)

"""Tests of plotting/plot_candidate_investigation.py: regions and profiles on synthetic inputs;
the plot_candidate_* wrappers on a CandidateInvestigation of synthetic stars (conftest.py); the listed
candidates and the picked endpoints on a skyproj map (Agg backend)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import skyproj
from matplotlib.backend_bases import MouseEvent
from scipy.ndimage import gaussian_filter

from streameucliddes.analysis.candidate_investigation import CandidateInvestigation, hess_error_table
from streameucliddes.analysis.candidate_investigation import region_windows
from streameucliddes.analysis.cmd_match_filter import match_filter_cmd_columns
from streameucliddes.catalogs.stream_candidates import add_candidate
from streameucliddes.plotting.plot_candidate_investigation import (
    EndpointPicker,
    draw_stream_regions,
    endpoints_arc_radec,
    plot_candidate_frame_check,
    plot_candidate_hess,
    plot_candidate_map,
    plot_candidate_null_test,
    plot_candidate_overview,
    plot_candidates,
    plot_phi1_profile,
    plot_phi2_profile,
    plot_photometric_errors,
    plot_width_fit_map,
    sky_axes_to_radec,
)
from streameucliddes.plotting.plot_cmd import DEFAULT_DIFFERENCE_SMOOTHING_BINS, plot_on_off_difference_hess
from streameucliddes.stream_frames import radec_to_phi12, stream_frame_from_endpoints

pytestmark = pytest.mark.plot_candidate_investigation


# Three closed outlines on the window edges: on, off below, off above; labelled once each
def test_draw_stream_regions_three_outlines():
    fig, ax = plt.subplots()
    regions = {"phi1_range": (-3.0, 2.0), "phi2_centre": 0.2, "sigma": 0.3, "on_n_sigma": 1.5, "off_n_sigma": (2.5, 4.0)}
    lines = draw_stream_regions(ax, regions)
    assert len(lines) == 3
    for line, window in zip(lines, [region_windows(regions)["on"], *region_windows(regions)["off"]]):
        x, y = line.get_xdata(), line.get_ydata()
        assert (x[0], y[0]) == (x[-1], y[-1])
        assert (x.min(), x.max(), y.min(), y.max()) == pytest.approx(window)
    assert [line.get_label() for line in lines][:2] == ["on", "off"] and lines[2].get_label().startswith("_")
    plt.close(fig)


FIT = {"amplitude": 5.0, "phi2_centre": 0.0, "sigma": 0.3, "background": 10.0, "background_coefficients": (10.0,),
       "background_reference": 0.0, "background_degree": 0, "phi2_fit_range": (-2.0, 2.0),
       "errors": {"phi2_centre": 0.02, "sigma": 0.03}}


# Points with error bars, the fitted range shaded, the model (solid in it, dashed beyond), the centre and
# +-sigma lines, the phi1 range in the title
def test_plot_phi2_profile(candidate_endpoints):
    centres = np.linspace(-2.95, 2.95, 60)
    profile = {"axis": "phi2", "centres": centres, "density": 10.0 + 5.0 * np.exp(-0.5 * (centres / 0.3) ** 2),
               "error": np.ones(60), "windows": [(-3.02, 3.02, -3.0, 3.0)]}
    fit = {**FIT, "model": profile["density"]}
    fig, ax = plot_phi2_profile(profile, fit)
    assert len(ax.lines) == 6 and len(ax.containers) == 1  # points, model x2, centre, two sigma lines; error bars
    assert len(ax.patches) == 1  # the fitted range
    assert "Gaussian + constant" in ax.get_legend().get_texts()[1].get_text()
    assert sorted(line.get_xdata()[0] for line in ax.lines[-3:]) == pytest.approx([-0.3, 0.0, 0.3])
    assert "-3.02" in ax.get_title() and "3.02" in ax.get_title()
    plt.close(fig)
    fig, ax = plot_phi2_profile(profile)  # without fit
    assert len(ax.lines) == 1 and len(ax.containers) == 1
    plt.close(fig)


# On - off with error bars, the zero line, the regions' phi1 range shaded, phi1 increasing to the left
def test_plot_phi1_profile():
    centres = np.linspace(-4.75, 4.75, 20)
    profile = {"centres": centres, "edges": np.linspace(-5.0, 5.0, 21),
               "difference": np.where(np.abs(centres) < 2, 30.0, 0.0), "error": np.full(20, 5.0)}
    fig, ax = plot_phi1_profile(profile, {"phi1_range": (-2.0, 2.0)})
    assert len(ax.containers) == 1 and len(ax.patches) == 1 and ax.get_xlim()[0] > ax.get_xlim()[1]
    plt.close(fig)


# The map of the scan closest to the distance modulus (given, else that of the object), else the zoom
# map; the arc between the endpoints drawn; no map: ValueError
def test_plot_candidate_map(candidate_with_maps):
    cand = candidate_with_maps
    fig, ax = plot_candidate_map(cand)  # no scan, no distance modulus: the zoom map
    assert "dm 17.0" in ax.get_title() and "endpoints arc" in {line.get_label() for line in ax.lines}
    plt.close(fig)
    with pytest.raises(ValueError, match="scan_distance_moduli"):
        plot_candidate_map(cand, 18.0)
    cand.scan_distance_moduli([16.0, 17.25, 18.0])
    fig, ax = plot_candidate_map(cand, 17.8)
    assert "dm 18.0" in ax.get_title()
    plt.close(fig)
    fig, ax = plot_candidate_map(cand, 17.3)  # two decimals when needed
    assert "dm 17.25 (" in ax.get_title()
    plt.close(fig)
    cand.distance_modulus = 16.2
    fig, ax = plot_candidate_map(cand)
    assert "dm 16.0" in ax.get_title()
    plt.close(fig)
    with pytest.raises(ValueError, match="from 16 to 18"):  # beyond one grid step (0.75) of the closest map
        plot_candidate_map(cand, 19.5)
    cand.regions = {"phi1_range": (-3.0, 2.0), "phi2_centre": 0.2, "sigma": 0.3, "on_n_sigma": 1.5,
                    "off_n_sigma": (2.5, 4.0)}
    fig, ax = plot_candidate_map(cand)  # with regions: the track and three outlines
    assert {"track", "on", "off"} <= {line.get_label() for line in ax.lines}
    plt.close(fig)
    fig, ax = plot_candidate_map(cand, show_regions=False)
    assert "on" not in {line.get_label() for line in ax.lines}
    plt.close(fig)



# The streams masked: hatched with a legend entry (not without show_stream_mask, nor when none is masked)
def test_plot_candidate_map_stream_mask(candidate_endpoints, candidate_catalog, no_object_masks, tmp_path,
                                        monkeypatch):
    from streameucliddes.analysis import candidate_investigation

    frame = stream_frame_from_endpoints((73.5, -48.0), (73.5, -35.0))
    crossing = {"name": "Cross-T00", "stream_name": "Cross", "ra": np.array([73.5, 73.5]),
                "dec": np.array([-48.0, -35.0]), "stream_frame": frame, "width_deg": 0.2, "dm_min": 17.0,
                "dm_max": 17.0}
    monkeypatch.setattr(candidate_investigation, "galstreams_mask_tracks", lambda *args, **kwargs: [crossing])
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=17.0,
                                  candidates_path=tmp_path / "none.csv",
                                  prepare_kwargs={"bin_size_deg": 0.2, **no_object_masks})
    cand.prepare(candidate_catalog).compute_zoom_map()
    assert cand.masked_streams == ["Cross-T00"]

    def labels(ax):
        return [text.get_text() for text in ax.get_legend().get_texts()]

    fig, ax = plot_candidate_map(cand)
    assert "masked streams" in labels(ax)
    plt.close(fig)
    fig, ax = plot_candidate_map(cand, show_stream_mask=False)
    assert "masked streams" not in labels(ax)
    plt.close(fig)
    cand.mask_streams = False
    fig, ax = plot_candidate_map(cand)
    assert "masked streams" not in labels(ax)
    plt.close(fig)


# Each of on, off, difference; the difference with symmetric limits; an unknown diagram or no Hess: ValueError
def test_plot_candidate_hess(candidate_endpoints, candidate_stream_catalog, candidate_stream, no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=candidate_stream["distance_modulus"],
                                  prepare_kwargs={"bin_size_deg": 0.2, **no_object_masks})
    cand.prepare(candidate_stream_catalog)
    with pytest.raises(ValueError, match="compute_hess"):
        plot_candidate_hess(cand)
    cand.fit_width().define_regions().compute_hess()
    for which in ("on", "off", "difference"):
        fig, ax = plot_candidate_hess(cand, which)
        assert which in ax.get_title()
        plt.close(fig)
    fig, ax = plot_candidate_hess(cand)
    vmin, vmax = ax.images[0].get_clim()
    assert vmin == -vmax and not ax.lines  # no isochrone computed yet
    assert "smoothed" in ax.get_title()
    assert not np.array_equal(ax.images[0].get_array(), cand.diff_hess["density"].T)  # smoothed by default
    plt.close(fig)
    fig, ax = plot_candidate_hess(cand, smooth=False)
    np.testing.assert_array_equal(ax.images[0].get_array(), cand.diff_hess["density"].T)
    plt.close(fig)
    cand.compute_isochrone()
    fig, ax = plot_candidate_hess(cand, match_filter=True)  # the isochrone and the match filter
    labels = [line.get_label() for line in ax.lines]
    assert len(labels) == 2 and "Match filter" in labels and ax.get_legend() is not None
    plt.close(fig)
    fig, ax = plot_candidate_hess(cand, isochrone=False)
    assert not ax.lines
    plt.close(fig)
    with pytest.raises(ValueError, match="which"):
        plot_candidate_hess(cand, "both")


# The table saved (hess_error_table) holds the raw diagrams, not the smoothed ones of the figures: after the smoothed
# plots, the densities are still the counts of the on region per deg² (np.histogram2d) and the difference is on - off,
# not its smoothed version
def test_saved_hess_not_smoothed(candidate_endpoints, candidate_stream_catalog, candidate_stream, no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=candidate_stream["distance_modulus"],
                                  prepare_kwargs={"bin_size_deg": 0.2, **no_object_masks})
    cand.prepare(candidate_stream_catalog).fit_width().define_regions().compute_hess()
    on_before = cand.on_hess["density"].copy()
    for fig in (plot_candidate_hess(cand)[0],
                plot_on_off_difference_hess(cand.on_hess, cand.off_hess, cand.diff_hess, smoothing_bins=2.0)[0]):
        plt.close(fig)
    np.testing.assert_array_equal(cand.on_hess["density"], on_before)
    table = hess_error_table(cand.on_hess, cand.off_hess)

    mag_column, color_column = match_filter_cmd_columns(cand.search.match_filter)[0]
    stars = cand.search.stars[np.asarray(cand.band_regions["on"], dtype=bool)]
    counts, _, _ = np.histogram2d(stars[mag_column] - stars[color_column], stars[mag_column],
                                  bins=[cand.on_hess["color_edges"], cand.on_hess["mag_edges"]])
    np.testing.assert_allclose(table["on_density"], (counts / cand.areas["on"]).ravel())
    raw_difference = (cand.on_hess["density"] - cand.off_hess["density"]).ravel()
    np.testing.assert_allclose(table["difference_density"], raw_difference)
    smoothed = gaussian_filter(cand.diff_hess["density"], DEFAULT_DIFFERENCE_SMOOTHING_BINS).ravel()
    assert not np.allclose(table["difference_density"], smoothed)


# The map with the shifted regions, the candidate's and the null test's difference Hess diagrams on the same scale,
# with their SNR; without the Hess diagrams or the null test: ValueError
def test_plot_candidate_null_test(candidate_endpoints, candidate_stream_catalog, candidate_stream, no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=candidate_stream["distance_modulus"],
                                  prepare_kwargs={"bin_size_deg": 0.2, **no_object_masks})
    cand.prepare(candidate_stream_catalog).compute_zoom_map()
    cand.define_regions(phi2_centre=candidate_stream["phi2_centre"], sigma=candidate_stream["sigma"])
    with pytest.raises(ValueError, match="compute_hess"):
        plot_candidate_null_test(cand)
    cand.compute_hess()
    with pytest.raises(ValueError, match="compute_null_test"):
        plot_candidate_null_test(cand)
    cand.compute_null_test().compute_isochrone()
    fig, (map_ax, candidate_ax, null_ax) = plot_candidate_null_test(cand)
    assert cand.null_test["direction"] in map_ax.get_title()
    assert "candidate, SNR" in candidate_ax.get_title() and "null test, SNR" in null_ax.get_title()
    assert candidate_ax.images[0].get_clim() == null_ax.images[0].get_clim()  # the same colour scale
    plt.close(fig)

# Four panels: the map with the regions, the phi2 and phi1 profiles, the difference Hess with the isochrone;
# before the steps: ValueError naming the missing one
def test_plot_candidate_overview(candidate_endpoints, candidate_stream_catalog, candidate_stream, no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=candidate_stream["distance_modulus"],
                                  prepare_kwargs={"bin_size_deg": 0.2, **no_object_masks})
    cand.prepare(candidate_stream_catalog).compute_zoom_map()
    with pytest.raises(ValueError, match="fit_width"):
        plot_candidate_overview(cand)
    cand.fit_width().define_regions()
    with pytest.raises(ValueError, match="compute_hess"):
        plot_candidate_overview(cand)
    cand.compute_hess().compute_isochrone()
    fig, axes = plot_candidate_overview(cand)
    map_ax, phi2_ax, phi1_ax, hess_ax = axes
    assert "on" in {line.get_label() for line in map_ax.lines}
    assert phi2_ax.lines and phi1_ax.containers and hess_ax.images and hess_ax.lines
    assert "excess" in fig.get_suptitle()
    plt.close(fig)


# Two panels at the distance modulus of the maps: the sky map with the great circle, the window and the
# endpoints; the map in the frame with phi2 = 0 and the endpoints
def test_plot_candidate_frame_check(candidate_with_maps):
    cand = candidate_with_maps
    fig, (sky_ax, zoom_ax) = plot_candidate_frame_check(cand)
    labels = {line.get_label() for line in sky_ax.lines}
    assert {"great circle", "window", "endpoints"} <= labels
    assert {"endpoints", r"$\phi_2 = 0$"} <= {line.get_label() for line in zoom_ax.lines}
    (ends,) = [line for line in zoom_ax.lines if line.get_label() == "endpoints"]
    assert ends.get_ydata() == pytest.approx([0.0, 0.0], abs=1e-8)
    assert "dm 17.0" in sky_ax.get_title() and "dm 17.0" in zoom_ax.get_title() and cand.name in fig.get_suptitle()
    plt.close(fig)


# Without the sky map or the zoom map: ValueError naming the step
def test_plot_candidate_frame_check_needs_maps(candidate_endpoints):
    with pytest.raises(ValueError, match="compute_sky_map"):
        plot_candidate_frame_check(CandidateInvestigation("test", *candidate_endpoints))


# Observed, model (constant along phi1, grey where no data), observed - model with symmetric limits
def test_plot_width_fit_map():
    phi1_edges, phi2_edges = np.linspace(-3.0, 3.0, 13), np.linspace(-3.0, 3.0, 31)
    phi2 = (phi2_edges[:-1] + phi2_edges[1:]) / 2
    density = np.tile(10.0 + 5.0 * np.exp(-0.5 * (phi2 / 0.3) ** 2), (12, 1))
    density[0, 0] = np.nan  # a cell without area
    width_map = {"phi1_edges": phi1_edges, "phi2_edges": phi2_edges, "density": density + 0.5}
    fig, axes = plot_width_fit_map(width_map, FIT)
    assert len(axes) == 3 and "constant" in axes[1].get_title()
    model = axes[1].images[0].get_array()
    assert model.mask[0, 0] and np.ptp(model[5, :]) == 0.0  # imshow of the transpose: row = phi2, constant along phi1
    vmin, vmax = axes[2].images[0].get_clim()
    assert vmin == -vmax and axes[2].images[0].get_array()[10, 6] == pytest.approx(0.5)
    assert axes[0].get_xlim()[0] > axes[0].get_xlim()[1]  # phi1 increasing to the left
    plt.close(fig)


# The view ranges of the width map: limits of the three panels and their colour scales, on the cells shown
def test_plot_width_fit_map_view_ranges():
    phi1_edges, phi2_edges = np.linspace(-3.0, 3.0, 13), np.linspace(-3.0, 3.0, 31)
    phi2 = (phi2_edges[:-1] + phi2_edges[1:]) / 2
    density = np.tile(10.0 + 5.0 * np.exp(-0.5 * (phi2 / 0.3) ** 2), (12, 1))
    density[:2, :] = 1.0e4  # a dense object at phi1 < -2, outside the view below
    width_map = {"phi1_edges": phi1_edges, "phi2_edges": phi2_edges, "density": density}
    fig, axes = plot_width_fit_map(width_map, FIT)
    assert axes[0].images[0].get_clim()[1] > 100.0 and axes[2].images[0].get_clim()[1] > 100.0
    plt.close(fig)
    fig, axes = plot_width_fit_map(width_map, FIT, phi1_range=(-2.0, 2.0), phi2_range=(-2.0, 2.0))
    assert axes[0].images[0].get_clim()[1] < 20.0 and axes[2].images[0].get_clim()[1] < 1.0
    assert all(ax.get_xlim() == (2.0, -2.0) and ax.get_ylim() == (-2.0, 2.0) for ax in axes)
    plt.close(fig)


# The list of candidates on the sky


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


# One line per band (each against its own magnitude) over the bins with stars; the title counts the stars of the
# reference band
def test_plot_photometric_errors():
    errors = {"mag_centres": np.array([20.5, 21.5, 22.5]), "bands": ("VIS", "H"), "statistic": "median",
              "error": {"VIS": np.array([0.02, 0.06, np.nan]), "H": np.array([0.02, 0.1, 0.3])},
              "n_stars": {"VIS": np.array([3, 2, 0]), "H": np.array([2, 2, 1])}}
    fig, ax = plot_photometric_errors(errors)
    assert ax.get_ylabel() == "median error [mag]"
    assert [line.get_label() for line in ax.lines] == ["VIS (against VIS)", "H (against H)"]
    assert [line.get_xdata().size for line in ax.lines] == [2, 3]
    assert ax.get_title() == "5 stars"
    plt.close(fig)

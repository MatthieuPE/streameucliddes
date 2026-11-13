"""Tests of plotting/plot_candidate_investigation.py: regions and profiles on synthetic inputs;
the plot_candidate_* wrappers on a CandidateInvestigation of synthetic stars (conftest.py)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from streameucliddes.analysis.candidate_investigation import CandidateInvestigation
from streameucliddes.analysis.candidate_investigation import region_windows
from streameucliddes.plotting.plot_candidate_investigation import (
    draw_stream_regions,
    plot_candidate_frame_check,
    plot_candidate_map,
    plot_phi1_profile,
    plot_phi2_profile,
    plot_width_fit_map,
)

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
    cand.regions = {"phi1_range": (-3.0, 2.0), "phi2_centre": 0.2, "sigma": 0.3, "on_n_sigma": 1.5,
                    "off_n_sigma": (2.5, 4.0)}
    fig, ax = plot_candidate_map(cand)  # with regions: the track and three outlines
    assert {"track", "on", "off"} <= {line.get_label() for line in ax.lines}
    plt.close(fig)
    fig, ax = plot_candidate_map(cand, show_regions=False)
    assert "on" not in {line.get_label() for line in ax.lines}
    plt.close(fig)


@pytest.mark.skip(reason="stub")
def test_plot_candidate_hess():
    """Each of on, off, difference, with the isochrone."""


@pytest.mark.skip(reason="stub")
def test_plot_candidate_overview():
    """Four panels."""


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

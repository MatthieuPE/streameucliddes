"""Tests of plotting/plot_candidate_investigation.py: regions and profiles on synthetic inputs;
the plot_candidate_* wrappers on a CandidateInvestigation of synthetic stars (conftest.py)."""

import matplotlib.pyplot as plt
import pytest

from streameucliddes.analysis.candidate_investigation import CandidateInvestigation
from streameucliddes.plotting.plot_candidate_investigation import plot_candidate_frame_check, plot_candidate_map

pytestmark = pytest.mark.plot_candidate_investigation


@pytest.mark.skip(reason="stub")
def test_draw_stream_regions_three_outlines():
    """Three closed outlines: on and the two off bands."""


@pytest.mark.skip(reason="stub")
def test_plot_phi2_profile():
    """Profile and fit drawn; returns (fig, ax)."""


@pytest.mark.skip(reason="stub")
def test_plot_phi1_profile():
    """Difference drawn, phi1_range shaded; returns (fig, ax)."""


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

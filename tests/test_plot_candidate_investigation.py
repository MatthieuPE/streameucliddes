"""Tests of plotting/plot_candidate_investigation.py: tracks, regions and profiles on synthetic
inputs; the plot_candidate_* wrappers on a CandidateInvestigation run on synthetic stars."""

import pytest

pytestmark = pytest.mark.plot_candidate_investigation


@pytest.mark.skip(reason="stub")
def test_draw_stream_track():
    """Adds one dashed line, shifted by phi2_offset_deg."""


@pytest.mark.skip(reason="stub")
def test_draw_stream_track_on_sky_split():
    """A track crossing the map edge is drawn in two parts."""


@pytest.mark.skip(reason="stub")
def test_draw_stream_regions_three_outlines():
    """Three closed outlines: on and the two off bands."""


@pytest.mark.skip(reason="stub")
def test_plot_phi2_profile():
    """Profile and fit drawn; returns (fig, ax)."""


@pytest.mark.skip(reason="stub")
def test_plot_phi1_profile():
    """Difference drawn, phi1_range shaded; returns (fig, ax)."""


@pytest.mark.skip(reason="stub")
def test_plot_candidate_frame_check():
    """Two panels: sky map with the great circle, window and endpoints; zoom map."""


@pytest.mark.skip(reason="stub")
def test_plot_candidate_frame_check_needs_maps():
    """Without sky_map or zoom_map: ValueError."""


@pytest.mark.skip(reason="stub")
def test_plot_candidate_map():
    """Map with the track and the regions."""


@pytest.mark.skip(reason="stub")
def test_plot_candidate_hess():
    """Each of on, off, difference, with the isochrone."""


@pytest.mark.skip(reason="stub")
def test_plot_candidate_overview():
    """Four panels."""

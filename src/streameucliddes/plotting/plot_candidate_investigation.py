"""Figures of a stream-candidate investigation (analysis.candidate_investigation).

Two kinds of functions:

- tracks, regions and profiles (``analysis.stream_regions``): the ``draw_*``
  functions draw on existing axes and return what they added; the
  ``plot_*`` functions make or take axes and return ``(fig, ax)``. The track
  functions take arrays, so they also draw galstreams tracks or the arc
  between two endpoints, in one style for all tracks;
- thin wrappers, ``plot_candidate_*``: each takes a
  ``CandidateInvestigation`` and calls the plots of ``plot_generic_search``,
  ``plot_stream_frame_search``, ``plot_cmd`` and the functions above on its
  results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from streameucliddes.plotting import plot_style  # noqa: F401  (sets the figure style)
from streameucliddes.plotting.plot_style import CATEGORICAL_COLORS

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from numpy.typing import ArrayLike

    from streameucliddes.analysis.candidate_investigation import CandidateInvestigation

TRACK_STYLE = {"color": CATEGORICAL_COLORS[0], "lw": 1.0, "ls": "--", "alpha": 0.7}  # thin dashed: not to hide the stream
ON_REGION_COLOR = CATEGORICAL_COLORS[1]
OFF_REGION_COLOR = CATEGORICAL_COLORS[2]
PROFILE_COLOR = CATEGORICAL_COLORS[3]
FIT_COLOR = CATEGORICAL_COLORS[4]


# ===================================================
# Tracks, regions and profiles
# ===================================================


def draw_stream_track(
    ax: Axes,
    phi1: ArrayLike,
    phi2: ArrayLike,
    *,
    label: str | None = None,
    phi2_offset_deg: float = 0.0,
    **line_kwargs: Any,
) -> list[Line2D]:
    """Track on a phi1/phi2 map, e.g. from ``plot_stream_frame_search_single_dm``.

    A thin dashed line (``TRACK_STYLE``), optionally shifted in phi2 not to
    hide the stream.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of a phi1/phi2 map.
    phi1, phi2 : array_like
        Track (deg).
    label : str, optional
        Legend label.
    phi2_offset_deg : float, optional
        Shift of the line in phi2 (deg).
    **line_kwargs
        Arguments of ``ax.plot``, updating ``TRACK_STYLE``.

    Returns
    -------
    lines : list of matplotlib.lines.Line2D
    """
    raise NotImplementedError


def draw_stream_track_on_sky(
    ax: Any,
    ra: ArrayLike,
    dec: ArrayLike,
    *,
    label: str | None = None,
    **line_kwargs: Any,
) -> list[Line2D]:
    """Track on a skyproj sky map, split at the map edge.

    Split with ``plot_stream_frame_search._split_at_map_edge`` (as the tracks
    of ``plot_generic_search_residual_map``), so that no line crosses the
    map.

    Parameters
    ----------
    ax : skyproj.SkyAxes
        Axes of a sky map, e.g. from ``plot_generic_search_residual_map``;
        its ``plot`` takes (ra, dec) in degrees.
    ra, dec : array_like
        Track (deg).
    label : str, optional
        Legend label, on the first part only.
    **line_kwargs
        Arguments of ``ax.plot``, updating ``TRACK_STYLE``.

    Returns
    -------
    lines : list of matplotlib.lines.Line2D
        One per part of the track.
    """
    raise NotImplementedError


def draw_stream_regions(
    ax: Axes,
    regions: dict[str, Any],
    *,
    on_color: str = ON_REGION_COLOR,
    off_color: str = OFF_REGION_COLOR,
) -> list[Line2D]:
    """Outlines of the on region and of the two off bands on a phi1/phi2 map.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of a phi1/phi2 map.
    regions : dict
        From ``analysis.stream_regions.stream_regions``.
    on_color, off_color : str, optional
        Colours of the outlines.

    Returns
    -------
    lines : list of matplotlib.lines.Line2D
        Three closed outlines: on, off below, off above.
    """
    raise NotImplementedError


def plot_phi2_profile(
    profile: dict[str, Any],
    fit: dict[str, Any] | None = None,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Density against phi2, with the fitted Gaussian plus background.

    Parameters
    ----------
    profile : dict
        From ``analysis.stream_regions.phi2_density_profile``.
    fit : dict, optional
        From ``fit_stream_width``: its model, and lines at
        ``phi2_centre ± sigma``.
    ax : matplotlib.axes.Axes, optional
        Default: a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    raise NotImplementedError


def plot_phi1_profile(
    profile: dict[str, Any],
    regions: dict[str, Any] | None = None,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """On - off density against phi1, to see where the stream starts and stops.

    Parameters
    ----------
    profile : dict
        From ``analysis.stream_regions.phi1_density_profile``.
    regions : dict, optional
        From ``stream_regions``: its ``phi1_range`` is shaded.
    ax : matplotlib.axes.Axes, optional
        Default: a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    raise NotImplementedError

# ===================================================
# Figures of a CandidateInvestigation
# ===================================================


def plot_candidate_frame_check(
    cand: CandidateInvestigation,
    *,
    sky_kwargs: Mapping[str, Any] | None = None,
    zoom_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Figure, list[Any]]:
    """Check of the endpoints: does the frame follow the stream?

    Left: ``cand.sky_map`` (corrected map of the loaded box,
    ``plot_generic_search_residual_map``) with the great circle of the frame,
    the outline of the window and the two endpoints. Right: ``cand.zoom_map``
    in phi1/phi2 (``plot_stream_frame_search_single_dm``) with the endpoints
    and the phi2 = 0 line. Same distance modulus, said in the title.

    Parameters
    ----------
    cand : CandidateInvestigation
        With ``sky_map`` and ``zoom_map`` computed.
    sky_kwargs : mapping, optional
        Arguments of ``plot_generic_search_residual_map``.
    zoom_kwargs : mapping, optional
        Arguments of ``plot_stream_frame_search_single_dm``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list
        The sky axes (``skyproj.SkyAxes``) and the phi1/phi2 axes.

    Raises
    ------
    ValueError
        ``sky_map`` or ``zoom_map`` not computed.
    """
    raise NotImplementedError


def plot_candidate_map(
    cand: CandidateInvestigation,
    distance_modulus: float | None = None,
    *,
    ax: Axes | None = None,
    show_regions: bool = True,
) -> tuple[Figure, Axes]:
    """Matched-filter map in the frame, with the track and the regions.

    ``plot_stream_frame_search_single_dm``, ``draw_stream_track`` and
    ``draw_stream_regions``.

    Parameters
    ----------
    cand : CandidateInvestigation
        With ``maps`` (or ``zoom_map``) computed.
    distance_modulus : float, optional
        Map of ``cand.maps`` closest to it. Default: ``cand.distance_modulus``,
        else ``cand.zoom_map``.
    ax : matplotlib.axes.Axes, optional
        Default: a new figure.
    show_regions : bool, optional
        Draw the regions, when defined.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    raise NotImplementedError


def plot_candidate_hess(
    cand: CandidateInvestigation,
    which: str = "difference",
    *,
    ax: Axes | None = None,
    isochrone: bool = True,
    match_filter: bool = False,
) -> tuple[Figure, Axes]:
    """One Hess diagram of the candidate, with its isochrone on top.

    Parameters
    ----------
    cand : CandidateInvestigation
        With the Hess diagrams computed (``compute_hess``).
    which : {"on", "off", "difference"}, optional
        The diagram.
    ax : matplotlib.axes.Axes, optional
        Default: a new figure.
    isochrone : bool, optional
        Draw ``cand.isochrone_track``, when computed (``plot_isochrone``).
    match_filter : bool, optional
        Draw the match-filter polygon at ``cand.distance_modulus``
        (``plot_match_filter_polygon``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    raise NotImplementedError


def plot_candidate_overview(cand: CandidateInvestigation) -> tuple[Figure, list[Axes]]:
    """Summary figure: map with the regions, phi2 profile, phi1 profile, difference Hess with the isochrone.

    Parameters
    ----------
    cand : CandidateInvestigation
        With every step done (``run_all_steps``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    raise NotImplementedError

"""Figures of a stream-candidate investigation (analysis.candidate_investigation).

Two kinds of functions:

- regions and profiles (``analysis.candidate_investigation``):
  ``draw_stream_regions`` draws on existing axes and returns what it added;
  the ``plot_*`` functions make or take axes and return ``(fig, ax)``. The
  tracks are drawn by ``plot_stream_frame_search.draw_stream_track`` and
  ``draw_stream_track_on_sky``, the same for every map;
- thin wrappers, ``plot_candidate_*``: each takes a
  ``CandidateInvestigation`` and calls the plots of ``plot_generic_search``,
  ``plot_stream_frame_search``, ``plot_cmd`` and the functions above on its
  results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from streameucliddes.plotting import plot_style  # noqa: F401  (sets the figure style)
from streameucliddes.plotting.plot_generic_search import plot_generic_search_residual_map
from streameucliddes.plotting.plot_stream_frame_search import (
    draw_stream_track,
    draw_stream_track_on_sky,
    plot_stream_frame_search_single_dm,
)
from streameucliddes.plotting.plot_style import CATEGORICAL_COLORS, escape_underscores
from streameucliddes.stream_frames import phi12_to_radec, phi12_window_outline, radec_to_phi12

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from numpy.typing import ArrayLike

    from streameucliddes.analysis.candidate_investigation import CandidateInvestigation

WINDOW_STYLE = {"color": CATEGORICAL_COLORS[1], "lw": 1.0, "ls": "-", "alpha": 0.9}  # outline of the window
ENDPOINT_STYLE = {"ls": "none", "marker": "o", "ms": 7, "mfc": "none", "mec": CATEGORICAL_COLORS[4], "mew": 1.5}
FRAME_CHECK_FIGSIZE = (17.0, 6.5)
ON_REGION_COLOR = CATEGORICAL_COLORS[1]
OFF_REGION_COLOR = CATEGORICAL_COLORS[2]
PROFILE_COLOR = CATEGORICAL_COLORS[3]
FIT_COLOR = CATEGORICAL_COLORS[4]


# ===================================================
# Tracks, regions and profiles
# ===================================================


def draw_stream_regions(
    ax: Axes,
    regions: dict[str, Any],
    *,
    on_color: str = ON_REGION_COLOR,
    off_color: str = OFF_REGION_COLOR,
) -> list[Line2D]:
    """Outlines of the on region and of the two off bands on a phi1/phi2 map.

    ``stream_frames.phi12_window_outline`` of each window of
    ``analysis.candidate_investigation.region_windows``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of a phi1/phi2 map.
    regions : dict
        From ``analysis.candidate_investigation.stream_regions``.
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
        From ``analysis.candidate_investigation.phi12_density_profile`` along phi2.
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
        From ``analysis.candidate_investigation.on_off_phi1_profile``.
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
    for name, step in (("sky_map", "compute_sky_map"), ("zoom_map", "compute_zoom_map")):
        if getattr(cand, name) is None:
            raise ValueError(f"No {name}: call cand.{step}() first.")
    fig, (sky_panel, zoom_panel) = plt.subplots(1, 2, figsize=FRAME_CHECK_FIGSIZE, width_ratios=(1.25, 1.0))
    ends_ra = [cand.endpoint_1[0], cand.endpoint_2[0]]
    ends_dec = [cand.endpoint_1[1], cand.endpoint_2[1]]

    # Left: the sky map, the great circle, the window, the endpoints
    sky_dm = cand.sky_map["distance_modulus"]
    _, sky_ax = plot_generic_search_residual_map(
        cand.sky_map, ax=sky_panel, title=f"Loaded box, dm {sky_dm:.2f}", **(sky_kwargs or {})
    )
    phi1_circle = np.linspace(-180.0, 180.0, 721)
    draw_stream_track_on_sky(sky_ax, *phi12_to_radec(phi1_circle, np.zeros_like(phi1_circle), cand.stream_frame),
                             label="great circle")
    draw_stream_track_on_sky(sky_ax, *phi12_to_radec(*phi12_window_outline(cand.window), cand.stream_frame),
                             label="window", **WINDOW_STYLE)
    sky_ax.plot(ends_ra, ends_dec, label="endpoints", **ENDPOINT_STYLE)
    sky_ax.legend(loc="upper right", fontsize="small", framealpha=0.8)

    # Right: the map in the frame, phi2 = 0, the endpoints
    zoom_dm = cand.zoom_map["distance_modulus"]
    _, zoom_ax = plot_stream_frame_search_single_dm(
        cand.zoom_map, ax=zoom_panel, title=f"In the frame, dm {zoom_dm:.2f}", **(zoom_kwargs or {})
    )
    draw_stream_track(zoom_ax, cand.window[:2], [0.0, 0.0], label=r"$\phi_2 = 0$")
    zoom_ax.plot(*radec_to_phi12(ends_ra, ends_dec, cand.stream_frame), label="endpoints", **ENDPOINT_STYLE)
    zoom_ax.legend(loc="upper right", fontsize="small", framealpha=0.8)

    fig.suptitle(f"{escape_underscores(cand.name)}: endpoints ({ends_ra[0]:g}, {ends_dec[0]:g}) and ({ends_ra[1]:g}, {ends_dec[1]:g}) deg")
    return fig, [sky_ax, zoom_ax]


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

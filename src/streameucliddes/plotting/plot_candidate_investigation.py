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

from streameucliddes.analysis.candidate_investigation import stream_width_model
from streameucliddes.plotting import plot_style  # noqa: F401  (sets the figure style)
from streameucliddes.plotting.plot_generic_search import plot_generic_search_residual_map
from streameucliddes.plotting.plot_stream_frame_search import (
    FOOTPRINT_OUTSIDE_GREY,
    draw_stream_track,
    draw_stream_track_on_sky,
    plot_stream_frame_search_single_dm,
)
from streameucliddes.plotting.plot_style import CATEGORICAL_COLORS, escape_underscores, format_distance_modulus
from streameucliddes.stream_frames import phi12_to_radec, phi12_window_outline, radec_to_phi12

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from numpy.typing import ArrayLike

    from streameucliddes.analysis.candidate_investigation import CandidateInvestigation

WINDOW_STYLE = {"color": CATEGORICAL_COLORS[1], "lw": 1.0, "ls": "-", "alpha": 0.9}  # outline of the window
ENDPOINT_STYLE = {"ls": "none", "marker": "o", "ms": 7, "mfc": "none", "mec": CATEGORICAL_COLORS[4], "mew": 1.5}
FRAME_CHECK_FIGSIZE = (17.0, 6.5)
PROFILE_FIGSIZE = (8.0, 4.5)
WIDTH_MAP_FIGSIZE = (15.0, 7.0)
DENSITY_PERCENTILES = (2.0, 98.0)  # colour scale of the observed and model densities
RESIDUAL_PERCENTILE = 98.0  # symmetric limit of the residual: this percentile of |observed - model|
RESIDUAL_CMAP = "RdBu_r"
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

    Points with Poisson error bars; the fitted range shaded, the model in it
    (dashed beyond, where it was not fitted), its centre and ``±sigma``
    (dotted), and the values in the legend. The title gives the phi1 range.

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
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=PROFILE_FIGSIZE)
    ax.errorbar(profile["centres"], profile["density"], yerr=profile["error"], fmt="o", ms=3, color=PROFILE_COLOR,
                elinewidth=0.8, label="stars in the match filter")
    if fit is not None:
        errors = fit["errors"]
        centres, (fit_min, fit_max) = profile["centres"], fit["phi2_fit_range"]
        in_range = (centres >= fit_min) & (centres <= fit_max)
        ax.axvspan(fit_min, fit_max, color=FIT_COLOR, alpha=0.08, lw=0, label="fitted range")
        ax.plot(centres, np.where(in_range, fit["model"], np.nan), color=FIT_COLOR, lw=1.5, label=(
            rf"Gaussian + {_background_name(fit)}: $\phi_2$ = {fit['phi2_centre']:.2f} $\pm$ {errors['phi2_centre']:.2f}, "
            rf"$\sigma$ = {fit['sigma']:.2f} $\pm$ {errors['sigma']:.2f} deg"))
        ax.plot(centres, fit["model"], color=FIT_COLOR, lw=0.8, ls="--")
        ax.axvline(fit["phi2_centre"], color=FIT_COLOR, lw=0.8)
        for edge in (fit["phi2_centre"] - fit["sigma"], fit["phi2_centre"] + fit["sigma"]):
            ax.axvline(edge, color=FIT_COLOR, lw=0.8, ls=":")
    phi1_min, phi1_max = profile["windows"][0][:2]
    ax.set(xlabel=r"$\phi_2$ [deg]", ylabel=r"Density [stars deg$^{-2}$]",
           title=rf"$\phi_1$ from {phi1_min:.2f} to {phi1_max:.2f} deg")
    ax.legend(loc="upper right", fontsize="small", framealpha=0.8)
    return fig, ax


def plot_width_fit_map(
    width_map: Mapping[str, Any],
    fit: Mapping[str, Any],
    *,
    axes: Sequence[Axes] | None = None,
) -> tuple[Figure, list[Axes]]:
    """The width fit in 2D: observed density, model, observed - model, in phi1/phi2.

    The model (``analysis.candidate_investigation.stream_width_model``) is
    constant along phi1: the residual shows whether the stream follows the
    straight track over the phi1 range, and where the field is not
    constant. Observed and model share one colour scale (stars/deg²); the
    residual a diverging one, symmetric. Cells without area are grey; the
    fitted phi2 range is marked by dotted lines, the centre by a solid one.

    Parameters
    ----------
    width_map : mapping
        From ``analysis.candidate_investigation.phi12_density_map``, e.g.
        ``cand.width_map``.
    fit : mapping
        From ``fit_stream_width``, e.g. ``cand.width_fit``.
    axes : sequence of three matplotlib.axes.Axes, optional
        Default: a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=WIDTH_MAP_FIGSIZE, layout="constrained")
    fig = axes[0].figure
    phi1_edges, phi2_edges = width_map["phi1_edges"], width_map["phi2_edges"]
    observed = width_map["density"]
    phi2_centres = (phi2_edges[:-1] + phi2_edges[1:]) / 2
    model = np.where(np.isfinite(observed), stream_width_model(fit, phi2_centres)[None, :], np.nan)
    residual = observed - model
    low, high = np.nanpercentile(observed, DENSITY_PERCENTILES)
    limit = float(np.nanpercentile(np.abs(residual), RESIDUAL_PERCENTILE))
    extent = (phi1_edges[0], phi1_edges[-1], phi2_edges[0], phi2_edges[-1])
    panels = [
        (observed, "Observed", "gray_r", low, high),
        (model, f"Model: Gaussian + {_background_name(fit)}", "gray_r", low, high),
        (residual, "Observed $-$ model", RESIDUAL_CMAP, -limit, limit),
    ]
    for ax, (values, title, cmap, vmin, vmax) in zip(axes, panels):
        colormap = plt.get_cmap(cmap).copy()
        colormap.set_bad(FOOTPRINT_OUTSIDE_GREY)
        image = ax.imshow(np.ma.masked_invalid(values).T, origin="lower", extent=extent, cmap=colormap, vmin=vmin,
                          vmax=vmax, aspect="equal", interpolation="nearest")
        fig.colorbar(image, ax=ax, label=r"stars deg$^{-2}$", shrink=0.8)
        for edge in fit["phi2_fit_range"]:
            ax.axhline(edge, color=FIT_COLOR, lw=0.8, ls=":")
        ax.axhline(fit["phi2_centre"], color=FIT_COLOR, lw=0.8)
        ax.set(xlim=(extent[1], extent[0]), ylim=extent[2:], xlabel=r"$\phi_1$ [deg]", ylabel=r"$\phi_2$ [deg]",
               title=title)  # phi1 increasing to the left, as the maps
    return fig, list(axes)


def _background_name(fit: Mapping[str, Any]) -> str:
    """Name of the background of a width fit: constant, linear, or its degree."""
    degree = fit["background_degree"]
    return {0: "constant", 1: "linear"}.get(degree, f"degree-{degree} polynomial")


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
        cand.sky_map, ax=sky_panel, title=f"Loaded box, dm {format_distance_modulus(sky_dm)}", **(sky_kwargs or {})
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
        cand.zoom_map, ax=zoom_panel, title=f"In the frame, dm {format_distance_modulus(zoom_dm)}", **(zoom_kwargs or {})
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

    ``plot_stream_frame_search_single_dm``, ``draw_stream_track`` of
    ``cand.track_phi12()`` (the arc between the endpoints before the regions)
    and ``draw_stream_regions``. The title gives the distance modulus of the
    map drawn.

    Parameters
    ----------
    cand : CandidateInvestigation
        With ``maps`` (or ``zoom_map``) computed.
    distance_modulus : float, optional
        Map of ``cand.maps`` closest to it. Default: ``cand.distance_modulus``;
        without it or without ``maps``, ``cand.zoom_map``.
    ax : matplotlib.axes.Axes, optional
        Default: a new figure.
    show_regions : bool, optional
        Draw the regions, when defined.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        No map: neither ``maps`` nor a ``zoom_map`` at this distance modulus.
    """
    fig, ax = plot_stream_frame_search_single_dm(_candidate_map(cand, distance_modulus), ax=ax)
    draw_stream_track(ax, *cand.track_phi12(), label="track" if cand.regions is not None else "endpoints arc")
    if show_regions and cand.regions is not None:
        draw_stream_regions(ax, cand.regions)
    ax.legend(loc="upper right", fontsize="small", framealpha=0.8)
    return fig, ax


def _candidate_map(cand: CandidateInvestigation, distance_modulus: float | None) -> Mapping[str, Any]:
    """Map of ``cand.maps`` closest to ``distance_modulus`` (default ``cand.distance_modulus``), else the zoom map."""
    if distance_modulus is None:
        distance_modulus = cand.distance_modulus
    if distance_modulus is not None and cand.maps:
        return cand.maps[min(cand.maps, key=lambda dm: abs(dm - distance_modulus))]
    zoom = cand.zoom_map
    if zoom is not None and (distance_modulus is None or np.isclose(zoom["distance_modulus"], distance_modulus)):
        return zoom
    raise ValueError("No map at this distance modulus: call cand.scan_distance_moduli() or cand.compute_zoom_map().")




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

"""Figures of a stream-candidate investigation (analysis.candidate_investigation).

Two kinds of functions:

- regions, profiles and photometric errors (``analysis.candidate_investigation``,
  ``plot_photometric_errors``):
  ``draw_stream_regions`` draws on existing axes and returns what it added;
  the ``plot_*`` functions make or take axes and return ``(fig, ax)``. The
  tracks are drawn by ``plot_stream_frame_search.draw_stream_track`` and
  ``draw_stream_track_on_sky``, the same for every map;
- thin wrappers, ``plot_candidate_*``: each takes a
  ``CandidateInvestigation`` and calls the plots of ``plot_generic_search``,
  ``plot_stream_frame_search``, ``plot_cmd`` and the functions above on its
  results; ``build_gif``, its maps over distance moduli as frames and a GIF,
  with its track and the known streams and other candidates
  (``draw_other_tracks``) in another colour;
- the list of candidates on the sky (``catalogs.stream_candidates``):
  ``plot_candidates`` draws the listed candidates over a skyproj sky map
  (the arc between the two endpoints of each, its endpoints and its id, in
  the colour of its status), and
  ``EndpointPicker`` gives the (ra, dec) of two clicks on such a map, the
  endpoints of a new candidate (needs ``%matplotlib widget``, from ipympl).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patheffects import withStroke
from scipy.ndimage import gaussian_filter

from streameucliddes.analysis.candidate_investigation import region_windows
from streameucliddes.analysis.fit_stream import stream_width_model
from streameucliddes.analysis.stream_frame_search import stream_frame_search_dm_scan
from streameucliddes.catalogs.survey_footprint import in_healpix_mask
from streameucliddes.catalogs.stream_candidates import (
    DEFAULT_CANDIDATES_PATH,
    STATUS_COLUMN,
    candidate_endpoints,
    load_candidates,
)
from streameucliddes.plotting import plot_style  # noqa: F401  (sets the figure style)
from streameucliddes.plotting.plot_cmd import (
    DEFAULT_DIFFERENCE_SMOOTHING_BINS,
    HESS_DIFFERENCE_PERCENTILE,
    plot_hess,
    plot_isochrone,
    plot_match_filter_polygon,
)
from streameucliddes.plotting.make_videos import make_gif_from_plots
from streameucliddes.plotting.plot_generic_search import plot_generic_search_residual_map
from streameucliddes.plotting.plot_stream_frame_search import (
    FOOTPRINT_OUTSIDE_GREY,
    _single_dm_title,
    _view_limits,
    bins_in_phi12_range,
    draw_stream_track,
    draw_stream_track_on_sky,
    plot_stream_frame_search_single_dm,
)
from streameucliddes.plotting.plot_style import (
    COLORBAR_LABEL_SIZE,
    CATEGORICAL_COLORS,
    escape_underscores,
    format_distance_modulus,
)
from streameucliddes.stream_frames import (
    phi12_to_radec,
    phi12_window_outline,
    radec_to_phi12,
    stream_frame_from_endpoints,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from numpy.typing import ArrayLike, NDArray

    from streameucliddes.analysis.candidate_investigation import CandidateInvestigation

WINDOW_STYLE = {"color": CATEGORICAL_COLORS[1], "lw": 1.0, "ls": "-", "alpha": 0.9}  # outline of the window
ENDPOINT_STYLE = {"ls": "none", "marker": "o", "ms": 7, "mfc": "none", "mec": CATEGORICAL_COLORS[4], "mew": 1.5}
FRAME_CHECK_FIGSIZE = (17.0, 6.5)
PROFILE_FIGSIZE = (8.0, 4.5)
WIDTH_MAP_FIGSIZE = (15.0, 7.0)
OVERVIEW_FIGSIZE = (16.0, 11.0)
DENSITY_PERCENTILES = (2.0, 98.0)  # colour scale of the observed and model densities
RESIDUAL_PERCENTILE = 98.0  # symmetric limit of the residual: this percentile of |observed - model|
RESIDUAL_CMAP = "RdBu_r"
ON_REGION_COLOR = CATEGORICAL_COLORS[1]
OFF_REGION_COLOR = CATEGORICAL_COLORS[2]
PROFILE_COLOR = CATEGORICAL_COLORS[3]
ERRORS_FIGSIZE = (6.5, 4.5)
# plot_photometric_errors: the reference band, the colour band
ERROR_COLORS = (CATEGORICAL_COLORS[0], CATEGORICAL_COLORS[1])
# The streams masked (cand.stream_healpix_mask) on the maps: hatched, outlined, the map still seen below
STREAM_MASK_COLOR = "0.15"
STREAM_MASK_HATCH = "////"
STREAM_MASK_LABEL = "masked streams"
FIT_COLOR = CATEGORICAL_COLORS[4]
# The known streams and the other candidates on the maps in the frame (cand.other_tracks_phi12): one colour, not
# that of the track of the candidate (plot_stream_frame_search.TRACK_COLOR) nor those of the regions and endpoints
OTHER_TRACK_COLOR = CATEGORICAL_COLORS[5]
OTHER_TRACK_STYLE = {"color": OTHER_TRACK_COLOR, "lw": 1.2, "ls": "--", "alpha": 0.9}
OTHER_TRACKS_LABEL = "other streams and candidates"
# build_gif: distance moduli around that of the candidate, time per frame, resolution of the frames
GIF_DM_HALF_RANGE = 1.0
GIF_DM_STEP = 0.1
GIF_FRAME_DURATION_MS = 300
GIF_DPI = 110
GIF_FRAMES_FOLDER = "frames"
# The list of candidates on the sky
CANDIDATE_COLOR = CATEGORICAL_COLORS[1]  # listed candidates (the known streams: CATEGORICAL_COLORS[0])
CANDIDATE_ARC_STYLE = {"color": CANDIDATE_COLOR, "lw": 1.5, "ls": "-", "alpha": 0.9}
CANDIDATE_ENDPOINT_STYLE = {"ls": "none", "marker": "o", "ms": 4, "color": CANDIDATE_COLOR}
# One colour per status of the candidates (stream_candidates.STATUS_COLUMN); without a status: CANDIDATE_NO_STATUS_COLOR
CANDIDATE_STATUS_COLORS = {"strong": CANDIDATE_COLOR, "good": CATEGORICAL_COLORS[2], "bad": CATEGORICAL_COLORS[4]}
CANDIDATE_NO_STATUS_COLOR = CATEGORICAL_COLORS[5]
PICK_ARC_STYLE = {"color": ENDPOINT_STYLE["mec"], "lw": 1.5, "ls": "-", "alpha": 0.9}  # endpoints being picked:
# ENDPOINT_STYLE, as the endpoints of the frame check
ARC_POINTS = 100
LABEL_FONTSIZE = 9  # ids of the candidates, outlined in white to stand out on the map
LEFT_BUTTON, RIGHT_BUTTON = 1, 3
# Press and release closer than this: a click (an endpoint), not a drag (zoom or pan). Below matplotlib's
# 5 pixels, under which the zoom tool does not zoom: a click never zooms.
CLICK_TOLERANCE_PIXELS = 4.0


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
    """Outlines of the on region and of the two off regions on a phi1/phi2 map.

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
    windows = region_windows(regions)
    lines = draw_stream_track(ax, *phi12_window_outline(windows["on"]), label="on", color=on_color, ls="-", alpha=0.9)
    for index, window in enumerate(windows["off"]):
        lines += draw_stream_track(ax, *phi12_window_outline(window), label="off" if index == 0 else None,
                                   color=off_color, ls="-", alpha=0.9)
    return lines


def draw_stream_mask(
    ax: Axes,
    cand: CandidateInvestigation,
    phi1_centres: ArrayLike,
    phi2_centres: ArrayLike,
) -> list[Artist]:
    """The streams masked for a candidate (``cand.stream_healpix_mask``), hatched and outlined on a phi1/phi2 map.

    The bins of the map whose centre is on the mask: the stars left out of
    the width, the regions, the SNR and the Hess diagrams (the map itself
    keeps them). Nothing is drawn without ``mask_streams``, before
    ``prepare`` or when the mask misses the bins.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of a phi1/phi2 map of the candidate.
    cand : CandidateInvestigation
        With ``stream_healpix_mask`` (``prepare``).
    phi1_centres, phi2_centres : array_like
        Bin centres of the map (deg).

    Returns
    -------
    artists : list of matplotlib.artist.Artist
        The hatched area, its outline and the legend entry (``STREAM_MASK_LABEL``); empty when nothing is drawn.
    """
    if not cand.mask_streams or cand.stream_healpix_mask is None:
        return []
    phi1_centres, phi2_centres = np.asarray(phi1_centres, dtype=float), np.asarray(phi2_centres, dtype=float)
    grid_1, grid_2 = np.meshgrid(phi1_centres, phi2_centres, indexing="ij")
    on_mask = in_healpix_mask(*phi12_to_radec(grid_1.ravel(), grid_2.ravel(), cand.stream_frame),
                              cand.stream_healpix_mask, nest=False).reshape(grid_1.shape).astype(float)
    if not on_mask.any():
        return []
    with plt.rc_context({"hatch.color": STREAM_MASK_COLOR, "hatch.linewidth": 0.6}):
        fill = ax.contourf(phi1_centres, phi2_centres, on_mask.T, levels=[0.5, 1.5], colors="none",
                           hatches=[STREAM_MASK_HATCH])
    outline = ax.contour(phi1_centres, phi2_centres, on_mask.T, levels=[0.5], colors=STREAM_MASK_COLOR,
                         linewidths=0.8)
    (entry,) = ax.fill([], [], facecolor="none", edgecolor=STREAM_MASK_COLOR, hatch=STREAM_MASK_HATCH,
                       label=STREAM_MASK_LABEL)
    return [fill, outline, entry]


def draw_other_tracks(
    ax: Axes,
    tracks: Sequence[tuple[str, ArrayLike, ArrayLike]],
) -> list[Artist]:
    """The known streams and the other candidates on a phi1/phi2 map, in ``OTHER_TRACK_COLOR``, each with its name.

    One legend entry for all (``OTHER_TRACKS_LABEL``); the name of each a
    quarter of the way along its points in the window (not in the middle: a
    track crossing the candidate crosses it near the centre of the map),
    outlined in white.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of a phi1/phi2 map of the candidate.
    tracks : sequence of (str, array_like, array_like)
        ``(name, phi1, phi2)`` (deg; NaN outside the window), e.g.
        ``cand.other_tracks_phi12()``.

    Returns
    -------
    artists : list of matplotlib.artist.Artist
        The lines and the names; empty without tracks.
    """
    artists: list[Artist] = []
    for index, (name, phi1, phi2) in enumerate(tracks):
        phi1, phi2 = np.asarray(phi1, dtype=float), np.asarray(phi2, dtype=float)
        artists += draw_stream_track(ax, phi1, phi2, label=OTHER_TRACKS_LABEL if index == 0 else None,
                                     **OTHER_TRACK_STYLE)
        shown = np.flatnonzero(np.isfinite(phi1) & np.isfinite(phi2))
        if shown.size:
            point = shown[shown.size // 4]
            artists.append(ax.text(phi1[point], phi2[point], f" {escape_underscores(name)}",
                                   color=OTHER_TRACK_COLOR, fontsize=LABEL_FONTSIZE, fontweight="bold", ha="left",
                                   va="bottom", clip_on=True,
                                   path_effects=[withStroke(linewidth=2.5, foreground="white")]))
    return artists


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
    phi1_range: tuple[float, float] | None = None,
    phi2_range: tuple[float, float] | None = None,
) -> tuple[Figure, list[Axes]]:
    """The width fit in 2D: observed density, model, observed - model, in phi1/phi2.

    The model (``analysis.fit_stream.stream_width_model``) is
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
    phi1_range, phi2_range : tuple of float, optional
        Part of the maps shown (deg), e.g. to leave out a dense object: the
        axis limits and the cells of the colour scales (those whose centre is
        inside, ``plot_stream_frame_search.bins_in_phi12_range``). Default:
        the whole map.

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
    phi1_centres, phi2_centres = (phi1_edges[:-1] + phi1_edges[1:]) / 2, (phi2_edges[:-1] + phi2_edges[1:]) / 2
    model = np.where(np.isfinite(observed), stream_width_model(fit, phi2_centres)[None, :], np.nan)
    residual = observed - model
    extent = (phi1_edges[0], phi1_edges[-1], phi2_edges[0], phi2_edges[-1])
    view = _view_limits(extent, phi1_range, phi2_range)
    shown = bins_in_phi12_range(phi1_centres, phi2_centres, view[:2], view[2:])  # the colour scales: these cells
    low, high = np.nanpercentile(observed[shown], DENSITY_PERCENTILES)
    limit = float(np.nanpercentile(np.abs(residual[shown]), RESIDUAL_PERCENTILE))
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
        fig.colorbar(image, ax=ax, shrink=0.8).set_label(r"stars deg$^{-2}$", fontsize=COLORBAR_LABEL_SIZE)
        for edge in fit["phi2_fit_range"]:
            ax.axhline(edge, color=FIT_COLOR, lw=0.8, ls=":")
        ax.axhline(fit["phi2_centre"], color=FIT_COLOR, lw=0.8)
        ax.set(xlim=(view[1], view[0]), ylim=view[2:], xlabel=r"$\phi_1$ [deg]", ylabel=r"$\phi_2$ [deg]",
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
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=PROFILE_FIGSIZE)
    if regions is not None:
        ax.axvspan(*regions["phi1_range"], color=ON_REGION_COLOR, alpha=0.1, lw=0, label=r"regions $\phi_1$ range")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.errorbar(profile["centres"], profile["difference"], yerr=profile["error"], fmt="o", ms=3, color=PROFILE_COLOR,
                elinewidth=0.8, label="on $-$ off")
    ax.set(xlim=(profile["edges"][-1], profile["edges"][0]),  # phi1 increasing to the left, as the maps
           xlabel=r"$\phi_1$ [deg]", ylabel=r"On $-$ off density [stars deg$^{-2}$]")
    ax.legend(loc="upper right", fontsize="small", framealpha=0.8)
    return fig, ax

def plot_photometric_errors(
    errors: Mapping[str, Any],
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Photometric error of each band against its own magnitude (median or mean), on the same axes.

    Draws ``analysis.cmd_match_filter.photometric_errors_by_magnitude`` (e.g.
    ``cand.photometric_errors``): one line per band (VIS against VIS, H
    against H), one point per bin with stars.

    Parameters
    ----------
    errors : mapping
        From ``photometric_errors_by_magnitude``.
    ax : matplotlib.axes.Axes, optional
        Default: a new figure.
    title : str, optional
        Default: the number of stars (of the reference band).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=ERRORS_FIGSIZE, layout="constrained")
    else:
        fig = ax.figure
    centres = np.asarray(errors["mag_centres"], dtype=float)
    for band, color in zip(errors["bands"], ERROR_COLORS):
        values = np.asarray(errors["error"][band], dtype=float)
        shown = np.isfinite(values)
        ax.plot(centres[shown], values[shown], marker="o", ms=3, color=color, label=f"{band} (against {band})")
    ax.set_xlabel("magnitude [mag]")
    ax.set_ylabel(f"{errors['statistic']} error [mag]")
    ax.legend(loc="upper left", fontsize="small")
    n_stars = int(np.sum(errors["n_stars"][errors["bands"][0]]))
    ax.set_title(f"{n_stars:,} stars" if title is None else title)
    return fig, ax


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
        Arguments of ``plot_stream_frame_search_single_dm``, e.g.
        ``{"phi1_range": (-5, 5), "phi2_range": (-2, 2)}`` to leave out a
        dense object (limits and colour scale).

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
    show_stream_mask: bool = True,
    other_tracks: Sequence[tuple[str, ArrayLike, ArrayLike]] | None = None,
    map_result: Mapping[str, Any] | None = None,
    phi1_range: tuple[float, float] | None = None,
    phi2_range: tuple[float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[Figure, Axes]:
    """Matched-filter map in the frame, with the track and the regions.

    ``plot_stream_frame_search_single_dm``, ``draw_stream_track`` of
    ``cand.track_phi12()`` (the arc between the endpoints before the regions),
    ``draw_stream_regions``, ``draw_stream_mask`` (the other streams
    masked, hatched) and ``draw_other_tracks`` (the known streams and the
    other candidates, when given). The title gives the distance modulus of
    the map drawn.

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
    show_stream_mask : bool, optional
        Hatch the streams masked (``draw_stream_mask``).
    other_tracks : sequence of (str, array_like, array_like), optional
        The known streams and the other candidates, ``(name, phi1, phi2)``,
        e.g. ``cand.other_tracks_phi12()`` (``draw_other_tracks``). Default:
        none drawn.
    map_result : mapping, optional
        A map of ``stream_frame_search_single_dm`` (or of a scan) in the
        frame of the candidate, drawn instead of one of ``cand.maps``
        (``distance_modulus`` is then not used), e.g. those of ``build_gif``.
    phi1_range, phi2_range : tuple of float, optional
        Part of the map shown (deg), e.g. to leave out a dense object: the
        axis limits and the colour scale (``plot_stream_frame_search_single_dm``).
        Default: the whole window.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        No map: neither ``maps`` nor a ``zoom_map`` at this distance modulus.
    """
    result = _candidate_map(cand, distance_modulus) if map_result is None else map_result
    fig, ax = plot_stream_frame_search_single_dm(result, ax=ax, phi1_range=phi1_range, phi2_range=phi2_range,
                                                 vmin=vmin, vmax=vmax)
    if show_stream_mask:
        draw_stream_mask(ax, cand, result["phi1_centres"], result["phi2_centres"])
    draw_stream_track(ax, *cand.track_phi12(), label="track" if cand.regions is not None else "endpoints arc")
    if show_regions and cand.regions is not None:
        draw_stream_regions(ax, cand.regions)
    if other_tracks:
        draw_other_tracks(ax, other_tracks)
    ax.legend(loc="upper right", fontsize="small", framealpha=0.8)
    return fig, ax


def _candidate_map(cand: CandidateInvestigation, distance_modulus: float | None) -> Mapping[str, Any]:
    """Map at ``distance_modulus`` (default ``cand.distance_modulus``).

    The map of ``cand.maps`` closest to it, within one step of their grid
    (only that distance modulus for a single map), else the zoom map at that
    distance modulus (or without one given).

    Raises
    ------
    ValueError
        No map at or near this distance modulus.
    """
    if distance_modulus is None:
        distance_modulus = cand.distance_modulus
    if distance_modulus is not None and cand.maps:
        grid = np.sort(np.asarray(list(cand.maps), dtype=float))
        step = float(np.min(np.diff(grid))) if grid.size > 1 else 0.0
        closest = min(cand.maps, key=lambda dm: abs(dm - distance_modulus))
        if abs(closest - distance_modulus) <= step + 1e-9:
            return cand.maps[closest]
    zoom = cand.zoom_map
    if zoom is not None and (distance_modulus is None or np.isclose(zoom["distance_modulus"], distance_modulus)):
        return zoom
    scanned = f" (the scanned maps go from {min(cand.maps):g} to {max(cand.maps):g})" if cand.maps else ""
    at = "no distance modulus" if distance_modulus is None else f"dm {distance_modulus:g}"
    raise ValueError(f"No map at {at}{scanned}: call cand.scan_distance_moduli() with it, or cand.compute_zoom_map().")




def plot_candidate_hess(
    cand: CandidateInvestigation,
    which: str = "difference",
    *,
    ax: Axes | None = None,
    isochrone: bool = True,
    match_filter: bool = False,
    smooth: bool = True,
) -> tuple[Figure, Axes]:
    """One Hess diagram of the candidate, with its isochrone on top.

    The difference is smoothed by ``plot_cmd.DEFAULT_DIFFERENCE_SMOOTHING_BINS``
    (only the figure), unless ``smooth`` is False.

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
    smooth : bool, optional
        Smooth the difference diagram.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        ``which`` not one of the three, or the Hess diagrams not computed.
    """
    diagrams = {"on": cand.on_hess, "off": cand.off_hess, "difference": cand.diff_hess}
    if which not in diagrams:
        raise ValueError(f"which must be one of {sorted(diagrams)}, not {which!r}.")
    if diagrams[which] is None:
        raise ValueError("No Hess diagrams: call cand.compute_hess() first.")
    smoothing_bins = DEFAULT_DIFFERENCE_SMOOTHING_BINS if smooth and which == "difference" else 0.0
    title = f"{escape_underscores(cand.name)}: {which}" + (f", smoothed ({smoothing_bins:g} bin{'' if smoothing_bins == 1 else 's'})" if smoothing_bins else "")
    fig, ax = plot_hess(diagrams[which], ax=ax, difference=which == "difference", smoothing_bins=smoothing_bins,
                        title=f"{title}, dm {format_distance_modulus(cand.distance_modulus)}")
    if isochrone and cand.isochrone_track is not None:
        plot_isochrone(ax, cand.isochrone_track)
    if match_filter:
        plot_match_filter_polygon(ax, cand.distance_modulus, cand.isochrone, cand.search.match_filter)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", fontsize="small", framealpha=0.8)
    return fig, ax


def plot_candidate_null_test(
    cand: CandidateInvestigation,
    *,
    isochrone: bool = True,
    match_filter: bool = True,
    smooth: bool = True,
) -> tuple[Figure, list[Axes]]:
    """Null test: the map with the shifted regions, and the on - off Hess diagrams of the candidate and of the null test.

    Left: ``plot_candidate_map`` with the regions of ``cand.null_test``
    (``compute_null_test``). Middle and right: the difference Hess diagrams
    of the candidate (``compute_hess``) and of the null test, on the same
    colour scale (the ``HESS_DIFFERENCE_PERCENTILE`` of the candidate's), each
    with its SNR; without the stream, the null test should show no isochrone.

    Parameters
    ----------
    cand : CandidateInvestigation
        With ``compute_hess`` and ``compute_null_test`` done.
    isochrone, match_filter, smooth : bool, optional
        See ``plot_candidate_hess``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        The map, the candidate's Hess diagram, the null test's.

    Raises
    ------
    ValueError
        The Hess diagrams or the null test not computed.
    """
    if cand.diff_hess is None:
        raise ValueError("No Hess diagrams: call cand.compute_hess() first.")
    null_test = cand.null_test
    if null_test is None:
        raise ValueError("No null test: call cand.compute_null_test() first.")
    smoothing_bins = DEFAULT_DIFFERENCE_SMOOTHING_BINS if smooth else 0.0
    density = np.asarray(cand.diff_hess["density"], dtype=float)
    density = gaussian_filter(density, smoothing_bins) if smoothing_bins > 0 else density
    values = np.abs(density[np.isfinite(density)])
    vlim = float(np.percentile(values, HESS_DIFFERENCE_PERCENTILE)) if values.size else 1.0
    vlim = vlim if vlim > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.5), layout="constrained",
                             gridspec_kw={"width_ratios": (1.4, 1.0, 1.0)})
    plot_candidate_map(cand, ax=axes[0], show_regions=False)
    draw_stream_regions(axes[0], null_test["regions"])
    axes[0].legend(loc="upper right", fontsize="small", framealpha=0.8)
    compact = ", off region next to it" if null_test.get("geometry") == "compact" else ""
    axes[0].set_title(f"Null test: regions shifted {null_test['direction']} by {null_test['shift_deg']:.2f} deg{compact}")
    snr = cand.summary()["snr"]
    for ax, diff_hess, label in ((axes[1], cand.diff_hess, f"candidate, SNR {snr:.1f}"),
                                 (axes[2], null_test["diff_hess"], f"null test, SNR {null_test['snr']:.1f}")):
        plot_hess(diff_hess, ax=ax, difference=True, smoothing_bins=smoothing_bins, vlim=vlim,
                  title=f"{escape_underscores(cand.name)}: on $-$ off, {label}")
        if isochrone and cand.isochrone_track is not None:
            plot_isochrone(ax, cand.isochrone_track)
        if match_filter:
            plot_match_filter_polygon(ax, cand.distance_modulus, cand.isochrone, cand.search.match_filter)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", fontsize="small", framealpha=0.8)
    return fig, list(axes)


def plot_candidate_overview(cand: CandidateInvestigation) -> tuple[Figure, list[Axes]]:
    """Summary figure: map with the regions, phi2 profile, phi1 profile, difference Hess with the isochrone.

    ``plot_candidate_map``, ``plot_phi2_profile`` (with the width fit),
    ``plot_phi1_profile`` and ``plot_candidate_hess``, on a 2 x 2 grid; the
    title gives the key numbers of ``cand.summary()``.

    Parameters
    ----------
    cand : CandidateInvestigation
        With every step done (``run_all_steps``); the isochrone is drawn
        when computed.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        Map, phi2 profile, phi1 profile, difference Hess.

    Raises
    ------
    ValueError
        A step not done (the message names it).
    """
    for name, step in (("width_fit", "fit_width"), ("phi1_profile", "define_regions"), ("diff_hess", "compute_hess")):
        if getattr(cand, name) is None:
            raise ValueError(f"No {name}: call cand.{step}() first.")
    fig, axes = plt.subplots(2, 2, figsize=OVERVIEW_FIGSIZE, layout="constrained")
    map_ax, phi2_ax, phi1_ax, hess_ax = axes.flat
    plot_candidate_map(cand, ax=map_ax)
    plot_phi2_profile(cand.phi2_profile, cand.width_fit, ax=phi2_ax)
    plot_phi1_profile(cand.phi1_profile, cand.regions, ax=phi1_ax)
    plot_candidate_hess(cand, ax=hess_ax)
    summary = cand.summary()
    fig.suptitle(
        f"{escape_underscores(cand.name)}: dm {format_distance_modulus(summary['distance_modulus'])} "
        f"({summary['distance_kpc']:.1f} kpc), $\\sigma$ = {summary['sigma']:.2f} deg, "
        f"excess {summary['n_excess']:.0f} $\\pm$ {summary['n_excess_error']:.0f} stars in the match filter"
    )
    return fig, [map_ax, phi2_ax, phi1_ax, hess_ax]


def build_gif(
    cand: CandidateInvestigation,
    distance_moduli: Sequence[float] | None = None,
    *,
    dm_half_range: float = GIF_DM_HALF_RANGE,
    dm_step: float = GIF_DM_STEP,
    show_other_tracks: bool = True,
    file_prefix: str | None = None,
    frame_duration_ms: int = GIF_FRAME_DURATION_MS,
    dpi: int = GIF_DPI,
    scan_kwargs: Mapping[str, Any] | None = None,
    **map_kwargs: Any,
) -> tuple[Path, list[Path]]:
    """GIF of the candidate over distance moduli: its map in its frame at each, with its track and the other streams.

    The maps of ``stream_frame_search_dm_scan`` on the prepared stars, with
    the isochrone of the object (``cand.maps`` is left as it is), each drawn
    by ``plot_candidate_map`` with the track of the candidate
    (``cand.track_phi12``: the arc between the endpoints, or the fitted track
    after ``define_regions``) and, in ``OTHER_TRACK_COLOR``, the known
    streams and the other candidates in the window
    (``cand.other_tracks_phi12``, ``draw_other_tracks``). The frame at
    ``cand.distance_modulus`` says "best dm" in its title. Frames
    ``<file_prefix>_dm<dm>.png`` in ``cand.figure_folder / "frames"``, GIF
    ``<file_prefix>_dm_scan.gif`` in ``cand.figure_folder``
    (``make_videos.make_gif_from_plots``, from the frames of this call only:
    older frames of the folder are left out). Each frame has the colour scale
    of its own map (``plot_stream_frame_search_single_dm``), unless ``vmin``
    and ``vmax`` are given.

    Parameters
    ----------
    cand : CandidateInvestigation
        Prepared (``prepare``).
    distance_moduli : sequence of float, optional
        Distance moduli of the frames. Default: from ``cand.distance_modulus
        - dm_half_range`` to ``cand.distance_modulus + dm_half_range`` every
        ``dm_step`` (for a listed candidate, its best distance modulus:
        ``from_candidate_id``); without ``distance_modulus``, the grid of
        the scan (``cand.search.distance_moduli``: ``cand.distance_moduli``,
        else that of ``prepare_search``).
    dm_half_range, dm_step : float, optional
        See ``distance_moduli``.
    show_other_tracks : bool, optional
        Draw the known streams and the other candidates.
    file_prefix : str, optional
        Start of the names of the frames and of the GIF. Default:
        ``cand.name``; another one keeps the files of another isochrone in
        the same folder, e.g. ``cand.name + "_fitted_isochrone"``.
    frame_duration_ms : int, optional
        Time per frame in the GIF (ms).
    dpi : int, optional
        Resolution of the frames.
    scan_kwargs : mapping, optional
        Arguments of ``stream_frame_search_dm_scan`` (``smoothing_deg``,
        ``n_workers``).
    **map_kwargs
        Arguments of ``plot_candidate_map`` (``phi1_range``, ``phi2_range``,
        ``vmin``, ``vmax``, ``show_regions`` and ``show_stream_mask``, both
        False by default here).

    Returns
    -------
    gif_path : pathlib.Path
    frame_paths : list of pathlib.Path
        In the order of ``distance_moduli``.

    Raises
    ------
    ValueError
        ``prepare`` not done, or ``dm_step`` not positive.
    """
    if cand.search is None:
        raise ValueError("No prepared stars: call cand.prepare(catalog) first.")
    if distance_moduli is None and cand.distance_modulus is None:
        distance_moduli = cand.search.distance_moduli
    elif distance_moduli is None:
        if dm_step <= 0:
            raise ValueError(f"dm_step must be positive, not {dm_step}.")
        n_steps = int(round(dm_half_range / dm_step))
        distance_moduli = cand.distance_modulus + dm_step * np.arange(-n_steps, n_steps + 1)
    distance_moduli = [round(float(dm), 6) for dm in distance_moduli]  # 16.35, not 16.350000000000001
    file_prefix = cand.name if file_prefix is None else file_prefix
    map_kwargs = {"show_regions": False, "show_stream_mask": False, **map_kwargs}
    other_tracks = cand.other_tracks_phi12() if show_other_tracks else None
    folder = Path(cand.figure_folder)
    frames_folder = folder / GIF_FRAMES_FOLDER
    frames_folder.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for dm, result in stream_frame_search_dm_scan(search=cand.search, distance_moduli=distance_moduli,
                                                  isochrone=cand.isochrone, **(scan_kwargs or {})):
        fig, ax = plot_candidate_map(cand, map_result=result, other_tracks=other_tracks, **map_kwargs)
        best = cand.distance_modulus is not None and np.isclose(dm, cand.distance_modulus)
        ax.set_title(f"{escape_underscores(cand.name)}: {_single_dm_title(result)}" + (", best dm" if best else ""),
                     color=ax.title.get_color())
        frame_paths.append(frames_folder / f"{file_prefix}_dm{dm:05.2f}.png")  # 05.2f: the names sort as the dm
        fig.savefig(frame_paths[-1], dpi=dpi)
        plt.close(fig)
    gif_path = folder / f"{file_prefix}_dm_scan.gif"
    make_gif_from_plots(gif_path, pattern=frame_paths, duration=frame_duration_ms)
    return gif_path, frame_paths


# ===================================================
# The list of candidates on the sky
# ===================================================


def endpoints_arc_radec(
    endpoint_1: ArrayLike, endpoint_2: ArrayLike, n_points: int = ARC_POINTS
) -> tuple[NDArray, NDArray]:
    """(ra, dec) of the great-circle arc between two endpoints (deg).

    The phi2 = 0 line of their frame (``stream_frames.stream_frame_from_endpoints``),
    between their phi1.

    Parameters
    ----------
    endpoint_1, endpoint_2 : array_like
        (ra, dec) (deg).
    n_points : int, optional
        Number of points.

    Returns
    -------
    ra, dec : numpy.ndarray
        (deg).
    """
    frame = stream_frame_from_endpoints(tuple(endpoint_1), tuple(endpoint_2))
    phi1, _ = radec_to_phi12([endpoint_1[0], endpoint_2[0]], [endpoint_1[1], endpoint_2[1]], frame)
    arc_phi1 = np.linspace(phi1[0], phi1[1], n_points)
    return phi12_to_radec(arc_phi1, np.zeros(n_points), frame)


def plot_candidates(
    ax: Any,
    candidates: pd.DataFrame | None = None,
    *,
    path: str | Path = DEFAULT_CANDIDATES_PATH,
    include_bad_candidates: bool = False,
    label_ids: bool = True,
    arc_style: dict[str, Any] | None = None,
    endpoint_style: dict[str, Any] | None = None,
) -> list[Artist]:
    """Overplot the listed candidates on a sky map: arc between the endpoints, endpoints, id.

    Each in the colour of its status (``CANDIDATE_STATUS_COLORS``,
    ``CANDIDATE_NO_STATUS_COLOR`` without one; ``CANDIDATE_COLOR`` for all
    when the list has no ``STATUS_COLUMN``), one legend entry per status
    drawn (``"<status> candidates"``, ``"candidates"`` without a status).

    Parameters
    ----------
    ax : skyproj.SkyAxes
        Axes of a sky map, e.g. from ``plot_generic_search_residual_map``;
        its ``plot`` and ``text`` take (ra, dec) in degrees.
    candidates : pandas.DataFrame, optional
        From ``catalogs.stream_candidates.load_candidates`` (all drawn).
        Default: those of the file ``path``.
    path : str or pathlib.Path, optional
        CSV file read when ``candidates`` is not given.
    include_bad_candidates : bool, optional
        Also draw the bad candidates of ``path`` (status ``bad``,
        ``load_candidates``) when ``candidates`` is not given.
    label_ids : bool, optional
        Write the id of each candidate next to its first endpoint.
    arc_style, endpoint_style : dict, optional
        Arguments of ``ax.plot`` updating ``CANDIDATE_ARC_STYLE`` and
        ``CANDIDATE_ENDPOINT_STYLE``; a ``color`` given here is that of
        every candidate.

    Returns
    -------
    artists : list of matplotlib.artist.Artist
        The arcs, endpoints and labels added; empty when there is no
        candidate.
    """
    if candidates is None:
        candidates = load_candidates(path, include_bad_candidates=include_bad_candidates)
    artists: list[Artist] = []
    labelled = set()
    for _, candidate in candidates.iterrows():
        status = candidate.get(STATUS_COLUMN)
        status = None if pd.isna(status) else str(status)
        color = (CANDIDATE_STATUS_COLORS.get(status, CANDIDATE_NO_STATUS_COLOR)
                 if STATUS_COLUMN in candidates.columns else CANDIDATE_COLOR)
        label = "candidates" if status is None else f"{status} candidates"
        candidate_arc_style = {**CANDIDATE_ARC_STYLE, "color": color, **(arc_style or {})}
        candidate_endpoint_style = {**CANDIDATE_ENDPOINT_STYLE, "color": color, **(endpoint_style or {})}
        endpoint_1, endpoint_2 = candidate_endpoints(candidate)
        artists += draw_stream_track_on_sky(ax, *endpoints_arc_radec(endpoint_1, endpoint_2),
                                            label=None if label in labelled else label, **candidate_arc_style)
        labelled.add(label)
        artists += ax.plot([endpoint_1[0], endpoint_2[0]], [endpoint_1[1], endpoint_2[1]],
                           **candidate_endpoint_style)
        if label_ids:
            artists.append(ax.text(*endpoint_1, f" {int(candidate['id'])}", color=candidate_arc_style["color"],
                                   fontsize=LABEL_FONTSIZE, fontweight="bold", ha="left", va="bottom",
                                   path_effects=[withStroke(linewidth=2.5, foreground="white")]))
    return artists


def sky_axes_to_radec(ax: Any, x: ArrayLike, y: ArrayLike) -> tuple[NDArray, NDArray]:
    """(ra, dec) of points of a skyproj map given in its data coordinates (deg).

    The inverse of the projection of the map (``ax.projection``, as skyproj
    does for the coordinates shown under the cursor). Points outside the sky
    (beyond the edge of the projection) are NaN.

    Parameters
    ----------
    ax : skyproj.SkyAxes
        Axes of a sky map.
    x, y : array_like
        Data coordinates, e.g. ``event.xdata`` and ``event.ydata`` of a click.

    Returns
    -------
    ra, dec : numpy.ndarray
        (deg), ra in [0, 360).
    """
    x, y = np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
    lonlat = ax.projection.transform_points(x, y, inverse=True)
    back = ax.projection.transform_points(lonlat[:, 0], lonlat[:, 1])  # outside the sky: not the same point back
    outside = ~(np.isclose(back[:, 0], x) & np.isclose(back[:, 1], y))
    ra, dec = lonlat[:, 0] % 360.0, lonlat[:, 1].copy()
    ra[outside], dec[outside] = np.nan, np.nan
    return ra, dec


class EndpointPicker:
    """Two clicks on a sky map: the endpoints of a new candidate.

    Left click: an endpoint (marked); the second one draws the great-circle
    arc between them; a third click starts a new pair. Right click: removes
    the last endpoint. A click is a press and a release at the same place
    (within ``CLICK_TOLERANCE_PIXELS``), so it works with the zoom or pan
    tool of the toolbar on: a drag zooms or pans and adds no endpoint.
    Needs an interactive backend: ``%matplotlib widget`` (ipympl) in a
    notebook.

    Parameters
    ----------
    ax : skyproj.SkyAxes
        Axes of a sky map, e.g. from ``plot_generic_search_residual_map``.

    Attributes
    ----------
    points : list of tuple of float
        (ra, dec) of the endpoints clicked (deg), at most two.

    Examples
    --------
    >>> fig, ax = plot_generic_search_residual_map(result)  # doctest: +SKIP
    >>> picker = EndpointPicker(ax)  # doctest: +SKIP
    >>> # two clicks on the map, then:
    >>> add_candidate(*picker.endpoints, best_dm_by_eye=17.0)  # doctest: +SKIP
    """

    def __init__(self, ax: Any) -> None:
        self.ax = ax
        self.points: list[tuple[float, float]] = []
        self._artists: list[Artist] = []
        self._press: tuple[float, float, Any] | None = None  # (x, y) in pixels and button of the last press
        canvas = ax.figure.canvas
        self._connections = [canvas.mpl_connect("button_press_event", self._on_press),
                             canvas.mpl_connect("button_release_event", self._on_release)]

    @property
    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The two endpoints clicked, (ra, dec) each (deg).

        Raises
        ------
        ValueError
            Fewer than two endpoints clicked.
        """
        if len(self.points) < 2:
            raise ValueError(f"{len(self.points)} endpoint(s) clicked: click two points on the map.")
        return self.points[0], self.points[1]

    def add_point(self, ra: float, dec: float) -> None:
        """Add an endpoint, as a left click at (ra, dec) does (a third one starts a new pair)."""
        if len(self.points) == 2:
            self.points = []
        self.points.append((float(ra) % 360.0, float(dec)))
        self._redraw()

    def remove_last_point(self) -> None:
        """Remove the last endpoint, as a right click does."""
        if self.points:
            self.points.pop()
            self._redraw()

    def disconnect(self) -> None:
        """Stop listening to the clicks; the marks stay."""
        for connection in self._connections:
            self.ax.figure.canvas.mpl_disconnect(connection)

    def _on_press(self, event: MouseEvent) -> None:
        """Keep where a press in the map starts: a click if released at the same place."""
        self._press = (event.x, event.y, event.button) if event.inaxes is self.ax else None

    def _on_release(self, event: MouseEvent) -> None:
        """A click (press and release at the same place): left adds an endpoint, right removes the last one."""
        press, self._press = self._press, None
        if press is None or event.inaxes is not self.ax or event.xdata is None or event.button != press[2]:
            return
        if np.hypot(event.x - press[0], event.y - press[1]) > CLICK_TOLERANCE_PIXELS:  # a drag: zoom or pan
            return
        if event.button == RIGHT_BUTTON:
            self.remove_last_point()
        elif event.button == LEFT_BUTTON:
            ra, dec = sky_axes_to_radec(self.ax, event.xdata, event.ydata)
            if np.isfinite(ra[0]):
                self.add_point(ra[0], dec[0])

    def _redraw(self) -> None:
        """Marks of the endpoints, and their arc when there are two."""
        for artist in self._artists:
            artist.remove()
        self._artists = []
        if self.points:
            ra, dec = zip(*self.points)
            self._artists += self.ax.plot(ra, dec, **ENDPOINT_STYLE)
        if len(self.points) == 2:
            self._artists += draw_stream_track_on_sky(self.ax, *endpoints_arc_radec(*self.points), **PICK_ARC_STYLE)
        self.ax.figure.canvas.draw_idle()

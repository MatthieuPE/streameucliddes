"""Colour-magnitude diagrams, and the match filters drawn on them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

from streameucliddes.analysis.catalog_preparation import check_columns, prepare_match_filter_catalog
from streameucliddes.analysis.cmd_match_filter import (
    DEFAULT_MATCH_FILTER,
    hess_diagram,
    match_filter_columns,
    match_filter_polygons,
    select_stars_in_match_filter,
)
from streameucliddes.catalogs.isochrone_parameters import default_isochrone_grid
from streameucliddes.plotting import plot_style  # noqa: F401  (sets the figure style)
from streameucliddes.plotting.plot_style import COLORBAR_LABEL_SIZE, CATEGORICAL_COLORS, format_distance_modulus
from streameucliddes.unit_conversions import convert_DM_to_kpc

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

MATCH_FILTER_COLOR = CATEGORICAL_COLORS[0]
# Default limits of a match-filter plot: the extent of the polygon plus this margin (mag)
MATCH_FILTER_MARGIN_MAG = 0.5
HESS_CMAP = "gray_r"  # on and off Hess diagrams
HESS_DIFFERENCE_CMAP = "RdBu_r"  # on - off: diverging, symmetric limits (celestial_currents hess.py)
HESS_DIFFERENCE_PERCENTILE = 98.0  # default limit of the difference: this percentile of |density|
HESS_COLORBAR_LABEL = r"stars deg$^{-2}$ bin$^{-1}$"  # mathtext: the serif font has no ²
HESS_FIGSIZE = (5.0, 6.0)
DEFAULT_DIFFERENCE_SMOOTHING_BINS = 1.0  # smoothing of the on - off diagram in the figures (Gaussian sigma, bins)
ISOCHRONE_COLOR = "k"

def plot_CMD(
    data,
    fig=None,
    ax=None,
    mag_cols=["mag_g", "mag_r"],
    ylim=(15, 28),
    xlim=(0.1, 1.2),
    xlabel=rf"$g-r$",
    ylabel=rf"$g$",
    color=None,
    cmap="cividis",
    norm="log",
    bins_colors=None,
    bins_mag=None,
):
    """
    To plot 2D hist in the CMD

    data must be data frame of dict.

    The counts are computed by ``analysis.cmd_match_filter.hess_diagram``
    and drawn by ``plot_hess``.
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    mag1 = pd.to_numeric(data[mag_cols[0]], errors="coerce")
    mag2 = pd.to_numeric(data[mag_cols[1]], errors="coerce")

    if bins_colors is None:
        if xlim is None:
            xlim = [None, None]
        if xlim[0] is None:
            xlim[0] = np.nanpercentile(mag1 - mag2, 1)
        if xlim[1] is None:
            xlim[1] = np.nanpercentile(mag1 - mag2, 99)
        bins_colors = np.arange(xlim[0], xlim[1], 0.01)
    if bins_mag is None:
        bins_mag = np.arange(ylim[0], ylim[1], 0.1)
    if norm not in (None, "log"):
        raise ValueError(f"Invalid norm: {norm}. Use None or 'log'.")

    # computed in analysis.cmd_match_filter, drawn by plot_hess: one way for every CMD histogram
    hess = hess_diagram(data, mag_cols, color_bins=bins_colors, mag_bins=bins_mag)
    plot_hess(hess, ax=ax, cmap=cmap, norm=norm, title="")

    ax.set_xlabel(f"{mag_cols[0]} - {mag_cols[1]}")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel(mag_cols[0])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ax.get_legend():
        ax.legend()

    ax.invert_yaxis()
    return fig, ax


def plot_match_filter(
    distance_modulus: float,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    catalog: pd.DataFrame | None = None,
    stars: pd.DataFrame | None = None,
    cuts: Mapping[str, Any] | None = None,
    color_lim: tuple[float, float] | None = None,
    mag_lim: tuple[float, float] | None = None,
    axes: Axes | Sequence[Axes] | None = None,
    cmap: str = "gray_r",
    title: str | None = None,
    **prepare_kwargs: Any,
) -> tuple[Figure, list[Axes]]:
    """Match filter at one distance modulus, on the CMD of the stars if given.

    One panel per colour band of the match filter: magnitude
    ``reference_band`` against colour ``reference_band - color_band``, the
    convention of the polygons (``analysis.cmd_match_filter.match_filter_polygons``).
    The stars are drawn as a 2D histogram (``plot_CMD``) of their
    ``<namespace>_<band>_obs`` columns.

    Parameters
    ----------
    distance_modulus : float
        Distance modulus of the filter.
    isochrone : mapping, optional
        ``{"age": age (Gyr), "metallicity": Z}``. Default: the first
        isochrone of ``catalogs.isochrone_parameters.default_isochrone_grid``.
    match_filter : mapping, optional
        See ``analysis.cmd_match_filter``. Default:
        ``analysis.cmd_match_filter.DEFAULT_MATCH_FILTER``.
    catalog : pandas.DataFrame, optional
        Stars, prepared here
        (``analysis.catalog_preparation.prepare_match_filter_catalog``:
        columns checked, ``cuts``, and the window if given in
        ``prepare_kwargs``). Only the needed columns of the stars kept are
        copied.
    stars : pandas.DataFrame, optional
        Stars already prepared, used as they are, e.g. ``search.stars`` of
        ``analysis.stream_frame_search.prepare_search``. To draw the filter
        at many distance moduli (frames of a gif), prepare the stars once.
    cuts : mapping, optional
        Cuts of ``catalog`` (format of ``catalogs.catalog_cuts.get_cut_mask``),
        updating the default ``analysis.catalog_preparation.default_error_cuts``
        (see ``prepare_match_filter_catalog``).
    color_lim, mag_lim : tuple of float, optional
        Colour and magnitude ranges (``mag_lim`` as (bright, faint)). Default:
        the extent of the polygon of each panel plus
        ``MATCH_FILTER_MARGIN_MAG``. Give them for frames of a gif, so that
        the axes do not move with the distance modulus.
    axes : matplotlib.axes.Axes or sequence of Axes, optional
        One per colour band. Default: a new figure.
    cmap : str, optional
        Colour map of the stars.
    title : str, optional
        Default: the distance modulus and distance, the isochrone, and the
        number of stars in the filter (all colour bands) if stars are given.
    **prepare_kwargs
        Other arguments of ``prepare_match_filter_catalog`` for ``catalog``:
        ``stream_frame``, ``window``, ``ra_col``, ``dec_col``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        One per colour band.

    Raises
    ------
    KeyError
        Magnitude columns missing from the stars (all of them listed).
    ValueError
        Both ``catalog`` and ``stars``; ``cuts`` or ``prepare_kwargs``
        without ``catalog``; not one axes per colour band.
    """
    isochrone = default_isochrone_grid()[0] if isochrone is None else isochrone
    match_filter = DEFAULT_MATCH_FILTER if match_filter is None else match_filter
    if catalog is not None and stars is not None:
        raise ValueError("give catalog or stars, not both")
    if catalog is not None:
        stars = prepare_match_filter_catalog(catalog, match_filter, cuts=cuts, **prepare_kwargs)
    elif cuts is not None or prepare_kwargs:
        given = (["cuts"] if cuts is not None else []) + list(prepare_kwargs)
        raise ValueError(f"{given} prepare a catalog: stars are used as they are")
    reference_column, *color_columns = match_filter_columns(match_filter)
    if stars is not None:
        check_columns(stars, [reference_column, *color_columns])

    polygons = match_filter_polygons(distance_modulus, isochrone, match_filter)
    if axes is None:
        fig, axes = plt.subplots(1, len(polygons), figsize=(4.5 * len(polygons), 4.5), squeeze=False)
        axes = list(axes.ravel())
    else:
        axes = list(np.atleast_1d(axes))
        if len(axes) != len(polygons):
            raise ValueError(f"{len(axes)} axes for {len(polygons)} colour bands")
        fig = axes[0].figure

    reference_band = match_filter["reference_band"][1]
    for color_index, (ax, polygon, color_column, (_, color_band)) in enumerate(zip(
        axes, polygons, color_columns, match_filter["color_bands"]
    )):
        colors, mags = polygon[:, 0], polygon[:, 1]
        panel_color_lim = color_lim or (colors.min() - MATCH_FILTER_MARGIN_MAG,
                                        colors.max() + MATCH_FILTER_MARGIN_MAG)
        panel_mag_lim = mag_lim or (mags.min() - MATCH_FILTER_MARGIN_MAG, mags.max() + MATCH_FILTER_MARGIN_MAG)
        xlabel, ylabel = f"{reference_band} - {color_band}", reference_band
        if stars is not None:
            plot_CMD(
                stars, fig=fig, ax=ax, mag_cols=[reference_column, color_column],
                xlim=panel_color_lim, ylim=panel_mag_lim, xlabel=xlabel, ylabel=ylabel, cmap=cmap,
                # no star in the ranges: linear scale (a log scale needs counts)
                norm="log" if _any_in_ranges(stars, reference_column, color_column,
                                             panel_color_lim, panel_mag_lim) else None,
            )
        else:
            ax.set(xlim=panel_color_lim, ylim=panel_mag_lim[::-1], xlabel=xlabel, ylabel=ylabel)
        plot_match_filter_polygon(ax, distance_modulus, isochrone, match_filter, color_index=color_index)
        ax.legend(loc="upper right")

    if title is None:
        title = (f"dm {format_distance_modulus(distance_modulus)} ({convert_DM_to_kpc(distance_modulus).value:.1f} kpc), "
                 f"{isochrone['age']:g} Gyr, Z = {isochrone['metallicity']:.2g}")
        if stars is not None:
            n_selected = select_stars_in_match_filter(stars, distance_modulus, isochrone, match_filter).sum()
            title += f"\n{n_selected} stars in the filter"
    if len(axes) == 1:
        axes[0].set_title(title)
    else:
        fig.suptitle(title)
    return fig, axes


def plot_hess(
    hess: Mapping[str, Any],
    *,
    ax: Axes | None = None,
    difference: bool = False,
    cmap: str | None = None,
    norm: str | None = None,
    vlim: float | None = None,
    smoothing_bins: float = 0.0,
    colorbar: bool = True,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Draw a colour-magnitude 2D histogram: counts, a Hess diagram, or on - off.

    The only function that draws CMD histograms: ``plot_CMD`` computes its
    counts with ``analysis.cmd_match_filter.hess_diagram`` and calls it.
    Colour on x, magnitude on y, inverted. With ``difference``, a diverging
    colour map and symmetric limits ``±vlim``, so that 0 is white. Draws
    only: the histogram is computed by ``hess_diagram`` or ``difference_hess``.

    Parameters
    ----------
    hess : mapping
        From ``analysis.cmd_match_filter.hess_diagram`` (counts, or per deg²
        with an area) or ``difference_hess``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Default: a new figure.
    difference : bool, optional
        Draw an on - off diagram.
    cmap : str, optional
        Default: ``HESS_DIFFERENCE_CMAP`` with ``difference``, else ``HESS_CMAP``.
    norm : {None, "log"}, optional
        Colour scale of a diagram that is not a difference.
    vlim : float, optional
        Limit of the colour scale. Default: with ``difference``, the
        ``HESS_DIFFERENCE_PERCENTILE`` percentile of ``|density|``; else the
        maximum density.
    smoothing_bins : float, optional
        Gaussian sigma of a smoothing of the drawn density, in bins
        (``scipy.ndimage.gaussian_filter``); 0: none. Only the figure is
        smoothed.
    colorbar : bool, optional
        Add a colour bar: ``HESS_COLORBAR_LABEL`` per deg², "N" for counts.
    title : str, optional
        Default: the number of stars and the area.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        ``norm`` neither None nor "log".

    Notes
    -----
    Style of celestial_currents ``hess.py::plot_difference_cmd``.
    """
    if norm not in (None, "log"):
        raise ValueError(f"Invalid norm: {norm}. Use None or 'log'.")
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=HESS_FIGSIZE)
    density = np.asarray(hess["density"], dtype=float)
    if smoothing_bins > 0:
        density = gaussian_filter(density, smoothing_bins)
    per_area = hess.get("area_deg2") is not None
    if difference:
        if vlim is None:
            values = np.abs(density[np.isfinite(density)])
            vlim = float(np.percentile(values, HESS_DIFFERENCE_PERCENTILE)) if values.size else 0.0
            vlim = vlim if vlim > 0 else 1.0
        scale = {"cmap": cmap or HESS_DIFFERENCE_CMAP, "vmin": -vlim, "vmax": vlim}
    elif norm == "log":
        scale = {"cmap": cmap or HESS_CMAP, "norm": LogNorm(vmax=vlim)}
    else:
        scale = {"cmap": cmap or HESS_CMAP, "vmin": 0.0, "vmax": vlim}
    color_edges, mag_edges = hess["color_edges"], hess["mag_edges"]
    image = ax.imshow(density.T, origin="lower", aspect="auto",
                      extent=[color_edges[0], color_edges[-1], mag_edges[0], mag_edges[-1]], **scale)
    if colorbar:
        colorbar_ = fig.colorbar(image, ax=ax)
        colorbar_.set_label(HESS_COLORBAR_LABEL if per_area else "N", fontsize=COLORBAR_LABEL_SIZE)
    ax.set(xlabel=hess["color_label"], ylabel=hess["mag_label"], xlim=(color_edges[0], color_edges[-1]),
           ylim=(mag_edges[-1], mag_edges[0]))  # bright at the top
    if title is None and difference:
        title = f"{hess['n_stars']:.0f} excess stars in {hess['area_deg2']:.2f} deg$^2$"
    elif title is None:
        title = f"{hess['n_stars']} stars" + (f", {hess['area_deg2']:.2f} deg$^2$" if per_area else "")
    ax.set_title(title)
    return fig, ax


def plot_on_off_difference_hess(
    on_hess: Mapping[str, Any],
    off_hess: Mapping[str, Any],
    diff_hess: Mapping[str, Any],
    *,
    axes: Sequence[Axes] | None = None,
    smoothed_difference: bool = True,
    smoothing_bins: float = DEFAULT_DIFFERENCE_SMOOTHING_BINS,
    isochrone_track: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[Figure, list[Axes]]:
    """On, off and on - off Hess diagrams side by side, same axes limits.

    ``plot_hess`` of the on and off diagrams, of the difference
    (``difference=True``), and, with ``smoothed_difference``, of the
    difference smoothed by ``smoothing_bins`` (only the figure is smoothed):
    its noise is then averaged over neighbouring bins, where a stream
    sequence stands out better.

    Parameters
    ----------
    on_hess, off_hess, diff_hess : mapping
        From ``analysis.cmd_match_filter.hess_diagram`` and ``difference_hess``.
    axes : sequence of matplotlib.axes.Axes, optional
        Four (three without ``smoothed_difference``). Default: a new figure.
    smoothed_difference : bool, optional
        Add the smoothed difference.
    smoothing_bins : float, optional
        Gaussian sigma of its smoothing, in bins.
    isochrone_track : mapping, optional
        From ``analysis.cmd_match_filter.isochrone_cmd_track``: drawn on every
        panel (``plot_isochrone``).
    **kwargs
        Arguments of ``plot_hess`` for every panel (``colorbar``, ``norm``...).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes

    Raises
    ------
    ValueError
        Not as many axes as panels.
    """
    n_panels = 4 if smoothed_difference else 3
    if axes is None:
        fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * HESS_FIGSIZE[0], HESS_FIGSIZE[1]), sharey=True,
                                 layout="constrained")
    axes = list(axes)
    if len(axes) != n_panels:
        raise ValueError(f"{len(axes)} axes for {n_panels} panels.")
    shared = {key: value for key, value in kwargs.items() if key not in ("difference", "title", "smoothing_bins")}
    plot_hess(on_hess, ax=axes[0], title=f"On: {_hess_title(on_hess)}", **shared)
    plot_hess(off_hess, ax=axes[1], title=f"Off: {_hess_title(off_hess)}", **shared)
    plot_hess(diff_hess, ax=axes[2], difference=True, title=f"On $-$ off: {diff_hess['n_stars']:.0f} excess stars",
              **shared)
    if smoothed_difference:
        plot_hess(diff_hess, ax=axes[3], difference=True, smoothing_bins=smoothing_bins,
                  title=f"On $-$ off, smoothed ({smoothing_bins:g} bin{'' if smoothing_bins == 1 else 's'})", **shared)
    if isochrone_track is not None:
        for ax in axes:
            plot_isochrone(ax, isochrone_track)
        axes[0].legend(loc="upper left", fontsize="small", framealpha=0.8)
    return axes[0].figure, axes


def _hess_title(hess: Mapping[str, Any]) -> str:
    """Number of stars and area of a Hess diagram."""
    return f"{hess['n_stars']} stars, {hess['area_deg2']:.2f} deg$^2$"


def plot_isochrone(
    ax: Axes,
    isochrone_track: Mapping[str, Any],
    *,
    color: str = ISOCHRONE_COLOR,
    lw: float = 1.0,
    label: str | None = None,
) -> Line2D:
    """Isochrone track on a CMD or Hess diagram.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of the CMD (colour on x, magnitude on y).
    isochrone_track : mapping
        From ``analysis.cmd_match_filter.isochrone_cmd_track``.
    color : str, optional
        Line colour.
    lw : float, optional
        Line width.
    label : str, optional
        Default: age, Z ([Fe/H] when given) and distance modulus.

    Returns
    -------
    line : matplotlib.lines.Line2D
    """
    if label is None:
        label = f"{isochrone_track['age']:g} Gyr, Z = {isochrone_track['metallicity']:.2g}"
        if isochrone_track.get("feh") is not None:
            label += f" ([Fe/H] = {isochrone_track['feh']:+.2f})"
        label += f", dm {format_distance_modulus(isochrone_track['distance_modulus'])}"
    (line,) = ax.plot(isochrone_track["color"], isochrone_track["mag"], color=color, lw=lw, label=label)
    return line


def plot_match_filter_polygon(
    ax: Axes,
    distance_modulus: float,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    color_index: int = 0,
    **line_kwargs: Any,
) -> Line2D:
    """Outline of a match-filter polygon on a CMD or Hess diagram.

    Factored out of ``plot_match_filter``, which then calls it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of the CMD.
    distance_modulus : float
        Distance modulus of the filter.
    isochrone : mapping, optional
        ``{"age", "metallicity"}`` (see ``match_filter_polygons``).
    match_filter : mapping, optional
        Default: ``DEFAULT_MATCH_FILTER``.
    color_index : int, optional
        Colour band: ``match_filter["color_bands"][color_index]``.
    **line_kwargs
        Arguments of ``ax.plot``. Default colour ``MATCH_FILTER_COLOR``.

    Returns
    -------
    line : matplotlib.lines.Line2D
        The closed outline.
    """
    polygon = match_filter_polygons(distance_modulus, isochrone, match_filter)[color_index]
    closed = np.vstack([polygon, polygon[:1]])
    (line,) = ax.plot(closed[:, 0], closed[:, 1], **{"color": MATCH_FILTER_COLOR, "lw": 1.5, "label": "Match filter",
                                                   **line_kwargs})
    return line


def _any_in_ranges(
    stars: pd.DataFrame,
    reference_column: str,
    color_column: str,
    color_lim: tuple[float, float],
    mag_lim: tuple[float, float],
) -> bool:
    """Whether a star is inside the colour and magnitude ranges of a CMD."""
    mag = stars[reference_column].to_numpy()
    color = mag - stars[color_column].to_numpy()
    return bool(np.any((color >= color_lim[0]) & (color <= color_lim[1])
                       & (mag >= mag_lim[0]) & (mag <= mag_lim[1])))

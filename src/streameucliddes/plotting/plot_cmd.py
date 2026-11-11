"""Colour-magnitude diagrams, and the match filters drawn on them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from streameucliddes.analysis.catalog_preparation import check_columns, prepare_match_filter_catalog
from streameucliddes.analysis.cmd_match_filter import (
    DEFAULT_MATCH_FILTER,
    match_filter_columns,
    match_filter_polygons,
    select_stars_in_match_filter,
)
from streameucliddes.catalogs.isochrone_parameters import default_isochrone_grid
from streameucliddes.plotting import plot_style  # noqa: F401  (sets the figure style)
from streameucliddes.plotting.plot_style import CATEGORICAL_COLORS
from streameucliddes.unit_conversions import convert_DM_to_kpc

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

MATCH_FILTER_COLOR = CATEGORICAL_COLORS[0]
# Default limits of a match-filter plot: the extent of the polygon plus this margin (mag)
MATCH_FILTER_MARGIN_MAG = 0.5

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
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    mag1 = pd.to_numeric(data[mag_cols[0]], errors="coerce")
    mag2 = pd.to_numeric(data[mag_cols[1]], errors="coerce")

    if bins_colors is None:
        bins_colors = np.arange(xlim[0], xlim[1], 0.01)
    if bins_mag is None:
        bins_mag = np.arange(ylim[0], ylim[1], 0.1)

    counts, xedges, yedges = np.histogram2d(
        mag1 - mag2, mag1, bins=[bins_colors, bins_mag]
    )
    if norm is None:
        label = "N"
    elif norm == "log":
        norm = LogNorm()
        label = "log10(N)"
    else:
        raise ValueError(f"Invalid norm: {norm}. Use None or 'log'.")

    im = ax.imshow(
        counts.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap,
        norm=norm,
    )
    # plt.colorbar(im, ax=ax, label=label)
    # plt.colorbar(im, ax=ax)
    cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label(label)

    # cbar.ax.yaxis.set_label_coords(0.5, 1.05)
    # cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label, loc="top")

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
        ``analysis.known_stream_search.prepare_search``. To draw the filter
        at many distance moduli (frames of a gif), prepare the stars once.
    cuts : mapping, optional
        Cuts of ``catalog`` (format of ``catalogs.catalog_cuts.get_cut_mask``).
        Default: ``analysis.catalog_preparation.default_error_cuts``.
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
    for ax, polygon, color_column, (_, color_band) in zip(
        axes, polygons, color_columns, match_filter["color_bands"]
    ):
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
        closed = np.vstack([polygon, polygon[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=MATCH_FILTER_COLOR, lw=1.5, label="Match filter")
        ax.legend(loc="upper right")

    if title is None:
        title = (f"dm {distance_modulus:.1f} ({convert_DM_to_kpc(distance_modulus).value:.1f} kpc), "
                 f"{isochrone['age']:g} Gyr, Z = {isochrone['metallicity']:.2g}")
        if stars is not None:
            n_selected = select_stars_in_match_filter(stars, distance_modulus, isochrone, match_filter).sum()
            title += f"\n{n_selected} stars in the filter"
    if len(axes) == 1:
        axes[0].set_title(title)
    else:
        fig.suptitle(title)
    return fig, axes


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

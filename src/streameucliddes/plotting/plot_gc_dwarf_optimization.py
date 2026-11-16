"""Plots of the tuning of cuts and match filters on known objects (``analysis.gc_dwarf_optimization``).

The map of the selected stars around an object with its on and off regions,
its CMDs (on stars, and off stars over the same area) and Hess diagrams (on,
off, on - off), one panel or row per colour, and the SNR of several
selections. Draws only: the regions, selections and Hess diagrams come from
``analysis.gc_dwarf_optimization``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter

from streameucliddes.analysis.cmd_match_filter import _band_name, isochrone_cmd_track
from streameucliddes.analysis.gc_dwarf_optimization import (
    match_filter_cmd_bins,
    match_filter_color_mag_pairs,
    object_hess_diagrams,
    selection_snr,
)
from streameucliddes.catalogs.isochrone_parameters import default_isochrone_grid
from streameucliddes.plotting import plot_style  # noqa: F401  (sets the figure style)
from streameucliddes.plotting.plot_cmd import (
    HESS_FIGSIZE,
    plot_isochrone,
    plot_match_filter_polygon,
    plot_on_off_difference_hess,
)
from streameucliddes.plotting.plot_style import (
    CATEGORICAL_COLORS,
    COLORBAR_LABEL_SIZE,
    escape_underscores,
    format_distance_modulus,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

ON_COLOR, OFF_COLOR = CATEGORICAL_COLORS[4], CATEGORICAL_COLORS[3]
MASKED_OBJECT_COLOR = "0.5"
MAP_CMAP = "gray_r"
MAP_BINS = 60  # default bins along each axis of the map
MAP_PADDING = 1.05  # map half-width: the outer off radius times this
CMD_PANEL_SIZE = 4.5  # inches
CMD_PERCENTILES = (0.5, 99.5)  # default CMD limits without a match filter: these percentiles of the on stars


def _object_label(regions: Mapping[str, Any]) -> str:
    """Name and distance modulus of the object of the regions."""
    obj = regions["object"]
    return f"{obj['name']} (dm {format_distance_modulus(float(obj['distance_modulus']))})"


def _selected(regions: Mapping[str, Any], selected: ArrayLike | None) -> np.ndarray:
    """The selected stars, all by default."""
    n_stars = len(regions["on"])
    return np.ones(n_stars, dtype=bool) if selected is None else np.asarray(selected, dtype=bool)


def draw_object_regions(ax: Axes, regions: Mapping[str, Any], *, masked_objects: bool = True) -> None:
    """Edges of the on and off regions, and the masks of the other objects, on a map in arcmin.

    The map is in the frame of the object: offsets east (x) and north (y),
    arcmin. Ellipses of ``matplotlib.patches.Ellipse``: the major axis at
    position angle PA (north to east) is at ``90 - PA`` from the x axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map of ``plot_object_map``.
    regions : mapping
        From ``analysis.gc_dwarf_optimization.object_regions``.
    masked_objects : bool, optional
        Draw the masks of the other objects (dashed).
    """
    obj = regions["object"]
    ellipticity = np.nan_to_num(float(obj["ellipticity"]))
    angle = 90.0 - np.nan_to_num(float(obj["position_angle"]))
    half_light_arcmin = regions["half_light_radius_deg"] * 60.0
    for radii, color, label in ((regions["on_radii"], ON_COLOR, "on"), (regions["off_radii"], OFF_COLOR, "off")):
        for i, radius in enumerate(radii):
            if radius <= 0:
                continue
            width = 2 * radius * half_light_arcmin
            ax.add_patch(Ellipse((0.0, 0.0), width, width * (1 - ellipticity), angle=angle, fill=False, color=color,
                                 lw=1.2, label=label if i == 1 else None))
    if not masked_objects:
        return
    for i, (_, other) in enumerate(regions["masked_objects"].iterrows()):
        width = 2 * other["mask_radius_arcmin"]
        ax.add_patch(Ellipse(
            (other["offset_east_deg"] * 60.0, other["offset_north_deg"] * 60.0), width,
            width * (1 - np.nan_to_num(float(other["ellipticity"]))),
            angle=90.0 - np.nan_to_num(float(other["position_angle"])), fill=False, color=MASKED_OBJECT_COLOR,
            ls="--", lw=0.8, label="masked" if i == 0 else None,
        ))


def plot_object_map(
    regions: Mapping[str, Any],
    selected: ArrayLike | None = None,
    *,
    bin_arcmin: float | None = None,
    smoothing_bins: float = 0.0,
    ax: Axes | None = None,
    cmap: str = MAP_CMAP,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Map of the selected stars around an object, with its on and off regions.

    Counts per bin of the stars selected and not excluded, in the frame of
    the object (offsets in arcmin, east to the left), with the edges of the
    regions and the masks of the other objects (``draw_object_regions``).

    Parameters
    ----------
    regions : mapping
        From ``analysis.gc_dwarf_optimization.object_regions``.
    selected : array_like of bool, optional
        Selected stars, one per row of the catalog. Default: all.
    bin_arcmin : float, optional
        Bin size (arcmin). Default: ``MAP_BINS`` bins across the map.
    smoothing_bins : float, optional
        Gaussian sigma of a smoothing of the counts, in bins; 0: none.
    ax : matplotlib.axes.Axes, optional
        Default: a new figure.
    cmap : str, optional
        Colour map.
    title : str, optional
        Default: the object, and the SNR of the selection (``selection_snr``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5.5, 4.5))
    selected = _selected(regions, selected)
    stars = selected & ~np.asarray(regions["excluded"], dtype=bool)
    half_width = MAP_PADDING * regions["off_radii"][1] * regions["half_light_radius_deg"] * 60.0
    n_bins = MAP_BINS if bin_arcmin is None else max(int(np.ceil(2 * half_width / bin_arcmin)), 1)
    edges = np.linspace(-half_width, half_width, n_bins + 1)
    counts, _, _ = np.histogram2d(regions["offset_east_deg"][stars] * 60.0, regions["offset_north_deg"][stars] * 60.0,
                                  bins=[edges, edges])
    if smoothing_bins > 0:
        counts = gaussian_filter(counts, smoothing_bins)
    image = ax.imshow(counts.T, origin="lower", extent=[edges[0], edges[-1], edges[0], edges[-1]], cmap=cmap,
                      aspect="equal")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("stars per bin", fontsize=COLORBAR_LABEL_SIZE)
    draw_object_regions(ax, regions)
    ax.set(xlim=(half_width, -half_width), ylim=(-half_width, half_width),  # east to the left
           xlabel=r"$\Delta$ east (arcmin)", ylabel=r"$\Delta$ north (arcmin)")
    ax.legend(loc="upper right", fontsize="small", framealpha=0.8)
    if title is None:
        snr = selection_snr(regions, selected)
        title = (f"{_object_label(regions)}: {snr['n_signal']:.0f} $\\pm$ {snr['n_signal_error']:.0f} excess stars, "
                 f"SNR {snr['snr']:.1f}")
    ax.set_title(title)
    return fig, ax


def _cmd_pairs(
    match_filter: Mapping[str, Any] | None, mag_cols: Sequence[Sequence[str]] | None, distance_modulus: float | None
) -> tuple[list[tuple[str, str]], bool]:
    """Magnitude columns of each panel, and whether the match filter is drawn on them."""
    if mag_cols is not None:
        return [tuple(pair) for pair in mag_cols], False
    return match_filter_color_mag_pairs(match_filter), distance_modulus is not None


def _draw_match_filter(
    ax: Axes,
    distance_modulus: float,
    isochrone: Mapping[str, float],
    match_filter: Mapping[str, Any] | None,
    color_index: int,
) -> None:
    """Polygon and isochrone track of one colour of the match filter."""
    plot_match_filter_polygon(ax, distance_modulus, isochrone, match_filter, color_index=color_index)
    plot_isochrone(ax, isochrone_cmd_track(distance_modulus, isochrone["age"], metallicity=isochrone["metallicity"],
                                           match_filter=match_filter, color_index=color_index))


def plot_object_cmd(
    catalog: pd.DataFrame,
    regions: Mapping[str, Any],
    distance_modulus: float | None = None,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    mag_cols: Sequence[Sequence[str]] | None = None,
    stars: ArrayLike | None = None,
    axes: Axes | Sequence[Axes] | None = None,
    color_lim: tuple[float, float] | None = None,
    mag_lim: tuple[float, float] | None = None,
    marker_size: float = 3.0,
    seed: int = 0,
) -> tuple[Figure, list[Axes]]:
    """CMDs of an object: its on stars, and off stars over the same area.

    The off stars are a random fraction ``on_area / off_area`` of those of
    the off region: as many as the field holds on the object, so that the
    two sets compare by eye. One panel per colour: those of the match filter
    (polygon and isochrone drawn when ``distance_modulus`` is given), or the
    pairs ``mag_cols``.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars.
    regions : mapping
        From ``analysis.gc_dwarf_optimization.object_regions``.
    distance_modulus : float, optional
        Of the match filter drawn, e.g. ``obj["distance_modulus"]``.
    isochrone : mapping, optional
        ``{"age", "metallicity"}``, e.g. ``object_isochrone(obj)``. Default:
        that of ``default_isochrone_grid``.
    match_filter : mapping, optional
        See ``analysis.cmd_match_filter``. Default: ``DEFAULT_MATCH_FILTER``.
    mag_cols : sequence of pairs of str, optional
        Instead of the match filter: one panel per (magnitude, other
        magnitude) pair, colour their difference. No polygon then.
    stars : array_like of bool, optional
        Stars drawn (e.g. a selection). Default: all.
    axes : Axes or sequence of Axes, optional
        One per panel. Default: a new figure.
    color_lim, mag_lim : tuple of float, optional
        Ranges (``mag_lim`` as (bright, faint)). Default: the polygon plus a
        margin, or the ``CMD_PERCENTILES`` of the on stars.
    marker_size : float, optional
    seed : int, optional
        Of the random choice of the off stars.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    isochrone = default_isochrone_grid()[0] if isochrone is None else isochrone
    pairs, draw_filter = _cmd_pairs(match_filter, mag_cols, distance_modulus)
    axes = _panel_axes(axes, len(pairs))
    shown = _selected(regions, stars)
    on = shown & regions["on"]
    off = np.flatnonzero(shown & regions["off"])
    rng = np.random.default_rng(seed)
    off = off[rng.random(off.size) < regions["on_area_deg2"] / regions["off_area_deg2"]]
    bins = match_filter_cmd_bins(distance_modulus, isochrone, match_filter) if draw_filter else None

    for i, (ax, (mag_col, other_col)) in enumerate(zip(axes, pairs)):
        mag, other = catalog[mag_col].to_numpy(dtype=float), catalog[other_col].to_numpy(dtype=float)
        ax.scatter(mag[off] - other[off], mag[off], s=marker_size, color=OFF_COLOR, lw=0, alpha=0.7,
                   label=f"off, scaled to the on area ({off.size})")
        ax.scatter(mag[on] - other[on], mag[on], s=marker_size, color="k", lw=0, label=f"on ({int(on.sum())})")
        if draw_filter:
            _draw_match_filter(ax, distance_modulus, isochrone, match_filter, i)
            panel_color_lim, panel_mag_lim = (bins[i][0][[0, -1]], bins[i][1][[0, -1]])
        else:
            color = mag[on] - other[on]
            panel_color_lim = np.nanpercentile(color, CMD_PERCENTILES) if on.any() else (-1.0, 2.0)
            panel_mag_lim = np.nanpercentile(mag[on], CMD_PERCENTILES) if on.any() else (16.0, 27.0)
        panel_color_lim = color_lim or panel_color_lim
        panel_mag_lim = mag_lim or panel_mag_lim
        ax.set(xlim=panel_color_lim, ylim=(max(panel_mag_lim), min(panel_mag_lim)),
               xlabel=f"{_band_name(mag_col)} - {_band_name(other_col)}", ylabel=_band_name(mag_col))
        ax.legend(loc="upper left", fontsize="small", framealpha=0.8, markerscale=3)
    axes[0].figure.suptitle(_object_label(regions))
    return axes[0].figure, axes


def plot_object_hess(
    catalog: pd.DataFrame,
    regions: Mapping[str, Any],
    distance_modulus: float | None = None,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    mag_cols: Sequence[Sequence[str]] | None = None,
    stars: ArrayLike | None = None,
    color_bins: ArrayLike | None = None,
    mag_bins: ArrayLike | None = None,
    smoothed_difference: bool = True,
    axes: Sequence[Sequence[Axes]] | None = None,
    **kwargs: Any,
) -> tuple[Figure, list[list[Axes]]]:
    """On, off and on - off Hess diagrams of an object, one row per colour.

    ``analysis.gc_dwarf_optimization.object_hess_diagrams`` drawn by
    ``plotting.plot_cmd.plot_on_off_difference_hess``. The colours are
    those of the match filter (polygon and isochrone on every panel when
    ``distance_modulus`` is given), or the pairs ``mag_cols``.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars.
    regions : mapping
        From ``analysis.gc_dwarf_optimization.object_regions``.
    distance_modulus, isochrone, match_filter, mag_cols :
        See ``plot_object_cmd``.
    stars : array_like of bool, optional
        Stars counted, e.g. ``stars_passing_cuts(catalog, match_filter)``.
        Default: all.
    color_bins, mag_bins : array_like, optional
        Bin edges, the same for every row. Default: around the polygon of
        each row (``match_filter_cmd_bins``) with the match filter, else
        those of ``cmd_match_filter.hess_diagram``.
    smoothed_difference : bool, optional
        Add the smoothed on - off panel.
    axes : sequence of rows of Axes, optional
        One row of 4 (3 without ``smoothed_difference``) per colour.
    **kwargs
        Of ``plot_on_off_difference_hess`` (``smoothing_bins``, ``norm``...).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of lists of matplotlib.axes.Axes
        One row per colour.
    """
    isochrone = default_isochrone_grid()[0] if isochrone is None else isochrone
    pairs, draw_filter = _cmd_pairs(match_filter, mag_cols, distance_modulus)
    n_columns = 4 if smoothed_difference else 3
    if axes is None:
        fig, axes = plt.subplots(len(pairs), n_columns, squeeze=False, layout="constrained",
                                 figsize=(n_columns * HESS_FIGSIZE[0], len(pairs) * HESS_FIGSIZE[1]))
    axes = [list(row) for row in axes]
    if len(axes) != len(pairs):
        raise ValueError(f"{len(axes)} rows of axes for {len(pairs)} colours.")
    filter_bins = match_filter_cmd_bins(distance_modulus, isochrone, match_filter) if draw_filter else None
    for i, (row, pair) in enumerate(zip(axes, pairs)):
        row_color_bins = color_bins if color_bins is not None or filter_bins is None else filter_bins[i][0]
        row_mag_bins = mag_bins if mag_bins is not None or filter_bins is None else filter_bins[i][1]
        on_hess, off_hess, difference = object_hess_diagrams(catalog, regions, pair, stars=stars,
                                                             color_bins=row_color_bins, mag_bins=row_mag_bins)
        track = None
        if draw_filter:
            track = isochrone_cmd_track(distance_modulus, isochrone["age"], metallicity=isochrone["metallicity"],
                                        match_filter=match_filter, color_index=i)
        plot_on_off_difference_hess(on_hess, off_hess, difference, axes=row, smoothed_difference=smoothed_difference,
                                    isochrone_track=track, **kwargs)
        if draw_filter:
            for ax in row:
                plot_match_filter_polygon(ax, distance_modulus, isochrone, match_filter, color_index=i)
    fig = axes[0][0].figure
    fig.suptitle(_object_label(regions))
    return fig, axes


def plot_selection_comparison(table: pd.DataFrame, *, ax: Axes | None = None) -> tuple[Figure, Axes]:
    """SNR of several selections (``compare_selections``), as horizontal bars.

    Each bar is labelled with its excess stars and their Poisson error.

    Parameters
    ----------
    table : pandas.DataFrame
        From ``analysis.gc_dwarf_optimization.compare_selections``.
    ax : matplotlib.axes.Axes, optional
        Default: a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.0, 0.5 + 0.4 * len(table)))
    positions = np.arange(len(table))
    snr = table["snr"].to_numpy(dtype=float)
    ax.barh(positions, np.where(np.isfinite(snr), snr, 0.0), color=CATEGORICAL_COLORS[1])
    for position, value, (_, row) in zip(positions, snr, table.iterrows()):
        ax.text(max(value, 0.0) if np.isfinite(value) else 0.0, position,
                f"  {row['n_signal']:.0f} $\\pm$ {row['n_signal_error']:.0f}", va="center", fontsize="small")
    ax.set_yticks(positions, [_display_text(str(name)) for name in table.index])
    ax.invert_yaxis()  # first selection at the top
    ax.set_xlabel("SNR (excess / $\\sqrt{\\mathrm{background}}$)")
    ax.margins(x=0.25)
    return fig, ax


def _display_text(text: str) -> str:
    """Text as displayed: the serif font (cmr10) has no ``_``, ``<`` nor ``>`` glyph: drawn by mathtext."""
    return escape_underscores(text).replace("<", "$<$").replace(">", "$>$")


def _panel_axes(axes: Axes | Sequence[Axes] | None, n_panels: int) -> list[Axes]:
    """One axes per panel: given, or a new figure."""
    if axes is None:
        _, axes = plt.subplots(1, n_panels, figsize=(CMD_PANEL_SIZE * n_panels, CMD_PANEL_SIZE), squeeze=False,
                               layout="constrained")
        return list(axes.ravel())
    axes = list(np.atleast_1d(axes))
    if len(axes) != n_panels:
        raise ValueError(f"{len(axes)} axes for {n_panels} panels.")
    return axes

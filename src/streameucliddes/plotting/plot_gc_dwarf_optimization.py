"""Plots of the tuning of cuts and match filters on known objects (``analysis.gc_dwarf_optimization``).

Thin wrappers around the shared figures, with the regions of an object:
its map in its frame (``plot_stream_frame_search.plot_stream_frame_search_single_dm``)
with the exact outlines of the regions and of the masked objects
(``build_masks.ellipse_outline_radec``), its CMDs (``plot_cmd.plot_on_off_cmd``)
and Hess diagrams (``plot_cmd.plot_on_off_hess_per_color``), one panel or
row per colour, and the SNR of several selections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from streameucliddes.analysis.build_masks import ellipse_outline_radec
from streameucliddes.analysis.gc_dwarf_optimization import object_density_map, selection_snr
from streameucliddes.plotting import plot_style  # noqa: F401  (sets the figure style)
from streameucliddes.plotting.plot_cmd import plot_on_off_cmd, plot_on_off_hess_per_color
from streameucliddes.plotting.plot_stream_frame_search import plot_stream_frame_search_single_dm
from streameucliddes.plotting.plot_style import CATEGORICAL_COLORS, escape_text, format_distance_modulus
from streameucliddes.stream_frames import radec_to_phi12

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

ON_COLOR, OFF_COLOR = CATEGORICAL_COLORS[4], CATEGORICAL_COLORS[3]
MASKED_OBJECT_STYLE = {"color": "0.4", "ls": "--", "lw": 0.8}
MAP_FIGSIZE = (6.5, 5.5)


def _object_label(regions: Mapping[str, Any]) -> str:
    """Name and distance modulus of the object of the regions; their ``label`` when they have one (a stream)."""
    if "label" in regions:
        return escape_text(str(regions["label"]))
    obj = regions["object"]
    return f"{escape_text(str(obj['name']))} (dm {format_distance_modulus(float(obj['distance_modulus']))})"


def _selected(regions: Mapping[str, Any], selected: ArrayLike | None) -> np.ndarray:
    """The selected stars, all by default."""
    return np.ones(len(regions["on"]), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)


def draw_object_regions(ax: Axes, regions: Mapping[str, Any], *, masked_objects: bool = True) -> None:
    """Exact outlines of the on and off regions, and of the masks of the other objects, on a map of the object.

    ``build_masks.ellipse_outline_radec`` (the ellipses of the regions and
    masks, exact on the sky) taken to the frame of the object
    (``stream_frames.radec_to_phi12``).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map in the frame of the object (``plot_object_map``).
    regions : mapping
        From ``analysis.gc_dwarf_optimization.object_regions``.
    masked_objects : bool, optional
        Draw the masks of the other objects (dashed).
    """
    obj, frame = regions["object"], regions["stream_frame"]
    centre = float(obj["ra"]), float(obj["dec"])
    shape = float(obj["ellipticity"]), float(obj["position_angle"])
    for radii, color, label in ((regions["on_radii"], ON_COLOR, "on"), (regions["off_radii"], OFF_COLOR, "off")):
        for i, radius in enumerate(radii):
            if radius > 0:
                outline = ellipse_outline_radec(*centre, radius * regions["half_light_radius_deg"], *shape)
                ax.plot(*radec_to_phi12(*outline, frame), color=color, lw=1.2, label=label if i == 1 else None)
    if not masked_objects:
        return
    for i, (_, other) in enumerate(regions["masked_objects"].iterrows()):
        outline = ellipse_outline_radec(float(other["ra"]), float(other["dec"]), other["mask_radius_arcmin"] / 60.0,
                                        float(other["ellipticity"]), float(other["position_angle"]))
        ax.plot(*radec_to_phi12(*outline, frame), label="masked" if i == 0 else None, **MASKED_OBJECT_STYLE)


def plot_object_map(
    regions: Mapping[str, Any],
    selected: ArrayLike | None = None,
    *,
    bin_size_deg: float | None = None,
    smoothing_deg: float = 0.0,
    ax: Axes | None = None,
    title: str | None = None,
    **map_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Map of the selected stars around an object, with its regions.

    ``analysis.gc_dwarf_optimization.object_density_map`` drawn by
    ``plot_stream_frame_search_single_dm`` (phi1, to the east, increasing to
    the left: the sky as seen, north up), with ``draw_object_regions``.

    Parameters
    ----------
    regions : mapping
        From ``analysis.gc_dwarf_optimization.object_regions``.
    selected : array_like of bool, optional
        Selected stars, one per row of the catalog. Default: all.
    bin_size_deg, smoothing_deg : float, optional
        Of ``object_density_map``.
    ax : matplotlib.axes.Axes, optional
        Default: a new figure.
    title : str, optional
        Default: the object, and the excess and SNR of the selection (``selection_snr``).
    **map_kwargs
        Of ``plot_stream_frame_search_single_dm`` (``vmax``, ``cmap``...).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    selected = _selected(regions, selected)
    density_map = object_density_map(regions, selected, bin_size_deg=bin_size_deg, smoothing_deg=smoothing_deg)
    if title is None:
        snr = selection_snr(regions, selected)
        title = (f"{_object_label(regions)}: {snr['n_signal']:.0f} $\\pm$ {snr['n_signal_error']:.0f} excess stars, "
                 f"SNR {snr['snr']:.1f}")
    if ax is None:
        _, ax = plt.subplots(figsize=MAP_FIGSIZE)
    fig, ax = plot_stream_frame_search_single_dm(density_map, ax=ax, title=title, **map_kwargs)
    draw_object_regions(ax, regions)
    ax.set(xlabel=r"$\phi_1$ (east) [deg]", ylabel=r"$\phi_2$ (north) [deg]")
    ax.legend(loc="upper right", fontsize="small", framealpha=0.8)
    return fig, ax


def plot_object_cmd(
    catalog: pd.DataFrame,
    regions: Mapping[str, Any],
    distance_modulus: float | None = None,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    stars: ArrayLike | None = None,
    **cmd_kwargs: Any,
) -> tuple[Figure, list[Axes]]:
    """CMDs of an object: its on stars, and off stars over the same area (``plot_cmd.plot_on_off_cmd``).

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars.
    regions : mapping
        From ``analysis.gc_dwarf_optimization.object_regions``.
    distance_modulus, isochrone, match_filter :
        Match filter drawn, e.g. ``obj["distance_modulus"]``,
        ``isochrone_from_feh(feh=obj["feh"])``; see ``plot_on_off_cmd``.
    stars : array_like of bool, optional
        Stars drawn (e.g. a selection). Default: all.
    **cmd_kwargs
        Of ``plot_on_off_cmd`` (``mag_cols``, ``axes``, ``color_lim``...).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    shown = _selected(regions, stars)
    fig, axes = plot_on_off_cmd(catalog, shown & regions["on"], shown & regions["off"],
                                regions["on_area_deg2"] / regions["off_area_deg2"], distance_modulus, isochrone,
                                match_filter, **cmd_kwargs)
    fig.suptitle(_object_label(regions))
    return fig, axes


def plot_object_hess(
    catalog: pd.DataFrame,
    regions: Mapping[str, Any],
    distance_modulus: float | None = None,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    stars: ArrayLike | None = None,
    **hess_kwargs: Any,
) -> tuple[Figure, list[list[Axes]]]:
    """On, off and on - off Hess diagrams of an object, one row per colour (``plot_cmd.plot_on_off_hess_per_color``).

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars.
    regions : mapping
        From ``analysis.gc_dwarf_optimization.object_regions``.
    distance_modulus, isochrone, match_filter :
        See ``plot_object_cmd``.
    stars : array_like of bool, optional
        Stars counted, e.g. ``prepare_match_filter_catalog(catalog,
        match_filter, as_mask=True)``. Default: all.
    **hess_kwargs
        Of ``plot_on_off_hess_per_color`` (``mag_cols``, ``color_bins``...).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of lists of matplotlib.axes.Axes
    """
    shown = _selected(regions, stars)
    fig, axes = plot_on_off_hess_per_color(
        catalog, shown & regions["on"], regions["on_area_deg2"], [shown & regions["off"]], [regions["off_area_deg2"]],
        distance_modulus, isochrone, match_filter, **hess_kwargs,
    )
    fig.suptitle(_object_label(regions))
    return fig, axes


def plot_selection_comparison(
    table: pd.DataFrame,
    *,
    ax: Axes | None = None,
    marker: str = "o",
    color: str | None = None,
    label: str | None = None,
) -> tuple[Figure, Axes]:
    """SNR of several selections (``compare_selections``), one point per selection, one row each.

    Call it again with the returned ``ax`` to overplot other tables (e.g. the
    same selections with another isochrone or star selection), each with its
    ``marker``, ``color`` and ``label``; then ``ax.legend()``. A selection
    name already on the axes keeps its row, a new one is added below.

    Parameters
    ----------
    table : pandas.DataFrame
        From ``analysis.gc_dwarf_optimization.compare_selections``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on, e.g. from a previous call. Default: a new figure.
    marker : str, optional
        Marker of the points.
    color : str, optional
        Colour of the points. Default: the next colour of the axes.
    label : str, optional
        Legend label of the points.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.0, 0.8 + 0.3 * len(table)))
    names = [escape_text(str(name)) for name in table.index]
    ax.plot(table["snr"].to_numpy(dtype=float), names, ls="none", marker=marker, color=color, label=label)
    if not ax.yaxis_inverted():  # first selection at the top
        ax.invert_yaxis()
    ax.set_xlabel("SNR (excess / $\\sqrt{\\mathrm{background}}$)")
    ax.grid(axis="y", color="0.9", lw=0.5)
    return fig, ax

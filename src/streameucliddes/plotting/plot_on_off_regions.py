"""Plots of on and off regions (``analysis.on_off_regions``): around an object, along a track.

Thin wrappers around the shared figures, with any regions: the map of the
selected stars in their frame (``plot_stream_frame_search.plot_stream_frame_search_single_dm``)
with the outlines of the regions, of the masked objects (exact ellipses,
``build_masks.ellipse_outline_radec``) and of the track
(``draw_region_outlines``); the CMDs (``plot_cmd.plot_on_off_cmd``) and Hess
diagrams (``plot_cmd.plot_on_off_hess_per_color``, the off diagram the mean
of those of the off parts), one panel or row per colour; the SNR of several
selections (``plot_selection_comparison``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from streameucliddes.analysis.build_masks import ellipse_outline_radec
from streameucliddes.analysis.on_off_regions import (
    off_part_masks_and_areas,
    region_density_map,
    selected_stars,
    selection_snr,
)
from streameucliddes.plotting import plot_style  # noqa: F401  (sets the figure style)
from streameucliddes.plotting.plot_cmd import plot_on_off_cmd, plot_on_off_hess_per_color
from streameucliddes.plotting.plot_stream_frame_search import draw_stream_track, plot_stream_frame_search_single_dm
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
# Maps: about square (an object) or long (a track, phi1 extent over phi2 extent above WIDE_MAP_ASPECT)
SQUARE_MAP_FIGSIZE, WIDE_MAP_FIGSIZE = (6.5, 5.5), (11.0, 4.0)
WIDE_MAP_ASPECT = 1.5


def region_title(regions: Mapping[str, Any]) -> str:
    """Title of the figures of regions: their ``label`` (an object: its name and distance modulus), escaped."""
    if "object" in regions:
        obj = regions["object"]
        return f"{escape_text(str(obj['name']))} (dm {format_distance_modulus(float(obj['distance_modulus']))})"
    return escape_text(str(regions.get("label", "")))


def draw_region_outlines(
    ax: Axes, regions: Mapping[str, Any], *, masked_objects: bool = True, track: bool = True
) -> None:
    """Outlines of the on and off regions, of the masked objects and of the track, on a map in their frame.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map in the frame of the regions (``plot_region_map``).
    regions : mapping
        From a builder of regions, with ``outlines`` (``{"on": [(phi1,
        phi2), ...], "off": [...]}``).
    masked_objects : bool, optional
        Draw the mask ellipses of the objects masked exactly (dashed).
    track : bool, optional
        Draw the track (``draw_stream_track``): its points
        (``track_points_phi12``) or the centre of the bands (``track_phi1``,
        ``track_phi2``), when the regions have one.
    """
    for name, color in (("on", ON_COLOR), ("off", OFF_COLOR)):
        for i, (phi1, phi2) in enumerate(regions.get("outlines", {}).get(name, [])):
            ax.plot(phi1, phi2, color=color, lw=1.2, label=name if i == 0 else None)
    if masked_objects and regions.get("masked_objects") is not None:
        frame = regions["stream_frame"]
        for i, (_, other) in enumerate(regions["masked_objects"].iterrows()):
            outline = ellipse_outline_radec(float(other["ra"]), float(other["dec"]),
                                            other["mask_radius_arcmin"] / 60.0, float(other["ellipticity"]),
                                            float(other["position_angle"]))
            ax.plot(*radec_to_phi12(*outline, frame), label="masked" if i == 0 else None, **MASKED_OBJECT_STYLE)
    if track:
        points = regions.get("track_points_phi12")
        if points is None and "track_phi1" in regions:
            points = (regions["track_phi1"], regions["track_phi2"])
        if points is not None:
            draw_stream_track(ax, *points, label=escape_text(str(regions.get("track_name", "track"))))


def plot_region_map(
    regions: Mapping[str, Any],
    selected: ArrayLike | None = None,
    *,
    bin_size_deg: float | None = None,
    smoothing_deg: float = 0.0,
    ax: Axes | None = None,
    title: str | None = None,
    **map_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Map of the selected stars of the regions, with their outlines.

    ``analysis.on_off_regions.region_density_map`` drawn by
    ``plot_stream_frame_search_single_dm`` (phi1 increasing to the left;
    around an object, phi1 to the east: the sky as seen, north up), with
    ``draw_region_outlines``.

    Parameters
    ----------
    regions : mapping
        From a builder of regions.
    selected : array_like of bool, optional
        Selected stars, one per row of the catalog. Default: all.
    bin_size_deg, smoothing_deg : float, optional
        Of ``region_density_map``.
    ax : matplotlib.axes.Axes, optional
        Default: a new figure, its shape from that of the window.
    title : str, optional
        Default: ``region_title``, and the excess and SNR of the selection (``selection_snr``).
    **map_kwargs
        Of ``plot_stream_frame_search_single_dm`` (``vmax``, ``cmap``...).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    selected = selected_stars(regions, selected)
    density_map = region_density_map(regions, selected, bin_size_deg=bin_size_deg, smoothing_deg=smoothing_deg)
    if title is None:
        snr = selection_snr(regions, selected)
        title = (f"{region_title(regions)}: {snr['n_signal']:.0f} $\\pm$ {snr['n_signal_error']:.0f} excess stars, "
                 f"SNR {snr['snr']:.1f}")
    if ax is None:
        window = regions["window"]
        wide = (window[1] - window[0]) > WIDE_MAP_ASPECT * (window[3] - window[2])
        _, ax = plt.subplots(figsize=WIDE_MAP_FIGSIZE if wide else SQUARE_MAP_FIGSIZE, layout="constrained")
    fig, ax = plot_stream_frame_search_single_dm(density_map, ax=ax, title=title, **map_kwargs)
    draw_region_outlines(ax, regions)
    if "object" in regions:
        ax.set(xlabel=r"$\phi_1$ (east) [deg]", ylabel=r"$\phi_2$ (north) [deg]")
    ax.legend(loc="upper right", fontsize="small", framealpha=0.8)
    return fig, ax


def plot_region_cmd(
    catalog: pd.DataFrame,
    regions: Mapping[str, Any],
    distance_modulus: float | None = None,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    stars: ArrayLike | None = None,
    **cmd_kwargs: Any,
) -> tuple[Figure, list[Axes]]:
    """CMDs of the on stars, and of off stars over the same area (``plot_cmd.plot_on_off_cmd``).

    The off stars drawn are a random fraction (on area / off area) of those
    of all the off parts.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars.
    regions : mapping
        From a builder of regions.
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
    shown = selected_stars(regions, stars)
    fig, axes = plot_on_off_cmd(catalog, shown & regions["on"], shown & regions["off"],
                                regions["on_area_deg2"] / regions["off_area_deg2"], distance_modulus, isochrone,
                                match_filter, **cmd_kwargs)
    fig.suptitle(region_title(regions))
    return fig, axes


def plot_region_hess(
    catalog: pd.DataFrame,
    regions: Mapping[str, Any],
    distance_modulus: float | None = None,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    stars: ArrayLike | None = None,
    **hess_kwargs: Any,
) -> tuple[Figure, list[list[Axes]]]:
    """On, off and on - off Hess diagrams, one row per colour (``plot_cmd.plot_on_off_hess_per_color``).

    The off diagram is the mean of those of the off parts
    (``on_off_regions.off_part_masks_and_areas``), as the background of
    ``selection_snr``.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars.
    regions : mapping
        From a builder of regions.
    distance_modulus, isochrone, match_filter :
        See ``plot_region_cmd``; along a track, e.g. ``np.mean(regions["track_dm"])``.
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
    shown = selected_stars(regions, stars)
    off_masks, off_areas = off_part_masks_and_areas(regions, shown)
    fig, axes = plot_on_off_hess_per_color(catalog, shown & regions["on"], regions["on_area_deg2"], off_masks,
                                           off_areas, distance_modulus, isochrone, match_filter, **hess_kwargs)
    fig.suptitle(region_title(regions))
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
        From ``analysis.on_off_regions.compare_selections``.
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

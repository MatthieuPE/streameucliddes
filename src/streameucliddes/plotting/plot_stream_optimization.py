"""Plots of the tuning of cuts and match filters on a known stream (``analysis.stream_optimization``).

The map of the selected stars along the track (``plot_stream_frame_search_single_dm``)
with the regions (``draw_stream_regions``). The CMDs and Hess diagrams are those of
the objects: ``plotting.plot_gc_dwarf_optimization.plot_object_cmd`` and
``plot_object_hess`` take the regions of ``stream_regions`` as they are (title: their
``label``), with the distance modulus of the track for the isochrone drawn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from streameucliddes.analysis.gc_dwarf_optimization import selection_snr
from streameucliddes.analysis.stream_optimization import DEFAULT_MAP_BIN_DEG, stream_density_map
from streameucliddes.plotting import plot_style  # noqa: F401  (sets the figure style)
from streameucliddes.plotting.plot_gc_dwarf_optimization import OFF_COLOR, ON_COLOR, _object_label, _selected
from streameucliddes.plotting.plot_stream_frame_search import draw_stream_track, plot_stream_frame_search_single_dm
from streameucliddes.stream_frames import radec_to_phi12

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

MAP_FIGSIZE = (11.0, 4.0)


def draw_stream_regions(ax: Axes, regions: Mapping[str, Any], *, track: bool = True) -> None:
    """Edges of the on and off regions along the track, slice by slice, on a map in its frame.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map in the frame of the track (``plot_stream_map``).
    regions : mapping
        From ``analysis.stream_optimization.stream_regions``.
    track : bool, optional
        Also draw the galstreams track (``draw_stream_track``).
    """
    sigma = regions["sigma_deg"]
    edges, phi2 = regions["slice_edges"], regions["slice_phi2"]
    on_half = regions["on_n_sigma"] * sigma
    off_inner, off_outer = (n_sigma * sigma for n_sigma in regions["off_n_sigma"])
    for offsets, color, label in (((-on_half, on_half), ON_COLOR, "on"),
                                  ((-off_outer, -off_inner, off_inner, off_outer), OFF_COLOR, "off")):
        for i, offset in enumerate(offsets):
            ax.stairs(phi2 + offset, edges, baseline=None, color=color, lw=1.0, label=label if i == 0 else None)
    if track:
        points = regions["track"].track
        draw_stream_track(ax, *radec_to_phi12(points.ra.deg, points.dec.deg, regions["stream_frame"]),
                          label=str(regions["track_name"]))


def plot_stream_map(
    regions: Mapping[str, Any],
    selected: ArrayLike | None = None,
    *,
    bin_size_deg: float = DEFAULT_MAP_BIN_DEG,
    smoothing_deg: float = 0.0,
    ax: Axes | None = None,
    title: str | None = None,
    **map_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Map of the selected stars along a stream, with its regions.

    ``analysis.stream_optimization.stream_density_map`` drawn by
    ``plot_stream_frame_search_single_dm`` (phi1 increasing to the left), with
    ``draw_stream_regions``.

    Parameters
    ----------
    regions : mapping
        From ``analysis.stream_optimization.stream_regions``.
    selected : array_like of bool, optional
        Selected stars, one per row of the catalog. Default: all.
    bin_size_deg, smoothing_deg : float, optional
        Of ``stream_density_map``.
    ax : matplotlib.axes.Axes, optional
        Default: a new figure.
    title : str, optional
        Default: the label of the regions, and the excess and SNR of the selection (``selection_snr``).
    **map_kwargs
        Of ``plot_stream_frame_search_single_dm`` (``vmax``, ``cmap``...).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    selected = _selected(regions, selected)
    density_map = stream_density_map(regions, selected, bin_size_deg=bin_size_deg, smoothing_deg=smoothing_deg)
    if title is None:
        snr = selection_snr(regions, selected)
        title = (f"{_object_label(regions)}: {snr['n_signal']:.0f} $\\pm$ {snr['n_signal_error']:.0f} excess stars, "
                 f"SNR {snr['snr']:.1f}")
    if ax is None:
        _, ax = plt.subplots(figsize=MAP_FIGSIZE, layout="constrained")
    fig, ax = plot_stream_frame_search_single_dm(density_map, ax=ax, title=title, **map_kwargs)
    draw_stream_regions(ax, regions)
    ax.legend(loc="upper right", fontsize="small", framealpha=0.8)
    return fig, ax

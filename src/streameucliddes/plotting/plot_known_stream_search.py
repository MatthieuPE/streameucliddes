"""Figures of the known-stream search (analysis.known_stream_search)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import skyproj
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke

from streameucliddes.plotting.plot_style import CATEGORICAL_COLORS

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from typing import Any

    import galstreams
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray

TRACK_COLOR = CATEGORICAL_COLORS[0]  # one color for every track: the labels give their names
FOOTPRINT_GREY = "0.88"
INK = "0.15"  # text
# Positions tried for a label, as fractions along the part of the track inside the footprint
LABEL_FRACTIONS = (0.5, 0.35, 0.65, 0.2, 0.8, 0.1, 0.9)


def plot_known_stream_search_single_dm(
    result: Mapping[str, Any],
    *,
    track_phi12: tuple[ArrayLike, ArrayLike] | None = None,
    other_tracks: Sequence[tuple[str, ArrayLike, ArrayLike]] = (),
    ax: Axes | None = None,
    vmax: float | None = None,
    cmap: str = "gray_r",
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Map of a known-stream search at one distance modulus.

    ``imshow`` of the density on the phi1/phi2 grid, phi1 increasing to the
    left, with the track of the stream and the other galstreams tracks in the
    window. Does not save the figure: the workflows do.

    Parameters
    ----------
    result : mapping
        Output of ``known_stream_search_single_dm``.
    track_phi12 : tuple of array_like, optional
        (phi1, phi2) of the stream track (deg), drawn as a line.
    other_tracks : sequence of (label, phi1, phi2), optional
        Other tracks in the window (deg), labelled.
    ax : matplotlib.axes.Axes, optional
        Axes to draw in. Default: a new figure.
    vmax : float, optional
        Top of the colour scale. Default: a high percentile of the density.
    cmap : str, optional
        Colour map.
    title : str, optional
        Default: the distance modulus and the distance.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Notes
    -----
    Code to reuse: celestial_currents
    ``visualization.py::plot_custom_stream_at_mu``, rewritten with ``imshow``
    on the grid and without saving; ``_find_overlapping_streams`` for the
    other tracks.
    """
    raise NotImplementedError


def plot_galstreams_selection(
    selection: pd.DataFrame,
    mws: galstreams.MWStreams,
    footprint_mask: NDArray[np.bool_],
    *,
    nest: bool,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Sky map of the footprint, with the known streams selected for the search.

    The footprint in light grey, zoomed on it, in a McBryde projection centred
    on its mean right ascension. The whole track of each selected stream is
    drawn in one color, and labelled with its stream name on its part inside
    the footprint: in the middle, or, if that label would overlap one already
    placed (longest tracks first), at the first free position along that part.

    Parameters
    ----------
    selection : pandas.DataFrame
        Output of ``catalogs.galstreams_catalog.select_galstreams_in_footprint``.
    mws : galstreams.MWStreams
        From ``catalogs.galstreams_catalog.load_galstreams``.
    footprint_mask : numpy.ndarray of bool
        HEALPix mask, True outside the footprint.
    nest : bool
        HEALPix ordering of ``footprint_mask``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw in, replaced by the sky axes. Default: a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : skyproj.SkyAxes
        Its ``plot`` and ``text`` take (ra, dec) in degrees.
    """
    footprint_mask = np.asarray(footprint_mask, dtype=bool)
    nside = hp.npix2nside(footprint_mask.size)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    lon_0 = _central_ra(footprint_mask, nest=nest)
    sky = skyproj.McBrydeSkyproj(ax=ax, lon_0=lon_0)
    footprint_map = np.where(footprint_mask, hp.UNSEEN, 1.0)  # UNSEEN: not drawn
    sky.draw_hpxmap(footprint_map, nest=nest, cmap=ListedColormap([FOOTPRINT_GREY]), vmin=0.0, vmax=2.0)

    for track_name in selection["track_name"]:
        points = mws[track_name].track
        for ra_part, dec_part in _split_at_map_edge(points.ra.deg, points.dec.deg, lon_0):
            sky.ax.plot(ra_part, dec_part, color=TRACK_COLOR, lw=1.5)

    renderer = fig.canvas.get_renderer()
    placed = []  # display boxes of the labels already placed
    for track_name, stream_name in zip(selection["track_name"], selection["stream_name"]):
        points = mws[track_name].track
        ra, dec = points.ra.deg, points.dec.deg
        inside = np.flatnonzero(~footprint_mask[hp.ang2pix(nside, ra, dec, lonlat=True, nest=nest)])
        if inside.size == 0:
            continue
        label = sky.ax.text(
            0.0, 0.0, stream_name, fontsize=7, color=INK, ha="center", va="bottom",
            path_effects=[withStroke(linewidth=2.5, foreground="white")],
        )
        for fraction in (*LABEL_FRACTIONS, LABEL_FRACTIONS[0]):  # all overlap: the middle
            point = inside[round(fraction * (inside.size - 1))]
            label.set_position((ra[point], dec[point]))
            box = label.get_window_extent(renderer)
            if not any(box.overlaps(other) for other in placed):
                break
        placed.append(label.get_window_extent(renderer))

    handles = [
        Patch(facecolor=FOOTPRINT_GREY, label="Footprint"),
        Line2D([], [], color=TRACK_COLOR, lw=1.5, label="Selected tracks"),
    ]
    # A fixed corner: loc="best" scans every track point, minutes for the galstreams tracks
    sky.ax.legend(handles=handles, loc="upper right", fontsize=8)
    sky.ax.set_title(f"galstreams tracks selected: {len(selection)}", pad=20)
    return fig, sky.ax


def _central_ra(footprint_mask: NDArray[np.bool_], *, nest: bool) -> float:
    """Mean right ascension of the footprint (deg), on the circle: the centre of the map."""
    nside = hp.npix2nside(footprint_mask.size)
    ra, _ = hp.pix2ang(nside, np.flatnonzero(~footprint_mask), lonlat=True, nest=nest)
    if ra.size == 0:
        return 0.0
    ra = np.radians(ra)
    return float(np.degrees(np.arctan2(np.sin(ra).mean(), np.cos(ra).mean())) % 360.0)


def _split_at_map_edge(
    ra: NDArray[np.float64], dec: NDArray[np.float64], lon_0: float
) -> Iterator[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Parts of a line that do not cross the edge of the map (``lon_0 + 180``), in order."""
    lon = (ra - lon_0 + 180.0) % 360.0 - 180.0
    breaks = np.flatnonzero(np.abs(np.diff(lon)) > 180.0) + 1
    return zip(np.split(ra, breaks), np.split(dec, breaks))

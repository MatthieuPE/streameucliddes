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
FOOTPRINT_OUTSIDE_GREY = "0.85"  # single-dm maps: bins outside the footprint
DENSITY_PERCENTILES = (1.0, 99.5)  # single-dm maps: default colour scale
COLORBAR_BOUNDS = (1.01, 0.0, 0.015, 1.0)  # single-dm maps: colour bar, in map axes fractions
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

    The colour scale goes from the 1st percentile of the density to
    ``vmax`` (default: its 99.5th percentile), to bring out faint streams.
    Bins outside the footprint are light grey. The track of the stream is a
    thin dashed line, not to hide the stream; the other tracks have one
    colour each and a legend.

    Parameters
    ----------
    result : mapping
        Output of ``known_stream_search_single_dm``, or a result of
        ``known_stream_search_dm_scan`` or ``known_stream_search_dm_along_track``.
    track_phi12 : tuple of array_like, optional
        (phi1, phi2) of the stream track (deg), drawn as a line.
    other_tracks : sequence of (label, phi1, phi2), optional
        Other tracks in the window (deg), labelled.
    ax : matplotlib.axes.Axes, optional
        Axes to draw in. Default: a new figure.
    vmax : float, optional
        Top of the colour scale. Default: the 99.5th percentile of the density
        (``DENSITY_PERCENTILES``).
    cmap : str, optional
        Colour map.
    title : str, optional
        Default: the track, the distance modulus and distance (or the offset
        along the track), and the isochrone.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Notes
    -----
    Adapted from celestial_currents
    ``visualization.py::plot_custom_stream_at_mu``, with ``imshow`` on the
    grid instead of a HEALPix map, and without saving.
    """
    density = np.ma.masked_invalid(np.ma.asarray(result["density"], dtype=float))
    phi1_centres, phi2_centres = result["phi1_centres"], result["phi2_centres"]
    half_bin = result["bin_size_deg"] / 2
    extent = (phi1_centres[0] - half_bin, phi1_centres[-1] + half_bin,
              phi2_centres[0] - half_bin, phi2_centres[-1] + half_bin)
    if ax is None:
        width = extent[1] - extent[0]
        height = extent[3] - extent[2]
        fig, ax = plt.subplots(figsize=(10.0, float(np.clip(10.0 * height / width, 2.0, 8.0)) + 1.0))
    else:
        fig = ax.figure

    values = density.compressed()
    vmin = float(np.percentile(values, DENSITY_PERCENTILES[0])) if values.size else 0.0
    if vmax is None:
        vmax = float(np.percentile(values, DENSITY_PERCENTILES[1])) if values.size else 1.0
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(FOOTPRINT_OUTSIDE_GREY)  # bins outside the footprint
    image = ax.imshow(density.T, origin="lower", extent=extent, cmap=colormap, vmin=vmin, vmax=vmax,
                      aspect="equal", interpolation="nearest")
    # inset of the map: the height of the map, and seen by constrained layout (no crop when saved)
    colorbar_ax = ax.inset_axes(COLORBAR_BOUNDS)
    fig.colorbar(image, cax=colorbar_ax, label="Stars per bin (smoothed)")

    if track_phi12 is not None:
        ax.plot(*track_phi12, color=TRACK_COLOR, lw=1.0, ls="--", alpha=0.8,
                label=result.get("track_name") or "Track")
    for color, (label, phi1, phi2) in zip(CATEGORICAL_COLORS[1:], other_tracks):
        ax.plot(phi1, phi2, color=color, lw=1.0, ls="--", alpha=0.8, label=label)
    if track_phi12 is not None or len(other_tracks):
        ax.legend(loc="upper right", fontsize="small", framealpha=0.8, labelcolor=INK)

    ax.set(xlim=(extent[1], extent[0]), ylim=(extent[2], extent[3]),  # phi1 increasing to the left
           xlabel=r"$\phi_1$ [deg]", ylabel=r"$\phi_2$ [deg]")
    ax.set_title(_single_dm_title(result) if title is None else title, color=INK)
    return fig, ax


def _single_dm_title(result: Mapping[str, Any]) -> str:
    """Track, distance modulus (or offset along the track) and isochrone of a map."""
    parts = [result["track_name"]] if result.get("track_name") else []
    if "distance_modulus" in result:
        parts.append(f"dm {result['distance_modulus']:.1f} ({result['distance_kpc']:.1f} kpc)")
    else:
        parts.append(f"dm along the track {result['dm_offset']:+.1f}")
    isochrone = result["isochrone"]
    parts.append(f"{isochrone['age']:g} Gyr, Z = {isochrone['metallicity']:.2g}")
    return ", ".join(parts)


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

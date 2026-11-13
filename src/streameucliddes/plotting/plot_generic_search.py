"""Figures of the generic stream search (analysis.generic_stream_search)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from streameucliddes.plotting.plot_cmd import plot_match_filter
from streameucliddes.plotting.plot_stream_frame_search import (
    DENSITY_PERCENTILES,
    TRACK_COLOR,
    _label_tracks,
    draw_stream_track_on_sky,
)
from streameucliddes.plotting.plot_sky_density import plot_2D_density_from_hp

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray

# The three maps of a result, with their titles
MAP_PANELS = (
    ("density", "Map before correction"),
    ("background", "Fitted background"),
    ("residual", "Corrected map (map - background)"),
)
# Known stream tracks on the corrected map: faint and dashed, to see the map through them
KNOWN_TRACK_STYLE = {"color": TRACK_COLOR, "lw": 1.0, "ls": "--", "alpha": 0.5}
KNOWN_TRACK_LABEL_ALPHA = 0.7
# Opacity of the coordinate grid of the sky maps (skyproj draws it opaque): the maps show through
GRID_ALPHA = 0.3


def plot_generic_search_result(
    result: Mapping[str, Any],
    *,
    cmd_stars: pd.DataFrame | None = None,
    color_lim: tuple[float, float] | None = None,
    mag_lim: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    lat_range: tuple[float, float] | None = None,
    cmap: str = "gray_r",
    grid_alpha: float = GRID_ALPHA,
    title: str | None = None,
) -> tuple[Figure, list[Axes]]:
    """Four panels of a generic-search map: CMD selection, map, background, corrected map.

    1. The match filter on the CMD of ``cmd_stars``
       (``plotting.plot_cmd.plot_match_filter``), with the number of stars
       the map selects.
    2. The map before correction (``density``), 3. the fitted background,
       4. the corrected map (``residual``): HEALPix maps
       (``plotting.plot_sky_density.plot_2D_density_from_hp``), zoomed on the
       footprint. The map and the background share their colour scale (1st
       to 99.5th percentile of the map, ``DENSITY_PERCENTILES``); the
       corrected map has its own. Their coordinate grid is faint
       (``grid_alpha``).

    The figure is not saved.

    Parameters
    ----------
    result : mapping
        Output of ``analysis.generic_stream_search.generic_search_single_dm``.
    cmd_stars : pandas.DataFrame, optional
        Stars of the CMD panel, e.g.
        ``analysis.generic_stream_search.generic_search_cmd_stars(search)``.
        Default: the match filter alone.
    color_lim, mag_lim : tuple of float, optional
        Ranges of the CMD panel (see ``plot_match_filter``): give them for
        frames of a gif.
    lon_range, lat_range : tuple of float, optional
        Ranges of the maps (deg). Default: those of the footprint
        (``_footprint_lonlat_ranges``).
    cmap : str, optional
        Colour map of the maps.
    grid_alpha : float, optional
        Opacity of the coordinate grid of the maps (1: as skyproj draws it).
    title : str, optional
        Default: the distance modulus and distance, and the isochrone.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        CMD, map, background, corrected map.

    Notes
    -----
    Adapted from the old ``plot_results_find_stream`` (2x2 panels); the
    overlays of known objects are left out for now.
    """
    if lon_range is None or lat_range is None:
        default_lon, default_lat = _footprint_lonlat_ranges(result["footprint_mask"], nest=result["nest"])
        lon_range = default_lon if lon_range is None else lon_range
        lat_range = default_lat if lat_range is None else lat_range

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    axes = list(axes.ravel())
    plot_match_filter(
        result["distance_modulus"], result["isochrone"], result["match_filter"], stars=cmd_stars,
        color_lim=color_lim, mag_lim=mag_lim, axes=axes[0],
        title=f"Match filter: {result['n_selected']:,} stars selected",
    )

    density_values = np.ma.compressed(result["density"])
    low, high = DENSITY_PERCENTILES
    density_scale = tuple(np.percentile(density_values, [low, high])) if density_values.size else (None, None)
    for index, (key, panel_title) in enumerate(MAP_PANELS, start=1):
        values = np.ma.compressed(result[key])
        vmin, vmax = density_scale if key != "residual" else (
            tuple(np.percentile(values, [low, high])) if values.size else (None, None)
        )
        _, axes[index] = plot_2D_density_from_hp(
            np.ma.filled(np.ma.asarray(result[key], dtype=float), hp.UNSEEN), nest=result["nest"],
            fig=fig, ax=axes[index], lon_range=lon_range, lat_range=lat_range, title=panel_title,
            cmap=cmap, vmin=vmin, vmax=vmax, label_fontsize=9, title_fontsize=11, tick_labelsize=7,
        )
        axes[index].gridlines.set_alpha(grid_alpha)  # skyproj's grid, a LineCollection

    fig.suptitle(_title(result) if title is None else title)
    return fig, axes


def plot_generic_search_residual_map(
    result: Mapping[str, Any],
    *,
    tracks: Sequence[tuple[str, ArrayLike, ArrayLike]] = (),
    label_tracks: bool = False,
    grid_alpha: float = GRID_ALPHA,
    lon_range: tuple[float, float] | None = None,
    lat_range: tuple[float, float] | None = None,
    cmap: str = "gray_r",
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Corrected map of a generic search, with the known stream tracks over it.

    The corrected map (``residual``) alone, as in the fourth panel of
    ``plot_generic_search_result`` (same colour scale), larger. The tracks
    are faint dashed lines, not to hide the map; with ``label_tracks``, each
    is labelled on its part inside the footprint. The coordinate grid is
    faint (``grid_alpha``).

    The figure is not saved.

    Parameters
    ----------
    result : mapping
        Output of ``analysis.generic_stream_search.generic_search_single_dm``.
    tracks : sequence of (label, ra, dec), optional
        Tracks to draw (deg), e.g. the galstreams tracks crossing the
        footprint.
    label_tracks : bool, optional
        Write the name of each track crossing the footprint.
    grid_alpha : float, optional
        Opacity of the coordinate grid (1: as skyproj draws it).
    lon_range, lat_range : tuple of float, optional
        Ranges of the map (deg). Default: those of the footprint.
    cmap : str, optional
        Colour map.
    title : str, optional
        Default: the distance modulus and distance, and the isochrone.
    ax : matplotlib.axes.Axes, optional
        A plain axes of an existing figure, turned into the sky map (e.g. one
        panel of several); the title is then that of the panel. Default: a
        new figure, the title above it.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : skyproj.SkyAxes
        Its ``plot`` and ``text`` take (ra, dec) in degrees.
    """
    footprint_mask = np.asarray(result["footprint_mask"], dtype=bool)
    if lon_range is None or lat_range is None:
        default_lon, default_lat = _footprint_lonlat_ranges(footprint_mask, nest=result["nest"])
        lon_range = default_lon if lon_range is None else lon_range
        lat_range = default_lat if lat_range is None else lat_range

    in_panel = ax is not None
    fig, ax = (ax.figure, ax) if in_panel else plt.subplots(figsize=(10.0, 7.0))
    values = np.ma.compressed(result["residual"])
    vmin, vmax = tuple(np.percentile(values, DENSITY_PERCENTILES)) if values.size else (None, None)
    fig, ax = plot_2D_density_from_hp(
        np.ma.filled(np.ma.asarray(result["residual"], dtype=float), hp.UNSEEN), nest=result["nest"],
        fig=fig, ax=ax, lon_range=lon_range, lat_range=lat_range, title=MAP_PANELS[2][1],
        cmap=cmap, vmin=vmin, vmax=vmax, label_fontsize=11, title_fontsize=12, tick_labelsize=9,
    )
    ax.gridlines.set_alpha(grid_alpha)  # skyproj's grid, a LineCollection
    for _, ra, dec in tracks:
        draw_stream_track_on_sky(ax, ra, dec, **KNOWN_TRACK_STYLE)
    if label_tracks:
        _label_tracks(ax, tracks, footprint_mask, nest=result["nest"], alpha=KNOWN_TRACK_LABEL_ALPHA)

    title = _title(result) if title is None else title
    if in_panel:
        ax.set_title(title, pad=20)
    else:
        fig.suptitle(title)
    return fig, ax


def _title(result: Mapping[str, Any]) -> str:
    """Distance modulus and distance, and isochrone of a map."""
    isochrone = result["isochrone"]
    return (f"dm {result['distance_modulus']:.1f} ({result['distance_kpc']:.1f} kpc), "
            f"{isochrone['age']:g} Gyr, Z = {isochrone['metallicity']:.2g}")


def _footprint_lonlat_ranges(
    footprint_mask: NDArray[np.bool_], *, nest: bool, margin_deg: float = 1.0
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Longitude and latitude ranges of a HEALPix footprint, for zoomed maps.

    The longitude range is taken around the footprint centre, so that a
    footprint across ra = 0 gives e.g. (-20, 100), not (0, 360).

    Parameters
    ----------
    footprint_mask : numpy.ndarray of bool
        HEALPix mask, True outside the footprint.
    nest : bool
        HEALPix ordering of ``footprint_mask``.
    margin_deg : float, optional
        Added on each side (deg).

    Returns
    -------
    lon_range, lat_range : tuple of float
        (deg), the longitudes in [-180, 360).
    """
    nside = hp.npix2nside(np.size(footprint_mask))
    lon, lat = hp.pix2ang(nside, np.flatnonzero(~np.asarray(footprint_mask, dtype=bool)), lonlat=True, nest=nest)
    if lon.size == 0:
        return (-180.0, 180.0), (-90.0, 90.0)
    centre = np.degrees(np.arctan2(np.sin(np.radians(lon)).mean(), np.cos(np.radians(lon)).mean()))
    offset = (lon - centre + 180.0) % 360.0 - 180.0  # around the centre
    lon_range = (float(centre + offset.min() - margin_deg), float(centre + offset.max() + margin_deg))
    lat_range = (float(max(lat.min() - margin_deg, -90.0)), float(min(lat.max() + margin_deg, 90.0)))
    return lon_range, lat_range

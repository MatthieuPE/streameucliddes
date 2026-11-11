"""Figures of the generic stream search (analysis.generic_stream_search)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from streameucliddes.plotting.plot_cmd import plot_match_filter
from streameucliddes.plotting.plot_known_stream_search import DENSITY_PERCENTILES
from streameucliddes.plotting.plot_sky_density import plot_2D_density_from_hp

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

# The three maps of a result, with their titles
MAP_PANELS = (
    ("density", "Map before correction"),
    ("background", "Fitted background"),
    ("residual", "Corrected map (map - background)"),
)


def plot_generic_search_result(
    result: Mapping[str, Any],
    *,
    cmd_stars: pd.DataFrame | None = None,
    color_lim: tuple[float, float] | None = None,
    mag_lim: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    lat_range: tuple[float, float] | None = None,
    cmap: str = "gray_r",
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
       corrected map has its own.

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

    if title is None:
        isochrone = result["isochrone"]
        title = (f"dm {result['distance_modulus']:.1f} ({result['distance_kpc']:.1f} kpc), "
                 f"{isochrone['age']:g} Gyr, Z = {isochrone['metallicity']:.2g}")
    fig.suptitle(title)
    return fig, axes


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

"""Star density maps on the sky (HEALPix, skyproj).

plot_overlapped_objects and plot_single_stream_track belong here, but were never
defined in this package (see docs/dev/package_structure.md)."""

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import skyproj

from streameucliddes.plotting import plot_style  # noqa: F401  (sets the figure style)

def plot_2D_density(
    data,
    columns=("ra", "dec"),
    nside=48,
    nest=False,
    fig=None,
    ax=None,
    lon_range=None,
    lat_range=None,
    title=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    add_colorbar=True,
    smoothing=False,
    smoothing_scale_degrees=0.3,
):
    """
    Plot a 2D stellar density map on the sky using skyproj + HEALPix.

    Parameters
    ----------
    data : pd.DataFrame
        Table containing at least RA/Dec columns (degrees).
    columns : tuple[str, str]
        Names of (RA, Dec) columns in degrees.
    nside : int
        HEALPix NSIDE resolution (power of 2).
    nest : bool
        HEALPix ordering scheme. False = RING, True = NESTED.
    fig, ax : matplotlib Figure/Axes, optional
        Existing figure and axes.
    lon_range, lat_range : tuple, optional
        Plot bounds in degrees. If None, inferred from data.
    title : str
        Plot title.
    cmap : str
        Colormap for density map.
    vmin, vmax : float, optional
        Color scale limits.
    add_colorbar : bool
        Whether to add a colorbar.

    Returns
    -------
    fig, ax
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    # Read coordinates and keep only valid values
    ra = np.asarray(data[columns[0]], dtype=float)
    dec = np.asarray(data[columns[1]], dtype=float)
    valid = np.isfinite(ra) & np.isfinite(dec) & (dec >= -90.0) & (dec <= 90.0)

    ra = ra[valid]
    dec = dec[valid]

    # Build HEALPix count map
    npix = hp.nside2npix(nside)
    pix_idx = hp.ang2pix(nside, ra, dec, lonlat=True, nest=nest)
    hp_map = np.bincount(pix_idx, minlength=npix).astype(np.float32)
    # convert degrees in radian
    smoothing_scale_radian = 2 * np.pi / 360.0 * smoothing_scale_degrees
    if smoothing:
        hp_map = hp.sphtfunc.smoothing(hp_map, sigma=smoothing_scale_radian)
    # Build projection
    sp = skyproj.McBrydeSkyproj(ax=ax)

    # Optional auto-zoom around data (with small padding)
    if lon_range is None and ra.size > 0:
        pad = 1.0
        lon_range = (max(0.0, np.nanmin(ra) - pad), min(360.0, np.nanmax(ra) + pad))
    if lat_range is None and dec.size > 0:
        pad = 1.0
        lat_range = (max(-90.0, np.nanmin(dec) - pad), min(90.0, np.nanmax(dec) + pad))

    im, *_ = sp.draw_hpxmap(
        hp_map,
        nest=nest,
        lon_range=lon_range,
        lat_range=lat_range,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if add_colorbar:
        fig.colorbar(im, ax=sp.ax, label="Counts per HEALPix pixel")
    sp.ax.set_title(title)
    return fig, sp.ax


def plot_2D_density_from_hp(
    hp_map,
    nest=False,
    fig=None,
    ax=None,
    lon_range=None,
    lat_range=None,
    title=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    add_colorbar=True,
    label_fontsize=14,
    title_fontsize=14,
    tick_labelsize=None,
):
    """
    Plot a 2D stellar density map on the sky directly from HEALPix map

    Parameters
    ----------
    hp_map : healpy map
    nest : bool
        HEALPix ordering scheme. False = RING, True = NESTED.
    fig, ax : matplotlib Figure/Axes, optional
        Existing figure and axes.
    lon_range, lat_range : tuple, optional
        Plot bounds in degrees. If None, inferred from data.
    title : str
        Plot title.
    cmap : str
        Colormap for density map.
    vmin, vmax : float, optional
        Color scale limits.
    add_colorbar : bool
        Whether to add a colorbar.
    label_fontsize, title_fontsize : float
        Font size for the axis labels / title. skyproj sets its own sizes
        rather than following rcParams, so these need to be passed explicitly
        (e.g. smaller values when placing several panels in one figure).
    tick_labelsize : float, optional
        Font size for the tick labels (axes and colorbar). If None, skyproj's
        default is used.

    Returns
    -------
    fig, ax
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    # Build projection
    sp = skyproj.McBrydeSkyproj(ax=ax)

    im, *_ = sp.draw_hpxmap(
        hp_map,
        nest=nest,
        lon_range=lon_range,
        lat_range=lat_range,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    sp.ax.xaxis.label.set_size(label_fontsize)
    sp.ax.yaxis.label.set_size(label_fontsize)
    sp.ax.yaxis.labelpad = -30
    if tick_labelsize is not None:
        sp.ax.tick_params(labelsize=tick_labelsize)

    if add_colorbar:
        cbar = fig.colorbar(im, ax=sp.ax, label="Counts per HEALPix pixel")
        cbar.set_label("Counts per HEALPix pixel", fontsize=label_fontsize)
        if tick_labelsize is not None:
            cbar.ax.tick_params(labelsize=tick_labelsize)
    sp.ax.set_title(title, fontsize=title_fontsize, pad=20)

    return fig, sp.ax

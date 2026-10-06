import os
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from matplotlib.colors import LogNorm
import skyproj
import healpy as hp

import glob
from PIL import Image

    
def plot_CMD(data, fig=None, ax=None, mag_cols =['mag_g', 'mag_r'], ylim=(15,28), xlim = (0.1, 1.2), xlabel = rf"$g-r$", ylabel = rf"$g$", color = None,  cmap='cividis', norm = 'log', bins_colors=None, bins_mag=None):
    """
    To plot 2D hist in the CMD

    data must be data frame of dict.
    """
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8,4))

    mag1 = pd.to_numeric(data[mag_cols[0]], errors='coerce')
    mag2 = pd.to_numeric(data[mag_cols[1]], errors='coerce')

    if bins_colors is None:
        bins_colors = np.arange(xlim[0], xlim[1], 0.01)
    if bins_mag is None:
        bins_mag = np.arange(ylim[0], ylim[1], 0.1)
        
    counts, xedges, yedges = np.histogram2d(mag1 - mag2, mag1, bins=[bins_colors, bins_mag])
    if norm is None:
        label = 'N'
    elif norm == 'log':
        norm = LogNorm()
        label = 'log10(N)'
    else:
        raise ValueError(f"Invalid norm: {norm}. Use None or 'log'.")

    im = ax.imshow(counts.T, origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap, norm=norm)
    #plt.colorbar(im, ax=ax, label=label)
    #plt.colorbar(im, ax=ax)
    cbar = plt.colorbar(im, ax=ax)
    #cbar.set_label(label)

    #cbar.ax.yaxis.set_label_coords(0.5, 1.05)
    #cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label, loc='top')

    ax.set_xlabel(f'{mag_cols[0]} - {mag_cols[1]}')
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
    smoothing= False,
    smoothing_scale_degrees=0.3
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
    smoothing_scale_radian = 2*np.pi/360.*smoothing_scale_degrees
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


def get_hp_map(
    data,
    columns=("ra", "dec"),
    nside=48,
    nest=False,
    lon_range=None,
    lat_range=None,
    title=None,
    vmin=None,
    vmax=None,
    smoothing= False,
    smoothing_scale_degrees=0.3
    ):
    """

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
    lon_range, lat_range : tuple, optional
        Plot bounds in degrees. If None, inferred from data.
    title : str
        Plot title.

    Returns
    -------
    hpmap
    """

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
    smoothing_scale_radian = 2*np.pi/360.*smoothing_scale_degrees
    if smoothing:
        hp_map = hp.sphtfunc.smoothing(hp_map, sigma=smoothing_scale_radian)
    # Build projection
    #sp = skyproj.McBrydeSkyproj(ax=ax)

    return hp_map


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



# ================================ Plotting functions for 2D polynomial fitting =====================================



def plot_results_find_stream(
    results,
    mag_bounds=(16.0, 28.0),
    color_bounds=(-0.5, 1.5),
    cmap="Greys",
    overlay_tracks=None,
    stream_name=None,
    show_overlapped_objects=True,
    overlap_kwargs=None,
    suptitle=None,
    save=False,
    show=True,
    fig_folder="../../figures",
    fname=None,
    fname_template="streamfind_dm{dm:.2f}.png",
    dpi=150,
):
    """
    2x2 panel figure from RubinStream.analysis.pipeline.find_stream() output:
    1. CMD + match filter polygon (+ faint magnitude-cut line, if applied)
    2. data_map, with dwarves/globular clusters/stream tracks overlaid (see
       `stream_name` and `show_overlapped_objects`)
    3. fit_map
    4. residual_map

    Parameters
    ----------
    stream_name : str, optional
        If given, the data_map panel highlights only this one galstreams
        stream by name (via plot_single_stream_track, individually labeled
        with its name/distance) plus globular clusters and dwarves (each
        collapsed to a single "type" legend entry). If None (default),
        every stream crossing the DP2 footprint is drawn instead, and all
        three object types (streams/GCs/dwarves) are each collapsed to a
        single "type" legend entry too (see plot_overlapped_objects).
    show_overlapped_objects : bool
        If False, skip the data_map overlay entirely -- e.g. for repeated
        calls (like scan_find_stream's gif frames) where the extra
        network/file I/O and visual clutter aren't wanted.
    overlap_kwargs : dict, optional
        Forwarded to plot_overlapped_objects (e.g. {"gc_kwargs": {"s": 10}}).

    Returns
    -------
    fig, ax, fname_path : figure, flattened (4,) axes array, path the figure
        was saved to (fname_path is None if save=False).
    """
    data = results["data"]
    mag_to_use = results["mag_to_use"]
    faint_mag_cut = results.get("faint_mag_cut")
    polygon_vertices = results["polygon_vertices"]
    ra_min, ra_max, dec_min, dec_max = results["footprint_bounds"]
    dm = results["distance_modulus"]

    figsize1 = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(2, 2, figsize=(2.5 * figsize1[0], 2.5 * figsize1[1]), constrained_layout=True)
    ax = ax.ravel()

    # skyproj lays out its own ticks/labels/colorbars rather than following
    # rcParams, and the global constrained-layout padding (set very tight in
    # plot_style.py for compact single-panel figures) is too small once two
    # skyproj panels sit stacked in the same figure -- widen it here and use
    # smaller fonts so the panels stop bleeding into each other.
    w_pad, h_pad, wspace, hspace = 0.05, 0.05, 0.15, 0.15
    try:
        #print(f"1 - Applying w_pad={w_pad}, h_pad={h_pad}, wspace={wspace}, hspace = {hspace}")
        fig.get_layout_engine().set(w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace)
    except AttributeError:
        #print(f"2 - Applying w_pad={w_pad}, h_pad={h_pad}, wspace={wspace}, hspace = {hspace}")
        fig.set_constrained_layout_pads(w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace)

    fig, ax[0] = plot_CMD(
        data, fig=fig, ax=ax[0], mag_cols=mag_to_use, ylim=mag_bounds, xlim=color_bounds, cmap=cmap,
    )
    ax[0].plot(polygon_vertices[:, 0], polygon_vertices[:, 1], color="C0", lw=2, label="Match filter")
    if faint_mag_cut is not None:
        ax[0].axhline(faint_mag_cut, color="C1", lw=1.5, ls="dashed", label=f"Magnitude cut ({faint_mag_cut:g})")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Match filter", fontsize=11)

    panels = [("data_map", "Data map"), ("fit_map", "Fit map"), ("residual_map", "Residual map")]
    for i, (key, panel_title) in enumerate(panels, start=1):
        fig, ax[i] = plot_2D_density_from_hp(
            results[key],
            lon_range=[ra_min, ra_max],
            lat_range=[dec_min, dec_max],
            fig=fig,
            ax=ax[i],
            cmap=cmap,
            title=panel_title,
            label_fontsize=9,
            title_fontsize=11,
            tick_labelsize=7,
        )
        _apply_overlay_tracks(ax[i], overlay_tracks)

    if show_overlapped_objects:
        overlap_kwargs = dict(overlap_kwargs or {})
        if stream_name is None:
            # Every crossing stream, all three object types collapsed to one
            # "type" legend entry each.
            overlap_kwargs.setdefault("stream_kwargs", {"label_mode": "type"})
            plot_overlapped_objects(fig=fig, ax=ax[1], plot_footprint=False, **overlap_kwargs)
        else:
            # Just GCs + dwarves (plot_overlapped_objects' own defaults already
            # collapse these to one "type" entry each) plus the one target stream,
            # individually labeled, combined into a single legend.
            plot_overlapped_objects(fig=fig, ax=ax[1], object_to_plot=("gc", "dwarf"),
                                     plot_footprint=False, **overlap_kwargs)
            plot_single_stream_track(stream_name, fig=fig, ax=ax[1], show_legend=False)
            handles, labels = ax[1].get_legend_handles_labels()
            if handles:
                ax[1].legend(handles, labels, loc="upper right", fontsize=8)

    if suptitle is None:
        suptitle = f"dm = {dm:.2f} ({results['distance_kpc']:.1f})"
    fig.suptitle(suptitle)

    fname_path = None
    if save:
        os.makedirs(fig_folder, exist_ok=True)
        fname_path = os.path.join(fig_folder, fname or fname_template.format(dm=dm))
        fig.savefig(fname_path, dpi=dpi)

    if not show:
        plt.close(fig)

    return fig, ax, fname_path




def make_gif(
    figures_folder="../../figures",
    pattern="streamfind_dm*.png",
    gif_name="streamfind.gif",
    fps=2,
    sort_numeric=True,
    dither=True,
):
    """
    Make a gif from several figures saved in `figures_folder`.

    Each frame is quantized to a 256-color palette with Pillow's octree
    quantizer (`Image.Quantize.FASTOCTREE`) plus Floyd-Steinberg dithering by
    default. GIF only supports 256 colors per frame; imageio's default writer
    quantizes without dithering, which visibly flattens/banding continuous
    colormaps (e.g. the density maps here). Dithering trades that banding for
    a bit of grain, which reads as noticeably more detail preserved.

    Parameters
    ----------
    pattern : str
        Glob pattern (relative to figures_folder) matching frame filenames.
    sort_numeric : bool
        If True, order frames by the first number found in each filename
        (e.g. the distance modulus) instead of lexicographic order.
    dither : bool
        If True (default), apply Floyd-Steinberg dithering during palette
        quantization. Set False for flatter, more compressible frames at the
        cost of visible color banding.

    Returns
    -------
    gif_path : str
    """
    files = glob.glob(os.path.join(figures_folder, pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in '{figures_folder}'.")

    if sort_numeric:
        def _sort_key(path):
            match = re.search(r"[-+]?\d*\.?\d+", os.path.basename(path))
            return float(match.group()) if match else path

        files = sorted(files, key=_sort_key)
    else:
        files = sorted(files)

    dither_mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    frames = [
        Image.open(f).convert("RGB").quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=dither_mode)
        for f in files
    ]

    gif_path = os.path.join(figures_folder, gif_name)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        disposal=2,
    )
    print(f"{gif_name} created ({len(files)} frames) at {gif_path}")
    return gif_path
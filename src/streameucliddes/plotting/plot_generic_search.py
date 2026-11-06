"""Figures of the generic stream search (analysis.generic_stream_search).

plot_generic_search_result calls _apply_overlay_tracks, plot_overlapped_objects and
plot_single_stream_track, which were never defined in this package (see
docs/dev/package_structure.md)."""

import os

import matplotlib.pyplot as plt

from streameucliddes.plotting.plot_cmd import plot_CMD
from streameucliddes.plotting.plot_sky_density import plot_2D_density_from_hp

def plot_generic_search_result(
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
    2x2 panel figure from analysis.generic_stream_search.generic_search_single_dm() output:
    1. CMD + match filter polygon (+ a line per bound of a range cut on the
       CMD magnitude, if any)
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
        calls (like run_generic_search's gif frames) where the extra
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
    cuts = results.get("cuts") or {}
    polygon_vertices = results["polygon_vertices"]
    ra_min, ra_max, dec_min, dec_max = results["footprint_bounds"]
    dm = results["distance_modulus"]

    figsize1 = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(
        2, 2, figsize=(2.5 * figsize1[0], 2.5 * figsize1[1]), constrained_layout=True
    )
    ax = ax.ravel()

    # skyproj lays out its own ticks/labels/colorbars rather than following
    # rcParams, and the global constrained-layout padding (set very tight in
    # plot_style.py for compact single-panel figures) is too small once two
    # skyproj panels sit stacked in the same figure -- widen it here and use
    # smaller fonts so the panels stop bleeding into each other.
    w_pad, h_pad, wspace, hspace = 0.05, 0.05, 0.15, 0.15
    try:
        # print(f"1 - Applying w_pad={w_pad}, h_pad={h_pad}, wspace={wspace}, hspace = {hspace}")
        fig.get_layout_engine().set(
            w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace
        )
    except AttributeError:
        # print(f"2 - Applying w_pad={w_pad}, h_pad={h_pad}, wspace={wspace}, hspace = {hspace}")
        fig.set_constrained_layout_pads(
            w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace
        )

    fig, ax[0] = plot_CMD(
        data,
        fig=fig,
        ax=ax[0],
        mag_cols=mag_to_use,
        ylim=mag_bounds,
        xlim=color_bounds,
        cmap=cmap,
    )
    ax[0].plot(
        polygon_vertices[:, 0],
        polygon_vertices[:, 1],
        color="C0",
        lw=2,
        label="Match filter",
    )
    mag_cut = cuts.get(mag_to_use[0])
    if isinstance(mag_cut, tuple):
        for bound in mag_cut:
            if bound is not None:
                ax[0].axhline(
                    bound,
                    color="C1",
                    lw=1.5,
                    ls="dashed",
                    label=f"Magnitude cut ({bound:g})",
                )
    ax[0].legend(fontsize=8)
    ax[0].set_title("Match filter", fontsize=11)

    panels = [
        ("data_map", "Data map"),
        ("fit_map", "Fit map"),
        ("residual_map", "Residual map"),
    ]
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
            plot_overlapped_objects(
                fig=fig, ax=ax[1], plot_footprint=False, **overlap_kwargs
            )
        else:
            # Just GCs + dwarves (plot_overlapped_objects' own defaults already
            # collapse these to one "type" entry each) plus the one target stream,
            # individually labeled, combined into a single legend.
            plot_overlapped_objects(
                fig=fig,
                ax=ax[1],
                object_to_plot=("gc", "dwarf"),
                plot_footprint=False,
                **overlap_kwargs,
            )
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

"""Generic stream search over a grid of distance moduli: figures, gif and results."""

import gc

from streameucliddes.analysis.generic_stream_search import generic_search_dm_scan
from streameucliddes.plotting.make_videos import make_gif
from streameucliddes.plotting.plot_generic_search import plot_generic_search_result


def run_generic_search(
    data,
    distance_modulus_array,
    match_filter_parameters,
    find_stream_kwargs=None,
    plot=False,
    save_figures=False,
    fig_folder="../../figures",
    plot_kwargs=None,
    make_gif_flag=False,
    gif_kwargs=None,
    keep_results=False,
    verbose=True,
):
    """
    Run generic_search_single_dm() over a grid of distance moduli, optionally plotting
    (and saving) each frame and building a gif from the saved figures.

    The scan itself (dust correction and cuts computed once, then one search per
    distance modulus) is analysis.generic_stream_search.generic_search_dm_scan.

    Parameters
    ----------
    data : pandas.DataFrame
    distance_modulus_array : iterable of float
    match_filter_parameters : dict
        Base kwargs for match_filter.build_match_filter; 'distance_modulus'
        is overridden for each value in `distance_modulus_array`.
    find_stream_kwargs : dict, optional
        Extra kwargs forwarded to generic_search_single_dm() (e.g. mag_cols, cuts,
        use_dust_corrected, nside...).
    plot : bool
        If True, display each frame (via plot_generic_search_result, show=True).
    save_figures : bool
        If True, save each frame to `fig_folder`.
    plot_kwargs : dict, optional
        Extra kwargs forwarded to plot_generic_search_result().
    make_gif_flag : bool
        If True, build a gif from `fig_folder` once the scan is done
        (requires save_figures=True).
    gif_kwargs : dict, optional
        Extra kwargs forwarded to make_gif().
    keep_results : bool
        If True, retain the full generic_search_single_dm() output (HEALPix maps, etc.)
        for every distance modulus. Default False: only a lightweight summary
        ('distance_kpc', 'n_selected', 'fname') is kept per distance modulus,
        and the heavy per-frame result is dropped as soon as it has been
        plotted/saved -- important when scanning many distance moduli over a
        large catalog, since retaining every frame's HEALPix maps would
        otherwise grow without bound.

    Returns
    -------
    results_by_dm : dict[float, dict]
        generic_search_single_dm() output (or the lightweight summary, see `keep_results`)
        keyed by distance modulus.
    """
    plot_kwargs = dict(plot_kwargs or {})
    gif_kwargs = dict(gif_kwargs or {})

    results_by_dm = {}
    for dm, results in generic_search_dm_scan(
        data,
        distance_modulus_array,
        match_filter_parameters,
        find_stream_kwargs=find_stream_kwargs,
        verbose=verbose,
    ):
        fname_path = None
        if plot or save_figures:
            _, _, fname_path = plot_generic_search_result(
                results,
                save=save_figures,
                show=plot,
                fig_folder=fig_folder,
                **plot_kwargs,
            )

        if keep_results:
            results_by_dm[dm] = results
        else:
            results_by_dm[dm] = {
                "distance_kpc": results["distance_kpc"],
                "n_selected": results["n_selected"],
                "fname": fname_path,
            }
            del results

        # Release any lingering matplotlib/array reference cycles before the next frame.
        gc.collect()

    if make_gif_flag:
        make_gif(figures_folder=fig_folder, **gif_kwargs)

    return results_by_dm

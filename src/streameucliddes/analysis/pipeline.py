"""Wrapper for the full match-filter / background-fit pipeline.
"""

import gc

import numpy as np

from streameucliddes.analysis import data_manipulation, fit
from streameucliddes.plotting.plotting import plot_results_find_stream, make_gif
from streameucliddes.plotting import plot_style
from streamobs import match_filter
from streameucliddes.utils import convert_DM_to_kpc

DEFAULT_MODEL_TYPE_FIT = {"polynomial": {"degree": 5}}


class SimpleLonLatProj:
    """Trivial "projection" that treats (lon, lat) as flat (x, y), in degrees.

    Used as the `proj` argument of fit.fit_background_model(method="polyfit2d").
    """

    def ang2xy(self, lon, lat, lonlat=True):
        return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)

    def get_extent(self):
        return (0.0, 360.0, -90.0, 90.0)


# Stateless -> shared across calls instead of re-instantiating on every find_stream() call.
_PROJ = SimpleLonLatProj()


def _resolve_mag_columns(data, mag_cols, use_dust_corrected):
    """Swap in '<col>_corrected' columns when available and requested."""
    resolved = []
    for col in mag_cols:
        corrected = f"{col}_corrected"
        if use_dust_corrected and corrected in data.columns:
            resolved.append(corrected)
        else:
            if use_dust_corrected:
                print(f"Warning: requested dust-corrected column '{corrected}' not found in data; using '{col}' instead.")
            resolved.append(col)
    return resolved


def _ensure_dust_corrected(data, mag_cols, ra_col, dec_col, dust_correction_kwargs=None, verbose=False):
    """Compute missing '<col>_corrected' columns in place via data_manipulation.correct_dust_map."""
    missing = [col for col in mag_cols if f"{col}_corrected" not in data.columns]
    if not missing:
        return

    if verbose:
        print(f"Applying dust correction for: {missing}")
    data_manipulation.correct_dust_map(
        data, mag_cols=list(mag_cols), pos_columns=[ra_col, dec_col], **dict(dust_correction_kwargs or {})
    )


def find_stream(
    data,
    match_filter_parameters,
    mag_cols=("mag_g", "mag_r"),
    use_dust_corrected=True,
    dust_correction_kwargs=None,
    ensure_dust_correction=True,
    faint_mag_cut=24.0,
    flag_mag=None,
    nside=512,
    smoothing=True,
    smoothing_scale_degrees=0.3,
    model_type_fit=None,
    fit_method="polyfit2d",
    ra_col="coord_ra",
    dec_col="coord_dec",
    footprint_bounds=None,
    fit_kwargs=None,
    verbose=False,
):
    """
    Run the full pipeline for a single distance modulus: dust correction,
    match filter, magnitude cut, smoothed HEALPix map, polynomial background
    fit, residual.

    Parameters
    ----------
    data : pandas.DataFrame
        Catalog with magnitude and (ra, dec) columns. Mutated in place to add
        '<col>_corrected' columns when `use_dust_corrected=True` and they are
        not already present.
    match_filter_parameters : dict
        Passed to match_filter.build_match_filter (must include 'distance_modulus').
    mag_cols : tuple of str
        Base (g, r) magnitude column names.
    use_dust_corrected : bool
        If True, use '<col>_corrected' columns instead of the raw ones.
    dust_correction_kwargs : dict, optional
        Extra kwargs forwarded to data_manipulation.correct_dust_map (e.g.
        'ebv_map' or 'ebv_map_path') when the correction needs to be computed.
    ensure_dust_correction : bool
        If True (default) and `use_dust_corrected`, compute the missing
        '<col>_corrected' columns in place before resolving magnitude columns.
        Set to False when the correction has already been applied upfront
        (e.g. by scan_find_stream) to skip the per-call existence check.
    faint_mag_cut : float
        Faint-end magnitude cut applied on mag_cols[0]. Ignored if `flag_mag`
        is provided.
    flag_mag : array-like of bool, optional
        Precomputed magnitude-cut mask (same length as `data`). If provided,
        the internal call to data_manipulation.apply_magnitude_cut is skipped
        -- useful when scanning several distance moduli, since the magnitude
        cut does not depend on distance modulus and only needs computing once.
    nside, smoothing, smoothing_scale_degrees : HEALPix map parameters.
    model_type_fit : dict
        Passed to fit.fit_background_model (default: degree-5 polynomial).
    fit_method : str
        Passed to fit.fit_background_model (default: "polyfit2d").
    footprint_bounds : tuple (ra_min, ra_max, dec_min, dec_max), optional
        If None, inferred from the match-filtered + magnitude-cut selection.

    Returns
    -------
    results : dict
        Includes 'data_map', 'fit_map', 'residual_map' (masked HEALPix maps),
        plus lightweight metadata needed to reproduce/plot them (CMD selection
        mask, polygon, footprint). Heavy intermediates (full-row selection,
        per-pixel lon/lat grids, the raw fit-solver workspace) are dropped
        rather than returned, to keep memory bounded when scanning many
        distance moduli over a large catalog.
    """
    if model_type_fit is None:
        model_type_fit = DEFAULT_MODEL_TYPE_FIT
    fit_kwargs = dict(fit_kwargs or {})

    if use_dust_corrected and ensure_dust_correction:
        _ensure_dust_corrected(data, mag_cols, ra_col, dec_col, dust_correction_kwargs, verbose=verbose)

    mag_to_use = _resolve_mag_columns(data, mag_cols, use_dust_corrected)
    if verbose:
        print(f"Magnitudes used for cut and match filter: {mag_to_use}")

    # --- Match filter + magnitude cut ---
    polygon_vertices = match_filter.build_match_filter(**match_filter_parameters)
    flag_match = match_filter.is_in_match_filter(
        data[mag_to_use[0]], data[mag_to_use[1]], polygon_vertices=polygon_vertices, verbose=verbose,
    )
    if flag_mag is None:
        flag_mag = data_manipulation.apply_magnitude_cut(
            data, mag_col=mag_to_use[0], mag_bounds=(None, faint_mag_cut)
        )
    selection_flag = flag_match & flag_mag
    del flag_match, flag_mag

    # Only pull the two coordinate columns for the selected rows, instead of a
    # full-row copy of (potentially very wide) `data` via `data.loc[mask]`.
    ra_values = data[ra_col].to_numpy()[selection_flag]
    dec_values = data[dec_col].to_numpy()[selection_flag]
    n_selected = int(ra_values.size)

    if verbose:
        print(f"Selected {n_selected}/{len(data)} stars in match filter + magnitude cut")

    if footprint_bounds is None:
        ra_min = float(np.min(ra_values))
        ra_max = float(np.max(ra_values))
        dec_min = float(np.min(dec_values))
        dec_max = float(np.max(dec_values))
    else:
        ra_min, ra_max, dec_min, dec_max = footprint_bounds

    # --- Smoothed HEALPix map ---
    data_map, lon, lat, npix = data_manipulation.get_healpy_data_from_table(
        {ra_col: ra_values, dec_col: dec_values},
        columns=[ra_col, dec_col],
        nside=nside,
        smoothing=smoothing,
        smoothing_scale_degrees=smoothing_scale_degrees,
    )
    del ra_values, dec_values

    footprint_mask = (lon < ra_min) | (lon > ra_max) | (lat < dec_min) | (lat > dec_max)
    del lon, lat

    partial_data = np.ma.array(data_map, mask=footprint_mask)
    del data_map

    # --- Polynomial background fit + residual ---
    params_fit, solver_res, prepared_fit = fit.fit_background_model(
        partial_data,
        method=fit_method,
        model_type=model_type_fit,
        proj=_PROJ,
        nside=nside,
        sigma=0.0,
        percent=None,
        mask=footprint_mask,
        **fit_kwargs,
    )

    model_map = np.asarray(prepared_fit["background_map"].filled(np.nan))
    residual_map = partial_data.filled(0.0) - model_map
    del prepared_fit

    distance_modulus = match_filter_parameters["distance_modulus"]

    return {
        "data": data,
        "mag_to_use": mag_to_use,
        "faint_mag_cut": faint_mag_cut,
        "match_filter_parameters": dict(match_filter_parameters),
        "distance_modulus": distance_modulus,
        "distance_kpc": convert_DM_to_kpc(distance_modulus),
        "polygon_vertices": polygon_vertices,
        "selection_flag": selection_flag,
        "n_selected": n_selected,
        "footprint_bounds": (ra_min, ra_max, dec_min, dec_max),
        "footprint_mask": footprint_mask,
        "nside": nside,
        "data_map": partial_data,
        "fit_map": np.ma.array(model_map, mask=footprint_mask),
        "residual_map": np.ma.array(residual_map, mask=footprint_mask),
        "model_type_fit": model_type_fit,
        "params_fit": params_fit,
        "solver_res": solver_res,
    }


def scan_find_stream(
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
    Run find_stream() over a grid of distance moduli, optionally plotting
    (and saving) each frame and building a gif from the saved figures.

    The dust correction and the (distance-modulus-independent) magnitude cut
    are each computed once up front and reused for every distance modulus,
    instead of being recomputed on every find_stream() call.

    Parameters
    ----------
    data : pandas.DataFrame
    distance_modulus_array : iterable of float
    match_filter_parameters : dict
        Base kwargs for match_filter.build_match_filter; 'distance_modulus'
        is overridden for each value in `distance_modulus_array`.
    find_stream_kwargs : dict, optional
        Extra kwargs forwarded to find_stream() (e.g. mag_cols, faint_mag_cut,
        use_dust_corrected, nside...).
    plot : bool
        If True, display each frame (via plot_results_find_stream, show=True).
    save_figures : bool
        If True, save each frame to `fig_folder`.
    plot_kwargs : dict, optional
        Extra kwargs forwarded to plot_results_find_stream().
    make_gif_flag : bool
        If True, build a gif from `fig_folder` once the scan is done
        (requires save_figures=True).
    gif_kwargs : dict, optional
        Extra kwargs forwarded to make_gif().
    keep_results : bool
        If True, retain the full find_stream() output (HEALPix maps, etc.)
        for every distance modulus. Default False: only a lightweight summary
        ('distance_kpc', 'n_selected', 'fname') is kept per distance modulus,
        and the heavy per-frame result is dropped as soon as it has been
        plotted/saved -- important when scanning many distance moduli over a
        large catalog, since retaining every frame's HEALPix maps would
        otherwise grow without bound.

    Returns
    -------
    results_by_dm : dict[float, dict]
        find_stream() output (or the lightweight summary, see `keep_results`)
        keyed by distance modulus.
    """
    find_stream_kwargs = dict(find_stream_kwargs or {})
    plot_kwargs = dict(plot_kwargs or {})
    gif_kwargs = dict(gif_kwargs or {})

    # --- Precompute what doesn't depend on distance modulus, once, for the whole scan ---
    mag_cols = find_stream_kwargs.get("mag_cols", ("mag_g", "mag_r"))
    ra_col = find_stream_kwargs.get("ra_col", "coord_ra")
    dec_col = find_stream_kwargs.get("dec_col", "coord_dec")
    use_dust_corrected = find_stream_kwargs.get("use_dust_corrected", True)
    faint_mag_cut = find_stream_kwargs.get("faint_mag_cut", 24.0)

    if use_dust_corrected:
        _ensure_dust_corrected(
            data, mag_cols, ra_col, dec_col, find_stream_kwargs.get("dust_correction_kwargs"), verbose=verbose,
        )
    find_stream_kwargs["ensure_dust_correction"] = False

    mag_to_use = _resolve_mag_columns(data, mag_cols, use_dust_corrected)
    find_stream_kwargs["faint_mag_cut"] = faint_mag_cut
    find_stream_kwargs["flag_mag"] = data_manipulation.apply_magnitude_cut(
        data, mag_col=mag_to_use[0], mag_bounds=(None, faint_mag_cut)
    )
    if verbose:
        print(f"Precomputed dust correction + magnitude cut once for the scan (mag columns: {mag_to_use}).")

    results_by_dm = {}
    for dm in distance_modulus_array:
        if verbose:
            print(f"Processing distance modulus = {dm:.2f}")

        mf_params = dict(match_filter_parameters)
        mf_params["distance_modulus"] = dm

        results = find_stream(data, mf_params, **find_stream_kwargs)

        fname_path = None
        if plot or save_figures:
            _, _, fname_path = plot_results_find_stream(
                results, save=save_figures, show=plot, fig_folder=fig_folder, **plot_kwargs,
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
"""Generic stream search: match filter, smoothed HEALPix map, polynomial background fit."""

import numpy as np
from streamobs import match_filter

from streameucliddes.analysis import background_polynomial_fit
from streameucliddes.analysis.background_polynomial_fit import SimpleLonLatProj
from streameucliddes.analysis.catalog_preparation import ensure_dust_corrected, resolve_mag_columns
from streameucliddes.analysis.star_density_maps import star_density_healpix
from streameucliddes.catalogs.catalog_cuts import get_cut_mask
from streameucliddes.unit_conversions import convert_DM_to_kpc

DEFAULT_MODEL_TYPE_FIT = {"polynomial": {"degree": 5}}

# Stateless -> shared across calls instead of re-instantiating on every generic_search_single_dm() call.
_PROJ = SimpleLonLatProj()


def generic_search_single_dm(
    data,
    match_filter_parameters,
    mag_cols=("mag_g", "mag_r"),
    use_dust_corrected=True,
    dust_correction_kwargs=None,
    ensure_dust_correction=True,
    cuts=None,
    flag_cuts=None,
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
    match filter, cuts, smoothed HEALPix map, polynomial background fit,
    residual.

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
        Extra kwargs forwarded to dust_correction.correct_dust_map (e.g.
        'ebv_map' or 'ebv_map_path') when the correction needs to be computed.
    ensure_dust_correction : bool
        If True (default) and `use_dust_corrected`, compute the missing
        '<col>_corrected' columns in place before resolving magnitude columns.
        Set to False when the correction has already been applied upfront
        (e.g. by generic_search_dm_scan) to skip the per-call existence check.
    cuts : dict, optional
        Selection applied on top of the match filter, as ``{column: spec}``
        (see catalog_cuts.get_cut_mask), e.g. ``{"mag_g_corrected": (None, 24.0)}``
        for a faint-end magnitude cut. Column names are used as is: name the
        '<col>_corrected' column to cut on a dust-corrected magnitude.
        Ignored if `flag_cuts` is provided. Default: no cuts.
    flag_cuts : array-like of bool, optional
        Precomputed mask of `cuts` (same length as `data`). If provided,
        the internal call to catalog_cuts.get_cut_mask is skipped
        -- useful when scanning several distance moduli, since the cuts
        do not depend on distance modulus and only need computing once.
    nside, smoothing, smoothing_scale_degrees : HEALPix map parameters.
    model_type_fit : dict
        Passed to background_polynomial_fit.fit_background_model (default: degree-5 polynomial).
    fit_method : str
        Passed to background_polynomial_fit.fit_background_model (default: "polyfit2d").
    footprint_bounds : tuple (ra_min, ra_max, dec_min, dec_max), optional
        If None, inferred from the match-filtered + cut selection.

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
        ensure_dust_corrected(
            data, mag_cols, ra_col, dec_col, dust_correction_kwargs, verbose=verbose
        )

    mag_to_use = resolve_mag_columns(data, mag_cols, use_dust_corrected)
    if verbose:
        print(f"Magnitudes used for cut and match filter: {mag_to_use}")

    # --- Match filter + cuts ---
    polygon_vertices = match_filter.build_match_filter(**match_filter_parameters)
    flag_match = match_filter.is_in_match_filter(
        data[mag_to_use[0]],
        data[mag_to_use[1]],
        polygon_vertices=polygon_vertices,
        verbose=verbose,
    )
    if flag_cuts is None:
        flag_cuts = get_cut_mask(data, cuts or {})
    selection_flag = flag_match & flag_cuts
    del flag_match, flag_cuts

    # Only pull the two coordinate columns for the selected rows, instead of a
    # full-row copy of (potentially very wide) `data` via `data.loc[mask]`.
    ra_values = data[ra_col].to_numpy()[selection_flag]
    dec_values = data[dec_col].to_numpy()[selection_flag]
    n_selected = int(ra_values.size)

    if verbose:
        print(f"Selected {n_selected}/{len(data)} stars in match filter + cuts")

    if footprint_bounds is None:
        ra_min = float(np.min(ra_values))
        ra_max = float(np.max(ra_values))
        dec_min = float(np.min(dec_values))
        dec_max = float(np.max(dec_values))
    else:
        ra_min, ra_max, dec_min, dec_max = footprint_bounds

    # --- Smoothed HEALPix map ---
    data_map, lon, lat, npix = star_density_healpix(
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
    params_fit, solver_res, prepared_fit = background_polynomial_fit.fit_background_model(
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
        "cuts": dict(cuts or {}),
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


def generic_search_dm_scan(
    data,
    distance_modulus_array,
    match_filter_parameters,
    find_stream_kwargs=None,
    verbose=True,
):
    """
    Run generic_search_single_dm() over a grid of distance moduli, one at a time.

    The dust correction and the (distance-modulus-independent) cuts
    are each computed once up front and reused for every distance modulus,
    instead of being recomputed on every generic_search_single_dm() call.

    The results are yielded one distance modulus at a time, so that the caller
    can plot or save each one and drop its HEALPix maps before the next one is
    computed: retaining every frame's maps would otherwise grow without bound
    when scanning many distance moduli over a large catalog (see
    workflows.run_generic_search.run_generic_search).

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
    verbose : bool

    Yields
    ------
    distance_modulus : float
    results : dict
        generic_search_single_dm() output for this distance modulus.
    """
    find_stream_kwargs = dict(find_stream_kwargs or {})

    # --- Precompute what doesn't depend on distance modulus, once, for the whole scan ---
    mag_cols = find_stream_kwargs.get("mag_cols", ("mag_g", "mag_r"))
    ra_col = find_stream_kwargs.get("ra_col", "coord_ra")
    dec_col = find_stream_kwargs.get("dec_col", "coord_dec")
    use_dust_corrected = find_stream_kwargs.get("use_dust_corrected", True)
    cuts = find_stream_kwargs.get("cuts")

    if use_dust_corrected:
        ensure_dust_corrected(
            data,
            mag_cols,
            ra_col,
            dec_col,
            find_stream_kwargs.get("dust_correction_kwargs"),
            verbose=verbose,
        )
    find_stream_kwargs["ensure_dust_correction"] = False

    find_stream_kwargs["flag_cuts"] = get_cut_mask(data, cuts or {})
    if verbose:
        print(f"Precomputed dust correction + cuts once for the scan (cuts: {cuts}).")

    for dm in distance_modulus_array:
        if verbose:
            print(f"Processing distance modulus = {dm:.2f}")

        mf_params = dict(match_filter_parameters)
        mf_params["distance_modulus"] = dm

        yield dm, generic_search_single_dm(data, mf_params, **find_stream_kwargs)

import logging
from types import SimpleNamespace

import healpy as hp
import numpy as np
from numpy.polynomial import polynomial
import pandas as pd

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - optional dependency at runtime
    minimize = None


LOGGER = logging.getLogger(__name__)

def fit_background_model(data, method="likelihood", model_type={"polynomial": {"degree": 5}}, **kwargs):
    """
    Fit background with either Poisson likelihood (default) or polyfit2d least-squares.
    """
    method_norm = str(method).lower()
    if method_norm in {"polyfit2d", "least-squares", "least_squares", "polyfit"}:
        return fit_background_model_polyfit2d(data, model_type=model_type, **kwargs)
    raise ValueError("Unknown method. Use 'likelihood' or 'polyfit2d'.")


############################################################################################################
################################# Fitting method as in DEs paper ###########################################
############################################################################################################
# https://stackoverflow.com/a/32297563/4075339


def polyfit2d(x, y, f, deg):
    """
    Fit a 2d polynomial.

    Parameters:
    -----------
    x : array of x values
    y : array of y values
    f : array of function return values
    deg : polynomial degree (length-2 list)

    Returns:
    --------
    c : polynomial coefficients
    """
    x = np.asarray(x)
    y = np.asarray(y)
    f = np.asarray(f)
    deg = np.asarray(deg)
    vander = polynomial.polyvander2d(x, y, deg)
    vander = vander.reshape((-1,vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c = np.linalg.lstsq(vander, f)[0]
    return c.reshape(deg+1)



def fit_background_model_polyfit2d(data, model_type={"polynomial": {"degree": 5}}, **kwargs):
    """
    Fit background with 2D polynomial least-squares using polyfit2d.
    
    Parameters
    ----------
    data : np.ndarray, np.ma.MaskedArray, pd.DataFrame, or dict
        Input sky data: HEALPix count map (1D array), DataFrame with 'ra'/'dec' columns,
        or masked array for sparse patches.
    model_type : dict
        Model configuration, e.g. {"polynomial": {"degree": 5}}.
    **kwargs
        proj : object
            Projection object with ang2xy(...) and get_extent() methods (required).
        nside : int
            HEALPix resolution (default: 64).
        nest : bool
            HEALPix ordering (default: False = RING).
        sigma : float
            Gaussian smoothing width in degrees (default: 0.0, no smoothing).
        percent : tuple/list or None
            Percentile clipping bounds, e.g. (2, 95) (default: None).
        mask : array-like of bool, optional
            Footprint mask for HEALPix maps. True values are excluded from fitting.
            Use when zero-count pixels are physical (e.g., deep observations) rather
            than indicating missing survey coverage.
        fit_log_space : bool
            Whether to fit log-transformed values (default: True).
        fit_log_eps : float
            Epsilon for log clipping (default: 1e-9).
    
    Returns
    -------
    params_fit : np.ndarray
        Fitted polynomial coefficients.
    solver_res : SimpleNamespace
        Result with: x, fval, success, message, nfev, nit, formula.
    prepared_poly : dict
        Data bundle with: background_map, model_type_used, coeff_2d, etc.
    """
    proj = kwargs.get("proj", None)
    sigma = float(kwargs.get("sigma", 0.0))
    percent = kwargs.get("percent", None)
    fit_log_space = bool(kwargs.get("fit_log_space", True))
    fit_log_eps = float(kwargs.get("fit_log_eps", 1e-9))
    kwargs_local = dict(kwargs)
    kwargs_local.pop("proj", None)
    kwargs_local.pop("sigma", None)
    kwargs_local.pop("percent", None)
    kwargs_local.pop("fit_log_space", None)
    kwargs_local.pop("fit_log_eps", None)

    prepared_poly = process_data_polyfit2d(data, proj=proj, sigma=sigma, percent=percent, **kwargs_local)

    cfg = _extract_polynomial_config(model_type)
    deg_x = int(cfg["deg_x"])
    deg_y = int(cfg["deg_y"])
    degree_mode = cfg["degree_mode"]
    total_degree = int(cfg["total_degree"])

    target_fit = np.asarray(prepared_poly["v_fit"], dtype=float)
    if fit_log_space:
        target_fit = np.log(np.clip(target_fit, fit_log_eps, None))

    # Defensive checks: ensure we have data points to fit and enough points
    n_fit_pts = int(np.asarray(prepared_poly.get("x_fit", [])).size)
    if n_fit_pts == 0:
        raise ValueError(
            "No valid projected pixels available for polyfit2d (after masking/projection). "
            "Check the footprint mask and projection extent passed to the fitter."
        )

    # Compute number of polynomial parameters for rectangular mode
    n_params_requested = (deg_x + 1) * (deg_y + 1)
    if n_fit_pts < n_params_requested:
        # Reduce polynomial degree to fit available data, keeping rectangular shape
        max_deg = max(int(np.floor(np.sqrt(n_fit_pts)) - 1), 0)
        if max_deg < min(deg_x, deg_y):
            LOGGER.warning(
                "Requested polynomial degrees (%d,%d) require %d params but only %d fit points available; reducing degree to %d.",
                deg_x,
                deg_y,
                n_params_requested,
                n_fit_pts,
                max_deg,
            )
        deg_x = deg_y = max_deg

    coeff_2d = polyfit2d(prepared_poly["x_fit"], prepared_poly["y_fit"], target_fit, [deg_x, deg_y])

    # Enforce triangular support for total-degree models.
    if degree_mode == "total":
        for i in range(coeff_2d.shape[0]):
            for j in range(coeff_2d.shape[1]):
                if i + j > total_degree:
                    coeff_2d[i, j] = 0.0

    model_plane_all = polynomial.polyval2d(prepared_poly["x_all"], prepared_poly["y_all"], coeff_2d)
    if fit_log_space:
        #bkg = np.exp(model_plane_all)
        # fix the fact that exp(710) is above float range
        model_plane_all_clipped = np.clip(model_plane_all, -700, 700)
        bkg = np.exp(model_plane_all_clipped)
    else:
        bkg = model_plane_all

    bkg = np.ma.array(bkg, mask=prepared_poly["mask"], fill_value=np.nan)

    model_plane_fit = polynomial.polyval2d(prepared_poly["x_fit"], prepared_poly["y_fit"], coeff_2d)
    if fit_log_space:
        pred_fit = np.exp(model_plane_fit)
    else:
        pred_fit = model_plane_fit
    sse = float(np.sum((prepared_poly["v_fit"] - pred_fit) ** 2))

    exponents = _polynomial_exponents(model_type)
    params_fit = np.asarray([coeff_2d[i, j] for i, j in exponents], dtype=float)
    solver_res = SimpleNamespace(
        x=params_fit,
        fval=sse,
        success=True,
        message="least-squares fit completed",
        nfev=1,
        nit=1,
        formula=polynomial_formula(params_fit, model_type),
    )

    prepared_poly["background_map"] = bkg
    prepared_poly["model_type_used"] = model_type
    prepared_poly["fit_log_space"] = fit_log_space
    prepared_poly["coeff_2d"] = coeff_2d
    return params_fit, solver_res, prepared_poly



############################################################################################################
############################## Data preparation for polyfit2d #######################################
############################################################################################################


def process_data(data, **kwargs):
    """
    Convert input data to a prepared data bundle for fitting.
    
    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        If np.ndarray (1D): treated as HEALPix count map; NaN and hp.UNSEEN pixels are masked.
        If pd.DataFrame: must contain 'ra' and 'dec' columns (degrees).
    **kwargs
        nside (int): HEALPix NSIDE (default 64).
        nest (bool): HEALPix ordering scheme (default False = RING).
        mask (array-like of bool, optional): footprint mask for HEALPix maps.
    
    Returns
    -------
    dict with keys:
        - data_map: full HEALPix map (npix,)
        - selection: boolean mask of valid pixels used for fitting (npix,)
        - x, y: normalized coordinates [-1,+1] of selected pixels
        - observed: observed counts at selected pixels
        - nside, npix: HEALPix metadata
    """
    nside = int(kwargs.get("nside", 64))
    nest = bool(kwargs.get("nest", False))
    user_mask = kwargs.get("mask", None)
    _validate_nside(nside)
    npix = hp.nside2npix(nside)

    def _coerce_mask(mask, shape):
        if mask is None:
            return np.zeros(shape, dtype=bool)
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape == ():
            mask_arr = np.full(shape, bool(mask_arr), dtype=bool)
        if mask_arr.shape != shape:
            raise ValueError(f"mask has shape {mask_arr.shape}, expected {shape}.")
        return mask_arr

    if isinstance(data, np.ma.MaskedArray):
        data_map = np.asarray(data.filled(0.0), dtype=float)
        mask = np.ma.getmaskarray(data)
    elif isinstance(data, np.ndarray) and data.ndim == 1:
        data_map = np.asarray(data, dtype=float)
        mask = np.zeros_like(data_map, dtype=bool)
    elif isinstance(data, dict):
        data = pd.DataFrame(data)
    if isinstance(data, pd.DataFrame):
        required_cols = ["ra", "dec"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        ra = data["ra"].to_numpy(dtype=float)
        dec = data["dec"].to_numpy(dtype=float)
        valid = np.isfinite(ra) & np.isfinite(dec)
        pix = hp.ang2pix(nside, ra[valid], dec[valid], lonlat=True, nest=nest)
        data_map = np.zeros(npix, dtype=float)
        np.add.at(data_map, pix, 1.0)
        mask = np.zeros(npix, dtype=bool)
    else:
        if not (isinstance(data, np.ndarray) or isinstance(data, np.ma.MaskedArray)):
            raise ValueError("Data format not recognized. Provide a HEALPix map or a DataFrame with 'ra'/'dec'.")

    if user_mask is not None:
        mask = mask | _coerce_mask(user_mask, data_map.shape)

    # Unobserved pixels of plain HEALPix maps are stored as NaN or hp.UNSEEN.
    # hp.mask_bad matches UNSEEN with a tolerance, so float32 maps are caught too.
    mask = mask | ~np.isfinite(data_map) | hp.mask_bad(data_map)

    if data_map.size != npix:
        raise ValueError(f"Input map size={data_map.size} is inconsistent with nside={nside} (expected npix={npix}).")

    lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True, nest=nest)
    x = _normalize_to_minus_one_plus_one(lon)
    y = _normalize_to_minus_one_plus_one(lat)

    selection = (~mask) & np.isfinite(x) & np.isfinite(y)
    fit_idx = np.where(selection)[0] # Indices of pixels used for fitting (unmasked and valid coordinates)
    # We will evaluate the likelihood only on these selected pixels to avoid
    # issues with masked/invalid data. This is better than selection mask, because it gives us the actual indices
    #  to index into the full data_map when computing expected counts, and avoid
    #  some jax issues with boolean indexing.

    return {
        "data_map": data_map,
        "mask": mask,
        "selection": selection,
        "fit_idx": fit_idx,
        "x": x[selection],
        "y": y[selection],
        "observed": np.clip(data_map[selection], 0.0, None),
        "nside": nside,
        "npix": npix,
    }



def process_data_polyfit2d(data, proj, sigma=0.0, percent=None, debug=False, **kwargs):
    """
    Prepare projected coordinates and values for polyfit2d from a HEALPix map.

    Parameters
    ----------
    data : np.ndarray, np.ma.MaskedArray, pd.DataFrame, or dict
        Input sky data accepted by process_data.
    proj : object
        Projection object with ang2xy(...) and optionally get_extent().
    sigma : float, default=0.0
        Gaussian smoothing width in degrees (0 disables smoothing).
    percent : tuple/list or None
        Optional percentile clipping, e.g. (2, 95).
    mask : array-like of bool, optional
        Optional footprint mask for HEALPix maps. True values are excluded from
        the fit. This is useful when a sparse map stores unobserved pixels as 0.

    Returns
    -------
    dict
        Prepared projected data for least-squares polynomial fitting.
    """
    if proj is None or not hasattr(proj, "ang2xy"):
        raise ValueError("polyfit2d method requires `proj` with an `ang2xy` method.")

    prepared = process_data(data, **kwargs)
    nside = prepared["nside"]
    npix = prepared["npix"]
    mask = prepared["mask"]

    data_ma = np.ma.array(prepared["data_map"], mask=mask)

    if percent is not None:
        p_lo, p_hi = percent
        vmin, vmax = np.percentile(data_ma.compressed(), [p_lo, p_hi])
        fill_med = float(np.ma.median(data_ma))
        clipped = np.clip(data_ma.filled(fill_med), vmin, vmax)
        data_ma = np.ma.array(clipped, mask=mask)

    if sigma is not None and float(sigma) > 0:
        fill_med = float(np.ma.median(data_ma))
        smoothed = hp.smoothing(data_ma.filled(fill_med), sigma=np.radians(float(sigma)))
        data_ma = np.ma.array(smoothed, mask=mask)

    lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    # Prefer the selection computed by process_data (handles mask and finite lon/lat).
    sel = prepared.get("selection", None)
    if sel is None:
        sel = ~data_ma.mask

    x_sel, y_sel = proj.ang2xy(lon[sel], lat[sel], lonlat=True)
    x_sel = np.asarray(x_sel, dtype=float)
    y_sel = np.asarray(y_sel, dtype=float)
    v_sel = np.asarray(data_ma[sel], dtype=float)

    sel2 = np.isfinite(x_sel) & np.isfinite(y_sel) & np.isfinite(v_sel)
    extent_count = None
    if hasattr(proj, "get_extent"):
        xmin, xmax, ymin, ymax = proj.get_extent()
        # Use inclusive bounds to avoid dropping pixels that lie exactly on the edge.
        extent_mask = (x_sel >= xmin) & (x_sel <= xmax) & (y_sel >= ymin) & (y_sel <= ymax)
        extent_count = int(np.count_nonzero(extent_mask))
        sel2 &= extent_mask

    if debug:
        print("[process_data_polyfit2d debug] sel_count:", int(np.count_nonzero(sel)))
        print("[process_data_polyfit2d debug] proj finite x/y:", int(np.count_nonzero(np.isfinite(x_sel))), int(np.count_nonzero(np.isfinite(y_sel))))
        print("[process_data_polyfit2d debug] v_sel finite:", int(np.count_nonzero(np.isfinite(v_sel))))
        if extent_count is not None:
            print("[process_data_polyfit2d debug] extent_count:", extent_count)
        print("[process_data_polyfit2d debug] sel2_count:", int(np.count_nonzero(sel2)))

    x_fit = x_sel[sel2]
    y_fit = y_sel[sel2]
    v_fit = v_sel[sel2]

    x_all, y_all = proj.ang2xy(lon, lat, lonlat=True)
    x_all = np.asarray(x_all, dtype=float)
    y_all = np.asarray(y_all, dtype=float)

    # Debug logging to help trace empty-fit issues
    try:
        LOGGER.debug("process_data_polyfit2d: sel_count=%d, sel2_count=%d, mask_sum=%d", np.count_nonzero(sel), np.count_nonzero(sel2), np.sum(data_ma.mask))
    except Exception:
        pass

    # Normalize projected coordinates for numerical stability of high-order fits.
    finite_x_all = np.isfinite(x_all)
    finite_y_all = np.isfinite(y_all)

    if np.any(finite_x_all):
        x_min, x_max = float(np.nanmin(x_all[finite_x_all])), float(np.nanmax(x_all[finite_x_all]))
    else:
        x_min, x_max = -1.0, 1.0
    if np.any(finite_y_all):
        y_min, y_max = float(np.nanmin(y_all[finite_y_all])), float(np.nanmax(y_all[finite_y_all]))
    else:
        y_min, y_max = -1.0, 1.0

    x_span = x_max - x_min
    y_span = y_max - y_min

    if x_span < 1e-12:
        x_all_norm = np.zeros_like(x_all)
        x_fit_norm = np.zeros_like(x_fit)
    else:
        x_all_norm = 2.0 * (x_all - x_min) / x_span - 1.0
        x_fit_norm = 2.0 * (x_fit - x_min) / x_span - 1.0

    if y_span < 1e-12:
        y_all_norm = np.zeros_like(y_all)
        y_fit_norm = np.zeros_like(y_fit)
    else:
        y_all_norm = 2.0 * (y_all - y_min) / y_span - 1.0
        y_fit_norm = 2.0 * (y_fit - y_min) / y_span - 1.0

    return {
        "nside": nside,
        "npix": npix,
        "mask": np.asarray(data_ma.mask, dtype=bool),
        "data_map": np.asarray(data_ma.filled(0.0), dtype=float),
        "x_fit": x_fit_norm,
        "y_fit": y_fit_norm,
        "v_fit": v_fit,
        "x_all": x_all_norm,
        "y_all": y_all_norm,
        "data_ma": data_ma,
    }




############################################################################################################
############################## Utils #######################################
############################################################################################################

def _normalize_to_minus_one_plus_one(arr):
    """Normalize array to [-1, +1] range."""
    arr = np.asarray(arr, dtype=float)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    if np.isclose(amax, amin):
        return np.zeros_like(arr)
    return 2.0 * (arr - amin) / (amax - amin) - 1.0


def _validate_nside(nside):
    """Check that nside is a valid HEALPix resolution (power of 2)."""
    if not hp.isnsideok(nside):
        raise ValueError(f"Invalid nside={nside}. nside must be a power of 2.")



def _extract_polynomial_config(model_type):
    """Extract and validate polynomial degree configuration from model_type dict."""
    poly_cfg = model_type.get("polynomial", {})

    if "degree" in poly_cfg:
        degree = int(poly_cfg["degree"])
        deg_x = degree
        deg_y = degree
        total_degree = degree
    else:
        deg_x = int(poly_cfg.get("deg_x", 5))
        deg_y = int(poly_cfg.get("deg_y", 5))
        total_degree = int(poly_cfg.get("total_degree", min(deg_x, deg_y)))

    degree_mode = poly_cfg.get("degree_mode", "rectangular")
    if degree_mode not in {"total", "rectangular"}:
        raise ValueError("degree_mode must be 'total' or 'rectangular'.")

    if deg_x < 0 or deg_y < 0 or total_degree < 0:
        raise ValueError("Polynomial degrees must be non-negative.")

    return {
        "deg_x": deg_x,
        "deg_y": deg_y,
        "total_degree": total_degree,
        "degree_mode": degree_mode,
    }


def _polynomial_exponents(model_type):
    """Return list of (i, j) exponent pairs for the polynomial basis."""
    cfg = _extract_polynomial_config(model_type)
    deg_x = cfg["deg_x"]
    deg_y = cfg["deg_y"]
    degree_mode = cfg["degree_mode"]
    total_degree = cfg["total_degree"]

    exponents = []
    for i in range(deg_x + 1):
        for j in range(deg_y + 1):
            if degree_mode == "total" and (i + j) > total_degree:
                continue
            exponents.append((i, j))
    return exponents



def polynomial_formula(params, model_type, precision=5):
    """Return a human-readable string of the fitted 2D polynomial."""
    exponents = _polynomial_exponents(model_type)
    coeffs = np.asarray(params, dtype=float)

    terms = []
    for coeff, (i, j) in zip(coeffs, exponents):
        if abs(coeff) < 10 ** (-precision):
            continue

        cstr = f"{coeff:.{precision}g}"
        if i == 0 and j == 0:
            terms.append(cstr)
            continue

        var_terms = []
        if i == 1:
            var_terms.append("x")
        elif i > 1:
            var_terms.append(f"x^{i}")
        if j == 1:
            var_terms.append("y")
        elif j > 1:
            var_terms.append(f"y^{j}")

        terms.append(f"{cstr}*{'*'.join(var_terms)}")

    if not terms:
        return "0"
    return " + ".join(terms)
import os

import healpy as hp
import numpy as np
import pandas as pd
import skyproj


def apply_magnitude_cut(data, mag_col="mag_g", mag_bounds=(None, None)):
    """
    Return a boolean mask selecting rows within magnitude bounds.

    Parameters
    ----------
    data : pandas.DataFrame or astropy.table.Table
    mag_col : str
        Name of magnitude column
    mag_bounds : tuple (min, max)
        Bounds on magnitude. Use None for open bound.

    Returns
    -------
    mask : array-like (bool)
        True for rows داخل bounds
    """
    mag = data[mag_col]

    mask = np.ones(len(mag), dtype=bool)

    if mag_bounds[0] is not None:
        mask &= mag >= mag_bounds[0]

    if mag_bounds[1] is not None:
        mask &= mag <= mag_bounds[1]

    return mask


def get_healpy_data_from_table(
    data,
    columns=("ra", "dec"),
    nside=512,
    smoothing=False,
    smoothing_scale_degrees=0.3,
    footprint_mask=None,
):
    """
    Build HEALPix map from catalog with optional smoothing and footprint masking.

    Unobserved pixels are masked to prevent smoothing from bleeding values into the
    unobserved footprint. Returns a masked array where masked pixels indicate regions
    outside the survey footprint.

    Parameters
    ----------
    data : table
        Catalog with coordinate columns
    columns : tuple of str
        Column names for ra/dec (default: ("ra", "dec"))
    nside : int
        HEALPix resolution (default: 512)
    smoothing : bool
        Whether to apply Gaussian smoothing (default: False)
    smoothing_scale_degrees : float
        Gaussian smoothing width in degrees (default: 0.3)
    footprint_mask : array-like of bool, optional
        Pre-computed footprint mask where True = unobserved (masked), False = observed.
        If None (default), unobserved pixels are auto-detected as those with count=0.
        Provide this when zero-count pixels are physical (e.g., from deep observations)
        rather than indicating missing data.

    Returns
    -------
    hp_map : np.ma.MaskedArray
        HEALPix count map with unobserved pixels masked
    lon, lat : np.ndarray
        Longitude and latitude of HEALPix pixel centers (degrees)
    npix : int
        Total number of HEALPix pixels at this resolution
    """
    # Read coordinates and keep only valid values
    ra = np.asarray(data[columns[0]], dtype=float)
    dec = np.asarray(data[columns[1]], dtype=float)
    valid = np.isfinite(ra) & np.isfinite(dec) & (dec >= -90.0) & (dec <= 90.0)

    ra = ra[valid]
    dec = dec[valid]

    # Build HEALPix count map
    npix = hp.nside2npix(nside)
    pix_idx = hp.ang2pix(nside, ra, dec, lonlat=True, nest=False)
    hp_map = np.bincount(pix_idx, minlength=npix).astype(np.float32)
    pix = np.arange(npix)
    lon, lat = hp.pix2ang(nside, pix, lonlat=True)

    # Determine footprint mask
    if footprint_mask is None:
        # Auto-detect: pixels that received no observations
        footprint_mask = hp_map == 0
    else:
        # Use provided mask; coerce to boolean array
        footprint_mask = np.asarray(footprint_mask, dtype=bool)
        if footprint_mask.shape != (npix,):
            raise ValueError(
                f"footprint_mask must have shape ({npix},), got {footprint_mask.shape}"
            )

    # Apply smoothing if requested
    if smoothing:
        smoothing_scale_radian = 2 * np.pi / 360.0 * smoothing_scale_degrees
        # To smooth safely: fill masked (unobserved) pixels with a benign value before smoothing,
        # then restore the mask afterwards. healpy.smoothing does not handle NaN gracefully.
        hp_map_temp = hp_map.copy().astype(float)

        # Fill unobserved pixels with 0 (not NaN) before smoothing
        hp_map_temp[footprint_mask] = 0.0

        # Apply smoothing on the filled array
        hp_map_smooth = hp.sphtfunc.smoothing(
            hp_map_temp, sigma=np.radians(smoothing_scale_degrees)
        )

        # Return as masked array with unobserved region masked
        hp_map = np.ma.array(hp_map_smooth, mask=footprint_mask, fill_value=np.nan)
    else:
        # Return as masked array even without smoothing for consistency
        hp_map = np.ma.array(hp_map, mask=footprint_mask)

    return hp_map, lon, lat, npix

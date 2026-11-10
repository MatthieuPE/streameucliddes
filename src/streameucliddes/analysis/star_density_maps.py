"""Star density maps: star counts on a grid, smoothed and masked.

On the sky (HEALPix) or in a stream frame (phi1/phi2). A phi1/phi2 grid is
given by a window ``(phi1_min, phi1_max, phi2_min, phi2_max)`` and a bin size
(deg), with ``round((max - min) / bin_size)`` bins along each axis. Its maps
have shape (n_phi1, n_phi2), row = phi1 bin (as ``numpy.histogram2d``):
transpose them for ``imshow``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def star_density_healpix(
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


def star_density_phi12(
    phi1: ArrayLike,
    phi2: ArrayLike,
    window: tuple[float, float, float, float],
    bin_size_deg: float,
    *,
    smoothing_deg: float = 0.0,
    footprint_mask: NDArray[np.bool_] | None = None,
) -> np.ma.MaskedArray:
    """Smoothed star counts on a phi1/phi2 grid.

    Counts the stars in each bin, then smooths the counts with a Gaussian of
    standard deviation ``smoothing_deg``. With a footprint, the smoothed
    counts are divided by the smoothed footprint (the fraction of each bin's
    neighbourhood inside the footprint), so that the density does not drop
    at the footprint edges. Bins outside the footprint are masked.

    Parameters
    ----------
    phi1, phi2 : array_like
        Stars in the stream frame (deg).
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` of the grid (deg).
    bin_size_deg : float
        Bin size of the grid (deg), the same in phi1 and phi2.
    smoothing_deg : float, optional
        Standard deviation of the Gaussian (deg). 0: no smoothing.
    footprint_mask : numpy.ndarray of bool, optional
        From ``catalogs.survey_footprint.survey_footprint_phi12_mask``, True
        outside the footprint.

    Returns
    -------
    density : numpy.ma.MaskedArray, shape (n_phi1, n_phi2)
        Smoothed counts per bin, masked outside the footprint.

    Notes
    -----
    Code to reuse: ``_compute_spline_phi12`` in celestial_currents
    ``notebooks/candidate_investigations/full_search/template.ipynb``
    (``numpy.histogram2d``, then ``scipy.ndimage.gaussian_filter`` with a
    sigma of ``smoothing_deg / bin_size_deg`` bins).
    """
    raise NotImplementedError

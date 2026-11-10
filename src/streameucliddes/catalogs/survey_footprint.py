"""Survey footprint: HEALPix mask, and its projection on a phi1/phi2 grid.

As in ``analysis.star_density_maps.star_density_healpix`` and
``analysis.background_polynomial_fit.process_data``, a mask is True outside
the footprint (masked). phi1/phi2 grids follow the convention of
``analysis.star_density_maps``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray


def survey_footprint_healpix_mask(
    ra: ArrayLike, dec: ArrayLike, nside: int, *, nest: bool, min_count: int = 1
) -> NDArray[np.bool_]:
    """HEALPix mask of the survey footprint, from the star counts.

    A pixel is in the footprint when it holds at least ``min_count`` stars.
    To replace by the DES Y6 and Euclid DR1 coverage maps when they are
    available.

    Parameters
    ----------
    ra, dec : array_like
        Stars of the survey catalog (deg).
    nside : int
        HEALPix resolution of the mask.
    nest : bool
        HEALPix ordering of the mask. No default: always explicit.
    min_count : int, optional
        Minimum number of stars of a pixel in the footprint.

    Returns
    -------
    mask : numpy.ndarray of bool, shape (12 * nside**2,)
        True outside the footprint.

    Notes
    -----
    Code to reuse: the footprint from the counts in
    ``analysis.star_density_maps.star_density_healpix`` (pixels without stars
    are masked), and the mask of
    ``analysis.background_polynomial_fit.process_data``.
    """
    raise NotImplementedError


def survey_footprint_phi12_mask(
    footprint_mask: NDArray[np.bool_],
    rotation_matrix: NDArray[np.float64],
    window: tuple[float, float, float, float],
    bin_size_deg: float,
    *,
    nest: bool,
) -> NDArray[np.bool_]:
    """Survey footprint on a phi1/phi2 grid.

    A bin is outside the footprint when its centre falls in a masked HEALPix
    pixel. Used to mask the known-stream maps, and to correct their edges:
    the smoothed counts are divided by the smoothed footprint (see
    ``analysis.star_density_maps.star_density_phi12``).

    Parameters
    ----------
    footprint_mask : numpy.ndarray of bool
        HEALPix mask from ``survey_footprint_healpix_mask``, True outside.
        nside is taken from its length.
    rotation_matrix : numpy.ndarray, shape (3, 3)
        Stream frame (see ``stream_frames``).
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` of the grid (deg).
    bin_size_deg : float
        Bin size of the grid (deg), the same in phi1 and phi2.
    nest : bool
        HEALPix ordering of ``footprint_mask``.

    Returns
    -------
    mask : numpy.ndarray of bool, shape (n_phi1, n_phi2)
        True outside the footprint. Row = phi1 bin, as the maps of
        ``analysis.star_density_maps.star_density_phi12``.

    Notes
    -----
    Code to reuse: celestial_currents
    ``visualization.py::transform_healpix_to_custom_frame`` (bin centres to
    ICRS, then ``healpy.ang2pix``), with ``stream_frames.phi12_to_radec``.
    """
    raise NotImplementedError

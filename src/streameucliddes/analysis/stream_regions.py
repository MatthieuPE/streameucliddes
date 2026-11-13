"""Width, track and on/off regions of a stream candidate, in its frame.

The track is straight for now: phi2 = ``phi2_centre`` over ``phi1_range``.
Its width is the sigma of a Gaussian plus a constant background, fitted to
the phi2 density profile of the stars (e.g. in the match filter). The on
region is ``|phi2 - phi2_centre| <= on_n_sigma * sigma``; the off region is
two bands, one on each side, ``off_n_sigma[0] * sigma`` to
``off_n_sigma[1] * sigma`` from ``phi2_centre``; both over ``phi1_range``.

Regions are plain dicts (``stream_regions``), to edit by hand in a notebook.

Areas and densities are computed outside an excluded HEALPix mask (RING):
the footprint OR the masked objects (``analysis.build_masks.excluded_healpix_mask``),
so that the stars, the maps and the areas see the same masks. Every area
comes from ``unmasked_area_deg2``.

References
----------
Shipp et al. 2018, ApJ 862, 114 (Gaussian width with a constant background;
on - off Hess diagrams). On band 1.4 sigma, off bands 2.5 to 4.5 sigma: user
choices, no reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    import astropy.coordinates as coord
    from numpy.typing import ArrayLike, NDArray

DEFAULT_ON_N_SIGMA = 1.4  # user choice
DEFAULT_OFF_N_SIGMA = (2.5, 4.5)  # user choice
DEFAULT_PHI2_BIN_DEG = 0.05
DEFAULT_PHI1_BIN_DEG = 0.5
DEFAULT_SIGMA_GUESS_DEG = 0.2  # initial sigma of the width fit
DEFAULT_AREA_STEP_DEG = 0.01  # fine grid of the areas
DEFAULT_TRACK_POINTS = 100


# ===================================================
# Regions
# ===================================================


def stream_regions(
    phi1_range: tuple[float, float],
    phi2_centre: float = 0.0,
    sigma: float = DEFAULT_SIGMA_GUESS_DEG,
    *,
    on_n_sigma: float = DEFAULT_ON_N_SIGMA,
    off_n_sigma: tuple[float, float] = DEFAULT_OFF_N_SIGMA,
) -> dict[str, Any]:
    """On and off regions of a stream, as a plain dict.

    Parameters
    ----------
    phi1_range : tuple of float
        ``(phi1_min, phi1_max)`` of the regions (deg).
    phi2_centre : float, optional
        phi2 of the track (deg).
    sigma : float, optional
        Gaussian width of the stream in phi2 (deg), e.g. from ``fit_stream_width``.
    on_n_sigma : float, optional
        Half-width of the on region, in units of ``sigma``.
    off_n_sigma : tuple of float, optional
        Inner and outer distances of the off bands to ``phi2_centre``, in
        units of ``sigma``.

    Returns
    -------
    regions : dict
        ``{"phi1_range", "phi2_centre", "sigma", "on_n_sigma", "off_n_sigma"}``.

    Raises
    ------
    ValueError
        ``off_n_sigma[0] < on_n_sigma`` (the regions overlap), or
        ``off_n_sigma[1] <= off_n_sigma[0]``, or ``sigma <= 0``.
    """
    raise NotImplementedError


def on_stream_mask(phi1: ArrayLike, phi2: ArrayLike, regions: dict[str, Any]) -> NDArray[np.bool_]:
    """Stars in the on region.

    Parameters
    ----------
    phi1, phi2 : array_like
        Coordinates of the stars in the stream frame (deg).
    regions : dict
        From ``stream_regions``.

    Returns
    -------
    mask : numpy.ndarray of bool
        True for ``|phi2 - phi2_centre| <= on_n_sigma * sigma`` and phi1 in
        ``phi1_range``.
    """
    raise NotImplementedError


def off_stream_mask(phi1: ArrayLike, phi2: ArrayLike, regions: dict[str, Any]) -> NDArray[np.bool_]:
    """Stars in the off region: the two bands, one on each side of the track.

    Parameters
    ----------
    phi1, phi2 : array_like
        Coordinates of the stars in the stream frame (deg).
    regions : dict
        From ``stream_regions``.

    Returns
    -------
    mask : numpy.ndarray of bool
        True for ``off_n_sigma[0] * sigma <= |phi2 - phi2_centre| <=
        off_n_sigma[1] * sigma`` and phi1 in ``phi1_range``.
    """
    raise NotImplementedError


# ===================================================
# Areas
# ===================================================


def unmasked_area_deg2(
    stream_frame: coord.BaseCoordinateFrame,
    phi1_edges: ArrayLike,
    phi2_edges: ArrayLike,
    excluded_healpix_mask: NDArray[np.bool_] | None,
    *,
    step_deg: float = DEFAULT_AREA_STEP_DEG,
) -> NDArray[np.float64]:
    """Area of each phi1/phi2 cell outside an excluded HEALPix mask.

    Each cell is sampled on a fine grid of ``step_deg``; each sample weighs
    its solid angle (``cos(phi2)`` weights), is converted to (ra, dec)
    (``stream_frames.phi12_to_radec``) and to its HEALPix pixel (RING, nside
    from the size of the mask), and is kept when the pixel is not excluded.

    Parameters
    ----------
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Frame of the cells.
    phi1_edges, phi2_edges : array_like
        Edges of the cells along each axis (deg), increasing.
    excluded_healpix_mask : numpy.ndarray of bool or None
        True where excluded (outside the footprint or on a masked object),
        RING. None: nothing excluded.
    step_deg : float, optional
        Step of the fine grid (deg).

    Returns
    -------
    area_deg2 : numpy.ndarray
        Shape (len(phi1_edges) - 1, len(phi2_edges) - 1) (deg²).
    """
    raise NotImplementedError


def region_area_deg2(
    regions: dict[str, Any],
    stream_frame: coord.BaseCoordinateFrame,
    excluded_healpix_mask: NDArray[np.bool_] | None,
    *,
    step_deg: float = DEFAULT_AREA_STEP_DEG,
) -> dict[str, float]:
    """Areas of the on and off regions outside an excluded HEALPix mask.

    ``unmasked_area_deg2`` on the edges of the on region and of the two off
    bands.

    Parameters
    ----------
    regions : dict
        From ``stream_regions``.
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Frame of the regions.
    excluded_healpix_mask : numpy.ndarray of bool or None
        See ``unmasked_area_deg2``.
    step_deg : float, optional
        Step of the fine grid (deg).

    Returns
    -------
    areas : dict
        ``{"on", "off"}`` (deg²); ``"off"`` is the sum of the two bands.
    """
    raise NotImplementedError


# ===================================================
# Profiles and width
# ===================================================


def phi2_density_profile(
    phi1: ArrayLike,
    phi2: ArrayLike,
    phi1_range: tuple[float, float],
    stream_frame: coord.BaseCoordinateFrame,
    excluded_healpix_mask: NDArray[np.bool_] | None,
    *,
    phi2_range: tuple[float, float],
    bin_deg: float = DEFAULT_PHI2_BIN_DEG,
    step_deg: float = DEFAULT_AREA_STEP_DEG,
) -> dict[str, Any]:
    """Density of stars in phi2 bins, over a phi1 range.

    Counts of the stars (e.g. those in the match filter) with phi1 in
    ``phi1_range``, in phi2 bins of ``bin_deg``, divided by the area of each
    bin outside the excluded mask (``unmasked_area_deg2``): tile edges and
    masked objects do not make dips.

    Parameters
    ----------
    phi1, phi2 : array_like
        Coordinates of the stars in the stream frame (deg).
    phi1_range : tuple of float
        ``(phi1_min, phi1_max)`` (deg).
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Frame of the stars.
    excluded_healpix_mask : numpy.ndarray of bool or None
        See ``unmasked_area_deg2``.
    phi2_range : tuple of float
        ``(phi2_min, phi2_max)`` of the bins (deg), e.g. that of the window.
    bin_deg : float, optional
        Bin size in phi2 (deg).
    step_deg : float, optional
        Step of the fine grid of the areas (deg).

    Returns
    -------
    profile : dict
        ``{"phi2_centres", "counts", "area_deg2", "density", "error",
        "bin_deg", "phi1_range"}``: density in stars/deg², error
        ``sqrt(counts) / area_deg2``. Bins with no unmasked area have NaN
        density and error.
    """
    raise NotImplementedError


def fit_stream_width(
    profile: dict[str, Any],
    *,
    phi2_centre_guess: float = 0.0,
    sigma_guess: float = DEFAULT_SIGMA_GUESS_DEG,
) -> dict[str, Any]:
    """Gaussian plus a constant, fitted to a phi2 density profile.

    ``scipy.optimize.curve_fit`` of ``amplitude * exp(-(phi2 - phi2_centre)²
    / (2 sigma²)) + background`` on the bins with a finite density, with the
    Poisson errors of the profile (empty bins: the error of one star).

    Parameters
    ----------
    profile : dict
        From ``phi2_density_profile``.
    phi2_centre_guess, sigma_guess : float, optional
        Initial values (deg).

    Returns
    -------
    fit : dict
        ``{"phi2_centre", "sigma", "amplitude", "background", "errors",
        "model"}``: ``errors`` a dict of the 1-sigma errors of the four
        parameters; ``model`` the fitted density at
        ``profile["phi2_centres"]``. Densities in stars/deg², angles in deg.

    Raises
    ------
    RuntimeError
        The fit does not converge.

    References
    ----------
    Shipp et al. 2018, ApJ 862, 114.
    """
    raise NotImplementedError


def phi1_density_profile(
    phi1: ArrayLike,
    phi2: ArrayLike,
    regions: dict[str, Any],
    stream_frame: coord.BaseCoordinateFrame,
    excluded_healpix_mask: NDArray[np.bool_] | None,
    *,
    phi1_range: tuple[float, float] | None = None,
    bin_deg: float = DEFAULT_PHI1_BIN_DEG,
    step_deg: float = DEFAULT_AREA_STEP_DEG,
) -> dict[str, Any]:
    """On - off density along phi1, to see where the stream starts and stops.

    In each phi1 bin, the stars of the on region and of the off bands
    (``on_stream_mask``, ``off_stream_mask``, phi2 limits only), each divided
    by its own area outside the excluded mask (``unmasked_area_deg2``).

    Parameters
    ----------
    phi1, phi2 : array_like
        Coordinates of the stars in the stream frame (deg).
    regions : dict
        From ``stream_regions``; its phi2 limits are used, not its
        ``phi1_range``.
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Frame of the stars.
    excluded_healpix_mask : numpy.ndarray of bool or None
        See ``unmasked_area_deg2``.
    phi1_range : tuple of float, optional
        Range of the bins (deg), e.g. that of the window. Default:
        ``regions["phi1_range"]``.
    bin_deg : float, optional
        Bin size in phi1 (deg).
    step_deg : float, optional
        Step of the fine grid of the areas (deg).

    Returns
    -------
    profile : dict
        ``{"phi1_centres", "on", "off", "difference", "error", "bin_deg"}``:
        densities in stars/deg²; ``error`` that of the difference (Poisson,
        in quadrature).
    """
    raise NotImplementedError


# ===================================================
# Track
# ===================================================


def stream_track_phi12(
    regions: dict[str, Any], n_points: int = DEFAULT_TRACK_POINTS
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(phi1, phi2) of the straight track: phi2 = ``phi2_centre`` over ``phi1_range``.

    Parameters
    ----------
    regions : dict
        From ``stream_regions``.
    n_points : int, optional
        Number of points.

    Returns
    -------
    phi1, phi2 : numpy.ndarray
        (deg), shape (n_points,).
    """
    raise NotImplementedError


def stream_track_radec(
    regions: dict[str, Any], stream_frame: coord.BaseCoordinateFrame, n_points: int = DEFAULT_TRACK_POINTS
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(ra, dec) of the straight track, for sky maps.

    ``stream_track_phi12`` converted with ``stream_frames.phi12_to_radec``.

    Parameters
    ----------
    regions : dict
        From ``stream_regions``.
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Frame of the regions.
    n_points : int, optional
        Number of points.

    Returns
    -------
    ra, dec : numpy.ndarray
        (deg), shape (n_points,).
    """
    raise NotImplementedError

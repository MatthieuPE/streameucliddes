"""Fits of a stream in its frame: for now, its width in phi2.

``fit_stream_width`` fits a Gaussian plus a polynomial background to a phi2
density profile (``analysis.candidate_investigation.phi12_density_profile``);
``stream_width_model`` gives the fitted density, e.g. on a 2D grid. The
profiles, the regions and the investigation stay in
``analysis.candidate_investigation``, which calls these fits.

References
----------
Shipp et al. 2018, ApJ 862, 114 (Gaussian width with a constant background).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.polynomial import polynomial
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

DEFAULT_SIGMA_GUESS_DEG = 0.2  # initial sigma of the width fit
DEFAULT_FIT_HALF_WIDTH_DEG = 2.0  # width fit on phi2_centre_guess +- this: the field density is not constant further
DEFAULT_BACKGROUND_DEGREE = 0  # background of the width fit: a constant (1: linear)


def fit_stream_width(
    profile: dict[str, Any],
    *,
    phi2_centre_guess: float = 0.0,
    sigma_guess: float = DEFAULT_SIGMA_GUESS_DEG,
    phi2_fit_range: tuple[float, float] | None = None,
    background_degree: int = DEFAULT_BACKGROUND_DEGREE,
) -> dict[str, Any]:
    """Gaussian plus a polynomial background, fitted to a phi2 density profile near the track.

    ``scipy.optimize.curve_fit`` of ``amplitude * exp(-(phi2 - phi2_centre)²
    / (2 sigma²)) + P(phi2 - background_reference)`` (``stream_width_model``),
    ``P`` a polynomial of degree ``background_degree`` (0: a constant, as
    Shipp et al. 2018; 1: linear) and ``background_reference`` the middle of
    the fitted range, on the bins of ``phi2_fit_range`` with a finite
    density, with the Poisson errors of the profile (empty bins: the error
    of one star). Only near the track: over a wider range the field density
    is not constant (e.g. a gradient of ~40 % over 10 deg in Euclid DR1),
    and a Gaussian takes the gradient instead of the stream.

    Parameters
    ----------
    profile : dict
        From ``phi12_density_profile`` along phi2.
    phi2_centre_guess, sigma_guess : float, optional
        Initial values (deg).
    phi2_fit_range : tuple of float, optional
        Bins fitted (deg). Default: ``phi2_centre_guess`` ±
        ``DEFAULT_FIT_HALF_WIDTH_DEG``.
    background_degree : int, optional
        Degree of the background polynomial: 0 constant, 1 linear (a field
        gradient left in the fitted range), more if needed.

    Returns
    -------
    fit : dict
        ``{"phi2_centre", "sigma", "amplitude", "background",
        "background_coefficients", "background_reference",
        "background_degree", "errors", "model", "chi2", "n_dof",
        "phi2_fit_range"}``: ``background`` the background at
        ``background_reference``; ``background_coefficients`` those of ``P``,
        constant first; ``errors`` a dict of the 1-sigma errors of
        ``amplitude``, ``phi2_centre``, ``sigma``, ``background`` and
        ``background_coefficients``; ``model`` the fitted density at
        ``profile["centres"]`` (all of them; it describes only
        ``phi2_fit_range``); ``chi2`` over ``n_dof`` degrees of freedom, on
        the bins fitted. Densities in stars/deg², angles in deg.

    Raises
    ------
    ValueError
        The profile is not along phi2, a negative ``background_degree``, or
        fewer bins with an area in ``phi2_fit_range`` than parameters + 2.
    RuntimeError
        The fit does not converge.

    Notes
    -----
    Bounds: amplitude and background at the reference >= 0, centre within
    the fitted range, 0.001 deg <= sigma <= its width. With a constant
    background, a field gradient left inside the range shifts the centre
    towards the denser side: in tests, by ~0.03 deg for a slope of 5 % of
    the stream amplitude per deg, within the errors at the statistics of the
    Euclid DR1 candidate, beyond them with far more stars; a linear
    background (``background_degree=1``) removes it.

    References
    ----------
    Shipp et al. 2018, ApJ 862, 114.
    """
    if profile["axis"] != "phi2":
        raise ValueError(f"fit_stream_width needs a profile along phi2, not {profile['axis']}.")
    if background_degree < 0:
        raise ValueError(f"background_degree must be >= 0, not {background_degree}.")
    if phi2_fit_range is None:
        phi2_fit_range = (phi2_centre_guess - DEFAULT_FIT_HALF_WIDTH_DEG, phi2_centre_guess + DEFAULT_FIT_HALF_WIDTH_DEG)
    n_parameters = 3 + background_degree + 1
    centres = profile["centres"]
    in_range = (centres >= phi2_fit_range[0]) & (centres <= phi2_fit_range[1])
    used = in_range & np.isfinite(profile["density"]) & (profile["area_deg2"] > 0)
    if used.sum() < n_parameters + 2:
        raise ValueError(f"Only {used.sum()} bins with an area in {phi2_fit_range}: too few to fit "
                         f"{n_parameters} parameters.")
    phi2, density = centres[used], profile["density"][used]
    error = np.sqrt(np.maximum(profile["counts"][used], 1.0)) / profile["area_deg2"][used]
    reference = (phi2_fit_range[0] + phi2_fit_range[1]) / 2
    extra = background_degree  # coefficients beyond the constant: free
    lower = [0.0, phi2.min(), 1e-3, 0.0] + [-np.inf] * extra
    upper = [np.inf, phi2.max(), phi2.max() - phi2.min(), np.inf] + [np.inf] * extra
    background = float(np.median(density))
    first_guess = np.clip([max(density.max() - background, 0.0), phi2_centre_guess, sigma_guess, background]
                          + [0.0] * extra, lower, upper)

    def model(x: NDArray[np.float64], *values: float) -> NDArray[np.float64]:
        return _gaussian_and_polynomial(x, *values[:3], values[3:], reference)

    values, covariance = curve_fit(model, phi2, density, p0=first_guess, sigma=error, absolute_sigma=True,
                                   bounds=(lower, upper))
    value_errors = np.sqrt(np.diag(covariance))
    fit = {
        "amplitude": float(values[0]), "phi2_centre": float(values[1]), "sigma": float(values[2]),
        "background": float(values[3]), "background_coefficients": tuple(float(v) for v in values[3:]),
        "background_reference": float(reference), "background_degree": int(background_degree),
    }
    fit["errors"] = {
        "amplitude": float(value_errors[0]), "phi2_centre": float(value_errors[1]), "sigma": float(value_errors[2]),
        "background": float(value_errors[3]), "background_coefficients": tuple(float(v) for v in value_errors[3:]),
    }
    fit["model"] = stream_width_model(fit, centres)
    fit["chi2"] = float(np.sum(((density - fit["model"][used]) / error) ** 2))
    fit["n_dof"] = int(used.sum() - n_parameters)
    fit["phi2_fit_range"] = (float(phi2_fit_range[0]), float(phi2_fit_range[1]))
    return fit


def stream_width_model(fit: dict[str, Any], phi2: ArrayLike) -> NDArray[np.float64]:
    """Density of the fitted Gaussian plus background at phi2 (stars/deg²), e.g. on a 2D grid.

    Parameters
    ----------
    fit : dict
        From ``fit_stream_width``.
    phi2 : array_like
        (deg).

    Returns
    -------
    density : numpy.ndarray
        Same shape as ``phi2``.
    """
    return _gaussian_and_polynomial(np.asarray(phi2, dtype=float), fit["amplitude"], fit["phi2_centre"],
                                    fit["sigma"], fit["background_coefficients"], fit["background_reference"])


def _gaussian_and_polynomial(
    phi2: NDArray[np.float64],
    amplitude: float,
    phi2_centre: float,
    sigma: float,
    background_coefficients: Sequence[float],
    background_reference: float,
) -> NDArray[np.float64]:
    """``amplitude * exp(-(phi2 - phi2_centre)² / (2 sigma²)) + P(phi2 - background_reference)``, constant first."""
    gaussian = amplitude * np.exp(-0.5 * ((phi2 - phi2_centre) / sigma) ** 2)
    return gaussian + polynomial.polyval(phi2 - background_reference, background_coefficients)

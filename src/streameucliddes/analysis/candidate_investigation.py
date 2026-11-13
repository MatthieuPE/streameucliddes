"""Investigation of one stream candidate: its regions and width, and one object holding every step.

Two layers, in this module:

- plain functions returning plain dicts, each callable alone and tunable:
  the on and off regions (``stream_regions``, ``region_windows``,
  ``region_area_deg2``), the densities (``phi12_density_map``,
  ``phi12_density_profile``, ``on_off_phi1_profile``) and the width
  (``fit_stream_width``, ``stream_width_model``);
- ``CandidateInvestigation``, which holds the inputs (two endpoints, the
  isochrone, the distance modulus, the region sizes) and the result of each
  step, and calls the functions in order. It adds no science: each method
  calls one or two functions of this module, ``stream_frames``,
  ``catalogs.survey_footprint``, ``analysis.generic_stream_search``,
  ``analysis.stream_frame_search``, ``analysis.build_masks`` and
  ``analysis.cmd_match_filter``, stores the result and returns the object,
  so that calls can be chained. A function can also be called alone and its
  result put back in the object (e.g. ``cand.regions = stream_regions(...)``).

The track is straight for now: phi2 = ``phi2_centre`` over ``phi1_range``.
Its width is the sigma of a Gaussian plus a background (a constant by
default, a polynomial of higher degree if needed), fitted to the phi2
density profile of the stars in the match filter, near the track
(``DEFAULT_FIT_HALF_WIDTH_DEG``). The on region is
``|phi2 - phi2_centre| <= on_n_sigma * sigma``; the off region is two bands,
one on each side, from ``off_n_sigma[0] * sigma`` to ``off_n_sigma[1] *
sigma``; all over ``phi1_range``. Each is a phi1/phi2 window
(``region_windows``), so the window functions of ``stream_frames`` select
their stars, draw them and, with ``catalogs.survey_footprint.unmasked_area_deg2``,
give their areas. Areas and densities are outside the excluded mask
(footprint OR masked objects, ``analysis.build_masks.excluded_healpix_mask``),
so that the stars, the maps and the areas see the same masks.

Steps of ``CandidateInvestigation``, in order:

1. frame, window and (ra, dec) box, at construction (``__post_init__``);
2. ``compute_sky_map``: matched-filter map of the loaded box, once;
3. ``prepare`` and ``compute_zoom_map``; ``set_endpoints`` to move the
   endpoints by hand until the frame follows the stream;
4. ``scan_distance_moduli``: maps in the frame over distance moduli; the
   user then sets ``distance_modulus`` by eye;
5. ``fit_width``: phi2 density profile and Gaussian width;
6. ``define_regions``: on and off regions, their areas, phi1 profile;
7. ``compute_hess``: on, off and on - off Hess diagrams;
8. ``compute_isochrone``: isochrone track over the Hess diagrams.

``run_all_steps`` does them all. The figures take the object
(``plotting.plot_candidate_investigation``).

References
----------
Shipp et al. 2018, ApJ 862, 114 (Gaussian width with a constant background;
on - off Hess diagrams). On band 1.4 sigma, off bands 3.5 to 6 sigma: user
choices, no reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.polynomial import polynomial
from scipy.optimize import curve_fit

from streameucliddes.analysis.build_masks import excluded_healpix_mask
from streameucliddes.analysis.cmd_match_filter import (
    difference_hess,
    hess_diagram,
    isochrone_cmd_track,
    match_filter_columns,
    select_stars_in_match_filter,
)
from streameucliddes.analysis.generic_stream_search import generic_search_single_dm
from streameucliddes.analysis.stream_frame_search import (
    prepare_search,
    stream_frame_search_dm_scan,
    stream_frame_search_single_dm,
)
from streameucliddes.catalogs.isochrone_parameters import DEFAULT_AGES_GYR, DEFAULT_METALLICITIES
from streameucliddes.catalogs.survey_footprint import survey_footprint_healpix_mask, unmasked_area_deg2
from streameucliddes.stream_frames import (
    DEFAULT_MIN_PHI2_HALF_WIDTH_DEG,
    DEFAULT_PADDING_PHI1_DEG,
    DEFAULT_PADDING_PHI2_DEG,
    in_phi12_window,
    phi12_to_radec,
    phi12_window_around,
    phi12_window_radec_box,
    radec_to_phi12,
    stream_frame_from_endpoints,
)
from streameucliddes.unit_conversions import convert_FeH_to_z

if TYPE_CHECKING:
    from collections.abc import Sequence

    import astropy.coordinates as coord
    import pandas as pd
    from numpy.typing import ArrayLike, NDArray

    from streameucliddes.analysis.stream_frame_search import PreparedSearch

DEFAULT_ON_N_SIGMA = 1.4  # user choice
DEFAULT_OFF_N_SIGMA = (3.5, 6)  # user choice
DEFAULT_SIGMA_GUESS_DEG = 0.2  # initial sigma of the width fit
DEFAULT_PHI2_BIN_DEG = 0.05  # bins of the phi2 profile
DEFAULT_PHI1_BIN_DEG = 0.5  # bins of the phi1 profile
DEFAULT_FIT_HALF_WIDTH_DEG = 2.0  # width fit on phi2_centre_guess +- this: the field density is not constant further
DEFAULT_BACKGROUND_DEGREE = 0  # background of the width fit: a constant (1: linear)
DEFAULT_WIDTH_MAP_BINS_DEG = (0.5, 0.2)  # (phi1, phi2) cells of the 2D map of the width fit
DEFAULT_FOOTPRINT_NSIDE = 512  # footprint from the stars, when there is no sky map
DEFAULT_TRACK_POINTS = 100
CATALOG_EDGE_TOLERANCE_DEG = 0.1  # window outline beyond the outermost stars by less than this: still inside
WINDOW_FIELDS = ("padding_phi1_deg", "padding_phi2_deg", "min_phi2_half_width_deg")
# Results that depend on the frame: cleared by prepare and set_endpoints (not sky_map nor isochrone_track)
FRAME_RESULTS = ("search", "excluded_healpix_mask", "zoom_map", "maps", "phi2_profile", "width_fit", "width_map",
                 "regions", "areas", "phi1_profile", "on_hess", "off_hess", "diff_hess")


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
    """On and off regions of a stream, as a plain dict to edit by hand.

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
        ``off_n_sigma[1] <= off_n_sigma[0]``, or ``sigma <= 0``, or
        ``phi1_range`` not increasing.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, not {sigma}.")
    if not off_n_sigma[0] >= on_n_sigma:
        raise ValueError(f"The off bands ({off_n_sigma} sigma) overlap the on region ({on_n_sigma} sigma).")
    if not off_n_sigma[1] > off_n_sigma[0]:
        raise ValueError(f"off_n_sigma must be increasing, not {off_n_sigma}.")
    if not phi1_range[1] > phi1_range[0]:
        raise ValueError(f"phi1_range must be increasing, not {phi1_range}.")
    return {
        "phi1_range": (float(phi1_range[0]), float(phi1_range[1])), "phi2_centre": float(phi2_centre),
        "sigma": float(sigma), "on_n_sigma": float(on_n_sigma),
        "off_n_sigma": (float(off_n_sigma[0]), float(off_n_sigma[1])),
    }


def region_windows(regions: dict[str, Any]) -> dict[str, Any]:
    """phi1/phi2 windows of the on region and of the two off bands.

    Computed from the regions each time, so that a region edited by hand
    stays consistent.

    Parameters
    ----------
    regions : dict
        From ``stream_regions``.

    Returns
    -------
    windows : dict
        ``{"on": window, "off": (window_below, window_above)}``, each
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg). Stars:
        ``stream_frames.in_phi12_window``; outlines:
        ``stream_frames.phi12_window_outline``.
    """
    phi1_min, phi1_max = regions["phi1_range"]
    centre, sigma = regions["phi2_centre"], regions["sigma"]
    on_half = regions["on_n_sigma"] * sigma
    off_inner, off_outer = (n_sigma * sigma for n_sigma in regions["off_n_sigma"])
    return {
        "on": (phi1_min, phi1_max, centre - on_half, centre + on_half),
        "off": ((phi1_min, phi1_max, centre - off_outer, centre - off_inner),
                (phi1_min, phi1_max, centre + off_inner, centre + off_outer)),
    }


def region_area_deg2(
    regions: dict[str, Any],
    stream_frame: coord.BaseCoordinateFrame,
    excluded_healpix_mask: NDArray[np.bool_] | None,
) -> dict[str, float]:
    """Areas of the on and off regions outside an excluded HEALPix mask.

    ``catalogs.survey_footprint.unmasked_area_deg2`` of each window of
    ``region_windows``.

    Parameters
    ----------
    regions : dict
        From ``stream_regions``.
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Frame of the regions.
    excluded_healpix_mask : numpy.ndarray of bool or None
        True where excluded, RING. None: nothing excluded.

    Returns
    -------
    areas : dict
        ``{"on", "off"}`` (deg²); ``"off"`` is the sum of the two bands.
    """
    windows = region_windows(regions)
    on = unmasked_area_deg2(stream_frame, windows["on"][:2], windows["on"][2:], excluded_healpix_mask)[0, 0]
    off = sum(unmasked_area_deg2(stream_frame, window[:2], window[2:], excluded_healpix_mask)[0, 0]
              for window in windows["off"])
    return {"on": float(on), "off": float(off)}


# ===================================================
# Profiles and width
# ===================================================


def phi12_density_profile(
    phi1: ArrayLike,
    phi2: ArrayLike,
    windows: Sequence[tuple[float, float, float, float]],
    stream_frame: coord.BaseCoordinateFrame,
    excluded_healpix_mask: NDArray[np.bool_] | None,
    *,
    axis: str,
    bin_deg: float,
    axis_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Density of stars along phi1 or phi2, over one or more windows.

    The stars in the windows (e.g. those in the match filter;
    ``stream_frames.in_phi12_window``) are counted in bins of ``bin_deg``
    along ``axis``; each bin is divided by its area inside the windows and
    outside the excluded mask (``catalogs.survey_footprint.unmasked_area_deg2``),
    so that tile edges and masked objects make no dips. One function for
    both profiles: along phi2 over the window of the maps (width), along
    phi1 over the on region or the two off bands (``on_off_phi1_profile``).

    Parameters
    ----------
    phi1, phi2 : array_like
        Coordinates of the stars in the stream frame (deg).
    windows : sequence of tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg), not overlapping;
        their counts and areas add up.
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Frame of the stars.
    excluded_healpix_mask : numpy.ndarray of bool or None
        True where excluded, RING. None: nothing excluded.
    axis : {"phi1", "phi2"}
        Axis of the profile.
    bin_deg : float
        Bin size along ``axis`` (deg), e.g. ``DEFAULT_PHI2_BIN_DEG``:
        ``axis_range`` is cut into the whole number of bins closest to it.
    axis_range : tuple of float, optional
        Range of the bins (deg). Default: that of the windows along ``axis``.

    Returns
    -------
    profile : dict
        ``{"axis", "centres", "edges", "counts", "area_deg2", "density",
        "error", "bin_deg", "windows"}``: density in stars/deg², error
        ``sqrt(counts) / area_deg2``, ``bin_deg`` the bin size used. Bins
        with no area have NaN density and error.

    Raises
    ------
    ValueError
        ``axis`` neither "phi1" nor "phi2".
    """
    if axis not in ("phi1", "phi2"):
        raise ValueError(f"axis must be 'phi1' or 'phi2', not {axis!r}.")
    windows = [tuple(float(value) for value in window) for window in windows]
    k = 0 if axis == "phi1" else 2  # index of the window minimum along the axis
    if axis_range is None:
        axis_range = (min(window[k] for window in windows), max(window[k + 1] for window in windows))
    edges = _bin_edges(axis_range, bin_deg)
    counts, area = np.zeros(edges.size - 1), np.zeros(edges.size - 1)
    for window in windows:
        clipped = np.clip(edges, window[k], window[k + 1])  # the bins inside the window; outside: zero width
        if axis == "phi1":
            cells = phi12_density_map(phi1, phi2, clipped, window[2:], stream_frame, excluded_healpix_mask)
        else:
            cells = phi12_density_map(phi1, phi2, window[:2], clipped, stream_frame, excluded_healpix_mask)
        counts += cells["counts"].ravel()
        area += cells["area_deg2"].ravel()
    density, error = _density_and_error(counts, area)
    return {
        "axis": axis, "centres": (edges[:-1] + edges[1:]) / 2, "edges": edges, "counts": counts,
        "area_deg2": area, "density": density, "error": error, "bin_deg": float(edges[1] - edges[0]),
        "windows": windows,
    }


def phi12_density_map(
    phi1: ArrayLike,
    phi2: ArrayLike,
    phi1_edges: ArrayLike,
    phi2_edges: ArrayLike,
    stream_frame: coord.BaseCoordinateFrame,
    excluded_healpix_mask: NDArray[np.bool_] | None,
) -> dict[str, Any]:
    """Density of stars on a phi1/phi2 grid: counts per cell over its area outside a mask.

    Not smoothed (unlike the maps of ``analysis.stream_frame_search``): each
    cell is the count of its stars divided by its area outside the excluded
    mask (``catalogs.survey_footprint.unmasked_area_deg2``), in stars/deg².
    ``phi12_density_profile`` adds its cells up along one axis.

    Parameters
    ----------
    phi1, phi2 : array_like
        Coordinates of the stars in the stream frame (deg).
    phi1_edges, phi2_edges : array_like
        Edges of the cells (deg), increasing.
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Frame of the stars.
    excluded_healpix_mask : numpy.ndarray of bool or None
        True where excluded, RING. None: nothing excluded.

    Returns
    -------
    density_map : dict
        ``{"phi1_edges", "phi2_edges", "counts", "area_deg2", "density",
        "error"}``, the arrays of shape (n_phi1, n_phi2): density in
        stars/deg², error ``sqrt(counts) / area_deg2``; cells with no area
        have NaN density and error.
    """
    phi1_edges, phi2_edges = np.asarray(phi1_edges, dtype=float), np.asarray(phi2_edges, dtype=float)
    counts, _, _ = np.histogram2d(np.asarray(phi1, dtype=float), np.asarray(phi2, dtype=float),
                                  bins=[phi1_edges, phi2_edges])
    area = unmasked_area_deg2(stream_frame, phi1_edges, phi2_edges, excluded_healpix_mask)
    density, error = _density_and_error(counts, area)
    return {"phi1_edges": phi1_edges, "phi2_edges": phi2_edges, "counts": counts, "area_deg2": area,
            "density": density, "error": error}


def _bin_edges(value_range: tuple[float, float], bin_deg: float) -> NDArray[np.float64]:
    """Edges cutting a range into the whole number of bins closest to bin_deg (at least one)."""
    n_bins = max(int(round((value_range[1] - value_range[0]) / bin_deg)), 1)
    return np.linspace(value_range[0], value_range[1], n_bins + 1)


def _density_and_error(
    counts: NDArray[np.float64], area: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """counts / area and sqrt(counts) / area; NaN where there is no area."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(area > 0, counts / area, np.nan), np.where(area > 0, np.sqrt(counts) / area, np.nan)


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


def on_off_phi1_profile(
    phi1: ArrayLike,
    phi2: ArrayLike,
    regions: dict[str, Any],
    stream_frame: coord.BaseCoordinateFrame,
    excluded_healpix_mask: NDArray[np.bool_] | None,
    *,
    phi1_range: tuple[float, float] | None = None,
    bin_deg: float = DEFAULT_PHI1_BIN_DEG,
) -> dict[str, Any]:
    """On - off density along phi1, to see where the stream starts and stops.

    ``phi12_density_profile`` along phi1 of the on region and of the two off
    bands (``region_windows``, over ``phi1_range``), and their difference.

    Parameters
    ----------
    phi1, phi2 : array_like
        Coordinates of the stars in the stream frame (deg).
    regions : dict
        From ``stream_regions``; its phi2 limits are used.
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Frame of the stars.
    excluded_healpix_mask : numpy.ndarray of bool or None
        True where excluded, RING. None: nothing excluded.
    phi1_range : tuple of float, optional
        Range of the bins (deg), e.g. that of the window of the maps, to
        extend beyond the regions. Default: ``regions["phi1_range"]``.
    bin_deg : float, optional
        Bin size in phi1 (deg).

    Returns
    -------
    profile : dict
        ``{"centres", "edges", "on", "off", "difference", "error",
        "on_error", "off_error", "bin_deg", "regions"}``: densities in
        stars/deg²; ``error`` that of the difference (Poisson, in
        quadrature). Bins with no area in the on region or the off bands
        have NaN.
    """
    if phi1_range is None:
        phi1_range = regions["phi1_range"]
    windows = region_windows({**regions, "phi1_range": tuple(phi1_range)})
    kwargs = {"axis": "phi1", "bin_deg": bin_deg, "axis_range": tuple(phi1_range)}
    on = phi12_density_profile(phi1, phi2, [windows["on"]], stream_frame, excluded_healpix_mask, **kwargs)
    off = phi12_density_profile(phi1, phi2, list(windows["off"]), stream_frame, excluded_healpix_mask, **kwargs)
    return {
        "centres": on["centres"], "edges": on["edges"], "on": on["density"], "off": off["density"],
        "difference": on["density"] - off["density"], "error": np.hypot(on["error"], off["error"]),
        "on_error": on["error"], "off_error": off["error"], "bin_deg": on["bin_deg"], "regions": dict(regions),
    }


# ===================================================
# The investigation
# ===================================================


@dataclass
class CandidateInvestigation:
    """All inputs and results of one stream-candidate investigation.

    Mutable: change a field and re-run the later steps. Results are None
    until their step is done.

    Parameters
    ----------
    name : str
        Name of the candidate, for titles.
    endpoint_1, endpoint_2 : tuple of float
        (ra, dec) of two points along the stream (deg).
    age : float, optional
        Isochrone age (Gyr).
    metallicity : float, optional
        Isochrone Z (Marigo et al. 2017 scale). At most one of
        ``metallicity`` and ``feh``; neither: Z =
        ``DEFAULT_METALLICITIES[0]`` of ``catalogs.isochrone_parameters``.
    feh : float, optional
        Isochrone [Fe/H], converted to Z (``unit_conversions.convert_FeH_to_z``)
        each time the isochrone is used (``isochrone``).
    distance_modulus : float, optional
        Distance modulus of the stream, chosen by eye after the scan
        (``scan_distance_moduli``). Needed from ``fit_width`` on.
    distance_moduli : sequence of float, optional
        Grid of the scan. Default: that of ``prepare_search``.
    match_filter : dict, optional
        See ``analysis.cmd_match_filter``. Default: ``DEFAULT_MATCH_FILTER``.
    padding_phi1_deg, padding_phi2_deg, min_phi2_half_width_deg : float, optional
        Window around the endpoints (``stream_frames.phi12_window_around``).
    on_n_sigma : float, optional
        Half-width of the on region, in units of sigma.
    off_n_sigma : tuple of float, optional
        Distances of the off bands, in units of sigma.
    prepare_kwargs : dict, optional
        More arguments of ``prepare_search`` (``bin_size_deg``, ``cuts``,
        ``mask_*``...).

    Attributes
    ----------
    stream_frame : gala.coordinates.GreatCircleICRSFrame
        From the endpoints (``stream_frame_from_endpoints``).
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg).
    radec_box : tuple
        ``(ra_range, dec_range)`` of the window (``phi12_window_radec_box``).
    catalog : pandas.DataFrame
        The stars given to ``prepare`` (a reference, not a copy), kept to
        prepare again after ``set_endpoints``.
    footprint_mask, footprint_nest : numpy.ndarray of bool, bool
        HEALPix footprint of the catalog (True outside) and its ordering, kept
        for ``set_endpoints``.
    search : PreparedSearch
        From ``prepare_search``.
    excluded_healpix_mask : numpy.ndarray of bool
        Footprint OR masked objects, RING (``build_masks.excluded_healpix_mask``).
    sky_map : dict
        ``generic_search_single_dm`` on the loaded box; does not depend on
        the frame.
    zoom_map : dict
        ``stream_frame_search_single_dm`` at the distance modulus of ``sky_map``.
    maps : dict
        ``{distance_modulus: result of stream_frame_search_single_dm}``.
    phi2_profile, width_fit, width_map : dict
        ``phi12_density_profile`` (along phi2), ``fit_stream_width`` and
        ``phi12_density_map`` (the same stars in 2D) results.
    regions, areas, phi1_profile : dict
        ``stream_regions``, ``region_area_deg2`` and ``on_off_phi1_profile``
        results.
    on_hess, off_hess, diff_hess : dict
        ``hess_diagram`` and ``difference_hess`` results.
    isochrone_track : dict
        ``isochrone_cmd_track`` result.
    """

    # Inputs
    name: str
    endpoint_1: tuple[float, float]
    endpoint_2: tuple[float, float]
    age: float = DEFAULT_AGES_GYR[0]
    metallicity: float | None = None
    feh: float | None = None
    distance_modulus: float | None = None
    distance_moduli: Sequence[float] | None = None
    match_filter: dict | None = None
    padding_phi1_deg: float = DEFAULT_PADDING_PHI1_DEG
    padding_phi2_deg: float = DEFAULT_PADDING_PHI2_DEG
    min_phi2_half_width_deg: float = DEFAULT_MIN_PHI2_HALF_WIDTH_DEG
    on_n_sigma: float = DEFAULT_ON_N_SIGMA
    off_n_sigma: tuple[float, float] = DEFAULT_OFF_N_SIGMA
    prepare_kwargs: dict = field(default_factory=dict)
    # Results, None until computed
    stream_frame: Any = None
    window: tuple | None = None
    radec_box: tuple | None = None
    catalog: Any = None
    footprint_mask: NDArray | None = None
    footprint_nest: bool = False
    search: PreparedSearch | None = None
    excluded_healpix_mask: NDArray | None = None
    sky_map: dict | None = None
    zoom_map: dict | None = None
    maps: dict | None = None
    phi2_profile: dict | None = None
    width_fit: dict | None = None
    width_map: dict | None = None
    regions: dict | None = None
    areas: dict | None = None
    phi1_profile: dict | None = None
    on_hess: dict | None = None
    off_hess: dict | None = None
    diff_hess: dict | None = None
    isochrone_track: dict | None = None

    def __post_init__(self) -> None:
        """Frame, window and (ra, dec) box from the endpoints (``stream_frames``).

        Raises
        ------
        ValueError
            Both ``metallicity`` and ``feh`` given.
        """
        _ = self.isochrone  # raises if both metallicity and feh are set
        self._set_frame_and_window()

    @property
    def isochrone(self) -> dict[str, float]:
        """``{"age", "metallicity"}`` of the match filter and of the isochrone track.

        Z from ``feh`` when given (``unit_conversions.convert_FeH_to_z``),
        else ``metallicity``, else ``DEFAULT_METALLICITIES[0]``. Computed each
        time: changing ``age``, ``metallicity`` or ``feh`` changes it.

        Raises
        ------
        ValueError
            Both ``metallicity`` and ``feh`` set.
        """
        if self.metallicity is not None and self.feh is not None:
            raise ValueError("Give metallicity (Z) or feh, not both.")
        if self.feh is not None:
            metallicity = float(convert_FeH_to_z(self.feh))
        elif self.metallicity is not None:
            metallicity = float(self.metallicity)
        else:
            metallicity = float(DEFAULT_METALLICITIES[0])
        return {"age": float(self.age), "metallicity": metallicity}

    def _set_frame_and_window(self) -> None:
        """Frame from the endpoints, window around them, (ra, dec) box of the window."""
        self.endpoint_1 = (float(self.endpoint_1[0]), float(self.endpoint_1[1]))
        self.endpoint_2 = (float(self.endpoint_2[0]), float(self.endpoint_2[1]))
        self.stream_frame = stream_frame_from_endpoints(self.endpoint_1, self.endpoint_2)
        phi1, phi2 = radec_to_phi12(
            [self.endpoint_1[0], self.endpoint_2[0]], [self.endpoint_1[1], self.endpoint_2[1]], self.stream_frame
        )
        self.window = phi12_window_around(
            phi1, phi2, padding_phi1_deg=self.padding_phi1_deg, padding_phi2_deg=self.padding_phi2_deg,
            min_phi2_half_width_deg=self.min_phi2_half_width_deg,
        )
        self.radec_box = phi12_window_radec_box(self.stream_frame, self.window)

    # ===================================================
    # Steps
    # ===================================================

    def compute_sky_map(
        self, catalog: pd.DataFrame, distance_modulus: float | None = None, **generic_kwargs: Any
    ) -> CandidateInvestigation:
        """Matched-filter map of the loaded box, once (``generic_search_single_dm``).

        The corrected map (``residual``) shows the streams of the box; the
        frame check draws the great circle of the endpoints over it. Does not
        depend on the frame: ``set_endpoints`` keeps it. The catalog is kept
        in ``self.catalog`` (a reference), the default of ``prepare``.

        Parameters
        ----------
        catalog : pandas.DataFrame
            Stars of a box larger than the candidate, e.g. from
            ``get_euclid_stars``. Not copied.
        distance_modulus : float, optional
            Default: ``self.distance_modulus``.
        **generic_kwargs
            Arguments of ``generic_search_single_dm`` (``smoothing_deg``,
            ``polynomial_degree``) and of ``prepare_generic_search``
            (``nside``, ``cuts``, ``mask_*``...). The isochrone and the match
            filter are those of the object.

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            No distance modulus given nor set.
        """
        distance_modulus = self.distance_modulus if distance_modulus is None else distance_modulus
        if distance_modulus is None:
            raise ValueError("compute_sky_map needs a distance_modulus: give one, or set cand.distance_modulus.")
        self.sky_map = generic_search_single_dm(
            catalog, distance_modulus, self.isochrone, match_filter=self.match_filter, **generic_kwargs
        )
        if catalog is not self.catalog:
            self.footprint_mask = None  # that of another catalog
        self.catalog = catalog
        return self

    def prepare(
        self,
        catalog: pd.DataFrame | None = None,
        *,
        footprint_mask: NDArray | None = None,
        nest: bool = False,
    ) -> CandidateInvestigation:
        """Stars of the window and excluded mask (``prepare_search``, ``excluded_healpix_mask``).

        ``prepare_search`` with the frame, window, isochrone, match filter and
        ``distance_moduli`` of the object, and ``prepare_kwargs``; then the
        excluded mask: the footprint OR the objects masked by
        ``prepare_search`` (``build_masks.excluded_healpix_mask``, RING). The
        results of the later steps are cleared.

        The footprint is not that of ``sky_map``: the generic search adds the
        masked objects to it at its own nside; here the objects come at the
        finer nside of ``prepare_search``, the one the stars are dropped with.

        Parameters
        ----------
        catalog : pandas.DataFrame, optional
            Stars, kept in ``self.catalog`` (a reference). Default: the
            catalog of ``compute_sky_map`` or of a previous call.
        footprint_mask : numpy.ndarray of bool, optional
            HEALPix footprint, True outside, kept in ``self.footprint_mask``.
            Default: the one kept, else from the stars
            (``survey_footprint_healpix_mask`` at ``DEFAULT_FOOTPRINT_NSIDE``).
        nest : bool, optional
            Ordering of a given ``footprint_mask``.

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            No catalog given nor kept; or the window leaves the (ra, dec) range
            of the catalog.
        """
        catalog = self.catalog if catalog is None else catalog
        if catalog is None:
            raise ValueError("prepare needs the stars: give a catalog, or call compute_sky_map first.")
        self._check_window_in_catalog(catalog)
        if catalog is not self.catalog:
            self.footprint_mask = None  # that of another catalog
        self.catalog = catalog
        if footprint_mask is not None:
            self.footprint_mask, self.footprint_nest = np.asarray(footprint_mask, dtype=bool), nest
        elif self.footprint_mask is None:
            self.footprint_mask = survey_footprint_healpix_mask(
                catalog["ra"], catalog["dec"], DEFAULT_FOOTPRINT_NSIDE, nest=False
            )
            self.footprint_nest = False
        self._prepare_stars()
        return self

    def set_endpoints(
        self,
        endpoint_1: tuple[float, float],
        endpoint_2: tuple[float, float],
        **window_kwargs: Any,
    ) -> CandidateInvestigation:
        """New endpoints: new frame, window and box; prepared again; later results cleared.

        Fast: the sky map, the footprint and the object mask (of the last
        ``prepare``, which does not depend on the frame) are kept; only the
        stars of the new window are prepared (when there is a catalog). Then
        call ``compute_zoom_map`` and ``plot_candidate_frame_check``. After a
        change of ``prepare_kwargs``, call ``prepare()`` instead.

        Parameters
        ----------
        endpoint_1, endpoint_2 : tuple of float
            (ra, dec) (deg).
        **window_kwargs
            ``padding_phi1_deg``, ``padding_phi2_deg``,
            ``min_phi2_half_width_deg``: set on the object.

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            The new window leaves the (ra, dec) range of the kept catalog:
            load a larger box. The object is then left unchanged.
        TypeError
            Another keyword than those of the window.
        """
        unknown = set(window_kwargs) - set(WINDOW_FIELDS)
        if unknown:
            raise TypeError(f"Unexpected arguments {sorted(unknown)}; set_endpoints takes {WINDOW_FIELDS}.")
        previous = {name: getattr(self, name) for name in ("endpoint_1", "endpoint_2", *WINDOW_FIELDS)}
        self.endpoint_1, self.endpoint_2 = endpoint_1, endpoint_2
        for name, value in window_kwargs.items():
            setattr(self, name, value)
        old_frame = (self.stream_frame, self.window, self.radec_box)
        self._set_frame_and_window()
        if self.catalog is not None:
            try:
                self._check_window_in_catalog(self.catalog)
            except ValueError:  # leave the object as it was
                for name, value in previous.items():
                    setattr(self, name, value)
                self.stream_frame, self.window, self.radec_box = old_frame
                raise
            self._prepare_stars(None if self.search is None else self.search.object_healpix_mask)
        else:
            self._clear_results(FRAME_RESULTS)
        return self

    def compute_zoom_map(self, distance_modulus: float | None = None, **map_kwargs: Any) -> CandidateInvestigation:
        """One matched-filter map in the frame (``stream_frame_search_single_dm``).

        Parameters
        ----------
        distance_modulus : float, optional
            Default: that of ``sky_map``, else ``self.distance_modulus``.
        **map_kwargs
            Arguments of ``stream_frame_search_single_dm`` (``smoothing_deg``).

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            ``prepare`` not done, or no distance modulus.
        """
        self._require("search", "prepare")
        if distance_modulus is None:
            distance_modulus = self.sky_map["distance_modulus"] if self.sky_map is not None else self.distance_modulus
        if distance_modulus is None:
            raise ValueError("compute_zoom_map needs a distance_modulus: give one, compute_sky_map, or set "
                             "cand.distance_modulus.")
        self.zoom_map = stream_frame_search_single_dm(
            search=self.search, distance_modulus=distance_modulus, isochrone=self.isochrone, **map_kwargs
        )
        return self

    # ===================================================
    # Checks
    # ===================================================

    def _prepare_stars(self, object_healpix_mask: NDArray | None = None) -> None:
        """``prepare_search`` on the kept catalog and footprint, and the excluded mask; later results cleared.

        ``object_healpix_mask``: the object mask of a previous search, not built again.
        """
        self._clear_results(FRAME_RESULTS)
        self.search = prepare_search(
            self.catalog, self.stream_frame, self.window, match_filter=self.match_filter, isochrone=self.isochrone,
            distance_moduli=self.distance_moduli, footprint_mask=self.footprint_mask, nest=self.footprint_nest,
            object_healpix_mask=object_healpix_mask, **self.prepare_kwargs,
        )
        self.excluded_healpix_mask, _ = excluded_healpix_mask(
            self.footprint_mask, self.search.object_healpix_mask, nest=self.footprint_nest
        )

    def _endpoints_phi1_range(self) -> tuple[float, float]:
        """phi1 of endpoint_1 and endpoint_2 (deg): negative, then positive."""
        phi1, _ = radec_to_phi12(
            [self.endpoint_1[0], self.endpoint_2[0]], [self.endpoint_1[1], self.endpoint_2[1]], self.stream_frame
        )
        return float(phi1[0]), float(phi1[1])

    def _stars_in_match_filter(self) -> NDArray[np.bool_]:
        """Prepared stars in the match filter at ``distance_modulus`` (``select_stars_in_match_filter``)."""
        if self.distance_modulus is None:
            raise ValueError("No distance_modulus: set cand.distance_modulus, e.g. from the maps of "
                             "scan_distance_moduli.")
        return select_stars_in_match_filter(
            self.search.stars, self.distance_modulus, self.isochrone, self.search.match_filter
        )

    def _require(self, field_name: str, step: str) -> None:
        """ValueError naming the missing step when ``field_name`` is not computed."""
        if getattr(self, field_name) is None:
            raise ValueError(f"No {field_name}: call {step}() first.")

    def _clear_results(self, names: Sequence[str]) -> None:
        """Set these results back to None."""
        for name in names:
            setattr(self, name, None)

    def _check_window_in_catalog(self, catalog: pd.DataFrame) -> None:
        """ValueError when the window leaves the (ra, dec) range of the catalog.

        Range of the stars around the ra of the window centre (no jump at ra =
        0/360); window outline from ``phi12_window_radec_box`` without margin.
        A window containing a pole is not checked.
        """
        (ra_min, ra_max), (dec_min, dec_max) = phi12_window_radec_box(self.stream_frame, self.window, margin_deg=0.0)
        if (ra_min, ra_max) == (0.0, 360.0):
            return
        centre_ra = (ra_min + ((ra_max - ra_min) % 360.0) / 2) % 360.0

        def relative(ra: Any) -> Any:
            return (np.asarray(ra) - centre_ra + 180.0) % 360.0 - 180.0

        star_ra = relative(catalog["ra"].to_numpy())
        star_dec = catalog["dec"].to_numpy()
        tolerance = CATALOG_EDGE_TOLERANCE_DEG
        if (relative(ra_min) < star_ra.min() - tolerance or relative(ra_max) > star_ra.max() + tolerance
                or dec_min < star_dec.min() - tolerance or dec_max > star_dec.max() + tolerance):
            raise ValueError(
                f"The window (ra {ra_min:.1f} to {ra_max:.1f}, dec {dec_min:.1f} to {dec_max:.1f} deg) leaves the "
                f"stars (ra {(star_ra.min() + centre_ra) % 360:.1f} to {(star_ra.max() + centre_ra) % 360:.1f}, dec "
                f"{star_dec.min():.1f} to {star_dec.max():.1f} deg): load a larger box."
            )

    def scan_distance_moduli(
        self, distance_moduli: Sequence[float] | None = None, **scan_kwargs: Any
    ) -> CandidateInvestigation:
        """Maps in the frame over distance moduli (``stream_frame_search_dm_scan``), in ``maps``.

        One map per distance modulus, with the isochrone of the object, on
        the prepared stars. The maps are small (the grid of the window): all
        are kept, ``{distance_modulus: result}``. Then choose
        ``distance_modulus`` by eye (``plot_candidate_map``).

        Parameters
        ----------
        distance_moduli : sequence of float, optional
            Default: ``self.distance_moduli``, else the grid of the search
            (``prepare_search``: 15 to 22 every 0.1).
        **scan_kwargs
            Arguments of ``stream_frame_search_dm_scan`` (``smoothing_deg``,
            ``n_workers``).

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            ``prepare`` not done.
        """
        self._require("search", "prepare")
        if distance_moduli is None:
            distance_moduli = self.distance_moduli
        self.maps = dict(stream_frame_search_dm_scan(
            search=self.search, distance_moduli=distance_moduli, isochrone=self.isochrone, **scan_kwargs
        ))
        return self

    def fit_width(
        self,
        phi1_range: tuple[float, float] | None = None,
        *,
        bin_deg: float = DEFAULT_PHI2_BIN_DEG,
        map_bins_deg: tuple[float, float] = DEFAULT_WIDTH_MAP_BINS_DEG,
        **fit_kwargs: Any,
    ) -> CandidateInvestigation:
        """phi2 density profile of the stars in the match filter, and its Gaussian width.

        Stars of ``search.stars`` in the match filter at
        ``distance_modulus`` (``select_stars_in_match_filter``), then
        ``phi12_density_profile`` along phi2 over ``phi1_range`` and the phi2
        range of the window (outside ``excluded_healpix_mask``), and
        ``fit_stream_width``: in ``phi2_profile`` and ``width_fit``. Also
        their 2D density over the same window (``phi12_density_map``), in
        ``width_map``, to compare with the model (``plot_width_fit_map``).

        Parameters
        ----------
        phi1_range : tuple of float, optional
            Default: the phi1 of the endpoints.
        bin_deg : float, optional
            Bins of the profile in phi2 (deg).
        map_bins_deg : tuple of float, optional
            (phi1, phi2) cells of the 2D map (deg): long along the stream.
        **fit_kwargs
            Arguments of ``fit_stream_width`` (``phi2_centre_guess``,
            ``sigma_guess``, ``phi2_fit_range``, ``background_degree``).

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            ``prepare`` not done, or ``distance_modulus`` not set.
        """
        self._require("search", "prepare")
        selected = self._stars_in_match_filter()
        phi1_range = self._endpoints_phi1_range() if phi1_range is None else tuple(phi1_range)
        stars = self.search.stars
        phi1, phi2 = stars["phi1"].to_numpy()[selected], stars["phi2"].to_numpy()[selected]
        phi2_range = self.window[2:]
        self.phi2_profile = phi12_density_profile(
            phi1, phi2, [(*phi1_range, *phi2_range)], self.stream_frame, self.excluded_healpix_mask,
            axis="phi2", bin_deg=bin_deg,
        )
        self.width_fit = fit_stream_width(self.phi2_profile, **fit_kwargs)
        self.width_map = phi12_density_map(
            phi1, phi2, _bin_edges(phi1_range, map_bins_deg[0]), _bin_edges(phi2_range, map_bins_deg[1]),
            self.stream_frame, self.excluded_healpix_mask,
        )
        return self

    def define_regions(
        self,
        phi1_range: tuple[float, float] | None = None,
        phi2_centre: float | None = None,
        sigma: float | None = None,
        *,
        bin_deg: float = DEFAULT_PHI1_BIN_DEG,
    ) -> CandidateInvestigation:
        """On and off regions, their areas and the phi1 profile.

        ``stream_regions`` (with ``on_n_sigma`` and ``off_n_sigma`` of the
        object), ``region_area_deg2`` (outside ``excluded_healpix_mask``) and
        ``on_off_phi1_profile`` of the stars in the match filter at
        ``distance_modulus``, its bins over the whole window: where the stream
        starts and stops, to choose ``phi1_range``. The Hess diagrams of
        earlier regions are cleared.

        Parameters
        ----------
        phi1_range : tuple of float, optional
            Default: the phi1 of the endpoints.
        phi2_centre, sigma : float, optional
            Default: from ``width_fit``.
        bin_deg : float, optional
            Bins of the phi1 profile (deg).

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            ``prepare`` not done; ``phi2_centre`` or ``sigma`` not given and
            ``fit_width`` not done; ``distance_modulus`` not set; or the off
            bands leave the window (make it wider: ``min_phi2_half_width_deg``).
        """
        self._require("search", "prepare")
        if (phi2_centre is None or sigma is None) and self.width_fit is None:
            raise ValueError("No width_fit: call fit_width() first, or give phi2_centre and sigma.")
        phi2_centre = self.width_fit["phi2_centre"] if phi2_centre is None else phi2_centre
        sigma = self.width_fit["sigma"] if sigma is None else sigma
        phi1_range = self._endpoints_phi1_range() if phi1_range is None else tuple(phi1_range)
        regions = stream_regions(phi1_range, phi2_centre, sigma, on_n_sigma=self.on_n_sigma,
                                 off_n_sigma=self.off_n_sigma)
        below, above = region_windows(regions)["off"]
        if below[2] < self.window[2] or above[3] > self.window[3] or phi1_range[0] < self.window[0] \
                or phi1_range[1] > self.window[1]:
            raise ValueError(f"The regions (phi1 {phi1_range}, phi2 {below[2]:.2f} to {above[3]:.2f} deg) leave the "
                             f"window {tuple(round(value, 2) for value in self.window)}: make it larger.")
        selected = self._stars_in_match_filter()
        stars = self.search.stars
        self._clear_results(("on_hess", "off_hess", "diff_hess"))
        self.regions = regions
        self.areas = region_area_deg2(regions, self.stream_frame, self.excluded_healpix_mask)
        self.phi1_profile = on_off_phi1_profile(
            stars["phi1"].to_numpy()[selected], stars["phi2"].to_numpy()[selected], regions, self.stream_frame,
            self.excluded_healpix_mask, phi1_range=self.window[:2], bin_deg=bin_deg,
        )
        return self

    def compute_hess(
        self, *, color_bins: Any = None, mag_bins: Any = None, color_index: int = 0
    ) -> CandidateInvestigation:
        """On, off and on - off Hess diagrams (``hess_diagram``, ``difference_hess``).

        Every prepared star of the on region and of the off bands
        (``region_windows``, ``in_phi12_window``), not only those in the
        match filter; colour ``reference_band - color_bands[color_index]`` of
        the match filter and magnitude ``reference_band``; densities per deg²
        with the areas of ``define_regions``.

        Parameters
        ----------
        color_bins, mag_bins : array_like, optional
            Bin edges, the same for on and off. Default: those of
            ``hess_diagram``.
        color_index : int, optional
            Colour band of the match filter.

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            ``define_regions`` not done.
        """
        self._require("regions", "define_regions")
        windows = region_windows(self.regions)
        stars = self.search.stars
        phi1, phi2 = stars["phi1"].to_numpy(), stars["phi2"].to_numpy()
        on = in_phi12_window(phi1, phi2, windows["on"])
        off = in_phi12_window(phi1, phi2, windows["off"][0]) | in_phi12_window(phi1, phi2, windows["off"][1])
        reference, *colors = match_filter_columns(self.search.match_filter)
        kwargs = {"color_bins": color_bins, "mag_bins": mag_bins}
        self.on_hess = hess_diagram(stars, (reference, colors[color_index]), mask=on, area_deg2=self.areas["on"],
                                    **kwargs)
        self.off_hess = hess_diagram(stars, (reference, colors[color_index]), mask=off,
                                     area_deg2=self.areas["off"], **kwargs)
        self.diff_hess = difference_hess(self.on_hess, self.off_hess)
        return self

    def compute_isochrone(
        self,
        *,
        age: float | None = None,
        metallicity: float | None = None,
        feh: float | None = None,
        distance_modulus: float | None = None,
        color_index: int = 0,
    ) -> CandidateInvestigation:
        """Isochrone track in the bands of the match filter (``isochrone_cmd_track``), in ``isochrone_track``.

        The colour of the Hess diagrams (``compute_hess`` with the same
        ``color_index``): the track can be drawn on them
        (``plot_candidate_hess``). Does not need the earlier steps.

        Parameters
        ----------
        age, metallicity, feh, distance_modulus : float, optional
            Default: those of the object (Z or [Fe/H] as given to it: ``feh``
            when set, else the Z of ``isochrone``). Given values are only used
            for this track, not stored as inputs; at most one of
            ``metallicity`` and ``feh``.
        color_index : int, optional
            Colour band of the match filter.

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            No distance modulus; both ``metallicity`` and ``feh``.
        """
        distance_modulus = self.distance_modulus if distance_modulus is None else distance_modulus
        if distance_modulus is None:
            raise ValueError("compute_isochrone needs a distance_modulus: give one, or set cand.distance_modulus.")
        if metallicity is not None and feh is not None:
            raise ValueError("Give metallicity (Z) or feh, not both.")
        if metallicity is None and feh is None:  # those of the object
            feh = self.feh
            metallicity = None if feh is not None else self.isochrone["metallicity"]
        self.isochrone_track = isochrone_cmd_track(
            distance_modulus, self.age if age is None else age, metallicity, feh=feh,
            match_filter=self.match_filter, color_index=color_index,
        )
        return self

    def run_all_steps(
        self, catalog: pd.DataFrame, *, footprint_mask: NDArray | None = None, nest: bool = False
    ) -> CandidateInvestigation:
        """Every step in order, with the inputs of the object.

        ``compute_sky_map``, ``prepare``, ``compute_zoom_map``,
        ``scan_distance_moduli``, ``fit_width``, ``define_regions``,
        ``compute_hess``, ``compute_isochrone``.

        Parameters
        ----------
        catalog : pandas.DataFrame
            Stars of a box containing the window. Not copied.
        footprint_mask : numpy.ndarray of bool, optional
            See ``prepare``.
        nest : bool, optional
            See ``prepare``.

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            ``distance_modulus`` not set.
        """
        raise NotImplementedError

    # ===================================================
    # Track and summary
    # ===================================================

    def track_phi12(self, n_points: int = DEFAULT_TRACK_POINTS) -> tuple[NDArray, NDArray]:
        """(phi1, phi2) of the straight track: phi2 = ``phi2_centre`` over ``phi1_range``; phi2 = 0 between the endpoints before ``define_regions``.

        Parameters
        ----------
        n_points : int, optional
            Number of points.

        Returns
        -------
        phi1, phi2 : numpy.ndarray
            (deg).
        """
        if self.regions is not None:
            phi1 = np.linspace(*self.regions["phi1_range"], n_points)
            return phi1, np.full(n_points, float(self.regions["phi2_centre"]))
        return np.linspace(*self._endpoints_phi1_range(), n_points), np.zeros(n_points)

    def track_radec(self, n_points: int = DEFAULT_TRACK_POINTS) -> tuple[NDArray, NDArray]:
        """(ra, dec) of ``track_phi12``, for sky maps.

        Parameters
        ----------
        n_points : int, optional
            Number of points.

        Returns
        -------
        ra, dec : numpy.ndarray
            (deg).
        """
        return phi12_to_radec(*self.track_phi12(n_points), self.stream_frame)

    def summary(self) -> dict[str, Any]:
        """Key numbers of the investigation, those computed so far.

        Returns
        -------
        summary : dict
            ``{"name", "endpoint_1", "endpoint_2", "window", "age",
            "metallicity", "feh", "distance_modulus", "distance_kpc",
            "phi2_centre", "sigma", "phi1_range", "area_on_deg2",
            "area_off_deg2", "n_on", "n_off", "n_excess"}``; None for the
            steps not done.
        """
        raise NotImplementedError

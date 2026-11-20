"""Investigation of one stream candidate: its regions and width, and one object holding every step.

Two layers, in this module:

- plain functions returning plain dicts, each callable alone and tunable:
  the on and off regions (``stream_regions``, ``region_windows``,
  ``stream_band_regions``), the densities (``phi12_density_map``,
  ``phi12_density_profile``, ``on_off_phi1_profile``), the on - off Hess
  diagram and the photometric errors as a table (``hess_error_table``); the width fit is in
  ``analysis.fit_stream`` (``fit_stream_width``, ``stream_width_model``);
- ``CandidateInvestigation``, which holds the inputs (two endpoints, the
  isochrone, the distance modulus, the region sizes) and the result of each
  step, and calls the functions in order. It adds no science: each method
  calls one or two functions of this module, ``stream_frames``,
  ``catalogs.survey_footprint``, ``analysis.fit_stream``, ``analysis.generic_stream_search``,
  ``analysis.stream_frame_search``, ``analysis.build_masks`` and
  ``analysis.cmd_match_filter``, stores the result and returns the object,
  so that calls can be chained. A function can also be called alone and its
  result put back in the object (e.g. ``cand.regions = stream_regions(...)``).

The track is straight for now: phi2 = ``phi2_centre`` over ``phi1_range``.
Its width is the sigma of a Gaussian plus a background (linear by default,
a constant or a polynomial of higher degree if needed), fitted to the phi2
density profile of the stars in the match filter between the endpoints
(counts and unmasked areas summed over that phi1 range), near the track
(``analysis.fit_stream.DEFAULT_FIT_HALF_WIDTH_DEG``). The on region is
``|phi2 - phi2_centre| <= on_n_sigma * sigma``; there are two off regions,
one below and one above the track, from ``off_n_sigma[0] * sigma`` to
``off_n_sigma[1] * sigma``; all over ``phi1_range``. The off density is the
mean of the densities of the two off regions (``mean_off_density``), so that
a linear field gradient across the track cancels even where the masks cut
one more than the other. The stars and areas of the regions are those of the
bands along a straight track of ``analysis.on_off_regions.track_band_regions``
(``stream_band_regions``), shared with the tuning on known objects and
streams; its ``selection_snr`` gives the excess and SNR of ``summary``. Each
region is also a phi1/phi2 window (``region_windows``), for the outlines and
the phi1 profile. Areas and densities are outside the excluded mask
(footprint OR masked objects, ``analysis.build_masks.excluded_healpix_mask``),
so that the stars, the maps and the areas see the same masks. With
``mask_streams`` (default), the other streams are excluded too: bands along
the galstreams tracks crossing the loaded box and along the other listed
candidates (``analysis.build_masks.streams_healpix_mask``), so that the width,
the regions, the SNR and the Hess diagrams of the candidate do not take their
stars; the maps (``search``) still show them.

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

Systematics, after ``define_regions`` (not in ``run_all_steps``):
``compute_photometric_errors``, the median photometric error of each band of
the stars of the on region per bin of its magnitude
(``analysis.cmd_match_filter.photometric_errors_by_magnitude``); after
``prepare``: ``compute_depth_dust_maps``, the depth maps of each band and
the dust map on the grid of the maps in the frame
(``catalogs.depth_dust_maps.depth_dust_phi12_maps``), to compare with them.

``run_all_steps`` does them all. The figures take the object
(``plotting.plot_candidate_investigation``); ``other_tracks_phi12`` gives
them the known streams and the other candidates in the window (e.g. the
distance-modulus scan GIF, ``plot_candidate_investigation.build_gif``).

References
----------
Shipp et al. 2018, ApJ 862, 114 (Gaussian width with a linear foreground;
on - off Hess diagrams). On region 1.4 sigma, off regions 3.5 to 6 sigma: user
choices, no reference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import healpy as hp
import numpy as np
import pandas as pd

from streameucliddes.analysis.build_masks import (
    CANDIDATE_MASK_MAX_DM_DIFFERENCE,
    STREAM_MASK_MAX_DM_DIFFERENCE,
    STREAM_MASK_MAX_WIDTH_DEG,
    candidate_mask_tracks,
    excluded_healpix_mask,
    fraction_on_stream_band,
    galstreams_mask_tracks,
    points_along_track,
    streams_healpix_mask,
    streams_near_distance_modulus,
    streams_not_wider_than,
)
from streameucliddes.analysis.cmd_match_filter import (
    DEFAULT_ERROR_STATISTIC,
    difference_hess,
    isochrone_cmd_track,
    match_filter_cmd_columns,
    mean_off_density,
    photometric_errors_by_magnitude,
    select_stars_in_match_filter,
)
from streameucliddes.analysis.fit_stream import DEFAULT_SIGMA_GUESS_DEG, fit_stream_width
from streameucliddes.analysis.generic_stream_search import generic_search_single_dm
from streameucliddes.analysis.on_off_regions import (
    DEFAULT_OFF_N_SIGMA,
    DEFAULT_ON_N_SIGMA,
    region_on_off_hess,
    selection_snr,
    track_band_regions,
)
from streameucliddes.analysis.stream_frame_search import (
    prepare_search,
    stream_frame_search_dm_scan,
    stream_frame_search_single_dm,
)
from streameucliddes.catalogs.depth_dust_maps import DEFAULT_DEPTH_QUANTITIES, depth_dust_phi12_maps
from streameucliddes.catalogs.euclid_catalog import DEFAULT_DUST_CORRECTION, LOADED_BOX_ATTR
from streameucliddes.catalogs.isochrone_parameters import DEFAULT_AGES_GYR, isochrone_from_feh
from streameucliddes.catalogs.stream_candidates import (
    BEST_DM_COLUMN,
    DEFAULT_CANDIDATES_PATH,
    candidate_best_dm,
    candidate_endpoints,
    get_candidate,
    load_candidates,
    write_candidates,
)
from streameucliddes.catalogs.survey_footprint import footprint_mask_from_stars, in_healpix_mask, unmasked_area_deg2
from streameucliddes.stream_frames import (
    DEFAULT_MIN_PHI2_HALF_WIDTH_DEG,
    DEFAULT_PADDING_PHI1_DEG,
    DEFAULT_PADDING_PHI2_DEG,
    phi12_to_radec,
    phi12_window_around,
    phi12_window_radec_box,
    radec_to_phi12,
    stream_frame_from_endpoints,
)
from streameucliddes.unit_conversions import convert_DM_to_kpc

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import astropy.coordinates as coord
    from numpy.typing import ArrayLike, NDArray

    from streameucliddes.analysis.stream_frame_search import PreparedSearch

LOGGER = logging.getLogger(__name__)

DEFAULT_PHI2_BIN_DEG = 0.05  # bins of the phi2 profile
DEFAULT_PHI1_BIN_DEG = 0.5  # bins of the phi1 profile
DEFAULT_WIDTH_MAP_BINS_DEG = (0.5, 0.2)  # (phi1, phi2) cells of the 2D map of the width fit
DEFAULT_FOOTPRINT_NSIDE = 512  # footprint from the stars (finest nside; coarser for sparse stars)
DEFAULT_TRACK_POINTS = 100
OTHER_TRACK_STEP_DEG = 0.05  # points along the other streams and candidates drawn on the maps (other_tracks_phi12)
# Figures of the candidates: figures/outputs/ of the repository (gitignored), one folder per candidate,
# "unnamed" for one given by its endpoints, its id for one of the list (from_candidate_id)
CANDIDATE_FIGURES_DIR = Path(__file__).resolve().parents[3] / "figures" / "outputs" / "candidates"
UNNAMED_FIGURE_FOLDER = "unnamed"
CROSS_CHECK_FOLDER = "cross_check"  # subfolder of figure_folder: the cross-checks (depth and dust maps)
CATALOG_EDGE_TOLERANCE_DEG = 0.1  # window outline beyond the outermost stars by less than this: still inside
# A stream whose band covers more of the track between the endpoints than this is not masked (warned): it is the
# candidate itself (e.g. one of the list investigated from its endpoints) or a stream along it
OWN_TRACK_OVERLAP_FRACTION = 0.5
# Steps of run_all_steps that take step_kwargs, in order (prepare, between compute_sky_map and compute_zoom_map, takes
# footprint_mask, nest and the prepare_kwargs of the object)
RUN_ALL_STEPS = ("compute_sky_map", "compute_zoom_map", "scan_distance_moduli", "fit_width", "define_regions",
                 "compute_hess", "compute_isochrone")
WINDOW_FIELDS = ("padding_phi1_deg", "padding_phi2_deg", "min_phi2_half_width_deg")
# Results that depend on the frame: cleared by prepare and set_endpoints (not sky_map nor isochrone_track)
FRAME_RESULTS = ("search", "footprint_and_objects_mask", "excluded_healpix_mask", "stars_not_excluded", "zoom_map",
                 "maps", "phi2_profile", "width_fit", "width_map", "regions", "band_regions", "areas", "phi1_profile",
                 "on_hess", "off_hess", "diff_hess", "photometric_errors", "null_test", "depth_dust_maps")
# Results computed from the distance modulus (the stars in the match filter, from fit_width on)
DISTANCE_MODULUS_RESULTS = ("phi2_profile", "width_fit", "width_map", "regions", "band_regions", "areas", "phi1_profile",
                            "on_hess", "off_hess", "diff_hess", "photometric_errors", "null_test", "isochrone_track")
REGION_RESULTS = ("regions", "band_regions", "areas", "phi1_profile", "on_hess", "off_hess", "diff_hess",
                  "photometric_errors", "null_test")
# The streams that can be masked, those masked and their mask: they depend on the footprint, not on the frame (kept
# by set_endpoints)
STREAM_MASK_RESULTS = ("stream_mask_tracks", "masked_streams", "overlapping_streams", "stream_healpix_mask")
# Results computed from the stars outside the excluded mask (the maps are not: they see every prepared star)
EXCLUDED_MASK_RESULTS = ("phi2_profile", "width_fit", "width_map", "regions", "band_regions", "areas", "phi1_profile",
                         "on_hess", "off_hess", "diff_hess", "photometric_errors", "null_test")
# Inputs of the stream mask: a change applies it again at once (_apply_stream_mask), no new prepare needed
STREAM_MASK_INPUTS = ("mask_streams", "stream_mask_max_dm_difference", "stream_mask_max_width_deg",
                      "always_masked_streams",
                      "unmasked_streams", "candidate_id", "candidates_path", "include_bad_candidates",
                      "candidate_mask_max_dm_difference")
# Results cleared when an input changes after a step: those computed from it (the maps and the zoom map carry their
# own distance modulus; prepare_search keeps the isochrone only as a default the candidate never uses)
INPUT_RESULTS = {
    "distance_modulus": DISTANCE_MODULUS_RESULTS,
    **dict.fromkeys(("age", "metallicity", "feh", "isochrone_model"),
                    ("sky_map", "zoom_map", "maps", *DISTANCE_MODULUS_RESULTS)),
    "match_filter": ("sky_map", *FRAME_RESULTS, "isochrone_track"),
    "prepare_kwargs": FRAME_RESULTS,
    **dict.fromkeys(("on_n_sigma", "off_n_sigma"), REGION_RESULTS),
    **dict.fromkeys(("mask_streams", "stream_mask_max_dm_difference", "stream_mask_max_width_deg",
                     "always_masked_streams"),
                    EXCLUDED_MASK_RESULTS),
    **dict.fromkeys(("unmasked_streams", "candidate_id", "candidates_path", "include_bad_candidates",
                     "candidate_mask_max_dm_difference"),
                    (*STREAM_MASK_RESULTS, *EXCLUDED_MASK_RESULTS)),
}


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
    on_phi2_offset_deg: float = 0.0,
    off_phi2_offsets_deg: tuple[float, float] = (0.0, 0.0),
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
        Inner and outer distances of the off regions to ``phi2_centre``, in
        units of ``sigma``.
    on_phi2_offset_deg : float, optional
        phi2 shift of the on region (deg), e.g. to move it off the stream as
        a null test.
    off_phi2_offsets_deg : tuple of float, optional
        phi2 shifts of the off region below and of the off region above (deg).

    Returns
    -------
    regions : dict
        ``{"phi1_range", "phi2_centre", "sigma", "on_n_sigma", "off_n_sigma",
        "on_phi2_offset_deg", "off_phi2_offsets_deg"}``.

    Raises
    ------
    ValueError
        An off region overlaps the on region, or ``off_n_sigma[1] <=
        off_n_sigma[0]``, or ``sigma <= 0``, or ``phi1_range`` not increasing.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, not {sigma}.")
    if not off_n_sigma[1] > off_n_sigma[0]:
        raise ValueError(f"off_n_sigma must be increasing, not {off_n_sigma}.")
    if not phi1_range[1] > phi1_range[0]:
        raise ValueError(f"phi1_range must be increasing, not {phi1_range}.")
    regions = {
        "phi1_range": (float(phi1_range[0]), float(phi1_range[1])), "phi2_centre": float(phi2_centre),
        "sigma": float(sigma), "on_n_sigma": float(on_n_sigma),
        "off_n_sigma": (float(off_n_sigma[0]), float(off_n_sigma[1])),
        "on_phi2_offset_deg": float(on_phi2_offset_deg),
        "off_phi2_offsets_deg": (float(off_phi2_offsets_deg[0]), float(off_phi2_offsets_deg[1])),
    }
    windows = region_windows(regions)
    on_min, on_max = windows["on"][2:]
    for off_min, off_max in (window[2:] for window in windows["off"]):
        if off_min < on_max and on_min < off_max:
            raise ValueError(f"An off region (phi2 {off_min:.2f} to {off_max:.2f} deg) overlaps the on region "
                             f"(phi2 {on_min:.2f} to {on_max:.2f} deg).")
    return regions


def region_windows(regions: dict[str, Any]) -> dict[str, Any]:
    """phi1/phi2 windows of the on region and of the two off regions.

    Computed from the regions each time, so that a region edited by hand
    stays consistent. Each window is shifted in phi2 by its offset
    (``on_phi2_offset_deg``, ``off_phi2_offsets_deg``; 0 when absent).

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
    on_offset = regions.get("on_phi2_offset_deg", 0.0)
    below_offset, above_offset = regions.get("off_phi2_offsets_deg", (0.0, 0.0))
    return {
        "on": (phi1_min, phi1_max, centre - on_half + on_offset, centre + on_half + on_offset),
        "off": ((phi1_min, phi1_max, centre - off_outer + below_offset, centre - off_inner + below_offset),
                (phi1_min, phi1_max, centre + off_inner + above_offset, centre + off_outer + above_offset)),
    }


def stream_band_regions(
    stars: pd.DataFrame,
    regions: dict[str, Any],
    stream_frame: coord.BaseCoordinateFrame,
    excluded_healpix_mask: NDArray[np.bool_] | None,
) -> dict[str, Any]:
    """Stars and areas of the regions: the bands along a straight track of ``on_off_regions.track_band_regions``.

    The track is at ``phi2_centre`` over ``phi1_range``; the bands are the
    windows of ``region_windows``, offsets included.

    Parameters
    ----------
    stars : pandas.DataFrame
        Stars, with ``ra`` and ``dec``. Not copied.
    regions : dict
        From ``stream_regions``.
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Frame of the regions.
    excluded_healpix_mask : numpy.ndarray of bool or None
        True where excluded, RING. None: nothing excluded.

    Returns
    -------
    band_regions : dict
        Those of ``track_band_regions`` (per star ``on``, ``off_below``,
        ``off_above``, ``off``; ``<region>_area_deg2``...), plus ``regions``
        (a copy of the regions they were built from).
    """
    band_regions = track_band_regions(
        stars, stream_frame, regions["phi2_centre"], regions["phi1_range"], regions["sigma"],
        on_n_sigma=regions["on_n_sigma"], off_n_sigma=regions["off_n_sigma"],
        on_phi2_offset_deg=regions.get("on_phi2_offset_deg", 0.0),
        off_phi2_offsets_deg=regions.get("off_phi2_offsets_deg", (0.0, 0.0)),
        excluded_healpix_mask=excluded_healpix_mask,
    )
    band_regions["regions"] = dict(regions)
    return band_regions


def region_areas(band_regions: Mapping[str, Any]) -> dict[str, float]:
    """``{"on", "off_below", "off_above", "off"}`` areas (deg²) of ``stream_band_regions``; ``"off"`` their sum."""
    return {name: band_regions[f"{name}_area_deg2"] for name in ("on", "off_below", "off_above", "off")}


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
    phi1 over the on region or the two off regions (``on_off_phi1_profile``).

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

    ``phi12_density_profile`` along phi1 of the on region and of the off
    regions below and above the track (``region_windows``, over
    ``phi1_range``); the off density of a bin is the mean of the two off
    regions (``mean_off_density``: the one with an area alone where the
    other has none); and the difference.

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
        "on_error", "off_error", "n_off_regions", "bin_deg", "regions"}``:
        densities in stars/deg²; ``error`` that of the difference (Poisson,
        in quadrature); ``n_off_regions`` the number of off regions in the
        mean of each bin: 1 where one has no area, and then a gradient
        across the track no longer cancels. Bins with no area in the on
        region or in both off regions have NaN.
    """
    if phi1_range is None:
        phi1_range = regions["phi1_range"]
    windows = region_windows({**regions, "phi1_range": tuple(phi1_range)})
    kwargs = {"axis": "phi1", "bin_deg": bin_deg, "axis_range": tuple(phi1_range)}
    on, below, above = (phi12_density_profile(phi1, phi2, [window], stream_frame, excluded_healpix_mask, **kwargs)
                        for window in (windows["on"], *windows["off"]))
    off, off_error, n_off_regions = mean_off_density(below["density"], below["error"], above["density"], above["error"])
    return {
        "centres": on["centres"], "edges": on["edges"], "on": on["density"], "off": off,
        "difference": on["density"] - off, "error": np.hypot(on["error"], off_error),
        "on_error": on["error"], "off_error": off_error, "n_off_regions": n_off_regions,
        "bin_deg": on["bin_deg"], "regions": dict(regions),
    }


def hess_error_table(
    on_hess: Mapping[str, Any],
    off_hess: Mapping[str, Any],
    photometric_errors: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """The Hess diagrams as a table, one row per bin, and the photometric errors per magnitude bin beside them.

    E.g. ``cand.on_hess``, ``cand.off_hess`` and ``cand.photometric_errors``,
    to save as CSV. Two blocks of columns of their own lengths: the Hess
    diagrams (one row per colour and magnitude bin) and the errors (one row
    per magnitude bin of the errors, which can differ from those of the
    diagrams), each from the first row; the shorter block is NaN below its
    end. Plot ``<statistic>_error_<band>`` against ``error_mag_centre_<band>``
    (the error of a band against its own magnitude). The densities are
    per deg²: times an area, the stars expected there (the areas of the
    regions, on every row, give back their stars).

    Parameters
    ----------
    on_hess, off_hess : mapping
        From ``compute_hess`` (``cand.on_hess``, ``cand.off_hess``): the on
        diagram and the off diagram (the mean of the two off regions), with
        the same bins.
    photometric_errors : mapping, optional
        From ``analysis.cmd_match_filter.photometric_errors_by_magnitude``
        (``cand.photometric_errors``), on any magnitude bins.

    Returns
    -------
    table : pandas.DataFrame
        Colour bin by colour bin, magnitude bin by magnitude bin:
        ``color_centre``, ``mag_centre`` (mag), ``difference_density`` (on -
        off, ``analysis.cmd_match_filter.difference_hess``), ``on_density``
        and ``off_density`` (stars/deg²/bin; off: the mean of the densities
        of the two off regions), and the areas of the regions outside the
        masks, the same on every row, ``area_on_deg2`` and ``area_off_deg2``
        (the two off regions together: when their areas differ,
        ``off_density * area_off_deg2`` is not exactly their number of
        stars). With ``photometric_errors``, beside, for each band (e.g. VIS,
        then H): ``error_mag_centre_<band>`` (the bin centres in that band's
        magnitude, mag) and ``<statistic>_error_<band>`` (its error, e.g.
        median, mag).

    Raises
    ------
    ValueError
        The on and off diagrams on different bins.
    """
    diff_hess = difference_hess(on_hess, off_hess)
    color_edges, mag_edges = (np.asarray(diff_hess[name], dtype=float) for name in ("color_edges", "mag_edges"))
    color_centres, mag_centres = (color_edges[:-1] + color_edges[1:]) / 2, (mag_edges[:-1] + mag_edges[1:]) / 2
    color_grid, mag_grid = np.meshgrid(color_centres, mag_centres, indexing="ij")  # (n_color, n_mag), as density
    table = pd.DataFrame({
        "color_centre": color_grid.ravel(), "mag_centre": mag_grid.ravel(),
        "difference_density": np.asarray(diff_hess["density"], dtype=float).ravel(),
        "on_density": np.asarray(on_hess["density"], dtype=float).ravel(),
        "off_density": np.asarray(off_hess["density"], dtype=float).ravel(),
        "area_on_deg2": float(on_hess["area_deg2"]), "area_off_deg2": float(off_hess["area_deg2"]),
    })
    if photometric_errors is None:
        return table
    statistic, centres = photometric_errors["statistic"], np.asarray(photometric_errors["mag_centres"], dtype=float)
    errors = pd.DataFrame({
        name: values for band in photometric_errors["bands"]
        for name, values in ((f"error_mag_centre_{band}", centres),
                             (f"{statistic}_error_{band}", photometric_errors["error"][band]))
    })
    return pd.concat([table, errors], axis=1)  # side by side from the first row; NaN below the shorter


# ===================================================
# The investigation
# ===================================================


def _changed(old: Any, new: Any) -> bool:
    """Whether a new input value differs from the old one (True when they cannot be compared)."""
    try:
        return bool(old != new)
    except (TypeError, ValueError):
        return True


@dataclass
class CandidateInvestigation:
    """All inputs and results of one stream-candidate investigation.

    Mutable: change a field and re-run the later steps. Results are None
    until their step is done. A candidate of the list
    (``catalogs.stream_candidates``): ``from_candidate_id``.

    Parameters
    ----------
    name : str
        Name of the candidate, for titles.
    endpoint_1, endpoint_2 : tuple of float
        (ra, dec) of two points along the stream (deg).
    age : float, optional
        Isochrone age (Gyr).
    metallicity : float, optional
        Isochrone Z (on the scale of ``isochrone_model``). At most one of
        ``metallicity`` and ``feh``; neither: the Z of
        ``DEFAULT_FEHS[0]`` of ``catalogs.isochrone_parameters`` in the model.
    feh : float, optional
        Isochrone [Fe/H], converted to Z on the scale of the isochrone model
        (``unit_conversions.convert_FeH_to_z``) each time the isochrone is used
        (``isochrone``).
    isochrone_model : str, optional
        Name of the ugali isochrone model of the match filter and of the
        isochrone track, e.g. ``"Dotter2016"``. None:
        ``unit_conversions.get_isochrone_model()`` (``DEFAULT_ISOCHRONE_MODEL``).
    distance_modulus : float, optional
        Distance modulus of the stream, chosen by eye after the scan
        (``scan_distance_moduli``). Needed from ``fit_width`` on.
    distance_moduli : sequence of float, optional
        Grid of the scan. Default: that of ``prepare_search``.
    match_filter : dict, optional
        See ``analysis.cmd_match_filter``. Default: ``get_match_filter()`` (``DEFAULT_MATCH_FILTER``).
    padding_phi1_deg, padding_phi2_deg, min_phi2_half_width_deg : float, optional
        Window around the endpoints (``stream_frames.phi12_window_around``).
    on_n_sigma : float, optional
        Half-width of the on region, in units of sigma.
    off_n_sigma : tuple of float, optional
        Distances of the off regions, in units of sigma.
    prepare_kwargs : dict, optional
        More arguments of ``prepare_search`` (``bin_size_deg``, ``cuts``,
        ``mask_*``...).
    figure_folder : str or pathlib.Path, optional
        Folder of the figures of the candidate. Default:
        ``CANDIDATE_FIGURES_DIR / "unnamed"`` (``figures/outputs/candidates/unnamed``);
        from ``from_candidate_id``, ``CANDIDATE_FIGURES_DIR / "<id>"``. Not
        created here. The figures of the cross-checks go to its subfolder
        ``cross_check_folder``.
    mask_streams : bool, optional
        Exclude the other streams (``excluded_healpix_mask``): the galstreams
        tracks crossing the footprint of the catalog
        (``galstreams_mask_tracks``: one per stream, no Ibata et al. 2024
        track, none of status ``not_masked`` in ``data/known_streams/``:
        ``catalogs.galstreams_catalog.load_known_streams``) and the listed
        candidates but this one
        (``candidate_mask_tracks``) not wider than
        ``stream_mask_max_width_deg`` (``build_masks.streams_not_wider_than``:
        a broad stream is a smooth background over the regions, subtracted
        by the off regions) and whose distance modulus comes within
        ``stream_mask_max_dm_difference`` of ``distance_modulus``
        (``candidate_mask_max_dm_difference`` for the candidates, from their
        best distance modulus), or whatever their distance modulus for those
        of ``always_masked_streams`` (but still not wider than
        ``stream_mask_max_width_deg``;
        ``build_masks.streams_near_distance_modulus``; all of them while
        ``distance_modulus`` is None), but those covering more than
        ``OWN_TRACK_OVERLAP_FRACTION`` of the track between the endpoints
        (the candidate itself, or a stream along it: ``overlapping_streams``,
        warned), over ``STREAM_MASK_N_SIGMA`` times
        their width, when unknown ``DEFAULT_STREAM_MASK_HALF_WIDTH_DEG`` for
        the known streams and ``CANDIDATE_MASK_HALF_WIDTH_DEG`` for the
        candidates (``build_masks.streams_healpix_mask``). A change of
        ``distance_modulus`` or of an input of the stream mask applies it
        again at once (no new ``prepare``).
    stream_mask_max_dm_difference, stream_mask_max_width_deg : float, optional
        See ``mask_streams``.
    candidate_mask_max_dm_difference : float, optional
        The other listed candidates are masked only when their best
        distance modulus (``candidate_best_dm``) comes within this of
        ``distance_modulus``; see ``mask_streams``.
    unmasked_streams : sequence of str, optional
        Names of streams not masked (a track name, a stream name or
        ``"candidate_<id>"``), e.g. the known stream the candidate is.
    always_masked_streams : sequence of str, optional
        Names of streams (as ``unmasked_streams``) masked whatever their
        distance modulus: one far from that of the candidate whose stars
        could still leak into its Hess diagrams, e.g. a known stream seen in
        the maps (``catalogs.galstreams_catalog.observed_known_streams``) or
        another candidate.
    candidate_id : int, optional
        Id of the candidate in the list, left out of the stream mask; set by
        ``from_candidate_id``.
    candidates_path : str or pathlib.Path, optional
        CSV list of the candidates masked. Default: ``DEFAULT_CANDIDATES_PATH``.
    include_bad_candidates : bool, optional
        Mask and draw (``other_tracks_phi12``) the other candidates of
        status ``bad`` too (``load_candidates``); by default they are ignored.

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
        Footprint OR masked objects, RING (``build_masks.excluded_healpix_mask``),
        OR ``stream_healpix_mask`` with ``mask_streams``.
    footprint_and_objects_mask : numpy.ndarray of bool
        Footprint OR masked objects, RING: ``excluded_healpix_mask`` without
        the streams.
    stream_mask_tracks : list of dict
        The tracks of the streams that can be masked (galstreams crossing the
        footprint and the other candidates), built once per footprint.
    masked_streams : list of str
        Names of those masked (near ``distance_modulus``, or in
        ``always_masked_streams``).
    overlapping_streams : list of str
        Names of those not masked because they cover the track of the
        candidate: their stars stay in its regions.
    stream_healpix_mask : numpy.ndarray of bool
        Their mask, RING, at the nside of ``excluded_healpix_mask``.
    stars_not_excluded : numpy.ndarray of bool
        True for the stars of ``search.stars`` outside ``excluded_healpix_mask``:
        those counted in the regions, whose areas leave the mask out
        (``prepare_search`` drops the stars on masked objects, not those
        outside the footprint).
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
    regions, phi1_profile : dict
        ``stream_regions`` and ``on_off_phi1_profile`` results.
    band_regions, areas : dict
        The stars and areas of the regions (``stream_band_regions``, built
        again when ``regions`` is changed by hand) and their areas
        (``region_areas``).
    photometric_errors : dict
        ``photometric_errors_by_magnitude`` of the stars of the on region
        (``compute_photometric_errors``).
    on_hess, off_hess, diff_hess : dict
        ``hess_diagram`` and ``difference_hess`` results.
    null_test : dict
        ``compute_null_test`` result: the regions moved off the stream, their
        Hess diagrams and SNR.
    isochrone_track : dict
        ``isochrone_cmd_track`` result.
    depth_dust_maps : dict
        ``depth_dust_phi12_maps`` result (``compute_depth_dust_maps``).
    """

    # Inputs
    name: str
    endpoint_1: tuple[float, float]
    endpoint_2: tuple[float, float]
    age: float = DEFAULT_AGES_GYR[0]
    metallicity: float | None = None
    feh: float | None = None
    isochrone_model: str | None = None
    distance_modulus: float | None = None
    distance_moduli: Sequence[float] | None = None
    match_filter: dict | None = None
    padding_phi1_deg: float = DEFAULT_PADDING_PHI1_DEG
    padding_phi2_deg: float = DEFAULT_PADDING_PHI2_DEG
    min_phi2_half_width_deg: float = DEFAULT_MIN_PHI2_HALF_WIDTH_DEG
    on_n_sigma: float = DEFAULT_ON_N_SIGMA
    off_n_sigma: tuple[float, float] = DEFAULT_OFF_N_SIGMA
    prepare_kwargs: dict = field(default_factory=dict)
    figure_folder: str | Path | None = None
    mask_streams: bool = True
    stream_mask_max_dm_difference: float = STREAM_MASK_MAX_DM_DIFFERENCE
    candidate_mask_max_dm_difference: float = CANDIDATE_MASK_MAX_DM_DIFFERENCE
    stream_mask_max_width_deg: float = STREAM_MASK_MAX_WIDTH_DEG
    unmasked_streams: Sequence[str] = ()
    always_masked_streams: Sequence[str] = ()
    candidate_id: int | None = None
    candidates_path: str | Path | None = None
    include_bad_candidates: bool = False
    # Results, None until computed
    stream_frame: Any = None
    window: tuple | None = None
    radec_box: tuple | None = None
    catalog: Any = None
    footprint_mask: NDArray | None = None
    footprint_nest: bool = False
    search: PreparedSearch | None = None
    excluded_healpix_mask: NDArray | None = None
    stars_not_excluded: NDArray | None = None
    footprint_and_objects_mask: NDArray | None = None
    stream_mask_tracks: list | None = None
    masked_streams: list | None = None
    overlapping_streams: list | None = None
    stream_healpix_mask: NDArray | None = None
    sky_map: dict | None = None
    zoom_map: dict | None = None
    maps: dict | None = None
    phi2_profile: dict | None = None
    width_fit: dict | None = None
    width_map: dict | None = None
    regions: dict | None = None
    band_regions: dict | None = None
    areas: dict | None = None
    phi1_profile: dict | None = None
    on_hess: dict | None = None
    off_hess: dict | None = None
    diff_hess: dict | None = None
    null_test: dict | None = None
    photometric_errors: dict | None = None
    isochrone_track: dict | None = None
    depth_dust_maps: dict | None = None

    def __post_init__(self) -> None:
        """Frame, window and (ra, dec) box from the endpoints (``stream_frames``).

        Raises
        ------
        ValueError
            Both ``metallicity`` and ``feh`` given.
        """
        _ = self.isochrone  # raises if both metallicity and feh are set
        if self.figure_folder is None:
            self.figure_folder = CANDIDATE_FIGURES_DIR / UNNAMED_FIGURE_FOLDER
        self._set_frame_and_window()
        object.__setattr__(self, "_cleared_by", {})  # result name: the input change that cleared it (_require)
        object.__setattr__(self, "_track_inputs", True)  # from now on, a changed input clears its results

    def __setattr__(self, name: str, value: Any) -> None:
        """Set a field; a changed input of ``INPUT_RESULTS`` clears the results computed from it.

        E.g. a new ``distance_modulus`` after ``define_regions`` clears the
        width, the regions and the Hess diagrams: ``summary`` and the plots
        never mix results of different inputs (a step asking for a cleared
        result names the change). The endpoints and the window change with
        ``set_endpoints``. A changed ``distance_modulus`` (with
        ``mask_streams``) or input of the stream mask (``STREAM_MASK_INPUTS``)
        applies the stream mask again, once the stars are prepared.
        """
        cleared_by = self.__dict__.get("_cleared_by")
        changed = bool(self.__dict__.get("_track_inputs")) and name in INPUT_RESULTS and _changed(
            self.__dict__.get(name), value)
        if changed:
            old = self.__dict__.get(name)
            change = f"cand.{name} was changed" + (
                f" ({old} -> {value})" if all(v is None or isinstance(v, (int, float)) for v in (old, value)) else "")
            for result in INPUT_RESULTS[name]:
                if self.__dict__.get(result) is not None:
                    cleared_by[result] = change
            self._clear_results(INPUT_RESULTS[name])
        elif cleared_by is not None and value is not None:
            cleared_by.pop(name, None)  # computed again
        object.__setattr__(self, name, value)
        if changed and self.footprint_and_objects_mask is not None and (
                name in STREAM_MASK_INPUTS or (name == "distance_modulus" and self.mask_streams)):
            self._apply_stream_mask()

    @classmethod
    def from_candidate_id(
        cls,
        candidate_id: int,
        *,
        path: str | Path = DEFAULT_CANDIDATES_PATH,
        name: str | None = None,
        **inputs: Any,
    ) -> CandidateInvestigation:
        """Investigation of a candidate of the list (``catalogs.stream_candidates``), from its id.

        Its endpoints and its distance modulus (``candidate_best_dm``:
        ``best_dm``, the one of its largest SNR, else ``best_dm_by_eye``) come
        from the list, its figures go to ``CANDIDATE_FIGURES_DIR / "<id>"``
        (``figure_folder``); it is left out of the stream mask
        (``candidate_id``), the other candidates of ``path`` masked
        (``candidates_path``); the other inputs are those of the class. For a candidate
        not in the list, give the endpoints to the class itself.

        Parameters
        ----------
        candidate_id : int
            Its ``id`` in the list.
        path : str or pathlib.Path, optional
            CSV file of the list.
        name : str, optional
            Default: ``"candidate_<id>"``.
        **inputs
            Other inputs of the class (``age``, ``metallicity`` or ``feh``,
            ``match_filter``, ``distance_moduli``...); a ``distance_modulus``
            or a ``figure_folder`` given here replaces that of the list.

        Returns
        -------
        cand : CandidateInvestigation

        Raises
        ------
        ValueError
            No candidate with this id (``get_candidate``).
        """
        candidate = get_candidate(candidate_id, path)
        endpoint_1, endpoint_2 = candidate_endpoints(candidate)
        inputs.setdefault("distance_modulus", candidate_best_dm(candidate))
        inputs.setdefault("figure_folder", CANDIDATE_FIGURES_DIR / str(int(candidate["id"])))
        inputs.setdefault("candidate_id", int(candidate["id"]))
        inputs.setdefault("candidates_path", path)
        return cls(f"candidate_{int(candidate['id'])}" if name is None else name, endpoint_1, endpoint_2, **inputs)

    @property
    def isochrone(self) -> dict[str, float]:
        """``{"age", "metallicity"}`` (and ``"isochrone_model"``) of the match filter and of the isochrone track.

        Z from ``feh`` when given (on the scale of ``isochrone_model``), else
        ``metallicity``, else that of ``DEFAULT_FEHS[0]``
        (``catalogs.isochrone_parameters.isochrone_from_feh``). Computed each
        time: changing ``age``, ``metallicity``, ``feh`` or
        ``isochrone_model`` changes it.

        Raises
        ------
        ValueError
            Both ``metallicity`` and ``feh`` set.
        """
        return isochrone_from_feh(self.age, feh=self.feh, metallicity=self.metallicity,
                                  isochrone_model=self.isochrone_model)

    @property
    def cross_check_folder(self) -> Path:
        """Folder of the figures of the cross-checks (e.g. the depth and dust maps): ``figure_folder / CROSS_CHECK_FOLDER``.

        ``figures/outputs/candidates/<id>/cross_check`` for a candidate of the
        list. Not created here.
        """
        return Path(self.figure_folder) / CROSS_CHECK_FOLDER

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
            self._clear_results(STREAM_MASK_RESULTS)
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
        ``prepare_search`` (``build_masks.excluded_healpix_mask``, RING), OR
        the other streams with ``mask_streams``. The results of the later
        steps are cleared.

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
            (``footprint_mask_from_stars``: ``survey_footprint_healpix_mask`` at
            ``DEFAULT_FOOTPRINT_NSIDE``, coarser if the pixels hold too few
            stars: ``star_count_nside``), less its edge pixels
            (``trim_footprint_edge``: partly covered,
            they would count their whole area). A given footprint is used as
            it is.
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
            self._clear_results(STREAM_MASK_RESULTS)
        self.catalog = catalog
        if footprint_mask is not None:
            self.footprint_mask, self.footprint_nest = np.asarray(footprint_mask, dtype=bool), nest
            self._clear_results(STREAM_MASK_RESULTS)
        elif self.footprint_mask is None:
            # coarser for a sparse catalog; the partly covered edge pixels left out
            self.footprint_mask = footprint_mask_from_stars(catalog["ra"].to_numpy(), catalog["dec"].to_numpy(),
                                                            DEFAULT_FOOTPRINT_NSIDE)
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
        """``prepare_search`` on the kept catalog and footprint, the excluded mask and the stars outside it;
        later results cleared.

        ``object_healpix_mask``: the object mask of a previous search, not built again.
        """
        self._clear_results(FRAME_RESULTS)
        self.search = prepare_search(
            self.catalog, self.stream_frame, self.window, match_filter=self.match_filter, isochrone=self.isochrone,
            distance_moduli=self.distance_moduli, footprint_mask=self.footprint_mask, nest=self.footprint_nest,
            object_healpix_mask=object_healpix_mask, **self.prepare_kwargs,
        )
        self.footprint_and_objects_mask, _ = excluded_healpix_mask(
            self.footprint_mask, self.search.object_healpix_mask, nest=self.footprint_nest
        )
        self._apply_stream_mask()

    def _apply_stream_mask(self) -> None:
        """``excluded_healpix_mask``: ``footprint_and_objects_mask``, OR the streams near ``distance_modulus`` with
        ``mask_streams``; and ``stars_not_excluded``."""
        mask = self.footprint_and_objects_mask
        if self.mask_streams:
            mask = mask | self._stream_healpix_mask(hp.npix2nside(mask.size))
        else:
            self.masked_streams = self.overlapping_streams = None
        self.excluded_healpix_mask = mask
        stars = self.search.stars
        self.stars_not_excluded = ~in_healpix_mask(stars["ra"].to_numpy(), stars["dec"].to_numpy(), mask, nest=False)

    def _stream_healpix_mask(self, nside: int) -> NDArray[np.bool_]:
        """Mask of the other streams near ``distance_modulus``, and of those of ``always_masked_streams``, at ``nside``
        (RING); the tracks kept, the mask built again when the streams masked or the nside change."""
        if self.stream_mask_tracks is None:
            tracks = galstreams_mask_tracks(self.footprint_mask, nest=self.footprint_nest)
            tracks += self._other_candidate_tracks()
            unmasked = set(self.unmasked_streams)
            self.stream_mask_tracks = [track for track in tracks
                                       if track["name"] not in unmasked and track["stream_name"] not in unmasked]
        narrow = streams_not_wider_than(self.stream_mask_tracks, self.stream_mask_max_width_deg)
        near = streams_near_distance_modulus(narrow, self.distance_modulus,
                                             max_dm_difference=self.stream_mask_max_dm_difference)
        near_names, always = {track["name"] for track in near}, set(self.always_masked_streams)
        near += [track for track in narrow if track["name"] not in near_names
                 and (track["name"] in always or track["stream_name"] in always)]
        phi1 = np.linspace(*self._endpoints_phi1_range(), DEFAULT_TRACK_POINTS)
        own_ra, own_dec = phi12_to_radec(phi1, np.zeros(phi1.size), self.stream_frame)
        masked, overlapping = [], []
        for track in near:
            covers = fraction_on_stream_band(own_ra, own_dec, track) > OWN_TRACK_OVERLAP_FRACTION
            (overlapping if covers else masked).append(track)
        overlapping_names = [track["name"] for track in overlapping]
        if overlapping_names and overlapping_names != self.overlapping_streams:
            LOGGER.warning("%s: not masked, they cover more than %.0f %% of the track between the endpoints: %s (the "
                           "candidate itself, or a stream along it, whose stars stay in the regions).", self.name,
                           100 * OWN_TRACK_OVERLAP_FRACTION, ", ".join(overlapping_names))
        self.overlapping_streams = overlapping_names
        names = [track["name"] for track in masked]
        if (self.stream_healpix_mask is None or self.stream_healpix_mask.size != hp.nside2npix(nside)
                or names != self.masked_streams):
            self.stream_healpix_mask = streams_healpix_mask(masked, nside, nest=False)
            self.masked_streams = names
        return self.stream_healpix_mask

    def _other_candidate_tracks(self) -> list[dict]:
        """The listed candidates of ``candidates_path`` but this one (``candidate_id``) and, unless
        ``include_bad_candidates``, the bad ones, as tracks (``candidate_mask_tracks``)."""
        path = DEFAULT_CANDIDATES_PATH if self.candidates_path is None else self.candidates_path
        return candidate_mask_tracks(load_candidates(path, include_bad_candidates=self.include_bad_candidates),
                                     exclude_ids=() if self.candidate_id is None else (self.candidate_id,),
                                     max_dm_difference=self.candidate_mask_max_dm_difference)

    def _endpoints_phi1_range(self) -> tuple[float, float]:
        """phi1 of endpoint_1 and endpoint_2 (deg): negative, then positive."""
        phi1, _ = radec_to_phi12(
            [self.endpoint_1[0], self.endpoint_2[0]], [self.endpoint_1[1], self.endpoint_2[1]], self.stream_frame
        )
        return float(phi1[0]), float(phi1[1])

    def _band_regions(self) -> dict[str, Any]:
        """The stars and areas of ``regions`` (``stream_band_regions``), built again when ``regions`` changed.

        Also sets ``areas``. ``regions`` can so be edited by hand after
        ``define_regions`` (e.g. ``cand.regions = stream_regions(...)``).
        """
        self._require("regions", "define_regions")
        if self.band_regions is None or self.band_regions["regions"] != self.regions:
            self.band_regions = stream_band_regions(self.search.stars, self.regions, self.stream_frame,
                                                    self.excluded_healpix_mask)
            self.areas = region_areas(self.band_regions)
        return self.band_regions

    def stars_in_match_filter(self, distance_modulus: float | None = None) -> NDArray[np.bool_]:
        """Prepared stars outside the excluded mask, in the match filter at ``distance_modulus``
        (``select_stars_in_match_filter``); default: that of the object."""
        distance_modulus = self.distance_modulus if distance_modulus is None else distance_modulus
        if distance_modulus is None:
            raise ValueError("No distance_modulus: set cand.distance_modulus, e.g. from the maps of "
                             "scan_distance_moduli.")
        return self.stars_not_excluded & select_stars_in_match_filter(
            self.search.stars, distance_modulus, self.isochrone, self.search.match_filter
        )

    def _require(self, field_name: str, step: str) -> None:
        """ValueError naming the missing step when ``field_name`` is not computed, and the input change that cleared
        it, if any."""
        if getattr(self, field_name) is None:
            cleared_by = self.__dict__.get("_cleared_by", {}).get(field_name)
            if cleared_by is not None:
                raise ValueError(f"No {field_name}: cleared because {cleared_by}; call {step}() again (and the steps "
                                 "before it that it needs).")
            raise ValueError(f"No {field_name}: call {step}() first.")

    def _clear_results(self, names: Sequence[str]) -> None:
        """Set these results back to None."""
        for name in names:
            setattr(self, name, None)

    def _check_window_in_catalog(self, catalog: pd.DataFrame) -> None:
        """ValueError when the window leaves the (ra, dec) range of the catalog.

        Range of the stars around the ra of the window centre (no jump at ra =
        0/360); window outline from ``phi12_window_radec_box`` without margin.
        A window containing a pole is not checked. When the catalog says in
        which box it was read (``attrs[euclid_catalog.LOADED_BOX_ATTR]``, set
        by ``get_euclid_stars``), that box is the range instead of the stars,
        on the sides it limits: a window beyond the stars but inside the box
        reaches past the survey edge, where there is no star to load (the
        footprint, from the stars, masks it).
        """
        (ra_min, ra_max), (dec_min, dec_max) = phi12_window_radec_box(self.stream_frame, self.window, margin_deg=0.0)
        if (ra_min, ra_max) == (0.0, 360.0):
            return
        centre_ra = (ra_min + ((ra_max - ra_min) % 360.0) / 2) % 360.0

        def relative(ra: Any) -> Any:
            return (np.asarray(ra) - centre_ra + 180.0) % 360.0 - 180.0

        star_ra = relative(catalog["ra"].to_numpy())
        star_dec = catalog["dec"].to_numpy()
        (ra_low, ra_high), (dec_low, dec_high) = (star_ra.min(), star_ra.max()), (star_dec.min(), star_dec.max())
        what = "stars"
        box_ra, box_dec = catalog.attrs.get(LOADED_BOX_ATTR) or (None, None)
        if box_ra is not None:
            full_circle = box_ra[1] - box_ra[0] >= 360.0
            ra_low, ra_high = (-np.inf, np.inf) if full_circle else (relative(box_ra[0]), relative(box_ra[1]))
            what = "loaded box"
        if box_dec is not None:
            (dec_low, dec_high), what = box_dec, "loaded box"
        tolerance = CATALOG_EDGE_TOLERANCE_DEG
        if (relative(ra_min) < ra_low - tolerance or relative(ra_max) > ra_high + tolerance
                or dec_min < dec_low - tolerance or dec_max > dec_high + tolerance):
            ra_text = ("every ra" if np.isinf(ra_low) else
                       f"ra {(ra_low + centre_ra) % 360:.1f} to {(ra_high + centre_ra) % 360:.1f}")
            raise ValueError(
                f"The window (ra {ra_min:.1f} to {ra_max:.1f}, dec {dec_min:.1f} to {dec_max:.1f} deg) leaves the "
                f"{what} ({ra_text}, dec {dec_low:.1f} to {dec_high:.1f} deg): load a larger box."
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

    def snr_against_distance_modulus(self, distance_moduli: Sequence[float] | None = None) -> pd.DataFrame:
        """SNR of the regions at each distance modulus, the regions fixed: where the candidate peaks.

        At each distance modulus, the stars outside the excluded mask in the
        match filter there (``stars_in_match_filter``), and their excess on
        the on region over the mean density of the off regions, over the
        square root of the background on the on region
        (``on_off_regions.selection_snr``, as ``summary``). The regions are
        those of ``define_regions`` (the same at every distance modulus);
        ``distance_modulus`` is not changed.

        Parameters
        ----------
        distance_moduli : sequence of float, optional
            Default: ``self.distance_moduli``, else the grid of the search
            (``prepare_search``).

        Returns
        -------
        snr : pandas.DataFrame
            One row per distance modulus: ``distance_modulus``, ``n_on``,
            ``n_off``, ``n_excess``, ``n_excess_error``, ``snr``.

        Raises
        ------
        ValueError
            ``prepare`` or ``define_regions`` not done.
        """
        self._require("search", "prepare")
        band_regions = self._band_regions()
        if distance_moduli is None:
            distance_moduli = self.search.distance_moduli if self.distance_moduli is None else self.distance_moduli
        rows = []
        for distance_modulus in distance_moduli:
            snr = selection_snr(band_regions, self.stars_in_match_filter(float(distance_modulus)))
            rows.append({"distance_modulus": float(distance_modulus), "n_on": snr["n_on"], "n_off": snr["n_off"],
                         "n_excess": snr["n_signal"], "n_excess_error": snr["n_signal_error"], "snr": snr["snr"]})
        return pd.DataFrame(rows)

    def fit_width(
        self,
        phi1_range: tuple[float, float] | None = None,
        *,
        bin_deg: float = DEFAULT_PHI2_BIN_DEG,
        map_bins_deg: tuple[float, float] = DEFAULT_WIDTH_MAP_BINS_DEG,
        **fit_kwargs: Any,
    ) -> CandidateInvestigation:
        """phi2 density profile of the stars in the match filter, and its Gaussian width.

        Stars of ``search.stars`` outside the excluded mask
        (``stars_not_excluded``) in the match filter at
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
        selected = self.stars_in_match_filter()
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
        on_phi2_offset_deg: float = 0.0,
        off_phi2_offsets_deg: tuple[float, float] = (0.0, 0.0),
        bin_deg: float = DEFAULT_PHI1_BIN_DEG,
    ) -> CandidateInvestigation:
        """On and off regions, their areas and the phi1 profile.

        ``stream_regions`` (with ``on_n_sigma`` and ``off_n_sigma`` of the
        object), ``stream_band_regions`` (its stars and areas, outside
        ``excluded_healpix_mask``; ``areas``) and
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
        on_phi2_offset_deg, off_phi2_offsets_deg : optional
            phi2 shifts of the on region and of the off regions (below, above)
            (deg), see ``stream_regions``; e.g. on region and top off region
            moved off the stream as a null test.
        bin_deg : float, optional
            Bins of the phi1 profile (deg).

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            ``prepare`` not done; ``phi2_centre`` or ``sigma`` not given and
            ``fit_width`` not done; ``distance_modulus`` not set; or the
            regions leave the window (make it wider: ``min_phi2_half_width_deg``).
        """
        self._require("search", "prepare")
        if (phi2_centre is None or sigma is None) and self.width_fit is None:
            cleared_by = self.__dict__.get("_cleared_by", {}).get("width_fit")
            reason = "" if cleared_by is None else f" (cleared because {cleared_by})"
            raise ValueError(f"No width_fit{reason}: call fit_width() first, or give phi2_centre and sigma.")
        phi2_centre = self.width_fit["phi2_centre"] if phi2_centre is None else phi2_centre
        sigma = self.width_fit["sigma"] if sigma is None else sigma
        phi1_range = self._endpoints_phi1_range() if phi1_range is None else tuple(phi1_range)
        regions = stream_regions(phi1_range, phi2_centre, sigma, on_n_sigma=self.on_n_sigma,
                                 off_n_sigma=self.off_n_sigma, on_phi2_offset_deg=on_phi2_offset_deg,
                                 off_phi2_offsets_deg=off_phi2_offsets_deg)
        outside = self._regions_outside_window(regions)
        if outside:
            raise ValueError(f"The regions ({outside}) leave the window "
                             f"{tuple(round(value, 2) for value in self.window)}: make it larger.")
        selected = self.stars_in_match_filter()
        stars = self.search.stars
        self._clear_results(("on_hess", "off_hess", "diff_hess", "null_test"))
        self.regions = regions
        self._band_regions()
        self.phi1_profile = on_off_phi1_profile(
            stars["phi1"].to_numpy()[selected], stars["phi2"].to_numpy()[selected], regions, self.stream_frame,
            self.excluded_healpix_mask, phi1_range=self.window[:2], bin_deg=bin_deg,
        )
        return self

    def compute_hess(
        self, *, color_bins: Any = None, mag_bins: Any = None, color_index: int = 0
    ) -> CandidateInvestigation:
        """On, off and on - off Hess diagrams (``hess_diagram``, ``difference_hess``).

        Every prepared star of the on region and of the two off regions
        (``stream_band_regions``) outside the excluded mask, not only those
        in the match filter; colour ``reference_band -
        color_bands[color_index]`` of the match filter and magnitude
        ``reference_band``; densities per deg² with the areas of the regions.
        The off diagram is the mean of those of the two off regions
        (``on_off_regions.region_on_off_hess``, ``mean_off_hess``).

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
        band_regions = self._band_regions()
        self.on_hess, self.off_hess, self.diff_hess = region_on_off_hess(
            self.search.stars, match_filter_cmd_columns(self.search.match_filter)[color_index], band_regions,
            color_bins=color_bins, mag_bins=mag_bins,
        )
        return self

    def compute_null_test(
        self,
        shift_n_sigma: float | None = None,
        direction: str = "auto",
        *,
        color_bins: Any = None,
        mag_bins: Any = None,
        color_index: int = 0,
    ) -> CandidateInvestigation:
        """Null test: the regions moved off the stream, their Hess diagrams and SNR, in ``null_test``.

        The on region of ``regions`` and the off region on the same side are
        shifted in phi2 by ``shift_n_sigma`` sigma away from the stream, the
        other off region kept: the stream is in none of the three, so the on -
        off Hess diagram should show no isochrone and the SNR be about 0.
        Above (to larger phi2) or below; with ``"auto"``, the side whose
        regions stay in the window, the one with the larger unmasked on region
        when both do. When no side fits (a wide stream, a window bounded by the
        survey edge), the off region on the shifted side is put right next to
        the shifted on region instead of keeping their gap ("compact"
        geometry, logged): the stream is still in none of the regions. The
        Hess diagrams as ``compute_hess`` (every prepared star), the SNR of the
        stars in the match filter at ``distance_modulus`` as ``summary``
        (``selection_snr``). ``regions`` and the other results are not changed.

        Parameters
        ----------
        shift_n_sigma : float, optional
            Shift, in units of the sigma of ``regions``. Default:
            ``on_n_sigma + off_n_sigma[0]``: the inner edge of the shifted on
            region at that of the off regions, where the stream is taken as
            absent.
        direction : {"auto", "above", "below"}, optional
            Side of the shift.
        color_bins, mag_bins, color_index :
            See ``compute_hess``.

        Returns
        -------
        self : CandidateInvestigation
            ``null_test``: ``{"direction", "shift_deg", "geometry"
            ("standard" or "compact"), "regions", "band_regions", "areas",
            "on_hess", "off_hess", "diff_hess", "n_on", "n_off", "n_excess",
            "n_excess_error", "snr"}``.

        Raises
        ------
        ValueError
            ``define_regions`` not done; ``direction`` unknown; or the shifted
            regions leave the window on the side(s) asked, in both geometries
            (make it larger: ``min_phi2_half_width_deg``).
        """
        self._band_regions()  # define_regions done
        if direction not in ("auto", "above", "below"):
            raise ValueError(f"direction must be 'auto', 'above' or 'below', not {direction!r}.")
        regions = self.regions
        shift_n_sigma = self.on_n_sigma + self.off_n_sigma[0] if shift_n_sigma is None else float(shift_n_sigma)
        shift = shift_n_sigma * regions["sigma"]
        on_offset, (below_offset, above_offset) = regions["on_phi2_offset_deg"], regions["off_phi2_offsets_deg"]
        # compact: the inner edge of the shifted off region at the outer edge of the shifted on region (a hair beyond,
        # not to overlap by rounding)
        next_to_on = shift + (regions["on_n_sigma"] - regions["off_n_sigma"][0]) * regions["sigma"] + 1e-9
        off_offsets = {
            "standard": {"above": (below_offset, above_offset + shift), "below": (below_offset - shift, above_offset)},
            "compact": {"above": (below_offset, on_offset + next_to_on), "below": (on_offset - next_to_on, above_offset)},
        }
        sides = ("above", "below") if direction == "auto" else (direction,)
        outside = {}
        for geometry in ("standard", "compact"):
            shifted = {side: stream_regions(regions["phi1_range"], regions["phi2_centre"], regions["sigma"],
                                            on_n_sigma=regions["on_n_sigma"], off_n_sigma=regions["off_n_sigma"],
                                            on_phi2_offset_deg=on_offset + (shift if side == "above" else -shift),
                                            off_phi2_offsets_deg=off_offsets[geometry][side])
                       for side in sides}
            outside[geometry] = {side: self._regions_outside_window(shifted[side]) for side in sides}
            fitting = [side for side in sides if not outside[geometry][side]]
            if fitting:
                break
        else:
            raise ValueError(f"The regions shifted by {shift:.2f} deg leave the window "
                             f"{tuple(round(value, 2) for value in self.window)}, also with the off region next to "
                             "the on region (" + "; ".join(f"{side}: {extent}" for side, extent in
                                                           outside["compact"].items())
                             + "): make it larger (min_phi2_half_width_deg).")
        if geometry == "compact":
            LOGGER.info("%s: null test with the off region next to the shifted on region (the regions shifted by "
                        "%.2f deg with their gap leave the window)", self.name, shift)
        sides = fitting
        band_regions = {side: stream_band_regions(self.search.stars, shifted[side], self.stream_frame,
                                                  self.excluded_healpix_mask) for side in sides}
        areas = {side: region_areas(band_regions[side]) for side in sides}
        side = max(sides, key=lambda name: areas[name]["on"])
        on_hess, off_hess, diff_hess = region_on_off_hess(
            self.search.stars, match_filter_cmd_columns(self.search.match_filter)[color_index], band_regions[side],
            color_bins=color_bins, mag_bins=mag_bins,
        )
        snr = selection_snr(band_regions[side], self.stars_in_match_filter())
        self.null_test = {
            "direction": side, "shift_deg": float(shift), "geometry": geometry, "regions": shifted[side],
            "band_regions": band_regions[side],
            "areas": areas[side], "on_hess": on_hess, "off_hess": off_hess, "diff_hess": diff_hess,
            "n_on": snr["n_on"], "n_off": snr["n_off"], "n_excess": snr["n_signal"],
            "n_excess_error": snr["n_signal_error"], "snr": snr["snr"],
        }
        return self

    def _regions_outside_window(self, regions: Mapping[str, Any]) -> str:
        """Where regions leave the window: their phi1 range and phi2 extent, empty when inside."""
        windows = region_windows(regions)
        phi2_min = min(window[2] for window in (windows["on"], *windows["off"]))
        phi2_max = max(window[3] for window in (windows["on"], *windows["off"]))
        phi1_range = regions["phi1_range"]
        if phi2_min < self.window[2] or phi2_max > self.window[3] or phi1_range[0] < self.window[0] \
                or phi1_range[1] > self.window[1]:
            return f"phi1 {phi1_range}, phi2 {phi2_min:.2f} to {phi2_max:.2f} deg"
        return ""

    def compute_photometric_errors(
        self,
        *,
        mag_bins: Any = None,
        color_index: int = 0,
        in_match_filter: bool = False,
        statistic: str = DEFAULT_ERROR_STATISTIC,
    ) -> CandidateInvestigation:
        """Median (or mean) photometric error of each band of the stars of the on region per bin of its
        magnitude, in ``photometric_errors``.

        ``photometric_errors_by_magnitude`` of the prepared stars of the on
        region outside the excluded mask (``stream_band_regions``, as
        ``compute_hess``): the median (``statistic``) error of the reference
        band (VIS) per bin of VIS and of the colour band (e.g. H) per bin of
        H, the bands of
        the match filter (colour ``color_index``). The ``in_match_filter`` and
        ``color_index`` used are added to the result.

        Parameters
        ----------
        mag_bins : array_like, optional
            Bin edges (mag). Default: those of the Hess diagrams
            (``default_hess_bins``).
        color_index : int, optional
            Colour band of the match filter.
        in_match_filter : bool, optional
            Only the stars in the match filter at ``distance_modulus``
            (``stars_in_match_filter``).
        statistic : {"median", "mean"}, optional
            Statistic of the errors in each bin.

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            ``define_regions`` not done; ``distance_modulus`` not set with
            ``in_match_filter``.
        """
        selected = np.asarray(self._band_regions()["on"], dtype=bool)
        if in_match_filter:
            selected = selected & self.stars_in_match_filter()
        errors = photometric_errors_by_magnitude(
            self.search.stars, match_filter_cmd_columns(self.search.match_filter)[color_index], mask=selected,
            mag_bins=mag_bins, statistic=statistic,
        )
        self.photometric_errors = {**errors, "in_match_filter": bool(in_match_filter), "color_index": int(color_index)}
        return self

    def compute_depth_dust_maps(
        self,
        *,
        dust_correction: str | None = DEFAULT_DUST_CORRECTION,
        bands: Sequence[str] | None = None,
        depth_quantities: Sequence[str] = DEFAULT_DEPTH_QUANTITIES,
    ) -> CandidateInvestigation:
        """Depth maps of each band and dust map in the frame, in ``depth_dust_maps``.

        ``depth_dust_phi12_maps`` on the grid of the maps in the frame (the
        window and bin size of ``search``): the depth or the dust can be
        compared bin by bin with the star maps, e.g. to see whether the
        candidate follows a fluctuation of either
        (``plot_candidate_investigation.plot_candidate_depth_dust_maps``).

        Parameters
        ----------
        dust_correction : str or None, optional
            Dust map: ``"SFD98"`` or ``"PLANCK"``, that of the dust correction
            given to ``get_euclid_stars``; None: no dust map.
        bands : sequence of str, optional
            Bands of the depth maps. Default: all those with a map
            (``depth_dust_maps.depth_map_bands``).
        depth_quantities : sequence of str, optional
            ``"MAGLIM"`` (limiting magnitude) and/or ``"FRACDET"`` (detection
            fraction).

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            ``prepare`` not done.
        """
        self._require("search", "prepare")
        self.depth_dust_maps = depth_dust_phi12_maps(
            self.stream_frame, self.search.window, self.search.bin_size_deg, bands=bands,
            depth_quantities=depth_quantities, dust_correction=dust_correction,
        )
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
            No distance modulus; both ``metallicity`` and ``feh``
            (``isochrone_cmd_track``).
        """
        distance_modulus = self.distance_modulus if distance_modulus is None else distance_modulus
        if distance_modulus is None:
            raise ValueError("compute_isochrone needs a distance_modulus: give one, or set cand.distance_modulus.")
        if metallicity is None and feh is None:  # those of the object: [Fe/H] when given (in the label), else Z
            metallicity, feh = (None, self.feh) if self.feh is not None else (self.isochrone["metallicity"], None)
        self.isochrone_track = isochrone_cmd_track(
            distance_modulus, self.age if age is None else age, metallicity, feh=feh,
            match_filter=self.match_filter if self.search is None else self.search.match_filter,
            color_index=color_index, isochrone_model=self.isochrone_model,
        )
        return self

    def run_all_steps(
        self,
        catalog: pd.DataFrame,
        *,
        footprint_mask: NDArray | None = None,
        nest: bool = False,
        step_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> CandidateInvestigation:
        """Every step in order, with the inputs of the object.

        ``compute_sky_map`` and ``compute_zoom_map`` at ``distance_modulus``,
        ``prepare``, ``scan_distance_moduli`` (over ``distance_moduli``),
        ``fit_width``, ``define_regions``, ``compute_hess``,
        ``compute_isochrone``: the steps of the example notebook, with the
        distance modulus chosen there.

        Parameters
        ----------
        catalog : pandas.DataFrame
            Stars of a box containing the window. Not copied.
        footprint_mask : numpy.ndarray of bool, optional
            See ``prepare``.
        nest : bool, optional
            See ``prepare``.
        step_kwargs : mapping, optional
            More arguments of a step, by method name (``RUN_ALL_STEPS``; not
            ``prepare``: ``prepare_kwargs``), e.g.
            ``{"fit_width": {"background_degree": 1}, "compute_hess":
            {"color_bins": ..., "mag_bins": ...}}``.

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            ``distance_modulus`` not set; a key of ``step_kwargs`` not a step.
        """
        if self.distance_modulus is None:
            raise ValueError("run_all_steps needs cand.distance_modulus: choose it from the maps of "
                             "scan_distance_moduli first.")
        step_kwargs = dict(step_kwargs or {})
        unknown = set(step_kwargs) - set(RUN_ALL_STEPS)
        if unknown:
            raise ValueError(f"Unknown steps {sorted(unknown)} in step_kwargs; the steps are {RUN_ALL_STEPS}.")
        kwargs = {step: dict(step_kwargs.get(step, {})) for step in RUN_ALL_STEPS}
        self.compute_sky_map(catalog, self.distance_modulus, **kwargs["compute_sky_map"])
        self.prepare(catalog, footprint_mask=footprint_mask, nest=nest)
        self.compute_zoom_map(self.distance_modulus, **kwargs["compute_zoom_map"])
        self.scan_distance_moduli(**kwargs["scan_distance_moduli"])
        self.fit_width(**kwargs["fit_width"])
        self.define_regions(**kwargs["define_regions"])
        self.compute_hess(**kwargs["compute_hess"])
        self.compute_isochrone(**kwargs["compute_isochrone"])
        return self

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

    def compute_snr(self) -> float | None:
        """SNR of the on region over the off regions, as in ``summary``."""
        if self.regions is None or self.distance_modulus is None:
            return None
        band_regions = self._band_regions()
        snr = selection_snr(band_regions, self.stars_in_match_filter())
        self.snr = snr
        return snr
    
    def other_tracks_phi12(self, step_deg: float = OTHER_TRACK_STEP_DEG) -> list[tuple[str, NDArray, NDArray]]:
        """The known streams and the other listed candidates in the window, in the frame of the candidate.

        For the maps: every galstreams track crossing the footprint
        (``galstreams_mask_tracks``, ``build_masks.NEVER_MASKED_STREAMS``
        included) and every listed candidate but this one
        (``_other_candidate_tracks``), whether masked or not (wide, far in
        distance modulus, in ``unmasked_streams``). Points at most
        ``step_deg`` apart along each (``build_masks.points_along_track``: the
        great circle between the endpoints of a candidate), NaN outside the
        window, so that the line stops at its edges; a track with no point in
        the window is left out.

        Parameters
        ----------
        step_deg : float, optional
            Largest distance between two points of a track (deg).

        Returns
        -------
        tracks : list of (str, numpy.ndarray, numpy.ndarray)
            ``(name, phi1, phi2)`` (deg): the galstreams track name or
            ``"candidate_<id>"``.

        Raises
        ------
        ValueError
            ``prepare`` not done (the footprint).
        """
        self._require("footprint_mask", "prepare")
        tracks = galstreams_mask_tracks(self.footprint_mask, nest=self.footprint_nest, never_masked=())
        tracks += self._other_candidate_tracks()
        phi1_min, phi1_max, phi2_min, phi2_max = self.window
        in_window = []
        for track in tracks:
            phi1, phi2 = radec_to_phi12(*points_along_track(track, step_deg), self.stream_frame)
            inside = (phi1 >= phi1_min) & (phi1 <= phi1_max) & (phi2 >= phi2_min) & (phi2 <= phi2_max)
            if inside.any():
                in_window.append((track["name"], np.where(inside, phi1, np.nan), np.where(inside, phi2, np.nan)))
        return in_window

    def summary(self) -> dict[str, Any]:
        """Key numbers of the investigation, those computed so far.

        Returns
        -------
        summary : dict
            ``{"name", "endpoint_1", "endpoint_2", "window", "age",
            "metallicity", "feh", "distance_modulus", "distance_kpc",
            "masked_streams", "overlapping_streams", "phi2_centre", "sigma", "phi1_range",
            "area_on_deg2", "area_off_deg2", "n_on", "n_off", "n_excess",
            "n_excess_error", "snr"}``; None for the steps not done.
            ``masked_streams``: names of the streams masked (near
            ``distance_modulus``), ``overlapping_streams`` those left in
            because they cover the candidate; None before ``prepare`` or
            without ``mask_streams``. ``metallicity`` is the Z
            used (``isochrone``). ``n_on`` and ``n_off``: stars in the match
            filter at ``distance_modulus`` outside the excluded mask, in the
            on region and in the two off regions; ``n_excess = area_on * (n_on
            / area_on - off density)``, the off density the mean of the two
            off regions, its Poisson error ``n_excess_error`` and ``snr``,
            ``n_excess`` over the square root of the background on the on
            region (``on_off_regions.selection_snr``, as the tuning on known
            objects and streams).
        """
        summary: dict[str, Any] = {
            "name": self.name, "endpoint_1": tuple(self.endpoint_1), "endpoint_2": tuple(self.endpoint_2),
            "window": None if self.window is None else tuple(float(value) for value in self.window),
            "age": self.isochrone["age"], "metallicity": self.isochrone["metallicity"], "feh": self.feh,
            "distance_modulus": self.distance_modulus,
            "distance_kpc": None if self.distance_modulus is None
            else float(convert_DM_to_kpc(self.distance_modulus).value),
            "masked_streams": None if self.masked_streams is None else list(self.masked_streams),
            "overlapping_streams": None if self.overlapping_streams is None else list(self.overlapping_streams),
        }
        fit = self.width_fit if self.regions is None else self.regions
        if self.regions is not None and self.search is not None:
            self._band_regions()  # the areas of regions edited by hand
        summary.update({
            "phi2_centre": None if fit is None else float(fit["phi2_centre"]),
            "sigma": None if fit is None else float(fit["sigma"]),
            "phi1_range": None if self.regions is None else self.regions["phi1_range"],
            "area_on_deg2": None if self.areas is None else self.areas["on"],
            "area_off_deg2": None if self.areas is None else self.areas["off"],
            "n_on": None, "n_off": None, "n_excess": None, "n_excess_error": None, "snr": None,
        })
        if self.regions is not None and self.distance_modulus is not None:
            snr = self.compute_snr()
            summary.update({"n_on": snr["n_on"], "n_off": snr["n_off"], "n_excess": snr["n_signal"],
                            "n_excess_error": snr["n_signal_error"], "snr": snr["snr"]})
        return summary


# ===================================================
# Best distance modulus of the listed candidates
# ===================================================


def update_candidate_best_dm(
    catalog: pd.DataFrame,
    candidate_ids: Sequence[int] | None = None,
    *,
    distance_moduli: Sequence[float] | None = None,
    path: str | Path = DEFAULT_CANDIDATES_PATH,
    replace_force: bool = False,
    include_bad_candidates: bool = False,
    footprint_mask: NDArray | None = None,
    nest: bool = False,
    **inputs: Any,
) -> pd.DataFrame:
    """Fill the ``best_dm`` column of the list: the distance modulus of the largest SNR of each candidate.

    For each candidate without ``best_dm`` (each one with ``replace_force``):
    ``CandidateInvestigation.from_candidate_id`` at its ``best_dm_by_eye``,
    ``prepare`` with ``catalog``, ``fit_width`` and ``define_regions`` there
    (as the candidate notebook), then ``snr_against_distance_modulus`` over
    ``distance_moduli``: ``best_dm`` is the distance modulus of the largest
    SNR. A candidate whose width fit or regions fail is left as it was, with a
    warning. The file is written after each candidate: a long run stopped
    and run again skips those done.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars of a box containing the windows of the candidates. Not copied.
    candidate_ids : sequence of int, optional
        Default: every candidate of the list, but the bad ones unless
        ``include_bad_candidates``.
    distance_moduli : sequence of float, optional
        Grid of the SNR. Default: ``distance_moduli`` of ``inputs``, else
        the grid of ``prepare_search``.
    path : str or pathlib.Path, optional
        CSV file of the list.
    replace_force : bool, optional
        Compute again the ``best_dm`` already in the file.
    include_bad_candidates : bool, optional
        The bad candidates (status ``bad``, ``load_candidates``) in the
        default ``candidate_ids`` and in the masks of the others
        (``CandidateInvestigation.include_bad_candidates``).
    footprint_mask : numpy.ndarray of bool, optional
        HEALPix footprint (True outside), see ``prepare``. Default: from the
        stars of ``catalog`` (``footprint_mask_from_stars``), computed once.
    nest : bool, optional
        Ordering of a given ``footprint_mask``.
    **inputs
        Other inputs of ``CandidateInvestigation`` (``match_filter``,
        ``isochrone_model``, ``prepare_kwargs``...), the same for every
        candidate.

    Returns
    -------
    candidates : pandas.DataFrame
        The list, as written (the bad candidates included).
    """
    candidates = load_candidates(path, include_bad_candidates=True)  # written back: the bad ones kept
    if candidate_ids is None:
        ids = load_candidates(path, include_bad_candidates=include_bad_candidates)["id"].tolist()
    else:
        ids = [int(candidate_id) for candidate_id in candidate_ids]
    inputs["include_bad_candidates"] = include_bad_candidates
    if footprint_mask is None:
        footprint_mask = footprint_mask_from_stars(catalog["ra"].to_numpy(), catalog["dec"].to_numpy(),
                                                   DEFAULT_FOOTPRINT_NSIDE)
        nest = False
    for index, candidate_id in enumerate(ids, start=1):
        row = candidates.index[candidates["id"] == candidate_id]
        if row.empty:
            raise ValueError(f"No candidate with id {candidate_id} in {path}; the ids are {candidates['id'].tolist()}.")
        best_dm, by_eye = candidates.loc[row[0], [BEST_DM_COLUMN, "best_dm_by_eye"]]
        if not pd.isna(best_dm) and not replace_force:
            LOGGER.info("candidate %d (%d/%d): best_dm already %.2f, skipped (replace_force=True to compute it "
                        "again)", candidate_id, index, len(ids), best_dm)
            continue
        LOGGER.info("candidate %d (%d/%d): SNR against the distance modulus, regions at best_dm_by_eye %.2f",
                    candidate_id, index, len(ids), by_eye)
        cand = CandidateInvestigation.from_candidate_id(candidate_id, path=path, distance_modulus=float(by_eye),
                                                        **inputs)
        cand.prepare(catalog, footprint_mask=footprint_mask, nest=nest)
        try:
            cand.fit_width().define_regions()
        except (RuntimeError, ValueError) as error:
            LOGGER.warning("candidate %d: no best_dm, the width fit or the regions failed: %s", candidate_id, error)
            continue
        snr = cand.snr_against_distance_modulus(distance_moduli)
        peak = snr.loc[snr["snr"].idxmax()]
        candidates.loc[row[0], BEST_DM_COLUMN] = float(peak["distance_modulus"])
        candidates = write_candidates(candidates, path)
        LOGGER.info("candidate %d: best_dm %.2f (SNR %.1f; by eye %.2f)", candidate_id, peak["distance_modulus"],
                    peak["snr"], by_eye)
    return candidates

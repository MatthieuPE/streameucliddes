"""Investigation of one stream candidate: its regions and width, and one object holding every step.

Two layers, in this module:

- plain functions returning plain dicts, each callable alone and tunable:
  the on and off regions (``stream_regions``, ``region_windows``,
  ``region_area_deg2``), the density profiles (``phi12_density_profile``,
  ``on_off_phi1_profile``) and the width (``fit_stream_width``);
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
Its width is the sigma of a Gaussian plus a constant background, fitted to
the phi2 density profile of the stars in the match filter. The on region is
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
on - off Hess diagrams). On band 1.4 sigma, off bands 2.5 to 4.5 sigma: user
choices, no reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from streameucliddes.analysis.generic_stream_search import generic_search_single_dm
from streameucliddes.catalogs.isochrone_parameters import DEFAULT_AGES_GYR, DEFAULT_METALLICITIES
from streameucliddes.stream_frames import (
    DEFAULT_MIN_PHI2_HALF_WIDTH_DEG,
    DEFAULT_PADDING_PHI1_DEG,
    DEFAULT_PADDING_PHI2_DEG,
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
DEFAULT_OFF_N_SIGMA = (2.5, 4.5)  # user choice
DEFAULT_SIGMA_GUESS_DEG = 0.2  # initial sigma of the width fit
DEFAULT_PHI2_BIN_DEG = 0.05  # bins of the phi2 profile
DEFAULT_PHI1_BIN_DEG = 0.5  # bins of the phi1 profile
DEFAULT_FOOTPRINT_NSIDE = 512  # footprint from the stars, when there is no sky map
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
        ``off_n_sigma[1] <= off_n_sigma[0]``, or ``sigma <= 0``.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
        Bin size along ``axis`` (deg), e.g. ``DEFAULT_PHI2_BIN_DEG``.
    axis_range : tuple of float, optional
        Range of the bins (deg). Default: that of the windows along ``axis``.

    Returns
    -------
    profile : dict
        ``{"axis", "centres", "counts", "area_deg2", "density", "error",
        "bin_deg"}``: density in stars/deg², error ``sqrt(counts) /
        area_deg2``. Bins with no area have NaN density and error.
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
        From ``phi12_density_profile`` along phi2.
    phi2_centre_guess, sigma_guess : float, optional
        Initial values (deg).

    Returns
    -------
    fit : dict
        ``{"phi2_centre", "sigma", "amplitude", "background", "errors",
        "model"}``: ``errors`` a dict of the 1-sigma errors of the four
        parameters; ``model`` the fitted density at ``profile["centres"]``.
        Densities in stars/deg², angles in deg.

    Raises
    ------
    ValueError
        The profile is not along phi2.
    RuntimeError
        The fit does not converge.

    References
    ----------
    Shipp et al. 2018, ApJ 862, 114.
    """
    raise NotImplementedError


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
        ``{"centres", "on", "off", "difference", "error", "bin_deg"}``:
        densities in stars/deg²; ``error`` that of the difference (Poisson,
        in quadrature).
    """
    raise NotImplementedError


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
    phi2_profile, width_fit : dict
        ``phi12_density_profile`` (along phi2) and ``fit_stream_width`` results.
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
    search: PreparedSearch | None = None
    excluded_healpix_mask: NDArray | None = None
    sky_map: dict | None = None
    zoom_map: dict | None = None
    maps: dict | None = None
    phi2_profile: dict | None = None
    width_fit: dict | None = None
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

        Parameters
        ----------
        catalog : pandas.DataFrame, optional
            Stars, kept in ``self.catalog`` (a reference). Default: the
            catalog of ``compute_sky_map`` or of a previous call.
        footprint_mask : numpy.ndarray of bool, optional
            HEALPix footprint, True outside. Default:
            ``sky_map["footprint_mask"]`` (with its ``nest``), else from the
            stars at ``DEFAULT_FOOTPRINT_NSIDE``.
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
        raise NotImplementedError

    def set_endpoints(
        self,
        endpoint_1: tuple[float, float],
        endpoint_2: tuple[float, float],
        **window_kwargs: Any,
    ) -> CandidateInvestigation:
        """New endpoints: new frame, window and box; prepared again; later results cleared.

        Fast: the sky map is kept; only ``prepare`` on the kept catalog is
        done again. Then call ``compute_zoom_map`` and
        ``plot_candidate_frame_check``.

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
            load a larger box.
        """
        raise NotImplementedError

    def compute_zoom_map(self, distance_modulus: float | None = None) -> CandidateInvestigation:
        """One matched-filter map in the frame (``stream_frame_search_single_dm``).

        Parameters
        ----------
        distance_modulus : float, optional
            Default: that of ``sky_map``.

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            ``prepare`` not done, or no distance modulus.
        """
        raise NotImplementedError

    def scan_distance_moduli(self, distance_moduli: Sequence[float] | None = None) -> CandidateInvestigation:
        """Maps in the frame over distance moduli (``stream_frame_search_dm_scan``), in ``maps``.

        Parameters
        ----------
        distance_moduli : sequence of float, optional
            Default: ``self.distance_moduli``, else the grid of the search.

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            ``prepare`` not done.
        """
        raise NotImplementedError

    def fit_width(self, phi1_range: tuple[float, float] | None = None) -> CandidateInvestigation:
        """phi2 density profile of the stars in the match filter, and its Gaussian width.

        Stars of ``search.stars`` in the match filter at
        ``distance_modulus`` (``select_stars_in_match_filter``), then
        ``phi12_density_profile`` along phi2 over the window and
        ``fit_stream_width``.

        Parameters
        ----------
        phi1_range : tuple of float, optional
            Default: the phi1 of the endpoints.

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            ``prepare`` not done, or ``distance_modulus`` not set.
        """
        raise NotImplementedError

    def define_regions(
        self,
        phi1_range: tuple[float, float] | None = None,
        phi2_centre: float | None = None,
        sigma: float | None = None,
    ) -> CandidateInvestigation:
        """On and off regions, their areas and the phi1 profile.

        ``stream_regions``, ``region_area_deg2`` and ``on_off_phi1_profile``
        (stars in the match filter, bins over the window).

        Parameters
        ----------
        phi1_range : tuple of float, optional
            Default: the phi1 of the endpoints.
        phi2_centre, sigma : float, optional
            Default: from ``width_fit``.

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            ``phi2_centre`` or ``sigma`` not given and ``fit_width`` not done;
            or the off bands leave the window.
        """
        raise NotImplementedError

    def compute_hess(self, *, color_bins: Any = None, mag_bins: Any = None) -> CandidateInvestigation:
        """On, off and on - off Hess diagrams (``hess_diagram``, ``difference_hess``).

        Parameters
        ----------
        color_bins, mag_bins : array_like, optional
            Bin edges, the same for on and off. Default: those of
            ``hess_diagram``.

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            ``define_regions`` not done.
        """
        raise NotImplementedError

    def compute_isochrone(
        self,
        *,
        age: float | None = None,
        metallicity: float | None = None,
        feh: float | None = None,
        distance_modulus: float | None = None,
    ) -> CandidateInvestigation:
        """Isochrone track in the bands of the match filter (``isochrone_cmd_track``).

        Parameters
        ----------
        age, metallicity, feh, distance_modulus : float, optional
            Default: those of the object. Given values are only used for
            this track, not stored as inputs.

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            No distance modulus.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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

"""Investigation of one stream candidate: all its inputs and results in one object.

``CandidateInvestigation`` holds the inputs (two endpoints, the isochrone,
the distance modulus, the region sizes) and the result of each step, and
calls the functions in order. It adds no science: each method calls one or
two functions of ``stream_frames``, ``analysis.generic_stream_search``,
``analysis.stream_frame_search``, ``analysis.stream_regions``,
``analysis.hess_diagrams`` and ``analysis.cmd_match_filter``, stores the
result and returns the object, so that calls can be chained. Each function
can also be called alone, and its result put back in the object (e.g.
``cand.regions = stream_regions(...)``).

Steps, in order:

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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from streameucliddes.analysis.stream_regions import DEFAULT_OFF_N_SIGMA, DEFAULT_ON_N_SIGMA
from streameucliddes.catalogs.isochrone_parameters import DEFAULT_AGES_GYR, DEFAULT_METALLICITIES
from streameucliddes.stream_frames import DEFAULT_PADDING_PHI1_DEG, DEFAULT_PHI2_HALFWIDTH_DEG

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from numpy.typing import NDArray

    from streameucliddes.analysis.stream_frame_search import PreparedSearch

DEFAULT_FOOTPRINT_NSIDE = 512  # footprint from the stars, when there is no sky map


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
        Isochrone Z (Marigo et al. 2017 scale). Exactly one of
        ``metallicity`` and ``feh``.
    feh : float, optional
        Isochrone [Fe/H], converted to Z (``unit_conversions.convert_FeH_to_z``).
    distance_modulus : float, optional
        Distance modulus of the stream, chosen by eye after the scan
        (``scan_distance_moduli``). Needed from ``fit_width`` on.
    distance_moduli : sequence of float, optional
        Grid of the scan. Default: that of ``prepare_search``.
    match_filter : dict, optional
        See ``analysis.cmd_match_filter``. Default: ``DEFAULT_MATCH_FILTER``.
    phi2_halfwidth_deg, padding_phi1_deg : float, optional
        Window around the endpoints (``stream_frames.endpoints_phi12_window``).
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
        ``phi2_density_profile`` and ``fit_stream_width`` results.
    regions, areas, phi1_profile : dict
        ``stream_regions``, ``region_area_deg2`` and ``phi1_density_profile``
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
    metallicity: float | None = DEFAULT_METALLICITIES[0]
    feh: float | None = None
    distance_modulus: float | None = None
    distance_moduli: Sequence[float] | None = None
    match_filter: dict | None = None
    phi2_halfwidth_deg: float = DEFAULT_PHI2_HALFWIDTH_DEG
    padding_phi1_deg: float = DEFAULT_PADDING_PHI1_DEG
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
            Neither or both of ``metallicity`` and ``feh``.
        """
        raise NotImplementedError

    # ===================================================
    # Steps
    # ===================================================

    def compute_sky_map(
        self, catalog: pd.DataFrame, distance_modulus: float | None = None, **generic_kwargs: Any
    ) -> CandidateInvestigation:
        """Matched-filter map of the loaded box, once (``generic_search_single_dm``).

        Does not depend on the frame: ``set_endpoints`` keeps it.

        Parameters
        ----------
        catalog : pandas.DataFrame
            Stars of a box larger than the candidate, e.g. from
            ``get_euclid_stars``. Not copied.
        distance_modulus : float, optional
            Default: ``self.distance_modulus``.
        **generic_kwargs
            Arguments of ``generic_search_single_dm`` (``smoothing_deg``,
            ``nside``...). The isochrone is that of the object.

        Returns
        -------
        self : CandidateInvestigation

        Raises
        ------
        ValueError
            No distance modulus given nor set.
        """
        raise NotImplementedError

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
            ``phi2_halfwidth_deg``, ``padding_phi1_deg``: set on the object.

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
        ``phi2_density_profile`` and ``fit_stream_width``.

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

        ``stream_regions``, ``region_area_deg2`` and ``phi1_density_profile``
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

    def track_phi12(self, n_points: int = 100) -> tuple[NDArray, NDArray]:
        """(phi1, phi2) of the track: ``stream_track_phi12(regions)``, or the arc between the endpoints before ``define_regions``.

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

    def track_radec(self, n_points: int = 100) -> tuple[NDArray, NDArray]:
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

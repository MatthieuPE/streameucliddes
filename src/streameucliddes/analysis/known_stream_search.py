"""Known-stream search: matched-filter maps of a known stream, in its frame.

For one galstreams stream and one isochrone: the stars inside the match
filter, counted on a phi1/phi2 grid in the stream frame, for each distance
modulus. No polynomial background is subtracted. Magnitudes are used as they are: the Euclid x DES
magnitudes are already dereddened.

Mirrors ``generic_stream_search``: ``_single_dm`` gives one map; ``_dm_scan``
goes over a grid of distance moduli one at a time (it yields ``(dm,
result)``), and the workflows build the stacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from typing import Any

    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray

if TYPE_CHECKING:  # galstreams: new dependency of phase 3, not installed yet
    import galstreams


def known_stream_search_single_dm(
    catalog: pd.DataFrame,
    rotation_matrix: NDArray[np.float64],
    window: tuple[float, float, float, float],
    distance_modulus: float,
    isochrone: Mapping[str, float],
    match_filter: Mapping[str, Any],
    *,
    bin_size_deg: float,
    smoothing_deg: float,
    cuts: Mapping[str, Any] | None = None,
    footprint_mask: NDArray[np.bool_] | None = None,
    phi12: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    flag_cuts: NDArray[np.bool_] | None = None,
) -> dict[str, Any]:
    """Matched-filter map of a known stream at one distance modulus.

    Selects the stars in the match filter of one isochrone
    (``cmd_match_filter.select_stars_in_match_filter``), and counts them on
    the phi1/phi2 grid (``star_density_maps.star_density_phi12``).

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars, with ``ra``, ``dec`` (deg) and the magnitude columns of the
        match filter, e.g. ``catalogs.euclid_des_catalog.get_eucliddes_stars()``.
    rotation_matrix : numpy.ndarray, shape (3, 3)
        Stream frame (see ``stream_frames``).
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` of the map (deg), e.g.
        from ``catalogs.galstreams_catalog.galstreams_phi12_window``.
    distance_modulus : float
        Distance modulus of the match filter.
    isochrone : mapping
        One isochrone, ``{"age": age (Gyr), "metallicity": Z}``: an item of
        ``isochrone_parameters.known_stream_isochrone_parameters``.
    match_filter : mapping
        See ``cmd_match_filter.select_stars_in_match_filter``.
    bin_size_deg : float
        Bin size of the grid (deg).
    smoothing_deg : float
        Standard deviation of the Gaussian smoothing (deg).
    cuts : mapping, optional
        Cuts of ``catalogs.catalog_cuts.get_cut_mask``.
    footprint_mask : numpy.ndarray of bool, optional
        From ``catalogs.survey_footprint.survey_footprint_phi12_mask``, True
        outside the footprint.
    phi12 : tuple of numpy.ndarray, optional
        Precomputed (phi1, phi2) of the stars, from
        ``stream_frames.radec_to_phi12``.
    flag_cuts : numpy.ndarray of bool, optional
        Precomputed mask of ``cuts``. ``phi12`` and ``flag_cuts`` do not
        depend on the distance modulus: ``known_stream_search_dm_scan``
        computes them once.

    Returns
    -------
    result : dict
        ``density`` (masked array, shape (n_phi1, n_phi2)), ``n_selected``,
        ``distance_modulus``, ``distance_kpc``, ``window``,
        ``bin_size_deg``, ``smoothing_deg``, ``isochrone`` and
        ``match_filter``.

    Notes
    -----
    Code to reuse: ``generic_stream_search.generic_search_single_dm``
    (structure, precomputed ``flag_cuts``); ``_compute_spline_phi12`` in
    celestial_currents
    ``notebooks/candidate_investigations/full_search/template.ipynb``.

    References
    ----------
    Rockosi et al. 2002, AJ 124, 349. Grillmair 2009, ApJ 693, 1118.
    Shipp et al. 2018, ApJ 862, 114.
    """
    raise NotImplementedError


def known_stream_search_dm_scan(
    catalog: pd.DataFrame,
    rotation_matrix: NDArray[np.float64],
    window: tuple[float, float, float, float],
    distance_moduli: Iterable[float],
    isochrone: Mapping[str, float],
    match_filter: Mapping[str, Any],
    *,
    bin_size_deg: float,
    smoothing_deg: float,
    cuts: Mapping[str, Any] | None = None,
    footprint_mask: NDArray[np.bool_] | None = None,
    n_workers: int = 1,
) -> Iterator[tuple[float, dict[str, Any]]]:
    """Matched-filter maps of a known stream over a grid of distance moduli.

    Runs ``known_stream_search_single_dm`` for each distance modulus, one at
    a time. The phi1/phi2 of the stars, the stars inside the window and the
    cuts are computed once, then reused for every distance modulus.

    Parameters
    ----------
    catalog, rotation_matrix, window, isochrone, match_filter
        See ``known_stream_search_single_dm``.
    distance_moduli : iterable of float
        Grid of distance moduli. Plan: 15 to 22.
    bin_size_deg, smoothing_deg, cuts
        See ``known_stream_search_single_dm``.
    footprint_mask : numpy.ndarray of bool, optional
        See ``known_stream_search_single_dm``.
    n_workers : int, optional
        Processes computing the distance moduli in parallel (1: none). The
        results are still yielded in the order of ``distance_moduli``.

    Yields
    ------
    distance_modulus : float
        One distance modulus of the grid.
    result : dict
        ``known_stream_search_single_dm`` output for it.

    Notes
    -----
    Code to reuse: ``generic_stream_search.generic_search_dm_scan`` (cuts
    computed once, results yielded); the ``ProcessPoolExecutor`` sweep cells
    of celestial_currents
    ``notebooks/candidate_investigations/full_search/template.ipynb``.
    """
    raise NotImplementedError


def known_stream_search_dm_along_track(
    catalog: pd.DataFrame,
    track: galstreams.Track6D,
    rotation_matrix: NDArray[np.float64],
    window: tuple[float, float, float, float],
    isochrone: Mapping[str, float],
    match_filter: Mapping[str, Any],
    *,
    dm_offsets: Iterable[float] = (0.0,),
    dm_step: float = 0.1,
    bin_size_deg: float,
    smoothing_deg: float,
    cuts: Mapping[str, Any] | None = None,
    footprint_mask: NDArray[np.bool_] | None = None,
) -> Iterator[tuple[float, dict[str, Any]]]:
    """Matched-filter maps that follow the distance of the track along phi1.

    In each phi1 bin, the match filter is at the distance modulus of the
    track there (``catalogs.galstreams_catalog.galstreams_dm_along_track``)
    plus an offset: one map per offset. The distance moduli of the track are
    rounded to ``dm_step``, so that each filter polygon is built once. Beyond
    the ends of the track, the distance modulus of the nearest end is used
    (proposal).

    Parameters
    ----------
    catalog, rotation_matrix, window, isochrone, match_filter
        See ``known_stream_search_single_dm``.
    track : galstreams.Track6D
        Track of the stream, from ``catalogs.galstreams_catalog.load_galstreams``.
    dm_offsets : iterable of float, optional
        Offsets added to the distance modulus of the track.
    dm_step : float, optional
        Rounding step of the distance moduli of the track.
    bin_size_deg, smoothing_deg, cuts
        See ``known_stream_search_single_dm``.
    footprint_mask : numpy.ndarray of bool, optional
        See ``known_stream_search_single_dm``.

    Yields
    ------
    dm_offset : float
        One offset of ``dm_offsets``.
    result : dict
        As ``known_stream_search_single_dm``, with ``dm_along_phi1`` (the
        distance modulus of each phi1 bin) instead of ``distance_modulus``.

    Notes
    -----
    Code to reuse: ``known_stream_search_single_dm`` and
    ``cmd_match_filter.select_stars_in_match_filter``, applied to the stars
    of each group of phi1 bins sharing a distance modulus.
    """
    raise NotImplementedError

"""Known-stream search: matched-filter maps of a stream, in its frame.

For one stream and one isochrone: the stars inside the match filter, counted
on a phi1/phi2 grid in the stream frame, for each distance modulus. No
polynomial background is subtracted. Magnitudes are used as they are: the
Euclid x DES magnitudes are already dereddened.

Every search starts with ``prepare_search``, done once: the values not given
taken from galstreams and the population table, the catalog prepared
(``catalog_preparation.prepare_match_filter_catalog``), the footprint put on
the grid. Then each map is ``cmd_match_filter.select_stars_in_match_filter``
and ``star_density_maps.star_density_phi12`` on the prepared stars:

- ``known_stream_search_single_dm``: one map;
- ``known_stream_search_dm_scan``: ``single_dm`` over a grid of distance
  moduli, for each isochrone of the stream, yielding ``(dm, result)`` one at
  a time;
- ``known_stream_search_dm_along_track``: maps following the distance of the
  track along phi1.

Each takes the arguments of ``prepare_search``, or a ``search`` already
prepared (e.g. for several calls on the same stream).
"""

from __future__ import annotations

import functools
import inspect
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np

from streameucliddes.analysis.catalog_preparation import prepare_match_filter_catalog
from streameucliddes.analysis.cmd_match_filter import DEFAULT_MATCH_FILTER, select_stars_in_match_filter
from streameucliddes.analysis.star_density_maps import star_density_phi12
from streameucliddes.catalogs.galstreams_catalog import (
    galstreams_dm_along_track,
    galstreams_phi12_window,
    load_galstreams,
    select_galstreams_in_footprint,
)
from streameucliddes.catalogs.isochrone_parameters import (
    default_isochrone_grid,
    known_stream_isochrone_parameters,
)
from streameucliddes.catalogs.survey_footprint import survey_footprint_phi12_mask
from streameucliddes.stream_frames import phi12_bin_centres
from streameucliddes.unit_conversions import convert_DM_to_kpc

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from typing import Any

    import astropy.coordinates as coord
    import galstreams
    import pandas as pd
    from numpy.typing import NDArray

DEFAULT_BIN_SIZE_DEG = 0.1  # about a HEALPix pixel at nside 512
DEFAULT_SMOOTHING_DEG = 0.3  # as the default of star_density_healpix
DM_STEP = 0.1  # step of the default grids of distance moduli
TRACK_DM_MARGIN = 1.0  # default grid: the distance moduli of the track in the footprint, +- this margin
# Default grid without a track distance: the plan's 15 to 22
DEFAULT_DISTANCE_MODULI = tuple(np.round(np.arange(15.0, 22.0 + DM_STEP / 2, DM_STEP), 1))
# Arguments of select_galstreams_in_footprint that prepare_search passes on when given
SELECTION_ARGUMENTS = tuple(
    name for name, parameter in inspect.signature(select_galstreams_in_footprint).parameters.items()
    if parameter.kind is inspect.Parameter.KEYWORD_ONLY and name != "nest"
)


# ===================================================
# Preparation
# ===================================================


@dataclass(frozen=True)
class PreparedSearch:
    """Everything a known-stream search uses, from ``prepare_search``.

    Attributes
    ----------
    stars : pandas.DataFrame
        The prepared catalog: stars of the window passing the cuts, with
        ``phi1``, ``phi2`` and the magnitudes of the match filter.
    stream_frame : astropy.coordinates.BaseCoordinateFrame or None
        Frame of the maps (None when only prepared stars were given).
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg).
    bin_size_deg : float
        Bin size of the grid (deg).
    phi1_centres, phi2_centres : numpy.ndarray
        Bins of the grid (``stream_frames.phi12_bin_centres``).
    footprint_mask : numpy.ndarray of bool or None
        Footprint on the grid, True outside.
    match_filter : dict
        See ``cmd_match_filter``.
    isochrones : list of dict
        ``{"age", "metallicity"}``, one set of maps each.
    distance_moduli : list of float
        Grid of distance moduli of the scan.
    track : galstreams.Track6D or None
        The galstreams track; None for a stream not in galstreams.
    track_name : str or None
        Its name.
    dm_min, dm_max : float
        Distance moduli of the track in the footprint
        (``select_galstreams_in_footprint``); NaN without a track distance.
    """

    stars: Any
    stream_frame: Any
    window: tuple[float, float, float, float]
    bin_size_deg: float
    phi1_centres: Any
    phi2_centres: Any
    footprint_mask: Any
    match_filter: dict
    isochrones: list
    distance_moduli: list
    track: Any = None
    track_name: str | None = None
    dm_min: float = np.nan
    dm_max: float = np.nan


def prepare_search(
    catalog: pd.DataFrame | None = None,
    stream_frame: coord.BaseCoordinateFrame | None = None,
    window: tuple[float, float, float, float] | None = None,
    *,
    stream_name: str | None = None,
    track: galstreams.Track6D | None = None,
    match_filter: Mapping[str, Any] | None = None,
    isochrone: Mapping[str, float] | None = None,
    distance_moduli: Iterable[float] | None = None,
    cuts: Mapping[str, Any] | None = None,
    stars: pd.DataFrame | None = None,
    mws: galstreams.MWStreams | None = None,
    population_table: pd.DataFrame | None = None,
    default_grid: Iterable[Mapping[str, float]] | None = None,
    bin_size_deg: float = DEFAULT_BIN_SIZE_DEG,
    footprint_mask: NDArray[np.bool_] | None = None,
    nest: bool | None = None,
    **selection_kwargs: Any,
) -> PreparedSearch:
    """Prepare a known-stream search: its values, its stars, its footprint.

    Every value given is kept. With a galstreams ``stream_name`` (a stream:
    its default track, galstreams ``On`` flag; or a track) or ``track``,
    those not given come from ``catalogs.galstreams_catalog``:

    - ``stream_frame``: the ``stream_frame`` of the track;
    - ``window``: ``galstreams_phi12_window``;
    - ``distance_moduli``: from ``dm_min - TRACK_DM_MARGIN`` to
      ``dm_max + TRACK_DM_MARGIN`` every ``DM_STEP``, ``dm_min`` and
      ``dm_max`` being those of the track in the footprint
      (``select_galstreams_in_footprint``, with its defaults or
      ``selection_kwargs``; on the whole sky without a HEALPix footprint).
      Without a track distance: ``DEFAULT_DISTANCE_MODULI`` (15 to 22).

    The isochrones: ``isochrone``, else those of the stream
    (``catalogs.isochrone_parameters.known_stream_isochrone_parameters``),
    else the default grid (``default_grid``, or
    ``catalogs.isochrone_parameters.default_isochrone_grid``). Then the catalog is prepared once
    (``catalog_preparation.prepare_match_filter_catalog``) and the footprint
    put on the grid (``catalogs.survey_footprint.survey_footprint_phi12_mask``).

    Parameters
    ----------
    catalog : pandas.DataFrame, optional
        Stars, with ``ra``, ``dec`` (deg) and the magnitude columns of the
        match filter, e.g. ``catalogs.euclid_des_catalog.get_eucliddes_stars()``.
        Not needed when ``stars`` is given.
    stream_frame : astropy.coordinates.BaseCoordinateFrame, optional
        Stream frame, e.g. a gala ``GreatCircleICRSFrame``.
    window : tuple of float, optional
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` of the maps (deg).
    stream_name : str, optional
        galstreams stream or track name; or the name of a stream not in
        galstreams (then ``stream_frame`` and ``window`` are needed), for its
        isochrones.
    track : galstreams.Track6D, optional
        The track itself, instead of ``stream_name``.
    match_filter : mapping, optional
        See ``cmd_match_filter``. Default: ``cmd_match_filter.DEFAULT_MATCH_FILTER``.
    isochrone : mapping, optional
        The one isochrone of the search, ``{"age": age (Gyr), "metallicity": Z}``.
    distance_moduli : iterable of float, optional
        Grid of distance moduli of the scan.
    cuts : mapping, optional
        Cuts of the preparation (default: errors <= 0.2 on the bands of the
        match filter). Ignored with ``stars``.
    stars : pandas.DataFrame, optional
        The catalog already prepared with this frame, window, match filter
        and cuts: used instead of ``catalog``.
    mws : galstreams.MWStreams, optional
        Default: ``catalogs.galstreams_catalog.load_galstreams()``, read only
        for ``stream_name``.
    population_table : pandas.DataFrame, optional
        See ``known_stream_isochrone_parameters``.
    default_grid : iterable of mapping, optional
        Isochrones (``{"age", "metallicity"}``) for the age and Z not known:
        those of a stream without a literature value, or all of them for a
        search without a stream. Default:
        ``catalogs.isochrone_parameters.default_isochrone_grid()``.
    bin_size_deg : float, optional
        Bin size of the grid (deg).
    footprint_mask : numpy.ndarray of bool, optional
        True outside the footprint: a HEALPix mask
        (``catalogs.survey_footprint.survey_footprint_healpix_mask``, needs
        ``nest``); or a mask on the grid, shape (n_phi1, n_phi2).
    nest : bool, optional
        HEALPix ordering of a HEALPix ``footprint_mask``.
    **selection_kwargs
        Arguments of ``select_galstreams_in_footprint`` (``SELECTION_ARGUMENTS``:
        ``dm_range``, ``min_length_deg``, ``add_ibata2024``), only when given.

    Returns
    -------
    search : PreparedSearch

    Raises
    ------
    ValueError
        A value needed is neither given nor found in galstreams, or the
        track is not selected by ``select_galstreams_in_footprint``.
    TypeError
        An argument that ``select_galstreams_in_footprint`` does not take.
    """
    unknown = set(selection_kwargs) - set(SELECTION_ARGUMENTS)
    if unknown:
        raise TypeError(f"Unexpected arguments {sorted(unknown)}; select_galstreams_in_footprint takes {SELECTION_ARGUMENTS}.")

    # Values not given: from galstreams
    if track is None and stream_name is not None:
        track = _galstreams_track(stream_name, mws)
    dm_min = dm_max = np.nan
    if track is not None:
        stream_frame = track.stream_frame if stream_frame is None else stream_frame
        window = galstreams_phi12_window(track) if window is None else window
        dm_min, dm_max = _dm_range_in_footprint(track, footprint_mask, nest, selection_kwargs)
    elif stream_name is not None and (window is None or (stream_frame is None and stars is None)):
        raise ValueError(f"{stream_name!r} is not in galstreams: give its stream_frame and window.")
    if window is None:
        raise ValueError("Give a window, or a galstreams stream_name.")
    window = tuple(float(value) for value in window)
    if distance_moduli is None:
        distance_moduli = _distance_moduli_grid(dm_min, dm_max)

    if isochrone is not None:
        isochrones = [dict(isochrone)]
    elif track is not None or stream_name is not None:
        isochrones = known_stream_isochrone_parameters(
            track.stream_name if track is not None else stream_name, population_table, default_grid
        )
    else:
        isochrones = default_isochrone_grid() if default_grid is None else [dict(item) for item in default_grid]
    match_filter = dict(DEFAULT_MATCH_FILTER if match_filter is None else match_filter)

    # Stars and footprint, once
    if stars is None:
        if catalog is None:
            raise ValueError("Give a catalog or the prepared stars.")
        if stream_frame is None:
            raise ValueError("A catalog needs a stream_frame (or a galstreams stream_name), for phi1/phi2.")
        stars = prepare_match_filter_catalog(catalog, match_filter, cuts=cuts, stream_frame=stream_frame, window=window)
    if footprint_mask is not None and np.ndim(footprint_mask) == 1:  # HEALPix: put on the grid
        if nest is None or stream_frame is None:
            raise ValueError("A HEALPix footprint_mask needs nest and a stream_frame.")
        footprint_mask = survey_footprint_phi12_mask(footprint_mask, stream_frame, window, bin_size_deg, nest=nest)
    phi1_centres, phi2_centres = phi12_bin_centres(window, bin_size_deg)

    return PreparedSearch(
        stars=stars, stream_frame=stream_frame, window=window, bin_size_deg=bin_size_deg,
        phi1_centres=phi1_centres, phi2_centres=phi2_centres, footprint_mask=footprint_mask,
        match_filter=match_filter, isochrones=list(isochrones), distance_moduli=[float(dm) for dm in distance_moduli],
        track=track, track_name=None if track is None else track.track_name, dm_min=dm_min, dm_max=dm_max,
    )


# ===================================================
# Searches
# ===================================================


def known_stream_search_single_dm(
    catalog: pd.DataFrame | None = None,
    stream_frame: coord.BaseCoordinateFrame | None = None,
    window: tuple[float, float, float, float] | None = None,
    distance_modulus: float | None = None,
    isochrone: Mapping[str, float] | None = None,
    *,
    search: PreparedSearch | None = None,
    smoothing_deg: float = DEFAULT_SMOOTHING_DEG,
    **prepare_kwargs: Any,
) -> dict[str, Any]:
    """Matched-filter map of a stream at one distance modulus.

    Selects the prepared stars in the match filter of one isochrone
    (``cmd_match_filter.select_stars_in_match_filter``), and counts them on
    the phi1/phi2 grid (``star_density_maps.star_density_phi12``).

    Parameters
    ----------
    catalog, stream_frame, window
        See ``prepare_search``.
    distance_modulus : float, optional
        Distance modulus of the match filter. Default: the middle of
        ``dm_min`` and ``dm_max`` of the track in the footprint, rounded to
        ``DM_STEP``.
    isochrone : mapping, optional
        ``{"age": age (Gyr), "metallicity": Z}``. Default: the first
        isochrone of the search.
    search : PreparedSearch, optional
        From ``prepare_search``; else prepared here from the other arguments.
    smoothing_deg : float, optional
        Standard deviation of the Gaussian smoothing (deg).
    **prepare_kwargs
        Other arguments of ``prepare_search`` (e.g. ``stream_name``,
        ``match_filter``, ``cuts``, ``bin_size_deg``, ``footprint_mask``,
        ``nest``); not with ``search``.

    Returns
    -------
    result : dict
        ``density`` (masked array, shape (n_phi1, n_phi2)), ``phi1_centres``
        and ``phi2_centres`` (the bins), ``n_stars`` (stars in the window),
        ``n_selected`` (in the match filter), ``distance_modulus``,
        ``distance_kpc``, ``window``, ``bin_size_deg``, ``smoothing_deg``,
        ``isochrone``, ``match_filter`` and ``track_name`` (None without a
        galstreams track).

    Raises
    ------
    ValueError
        No distance modulus given nor in galstreams; preparation arguments
        with ``search``; see ``prepare_search``.

    References
    ----------
    Rockosi et al. 2002, AJ 124, 349. Grillmair 2009, ApJ 693, 1118.
    Shipp et al. 2018, ApJ 862, 114.
    """
    search = _search(search, catalog, stream_frame, window, prepare_kwargs)
    if distance_modulus is None:
        if np.isnan(search.dm_min):
            raise ValueError("Give distance_modulus: no galstreams track distance to take it from.")
        distance_modulus = float(np.round(np.round((search.dm_min + search.dm_max) / 2 / DM_STEP) * DM_STEP, 1))
    isochrone = search.isochrones[0] if isochrone is None else dict(isochrone)

    selected = select_stars_in_match_filter(search.stars, distance_modulus, isochrone, search.match_filter)
    result = _map_of_selected(search, selected, smoothing_deg)
    result.update(
        distance_modulus=float(distance_modulus),
        distance_kpc=float(convert_DM_to_kpc(distance_modulus).value),
        isochrone=isochrone,
    )
    return result


def known_stream_search_dm_scan(
    catalog: pd.DataFrame | None = None,
    stream_frame: coord.BaseCoordinateFrame | None = None,
    window: tuple[float, float, float, float] | None = None,
    distance_moduli: Iterable[float] | None = None,
    isochrone: Mapping[str, float] | None = None,
    *,
    search: PreparedSearch | None = None,
    smoothing_deg: float = DEFAULT_SMOOTHING_DEG,
    n_workers: int = 1,
    **prepare_kwargs: Any,
) -> Iterator[tuple[float, dict[str, Any]]]:
    """Matched-filter maps of a stream over a grid of distance moduli, per isochrone.

    The search is prepared once (``prepare_search``); then each isochrone
    and distance modulus is a call of ``known_stream_search_single_dm``.
    The maps are yielded one at a time: the caller can save or plot each
    one and drop it before the next one.

    Parameters
    ----------
    catalog, stream_frame, window
        See ``prepare_search``.
    distance_moduli : iterable of float, optional
        Grid of distance moduli. Default: that of the search (see
        ``prepare_search``).
    isochrone : mapping, optional
        The one isochrone of the scan. Default: every isochrone of the search.
    search : PreparedSearch, optional
        From ``prepare_search``; else prepared here from the other arguments.
    smoothing_deg : float, optional
        See ``known_stream_search_single_dm``.
    n_workers : int, optional
        Processes computing the maps in parallel (1: none). Each process
        receives the search once. The results are still yielded in order.
    **prepare_kwargs
        Other arguments of ``prepare_search``; not with ``search``.

    Yields
    ------
    distance_modulus : float
        One distance modulus of the grid; every distance modulus for the
        first isochrone, then for the next one.
    result : dict
        ``known_stream_search_single_dm`` output for it (its ``isochrone``
        says which).

    Notes
    -----
    Processes pay off for large windows: a map takes about 0.15 s for
    300 000 stars in the window, and a process takes a few seconds to start.
    """
    search = _search(search, catalog, stream_frame, window, prepare_kwargs)
    distance_moduli = search.distance_moduli if distance_moduli is None else [float(dm) for dm in distance_moduli]
    isochrones = search.isochrones if isochrone is None else [dict(isochrone)]
    tasks = [(isochrone, dm) for isochrone in isochrones for dm in distance_moduli]

    if n_workers <= 1:
        for isochrone, dm in tasks:
            yield dm, known_stream_search_single_dm(
                distance_modulus=dm, isochrone=isochrone, search=search, smoothing_deg=smoothing_deg
            )
        return
    single_dm = functools.partial(_worker_single_dm, smoothing_deg=smoothing_deg)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_set_worker_search, initargs=(search,)) as executor:
        yield from zip((dm for _, dm in tasks), executor.map(single_dm, tasks))


def known_stream_search_dm_along_track(
    catalog: pd.DataFrame | None = None,
    track: galstreams.Track6D | None = None,
    stream_frame: coord.BaseCoordinateFrame | None = None,
    window: tuple[float, float, float, float] | None = None,
    isochrone: Mapping[str, float] | None = None,
    *,
    search: PreparedSearch | None = None,
    dm_offsets: Iterable[float] = (0.0,),
    dm_step: float = DM_STEP,
    smoothing_deg: float = DEFAULT_SMOOTHING_DEG,
    **prepare_kwargs: Any,
) -> Iterator[tuple[float, dict[str, Any]]]:
    """Matched-filter maps that follow the distance of the track along phi1, per isochrone.

    In each phi1 bin, the match filter is at the distance modulus of the
    track there plus an offset: one map per offset. Along the track, its
    distance moduli; beyond its ends, over the rest of the window, their
    linear extrapolation, to follow a possible extension of the stream
    (``catalogs.galstreams_catalog.galstreams_dm_along_track`` with
    ``extrapolate``). They are rounded to ``dm_step``: the stars of the phi1
    bins sharing a distance modulus go through one filter.

    Parameters
    ----------
    catalog : pandas.DataFrame, optional
        See ``prepare_search``.
    track : galstreams.Track6D, optional
        Track of the stream. Default: the galstreams track of ``stream_name``.
    stream_frame : astropy.coordinates.BaseCoordinateFrame, optional
        Frame of the maps; its phi1 must be the track's. Default:
        ``track.stream_frame``.
    window : tuple of float, optional
        See ``prepare_search``.
    isochrone : mapping, optional
        See ``known_stream_search_dm_scan``.
    search : PreparedSearch, optional
        From ``prepare_search`` with a track; else prepared here.
    dm_offsets : iterable of float, optional
        Offsets added to the distance modulus of the track.
    dm_step : float, optional
        Rounding step of the distance moduli of the track.
    smoothing_deg : float, optional
        See ``known_stream_search_single_dm``.
    **prepare_kwargs
        Other arguments of ``prepare_search``; not with ``search``.

    Yields
    ------
    dm_offset : float
        One offset of ``dm_offsets``; every offset for the first isochrone,
        then for the next one.
    result : dict
        As ``known_stream_search_single_dm``, with ``dm_along_phi1`` (the
        distance modulus of each phi1 bin, offset included) and
        ``dm_offset`` instead of ``distance_modulus`` and ``distance_kpc``.

    Raises
    ------
    ValueError
        No track, or a track without distance (e.g. the Ibata et al. 2024
        tracks).
    """
    if track is not None:
        prepare_kwargs["track"] = track
    search = _search(search, catalog, stream_frame, window, prepare_kwargs)
    if search.track is None:
        raise ValueError("Give a track or a galstreams stream_name: the distance follows the track.")
    dm_along_phi1 = galstreams_dm_along_track(search.track, search.phi1_centres, extrapolate=True)
    if np.isnan(dm_along_phi1).any():
        raise ValueError(f"Track {search.track_name} has no distance in galstreams.")
    dm_along_phi1 = np.round(dm_along_phi1 / dm_step) * dm_step

    stars = search.stars
    # phi1 bin of each star (the last bin may end within half a bin of the window)
    star_bin = np.clip(
        np.floor((stars["phi1"].to_numpy() - search.window[0]) / search.bin_size_deg).astype(int),
        0, search.phi1_centres.size - 1,
    )
    star_dm = dm_along_phi1[star_bin]
    isochrones = search.isochrones if isochrone is None else [dict(isochrone)]
    for isochrone in isochrones:
        for dm_offset in dm_offsets:
            selected = np.zeros(len(stars), dtype=bool)
            for dm in np.unique(star_dm):
                selected |= select_stars_in_match_filter(
                    stars, dm + dm_offset, isochrone, search.match_filter, mask=star_dm == dm
                )
            result = _map_of_selected(search, selected, smoothing_deg)
            result.update(dm_along_phi1=dm_along_phi1 + dm_offset, dm_offset=float(dm_offset), isochrone=isochrone)
            yield float(dm_offset), result


# ===================================================
# Helpers
# ===================================================


def _search(search, catalog, stream_frame, window, prepare_kwargs):
    """The search given, else prepared from the arguments."""
    if search is None:
        return prepare_search(catalog, stream_frame, window, **prepare_kwargs)
    given = [name for name, value in (("catalog", catalog), ("stream_frame", stream_frame), ("window", window))
             if value is not None]
    if given or prepare_kwargs:
        raise ValueError(f"Already in the search: {given + sorted(prepare_kwargs)}; give them to prepare_search.")
    return search


def _galstreams_track(stream_name, mws):
    """The galstreams track of a track name, or the default ('On') track of a stream name; None if neither."""
    mws = load_galstreams() if mws is None else mws
    if stream_name in mws:  # a track name
        return mws[stream_name]
    summary = mws.summary  # one row per track: its stream ("Name") and default track ("On")
    default_tracks = summary.loc[(summary["Name"] == stream_name) & summary["On"], "TrackName"]
    return mws[default_tracks.iloc[0]] if len(default_tracks) else None


def _dm_range_in_footprint(track, footprint_mask, nest, selection_kwargs):
    """dm_min, dm_max of the track in the footprint, from select_galstreams_in_footprint."""
    if footprint_mask is not None and np.ndim(footprint_mask) == 1:
        if nest is None:
            raise ValueError("A HEALPix footprint_mask needs nest.")
        healpix_mask, healpix_nest = footprint_mask, nest
    else:  # no HEALPix footprint: the whole sky
        healpix_mask, healpix_nest = np.zeros(hp.nside2npix(1), dtype=bool), False
    selection = select_galstreams_in_footprint(
        {track.track_name: track}, healpix_mask, nest=healpix_nest, **selection_kwargs
    )
    if selection.empty:
        raise ValueError(
            f"Track {track.track_name} is not selected by select_galstreams_in_footprint "
            f"(arguments {selection_kwargs or 'by default'}): not enough of it in the footprint, "
            "in the dm range, or an Ibata et al. 2024 track without add_ibata2024=True."
        )
    return float(selection["dm_min"].iloc[0]), float(selection["dm_max"].iloc[0])


def _distance_moduli_grid(dm_min, dm_max):
    """dm_min - TRACK_DM_MARGIN to dm_max + TRACK_DM_MARGIN, every DM_STEP; 15 to 22 without them."""
    if np.isnan(dm_min):
        return list(DEFAULT_DISTANCE_MODULI)
    low = np.floor((dm_min - TRACK_DM_MARGIN) / DM_STEP) * DM_STEP
    high = np.ceil((dm_max + TRACK_DM_MARGIN) / DM_STEP) * DM_STEP
    return list(np.round(np.arange(low, high + DM_STEP / 2, DM_STEP), 1))


def _map_of_selected(search, selected, smoothing_deg):
    """Map of the selected stars on the grid of the search, with the parameters of the search."""
    stars = search.stars
    density = star_density_phi12(
        stars["phi1"].to_numpy()[selected], stars["phi2"].to_numpy()[selected], search.window,
        search.bin_size_deg, smoothing_deg=smoothing_deg, footprint_mask=search.footprint_mask,
    )
    return {
        "density": density, "phi1_centres": search.phi1_centres, "phi2_centres": search.phi2_centres,
        "n_stars": len(stars), "n_selected": int(selected.sum()), "window": search.window,
        "bin_size_deg": search.bin_size_deg, "smoothing_deg": smoothing_deg,
        "match_filter": search.match_filter, "track_name": search.track_name,
    }


# Search of a worker process, set once by its initializer
_WORKER_SEARCH = None


def _set_worker_search(search):
    global _WORKER_SEARCH
    _WORKER_SEARCH = search


def _worker_single_dm(task, *, smoothing_deg):
    isochrone, distance_modulus = task
    return known_stream_search_single_dm(
        distance_modulus=distance_modulus, isochrone=isochrone, search=_WORKER_SEARCH, smoothing_deg=smoothing_deg
    )

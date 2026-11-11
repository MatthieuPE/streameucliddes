"""Known-stream search: matched-filter maps of a stream, in its frame.

For one stream and one isochrone: the stars inside the match filter, counted
on a phi1/phi2 grid in the stream frame, for each distance modulus. No
polynomial background is subtracted. Magnitudes are used as they are: the
Euclid x DES magnitudes are already dereddened.

Built from the shared pieces: ``catalog_preparation.prepare_match_filter_catalog``
(columns, cuts, phi1/phi2, window), ``cmd_match_filter.select_stars_in_match_filter``
and ``star_density_maps.star_density_phi12``.

Every argument but the catalog is optional. Give ``stream_name``, a
galstreams stream (e.g. ``"AAU-ATLAS"``, its default track) or track (e.g.
``"AAU-ATLAS-L21"``), and what is not given is taken from galstreams and the
population table: the frame and window of the track, its distance moduli, the
isochrones of the stream. A given value is always kept: a stream not in
galstreams works with its ``stream_frame`` and ``window``. The prepared stars
can be passed (``stars``) to prepare a catalog only once.

Mirrors ``generic_stream_search``: ``_single_dm`` gives one map;
``_dm_scan`` calls it over a grid of distance moduli (and the isochrones of
the stream) and yields ``(dm, result)`` one at a time; the workflows build
the stacks.
"""

from __future__ import annotations

import functools
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from streameucliddes.analysis.catalog_preparation import prepare_match_filter_catalog
from streameucliddes.analysis.cmd_match_filter import DEFAULT_MATCH_FILTER, select_stars_in_match_filter
from streameucliddes.analysis.star_density_maps import star_density_phi12
from streameucliddes.catalogs.galstreams_catalog import (
    galstreams_dm_along_track,
    galstreams_phi12_window,
    load_galstreams,
)
from streameucliddes.catalogs.isochrone_parameters import (
    default_isochrone_grid,
    known_stream_isochrone_parameters,
)
from streameucliddes.catalogs.survey_footprint import survey_footprint_phi12_mask
from streameucliddes.stream_frames import phi12_bin_centres, radec_to_phi12
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
TRACK_DM_MARGIN = 1.0  # default grid: the distance moduli of the track, +- this margin
# Default grid without a track distance: the plan's 15 to 22
DEFAULT_DISTANCE_MODULI = tuple(np.round(np.arange(15.0, 22.0 + DM_STEP / 2, DM_STEP), 1))


def known_stream_search_single_dm(
    catalog: pd.DataFrame | None = None,
    stream_frame: coord.BaseCoordinateFrame | None = None,
    window: tuple[float, float, float, float] | None = None,
    distance_modulus: float | None = None,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    stream_name: str | None = None,
    mws: galstreams.MWStreams | None = None,
    population_table: pd.DataFrame | None = None,
    stars: pd.DataFrame | None = None,
    cuts: Mapping[str, Any] | None = None,
    bin_size_deg: float = DEFAULT_BIN_SIZE_DEG,
    smoothing_deg: float = DEFAULT_SMOOTHING_DEG,
    footprint_mask: NDArray[np.bool_] | None = None,
    nest: bool | None = None,
) -> dict[str, Any]:
    """Matched-filter map of a stream at one distance modulus.

    Prepares the catalog (``catalog_preparation.prepare_match_filter_catalog``),
    selects the stars in the match filter of one isochrone
    (``cmd_match_filter.select_stars_in_match_filter``), and counts them on
    the phi1/phi2 grid (``star_density_maps.star_density_phi12``).

    Parameters
    ----------
    catalog : pandas.DataFrame, optional
        Stars, with ``ra``, ``dec`` (deg) and the magnitude columns of the
        match filter, e.g. ``catalogs.euclid_des_catalog.get_eucliddes_stars()``.
        Not needed when ``stars`` is given.
    stream_frame : astropy.coordinates.BaseCoordinateFrame, optional
        Stream frame (e.g. a gala ``GreatCircleICRSFrame``). Default: the
        ``stream_frame`` of the galstreams track of ``stream_name``.
    window : tuple of float, optional
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` of the map (deg).
        Default: ``catalogs.galstreams_catalog.galstreams_phi12_window`` of
        the track.
    distance_modulus : float, optional
        Distance modulus of the match filter. Default: the median distance
        modulus of the track.
    isochrone : mapping, optional
        ``{"age": age (Gyr), "metallicity": Z}``. Default: the first isochrone
        of the stream (``catalogs.isochrone_parameters.known_stream_isochrone_parameters``),
        or of ``default_isochrone_grid`` without a galstreams stream.
    match_filter : mapping, optional
        See ``cmd_match_filter``. Default: ``cmd_match_filter.DEFAULT_MATCH_FILTER``.
    stream_name : str, optional
        galstreams stream (its default track, galstreams ``On`` flag) or
        track name, to fill the arguments not given.
    mws : galstreams.MWStreams, optional
        Default: ``catalogs.galstreams_catalog.load_galstreams()``, read only
        for ``stream_name``.
    population_table : pandas.DataFrame, optional
        See ``known_stream_isochrone_parameters``.
    stars : pandas.DataFrame, optional
        The catalog already prepared by ``prepare_match_filter_catalog`` with
        this frame, window, match filter and cuts: used instead of
        ``catalog``.
    cuts : mapping, optional
        Cuts of the preparation (default: errors <= 0.2 on the bands of the
        match filter). Ignored with ``stars``.
    bin_size_deg : float, optional
        Bin size of the grid (deg).
    smoothing_deg : float, optional
        Standard deviation of the Gaussian smoothing (deg).
    footprint_mask : numpy.ndarray of bool, optional
        True outside the footprint: on the grid, shape (n_phi1, n_phi2), as
        ``catalogs.survey_footprint.survey_footprint_phi12_mask``; or a
        HEALPix mask, converted to the grid (needs ``nest``).
    nest : bool, optional
        HEALPix ordering of a HEALPix ``footprint_mask``.

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
        A value needed is neither given nor found in galstreams.

    References
    ----------
    Rockosi et al. 2002, AJ 124, 349. Grillmair 2009, ApJ 693, 1118.
    Shipp et al. 2018, ApJ 862, 114.
    """
    search = _resolve_search(
        catalog, stream_frame, window, match_filter, stream_name=stream_name, mws=mws,
        population_table=population_table, stars=stars, cuts=cuts, bin_size_deg=bin_size_deg,
        footprint_mask=footprint_mask, nest=nest, isochrone=isochrone,
    )
    if distance_modulus is None:
        if search["track_dm"] is None:
            raise ValueError("Give distance_modulus: no galstreams track distance to take it from.")
        distance_modulus = float(np.round(np.median(search["track_dm"]) / DM_STEP) * DM_STEP)

    selected = select_stars_in_match_filter(
        search["stars"], distance_modulus, search["isochrones"][0], search["match_filter"]
    )
    result = _map_of_selected(search, selected, bin_size_deg, smoothing_deg)
    result.update(
        distance_modulus=float(distance_modulus),
        distance_kpc=float(convert_DM_to_kpc(distance_modulus).value),
        isochrone=dict(search["isochrones"][0]),
    )
    return result


def known_stream_search_dm_scan(
    catalog: pd.DataFrame | None = None,
    stream_frame: coord.BaseCoordinateFrame | None = None,
    window: tuple[float, float, float, float] | None = None,
    distance_moduli: Iterable[float] | None = None,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    stream_name: str | None = None,
    mws: galstreams.MWStreams | None = None,
    population_table: pd.DataFrame | None = None,
    stars: pd.DataFrame | None = None,
    cuts: Mapping[str, Any] | None = None,
    bin_size_deg: float = DEFAULT_BIN_SIZE_DEG,
    smoothing_deg: float = DEFAULT_SMOOTHING_DEG,
    footprint_mask: NDArray[np.bool_] | None = None,
    nest: bool | None = None,
    n_workers: int = 1,
) -> Iterator[tuple[float, dict[str, Any]]]:
    """Matched-filter maps of a stream over a grid of distance moduli, per isochrone.

    The catalog is prepared and the footprint put on the grid once; then
    ``known_stream_search_single_dm`` gives the map of each isochrone and
    distance modulus, yielded one at a time: the caller can save or plot it
    and drop it before the next one.

    Parameters
    ----------
    catalog, stream_frame, window, match_filter
        See ``known_stream_search_single_dm``.
    distance_moduli : iterable of float, optional
        Grid of distance moduli. Default: those of the galstreams track, +-
        ``TRACK_DM_MARGIN``, every ``DM_STEP``; without a track distance,
        ``DEFAULT_DISTANCE_MODULI`` (15 to 22).
    isochrone : mapping, optional
        The one isochrone of the scan. Default: every isochrone of the
        stream (``known_stream_isochrone_parameters``), or of
        ``default_isochrone_grid`` without a galstreams stream.
    stream_name, mws, population_table, stars, cuts
        See ``known_stream_search_single_dm``.
    bin_size_deg, smoothing_deg, footprint_mask, nest
        See ``known_stream_search_single_dm``.
    n_workers : int, optional
        Processes computing the maps in parallel (1: none). Each process
        receives the prepared stars once. The results are still yielded in
        order.

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
    search = _resolve_search(
        catalog, stream_frame, window, match_filter, stream_name=stream_name, mws=mws,
        population_table=population_table, stars=stars, cuts=cuts, bin_size_deg=bin_size_deg,
        footprint_mask=footprint_mask, nest=nest, isochrone=isochrone,
    )
    if distance_moduli is None:
        distance_moduli = _default_distance_moduli(search["track_dm"])
    tasks = [(isochrone, float(dm)) for isochrone in search["isochrones"] for dm in distance_moduli]
    single_dm = functools.partial(
        _single_dm_of_task, window=search["window"], match_filter=search["match_filter"],
        bin_size_deg=bin_size_deg, smoothing_deg=smoothing_deg, footprint_mask=search["footprint_mask"],
        track_name=search["track_name"],
    )

    if n_workers <= 1:
        for task in tasks:
            yield task[1], single_dm(task, stars=search["stars"])
        return
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_set_worker_stars, initargs=(search["stars"],)
    ) as executor:
        yield from zip((dm for _, dm in tasks), executor.map(functools.partial(_worker_single_dm, single_dm), tasks))


def known_stream_search_dm_along_track(
    catalog: pd.DataFrame | None = None,
    track: galstreams.Track6D | None = None,
    stream_frame: coord.BaseCoordinateFrame | None = None,
    window: tuple[float, float, float, float] | None = None,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    stream_name: str | None = None,
    mws: galstreams.MWStreams | None = None,
    population_table: pd.DataFrame | None = None,
    stars: pd.DataFrame | None = None,
    dm_offsets: Iterable[float] = (0.0,),
    dm_step: float = DM_STEP,
    cuts: Mapping[str, Any] | None = None,
    bin_size_deg: float = DEFAULT_BIN_SIZE_DEG,
    smoothing_deg: float = DEFAULT_SMOOTHING_DEG,
    footprint_mask: NDArray[np.bool_] | None = None,
    nest: bool | None = None,
) -> Iterator[tuple[float, dict[str, Any]]]:
    """Matched-filter maps that follow the distance of the track along phi1, per isochrone.

    In each phi1 bin, the match filter is at the distance modulus of the
    track there plus an offset: one map per offset. The distance moduli are
    those of ``catalogs.galstreams_catalog.galstreams_dm_along_track``,
    held at the ends of the track, rounded to ``dm_step``: the stars of the
    phi1 bins sharing a distance modulus go through one filter.

    Parameters
    ----------
    catalog, window, match_filter
        See ``known_stream_search_single_dm``.
    track : galstreams.Track6D, optional
        Track of the stream. Default: the galstreams track of ``stream_name``.
    stream_frame : astropy.coordinates.BaseCoordinateFrame, optional
        Frame of the maps; its phi1 must be the track's. Default:
        ``track.stream_frame``.
    isochrone : mapping, optional
        See ``known_stream_search_dm_scan``.
    stream_name, mws, population_table, stars
        See ``known_stream_search_single_dm``.
    dm_offsets : iterable of float, optional
        Offsets added to the distance modulus of the track.
    dm_step : float, optional
        Rounding step of the distance moduli of the track.
    cuts, bin_size_deg, smoothing_deg, footprint_mask, nest
        See ``known_stream_search_single_dm``.

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
    search = _resolve_search(
        catalog, stream_frame, window, match_filter, stream_name=stream_name, mws=mws,
        population_table=population_table, stars=stars, cuts=cuts, bin_size_deg=bin_size_deg,
        footprint_mask=footprint_mask, nest=nest, isochrone=isochrone, track=track,
    )
    track = search["track"]
    if track is None:
        raise ValueError("Give a track or a galstreams stream_name: the distance follows the track.")
    track_dm = galstreams_dm_along_track(track, search["phi1_centres"], hold_ends=True)
    if not np.isfinite(track_dm).all():
        raise ValueError(f"Track {search['track_name']} has no distance in galstreams.")
    track_dm = np.round(track_dm / dm_step) * dm_step

    stars = search["stars"]
    # phi1 bin of each star (the last bin may end within half a bin of the window)
    star_bin = np.clip(
        np.floor((stars["phi1"].to_numpy() - search["window"][0]) / bin_size_deg).astype(int),
        0, search["phi1_centres"].size - 1,
    )
    star_dm = track_dm[star_bin]
    for isochrone in search["isochrones"]:
        for dm_offset in dm_offsets:
            selected = np.zeros(len(stars), dtype=bool)
            for dm in np.unique(star_dm):
                selected |= select_stars_in_match_filter(
                    stars, dm + dm_offset, isochrone, search["match_filter"], mask=star_dm == dm
                )
            result = _map_of_selected(search, selected, bin_size_deg, smoothing_deg)
            result.update(dm_along_phi1=track_dm + dm_offset, dm_offset=float(dm_offset), isochrone=dict(isochrone))
            yield float(dm_offset), result


# ===================================================
# Helpers
# ===================================================


def _galstreams_track(stream_name, mws):
    """The galstreams track of a track name, or the default track of a stream name; None if neither."""
    mws = load_galstreams() if mws is None else mws
    if stream_name in mws:
        return mws[stream_name]
    # get_track_names_for_stream matches parts of names ("AAU" gives both AAU streams): keep exact ones
    names = [
        name for name in mws.get_track_names_for_stream(stream_name, On_only=True)
        if mws[name].stream_name == stream_name
    ]
    return mws[names[0]] if names else None


def _resolve_search(
    catalog, stream_frame, window, match_filter, *, stream_name, mws, population_table, stars, cuts,
    bin_size_deg, footprint_mask, nest, isochrone, track=None,
):
    """Every value of a search: those given, else from galstreams; the prepared stars and footprint."""
    if track is None and stream_name is not None:
        track = _galstreams_track(stream_name, mws)
        if track is None and (window is None or (stream_frame is None and stars is None)):
            raise ValueError(f"{stream_name!r} is not in galstreams: give its stream_frame and window.")
    if track is not None:
        stream_frame = track.stream_frame if stream_frame is None else stream_frame
        window = galstreams_phi12_window(track) if window is None else window
    if window is None:
        raise ValueError("Give a window, or a galstreams stream_name.")
    match_filter = DEFAULT_MATCH_FILTER if match_filter is None else match_filter

    if isochrone is not None:
        isochrones = [dict(isochrone)]
    elif track is not None:
        isochrones = known_stream_isochrone_parameters(track.stream_name, population_table)
    else:
        isochrones = default_isochrone_grid()

    if stars is None:
        if catalog is None:
            raise ValueError("Give a catalog or the prepared stars.")
        if stream_frame is None:
            raise ValueError("A catalog needs a stream_frame (or a galstreams stream_name), for phi1/phi2.")
        stars = prepare_match_filter_catalog(catalog, match_filter, cuts=cuts, stream_frame=stream_frame, window=window)

    phi1_centres, phi2_centres = phi12_bin_centres(window, bin_size_deg)
    if footprint_mask is not None and np.ndim(footprint_mask) == 1:  # HEALPix: put on the grid
        if nest is None or stream_frame is None:
            raise ValueError("A HEALPix footprint_mask needs nest and a stream_frame.")
        footprint_mask = survey_footprint_phi12_mask(footprint_mask, stream_frame, window, bin_size_deg, nest=nest)

    track_dm = None
    if track is not None:  # distance moduli of the track points (ends held: no NaN from rounding)
        track_phi1, _ = radec_to_phi12(track.track.ra.deg, track.track.dec.deg, track.stream_frame)
        track_dm = galstreams_dm_along_track(track, track_phi1, hold_ends=True)
        track_dm = track_dm if np.isfinite(track_dm).all() else None  # no distance (Ibata et al. 2024)

    return {
        "stars": stars, "window": tuple(window), "match_filter": match_filter, "isochrones": isochrones,
        "track": track, "track_name": None if track is None else track.track_name, "track_dm": track_dm,
        "footprint_mask": footprint_mask, "phi1_centres": phi1_centres, "phi2_centres": phi2_centres,
    }


def _default_distance_moduli(track_dm):
    """The track's distance moduli +- TRACK_DM_MARGIN every DM_STEP; 15 to 22 without them."""
    if track_dm is None:
        return list(DEFAULT_DISTANCE_MODULI)
    low = np.floor((np.min(track_dm) - TRACK_DM_MARGIN) / DM_STEP) * DM_STEP
    high = np.ceil((np.max(track_dm) + TRACK_DM_MARGIN) / DM_STEP) * DM_STEP
    return list(np.round(np.arange(low, high + DM_STEP / 2, DM_STEP), 1))


def _map_of_selected(search, selected, bin_size_deg, smoothing_deg):
    """Map of the selected stars, with the numbers of stars and the parameters of the search."""
    stars = search["stars"]
    density = star_density_phi12(
        stars["phi1"].to_numpy()[selected], stars["phi2"].to_numpy()[selected], search["window"], bin_size_deg,
        smoothing_deg=smoothing_deg, footprint_mask=search["footprint_mask"],
    )
    return {
        "density": density, "phi1_centres": search["phi1_centres"], "phi2_centres": search["phi2_centres"],
        "n_stars": len(stars), "n_selected": int(selected.sum()), "window": search["window"],
        "bin_size_deg": bin_size_deg, "smoothing_deg": smoothing_deg, "match_filter": search["match_filter"],
        "track_name": search["track_name"],
    }


def _single_dm_of_task(task, *, stars, window, match_filter, bin_size_deg, smoothing_deg, footprint_mask,
                       track_name):
    """known_stream_search_single_dm of one (isochrone, distance modulus) on prepared stars."""
    isochrone, distance_modulus = task
    result = known_stream_search_single_dm(
        None, None, window, distance_modulus, isochrone, match_filter, stars=stars,
        bin_size_deg=bin_size_deg, smoothing_deg=smoothing_deg, footprint_mask=footprint_mask,
    )
    result["track_name"] = track_name
    return result


# Prepared stars of a worker process, set once by its initializer
_WORKER_STARS = None


def _set_worker_stars(stars):
    global _WORKER_STARS
    _WORKER_STARS = stars


def _worker_single_dm(single_dm, task):
    return single_dm(task, stars=_WORKER_STARS)

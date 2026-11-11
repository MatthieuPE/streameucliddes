"""Known-stream search: matched-filter maps of a stream, in its frame.

For one stream and one isochrone: the stars inside the match filter, counted
on a phi1/phi2 grid in the stream frame, for each distance modulus. No
polynomial background is subtracted. Magnitudes are used as they are: the
Euclid x DES magnitudes are already dereddened.

Built from the shared pieces: ``catalog_preparation.prepare_match_filter_catalog``
(columns, cuts, phi1/phi2, window), ``cmd_match_filter.select_stars_in_match_filter``
and ``star_density_maps.star_density_phi12``.

Every argument but the catalog is optional. With ``stream_name``, a
galstreams stream (its default track, galstreams ``On`` flag, e.g.
``"AAU-ATLAS"``) or track (e.g. ``"AAU-ATLAS-L21"``), the values not given
come from ``catalogs.galstreams_catalog``:

- the frame: the ``stream_frame`` of the track;
- the window: ``galstreams_phi12_window``;
- the distance moduli: ``dm_min`` to ``dm_max`` of the part of the track in
  the footprint (``select_galstreams_in_footprint``), +- ``TRACK_DM_MARGIN``;

and the isochrones from ``catalogs.isochrone_parameters.known_stream_isochrone_parameters``.
A given value is always kept: a stream not in galstreams works with its
``stream_frame`` and ``window``. The prepared stars can be passed (``stars``)
to prepare a catalog only once.

Mirrors ``generic_stream_search``: ``_single_dm`` gives one map;
``_dm_scan`` calls it over a grid of distance moduli (and the isochrones of
the stream) and yields ``(dm, result)`` one at a time; the workflows build
the stacks.
"""

from __future__ import annotations

import functools
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
        Distance modulus of the match filter. Default: the middle of
        ``dm_min`` and ``dm_max`` of the track in the footprint, rounded to
        ``DM_STEP``.
    isochrone : mapping, optional
        ``{"age": age (Gyr), "metallicity": Z}``. Default: the first isochrone
        of ``catalogs.isochrone_parameters.known_stream_isochrone_parameters``
        for the stream (the default grid for a stream not in the population
        table, or without ``stream_name``).
    match_filter : mapping, optional
        See ``cmd_match_filter``. Default: ``cmd_match_filter.DEFAULT_MATCH_FILTER``.
    stream_name : str, optional
        galstreams stream or track name, to fill the arguments not given
        (see the module docstring); or the name of a stream not in
        galstreams, for its isochrones.
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
        True outside the footprint: a HEALPix mask
        (``catalogs.survey_footprint.survey_footprint_healpix_mask``, needs
        ``nest``), put on the grid and used for the distance moduli of the
        track in the footprint; or a mask on the grid, shape (n_phi1,
        n_phi2) (``survey_footprint_phi12_mask``).
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
    known = _find_galstreams_stream(stream_name, mws, footprint_mask, nest)
    stream_frame, window = _frame_and_window(known, stream_name, stream_frame, window, stars)
    if distance_modulus is None:
        if known is None or np.isnan(known.dm_min):
            raise ValueError("Give distance_modulus: no galstreams track distance to take it from.")
        distance_modulus = _round_dm((known.dm_min + known.dm_max) / 2)
    isochrone = _isochrones(isochrone, known, stream_name, population_table)[0]
    match_filter = DEFAULT_MATCH_FILTER if match_filter is None else match_filter
    stars = _search_stars(catalog, stars, stream_frame, window, match_filter, cuts)
    grid_footprint = _grid_footprint(footprint_mask, stream_frame, window, bin_size_deg, nest)

    selected = select_stars_in_match_filter(stars, distance_modulus, isochrone, match_filter)
    result = _map_of_selected(stars, selected, window, bin_size_deg, smoothing_deg, grid_footprint)
    result.update(
        distance_modulus=float(distance_modulus),
        distance_kpc=float(convert_DM_to_kpc(distance_modulus).value),
        isochrone=dict(isochrone), match_filter=match_filter,
        track_name=None if known is None else known.track_name,
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

    The values not given are taken once (see the module docstring), the
    catalog prepared and the footprint put on the grid once; then each
    isochrone and distance modulus is a call of
    ``known_stream_search_single_dm`` on the prepared stars. The maps are
    yielded one at a time: the caller can save or plot each one and drop
    it before the next one.

    Parameters
    ----------
    catalog, stream_frame, window, match_filter
        See ``known_stream_search_single_dm``.
    distance_moduli : iterable of float, optional
        Grid of distance moduli. Default: from ``dm_min - TRACK_DM_MARGIN``
        to ``dm_max + TRACK_DM_MARGIN`` of the track in the footprint, every
        ``DM_STEP``; without a track distance, ``DEFAULT_DISTANCE_MODULI``
        (15 to 22).
    isochrone : mapping, optional
        The one isochrone of the scan. Default: every isochrone of
        ``known_stream_isochrone_parameters`` for the stream (the default
        grid without ``stream_name``).
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
    known = _find_galstreams_stream(stream_name, mws, footprint_mask, nest)
    stream_frame, window = _frame_and_window(known, stream_name, stream_frame, window, stars)
    if distance_moduli is None:
        distance_moduli = _distance_moduli_grid(known)
    isochrones = _isochrones(isochrone, known, stream_name, population_table)
    match_filter = DEFAULT_MATCH_FILTER if match_filter is None else match_filter
    stars = _search_stars(catalog, stars, stream_frame, window, match_filter, cuts)
    grid_footprint = _grid_footprint(footprint_mask, stream_frame, window, bin_size_deg, nest)

    single_dm = functools.partial(
        _single_dm_on_stars, window=window, match_filter=match_filter, bin_size_deg=bin_size_deg,
        smoothing_deg=smoothing_deg, footprint_mask=grid_footprint,
        track_name=None if known is None else known.track_name,
    )
    tasks = [(isochrone, float(dm)) for isochrone in isochrones for dm in distance_moduli]
    if n_workers <= 1:
        for task in tasks:
            yield task[1], single_dm(task, stars=stars)
        return
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_set_worker_stars, initargs=(stars,)) as executor:
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
    if track is not None:
        known = _galstreams_stream_of_track(track, footprint_mask, nest)
    else:
        known = _find_galstreams_stream(stream_name, mws, footprint_mask, nest)
        if known is None:
            raise ValueError("Give a track or a galstreams stream_name: the distance follows the track.")
    stream_frame, window = _frame_and_window(known, stream_name, stream_frame, window, stars)
    isochrones = _isochrones(isochrone, known, stream_name, population_table)
    match_filter = DEFAULT_MATCH_FILTER if match_filter is None else match_filter
    stars = _search_stars(catalog, stars, stream_frame, window, match_filter, cuts)
    grid_footprint = _grid_footprint(footprint_mask, stream_frame, window, bin_size_deg, nest)

    phi1_centres, _ = phi12_bin_centres(window, bin_size_deg)
    dm_along_phi1 = galstreams_dm_along_track(known.track, phi1_centres, hold_ends=True)
    if np.isnan(dm_along_phi1).any():
        raise ValueError(f"Track {known.track_name} has no distance in galstreams.")
    dm_along_phi1 = np.round(dm_along_phi1 / dm_step) * dm_step

    # phi1 bin of each star (the last bin may end within half a bin of the window)
    star_bin = np.clip(
        np.floor((stars["phi1"].to_numpy() - window[0]) / bin_size_deg).astype(int), 0, phi1_centres.size - 1
    )
    star_dm = dm_along_phi1[star_bin]
    for isochrone in isochrones:
        for dm_offset in dm_offsets:
            selected = np.zeros(len(stars), dtype=bool)
            for dm in np.unique(star_dm):
                selected |= select_stars_in_match_filter(
                    stars, dm + dm_offset, isochrone, match_filter, mask=star_dm == dm
                )
            result = _map_of_selected(stars, selected, window, bin_size_deg, smoothing_deg, grid_footprint)
            result.update(
                dm_along_phi1=dm_along_phi1 + dm_offset, dm_offset=float(dm_offset),
                isochrone=dict(isochrone), match_filter=match_filter, track_name=known.track_name,
            )
            yield float(dm_offset), result


# ===================================================
# Values from galstreams
# ===================================================


@dataclass(frozen=True)
class _GalstreamsStream:
    """What ``catalogs.galstreams_catalog`` gives for one galstreams track."""

    track_name: str
    stream_name: str  # galstreams stream name: the key of the population table
    track: Any  # galstreams.Track6D
    window: tuple[float, float, float, float]  # galstreams_phi12_window
    dm_min: float  # of the track in the footprint (select_galstreams_in_footprint); NaN without distance
    dm_max: float


def _find_galstreams_stream(stream_name, mws, footprint_mask, nest):
    """The galstreams track of a track name, or the default ('On') track of a stream name; None if neither."""
    if stream_name is None:
        return None
    mws = load_galstreams() if mws is None else mws
    if stream_name in mws:  # a track name
        return _galstreams_stream_of_track(mws[stream_name], footprint_mask, nest)
    summary = mws.summary  # one row per track: its stream ("Name") and default track ("On")
    default_tracks = summary.loc[(summary["Name"] == stream_name) & summary["On"], "TrackName"]
    if default_tracks.empty:
        return None
    return _galstreams_stream_of_track(mws[default_tracks.iloc[0]], footprint_mask, nest)


def _galstreams_stream_of_track(track, footprint_mask, nest):
    """Window and distance moduli of a track in the footprint (the whole sky without a HEALPix footprint)."""
    if footprint_mask is not None and np.ndim(footprint_mask) == 1:
        if nest is None:
            raise ValueError("A HEALPix footprint_mask needs nest.")
        healpix_mask, healpix_nest = footprint_mask, nest
    else:
        healpix_mask, healpix_nest = np.zeros(hp.nside2npix(1), dtype=bool), False  # whole sky
    selection = select_galstreams_in_footprint(
        {track.track_name: track}, healpix_mask, nest=healpix_nest,
        dm_range=None, min_length_deg=0.0, add_ibata2024=True,
    )
    if selection.empty:
        raise ValueError(f"Track {track.track_name} does not cross the footprint.")
    return _GalstreamsStream(
        track_name=track.track_name, stream_name=track.stream_name, track=track,
        window=galstreams_phi12_window(track),
        dm_min=float(selection["dm_min"].iloc[0]), dm_max=float(selection["dm_max"].iloc[0]),
    )


def _frame_and_window(known, stream_name, stream_frame, window, stars):
    """The given frame and window, else those of the galstreams track."""
    if known is not None:
        stream_frame = known.track.stream_frame if stream_frame is None else stream_frame
        window = known.window if window is None else window
    elif stream_name is not None and (window is None or (stream_frame is None and stars is None)):
        raise ValueError(f"{stream_name!r} is not in galstreams: give its stream_frame and window.")
    if window is None:
        raise ValueError("Give a window, or a galstreams stream_name.")
    return stream_frame, tuple(window)


def _distance_moduli_grid(known):
    """dm_min - TRACK_DM_MARGIN to dm_max + TRACK_DM_MARGIN of the track, every DM_STEP; 15 to 22 without."""
    if known is None or np.isnan(known.dm_min):
        return list(DEFAULT_DISTANCE_MODULI)
    low = np.floor((known.dm_min - TRACK_DM_MARGIN) / DM_STEP) * DM_STEP
    high = np.ceil((known.dm_max + TRACK_DM_MARGIN) / DM_STEP) * DM_STEP
    return list(np.round(np.arange(low, high + DM_STEP / 2, DM_STEP), 1))


def _round_dm(distance_modulus):
    return float(np.round(np.round(distance_modulus / DM_STEP) * DM_STEP, 1))


# ===================================================
# Isochrones, stars and footprint of a search
# ===================================================


def _isochrones(isochrone, known, stream_name, population_table):
    """The given isochrone, else those of the stream in the population table, else the default grid."""
    if isochrone is not None:
        return [dict(isochrone)]
    name = known.stream_name if known is not None else stream_name
    if name is None:
        return default_isochrone_grid()
    return known_stream_isochrone_parameters(name, population_table)


def _search_stars(catalog, stars, stream_frame, window, match_filter, cuts):
    """The prepared stars given, else the catalog prepared for this frame and window."""
    if stars is not None:
        return stars
    if catalog is None:
        raise ValueError("Give a catalog or the prepared stars.")
    if stream_frame is None:
        raise ValueError("A catalog needs a stream_frame (or a galstreams stream_name), for phi1/phi2.")
    return prepare_match_filter_catalog(catalog, match_filter, cuts=cuts, stream_frame=stream_frame, window=window)


def _grid_footprint(footprint_mask, stream_frame, window, bin_size_deg, nest):
    """The footprint on the phi1/phi2 grid: a HEALPix mask put on it, a grid mask as it is."""
    if footprint_mask is None or np.ndim(footprint_mask) == 2:
        return footprint_mask
    if nest is None or stream_frame is None:
        raise ValueError("A HEALPix footprint_mask needs nest and a stream_frame.")
    return survey_footprint_phi12_mask(footprint_mask, stream_frame, window, bin_size_deg, nest=nest)


def _map_of_selected(stars, selected, window, bin_size_deg, smoothing_deg, grid_footprint):
    """Map of the selected stars on the grid of the window, with the numbers of stars."""
    density = star_density_phi12(
        stars["phi1"].to_numpy()[selected], stars["phi2"].to_numpy()[selected], window, bin_size_deg,
        smoothing_deg=smoothing_deg, footprint_mask=grid_footprint,
    )
    phi1_centres, phi2_centres = phi12_bin_centres(window, bin_size_deg)
    return {
        "density": density, "phi1_centres": phi1_centres, "phi2_centres": phi2_centres,
        "n_stars": len(stars), "n_selected": int(selected.sum()), "window": tuple(window),
        "bin_size_deg": bin_size_deg, "smoothing_deg": smoothing_deg,
    }


# ===================================================
# Scan: one known_stream_search_single_dm call per (isochrone, distance modulus)
# ===================================================


def _single_dm_on_stars(task, *, stars, window, match_filter, bin_size_deg, smoothing_deg, footprint_mask,
                        track_name):
    """known_stream_search_single_dm of one (isochrone, distance modulus) on the prepared stars."""
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

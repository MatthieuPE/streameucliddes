"""Known-stream search: matched-filter maps of a stream, in its frame.

For one stream and one isochrone: the stars inside the match filter, counted
on a phi1/phi2 grid in the stream frame, for each distance modulus. No
polynomial background is subtracted. Magnitudes are used as they are: the
Euclid x DES magnitudes are already dereddened.

Built from the shared pieces: ``catalog_preparation.prepare_match_filter_catalog``
(columns, cuts, phi1/phi2, window), ``cmd_match_filter.select_stars_in_match_filter``
and ``star_density_maps.star_density_phi12``. The catalog is prepared once per
call: pass ``stars``, the prepared catalog, to reuse it (e.g. for several
isochrones). Any stream frame works, from galstreams or not.

Mirrors ``generic_stream_search``: ``_single_dm`` gives one map; ``_dm_scan``
goes over a grid of distance moduli one at a time (it yields ``(dm,
result)``), and the workflows build the stacks.
"""

from __future__ import annotations

import functools
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from streameucliddes.analysis.catalog_preparation import prepare_match_filter_catalog
from streameucliddes.analysis.cmd_match_filter import DEFAULT_MATCH_FILTER, select_stars_in_match_filter
from streameucliddes.analysis.star_density_maps import star_density_phi12
from streameucliddes.catalogs.galstreams_catalog import galstreams_dm_along_track
from streameucliddes.catalogs.isochrone_parameters import default_isochrone_grid
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


def known_stream_search_single_dm(
    catalog: pd.DataFrame | None,
    stream_frame: coord.BaseCoordinateFrame | None,
    window: tuple[float, float, float, float],
    distance_modulus: float,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    stars: pd.DataFrame | None = None,
    cuts: Mapping[str, Any] | None = None,
    bin_size_deg: float = DEFAULT_BIN_SIZE_DEG,
    smoothing_deg: float = DEFAULT_SMOOTHING_DEG,
    footprint_mask: NDArray[np.bool_] | None = None,
) -> dict[str, Any]:
    """Matched-filter map of a stream at one distance modulus.

    Prepares the catalog (``catalog_preparation.prepare_match_filter_catalog``),
    selects the stars in the match filter of one isochrone
    (``cmd_match_filter.select_stars_in_match_filter``), and counts them on
    the phi1/phi2 grid (``star_density_maps.star_density_phi12``).

    Parameters
    ----------
    catalog : pandas.DataFrame or None
        Stars, with ``ra``, ``dec`` (deg) and the magnitude columns of the
        match filter, e.g. ``catalogs.euclid_des_catalog.get_eucliddes_stars()``.
        None when ``stars`` is given.
    stream_frame : astropy.coordinates.BaseCoordinateFrame or None
        Stream frame, e.g. the ``stream_frame`` of a galstreams track (a gala
        ``GreatCircleICRSFrame``), or any frame for a stream not in
        galstreams. None when ``stars`` is given.
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` of the map (deg), e.g.
        from ``catalogs.galstreams_catalog.galstreams_phi12_window``.
    distance_modulus : float
        Distance modulus of the match filter.
    isochrone : mapping, optional
        ``{"age": age (Gyr), "metallicity": Z}``, e.g. an item of
        ``catalogs.isochrone_parameters.known_stream_isochrone_parameters``.
        Default: the first isochrone of
        ``catalogs.isochrone_parameters.default_isochrone_grid``.
    match_filter : mapping, optional
        See ``cmd_match_filter``. Default: ``cmd_match_filter.DEFAULT_MATCH_FILTER``.
    stars : pandas.DataFrame, optional
        ``catalog`` already prepared by ``prepare_match_filter_catalog`` with
        this frame, window, match filter and cuts (``phi1``, ``phi2`` and
        magnitude columns): used instead of ``catalog``.
    cuts : mapping, optional
        Cuts of the preparation (default: errors <= 0.2 on the bands of the
        match filter). Ignored with ``stars``.
    bin_size_deg : float, optional
        Bin size of the grid (deg).
    smoothing_deg : float, optional
        Standard deviation of the Gaussian smoothing (deg).
    footprint_mask : numpy.ndarray of bool, optional
        From ``catalogs.survey_footprint.survey_footprint_phi12_mask``, True
        outside the footprint.

    Returns
    -------
    result : dict
        ``density`` (masked array, shape (n_phi1, n_phi2)), ``phi1_centres``
        and ``phi2_centres`` (the bins), ``n_stars`` (stars in the window),
        ``n_selected`` (in the match filter), ``distance_modulus``,
        ``distance_kpc``, ``window``, ``bin_size_deg``, ``smoothing_deg``,
        ``isochrone`` and ``match_filter``.

    Raises
    ------
    ValueError
        Neither ``catalog`` nor ``stars``, or a catalog without
        ``stream_frame``.

    References
    ----------
    Rockosi et al. 2002, AJ 124, 349. Grillmair 2009, ApJ 693, 1118.
    Shipp et al. 2018, ApJ 862, 114.
    """
    isochrone, match_filter = _resolve_isochrone_match_filter(isochrone, match_filter)
    stars = _prepared_stars(catalog, stream_frame, window, match_filter, cuts, stars)
    return _single_dm_map(
        stars, window, distance_modulus, isochrone, match_filter,
        bin_size_deg=bin_size_deg, smoothing_deg=smoothing_deg, footprint_mask=footprint_mask,
    )


def known_stream_search_dm_scan(
    catalog: pd.DataFrame | None,
    stream_frame: coord.BaseCoordinateFrame | None,
    window: tuple[float, float, float, float],
    distance_moduli: Iterable[float],
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    stars: pd.DataFrame | None = None,
    cuts: Mapping[str, Any] | None = None,
    bin_size_deg: float = DEFAULT_BIN_SIZE_DEG,
    smoothing_deg: float = DEFAULT_SMOOTHING_DEG,
    footprint_mask: NDArray[np.bool_] | None = None,
    n_workers: int = 1,
) -> Iterator[tuple[float, dict[str, Any]]]:
    """Matched-filter maps of a stream over a grid of distance moduli.

    The catalog is prepared once (columns, cuts, phi1/phi2, window), then
    each distance modulus gives one map (see
    ``known_stream_search_single_dm``), yielded one at a time: the caller
    can save or plot it and drop it before the next one.

    Parameters
    ----------
    catalog, stream_frame, window, isochrone, match_filter
        See ``known_stream_search_single_dm``.
    distance_moduli : iterable of float
        Grid of distance moduli.
    stars, cuts, bin_size_deg, smoothing_deg, footprint_mask
        See ``known_stream_search_single_dm``.
    n_workers : int, optional
        Processes computing the distance moduli in parallel (1: none). Each
        process receives the prepared stars once. The results are still
        yielded in the order of ``distance_moduli``.

    Yields
    ------
    distance_modulus : float
        One distance modulus of the grid.
    result : dict
        ``known_stream_search_single_dm`` output for it.

    Notes
    -----
    Processes pay off for large windows: a map takes about 0.15 s for
    300 000 stars in the window, and a process takes a few seconds to start.
    """
    isochrone, match_filter = _resolve_isochrone_match_filter(isochrone, match_filter)
    stars = _prepared_stars(catalog, stream_frame, window, match_filter, cuts, stars)
    map_kwargs = {"bin_size_deg": bin_size_deg, "smoothing_deg": smoothing_deg, "footprint_mask": footprint_mask}
    distance_moduli = [float(dm) for dm in distance_moduli]

    if n_workers <= 1:
        for dm in distance_moduli:
            yield dm, _single_dm_map(stars, window, dm, isochrone, match_filter, **map_kwargs)
        return

    worker = functools.partial(
        _worker_single_dm_map, window=window, isochrone=isochrone, match_filter=match_filter, **map_kwargs
    )
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_set_worker_stars, initargs=(stars,)) as executor:
        yield from zip(distance_moduli, executor.map(worker, distance_moduli))


def known_stream_search_dm_along_track(
    catalog: pd.DataFrame | None,
    track: galstreams.Track6D,
    stream_frame: coord.BaseCoordinateFrame | None,
    window: tuple[float, float, float, float],
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    stars: pd.DataFrame | None = None,
    dm_offsets: Iterable[float] = (0.0,),
    dm_step: float = 0.1,
    cuts: Mapping[str, Any] | None = None,
    bin_size_deg: float = DEFAULT_BIN_SIZE_DEG,
    smoothing_deg: float = DEFAULT_SMOOTHING_DEG,
    footprint_mask: NDArray[np.bool_] | None = None,
) -> Iterator[tuple[float, dict[str, Any]]]:
    """Matched-filter maps that follow the distance of the track along phi1.

    In each phi1 bin, the match filter is at the distance modulus of the
    track there (``catalogs.galstreams_catalog.galstreams_dm_along_track``)
    plus an offset: one map per offset. The distance moduli of the track are
    rounded to ``dm_step``, so that the stars of the phi1 bins sharing a
    distance modulus go through one filter. Beyond the ends of the track,
    the distance modulus of the nearest end is used.

    Parameters
    ----------
    catalog, window, isochrone, match_filter
        See ``known_stream_search_single_dm``.
    track : galstreams.Track6D
        Track of the stream, from ``catalogs.galstreams_catalog.load_galstreams``.
    stream_frame : astropy.coordinates.BaseCoordinateFrame or None
        Frame of the maps; its phi1 must be the track's. Default (None):
        ``track.stream_frame``.
    stars : pandas.DataFrame, optional
        See ``known_stream_search_single_dm``.
    dm_offsets : iterable of float, optional
        Offsets added to the distance modulus of the track.
    dm_step : float, optional
        Rounding step of the distance moduli of the track.
    cuts, bin_size_deg, smoothing_deg, footprint_mask
        See ``known_stream_search_single_dm``.

    Yields
    ------
    dm_offset : float
        One offset of ``dm_offsets``.
    result : dict
        As ``known_stream_search_single_dm``, with ``dm_along_phi1`` (the
        distance modulus of each phi1 bin, offset included) and
        ``dm_offset`` instead of ``distance_modulus`` and ``distance_kpc``.

    Raises
    ------
    ValueError
        The track has no distance (e.g. the Ibata et al. 2024 tracks).
    """
    stream_frame = track.stream_frame if stream_frame is None else stream_frame
    isochrone, match_filter = _resolve_isochrone_match_filter(isochrone, match_filter)
    stars = _prepared_stars(catalog, stream_frame, window, match_filter, cuts, stars)

    phi1_centres, phi2_centres = phi12_bin_centres(window, bin_size_deg)
    track_dm = galstreams_dm_along_track(track, phi1_centres)
    known = np.isfinite(track_dm)
    if not known.any():
        raise ValueError(f"Track {getattr(track, 'track_name', track)} has no distance in the window.")
    # Nearest end beyond the track: np.interp holds the end values
    track_dm = np.interp(phi1_centres, phi1_centres[known], track_dm[known])
    track_dm = np.round(track_dm / dm_step) * dm_step

    # phi1 bin of each star (stars of the window: the last bin may end within half a bin of it)
    star_bin = np.clip(
        np.floor((stars["phi1"].to_numpy() - window[0]) / bin_size_deg).astype(int), 0, phi1_centres.size - 1
    )
    star_dm = track_dm[star_bin]
    for dm_offset in dm_offsets:
        selected = np.zeros(len(stars), dtype=bool)
        for dm in np.unique(star_dm):
            same_dm = star_dm == dm
            selected |= select_stars_in_match_filter(stars, dm + dm_offset, isochrone, match_filter, mask=same_dm)
        result = _map_of_selected(stars, selected, window, bin_size_deg, smoothing_deg, footprint_mask)
        result.update(
            phi1_centres=phi1_centres, phi2_centres=phi2_centres,
            dm_along_phi1=track_dm + dm_offset, dm_offset=float(dm_offset),
            window=tuple(window), bin_size_deg=bin_size_deg, smoothing_deg=smoothing_deg,
            isochrone=dict(isochrone), match_filter=match_filter,
        )
        yield float(dm_offset), result


# ===================================================
# Helpers
# ===================================================


def _resolve_isochrone_match_filter(isochrone, match_filter):
    """The given isochrone and match filter, or the defaults."""
    isochrone = default_isochrone_grid()[0] if isochrone is None else isochrone
    match_filter = DEFAULT_MATCH_FILTER if match_filter is None else match_filter
    return isochrone, match_filter


def _prepared_stars(catalog, stream_frame, window, match_filter, cuts, stars):
    """``stars`` when given, else the catalog prepared for this frame and window."""
    if stars is not None:
        return stars
    if catalog is None:
        raise ValueError("Give a catalog (with its stream_frame) or the prepared stars.")
    if stream_frame is None:
        raise ValueError("A catalog needs a stream_frame, to compute phi1/phi2.")
    return prepare_match_filter_catalog(catalog, match_filter, cuts=cuts, stream_frame=stream_frame, window=window)


def _map_of_selected(stars, selected, window, bin_size_deg, smoothing_deg, footprint_mask):
    """Map of the selected stars, with the numbers of stars."""
    density = star_density_phi12(
        stars["phi1"].to_numpy()[selected], stars["phi2"].to_numpy()[selected], window, bin_size_deg,
        smoothing_deg=smoothing_deg, footprint_mask=footprint_mask,
    )
    return {"density": density, "n_stars": len(stars), "n_selected": int(selected.sum())}


def _single_dm_map(stars, window, distance_modulus, isochrone, match_filter, *, bin_size_deg, smoothing_deg,
                   footprint_mask):
    """Map at one distance modulus of the prepared stars."""
    selected = select_stars_in_match_filter(stars, distance_modulus, isochrone, match_filter)
    result = _map_of_selected(stars, selected, window, bin_size_deg, smoothing_deg, footprint_mask)
    phi1_centres, phi2_centres = phi12_bin_centres(window, bin_size_deg)
    result.update(
        phi1_centres=phi1_centres, phi2_centres=phi2_centres,
        distance_modulus=float(distance_modulus), distance_kpc=float(convert_DM_to_kpc(distance_modulus).value),
        window=tuple(window), bin_size_deg=bin_size_deg, smoothing_deg=smoothing_deg,
        isochrone=dict(isochrone), match_filter=match_filter,
    )
    return result


# Prepared stars of a worker process, set once by its initializer
_WORKER_STARS = None


def _set_worker_stars(stars):
    global _WORKER_STARS
    _WORKER_STARS = stars


def _worker_single_dm_map(distance_modulus, **kwargs):
    return _single_dm_map(_WORKER_STARS, distance_modulus=distance_modulus, **kwargs)

"""Generic stream search over the survey footprint: match filter, smoothed HEALPix map,
polynomial background, residual.

Mirrors ``analysis.known_stream_search``: the search is prepared once
(``prepare_generic_search``), then one map per distance modulus and isochrone
(``generic_search_single_dm``, ``generic_search_dm_scan``). The match filter
is the shared one (``analysis.cmd_match_filter``, default Euclid VIS vs
VIS - H, ``<namespace>_<band>_obs`` columns used as they are: the Euclid x DES
magnitudes are already dereddened).

Every star of the catalog is in the search, and the catalog may be huge: it
is neither copied nor modified. The preparation keeps, per star, whether it
passes the cuts (1 byte,
``catalog_preparation.prepare_match_filter_catalog`` with ``as_mask``) and
its HEALPix pixel (4 bytes, ``catalog_preparation.star_healpix_pixels``);
each map then reads
the magnitude columns by chunks of ``CHUNK_SIZE`` stars (views) and counts the
selected stars in their pixels.

HEALPix maps are in RING ordering (``nest=False``), as ``healpy.smoothing``
needs.
"""

from __future__ import annotations

import collections
import functools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
import pandas as pd

from streameucliddes.analysis.background_polynomial_fit import fit_background_model
from streameucliddes.analysis.build_masks import DEFAULT_RADIUS_SCALES, overlapped_objects_healpix_mask
from streameucliddes.analysis.catalog_preparation import (
    CHUNK_SIZE,
    prepare_match_filter_catalog,
    star_healpix_pixels,
)
from streameucliddes.analysis.cmd_match_filter import (
    DEFAULT_MATCH_FILTER,
    match_filter_columns,
    select_stars_in_match_filter,
)
from streameucliddes.analysis.known_stream_search import DEFAULT_DISTANCE_MODULI
from streameucliddes.catalogs.isochrone_parameters import default_isochrone_grid
from streameucliddes.catalogs.survey_footprint import (
    convert_footprint_healpix_mask,
    survey_footprint_healpix_mask,
)
from streameucliddes.unit_conversions import convert_DM_to_kpc

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from typing import Any

    from numpy.typing import NDArray

DEFAULT_NSIDE = 512  # pixels of about 0.11 deg
DEFAULT_SMOOTHING_DEG = 0.3  # Gaussian sigma (deg), as the default of star_density_healpix
DEFAULT_POLYNOMIAL_DEGREE = 5  # fixed degree for now (plan, section 1)
# Stars drawn for a CMD of the search (its shape, not its counts): bounds the copy
CMD_SAMPLE_SIZE = 200_000
CMD_SAMPLE_SEED = 0


# ===================================================
# Preparation
# ===================================================


@dataclass(frozen=True)
class PreparedGenericSearch:
    """A generic search prepared once, reused for every distance modulus and isochrone.

    Attributes
    ----------
    catalog : pandas.DataFrame
        The catalog itself (a reference, not a copy).
    keep : numpy.ndarray of bool
        Stars passing the cuts, with finite coordinates.
    pixels : numpy.ndarray of int32
        HEALPix pixel (RING, ``nside``) of each star; -1 without coordinates.
    nside : int
    footprint_mask : numpy.ndarray of bool
        HEALPix mask (RING), True outside the footprint or on a masked
        object (``object_mask``): the pixels left out of the maps.
    object_mask : numpy.ndarray of bool
        HEALPix mask (RING) of the masked objects (LMC, SMC, dwarf
        galaxies, globular clusters, bright galaxies; ``analysis.build_masks``);
        all False when none.
    footprint_centre_lon : float
        Mean longitude of the footprint (deg): the background polynomial is
        fitted on the longitude around it, so that ra = 0 does not cut the
        footprint.
    match_filter : dict
    isochrones : list of dict
    distance_moduli : list of float
    """

    catalog: Any
    keep: Any
    pixels: Any
    nside: int
    footprint_mask: Any
    object_mask: Any
    footprint_centre_lon: float
    match_filter: dict
    isochrones: list
    distance_moduli: list
    # smoothed footprint, per smoothing scale: the same for every map
    _smoothed_footprint: dict = field(default_factory=dict, compare=False, repr=False)


def prepare_generic_search(
    catalog: pd.DataFrame,
    *,
    match_filter: Mapping[str, Any] | None = None,
    isochrone: Mapping[str, float] | None = None,
    default_grid: Iterable[Mapping[str, float]] | None = None,
    distance_moduli: Iterable[float] | None = None,
    cuts: Mapping[str, Any] | None = None,
    nside: int = DEFAULT_NSIDE,
    footprint_mask: NDArray[np.bool_] | None = None,
    nest: bool | None = None,
    mask_lmc: bool = True,
    mask_smc: bool = True,
    mask_dwarfs: bool = True,
    mask_gc: bool = True,
    mask_galaxies: bool = True,
    mask_radius_scale: float | Mapping[str, float] = DEFAULT_RADIUS_SCALES,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> PreparedGenericSearch:
    """Prepare a generic search: the stars kept, their pixels, the footprint.

    The catalog is neither copied nor modified: its columns are read by
    chunks of ``CHUNK_SIZE`` stars. Kept per star: whether it passes the cuts
    (1 byte: ``analysis.catalog_preparation.prepare_match_filter_catalog``
    with ``as_mask``), and its HEALPix pixel (4 bytes:
    ``analysis.catalog_preparation.star_healpix_pixels``). The stars without
    finite coordinates are not kept.

    Objects that overlap the footprint (the LMC, the SMC, the dwarf
    galaxies, the globular clusters, the bright galaxies) are masked by default
    (``analysis.build_masks.overlapped_objects_healpix_mask``): their pixels
    are added to the footprint mask, and their stars are not kept
    (``prepare_match_filter_catalog`` with ``excluded_pixels``). No star
    is counted there, the smoothed counts are corrected at the edges of the
    masks as at those of the footprint, and the background is fitted on
    the other pixels.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars, with ``ra_col``, ``dec_col`` (deg) and the magnitude columns of
        the match filter, e.g. ``catalogs.euclid_catalog.get_euclid_stars()``.
    match_filter : mapping, optional
        See ``analysis.cmd_match_filter``. Default:
        ``cmd_match_filter.DEFAULT_MATCH_FILTER``.
    isochrone : mapping, optional
        The one isochrone of the search, ``{"age": age (Gyr), "metallicity": Z}``.
    default_grid : iterable of mapping, optional
        The isochrones otherwise. Default:
        ``catalogs.isochrone_parameters.default_isochrone_grid()``.
    distance_moduli : iterable of float, optional
        Grid of the scan. Default: ``known_stream_search.DEFAULT_DISTANCE_MODULI``
        (15 to 22, every 0.1).
    cuts : mapping, optional
        Format of ``catalogs.catalog_cuts.get_cut_mask``. They update the
        default cuts, errors <= 0.2 on the bands of the match filter
        (``catalog_preparation.default_error_cuts``): a cut on an error column
        replaces its default one, ``(None, None)`` removes it.
    nside : int, optional
        HEALPix resolution of the maps.
    footprint_mask : numpy.ndarray of bool, optional
        HEALPix mask, True outside the footprint (needs ``nest``), e.g.
        ``catalogs.survey_footprint.survey_footprint_healpix_mask``; of any
        nside and ordering, put in RING at ``nside``
        (``catalogs.survey_footprint.convert_footprint_healpix_mask``: a
        pixel is outside if any part of it is). Default:
        the pixels with at least one star of the catalog, cut or not
        (``survey_footprint_healpix_mask``, by chunks).
    nest : bool, optional
        HEALPix ordering of ``footprint_mask``.
    mask_lmc, mask_smc, mask_dwarfs, mask_gc, mask_galaxies : bool, optional
        Mask the LMC, the SMC, the dwarf galaxies (but the Clouds), the
        globular clusters (LVDB: half-light ellipses), the bright galaxies
        (Siena Galaxy Atlas: circles of D26 / 2), scaled by
        ``mask_radius_scale`` (``catalogs.overlapped_objects``; downloaded
        at the first use). False: not masked.
    mask_radius_scale : float or mapping, optional
        Semi-major axis of the ellipses, in units of the reference radius of
        the objects: the same for every type, or per type, ``{"lmc": ...,
        "smc": ..., "dwarfs": ..., "gc": ..., "galaxies": ...}`` (default:
        6.5 for the LMC, 21 deg; 11 for the SMC, 11 deg; 4 for the dwarfs
        and clusters; 5 for the galaxies; see
        ``analysis.build_masks.overlapped_objects_healpix_mask``).
    ra_col, dec_col : str, optional
        Coordinate columns (deg).

    Returns
    -------
    search : PreparedGenericSearch

    Raises
    ------
    KeyError
        Columns missing from the catalog (all of them listed).
    ValueError
        ``footprint_mask`` without ``nest``, an empty footprint, or a
        footprint all masked by the objects.
    """
    match_filter = dict(DEFAULT_MATCH_FILTER if match_filter is None else match_filter)
    object_mask = overlapped_objects_healpix_mask(
        nside, nest=False, mask_lmc=mask_lmc, mask_smc=mask_smc, mask_dwarfs=mask_dwarfs, mask_gc=mask_gc,
        mask_galaxies=mask_galaxies, radius_scale=mask_radius_scale,
    )
    keep = prepare_match_filter_catalog(  # 1 byte per star; not the stars on the masked objects
        catalog, match_filter, cuts=cuts, excluded_pixels=object_mask, excluded_pixels_nest=False,
        ra_col=ra_col, dec_col=dec_col, as_mask=True,
    )
    pixels = star_healpix_pixels(catalog, nside, nest=False, ra_col=ra_col, dec_col=dec_col)  # 4 bytes per star
    for start in range(0, len(catalog), CHUNK_SIZE):  # finite coordinates; temporaries of a chunk only
        keep[start:start + CHUNK_SIZE] &= pixels[start:start + CHUNK_SIZE] >= 0

    if footprint_mask is None:
        footprint_mask = survey_footprint_healpix_mask(
            catalog[ra_col].to_numpy(), catalog[dec_col].to_numpy(), nside, nest=False
        )
    else:
        if nest is None:
            raise ValueError("A footprint_mask needs nest.")
        # in RING at nside; a pixel is outside if any part of it is
        footprint_mask = convert_footprint_healpix_mask(footprint_mask, nside, nest=nest, nest_out=False)
    if footprint_mask.all():
        raise ValueError("Empty footprint: no pixel inside.")
    inside_lon, _ = hp.pix2ang(nside, np.flatnonzero(~footprint_mask), lonlat=True)
    centre_lon = np.degrees(np.arctan2(np.sin(np.radians(inside_lon)).mean(), np.cos(np.radians(inside_lon)).mean()))
    footprint_mask = footprint_mask | object_mask  # the pixels left out of the maps
    if footprint_mask.all():
        raise ValueError(
            "Footprint all masked by the objects (mask_lmc, mask_smc, mask_dwarfs, mask_gc, mask_galaxies)."
        )

    if isochrone is not None:
        isochrones = [dict(isochrone)]
    else:
        isochrones = default_isochrone_grid() if default_grid is None else [dict(item) for item in default_grid]
    distance_moduli = DEFAULT_DISTANCE_MODULI if distance_moduli is None else distance_moduli
    return PreparedGenericSearch(
        catalog=catalog, keep=keep, pixels=pixels, nside=nside, footprint_mask=footprint_mask,
        object_mask=object_mask, footprint_centre_lon=float(centre_lon % 360.0), match_filter=match_filter,
        isochrones=list(isochrones),
        distance_moduli=[float(dm) for dm in distance_moduli],
    )


# ===================================================
# Searches
# ===================================================


def generic_search_single_dm(
    catalog: pd.DataFrame | None = None,
    distance_modulus: float | None = None,
    isochrone: Mapping[str, float] | None = None,
    *,
    search: PreparedGenericSearch | None = None,
    smoothing_deg: float = DEFAULT_SMOOTHING_DEG,
    polynomial_degree: int = DEFAULT_POLYNOMIAL_DEGREE,
    **prepare_kwargs: Any,
) -> dict[str, Any]:
    """Matched-filter HEALPix map at one distance modulus, its background and residual.

    The stars in the match filter
    (``analysis.cmd_match_filter.select_stars_in_match_filter``) are counted
    in their pixels, by chunks. The counts are smoothed with a Gaussian and
    divided by the smoothed footprint: no drop at the footprint edges. A 2D
    polynomial of the log density
    (``analysis.background_polynomial_fit.fit_background_model``, method
    ``"polyfit2d"``), on (longitude around the footprint centre, latitude),
    gives the background; the residual is the density minus the background.

    Parameters
    ----------
    catalog : pandas.DataFrame, optional
        Stars: prepared here (``prepare_generic_search``). Not with ``search``.
    distance_modulus : float
        Distance modulus of the match filter.
    isochrone : mapping, optional
        ``{"age", "metallicity"}``. Default: the first isochrone of the search.
    search : PreparedGenericSearch, optional
        The search, prepared once for many maps.
    smoothing_deg : float, optional
        Gaussian sigma (deg); 0 for no smoothing.
    polynomial_degree : int, optional
        Degree of the background polynomial (in each coordinate).
    **prepare_kwargs
        Arguments of ``prepare_generic_search``. Not with ``search``.

    Returns
    -------
    result : dict
        ``density``, ``background``, ``residual``: HEALPix maps (RING, masked
        outside the footprint; stars per pixel); ``footprint_mask``,
        ``nside``, ``nest`` (False); ``n_selected``; ``distance_modulus``,
        ``distance_kpc``, ``isochrone``, ``match_filter``, ``smoothing_deg``,
        ``polynomial_degree``.

    Raises
    ------
    ValueError
        No ``distance_modulus``; preparation arguments with ``search``.

    References
    ----------
    Rockosi et al. 2002, AJ 124, 349. Grillmair 2009, ApJ 693, 1118.
    Shipp et al. 2018, ApJ 862, 114.
    """
    search = _search(search, catalog, prepare_kwargs)
    if distance_modulus is None:
        raise ValueError("Give a distance_modulus.")
    isochrone = dict(search.isochrones[0] if isochrone is None else isochrone)

    counts = _selected_counts(search, distance_modulus, isochrone)
    density = _smoothed_density(counts, search, smoothing_deg)
    background = _background(density, search, polynomial_degree)
    return {
        "density": density,
        "background": background,
        "residual": density - background,
        "footprint_mask": search.footprint_mask,
        "nside": search.nside,
        "nest": False,
        "n_selected": int(counts.sum()),
        "distance_modulus": float(distance_modulus),
        "distance_kpc": float(convert_DM_to_kpc(distance_modulus).value),
        "isochrone": isochrone,
        "match_filter": search.match_filter,
        "smoothing_deg": float(smoothing_deg),
        "polynomial_degree": int(polynomial_degree),
    }


def generic_search_dm_scan(
    catalog: pd.DataFrame | None = None,
    distance_moduli: Iterable[float] | None = None,
    isochrone: Mapping[str, float] | None = None,
    *,
    search: PreparedGenericSearch | None = None,
    smoothing_deg: float = DEFAULT_SMOOTHING_DEG,
    polynomial_degree: int = DEFAULT_POLYNOMIAL_DEGREE,
    n_workers: int = 1,
    **prepare_kwargs: Any,
) -> Iterator[tuple[float, dict[str, Any]]]:
    """Generic search over a grid of distance moduli, per isochrone, one map at a time.

    ``generic_search_single_dm`` for every isochrone of the search (or
    ``isochrone``), every distance modulus for the first, then the next.
    The maps are yielded one at a time: save or plot each, then drop it.

    With ``n_workers`` > 1, the maps are computed in parallel by forked
    processes: they share the catalog of the parent (copy on write), not a
    copy of it; at most ``n_workers`` maps wait in memory to be yielded.

    Parameters
    ----------
    catalog : pandas.DataFrame, optional
        Stars: prepared here (``prepare_generic_search``). Not with ``search``.
    distance_moduli : iterable of float, optional
        Default: those of the search.
    isochrone : mapping, optional
        The one isochrone. Default: those of the search.
    search : PreparedGenericSearch, optional
        The search, prepared once.
    smoothing_deg, polynomial_degree
        See ``generic_search_single_dm``.
    n_workers : int, optional
        Processes over the maps (needs the "fork" start method: Linux, macOS).
    **prepare_kwargs
        Arguments of ``prepare_generic_search``. Not with ``search``.

    Yields
    ------
    distance_modulus : float
    result : dict
        Output of ``generic_search_single_dm``.
    """
    search = _search(search, catalog, prepare_kwargs)
    distance_moduli = search.distance_moduli if distance_moduli is None else [float(dm) for dm in distance_moduli]
    isochrones = search.isochrones if isochrone is None else [dict(isochrone)]
    tasks = [(isochrone, dm) for isochrone in isochrones for dm in distance_moduli]
    single_dm = functools.partial(_worker_single_dm, smoothing_deg=smoothing_deg, polynomial_degree=polynomial_degree)

    if n_workers <= 1:
        for isochrone, dm in tasks:
            yield dm, generic_search_single_dm(
                distance_modulus=dm, isochrone=isochrone, search=search,
                smoothing_deg=smoothing_deg, polynomial_degree=polynomial_degree,
            )
        return

    global _WORKER_SEARCH
    _WORKER_SEARCH = search  # inherited by the forked workers, not pickled
    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("fork")) as executor:
            pending = collections.deque()
            for task in tasks:
                pending.append((task[1], executor.submit(single_dm, task)))
                if len(pending) >= n_workers:  # bounded: the maps not yet yielded stay few
                    dm, future = pending.popleft()
                    yield dm, future.result()
            while pending:
                dm, future = pending.popleft()
                yield dm, future.result()
    finally:
        _WORKER_SEARCH = None



def generic_search_cmd_stars(
    search: PreparedGenericSearch, max_stars: int = CMD_SAMPLE_SIZE
) -> pd.DataFrame:
    """Stars kept by a search, at most ``max_stars``: for a CMD of the search.

    For figures only: the CMD panel of
    ``plotting.plot_generic_search.plot_generic_search_result``, which shows
    where the match filter falls on the colour-magnitude distribution of the
    stars. The maps do not use it: they count every star passing the cuts
    (``generic_search_single_dm``), whatever ``max_stars``.

    Each star that passes the cuts is drawn with the same probability (about
    ``max_stars`` drawn; a fixed seed, ``CMD_SAMPLE_SEED``: the same stars at
    each call), by chunks. Only the magnitude columns of the match filter are
    copied: ``max_stars`` rows at most, whatever the size of the catalog. E.g.
    ``plotting.plot_cmd.plot_match_filter(..., stars=generic_search_cmd_stars(search))``.

    Parameters
    ----------
    search : PreparedGenericSearch
    max_stars : int, optional
        Maximum number of stars.

    Returns
    -------
    stars : pandas.DataFrame
        The magnitude columns of the match filter, with the index of the
        catalog.
    """
    probability = min(1.0, max_stars / max(int(search.keep.sum()), 1))
    rng = np.random.default_rng(CMD_SAMPLE_SEED)
    parts = []
    for start in range(0, search.keep.size, CHUNK_SIZE):
        keep = search.keep[start:start + CHUNK_SIZE]
        parts.append(np.flatnonzero(keep & (rng.random(keep.size) < probability)) + start)
    rows = np.concatenate(parts)[:max_stars] if parts else np.empty(0, dtype=np.int64)
    columns = match_filter_columns(search.match_filter)
    return pd.DataFrame(
        {column: search.catalog[column].to_numpy()[rows] for column in columns},
        index=search.catalog.index.take(rows),
        copy=False,
    )


# ===================================================
# Helpers
# ===================================================

_WORKER_SEARCH = None  # the search of the forked workers of generic_search_dm_scan


def _worker_single_dm(task, *, smoothing_deg, polynomial_degree):
    """One map in a worker, from the search inherited at the fork."""
    isochrone, dm = task
    return generic_search_single_dm(
        distance_modulus=dm, isochrone=isochrone, search=_WORKER_SEARCH,
        smoothing_deg=smoothing_deg, polynomial_degree=polynomial_degree,
    )


def _search(search, catalog, prepare_kwargs):
    """The prepared search: given, or prepared from the catalog."""
    if search is None:
        if catalog is None:
            raise ValueError("Give a catalog or a prepared search.")
        return prepare_generic_search(catalog, **prepare_kwargs)
    if catalog is not None or prepare_kwargs:
        given = (["catalog"] if catalog is not None else []) + list(prepare_kwargs)
        raise ValueError(f"{given} prepare a search: not with search")
    return search


def _selected_counts(search, distance_modulus, isochrone):
    """Stars in the match filter per HEALPix pixel, counted by chunks of the catalog."""
    npix = hp.nside2npix(search.nside)
    counts = np.zeros(npix)
    for start in range(0, len(search.catalog), CHUNK_SIZE):
        chunk = slice(start, start + CHUNK_SIZE)
        keep = search.keep[chunk]
        if not keep.any():
            continue
        selected = select_stars_in_match_filter(
            search.catalog.iloc[chunk], distance_modulus, isochrone, search.match_filter, mask=keep
        )
        counts += np.bincount(search.pixels[chunk][selected], minlength=npix)
    return counts


def _smoothed_density(counts, search, smoothing_deg):
    """Counts smoothed and divided by the smoothed footprint, masked outside the footprint."""
    outside = search.footprint_mask
    if smoothing_deg <= 0:
        return np.ma.array(counts, mask=outside)
    sigma = np.radians(smoothing_deg)
    if smoothing_deg not in search._smoothed_footprint:
        search._smoothed_footprint[smoothing_deg] = hp.smoothing((~outside).astype(float), sigma=sigma)
    weight = search._smoothed_footprint[smoothing_deg]
    smoothed = hp.smoothing(np.where(outside, 0.0, counts), sigma=sigma)
    density = np.divide(smoothed, weight, out=np.zeros_like(smoothed), where=~outside & (weight > 0))
    return np.ma.array(density, mask=outside)


class _FootprintCentredProj:
    """(longitude around the footprint centre, wrapped to [-180, 180), latitude), in deg.

    The ``proj`` of ``fit_background_model``: ra = 0 does not cut a footprint across it.
    """

    def __init__(self, centre_lon):
        self.centre_lon = centre_lon

    def ang2xy(self, lon, lat, lonlat=True):
        x = (np.asarray(lon, dtype=float) - self.centre_lon + 180.0) % 360.0 - 180.0
        return x, np.asarray(lat, dtype=float)

    def get_extent(self):
        return (-180.0, 180.0, -90.0, 90.0)


def _background(density, search, polynomial_degree):
    """2D polynomial background of a density map, masked outside the footprint."""
    _, _, prepared = fit_background_model(
        density,
        method="polyfit2d",
        model_type={"polynomial": {"degree": int(polynomial_degree)}},
        proj=_FootprintCentredProj(search.footprint_centre_lon),
        nside=search.nside,
        nest=False,
        sigma=0.0,
        percent=None,
        mask=search.footprint_mask,
    )
    return np.ma.array(prepared["background_map"].filled(np.nan), mask=search.footprint_mask)

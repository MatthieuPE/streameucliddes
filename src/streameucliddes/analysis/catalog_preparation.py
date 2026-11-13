"""Preparation of the star catalogs for the searches, before the match filter.

Columns checked, magnitude columns resolved (dust-corrected or not), cuts,
stream-frame coordinates and window, HEALPix pixels of the stars. The
selection itself is in ``analysis.cmd_match_filter``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
import pandas as pd

from streameucliddes.analysis.cmd_match_filter import DEFAULT_MATCH_FILTER, match_filter_columns
from streameucliddes.catalogs import dust_correction
from streameucliddes.catalogs.catalog_cuts import get_cut_mask
from streameucliddes.stream_frames import phi12_to_radec, radec_to_phi12

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any

    import astropy.coordinates as coord
    from numpy.typing import NDArray

# Default cut on the errors of the match-filter bands (as the SNR forecasts of injected_stream_snr)
DEFAULT_ERR_CUT = (None, 0.2)
# HEALPix pixels (about 0.9 deg) of the pre-selection of the stars near a phi1/phi2 window
WINDOW_PRESELECTION_NSIDE = 64
# Stars per chunk of the cuts, the window pre-selection, the phi1/phi2 transformation and the
# HEALPix pixels: bounds their temporary arrays
CHUNK_SIZE = 500_000


# ===================================================
# Magnitude columns and dust correction (generic search)
# ===================================================


def resolve_mag_columns(data, mag_cols, use_dust_corrected):
    """Swap in '<col>_corrected' columns when available and requested."""
    resolved = []
    for col in mag_cols:
        corrected = f"{col}_corrected"
        if use_dust_corrected and corrected in data.columns:
            resolved.append(corrected)
        else:
            if use_dust_corrected:
                print(
                    f"Warning: requested dust-corrected column '{corrected}' not found in data; using '{col}' instead."
                )
            resolved.append(col)
    return resolved


def ensure_dust_corrected(
    data, mag_cols, ra_col, dec_col, dust_correction_kwargs=None, verbose=False
):
    """Compute missing '<col>_corrected' columns in place via dust_correction.correct_dust_map."""
    missing = [col for col in mag_cols if f"{col}_corrected" not in data.columns]
    if not missing:
        return

    if verbose:
        print(f"Applying dust correction for: {missing}")
    dust_correction.correct_dust_map(
        data,
        mag_cols=list(mag_cols),
        pos_columns=[ra_col, dec_col],
        **dict(dust_correction_kwargs or {}),
    )


# ===================================================
# Match-filter catalog
# ===================================================


def default_error_cuts(match_filter: Mapping[str, Any] | None = None) -> dict[str, tuple]:
    """Default cuts of a match filter: errors <= 0.2 on each of its bands.

    Parameters
    ----------
    match_filter : mapping, optional
        See ``cmd_match_filter``. Default: ``cmd_match_filter.DEFAULT_MATCH_FILTER``.

    Returns
    -------
    cuts : dict
        ``{"<namespace>_<band>_err": DEFAULT_ERR_CUT}`` for the reference band
        and each colour band (format of ``catalogs.catalog_cuts.get_cut_mask``).
    """
    match_filter = DEFAULT_MATCH_FILTER if match_filter is None else match_filter
    bands = [match_filter["reference_band"], *match_filter["color_bands"]]
    return {f"{namespace}_{band}_err": DEFAULT_ERR_CUT for namespace, band in bands}


def check_columns(catalog: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a KeyError naming every column missing from the catalog.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars.
    columns : iterable of str
        Columns needed.

    Raises
    ------
    KeyError
        Some columns are missing; the message lists them all.
    """
    missing = [column for column in dict.fromkeys(columns) if column not in catalog.columns]
    if missing:
        raise KeyError(f"columns missing from the catalog: {missing}")


def prepare_match_filter_catalog(
    catalog: pd.DataFrame,
    match_filter: Mapping[str, Any] | None = None,
    *,
    cuts: Mapping[str, Any] | None = None,
    stream_frame: coord.BaseCoordinateFrame | None = None,
    window: tuple[float, float, float, float] | None = None,
    excluded_pixels: NDArray[np.bool_] | None = None,
    excluded_pixels_nest: bool | None = None,
    ra_col: str = "ra",
    dec_col: str = "dec",
    as_mask: bool = False,
) -> pd.DataFrame | NDArray[np.bool_]:
    """Stars ready for a match filter: columns checked, cuts applied, phi1/phi2.

    Done once for a scan, then reused for every distance modulus and
    isochrone. Keeps only the columns needed: the magnitudes of the match
    filter, the coordinates when a stream frame is given, and the cut
    columns.

    The catalog may be huge: it is neither modified nor copied. The rows to
    keep are found first, as a boolean mask (cuts, excluded pixels, then
    window), and only
    their needed columns are copied, once; with ``as_mask``, nothing is
    copied: the mask is returned (e.g. for a search over the whole catalog,
    ``analysis.generic_stream_search``). With a window, the stars far
    from it are dropped before the transformation to phi1/phi2, the slow
    part: only those in the HEALPix pixels (nside
    ``WINDOW_PRESELECTION_NSIDE``) of the window, or next to them, are
    transformed. The stars in ``excluded_pixels`` (e.g. on the LMC, a dwarf
    galaxy, a globular cluster: ``analysis.build_masks``) are dropped
    before the window, too. The cuts, the exclusion, the pre-selection and
    the transformation go by chunks of
    ``CHUNK_SIZE`` stars: their temporary arrays do not grow with the
    catalog.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars, e.g. ``catalogs.euclid_catalog.get_euclid_stars()``.
    match_filter : mapping, optional
        See ``cmd_match_filter``. Default: ``cmd_match_filter.DEFAULT_MATCH_FILTER``.
    cuts : mapping, optional
        Format of ``catalogs.catalog_cuts.get_cut_mask``. They update the
        default error cuts ``default_error_cuts(match_filter)``, which are
        always applied: a cut on an error column replaces its default one,
        ``(None, None)`` removes it.
    stream_frame : astropy.coordinates.BaseCoordinateFrame, optional
        Adds the ``phi1`` and ``phi2`` columns (deg) in this frame
        (``stream_frames.radec_to_phi12``).
    window : tuple of float, optional
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg, inclusive): keeps
        only the stars inside. Needs ``stream_frame``.
    excluded_pixels : numpy.ndarray of bool, optional
        HEALPix mask, of any nside: the stars in its True pixels are
        dropped (those without finite coordinates are not). Needs
        ``excluded_pixels_nest``.
    excluded_pixels_nest : bool, optional
        HEALPix ordering of ``excluded_pixels``.
    ra_col, dec_col : str, optional
        Coordinate columns (deg), read when ``stream_frame`` or
        ``excluded_pixels`` is given.
    as_mask : bool, optional
        Return the mask of the rows kept instead of the stars.

    Returns
    -------
    stars : pandas.DataFrame
        The stars passing the cuts (and inside the window), with the index of
        ``catalog``.
    keep : numpy.ndarray of bool
        With ``as_mask``: True for the rows of ``catalog`` kept (1 byte per
        star).

    Raises
    ------
    KeyError
        Columns missing from the catalog (all of them listed).
    ValueError
        ``window`` without ``stream_frame``; ``excluded_pixels`` without
        ``excluded_pixels_nest``.
    """
    if window is not None and stream_frame is None:
        raise ValueError("a window needs a stream_frame")
    if excluded_pixels is not None and excluded_pixels_nest is None:
        raise ValueError("excluded_pixels needs excluded_pixels_nest")
    match_filter = DEFAULT_MATCH_FILTER if match_filter is None else match_filter
    cuts = {**default_error_cuts(match_filter), **(cuts or {})}  # given cuts replace the defaults

    columns = match_filter_columns(match_filter)
    if stream_frame is not None:
        columns += [ra_col, dec_col]
    columns = list(dict.fromkeys([*columns, *cuts]))
    check_columns(catalog, columns if excluded_pixels is None else [*columns, ra_col, dec_col])

    keep = np.zeros(len(catalog), dtype=bool)  # rows kept, 1 byte per star
    for start in range(0, len(catalog), CHUNK_SIZE):  # temporaries of a chunk only
        keep[start:start + CHUNK_SIZE] = get_cut_mask(catalog.iloc[start:start + CHUNK_SIZE], cuts)
    if excluded_pixels is not None:
        _drop_excluded_pixels(catalog, keep, excluded_pixels, excluded_pixels_nest, ra_col, dec_col)
    phi12 = {}
    if stream_frame is not None:
        ra = catalog[ra_col].to_numpy()  # views of numpy-backed columns, not copies
        dec = catalog[dec_col].to_numpy()
        pixel_near = None if window is None else _window_pixels(stream_frame, window)
        phi1_parts, phi2_parts = [], []
        for start in range(0, len(catalog), CHUNK_SIZE):
            chunk = slice(start, start + CHUNK_SIZE)
            keep_chunk = keep[chunk]  # a view: setting it updates keep
            chunk_ra, chunk_dec = ra[chunk][keep_chunk], dec[chunk][keep_chunk]
            if pixel_near is not None:
                near = _in_pixels(chunk_ra, chunk_dec, pixel_near)
                keep_chunk[keep_chunk] = near
                chunk_ra, chunk_dec = chunk_ra[near], chunk_dec[near]
            phi1, phi2 = radec_to_phi12(chunk_ra, chunk_dec, stream_frame)
            if window is not None:
                phi1_min, phi1_max, phi2_min, phi2_max = window
                inside = (phi1 >= phi1_min) & (phi1 <= phi1_max) & (phi2 >= phi2_min) & (phi2 <= phi2_max)
                keep_chunk[keep_chunk] = inside
                phi1, phi2 = phi1[inside], phi2[inside]
            if not as_mask:
                phi1_parts.append(phi1)
                phi2_parts.append(phi2)
        phi12 = {
            "phi1": np.concatenate(phi1_parts) if phi1_parts else np.empty(0),
            "phi2": np.concatenate(phi2_parts) if phi2_parts else np.empty(0),
        }

    if as_mask:
        return keep
    return pd.DataFrame(
        {**{column: catalog[column].to_numpy()[keep] for column in columns}, **phi12},
        # take: indexing a RangeIndex (with a mask or positions) builds and caches all its values
        index=catalog.index.take(np.flatnonzero(keep)),
        copy=False,  # the selected rows are already copies: not copied again
    )


def star_healpix_pixels(
    catalog: pd.DataFrame,
    nside: int,
    *,
    nest: bool,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> NDArray[np.int32]:
    """HEALPix pixel of each star of a catalog, by chunks.

    The catalog is neither modified nor copied: its coordinate columns are
    read by chunks of ``CHUNK_SIZE`` stars (views). 4 bytes per star.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars, with ``ra_col``, ``dec_col`` (deg).
    nside : int
        HEALPix resolution.
    nest : bool
        HEALPix ordering of the pixels.
    ra_col, dec_col : str, optional
        Coordinate columns (deg).

    Returns
    -------
    pixels : numpy.ndarray of int32
        Pixel of each star, in the order of ``catalog``; -1 for a star
        without finite coordinates.

    Raises
    ------
    KeyError
        Coordinate columns missing from the catalog.
    """
    check_columns(catalog, [ra_col, dec_col])
    ra, dec = catalog[ra_col].to_numpy(), catalog[dec_col].to_numpy()  # views
    pixels = np.full(len(catalog), -1, dtype=np.int32)
    for start in range(0, len(catalog), CHUNK_SIZE):
        chunk = slice(start, start + CHUNK_SIZE)
        finite = np.isfinite(ra[chunk]) & np.isfinite(dec[chunk])
        pixels[chunk][finite] = hp.ang2pix(nside, ra[chunk][finite], dec[chunk][finite], lonlat=True, nest=nest)
    return pixels


def _window_pixels(
    stream_frame: coord.BaseCoordinateFrame,
    window: tuple[float, float, float, float],
    nside: int = WINDOW_PRESELECTION_NSIDE,
) -> NDArray[np.bool_]:
    """HEALPix pixels (RING) of a phi1/phi2 window and their neighbours, as a mask.

    The window is sampled every half pixel and its points taken to the sky
    (``stream_frames.phi12_to_radec``): every point of the window is in the
    pixel of one of these points or in a neighbour of it.
    """
    phi1_min, phi1_max, phi2_min, phi2_max = window
    step = np.degrees(hp.nside2resol(nside)) / 2
    phi1_grid = np.linspace(phi1_min, phi1_max, int(np.ceil((phi1_max - phi1_min) / step)) + 1)
    phi2_grid = np.linspace(phi2_min, phi2_max, int(np.ceil((phi2_max - phi2_min) / step)) + 1)
    grid_ra, grid_dec = phi12_to_radec(*np.meshgrid(phi1_grid, phi2_grid), stream_frame)
    pixels = np.unique(hp.ang2pix(nside, grid_ra, grid_dec, lonlat=True))
    neighbours = hp.get_all_neighbours(nside, pixels).ravel()
    pixel_near = np.zeros(hp.nside2npix(nside), dtype=bool)
    pixel_near[pixels] = True
    pixel_near[neighbours[neighbours >= 0]] = True  # -1: no neighbour there
    return pixel_near


def _drop_excluded_pixels(catalog, keep, excluded_pixels, nest, ra_col, dec_col):
    """Set keep to False for the stars in the True pixels of excluded_pixels, by chunks."""
    excluded_pixels = np.asarray(excluded_pixels, dtype=bool)
    if not excluded_pixels.any():
        return
    ra = catalog[ra_col].to_numpy()  # views of numpy-backed columns, not copies
    dec = catalog[dec_col].to_numpy()
    for start in range(0, len(catalog), CHUNK_SIZE):
        chunk = slice(start, start + CHUNK_SIZE)
        keep_chunk = keep[chunk]  # a view: setting it updates keep
        keep_chunk[keep_chunk] = ~_in_pixels(ra[chunk][keep_chunk], dec[chunk][keep_chunk], excluded_pixels, nest=nest)


def _in_pixels(
    ra: NDArray[np.float64], dec: NDArray[np.float64], pixel_mask: NDArray[np.bool_], *, nest: bool = False
) -> NDArray[np.bool_]:
    """Stars in the True pixels of a HEALPix mask (RING by default); NaN coordinates are not."""
    nside = hp.npix2nside(pixel_mask.size)
    finite = np.isfinite(ra) & np.isfinite(dec)
    inside = np.zeros(ra.size, dtype=bool)
    inside[finite] = pixel_mask[hp.ang2pix(nside, ra[finite], dec[finite], lonlat=True, nest=nest)]
    return inside

"""Survey footprint: HEALPix mask, its projection on a phi1/phi2 grid, and areas outside a mask.

As in ``analysis.star_density_maps.star_density_healpix`` and
``analysis.background_polynomial_fit.process_data``, a mask is True outside
the footprint (masked). phi1/phi2 grids are those of
``stream_frames.phi12_bin_centres``.

The footprint map of a survey, when there is one (e.g. the DES Y6 or Euclid
DR1 coverage map), is a HEALPix FITS file ``FOOTPRINT_DIR/<survey_name>.fits``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np

from streameucliddes.stream_frames import phi12_bin_centres, phi12_to_radec

if TYPE_CHECKING:
    import astropy.coordinates as coord
    from numpy.typing import ArrayLike, NDArray

LOGGER = logging.getLogger(__name__)

# Footprint maps of the surveys, next to the catalogs (see
# euclid_catalog.DATA_DIR): the data/ folder of the repository.
FOOTPRINT_DIR = Path(__file__).resolve().parents[3] / "data" / "footprints"
# Stars per chunk of the star counts: bounds their temporary arrays, whatever the catalog size
CHUNK_SIZE = 500_000


def survey_footprint_healpix_mask(
    ra: ArrayLike,
    dec: ArrayLike,
    nside: int,
    *,
    nest: bool,
    min_count: int = 1,
    survey_name: str | None = None,
) -> NDArray[np.bool_]:
    """HEALPix mask of the survey footprint.

    Read from the footprint map of ``survey_name`` when there is one, brought to
    ``nside``. Otherwise computed from the star counts: a pixel is in the
    footprint when it holds at least ``min_count`` stars. The stars are
    counted by chunks of ``CHUNK_SIZE``: the temporary arrays do not grow
    with the catalog, and ``ra``, ``dec`` (e.g. ``catalog["ra"].to_numpy()``,
    views) are not copied.

    A footprint map is a HEALPix FITS file ``FOOTPRINT_DIR/<survey_name>.fits``, of
    any nside and ordering. A pixel is in the footprint when its value is
    positive (UNSEEN and NaN are outside), so both a 0/1 map and a coverage
    fraction work. At a coarser nside, a pixel is in the footprint when at
    least half of its sub-pixels are. At a finer nside, the sub-pixels take
    the value of their pixel.

    Parameters
    ----------
    ra, dec : array_like
        Stars of the survey catalog (deg). Stars with a non-finite position
        are left out. Not used when the footprint map is read.
    nside : int
        HEALPix resolution of the mask.
    nest : bool
        HEALPix ordering of the mask. No default: always explicit.
    min_count : int, optional
        Minimum number of stars of a pixel in the footprint, when the mask is
        computed from the stars.
    survey_name : str, optional
        Name of the footprint map, without ``.fits`` (e.g. ``"des_y6"``).
        None, or no such file: the mask is computed from the stars.

    Returns
    -------
    mask : numpy.ndarray of bool, shape (12 * nside**2,)
        True outside the footprint.
    """
    if survey_name is not None:
        path = FOOTPRINT_DIR / f"{survey_name}.fits"
        if path.exists():
            return _footprint_mask_from_map(path, nside, nest=nest)
        LOGGER.info("No footprint map %s: footprint from the star counts.", path)

    ra = np.asarray(ra, dtype=float)  # no copy of float arrays
    dec = np.asarray(dec, dtype=float)
    counts = np.zeros(hp.nside2npix(nside), dtype=np.int64)
    for start in range(0, ra.size, CHUNK_SIZE):
        chunk_ra, chunk_dec = ra[start:start + CHUNK_SIZE], dec[start:start + CHUNK_SIZE]
        valid = np.isfinite(chunk_ra) & np.isfinite(chunk_dec)
        pixels = hp.ang2pix(nside, chunk_ra[valid], chunk_dec[valid], lonlat=True, nest=nest)
        counts += np.bincount(pixels, minlength=counts.size)
    return counts < min_count


def convert_footprint_healpix_mask(
    footprint_mask: ArrayLike,
    nside: int,
    *,
    nest: bool,
    nest_out: bool,
) -> NDArray[np.bool_]:
    """A HEALPix footprint mask at another nside and ordering.

    At a coarser nside, a pixel is outside the footprint when any of its
    sub-pixels is: the pixels left inside are fully covered (no partial
    pixels at the edges, e.g. for star counts per pixel). At a finer nside,
    the sub-pixels take the value of their pixel. (A footprint map read by
    ``survey_footprint_healpix_mask`` keeps instead the pixels at least half
    covered.)

    Parameters
    ----------
    footprint_mask : array_like of bool
        HEALPix mask, True outside the footprint.
    nside : int
        HEALPix resolution of the output.
    nest : bool
        HEALPix ordering of ``footprint_mask``.
    nest_out : bool
        HEALPix ordering of the output.

    Returns
    -------
    mask : numpy.ndarray of bool, shape (12 * nside**2,)
        True outside the footprint.
    """
    footprint_mask = np.asarray(footprint_mask, dtype=bool)
    order_in, order_out = ("NESTED" if nest else "RING"), ("NESTED" if nest_out else "RING")
    if hp.npix2nside(footprint_mask.size) == nside:
        if nest == nest_out:
            return footprint_mask.copy()
        return hp.reorder(footprint_mask, inp=order_in, out=order_out).astype(bool)
    outside_fraction = hp.ud_grade(footprint_mask.astype(float), nside, order_in=order_in, order_out=order_out)
    return outside_fraction > 0


def _footprint_mask_from_map(path: Path, nside: int, *, nest: bool) -> NDArray[np.bool_]:
    """Mask (True outside) of the footprint map in ``path``, brought to ``nside``."""
    values = hp.read_map(path, nest=nest)  # converted to the requested ordering
    inside = (values > 0).astype(float)  # UNSEEN (negative) and NaN: outside
    map_nside = hp.npix2nside(inside.size)
    if map_nside != nside:
        # Coarser: fraction of the sub-pixels inside. Finer: value of the pixel.
        order = "NESTED" if nest else "RING"
        inside = hp.ud_grade(inside, nside, order_in=order, order_out=order)
    LOGGER.info("Footprint read from %s (nside %d, used at nside %d).", path, map_nside, nside)
    return inside < 0.5


def survey_footprint_phi12_mask(
    footprint_mask: NDArray[np.bool_],
    stream_frame: coord.BaseCoordinateFrame,
    window: tuple[float, float, float, float],
    bin_size_deg: float,
    *,
    nest: bool,
) -> NDArray[np.bool_]:
    """Survey footprint on a phi1/phi2 grid.

    A bin is outside the footprint when its centre falls in a masked HEALPix
    pixel. Used to mask the known-stream maps, and to correct their edges:
    the smoothed counts are divided by the smoothed footprint (see
    ``analysis.star_density_maps.star_density_phi12``).

    Parameters
    ----------
    footprint_mask : numpy.ndarray of bool
        HEALPix mask from ``survey_footprint_healpix_mask``, True outside.
        nside is taken from its length.
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Stream frame, e.g. the ``stream_frame`` of a galstreams track (a gala
        ``GreatCircleICRSFrame``).
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` of the grid (deg).
    bin_size_deg : float
        Bin size of the grid (deg), the same in phi1 and phi2 (see
        ``stream_frames.phi12_bin_centres``).
    nest : bool
        HEALPix ordering of ``footprint_mask``.

    Returns
    -------
    mask : numpy.ndarray of bool, shape (n_phi1, n_phi2)
        True outside the footprint. Row = phi1 bin, as the maps of
        ``analysis.star_density_maps.star_density_phi12``.

    Notes
    -----
    Adapted from celestial_currents
    ``visualization.py::transform_healpix_to_custom_frame``: the bin centres
    are taken to ICRS (``stream_frames.phi12_to_radec``), then to HEALPix
    pixels.
    """
    phi1_centres, phi2_centres = phi12_bin_centres(window, bin_size_deg)
    phi1, phi2 = np.meshgrid(phi1_centres, phi2_centres, indexing="ij")
    return _mask_at_phi12(footprint_mask, phi1, phi2, stream_frame, nest=nest)


def _mask_at_phi12(
    healpix_mask: ArrayLike, phi1: ArrayLike, phi2: ArrayLike, stream_frame: coord.BaseCoordinateFrame, *, nest: bool
) -> NDArray[np.bool_]:
    """Value of a HEALPix mask at phi1/phi2 points (nside from its length)."""
    healpix_mask = np.asarray(healpix_mask, dtype=bool)
    ra, dec = phi12_to_radec(phi1, phi2, stream_frame)
    return healpix_mask[hp.ang2pix(hp.npix2nside(healpix_mask.size), ra, dec, lonlat=True, nest=nest)]


def unmasked_area_deg2(
    stream_frame: coord.BaseCoordinateFrame,
    phi1_edges: ArrayLike,
    phi2_edges: ArrayLike,
    excluded_healpix_mask: NDArray[np.bool_] | None,
    *,
    step_deg: float = 0.01,
) -> NDArray[np.float64]:
    """Area of each phi1/phi2 cell outside a HEALPix mask.

    Each cell is cut into a fine grid of about ``step_deg``; each piece
    weighs its exact solid angle, ``dphi1 (sin phi2_top - sin phi2_bottom)``,
    and is kept when the HEALPix pixel of its centre is not masked: the same
    look-up as ``survey_footprint_phi12_mask``, at the pieces instead of the
    bin centres. The area is exact without a mask; with a mask, its error
    comes from the pieces across a pixel edge (``step_deg`` well below the
    pixel size). A window is one cell: ``phi1_edges = (phi1_min, phi1_max)``,
    ``phi2_edges = (phi2_min, phi2_max)``; a cell of zero width has zero area.

    Parameters
    ----------
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Frame of the cells.
    phi1_edges, phi2_edges : array_like
        Edges of the cells along each axis (deg), increasing.
    excluded_healpix_mask : numpy.ndarray of bool or None
        True where excluded (outside the footprint, on a masked object),
        RING, e.g. from ``analysis.build_masks.excluded_healpix_mask``; nside
        from its length. None: nothing excluded.
    step_deg : float, optional
        Step of the fine grid (deg).

    Returns
    -------
    area_deg2 : numpy.ndarray
        Shape (len(phi1_edges) - 1, len(phi2_edges) - 1) (deg²).
    """
    phi1, cell_1, width_1 = _fine_pieces(phi1_edges, step_deg)
    phi2, cell_2, width_2 = _fine_pieces(phi2_edges, step_deg)
    half = np.radians(width_2) / 2
    strip = np.sin(np.radians(phi2) + half) - np.sin(np.radians(phi2) - half)  # solid angle per radian of phi1
    weight = np.outer(np.radians(width_1), strip) * np.degrees(1.0) ** 2  # deg²
    if excluded_healpix_mask is not None:
        grid_1, grid_2 = np.meshgrid(phi1, phi2, indexing="ij")
        weight = np.where(_mask_at_phi12(excluded_healpix_mask, grid_1, grid_2, stream_frame, nest=False), 0.0, weight)
    n_1, n_2 = np.size(phi1_edges) - 1, np.size(phi2_edges) - 1
    cells = (cell_1[:, None] * n_2 + cell_2[None, :]).ravel()
    return np.bincount(cells, weights=weight.ravel(), minlength=n_1 * n_2).reshape(n_1, n_2)


def _fine_pieces(edges: ArrayLike, step_deg: float) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Centres, cell indices and widths of the pieces of about ``step_deg`` cutting each cell between edges."""
    edges = np.asarray(edges, dtype=float)
    widths = np.diff(edges)
    if np.any(widths < 0):
        raise ValueError("Edges must be increasing.")
    n_pieces = np.maximum(np.ceil(widths / step_deg).astype(int), 1)
    cell = np.repeat(np.arange(widths.size), n_pieces)
    width = np.repeat(widths / n_pieces, n_pieces)
    rank = np.arange(cell.size) - np.repeat(np.cumsum(n_pieces) - n_pieces, n_pieces)  # rank of the piece in its cell
    return edges[cell] + (rank + 0.5) * width, cell, width

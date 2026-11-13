"""HEALPix masks of the objects that overlap the stream searches.

One mask per type of object: the LMC, the SMC, the dwarf galaxies, the
globular clusters (LVDB) and the bright galaxies (Siena Galaxy Atlas);
positions, sizes and shapes from ``catalogs.overlapped_objects``. Each object
is masked over its reference ellipse (``radius_arcmin``, ``ellipticity``,
``position_angle``: the half-light ellipse of the LVDB, the 26 mag/arcsec^2
isophote of the atlas) scaled by ``radius_scale``: semi-major axis
``radius_scale * radius_arcmin``, same ellipticity and position angle; a
circle when the ellipticity is unknown. The bright galaxies are masked as
circles. As the footprint masks (``catalogs.survey_footprint``), a mask is
True where masked.

Both searches mask the five types by default (``mask_lmc``, ``mask_smc``,
``mask_dwarfs``, ``mask_gc``, ``mask_galaxies`` of ``analysis.generic_stream_search.prepare_generic_search`` and
``analysis.stream_frame_search.prepare_search``): the stars on them are
dropped (``catalog_preparation.prepare_match_filter_catalog`` with
``excluded_pixels``), and their pixels or bins added to the footprint mask,
so that no star is counted there and the background is fitted on the rest.
"""

from __future__ import annotations

import logging
import types
from collections.abc import Mapping
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
from astropy.coordinates import angular_separation, position_angle

from streameucliddes.catalogs import overlapped_objects

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

# Mask semi-major axis, in half-light radii (as celestial_currents investigate/postprocess.py::build_lvdb_mask)
DEFAULT_RADIUS_SCALE = 4.0
# The Magellanic Clouds extend far beyond a few half-light radii. Measured in Euclid DR1 south (match
# filter at their distance moduli, 18.5 and 19.0), the star counts reach the background:
# - LMC: about 21 deg along its LVDB ellipse, in every direction: 6.5 half-light radii (3.2 deg);
# - SMC: about 11 deg from its centre: 11 half-light radii (1.0 deg; a circle, no LVDB ellipticity).
LMC_RADIUS_SCALE = 6.5
SMC_RADIUS_SCALE = 11.0
# Bright galaxies: circles of 5 times D26/2. In Euclid DR1 northeast (match filter at distance moduli 15
# to 17), the point-like sources of M81 and M82 are at the background beyond 4 to 5 times D26/2, along
# their major axis; along their minor axis about half as far, rounder than the isophote (M82, b/a 0.25)
GALAXY_RADIUS_SCALE = 5.0
# Types of objects of overlapped_objects_healpix_mask, as the keys of a radius_scale mapping
OBJECT_TYPES = ("lmc", "smc", "dwarfs", "gc", "galaxies")
DEFAULT_RADIUS_SCALES = types.MappingProxyType({
    "lmc": LMC_RADIUS_SCALE, "smc": SMC_RADIUS_SCALE, "dwarfs": DEFAULT_RADIUS_SCALE, "gc": DEFAULT_RADIUS_SCALE,
    "galaxies": GALAXY_RADIUS_SCALE,
})


def objects_healpix_mask(
    objects: pd.DataFrame,
    nside: int,
    *,
    nest: bool,
    radius_scale: float = DEFAULT_RADIUS_SCALE,
    mask_centre_pixel: bool = True,
) -> NDArray[np.bool_]:
    """HEALPix mask of ellipses around objects.

    Each object is masked over an ellipse of semi-major axis
    ``radius_scale * radius_arcmin``, semi-minor axis ``(1 - ellipticity)``
    times it, major axis at ``position_angle`` (north to east); a circle
    when the ellipticity is unknown. A pixel is masked when its centre is in
    an ellipse; with ``mask_centre_pixel``, the pixel of the centre of an
    object is masked too (an object smaller than a pixel masks its pixel).
    Objects without a radius are left out (logged).

    The ellipse is drawn in the coordinates (separation, position angle)
    around the centre of the object (``astropy.coordinates.angular_separation``
    and ``position_angle``): exact on the sky whatever its size.

    Parameters
    ----------
    objects : pandas.DataFrame
        Columns ``ra``, ``dec`` (deg), ``radius_arcmin``, ``ellipticity`` and
        ``position_angle`` (deg), e.g.
        ``catalogs.overlapped_objects.get_globular_clusters()``.
    nside : int
        HEALPix resolution of the mask.
    nest : bool
        HEALPix ordering of the mask. No default: always explicit.
    radius_scale : float, optional
        Semi-major axis of the ellipses, in units of ``radius_arcmin``.
    mask_centre_pixel : bool, optional
        Mask the pixel of the centre of each object, whatever its size: for
        compact objects (a cluster smaller than a pixel still has its stars
        in it).

    Returns
    -------
    mask : numpy.ndarray of bool, shape (12 * nside**2,)
        True on the ellipses.

    Notes
    -----
    Adapted from celestial_currents ``investigate/postprocess.py::build_lvdb_mask``
    (circles there).
    """
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    semi_major_deg = radius_scale * objects["radius_arcmin"].to_numpy(dtype=float) / 60.0
    known = np.isfinite(semi_major_deg)
    if (~known).any():
        LOGGER.info("%d objects without a radius: not masked.", int((~known).sum()))
    ellipticity = np.nan_to_num(objects["ellipticity"].to_numpy(dtype=float), nan=0.0)  # unknown: a circle
    major_angle = np.nan_to_num(objects["position_angle"].to_numpy(dtype=float), nan=0.0)
    ra, dec = objects["ra"].to_numpy(dtype=float), objects["dec"].to_numpy(dtype=float)
    for i in np.flatnonzero(known):
        if mask_centre_pixel:
            mask[hp.ang2pix(nside, ra[i], dec[i], lonlat=True, nest=nest)] = True
        vector = hp.ang2vec(ra[i], dec[i], lonlat=True)
        candidates = hp.query_disc(nside, vector, np.radians(semi_major_deg[i]), nest=nest)  # centres in the circle
        lon, lat = np.radians(hp.pix2ang(nside, candidates, nest=nest, lonlat=True))
        centre_lon, centre_lat = np.radians(ra[i]), np.radians(dec[i])
        separation = np.degrees(angular_separation(centre_lon, centre_lat, lon, lat))
        angle = np.asarray(position_angle(centre_lon, centre_lat, lon, lat)) - np.radians(major_angle[i])
        along_major, along_minor = separation * np.cos(angle), separation * np.sin(angle)
        semi_minor_deg = semi_major_deg[i] * (1.0 - ellipticity[i])
        inside = (along_major / semi_major_deg[i]) ** 2 + (along_minor / semi_minor_deg) ** 2 <= 1.0
        mask[candidates[inside]] = True
    return mask


def object_mask_nside(bin_size_deg: float) -> int:
    """HEALPix resolution of the object masks of a phi1/phi2 grid: pixels at most half a bin.

    Used by ``analysis.stream_frame_search.prepare_search``: the stars are
    dropped by pixel, the bins masked by their centre; at the edge of a
    mask, a bin not masked loses at most the stars of the pixels it
    overlaps, less than half a bin wide.

    Parameters
    ----------
    bin_size_deg : float
        Bin size of the grid (deg).

    Returns
    -------
    nside : int
        The smallest power of 2 whose pixels (``healpy.nside2resol``) are at
        most ``bin_size_deg / 2``; e.g. 2048 (0.03 deg) for 0.1 deg bins.
    """
    nside = 1
    while np.degrees(hp.nside2resol(nside)) > bin_size_deg / 2:
        nside *= 2
    return nside


def lmc_healpix_mask(nside: int, *, nest: bool, radius_scale: float = LMC_RADIUS_SCALE) -> NDArray[np.bool_]:
    """HEALPix mask of the LMC (``catalogs.overlapped_objects.get_lmc``).

    By default, 21 deg along the major axis (LVDB v1.1.2: half-light radius
    3.2 deg, ellipticity 0.16, major axis at 227 deg). Parameters and returns:
    see ``objects_healpix_mask``.
    """
    return objects_healpix_mask(overlapped_objects.get_lmc(), nside, nest=nest, radius_scale=radius_scale)


def smc_healpix_mask(nside: int, *, nest: bool, radius_scale: float = SMC_RADIUS_SCALE) -> NDArray[np.bool_]:
    """HEALPix mask of the SMC (``catalogs.overlapped_objects.get_smc``).

    By default, a circle of 11 deg (LVDB v1.1.2: half-light radius 1.0 deg, no
    ellipticity). Parameters and returns: see ``objects_healpix_mask``.
    """
    return objects_healpix_mask(overlapped_objects.get_smc(), nside, nest=nest, radius_scale=radius_scale)


def dwarf_galaxies_healpix_mask(
    nside: int, *, nest: bool, radius_scale: float = DEFAULT_RADIUS_SCALE
) -> NDArray[np.bool_]:
    """HEALPix mask of the dwarf galaxies but the LMC and SMC (``catalogs.overlapped_objects.get_dwarf_galaxies``).

    Parameters and returns: see ``objects_healpix_mask``.
    """
    return objects_healpix_mask(overlapped_objects.get_dwarf_galaxies(), nside, nest=nest, radius_scale=radius_scale)


def globular_clusters_healpix_mask(
    nside: int, *, nest: bool, radius_scale: float = DEFAULT_RADIUS_SCALE
) -> NDArray[np.bool_]:
    """HEALPix mask of the globular clusters (``catalogs.overlapped_objects.get_globular_clusters``).

    Parameters and returns: see ``objects_healpix_mask``.
    """
    return objects_healpix_mask(
        overlapped_objects.get_globular_clusters(), nside, nest=nest, radius_scale=radius_scale
    )


def galaxies_healpix_mask(nside: int, *, nest: bool, radius_scale: float = GALAXY_RADIUS_SCALE) -> NDArray[np.bool_]:
    """HEALPix mask of the bright galaxies (``catalogs.overlapped_objects.get_galaxies``).

    Circles of ``radius_scale`` times ``D26 / 2`` (their point-like sources
    spread rounder than their isophote), without the pixel of their centre
    forced: a galaxy smaller than a pixel has few point-like sources, and
    forcing the pixels of the 82 000 galaxies would mask about 1000 deg2 of
    the sky at nside 512. E.g. M81: 1.1 deg, M82: 0.6 deg. Parameters and
    returns: see ``objects_healpix_mask``.
    """
    galaxies = overlapped_objects.get_galaxies().assign(ellipticity=np.nan)  # circles
    return objects_healpix_mask(galaxies, nside, nest=nest, radius_scale=radius_scale, mask_centre_pixel=False)


def overlapped_objects_healpix_mask(
    nside: int,
    *,
    nest: bool,
    mask_lmc: bool = False,
    mask_smc: bool = False,
    mask_dwarfs: bool = False,
    mask_gc: bool = False,
    mask_galaxies: bool = False,
    radius_scale: float | Mapping[str, float] = DEFAULT_RADIUS_SCALES,
) -> NDArray[np.bool_]:
    """HEALPix mask of the objects of the chosen types: the union of their masks.

    Parameters
    ----------
    nside : int
        HEALPix resolution of the mask.
    nest : bool
        HEALPix ordering of the mask.
    mask_lmc : bool, optional
        Mask the LMC (``lmc_healpix_mask``).
    mask_smc : bool, optional
        Mask the SMC (``smc_healpix_mask``).
    mask_dwarfs : bool, optional
        Mask the dwarf galaxies, but the LMC and SMC (``dwarf_galaxies_healpix_mask``).
    mask_gc : bool, optional
        Mask the globular clusters (``globular_clusters_healpix_mask``).
    mask_galaxies : bool, optional
        Mask the bright galaxies (``galaxies_healpix_mask``).
    radius_scale : float or mapping, optional
        Semi-major axis of the ellipses, in units of the reference radius of
        the objects (``radius_arcmin``): the same for every type, or per
        type, ``{"lmc": ..., "smc": ..., "dwarfs": ..., "gc": ...,
        "galaxies": ...}`` (``OBJECT_TYPES``; a type left out has its
        default, ``DEFAULT_RADIUS_SCALES``: 6.5 for the LMC, 11 for the SMC,
        4 for the dwarfs and clusters, 5 for the galaxies).

    Returns
    -------
    mask : numpy.ndarray of bool, shape (12 * nside**2,)
        True on the masked objects; all False when no type is chosen.

    Raises
    ------
    ValueError
        A key of ``radius_scale`` that is not in ``OBJECT_TYPES``.
    """
    if isinstance(radius_scale, Mapping):
        unknown = set(radius_scale) - set(OBJECT_TYPES)
        if unknown:
            raise ValueError(f"radius_scale keys {sorted(unknown)} not in {OBJECT_TYPES}.")
        scales = {**DEFAULT_RADIUS_SCALES, **radius_scale}
    else:
        scales = dict.fromkeys(OBJECT_TYPES, radius_scale)

    masks = {"lmc": lmc_healpix_mask, "smc": smc_healpix_mask, "dwarfs": dwarf_galaxies_healpix_mask,
             "gc": globular_clusters_healpix_mask, "galaxies": galaxies_healpix_mask}
    chosen = {"lmc": mask_lmc, "smc": mask_smc, "dwarfs": mask_dwarfs, "gc": mask_gc, "galaxies": mask_galaxies}
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    for object_type in OBJECT_TYPES:
        if chosen[object_type]:
            mask |= masks[object_type](nside, nest=nest, radius_scale=scales[object_type])
    return mask

"""HEALPix masks of the objects that overlap the stream searches.

One mask per type of object: the LMC, the SMC, the dwarf galaxies, the
globular clusters (LVDB) and the bright galaxies (Siena Galaxy Atlas);
positions, sizes and shapes from ``catalogs.overlapped_objects``. Each object
is masked over its reference ellipse (``radius_arcmin``, ``ellipticity``,
``position_angle``: the half-light ellipse of the LVDB, the 26 mag/arcsec^2
isophote of the atlas) scaled by ``radius_scale``: semi-major axis
``radius_scale * radius_arcmin``, same ellipticity and position angle; a
circle when the ellipticity is unknown. The bright galaxies are masked as
circles. The scale is one per type, but for the globular clusters: by
default, each has its own, larger for a brighter surface brightness
(``globular_cluster_radius_scales``). As the footprint masks (``catalogs.survey_footprint``), a mask is
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

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord, angular_separation, position_angle

from streameucliddes.catalogs import overlapped_objects
from streameucliddes.catalogs.survey_footprint import convert_footprint_healpix_mask

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike, NDArray

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
# Globular clusters: the brighter their surface brightness, the farther their stars stand out of the field. The
# masks are built from the LVDB alone (V, half-light radius, ellipticity); Euclid DR1 only set the two constants
# below, once. Measured there (VIS - H match filter at their distance modulus, background 40 to 60 arcmin away),
# the excess of the cluster falls to 3 times the background (chosen with the user, 2026-11-16: masks no larger
# than needed) at 27.4 half-light radii for NGC 1851 (V = 7.0, half-light radius 0.50 arcmin: 13.7 arcmin) and
# 18.0 for NGC 1261 (V = 8.3, 0.68 arcmin: 12.3 arcmin); the few stars of the faint Eridanus III (V = 17.7) are
# within 4. (At 1 times the background: 37 and 23.) The scale of a cluster of mean surface brightness mu within
# its half-light ellipse, mu = V + 2.5 log10(2 pi rhalf**2 (1 - ellipticity)) (mag/arcmin**2; rhalf its
# semi-major axis), is where a profile falling as radius**-GC_PROFILE_SLOPE reaches that level, from that of
# NGC 1851 (the slope fitted on the two):
# GC_REFERENCE_RADIUS_SCALE * 10**(-0.4 * (mu - GC_REFERENCE_SURFACE_BRIGHTNESS) / GC_PROFILE_SLOPE), and at
# least DEFAULT_RADIUS_SCALE (faint clusters, or V unknown). No cap: the brightest clusters (omega Cen, 47 Tuc,
# NGC 6121; 1 to 2 deg by this rule) are not in DR1
GC_REFERENCE_SURFACE_BRIGHTNESS = 7.50  # NGC 1851, mag/arcmin**2
GC_REFERENCE_RADIUS_SCALE = 27.4
GC_PROFILE_SLOPE = 4.26
# Types of objects of overlapped_objects_healpix_mask, as the keys of a radius_scale mapping. None: the scale of
# each cluster from its surface brightness (globular_cluster_radius_scales)
OBJECT_TYPES = overlapped_objects.OBJECT_TYPES
DEFAULT_RADIUS_SCALES = types.MappingProxyType({
    "lmc": LMC_RADIUS_SCALE, "smc": SMC_RADIUS_SCALE, "dwarfs": DEFAULT_RADIUS_SCALE, "gc": None,
    "galaxies": GALAXY_RADIUS_SCALE,
})


def objects_healpix_mask(
    objects: pd.DataFrame,
    nside: int,
    *,
    nest: bool,
    radius_scale: float | ArrayLike = DEFAULT_RADIUS_SCALE,
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
    radius_scale : float or array_like, optional
        Semi-major axis of the ellipses, in units of ``radius_arcmin``: the
        same for every object, or one per object (e.g. ``mask_radius_scales``).
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
    semi_major_deg = np.asarray(radius_scale, dtype=float) * objects["radius_arcmin"].to_numpy(dtype=float) / 60.0
    semi_major_deg = np.broadcast_to(semi_major_deg, len(objects))
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
        lon, lat = hp.pix2ang(nside, candidates, nest=nest, lonlat=True)
        radius = elliptical_radius_deg(lon, lat, ra[i], dec[i], ellipticity[i], major_angle[i])
        mask[candidates[radius <= semi_major_deg[i]]] = True
    return mask


def elliptical_radius_deg(
    ra: ArrayLike,
    dec: ArrayLike,
    centre_ra: float,
    centre_dec: float,
    ellipticity: float,
    major_axis_angle: float,
) -> NDArray[np.float64]:
    """Elliptical radius of points around a centre: the semi-major axis of the ellipse through each point.

    A point is inside the ellipse of semi-major axis ``a`` (and semi-minor
    axis ``(1 - ellipticity) * a``, major axis at ``major_axis_angle``) when
    its elliptical radius is at most ``a``. Computed in the coordinates
    (separation, position angle) around the centre
    (``astropy.coordinates.angular_separation`` and ``position_angle``):
    exact on the sky whatever the size.

    Parameters
    ----------
    ra, dec : array_like
        Points (deg).
    centre_ra, centre_dec : float
        Centre (deg).
    ellipticity : float
        1 - b/a; 0 for a circle (the elliptical radius is then the separation).
    major_axis_angle : float
        Position angle of the major axis (deg, north to east).

    Returns
    -------
    radius : numpy.ndarray
        Elliptical radius (deg), of the shape of ``ra``.
    """
    lon, lat = np.radians(np.asarray(ra, dtype=float)), np.radians(np.asarray(dec, dtype=float))
    centre_lon, centre_lat = np.radians(centre_ra), np.radians(centre_dec)
    separation = np.degrees(angular_separation(centre_lon, centre_lat, lon, lat))
    angle = np.asarray(position_angle(centre_lon, centre_lat, lon, lat)) - np.radians(major_axis_angle)
    along_major, along_minor = separation * np.cos(angle), separation * np.sin(angle)
    return np.hypot(along_major, along_minor / (1.0 - ellipticity))


def ellipse_outline_radec(
    centre_ra: float,
    centre_dec: float,
    semi_major_deg: float,
    ellipticity: float,
    major_axis_angle: float,
    *,
    n_points: int = 180,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Outline of an ellipse on the sky: the points of elliptical radius ``semi_major_deg``.

    The ellipse of ``elliptical_radius_deg``: at each position angle around
    the centre, the point at the separation where the elliptical radius is
    ``semi_major_deg`` (``astropy.coordinates.SkyCoord.directional_offset_by``).
    Exact on the sky; take it to a stream frame with
    ``stream_frames.radec_to_phi12`` to draw it on a map.

    Parameters
    ----------
    centre_ra, centre_dec : float
        Centre (deg).
    semi_major_deg : float
        Semi-major axis (deg).
    ellipticity : float
        1 - b/a; NaN or 0 for a circle.
    major_axis_angle : float
        Position angle of the major axis (deg, north to east); NaN: 0.
    n_points : int, optional
        Points along the outline.

    Returns
    -------
    ra, dec : numpy.ndarray
        (deg), a closed line of ``n_points + 1`` points.
    """
    ellipticity = 0.0 if not np.isfinite(ellipticity) else float(ellipticity)
    major_axis_angle = 0.0 if not np.isfinite(major_axis_angle) else float(major_axis_angle)
    angle = np.linspace(0.0, 360.0, n_points + 1)
    relative = np.radians(angle - major_axis_angle)
    separation = semi_major_deg / np.hypot(np.cos(relative), np.sin(relative) / (1.0 - ellipticity))
    centre = SkyCoord(centre_ra * u.deg, centre_dec * u.deg)
    outline = centre.directional_offset_by(angle * u.deg, separation * u.deg)
    return outline.ra.deg, outline.dec.deg


def globular_cluster_radius_scales(clusters: pd.DataFrame) -> NDArray[np.float64]:
    """Mask scale of each globular cluster, in half-light radii, from its surface brightness.

    ``GC_REFERENCE_RADIUS_SCALE * 10**(-0.4 * (mu - GC_REFERENCE_SURFACE_BRIGHTNESS) / GC_PROFILE_SLOPE)``,
    at least ``DEFAULT_RADIUS_SCALE``, with ``mu = V + 2.5 log10(2 pi
    rhalf**2 (1 - ellipticity))`` the mean surface brightness within the
    half-light ellipse (mag/arcmin²; half the light over its area, ``rhalf``
    its semi-major axis; a circle when the ellipticity is unknown): see the
    comment of these constants for the measurement in Euclid DR1. E.g.
    NGC 1851 27 (14 arcmin), NGC 1261 18 (12 arcmin), NGC 1904 20 (13
    arcmin), Eridanus III 4.

    Parameters
    ----------
    clusters : pandas.DataFrame
        Columns ``apparent_magnitude_v`` (V), ``radius_arcmin`` (half-light
        radius) and ``ellipticity``, e.g.
        ``catalogs.overlapped_objects.get_globular_clusters()``.

    Returns
    -------
    scales : numpy.ndarray
        One per cluster; ``DEFAULT_RADIUS_SCALE`` when V (or its column) or the
        radius is unknown.
    """
    if "apparent_magnitude_v" not in clusters:
        return np.full(len(clusters), DEFAULT_RADIUS_SCALE)
    v_mag = clusters["apparent_magnitude_v"].to_numpy(dtype=float)
    radius = clusters["radius_arcmin"].to_numpy(dtype=float)
    ellipticity = clusters["ellipticity"].to_numpy(dtype=float) if "ellipticity" in clusters else 0.0
    axis_ratio = 1.0 - np.nan_to_num(ellipticity, nan=0.0)  # unknown: a circle
    with np.errstate(divide="ignore", invalid="ignore"):
        surface_brightness = v_mag + 2.5 * np.log10(2 * np.pi * radius**2 * axis_ratio)
        scales = GC_REFERENCE_RADIUS_SCALE * 10 ** (
            -0.4 * (surface_brightness - GC_REFERENCE_SURFACE_BRIGHTNESS) / GC_PROFILE_SLOPE
        )
    return np.where(np.isfinite(scales), np.maximum(scales, DEFAULT_RADIUS_SCALE), DEFAULT_RADIUS_SCALE)


def mask_radius_scales(
    objects: pd.DataFrame, object_type: str, radius_scale: float | None = None
) -> NDArray[np.float64]:
    """Mask scale of each object of a type, in units of its reference radius.

    Parameters
    ----------
    objects : pandas.DataFrame
        Objects of the type (columns of ``catalogs.overlapped_objects``).
    object_type : str
        One of ``OBJECT_TYPES``.
    radius_scale : float, optional
        The same scale for every object. None: the default of the type,
        ``DEFAULT_RADIUS_SCALES[object_type]``; for the globular clusters,
        ``globular_cluster_radius_scales``.

    Returns
    -------
    scales : numpy.ndarray
        One per object.

    Raises
    ------
    ValueError
        ``object_type`` not in ``OBJECT_TYPES``.
    """
    if object_type not in OBJECT_TYPES:
        raise ValueError(f"object_type {object_type!r} not in {OBJECT_TYPES}.")
    if radius_scale is None:
        radius_scale = DEFAULT_RADIUS_SCALES[object_type]
    if radius_scale is None:
        return globular_cluster_radius_scales(objects)
    return np.full(len(objects), float(radius_scale))


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


def lmc_healpix_mask(nside: int, *, nest: bool, radius_scale: float | None = LMC_RADIUS_SCALE) -> NDArray[np.bool_]:
    """HEALPix mask of the LMC (``catalogs.overlapped_objects.get_lmc``).

    By default, 21 deg along the major axis (LVDB v1.1.2: half-light radius
    3.2 deg, ellipticity 0.16, major axis at 227 deg). Parameters and returns:
    see ``objects_healpix_mask``.
    """
    lmc = overlapped_objects.get_lmc()
    return objects_healpix_mask(lmc, nside, nest=nest, radius_scale=mask_radius_scales(lmc, "lmc", radius_scale))


def smc_healpix_mask(nside: int, *, nest: bool, radius_scale: float | None = SMC_RADIUS_SCALE) -> NDArray[np.bool_]:
    """HEALPix mask of the SMC (``catalogs.overlapped_objects.get_smc``).

    By default, a circle of 11 deg (LVDB v1.1.2: half-light radius 1.0 deg, no
    ellipticity). Parameters and returns: see ``objects_healpix_mask``.
    """
    smc = overlapped_objects.get_smc()
    return objects_healpix_mask(smc, nside, nest=nest, radius_scale=mask_radius_scales(smc, "smc", radius_scale))


def dwarf_galaxies_healpix_mask(
    nside: int, *, nest: bool, radius_scale: float | None = DEFAULT_RADIUS_SCALE
) -> NDArray[np.bool_]:
    """HEALPix mask of the dwarf galaxies but the LMC and SMC (``catalogs.overlapped_objects.get_dwarf_galaxies``).

    Parameters and returns: see ``objects_healpix_mask``.
    """
    dwarfs = overlapped_objects.get_dwarf_galaxies()
    return objects_healpix_mask(dwarfs, nside, nest=nest,
                                radius_scale=mask_radius_scales(dwarfs, "dwarfs", radius_scale))


def globular_clusters_healpix_mask(
    nside: int, *, nest: bool, radius_scale: float | None = None
) -> NDArray[np.bool_]:
    """HEALPix mask of the globular clusters (``catalogs.overlapped_objects.get_globular_clusters``).

    By default (``radius_scale`` None), each cluster has its own scale, from
    its surface brightness (``globular_cluster_radius_scales``); a float: the
    same for every cluster. Other parameters and returns: see ``objects_healpix_mask``.
    """
    clusters = overlapped_objects.get_globular_clusters()
    return objects_healpix_mask(clusters, nside, nest=nest,
                                radius_scale=mask_radius_scales(clusters, "gc", radius_scale))


def galaxies_healpix_mask(
    nside: int, *, nest: bool, radius_scale: float | None = GALAXY_RADIUS_SCALE
) -> NDArray[np.bool_]:
    """HEALPix mask of the bright galaxies (``catalogs.overlapped_objects.get_galaxies``).

    Circles of ``radius_scale`` times ``D26 / 2`` (their point-like sources
    spread rounder than their isophote), without the pixel of their centre
    forced: a galaxy smaller than a pixel has few point-like sources, and
    forcing the pixels of the 82 000 galaxies would mask about 1000 deg2 of
    the sky at nside 512. E.g. M81: 1.1 deg, M82: 0.6 deg. Parameters and
    returns: see ``objects_healpix_mask``.
    """
    galaxies = overlapped_objects.get_galaxies().assign(ellipticity=np.nan)  # circles
    return objects_healpix_mask(galaxies, nside, nest=nest,
                                radius_scale=mask_radius_scales(galaxies, "galaxies", radius_scale),
                                mask_centre_pixel=False)


def overlapped_objects_healpix_mask(
    nside: int,
    *,
    nest: bool,
    mask_lmc: bool = False,
    mask_smc: bool = False,
    mask_dwarfs: bool = False,
    mask_gc: bool = False,
    mask_galaxies: bool = False,
    radius_scale: float | Mapping[str, float | None] = DEFAULT_RADIUS_SCALES,
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
        "galaxies": ...}`` (``OBJECT_TYPES``; a type left out, or None, has
        its default, ``DEFAULT_RADIUS_SCALES``: 6.5 for the LMC, 11 for the
        SMC, 4 for the dwarfs, 5 for the galaxies, and for the clusters one
        per cluster from its surface brightness,
        ``globular_cluster_radius_scales``).

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
        given = {key: value for key, value in radius_scale.items() if value is not None}  # None: the default
        scales = {**DEFAULT_RADIUS_SCALES, **given}
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


def excluded_healpix_mask(
    footprint_mask: NDArray[np.bool_],
    object_healpix_mask: NDArray[np.bool_] | None,
    *,
    nest: bool,
) -> tuple[NDArray[np.bool_], int]:
    """HEALPix mask of the excluded sky: outside the footprint, or on a masked object.

    One mask for the stars, the maps and the areas of a search in a stream
    frame. The footprint is brought to the nside of the object mask and to
    RING (``catalogs.survey_footprint.convert_footprint_healpix_mask``: each
    sub-pixel takes the value of its pixel, exact), then OR the objects.

    Parameters
    ----------
    footprint_mask : numpy.ndarray of bool
        True outside the footprint
        (``catalogs.survey_footprint.survey_footprint_healpix_mask``).
    object_healpix_mask : numpy.ndarray of bool or None
        True on the masked objects, RING, e.g. ``object_healpix_mask`` of
        ``analysis.stream_frame_search.PreparedSearch``
        (``overlapped_objects_healpix_mask`` at ``object_mask_nside``). None:
        the footprint alone, in RING at its own nside.
    nest : bool
        HEALPix ordering of ``footprint_mask``.

    Returns
    -------
    mask : numpy.ndarray of bool
        True where excluded, RING.
    nside : int
        Its resolution: that of ``object_healpix_mask``, or of the footprint.

    Raises
    ------
    ValueError
        The object mask is coarser than the footprint (going down would
        lose excluded pixels).
    """
    footprint_mask = np.asarray(footprint_mask, dtype=bool)
    footprint_nside = hp.npix2nside(footprint_mask.size)
    if object_healpix_mask is None:
        return convert_footprint_healpix_mask(footprint_mask, footprint_nside, nest=nest, nest_out=False), footprint_nside
    object_healpix_mask = np.asarray(object_healpix_mask, dtype=bool)
    nside = hp.npix2nside(object_healpix_mask.size)
    if nside < footprint_nside:
        raise ValueError(f"Object mask at nside {nside}, coarser than the footprint ({footprint_nside}).")
    footprint = convert_footprint_healpix_mask(footprint_mask, nside, nest=nest, nest_out=False)
    return footprint | object_healpix_mask, nside

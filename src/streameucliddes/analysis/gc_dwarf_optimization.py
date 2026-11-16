"""Tune the cuts and match filters on known dwarf galaxies and globular clusters.

A known object at a distance modulus of interest is a real old, metal-poor
population of known distance: the signal of a selection (cuts, colour or
magnitude ranges, a match filter in some bands) is its excess of stars on
the object over the field around it. Steps, in a notebook:

1. ``objects_near_distance_modulus``: the LVDB dwarf galaxies and globular
   clusters (Local Volume Database, Pace 2024, through
   ``catalogs.overlapped_objects``) near a distance modulus, flagged when
   they are in the catalog (``in_catalog``). ``get_object`` picks one by name.
2. ``object_radec_box``: the (ra, dec) box of its field, to read the stars
   (``catalogs.euclid_catalog.get_euclid_stars(..., ra_range=, dec_range=)``).
3. ``object_regions``: the on region (an ellipse, or an elliptical annulus,
   between ``on_radii`` times the half-light radius of the object), the off
   region (an elliptical annulus between ``off_radii``), the stars excluded
   (outside the footprint, or on another known object) and the areas. For
   a globular cluster, start the on annulus beyond its core, where the
   catalog misses stars (crowding; NGC 1851 in DR1: no star within 4
   half-light radii), and the off annulus beyond its stars (NGC 1851: about
   20 half-light radii).
4. Selections: boolean arrays over the rows of the catalog, combined with
   ``&`` and ``|``: ``stars_in_match_filter``, ``stars_passing_cuts``, or any
   expression on the catalog columns, e.g.
   ``catalog["des_yr6_g_obs"] - catalog["des_yr6_r_obs"] < 0.8``.
5. ``selection_snr`` of one selection, ``compare_selections`` of several:
   SNR = excess on the object / sqrt(background expected there), the
   definition of ``analysis.injected_stream_snr``.

The plots (map, CMD, Hess diagrams) are in
``plotting.plot_gc_dwarf_optimization``. The catalog is neither copied nor
modified: the regions and the selections are arrays of one value per star.
The catalog magnitudes are already dereddened.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

import astropy.units as u
import healpy as hp
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, angular_separation

from streameucliddes.analysis.build_masks import DEFAULT_RADIUS_SCALES, elliptical_radius_deg
from streameucliddes.analysis.catalog_preparation import prepare_match_filter_catalog
from streameucliddes.analysis.cmd_match_filter import (
    DEFAULT_COLOR_BIN,
    DEFAULT_MAG_BIN,
    difference_hess,
    hess_diagram,
    match_filter_columns,
    match_filter_polygons,
    select_stars_in_match_filter,
)
from streameucliddes.catalogs import overlapped_objects
from streameucliddes.catalogs.euclid_catalog import catalog_point_map
from streameucliddes.catalogs.isochrone_parameters import default_isochrone_grid
from streameucliddes.catalogs.survey_footprint import (
    star_count_nside,
    survey_footprint_healpix_mask,
    trim_footprint_edge,
)
from streameucliddes.unit_conversions import convert_FeH_to_z, convert_kpc_to_DM

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from numpy.typing import ArrayLike, NDArray

# Types of known objects (keys of build_masks.DEFAULT_RADIUS_SCALES): those searched by distance modulus,
# and those masked around an object
OBJECT_TYPES = ("lmc", "smc", "dwarfs", "gc")
DEFAULT_OBJECT_TYPES = ("dwarfs", "gc")
DEFAULT_MASKED_OBJECT_TYPES = ("dwarfs", "gc", "galaxies")
DEFAULT_MAX_DM_OFFSET = 0.5  # mag
# Regions, in half-light radii (semi-major axis of the LVDB half-light ellipse). The off annulus starts beyond
# the mask of the objects in the searches (build_masks.DEFAULT_RADIUS_SCALE, 4)
DEFAULT_ON_RADII = (0.0, 2.0)
DEFAULT_OFF_RADII = (5.0, 10.0)
# Box read around an object: its field plus this margin (deg), room for the footprint edge pixels trimmed
BOX_MARGIN_DEG = 0.3
# Footprint from the stars of the field (as CandidateInvestigation.prepare): finest nside, coarser when sparse
DEFAULT_FOOTPRINT_NSIDE = 1024
# Areas: grid of points in the frame centred on the object, so many points across the on region (its outer
# minus inner radius), at most so many points along an axis
AREA_GRID_POINTS_ACROSS_ON = 20
AREA_GRID_MAX_POINTS = 1000
# CMDs of a match filter: the extent of its polygons plus this margin (mag)
CMD_MARGIN_MAG = 0.5
# Two LVDB entries closer than this are the same object (e.g. in two tables): not masked around it
SAME_OBJECT_SEPARATION_ARCSEC = 10.0

_LVDB_TABLES = {
    "lmc": (overlapped_objects.DWARF_TABLE,), "smc": (overlapped_objects.DWARF_TABLE,),
    "dwarfs": (overlapped_objects.DWARF_TABLE,), "gc": overlapped_objects.GLOBULAR_CLUSTER_TABLES,
}


# ===================================================
# Known objects
# ===================================================


def known_objects(object_types: Sequence[str] = DEFAULT_OBJECT_TYPES) -> pd.DataFrame:
    """Known objects of the LVDB, with their distance modulus and [Fe/H].

    Parameters
    ----------
    object_types : sequence of str, optional
        Among ``OBJECT_TYPES``: ``"lmc"``, ``"smc"``, ``"dwarfs"`` (but the
        Magellanic Clouds), ``"gc"`` (globular clusters).

    Returns
    -------
    objects : pandas.DataFrame
        One row per object: ``type``, the ``catalogs.overlapped_objects.OBJECT_COLUMNS``
        (``radius_arcmin`` the half-light radius), ``distance_modulus`` (LVDB;
        from ``distance_kpc`` when missing) and ``feh`` (LVDB ``metallicity``,
        its best [Fe/H]; NaN when unknown).

    Raises
    ------
    ValueError
        A type not in ``OBJECT_TYPES``.
    """
    unknown = set(object_types) - set(OBJECT_TYPES)
    if unknown:
        raise ValueError(f"object types {sorted(unknown)} not in {OBJECT_TYPES}.")
    getters = {"lmc": overlapped_objects.get_lmc, "smc": overlapped_objects.get_smc,
               "dwarfs": overlapped_objects.get_dwarf_galaxies, "gc": overlapped_objects.get_globular_clusters}
    tables = []
    for object_type in object_types:
        objects = getters[object_type]()
        lvdb = [overlapped_objects.load_lvdb_table(table) for table in _LVDB_TABLES[object_type]]
        lvdb = pd.concat([table for table in lvdb if not table.empty] or lvdb[:1], ignore_index=True)
        lvdb = (lvdb.reindex(columns=["key", "distance_modulus", "metallicity"])
                .drop_duplicates("key").rename(columns={"metallicity": "feh"}))
        tables.append(objects.merge(lvdb, on="key", how="left").assign(type=object_type))
    objects = pd.concat(tables, ignore_index=True)
    objects["distance_modulus"] = pd.to_numeric(objects["distance_modulus"], errors="coerce")
    objects["feh"] = pd.to_numeric(objects["feh"], errors="coerce")
    missing = objects["distance_modulus"].isna() & (objects["distance_kpc"] > 0)
    if missing.any():
        objects.loc[missing, "distance_modulus"] = convert_kpc_to_DM(objects.loc[missing, "distance_kpc"]).value
    return objects[["type", *overlapped_objects.OBJECT_COLUMNS, "distance_modulus", "feh"]]


def objects_near_distance_modulus(
    distance_modulus: float,
    max_offset: float = DEFAULT_MAX_DM_OFFSET,
    *,
    object_types: Sequence[str] = DEFAULT_OBJECT_TYPES,
    catalog: str | None = None,
    data_path: Any = None,
) -> pd.DataFrame:
    """Known objects whose distance modulus is within ``max_offset`` of ``distance_modulus``.

    Parameters
    ----------
    distance_modulus : float
        Distance modulus of interest (mag).
    max_offset : float, optional
        Largest ``|object distance modulus - distance_modulus|`` (mag).
    object_types : sequence of str, optional
        See ``known_objects``.
    catalog : str, optional
        Key of ``catalogs.euclid_catalog.CATALOGS``: adds
        ``n_sources_at_centre``, the sources of the catalog in the HEALPix
        pixel of the centre of each object (``catalog_point_map``, nside
        1024 for DR1, about 3.4 arcmin), and ``in_catalog`` (at least one).
    data_path : str or Path, optional
        Folder of the HATS catalog, when it is not the default one.

    Returns
    -------
    objects : pandas.DataFrame
        The columns of ``known_objects``, plus ``dm_offset`` (object minus
        ``distance_modulus``), sorted by ``|dm_offset|``.
    """
    objects = known_objects(object_types)
    objects["dm_offset"] = objects["distance_modulus"] - distance_modulus
    objects = objects[objects["dm_offset"].abs() <= max_offset]
    objects = objects.iloc[np.argsort(objects["dm_offset"].abs().to_numpy(), kind="stable")].reset_index(drop=True)
    if catalog is not None:
        counts = catalog_point_map(catalog, data_path)
        pixels = hp.ang2pix(hp.npix2nside(counts.size), objects["ra"].to_numpy(), objects["dec"].to_numpy(),
                            nest=True, lonlat=True)
        objects["n_sources_at_centre"] = counts[pixels].astype(int)
        objects["in_catalog"] = objects["n_sources_at_centre"] > 0
    return objects


def get_object(name: str, object_types: Sequence[str] = OBJECT_TYPES) -> pd.Series:
    """One known object, by its LVDB key or name.

    Case, spaces, underscores and hyphens are ignored: ``"NGC 1851"``,
    ``"ngc_1851"`` and ``"ngc1851"`` are the same.

    Parameters
    ----------
    name : str
        LVDB key or name.
    object_types : sequence of str, optional
        Types searched (see ``known_objects``).

    Returns
    -------
    object : pandas.Series
        The row of ``known_objects``.

    Raises
    ------
    KeyError
        No such object; the message lists close names.
    ValueError
        Several objects of different keys have this name.
    """
    objects = known_objects(object_types)
    wanted = _simple_name(name)
    match = objects[(objects["key"].map(_simple_name) == wanted) | (objects["name"].map(_simple_name) == wanted)]
    match = match.drop_duplicates("key")
    if match.empty:
        close = difflib.get_close_matches(name, objects["name"].astype(str).tolist(), n=5)
        raise KeyError(f"No known object {name!r}. Close names: {close}")
    if len(match) > 1:
        raise ValueError(f"Several objects named {name!r}: keys {match['key'].tolist()}")
    return match.iloc[0]


def _simple_name(name: Any) -> str:
    """Name without case, spaces, underscores and hyphens."""
    return str(name).lower().replace(" ", "").replace("_", "").replace("-", "")


def object_isochrone(obj: Mapping[str, Any], age: float | None = None) -> dict[str, float]:
    """Isochrone of an object: its [Fe/H] (LVDB) as Z, and an old age.

    Parameters
    ----------
    obj : mapping
        A row of ``known_objects`` (``get_object``).
    age : float, optional
        Age (Gyr). Default: that of ``catalogs.isochrone_parameters.default_isochrone_grid``.

    Returns
    -------
    isochrone : dict
        ``{"age": age (Gyr), "metallicity": Z}``, Z from ``feh``
        (``unit_conversions.convert_FeH_to_z``, Marigo et al. 2017 scale);
        the Z of the default grid when ``feh`` is unknown.
    """
    default = default_isochrone_grid()[0]
    feh = obj.get("feh", np.nan)
    metallicity = default["metallicity"] if not np.isfinite(feh) else float(convert_FeH_to_z(feh))
    return {"age": float(default["age"] if age is None else age), "metallicity": metallicity}


# ===================================================
# Regions around an object
# ===================================================


def object_field_radius_deg(obj: Mapping[str, Any], off_radii: tuple[float, float] = DEFAULT_OFF_RADII) -> float:
    """Radius of the field of an object (deg): the semi-major axis of the outer edge of the off annulus."""
    return float(off_radii[1] * obj["radius_arcmin"] / 60.0)


def object_radec_box(
    obj: Mapping[str, Any],
    radius_deg: float | None = None,
    *,
    margin_deg: float = BOX_MARGIN_DEG,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """(ra, dec) box around an object, to read the stars of its field.

    Parameters
    ----------
    obj : mapping
        A row of ``known_objects``.
    radius_deg : float, optional
        Radius around the centre (deg). Default: ``object_field_radius_deg``.
    margin_deg : float, optional
        Added to the radius.

    Returns
    -------
    ra_range, dec_range : tuple of float
        For ``catalogs.euclid_catalog.get_euclid_stars``; ``ra_range`` may
        cross RA = 0 (``ra_min > ra_max``); the whole RA range near a pole.
    """
    radius = (object_field_radius_deg(obj) if radius_deg is None else radius_deg) + margin_deg
    ra, dec = float(obj["ra"]), float(obj["dec"])
    dec_range = (max(dec - radius, -90.0), min(dec + radius, 90.0))
    widest_dec = abs(dec) + radius
    if widest_dec >= 90.0:
        return (0.0, 360.0), dec_range
    half_width = radius / np.cos(np.radians(widest_dec))
    if half_width >= 180.0:
        return (0.0, 360.0), dec_range
    return ((ra - half_width) % 360.0, (ra + half_width) % 360.0), dec_range


def object_regions(
    catalog: pd.DataFrame,
    obj: Mapping[str, Any],
    *,
    on_radii: tuple[float, float] = DEFAULT_ON_RADII,
    off_radii: tuple[float, float] = DEFAULT_OFF_RADII,
    masked_object_types: Sequence[str] = DEFAULT_MASKED_OBJECT_TYPES,
    footprint_mask: NDArray[np.bool_] | None = None,
    footprint_nest: bool = False,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> dict[str, Any]:
    """On and off regions of an object: the stars in each and their areas.

    Radii are elliptical, in half-light radii: a star at radius ``r`` is on
    the ellipse of semi-major axis ``r`` times the half-light radius, with
    the ellipticity and position angle of the object (LVDB;
    ``build_masks.elliptical_radius_deg``, a circle when unknown). The on
    region is ``on_radii[0] <= r <= on_radii[1]``, the off region
    ``off_radii[0] <= r <= off_radii[1]``. Stars are excluded outside the footprint, or on another
    known object, masked as in the searches
    (``build_masks.DEFAULT_RADIUS_SCALES`` times its reference radius); the
    areas leave out the same sky. They are computed on a grid of points
    around the object (``AREA_GRID_POINTS_ACROSS_ON``).

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars of the field, e.g. read in ``object_radec_box``. Not copied.
    obj : mapping
        A row of ``known_objects`` (``get_object``), with a half-light radius.
    on_radii : tuple of float, optional
        Inner and outer radii of the on region (half-light radii): inner 0
        for an ellipse.
    off_radii : tuple of float, optional
        Inner and outer radii of the off annulus (half-light radii).
    masked_object_types : sequence of str, optional
        Types of the other objects masked: among ``"lmc"``, ``"smc"``,
        ``"dwarfs"``, ``"gc"``, ``"galaxies"`` (Siena Galaxy Atlas).
    footprint_mask : numpy.ndarray of bool, optional
        HEALPix footprint, True outside. Default: from the stars of
        ``catalog`` (``survey_footprint_healpix_mask`` at
        ``DEFAULT_FOOTPRINT_NSIDE``, coarser for sparse stars:
        ``star_count_nside``), less its edge pixels (``trim_footprint_edge``).
    footprint_nest : bool, optional
        Ordering of a given ``footprint_mask``.
    ra_col, dec_col : str, optional
        Coordinate columns (deg).

    Returns
    -------
    regions : dict
        Per star (arrays of ``len(catalog)``): ``elliptical_radius``
        (half-light radii), ``offset_east_deg`` and ``offset_north_deg``
        (in the frame centred on the object, ``astropy.coordinates.SkyOffsetFrame``),
        ``excluded``, ``on`` and ``off`` (bool, not excluded). And
        ``on_area_deg2``, ``off_area_deg2``, ``on_area_fraction`` and
        ``off_area_fraction`` (of the full ellipses not excluded),
        ``object``, ``half_light_radius_deg``, ``on_radii``, ``off_radii``,
        ``masked_objects`` (the other objects masked in the field, with
        their ``mask_radius_arcmin`` and offsets), ``footprint_mask``,
        ``footprint_nest``.

    Raises
    ------
    ValueError
        The object has no half-light radius; radii not increasing.
    """
    half_light_deg = float(obj["radius_arcmin"]) / 60.0
    if not np.isfinite(half_light_deg) or half_light_deg <= 0:
        raise ValueError(f"{obj['name']} has no half-light radius in the LVDB.")
    if not 0 <= on_radii[0] < on_radii[1] <= off_radii[0] < off_radii[1]:
        raise ValueError(f"Need 0 <= on_radii[0] < on_radii[1] <= off_radii[0] < off_radii[1], not {on_radii}, "
                         f"{off_radii}.")
    ellipticity = _finite_or_zero(obj.get("ellipticity", np.nan))
    major_axis_angle = _finite_or_zero(obj.get("position_angle", np.nan))

    ra, dec = catalog[ra_col].to_numpy(dtype=float), catalog[dec_col].to_numpy(dtype=float)
    if footprint_mask is None:
        nside = star_count_nside(ra, dec, DEFAULT_FOOTPRINT_NSIDE)
        footprint_mask = trim_footprint_edge(survey_footprint_healpix_mask(ra, dec, nside, nest=False), nest=False)
        footprint_nest = False
    footprint_mask = np.asarray(footprint_mask, dtype=bool)

    field_radius_deg = off_radii[1] * half_light_deg
    masked_objects = _masked_objects_in_field(obj, field_radius_deg, masked_object_types)

    def radius_and_excluded(points_ra, points_dec):
        radius = elliptical_radius_deg(points_ra, points_dec, obj["ra"], obj["dec"], ellipticity,
                                       major_axis_angle) / half_light_deg
        excluded = _outside_footprint(points_ra, points_dec, footprint_mask, footprint_nest)
        for _, other in masked_objects.iterrows():
            excluded |= elliptical_radius_deg(
                points_ra, points_dec, other["ra"], other["dec"], _finite_or_zero(other["ellipticity"]),
                _finite_or_zero(other["position_angle"]),
            ) <= other["mask_radius_arcmin"] / 60.0
        return radius, excluded

    radius, excluded = radius_and_excluded(ra, dec)
    centre = SkyCoord(float(obj["ra"]) * u.deg, float(obj["dec"]) * u.deg)
    offsets = SkyCoord(ra * u.deg, dec * u.deg).transform_to(centre.skyoffset_frame())
    masked_offsets = SkyCoord(masked_objects["ra"].to_numpy(dtype=float) * u.deg,
                              masked_objects["dec"].to_numpy(dtype=float) * u.deg).transform_to(centre.skyoffset_frame())
    masked_objects = masked_objects.assign(offset_east_deg=masked_offsets.lon.wrap_at(180 * u.deg).deg,
                                           offset_north_deg=masked_offsets.lat.deg)

    # Areas: points of a grid in the frame of the object, each of area cos(lat) * step**2
    step = max((on_radii[1] - on_radii[0]) * half_light_deg / AREA_GRID_POINTS_ACROSS_ON,
               2 * field_radius_deg / AREA_GRID_MAX_POINTS)
    axis = np.arange(-field_radius_deg + step / 2, field_radius_deg, step)
    grid_lon, grid_lat = np.meshgrid(axis, axis)
    grid = SkyCoord(grid_lon.ravel() * u.deg, grid_lat.ravel() * u.deg, frame=centre.skyoffset_frame()).icrs
    grid_radius, grid_excluded = radius_and_excluded(grid.ra.deg, grid.dec.deg)
    cell_area = np.cos(np.radians(grid_lat.ravel())) * step**2
    on_grid = (grid_radius >= on_radii[0]) & (grid_radius <= on_radii[1])
    off_grid = (grid_radius >= off_radii[0]) & (grid_radius <= off_radii[1])
    on_full_area = cell_area[on_grid].sum()
    off_full_area = cell_area[off_grid].sum()
    on_area = cell_area[on_grid & ~grid_excluded].sum()
    off_area = cell_area[off_grid & ~grid_excluded].sum()

    return {
        "object": obj, "half_light_radius_deg": half_light_deg, "on_radii": (float(on_radii[0]), float(on_radii[1])),
        "off_radii": (float(off_radii[0]), float(off_radii[1])),
        "elliptical_radius": radius,
        "offset_east_deg": offsets.lon.wrap_at(180 * u.deg).deg, "offset_north_deg": offsets.lat.deg,
        "excluded": excluded,
        "on": ~excluded & (radius >= on_radii[0]) & (radius <= on_radii[1]),
        "off": ~excluded & (radius >= off_radii[0]) & (radius <= off_radii[1]),
        "on_area_deg2": float(on_area), "off_area_deg2": float(off_area),
        "on_area_fraction": float(on_area / on_full_area), "off_area_fraction": float(off_area / off_full_area),
        "masked_objects": masked_objects,
        "footprint_mask": footprint_mask, "footprint_nest": footprint_nest,
    }


def _finite_or_zero(value: Any) -> float:
    """value as a float, 0 when unknown (NaN)."""
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _outside_footprint(
    ra: NDArray[np.float64], dec: NDArray[np.float64], footprint_mask: NDArray[np.bool_], nest: bool
) -> NDArray[np.bool_]:
    """Points outside the footprint (True pixels), or without finite coordinates."""
    finite = np.isfinite(ra) & np.isfinite(dec)
    outside = ~finite
    nside = hp.npix2nside(footprint_mask.size)
    outside[finite] = footprint_mask[hp.ang2pix(nside, ra[finite], dec[finite], nest=nest, lonlat=True)]
    return outside


def _masked_objects_in_field(
    obj: Mapping[str, Any], field_radius_deg: float, object_types: Sequence[str]
) -> pd.DataFrame:
    """The other known objects whose mask reaches the field, with ``mask_radius_arcmin``."""
    tables = []
    lvdb_types = [object_type for object_type in object_types if object_type != "galaxies"]
    if lvdb_types:
        tables.append(known_objects(lvdb_types))
    if "galaxies" in object_types:  # circles, as build_masks.galaxies_healpix_mask
        tables.append(overlapped_objects.get_galaxies().assign(type="galaxies", ellipticity=np.nan))
    columns = ["type", *overlapped_objects.OBJECT_COLUMNS, "mask_radius_arcmin"]
    if not tables:
        return pd.DataFrame(columns=columns)
    objects = pd.concat([table for table in tables if not table.empty] or tables[:1], ignore_index=True)
    objects["mask_radius_arcmin"] = objects["type"].map(DEFAULT_RADIUS_SCALES) * objects["radius_arcmin"]
    separation_deg = np.degrees(angular_separation(
        np.radians(float(obj["ra"])), np.radians(float(obj["dec"])),
        np.radians(objects["ra"].to_numpy(dtype=float)), np.radians(objects["dec"].to_numpy(dtype=float)),
    ))
    reaches = separation_deg <= field_radius_deg + objects["mask_radius_arcmin"].to_numpy() / 60.0
    itself = (objects["key"].to_numpy() == obj["key"]) | (separation_deg * 3600.0 < SAME_OBJECT_SEPARATION_ARCSEC)
    keep = reaches & ~itself & np.isfinite(objects["mask_radius_arcmin"].to_numpy())
    return objects.loc[keep, columns].reset_index(drop=True)


# ===================================================
# Selections and SNR
# ===================================================


def stars_passing_cuts(
    catalog: pd.DataFrame,
    match_filter: Mapping[str, Any] | None = None,
    cuts: Mapping[str, Any] | None = None,
) -> NDArray[np.bool_]:
    """Stars passing the cuts of a match filter: errors <= 0.2 in its bands, updated by ``cuts``.

    ``catalog_preparation.prepare_match_filter_catalog`` with ``as_mask``:
    nothing copied.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars.
    match_filter : mapping, optional
        See ``cmd_match_filter``. Default: ``DEFAULT_MATCH_FILTER``.
    cuts : mapping, optional
        Format of ``catalogs.catalog_cuts.get_cut_mask``; they update the
        default error cuts (``(None, None)`` removes one).

    Returns
    -------
    selected : numpy.ndarray of bool, shape (len(catalog),)
    """
    return prepare_match_filter_catalog(catalog, match_filter, cuts=cuts, as_mask=True)


def stars_in_match_filter(
    catalog: pd.DataFrame,
    distance_modulus: float,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    cuts: Mapping[str, Any] | None = None,
    mask: ArrayLike | None = None,
) -> NDArray[np.bool_]:
    """Stars passing the cuts (``stars_passing_cuts``) and inside the match filter.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars.
    distance_modulus : float
        Distance modulus of the filter, e.g. ``obj["distance_modulus"]``.
    isochrone : mapping, optional
        ``{"age", "metallicity"}``, e.g. ``object_isochrone(obj)``. Default:
        that of ``default_isochrone_grid``.
    match_filter : mapping, optional
        See ``cmd_match_filter``. Default: ``DEFAULT_MATCH_FILTER``.
    cuts : mapping, optional
        See ``stars_passing_cuts``.
    mask : array_like of bool, optional
        Stars to consider (e.g. your own colour or magnitude cut); the
        others are not selected.

    Returns
    -------
    selected : numpy.ndarray of bool, shape (len(catalog),)
    """
    keep = stars_passing_cuts(catalog, match_filter, cuts)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    return select_stars_in_match_filter(catalog, distance_modulus, isochrone, match_filter, mask=keep)


def selection_snr(regions: Mapping[str, Any], selected: ArrayLike | None = None) -> dict[str, float]:
    """Excess of the selected stars on the object over the off region, and its SNR.

    The background expected on the object is the off count scaled by the
    ratio of the areas. SNR = excess / sqrt(background on the object), as
    ``analysis.injected_stream_snr``.

    Parameters
    ----------
    regions : mapping
        From ``object_regions``.
    selected : array_like of bool, optional
        Selected stars, one per row of the catalog. Default: all.

    Returns
    -------
    snr : dict
        ``n_on``, ``n_off`` (selected stars in each region),
        ``n_background_on`` (``n_off * on_area / off_area``), ``n_signal``
        (``n_on - n_background_on``), ``n_signal_error`` (Poisson, on and
        off), ``snr`` (``n_signal / sqrt(n_background_on)``; inf without
        background), ``background_density`` (stars/deg²), ``on_area_deg2``,
        ``off_area_deg2``.
    """
    on, off = np.asarray(regions["on"], dtype=bool), np.asarray(regions["off"], dtype=bool)
    if selected is not None:
        selected = np.asarray(selected, dtype=bool)
        on, off = on & selected, off & selected
    n_on, n_off = int(on.sum()), int(off.sum())
    on_area, off_area = regions["on_area_deg2"], regions["off_area_deg2"]
    area_ratio = on_area / off_area
    n_background = n_off * area_ratio
    n_signal = n_on - n_background
    if n_background > 0:
        snr = n_signal / np.sqrt(n_background)
    else:
        snr = np.inf if n_signal > 0 else 0.0
    return {
        "n_on": n_on, "n_off": n_off, "n_background_on": float(n_background), "n_signal": float(n_signal),
        "n_signal_error": float(np.sqrt(n_on + area_ratio**2 * n_off)), "snr": float(snr),
        "background_density": n_off / off_area, "on_area_deg2": on_area, "off_area_deg2": off_area,
    }


def compare_selections(regions: Mapping[str, Any], selections: Mapping[str, ArrayLike]) -> pd.DataFrame:
    """``selection_snr`` of several selections, one row each.

    Parameters
    ----------
    regions : mapping
        From ``object_regions``.
    selections : mapping
        ``{name: selected stars}``, e.g. ``{"VIS-H": stars_in_match_filter(...),
        "g-r, g < 24": ... & (catalog["des_yr6_g_obs"] < 24)}``.

    Returns
    -------
    table : pandas.DataFrame
        Index: the names, in order; columns: those of ``selection_snr``.
    """
    rows = {name: selection_snr(regions, selected) for name, selected in selections.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


# ===================================================
# Colour-magnitude diagrams
# ===================================================


def match_filter_color_mag_pairs(match_filter: Mapping[str, Any] | None = None) -> list[tuple[str, str]]:
    """Magnitude columns of each CMD of a match filter: (reference band, colour band), one per colour band."""
    reference_column, *color_columns = match_filter_columns(match_filter)
    return [(reference_column, color_column) for color_column in color_columns]


def match_filter_cmd_bins(
    distance_modulus: float,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    margin_mag: float = CMD_MARGIN_MAG,
    color_bin: float = DEFAULT_COLOR_BIN,
    mag_bin: float = DEFAULT_MAG_BIN,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Hess bins around the polygons of a match filter: their extent plus ``margin_mag``.

    Returns
    -------
    bins : list of (color_edges, mag_edges)
        One per colour band of the match filter.
    """
    bins = []
    for polygon in match_filter_polygons(distance_modulus, isochrone, match_filter):
        color, mag = polygon[:, 0], polygon[:, 1]
        bins.append((np.arange(color.min() - margin_mag, color.max() + margin_mag + color_bin / 2, color_bin),
                     np.arange(mag.min() - margin_mag, mag.max() + margin_mag + mag_bin / 2, mag_bin)))
    return bins


def object_hess_diagrams(
    catalog: pd.DataFrame,
    regions: Mapping[str, Any],
    mag_cols: Sequence[str],
    *,
    stars: ArrayLike | None = None,
    color_bins: ArrayLike | None = None,
    mag_bins: ArrayLike | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """On, off and on - off Hess diagrams of an object (stars/deg²/bin).

    ``cmd_match_filter.hess_diagram`` of the stars of each region over its
    area, and ``difference_hess``. Any pair of magnitude columns: colour
    ``mag_cols[0] - mag_cols[1]``, magnitude ``mag_cols[0]``.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars, with the ``mag_cols`` columns.
    regions : mapping
        From ``object_regions``.
    mag_cols : sequence of two str
        Magnitude columns.
    stars : array_like of bool, optional
        Stars drawn, e.g. ``stars_passing_cuts(catalog, match_filter)`` or
        your own cut. Default: all.
    color_bins, mag_bins : array_like, optional
        Bin edges (default: those of ``hess_diagram``).

    Returns
    -------
    on_hess, off_hess, difference : dict
        See ``hess_diagram`` and ``difference_hess``.
    """
    on, off = np.asarray(regions["on"], dtype=bool), np.asarray(regions["off"], dtype=bool)
    if stars is not None:
        stars = np.asarray(stars, dtype=bool)
        on, off = on & stars, off & stars
    bins = {"color_bins": color_bins, "mag_bins": mag_bins}
    on_hess = hess_diagram(catalog, mag_cols, mask=on, area_deg2=regions["on_area_deg2"], **bins)
    off_hess = hess_diagram(catalog, mag_cols, mask=off, area_deg2=regions["off_area_deg2"], **bins)
    return on_hess, off_hess, difference_hess(on_hess, off_hess)

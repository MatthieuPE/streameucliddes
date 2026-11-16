"""Tune the cuts and match filters on known dwarf galaxies and globular clusters.

A known object at a distance modulus of interest is a real old, metal-poor
population of known distance: the signal of a selection (cuts, colour or
magnitude ranges, a match filter in some bands) is its excess of stars on
the object over the field around it. Steps, in a notebook:

1. ``catalogs.overlapped_objects.objects_near_distance_modulus``: the LVDB
   dwarf galaxies and globular clusters near a distance modulus, flagged
   when they are in the catalog; ``find_object`` picks one by name;
   ``catalogs.isochrone_parameters.isochrone_from_feh`` gives its isochrone.
2. ``object_radec_box``: the (ra, dec) box of its field, to read the stars
   (``catalogs.euclid_catalog.get_euclid_stars(..., ra_range=, dec_range=)``).
3. ``object_regions``: the on region (an ellipse, or an elliptical annulus,
   between ``on_radii`` times the half-light radius of the object), the off
   region (an elliptical annulus between ``off_radii``), the stars excluded
   (outside the footprint, or on another known object) and the areas. For
   a globular cluster, start the on annulus beyond its core, where the
   catalog misses stars (crowding; NGC 1851 in DR1: no star within 4
   half-light radii), and the off annulus beyond its stars (NGC 1851: about
   37 half-light radii, where its excess falls to the background; its mask,
   ``analysis.build_masks.globular_cluster_radius_scales``, stops at 27, where
   it is 3 times the background).
4. Selections: boolean arrays over the rows of the catalog, combined with
   ``&`` and ``|``: ``analysis.catalog_preparation.catalog_match_filter_mask``,
   ``prepare_match_filter_catalog(..., as_mask=True)`` (the cuts alone), or
   any expression on the catalog columns, e.g.
   ``catalog["des_yr6_g_obs"] - catalog["des_yr6_r_obs"] < 0.8``.
5. ``selection_snr`` of one selection, ``compare_selections`` of several:
   SNR = excess on the object / sqrt(background expected there),
   ``analysis.injected_stream_snr.compute_snr``.

Geometry. The regions are those of the masks (``analysis.build_masks``):
ellipses in (separation, position angle) around the centre,
``elliptical_radius_deg``, exact on the sky; the LVDB ``rhalf`` is the
semi-major axis of the half-light ellipse. The maps and areas are in the
frame centred on the object (``stream_frames.frame_centred_on_point``:
phi1 to the east, phi2 to the north); each area is the sum of the exact
solid angles of the small pieces of that frame inside the region
(``catalogs.survey_footprint.phi12_area_pieces``).

The plots (map, CMD, Hess diagrams) are in
``plotting.plot_gc_dwarf_optimization``. The catalog is neither copied nor
modified: the regions and the selections are arrays of one value per star.
The catalog magnitudes are already dereddened.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from astropy.coordinates import angular_separation

from streameucliddes.analysis.build_masks import elliptical_radius_deg, mask_radius_scales
from streameucliddes.analysis.injected_stream_snr import compute_snr
from streameucliddes.analysis.star_density_maps import star_density_phi12
from streameucliddes.catalogs import overlapped_objects
from streameucliddes.catalogs.survey_footprint import (
    footprint_mask_from_stars,
    in_healpix_mask,
    phi12_area_pieces,
    survey_footprint_phi12_mask,
)
from streameucliddes.stream_frames import (
    frame_centred_on_point,
    phi12_bin_centres,
    phi12_to_radec,
    phi12_window_radec_box,
    radec_to_phi12,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from numpy.typing import ArrayLike, NDArray

# Other objects masked around an object (catalogs.overlapped_objects.OBJECT_TYPES)
DEFAULT_MASKED_OBJECT_TYPES = ("dwarfs", "gc", "galaxies")
# Regions, in half-light radii (semi-major axis of the LVDB half-light ellipse). The off annulus starts beyond
# the default mask of the dwarf galaxies in the searches (build_masks.DEFAULT_RADIUS_SCALE, 4)
DEFAULT_ON_RADII = (0.0, 2.0)
DEFAULT_OFF_RADII = (5.0, 10.0)
# Box read around an object: its field plus this margin (deg), room for the footprint edge pixels trimmed
BOX_MARGIN_DEG = 0.3
# Footprint from the stars of the field (footprint_mask_from_stars, as CandidateInvestigation.prepare)
DEFAULT_FOOTPRINT_NSIDE = 1024
# Areas: pieces of the field so that so many fit across the on region (its outer minus inner radius), at most
# so many along an axis of the field
AREA_PIECES_ACROSS_ON = 20
AREA_MAX_PIECES = 1000
# Another entry is the object itself (not masked around it) when its centre is inside the half-light ellipse of
# the object and its reference radius at least this fraction of the half-light radius: the same cluster in two LVDB
# tables, or a dwarf galaxy also in the Siena Galaxy Atlas, its centre 10 to 30 arcsec away (in DR1: Phoenix, DDO
# 165, ESO 115-021, ESO 245-005, KDG 063, NGC 2976), whose galaxy mask would cover the whole field. A small
# background galaxy in a large dwarf (e.g. 5 arcmin from Fornax, half-light radius 20 arcmin) stays masked
SAME_OBJECT_MIN_RADIUS_FRACTION = 0.5
# Maps of an object: bins across the field
DEFAULT_MAP_BINS = 60


# ===================================================
# Field and regions around an object
# ===================================================


def object_field_radius_deg(obj: Mapping[str, Any], off_radii: tuple[float, float] = DEFAULT_OFF_RADII) -> float:
    """Radius of the field of an object (deg): the semi-major axis of the outer edge of the off annulus."""
    return float(off_radii[1] * obj["radius_arcmin"] / 60.0)


def object_field_window(radius_deg: float) -> tuple[float, float, float, float]:
    """phi1/phi2 window of a field of radius ``radius_deg`` in the frame centred on its object.

    Every point within ``radius_deg`` of the centre is inside: its phi1 and
    phi2 are at most its separation.
    """
    return (-radius_deg, radius_deg, -radius_deg, radius_deg)


def object_radec_box(
    obj: Mapping[str, Any],
    radius_deg: float | None = None,
    *,
    margin_deg: float = BOX_MARGIN_DEG,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """(ra, dec) box around an object, to read the stars of its field.

    ``stream_frames.phi12_window_radec_box`` of the window of the field
    (``object_field_window``) in the frame centred on the object: across
    ra = 0 (``ra_min > ra_max``) and around a pole as the other windows.

    Parameters
    ----------
    obj : mapping
        A row of ``catalogs.overlapped_objects.get_objects`` (``find_object``).
    radius_deg : float, optional
        Radius around the centre (deg). Default: ``object_field_radius_deg``.
    margin_deg : float, optional
        Margin added on each side of the box (deg).

    Returns
    -------
    ra_range, dec_range : tuple of float
        For ``catalogs.euclid_catalog.get_euclid_stars``.
    """
    radius_deg = object_field_radius_deg(obj) if radius_deg is None else radius_deg
    frame = frame_centred_on_point(float(obj["ra"]), float(obj["dec"]))
    return phi12_window_radec_box(frame, object_field_window(radius_deg), margin_deg=margin_deg)


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
    the ellipticity and position angle of the object
    (``build_masks.elliptical_radius_deg``, a circle when unknown). The on
    region is ``on_radii[0] <= r <= on_radii[1]``, the off region
    ``off_radii[0] <= r <= off_radii[1]``. Stars are excluded outside the
    footprint, or on another known object, masked as in the searches
    (``build_masks.mask_radius_scales`` times its reference radius); the
    areas leave out the same sky. The object itself is never masked: its own
    key, and the other entries of the same object (in another LVDB table, in
    the Siena Galaxy Atlas: same name, or centre inside its half-light
    ellipse and size at least ``SAME_OBJECT_MIN_RADIUS_FRACTION`` of its
    half-light radius). A
    warning names the other objects whose mask covers its centre (e.g. the
    host galaxy of a satellite): its on region is then excluded. Each area is the sum of the pieces
    (``survey_footprint.phi12_area_pieces``, ``AREA_PIECES_ACROSS_ON``
    across the on region) of the field whose centre is in the region and
    not excluded.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars of the field, e.g. read in ``object_radec_box``. Not copied.
    obj : mapping
        A row of ``catalogs.overlapped_objects.get_objects`` (``find_object``),
        with a half-light radius.
    on_radii : tuple of float, optional
        Inner and outer radii of the on region (half-light radii): inner 0
        for an ellipse.
    off_radii : tuple of float, optional
        Inner and outer radii of the off annulus (half-light radii).
    masked_object_types : sequence of str, optional
        Types of the other objects masked (``catalogs.overlapped_objects.OBJECT_TYPES``).
    footprint_mask : numpy.ndarray of bool, optional
        HEALPix footprint, True outside. Default: from the stars of
        ``catalog`` (``survey_footprint.footprint_mask_from_stars`` at
        ``DEFAULT_FOOTPRINT_NSIDE``).
    footprint_nest : bool, optional
        Ordering of a given ``footprint_mask``.
    ra_col, dec_col : str, optional
        Coordinate columns (deg).

    Returns
    -------
    regions : dict
        Per star (arrays of ``len(catalog)``): ``elliptical_radius``
        (half-light radii), ``phi1`` and ``phi2`` (deg, in ``stream_frame``),
        ``excluded``, ``on`` and ``off`` (bool, not excluded). And
        ``on_area_deg2``, ``off_area_deg2``, ``on_area_fraction`` and
        ``off_area_fraction`` (of the full regions, not excluded),
        ``object``, ``stream_frame`` (``frame_centred_on_point``),
        ``window`` (``object_field_window``), ``half_light_radius_deg``,
        ``on_radii``, ``off_radii``, ``masked_objects`` (the other objects
        masked in the field, with their ``mask_radius_arcmin``),
        ``same_object_entries`` (the other entries of the object, not
        masked), ``footprint_mask``, ``footprint_nest``.

    Raises
    ------
    ValueError
        The object has no half-light radius; radii not increasing.
    """
    half_light_deg = float(obj["radius_arcmin"]) / 60.0
    if not np.isfinite(half_light_deg) or half_light_deg <= 0:
        raise ValueError(f"{obj['name']} has no half-light radius.")
    if not 0 <= on_radii[0] < on_radii[1] <= off_radii[0] < off_radii[1]:
        raise ValueError(f"Need 0 <= on_radii[0] < on_radii[1] <= off_radii[0] < off_radii[1], not {on_radii}, "
                         f"{off_radii}.")
    ra, dec = catalog[ra_col].to_numpy(dtype=float), catalog[dec_col].to_numpy(dtype=float)
    if footprint_mask is None:
        footprint_mask, footprint_nest = footprint_mask_from_stars(ra, dec, DEFAULT_FOOTPRINT_NSIDE), False
    footprint_mask = np.asarray(footprint_mask, dtype=bool)
    field_radius_deg = off_radii[1] * half_light_deg
    frame = frame_centred_on_point(float(obj["ra"]), float(obj["dec"]))
    window = object_field_window(field_radius_deg)
    masked_objects, same_object_entries = _masked_objects_in_field(obj, field_radius_deg, masked_object_types)
    covering = masked_objects[_centre_in_masks(obj, masked_objects)]
    if not covering.empty:
        warnings.warn(f"The masks of {covering['name'].tolist()} cover the centre of {obj['name']}: its on region "
                      "is excluded. Leave their type out of masked_object_types to keep it.", stacklevel=2)

    def radius_and_excluded(points_ra, points_dec):
        radius = _elliptical_radius(points_ra, points_dec, obj) / half_light_deg
        excluded = ~np.isfinite(points_ra) | ~np.isfinite(points_dec)
        excluded |= in_healpix_mask(points_ra, points_dec, footprint_mask, nest=footprint_nest)
        for _, other in masked_objects.iterrows():
            excluded |= _elliptical_radius(points_ra, points_dec, other) <= other["mask_radius_arcmin"] / 60.0
        return radius, excluded

    radius, excluded = radius_and_excluded(ra, dec)
    phi1, phi2 = radec_to_phi12(ra, dec, frame)

    step = max((on_radii[1] - on_radii[0]) * half_light_deg / AREA_PIECES_ACROSS_ON,
               2 * field_radius_deg / AREA_MAX_PIECES)
    piece_phi1, piece_phi2, piece_area, _ = phi12_area_pieces(window[:2], window[2:], step_deg=step)
    piece_radius, piece_excluded = radius_and_excluded(*phi12_to_radec(piece_phi1, piece_phi2, frame))
    areas = {}
    for name, (inner, outer) in (("on", on_radii), ("off", off_radii)):
        inside = (piece_radius >= inner) & (piece_radius <= outer)
        areas[name] = (float(piece_area[inside & ~piece_excluded].sum()), float(piece_area[inside].sum()))

    return {
        "object": obj, "stream_frame": frame, "window": window, "half_light_radius_deg": half_light_deg,
        "on_radii": (float(on_radii[0]), float(on_radii[1])), "off_radii": (float(off_radii[0]), float(off_radii[1])),
        "elliptical_radius": radius, "phi1": phi1, "phi2": phi2, "excluded": excluded,
        "on": ~excluded & (radius >= on_radii[0]) & (radius <= on_radii[1]),
        "off": ~excluded & (radius >= off_radii[0]) & (radius <= off_radii[1]),
        "on_area_deg2": areas["on"][0], "off_area_deg2": areas["off"][0],
        "on_area_fraction": areas["on"][0] / areas["on"][1], "off_area_fraction": areas["off"][0] / areas["off"][1],
        "masked_objects": masked_objects, "same_object_entries": same_object_entries,
        "footprint_mask": footprint_mask, "footprint_nest": footprint_nest,
    }


def _elliptical_radius(ra: ArrayLike, dec: ArrayLike, obj: Mapping[str, Any]) -> NDArray[np.float64]:
    """Elliptical radius (deg) of points around an object (``elliptical_radius_deg``); a circle when unknown."""
    ellipticity, major_axis_angle = (float(obj[key]) for key in ("ellipticity", "position_angle"))
    return elliptical_radius_deg(ra, dec, float(obj["ra"]), float(obj["dec"]),
                                 ellipticity if np.isfinite(ellipticity) else 0.0,
                                 major_axis_angle if np.isfinite(major_axis_angle) else 0.0)


def _centre_in_masks(obj: Mapping[str, Any], masked_objects: pd.DataFrame) -> NDArray[np.bool_]:
    """Masked objects whose mask covers the centre of ``obj``."""
    covers = [_elliptical_radius(float(obj["ra"]), float(obj["dec"]), other) <= other["mask_radius_arcmin"] / 60.0
              for _, other in masked_objects.iterrows()]
    return np.array(covers, dtype=bool).reshape(len(masked_objects))


def _masked_objects_in_field(
    obj: Mapping[str, Any], field_radius_deg: float, object_types: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The other objects whose mask reaches the field, with ``mask_radius_arcmin``; the other entries of ``obj``.

    Galaxies are circles, as in ``build_masks.galaxies_healpix_mask``. An
    entry is ``obj`` itself when it has its key or its name (``simplified_name``,
    e.g. "ESO 115-021" and "ESO115-021", whose centres are 0.5 arcmin apart
    across its minor axis), or when its centre is inside the half-light
    ellipse of ``obj`` and its reference radius is at least
    ``SAME_OBJECT_MIN_RADIUS_FRACTION`` of the half-light radius of ``obj``.
    """
    objects = overlapped_objects.get_objects(object_types)
    objects.loc[objects["type"] == "galaxies", "ellipticity"] = np.nan
    scales = np.full(len(objects), np.nan)
    for object_type, rows in objects.groupby("type").indices.items():
        scales[rows] = mask_radius_scales(objects.iloc[rows], object_type)
    objects["mask_radius_arcmin"] = scales * objects["radius_arcmin"].to_numpy(dtype=float)
    separation_deg = np.degrees(angular_separation(
        np.radians(float(obj["ra"])), np.radians(float(obj["dec"])),
        np.radians(objects["ra"].to_numpy(dtype=float)), np.radians(objects["dec"].to_numpy(dtype=float)),
    ))
    reaches = separation_deg <= field_radius_deg + objects["mask_radius_arcmin"].to_numpy() / 60.0
    objects = objects[reaches].reset_index(drop=True)
    half_light_arcmin = float(obj["radius_arcmin"])
    centre_inside = (_elliptical_radius(objects["ra"].to_numpy(dtype=float), objects["dec"].to_numpy(dtype=float), obj)
                     * 60.0 <= half_light_arcmin)
    as_large = objects["radius_arcmin"].to_numpy(dtype=float) >= SAME_OBJECT_MIN_RADIUS_FRACTION * half_light_arcmin
    same_name = objects["name"].map(overlapped_objects.simplified_name).to_numpy() == overlapped_objects.simplified_name(
        obj["name"])
    itself = (objects["key"].to_numpy() == obj["key"]) | same_name | (centre_inside & as_large)
    keep = ~itself & np.isfinite(objects["mask_radius_arcmin"].to_numpy())
    same_entries = objects[itself & (objects["key"].to_numpy() != obj["key"])].reset_index(drop=True)
    return objects[keep].reset_index(drop=True), same_entries


def object_density_map(
    regions: Mapping[str, Any],
    selected: ArrayLike | None = None,
    *,
    bin_size_deg: float | None = None,
    smoothing_deg: float = 0.0,
) -> dict[str, Any]:
    """Map of the selected stars around an object, in its frame.

    ``star_density_maps.star_density_phi12`` of the selected stars not
    excluded, on the grid of the window of the field; the bins outside the
    footprint (``survey_footprint_phi12_mask``) or with their centre on a
    masked object are masked (NaN).

    Parameters
    ----------
    regions : mapping
        From ``object_regions``.
    selected : array_like of bool, optional
        Selected stars, one per row of the catalog. Default: all.
    bin_size_deg : float, optional
        Default: ``DEFAULT_MAP_BINS`` bins across the field.
    smoothing_deg : float, optional
        Gaussian smoothing (deg); 0: none.

    Returns
    -------
    density_map : dict
        ``{"density", "phi1_centres", "phi2_centres", "bin_size_deg",
        "window", "smoothing_deg", "n_selected"}``, the keys of the maps of
        ``analysis.stream_frame_search`` drawn by
        ``plotting.plot_stream_frame_search.plot_stream_frame_search_single_dm``.
    """
    window, frame = regions["window"], regions["stream_frame"]
    bin_size_deg = (window[1] - window[0]) / DEFAULT_MAP_BINS if bin_size_deg is None else bin_size_deg
    stars = ~np.asarray(regions["excluded"], dtype=bool)
    if selected is not None:
        stars &= np.asarray(selected, dtype=bool)
    grid_mask = survey_footprint_phi12_mask(regions["footprint_mask"], frame, window, bin_size_deg,
                                            nest=regions["footprint_nest"])
    phi1_centres, phi2_centres = phi12_bin_centres(window, bin_size_deg)
    centres_ra, centres_dec = phi12_to_radec(*np.meshgrid(phi1_centres, phi2_centres, indexing="ij"), frame)
    for _, other in regions["masked_objects"].iterrows():
        grid_mask |= _elliptical_radius(centres_ra, centres_dec, other) <= other["mask_radius_arcmin"] / 60.0
    density = star_density_phi12(regions["phi1"][stars], regions["phi2"][stars], window, bin_size_deg,
                                 smoothing_deg=smoothing_deg, footprint_mask=grid_mask)
    return {"density": density, "phi1_centres": phi1_centres, "phi2_centres": phi2_centres,
            "bin_size_deg": bin_size_deg, "window": window, "smoothing_deg": smoothing_deg,
            "n_selected": int(stars.sum())}


# ===================================================
# SNR of selections
# ===================================================


def selection_snr(regions: Mapping[str, Any], selected: ArrayLike | None = None) -> dict[str, float]:
    """Excess of the selected stars on the object over the off region, and its SNR.

    The background expected on the object is the off count scaled by the
    ratio of the areas; SNR = excess / sqrt(background on the object)
    (``injected_stream_snr.compute_snr``).

    Parameters
    ----------
    regions : mapping
        From ``object_regions``: ``on`` and ``off`` (one value per star) and
        their areas ``on_area_deg2`` and ``off_area_deg2``.
    selected : array_like of bool, optional
        Selected stars, one per row of the catalog. Default: all.

    Returns
    -------
    snr : dict
        ``n_on``, ``n_off`` (selected stars in each region),
        ``n_background_on`` (``n_off * on_area / off_area``), ``n_signal``
        (``n_on - n_background_on``), ``n_signal_error`` (Poisson, on and
        off), ``snr``, ``background_density`` (stars/deg²),
        ``on_area_deg2``, ``off_area_deg2``.
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
    return {
        "n_on": n_on, "n_off": n_off, "n_background_on": float(n_background), "n_signal": float(n_signal),
        "n_signal_error": float(np.sqrt(n_on + area_ratio**2 * n_off)),
        "snr": float(compute_snr(n_signal, n_background)),
        "background_density": n_off / off_area, "on_area_deg2": on_area, "off_area_deg2": off_area,
    }


def compare_selections(regions: Mapping[str, Any], selections: Mapping[str, ArrayLike]) -> pd.DataFrame:
    """``selection_snr`` of several selections, one row each.

    Parameters
    ----------
    regions : mapping
        From ``object_regions``.
    selections : mapping
        ``{name: selected stars}``, e.g. ``{"VIS-H": catalog_match_filter_mask(...),
        "VIS-H, VIS < 24": ... & (catalog["euclid_dr1_VIS_obs"] < 24)}``.

    Returns
    -------
    table : pandas.DataFrame
        Index: the names, in order; columns: those of ``selection_snr``.
    """
    rows = {name: selection_snr(regions, selected) for name, selected in selections.items()}
    return pd.DataFrame.from_dict(rows, orient="index")

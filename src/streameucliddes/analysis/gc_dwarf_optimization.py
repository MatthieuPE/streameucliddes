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
5. ``analysis.on_off_regions.selection_snr`` of one selection,
   ``compare_selections`` of several: SNR = excess on the object /
   sqrt(background expected there); ``region_density_map`` and
   ``region_on_off_hess``, and the plots of ``plotting.plot_on_off_regions``.

Geometry. The regions are those of ``analysis.on_off_regions.on_off_regions``,
intervals of the elliptical radius around the centre
(``on_off_regions.object_elliptical_radius_deg``, exact on the sky; the LVDB
``rhalf`` is the semi-major axis of the half-light ellipse), in the frame
centred on the object (``stream_frames.frame_centred_on_point``: phi1 to the
east, phi2 to the north). The other objects are masked by their exact
ellipses (a neighbour can be smaller than a HEALPix pixel), the footprint by
its HEALPix mask. The catalog is neither copied nor modified. The catalog
magnitudes are already dereddened.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from streameucliddes.analysis.build_masks import ellipse_outline_radec, excluded_healpix_mask
from streameucliddes.analysis.on_off_regions import (
    DEFAULT_MASKED_OBJECT_TYPES,
    masked_objects_in_window,
    object_elliptical_radius_deg,
    on_off_regions,
)
from streameucliddes.catalogs import overlapped_objects
from streameucliddes.catalogs.survey_footprint import footprint_mask_from_stars
from streameucliddes.stream_frames import frame_centred_on_point, phi12_window_radec_box, radec_to_phi12

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import pandas as pd
    from numpy.typing import NDArray

# Regions, in half-light radii (semi-major axis of the LVDB half-light ellipse). The off annulus starts beyond
# the default mask of the dwarf galaxies in the searches (build_masks.DEFAULT_RADIUS_SCALE, 4)
DEFAULT_ON_RADII = (0.0, 2.0)
DEFAULT_OFF_RADII = (5.0, 10.0)
# Box read around an object: its field plus this margin (deg), room for the footprint edge pixels trimmed
BOX_MARGIN_DEG = 0.3
# Footprint from the stars of the field (footprint_mask_from_stars, as CandidateInvestigation.prepare)
DEFAULT_FOOTPRINT_NSIDE = 1024
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
    area_step_deg: float | None = None,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> dict[str, Any]:
    """On and off regions of an object: the stars in each and their areas (``on_off_regions``).

    Radii are elliptical, in half-light radii: a star at radius ``r`` is on
    the ellipse of semi-major axis ``r`` times the half-light radius, with
    the ellipticity and position angle of the object
    (``object_elliptical_radius_deg``, a circle when unknown). The on
    region is ``on_radii[0] <= r <= on_radii[1]``, the off region (one off
    part, ``off``) ``off_radii[0] <= r <= off_radii[1]``. Stars are
    excluded outside the footprint, or on another known object, masked by
    its exact ellipse as in the searches (``masked_objects_in_window``); the
    areas leave out the same sky. The object itself is never masked: its own
    key, and the other entries of the same object (in another LVDB table, in
    the Siena Galaxy Atlas: same name, or centre inside its half-light
    ellipse and size at least ``SAME_OBJECT_MIN_RADIUS_FRACTION`` of its
    half-light radius). A warning names the other objects whose mask covers
    its centre (e.g. the host galaxy of a satellite): its on region is then
    excluded.

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
    area_step_deg : float, optional
        Size of the area pieces (deg), see ``on_off_regions``.
    ra_col, dec_col : str, optional
        Coordinate columns (deg).

    Returns
    -------
    regions : dict
        Those of ``on_off_regions``, plus ``elliptical_radius`` (per star,
        half-light radii), ``object``, ``half_light_radius_deg``,
        ``on_radii``, ``off_radii``, ``same_object_entries`` (the other
        entries of the object, not masked), ``outlines`` (the ellipses of the
        regions in the frame), ``map_bin_size_deg``, ``footprint_mask``,
        ``footprint_nest``; ``masked_objects`` has their
        ``mask_radius_arcmin``.

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
    if footprint_mask is None:
        footprint_mask, footprint_nest = footprint_mask_from_stars(
            catalog[ra_col].to_numpy(dtype=float), catalog[dec_col].to_numpy(dtype=float),
            DEFAULT_FOOTPRINT_NSIDE), False
    footprint_mask = np.asarray(footprint_mask, dtype=bool)
    frame = frame_centred_on_point(float(obj["ra"]), float(obj["dec"]))
    window = object_field_window(off_radii[1] * half_light_deg)
    masked_objects, same_object_entries = _split_same_object(
        obj, masked_objects_in_window(frame, window, masked_object_types))
    covering = [name for name, (_, other) in zip(masked_objects["name"], masked_objects.iterrows())
                if object_elliptical_radius_deg(float(obj["ra"]), float(obj["dec"]), other)
                <= other["mask_radius_arcmin"] / 60.0]
    if covering:
        warnings.warn(f"The masks of {covering} cover the centre of {obj['name']}: its on region "
                      "is excluded. Leave their type out of masked_object_types to keep it.", stacklevel=2)

    def elliptical_radius(ra, dec, phi1, phi2):
        return object_elliptical_radius_deg(ra, dec, obj)

    regions = on_off_regions(
        catalog, frame, window, elliptical_radius,
        {"on": tuple(radius * half_light_deg for radius in on_radii),
         "off": tuple(radius * half_light_deg for radius in off_radii)},
        excluded_healpix_mask=excluded_healpix_mask(footprint_mask, None, nest=footprint_nest)[0],
        masked_objects=masked_objects, area_step_deg=area_step_deg, label=str(obj["name"]),
        ra_col=ra_col, dec_col=dec_col,
    )
    centre, shape = (float(obj["ra"]), float(obj["dec"])), (float(obj["ellipticity"]), float(obj["position_angle"]))
    outlines = {name: [radec_to_phi12(*ellipse_outline_radec(*centre, radius * half_light_deg, *shape), frame)
                       for radius in radii if radius > 0]
                for name, radii in (("on", on_radii), ("off", off_radii))}
    regions.update({
        "elliptical_radius": regions["region_coordinate"] / half_light_deg, "object": obj,
        "half_light_radius_deg": half_light_deg, "on_radii": (float(on_radii[0]), float(on_radii[1])),
        "off_radii": (float(off_radii[0]), float(off_radii[1])), "same_object_entries": same_object_entries,
        "outlines": outlines, "map_bin_size_deg": (window[1] - window[0]) / DEFAULT_MAP_BINS,
        "footprint_mask": footprint_mask, "footprint_nest": footprint_nest,
    })
    return regions


def _split_same_object(obj: Mapping[str, Any], objects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The objects to mask around ``obj``, and the other entries of ``obj`` among ``objects``.

    An entry is ``obj`` itself when it has its key or its name (``simplified_name``,
    e.g. "ESO 115-021" and "ESO115-021", whose centres are 0.5 arcmin apart
    across its minor axis), or when its centre is inside the half-light
    ellipse of ``obj`` and its reference radius is at least
    ``SAME_OBJECT_MIN_RADIUS_FRACTION`` of the half-light radius of ``obj``.
    """
    half_light_arcmin = float(obj["radius_arcmin"])
    centre_inside = (object_elliptical_radius_deg(objects["ra"].to_numpy(dtype=float),
                                                  objects["dec"].to_numpy(dtype=float), obj) * 60.0
                     <= half_light_arcmin)
    as_large = objects["radius_arcmin"].to_numpy(dtype=float) >= SAME_OBJECT_MIN_RADIUS_FRACTION * half_light_arcmin
    same_name = objects["name"].map(overlapped_objects.simplified_name).to_numpy() == overlapped_objects.simplified_name(
        obj["name"])
    itself = (objects["key"].to_numpy() == obj["key"]) | same_name | (centre_inside & as_large)
    same_entries = objects[itself & (objects["key"].to_numpy() != obj["key"])].reset_index(drop=True)
    return objects[~itself].reset_index(drop=True), same_entries

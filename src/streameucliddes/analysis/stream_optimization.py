"""Tune the cuts and match filters on a known stream, as ``analysis.gc_dwarf_optimization`` on known objects.

A known stream crossing the footprint is a real old, metal-poor population
of known track and distance: the signal of a selection is its excess of
stars along the track over the field on both sides. Steps, in a notebook:

1. ``catalogs.galstreams_catalog``: ``load_galstreams`` and
   ``select_galstreams_in_footprint`` (the tracks crossing the footprint);
   one track, e.g. ``mws["AAU-AliqaUma-L21"]``, its width
   ``galstreams_width_phi2``.
2. ``stream_field``: the phi1 range of the track in the footprint, the
   window of its regions and the (ra, dec) box to read the stars
   (``catalogs.euclid_catalog.get_euclid_stars(..., ra_range=, dec_range=)``).
3. ``stream_regions``: the on band (within ``on_n_sigma`` times the
   width of the track), the off bands (from ``off_n_sigma[0]`` to
   ``off_n_sigma[1]`` times the width, below and above), the stars excluded
   (outside the footprint, or on a known object) and the areas.
4. Selections: boolean arrays over the rows of the catalog, as for the
   objects; ``track_match_filter_mask`` is the match filter at the distance
   of the track along it.
5. ``analysis.on_off_regions.selection_snr`` and ``compare_selections`` (the
   background: the mean of the densities of the two off bands, so that a
   gradient across the track cancels), ``region_density_map``,
   ``region_on_off_hess``, and the plots of ``plotting.plot_on_off_regions``.

Geometry. The regions are the bands of ``analysis.on_off_regions.track_band_regions``
along the track, in its ``stream_frame``: a star belongs to a band by its
phi2 offset from the track at its phi1 (``galstreams_phi2_along_track``,
plus ``phi2_offset_deg``); its distance modulus is that of the track at its
phi1 (``galstreams_dm_along_track``, rounded to ``dm_step``). The known
objects are masked as in the searches, by a HEALPix mask
(``analysis.build_masks.excluded_healpix_mask``), or by their exact ellipses
(``object_masks="ellipses"``). The catalog is neither copied nor modified.
The catalog magnitudes are already dereddened.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from streameucliddes.analysis.build_masks import excluded_healpix_mask, overlapped_objects_healpix_mask
from streameucliddes.analysis.catalog_preparation import catalog_match_filter_mask
from streameucliddes.analysis.fit_stream import DEFAULT_SIGMA_GUESS_DEG
from streameucliddes.analysis.on_off_regions import (
    DEFAULT_MASKED_OBJECT_TYPES,
    DEFAULT_OFF_N_SIGMA,
    DEFAULT_ON_N_SIGMA,
    masked_objects_in_window,
    track_band_regions,
    track_band_window,
)
from streameucliddes.catalogs.galstreams_catalog import (
    galstreams_dm_along_track,
    galstreams_phi2_along_track,
    galstreams_phi12_window,
)
from streameucliddes.catalogs.survey_footprint import footprint_mask_from_stars
from streameucliddes.stream_frames import phi12_window_radec_box, radec_to_phi12

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import galstreams
    import pandas as pd
    from numpy.typing import NDArray

# Distance moduli along the track rounded to this step: the stars sharing one go through one match filter
DEFAULT_DM_STEP = 0.05
# Footprint from the stars (as object_regions) and resolution of the object masks (pixels of 0.03 deg, as the
# searches with 0.1 deg bins: build_masks.object_mask_nside(0.1))
DEFAULT_FOOTPRINT_NSIDE = 1024
DEFAULT_OBJECT_MASK_NSIDE = 2048
# Masks of the known objects: "healpix" (as the searches) or "ellipses" (exact)
OBJECT_MASKS = ("healpix", "ellipses")
# Box read around the window of the regions (deg)
BOX_MARGIN_DEG = 0.3
# Maps of a stream: bin size (deg)
DEFAULT_MAP_BIN_DEG = 0.05


# ===================================================
# Field and regions along a track
# ===================================================


def _track_phi2_function(track: galstreams.Track6D, phi2_offset_deg: float):
    """phi2 of the track plus an offset, as a function of phi1 (``galstreams_phi2_along_track``)."""
    def track_phi2(phi1):
        return galstreams_phi2_along_track(track, phi1) + phi2_offset_deg
    return track_phi2


def stream_field(
    track: galstreams.Track6D,
    *,
    sigma_deg: float = DEFAULT_SIGMA_GUESS_DEG,
    off_n_sigma: tuple[float, float] = DEFAULT_OFF_N_SIGMA,
    footprint_mask: NDArray[np.bool_] | None = None,
    nest: bool | None = None,
    phi1_range: tuple[float, float] | None = None,
    phi2_offset_deg: float = 0.0,
) -> dict[str, Any]:
    """phi1 range, window and (ra, dec) box of the regions of a stream, before reading its stars.

    Parameters
    ----------
    track : galstreams.Track6D
        One track of ``catalogs.galstreams_catalog.load_galstreams()``.
    sigma_deg : float, optional
        Width of the stream in phi2 (deg), e.g. ``galstreams_width_phi2``.
    off_n_sigma : tuple of float, optional
        Inner and outer distances of the off regions to the track, in units of ``sigma_deg``.
    footprint_mask : numpy.ndarray of bool, optional
        HEALPix footprint, True outside (needs ``nest``), e.g. ``catalog_point_map(catalog) == 0``:
        the phi1 range is then that of the track in the footprint.
    nest : bool, optional
        Ordering of ``footprint_mask``.
    phi1_range : tuple of float, optional
        phi1 range of the regions (deg). Default: the track, in the footprint with ``footprint_mask``.
    phi2_offset_deg : float, optional
        Shift of the regions from the track in phi2 (deg), as in ``stream_regions``.

    Returns
    -------
    field : dict
        ``{"phi1_range", "window", "ra_range", "dec_range"}``: the window
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` covers the off regions
        along the track plus ``on_off_regions.DEFAULT_WINDOW_MARGIN_DEG`` (``track_band_window``);
        the box (``stream_frames.phi12_window_radec_box``) is for ``get_euclid_stars``.
    """
    if phi1_range is None:
        phi1_range = galstreams_phi12_window(track, footprint_mask=footprint_mask, nest=nest, padding_phi1_deg=0.0,
                                             padding_phi2_deg=0.0, min_phi2_half_width_deg=0.0)[:2]
    phi1_range = (float(phi1_range[0]), float(phi1_range[1]))
    window = track_band_window(_track_phi2_function(track, phi2_offset_deg), phi1_range, off_n_sigma[1] * sigma_deg)
    ra_range, dec_range = phi12_window_radec_box(track.stream_frame, window, margin_deg=BOX_MARGIN_DEG)
    return {"phi1_range": phi1_range, "window": window, "ra_range": ra_range, "dec_range": dec_range}


def stream_object_healpix_mask(
    masked_object_types: Sequence[str] = DEFAULT_MASKED_OBJECT_TYPES,
    nside: int = DEFAULT_OBJECT_MASK_NSIDE,
) -> NDArray[np.bool_] | None:
    """HEALPix mask (RING, True on them) of the known objects masked around a stream, as in the searches.

    ``analysis.build_masks.overlapped_objects_healpix_mask`` of the types
    among ``"lmc"``, ``"smc"``, ``"dwarfs"``, ``"gc"``, ``"galaxies"``. Once
    for several catalogs of the same stream (``stream_regions(object_healpix_mask=)``).

    Returns
    -------
    mask : numpy.ndarray of bool or None
        None when no type is given.
    """
    if not masked_object_types:
        return None
    return overlapped_objects_healpix_mask(nside, nest=False, **{f"mask_{object_type}": True
                                                                for object_type in masked_object_types})


def stream_regions(
    catalog: pd.DataFrame,
    track: galstreams.Track6D,
    *,
    phi1_range: tuple[float, float],
    sigma_deg: float = DEFAULT_SIGMA_GUESS_DEG,
    on_n_sigma: float = DEFAULT_ON_N_SIGMA,
    off_n_sigma: tuple[float, float] = DEFAULT_OFF_N_SIGMA,
    dm_step: float = DEFAULT_DM_STEP,
    object_masks: str = "healpix",
    masked_object_types: Sequence[str] = DEFAULT_MASKED_OBJECT_TYPES,
    object_healpix_mask: NDArray[np.bool_] | None = None,
    footprint_mask: NDArray[np.bool_] | None = None,
    footprint_nest: bool = False,
    phi2_offset_deg: float = 0.0,
    area_step_deg: float | None = None,
    label: str | None = None,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> dict[str, Any]:
    """On and off bands along a stream track: the stars in each and their areas (``track_band_regions``).

    A star is on the stream when its phi2 offset from the track at its phi1
    is at most ``on_n_sigma * sigma_deg``, off when it is between
    ``off_n_sigma[0] * sigma_deg`` and ``off_n_sigma[1] * sigma_deg``, below
    (``off_below``) or above (``off_above``). Stars outside the footprint,
    or on a masked known object, are excluded; the areas leave out the same
    sky.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars of the field, e.g. read in the box of ``stream_field``. Not copied.
    track : galstreams.Track6D
        Track of the stream; the regions are in its ``stream_frame``.
    phi1_range : tuple of float
        phi1 range of the regions (deg), e.g. ``stream_field(...)["phi1_range"]``.
    sigma_deg : float, optional
        Width of the stream in phi2 (deg), e.g. ``galstreams_width_phi2``.
    on_n_sigma : float, optional
        Half-width of the on band, in units of ``sigma_deg``.
    off_n_sigma : tuple of float, optional
        Inner and outer distances of the off bands to the track, in units of ``sigma_deg``.
    dm_step : float, optional
        Rounding step of the distance moduli of the stars.
    object_masks : {"healpix", "ellipses"}, optional
        Known objects masked by a HEALPix mask, as in the searches
        (``stream_object_healpix_mask``), or by their exact ellipses
        (``on_off_regions.masked_objects_in_window``).
    masked_object_types : sequence of str, optional
        Types of the known objects masked; ignored with ``object_healpix_mask``.
    object_healpix_mask : numpy.ndarray of bool, optional
        Masked objects, RING, from ``stream_object_healpix_mask``: computed once for several catalogs
        (``object_masks="healpix"``).
    footprint_mask : numpy.ndarray of bool, optional
        HEALPix footprint, True outside. Default: from the stars of ``catalog``
        (``survey_footprint.footprint_mask_from_stars`` at ``DEFAULT_FOOTPRINT_NSIDE``).
    footprint_nest : bool, optional
        Ordering of a given ``footprint_mask``.
    phi2_offset_deg : float, optional
        Shift of the regions from the galstreams track in phi2 (deg), set by hand when
        the stream is not on its galstreams track in the catalog (e.g. Aliqa Uma in DR1: about +0.6 deg from
        the galstreams track in the frame of ``galstreams_track_in_own_frame``): the centre of the regions is the
        track phi2 plus this offset.
    area_step_deg : float, optional
        Size of the area pieces (deg), see ``on_off_regions.on_off_regions``.
    label : str, optional
        Title of the figures. Default: the track name and its distance moduli.
    ra_col, dec_col : str, optional
        Coordinate columns (deg).

    Returns
    -------
    regions : dict
        Those of ``track_band_regions`` (``track_phi2`` with the offset: the
        centre of the bands), plus ``star_dm`` (per star: distance modulus of
        the track at its phi1, rounded; NaN outside ``phi1_range``),
        ``track_dm`` (the same on ``track_phi1``), ``track``, ``track_name``,
        ``track_points_phi12`` (the galstreams track in the frame),
        ``phi2_offset_deg``, ``object_masks``, ``map_bin_size_deg``,
        ``footprint_mask``, ``footprint_nest``.

    Raises
    ------
    ValueError
        ``sigma_deg`` not positive, the off bands overlap the on band,
        ``phi1_range`` not increasing, the track has no distance, or
        ``object_masks`` unknown.
    """
    if not on_n_sigma <= off_n_sigma[0] < off_n_sigma[1]:
        raise ValueError(f"Need on_n_sigma <= off_n_sigma[0] < off_n_sigma[1], not {on_n_sigma}, {off_n_sigma}.")
    if object_masks not in OBJECT_MASKS:
        raise ValueError(f"object_masks must be one of {OBJECT_MASKS}, not {object_masks!r}.")
    if not phi1_range[1] > phi1_range[0]:
        raise ValueError(f"phi1_range must be increasing, not {phi1_range}.")
    if np.isnan(galstreams_dm_along_track(track, np.asarray(phi1_range, dtype=float), extrapolate=True)).any():
        raise ValueError(f"Track {track.track_name} has no distance in galstreams.")
    frame = track.stream_frame
    if footprint_mask is None:
        footprint_mask, footprint_nest = footprint_mask_from_stars(
            catalog[ra_col].to_numpy(dtype=float), catalog[dec_col].to_numpy(dtype=float),
            DEFAULT_FOOTPRINT_NSIDE), False
    track_phi2 = _track_phi2_function(track, phi2_offset_deg)
    masked_objects = None
    if object_masks == "healpix":
        if object_healpix_mask is None:
            object_healpix_mask = stream_object_healpix_mask(masked_object_types)
    else:
        object_healpix_mask = None
        window = track_band_window(track_phi2, phi1_range, off_n_sigma[1] * sigma_deg)
        masked_objects = masked_objects_in_window(frame, window, masked_object_types)
    excluded_mask, _ = excluded_healpix_mask(footprint_mask, object_healpix_mask, nest=footprint_nest)

    regions = track_band_regions(
        catalog, frame, track_phi2, phi1_range, sigma_deg, on_n_sigma=on_n_sigma, off_n_sigma=off_n_sigma,
        excluded_healpix_mask=excluded_mask, masked_objects=masked_objects, area_step_deg=area_step_deg,
        ra_col=ra_col, dec_col=dec_col,
    )

    def rounded_dm(phi1):
        return np.round(galstreams_dm_along_track(track, phi1, extrapolate=True) / dm_step) * dm_step

    in_range = np.isfinite(regions["delta_phi2"])
    star_dm = np.full(in_range.size, np.nan)
    star_dm[in_range] = rounded_dm(regions["phi1"][in_range])
    track_dm = rounded_dm(regions["track_phi1"])
    if label is None:
        label = f"{track.track_name} (dm {track_dm.min():.2f} to {track_dm.max():.2f})"
        if phi2_offset_deg:
            label += f", regions {phi2_offset_deg:+g} deg from the track"
    points = track.track
    regions.update({
        "label": label, "star_dm": star_dm, "track_dm": track_dm, "track": track, "track_name": track.track_name,
        "track_points_phi12": radec_to_phi12(points.ra.deg, points.dec.deg, frame),
        "phi2_offset_deg": float(phi2_offset_deg), "object_masks": object_masks,
        "map_bin_size_deg": DEFAULT_MAP_BIN_DEG, "footprint_mask": footprint_mask, "footprint_nest": footprint_nest,
    })
    return regions


# ===================================================
# Match filter along the track
# ===================================================


def track_match_filter_mask(
    catalog: pd.DataFrame,
    regions: Mapping[str, Any],
    isochrone: Mapping[str, Any] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    dm_offset: float = 0.0,
    cuts: Mapping[str, Any] | None = None,
) -> NDArray[np.bool_]:
    """Stars in the match filter at the distance modulus of the track where they are, as a mask.

    ``analysis.catalog_preparation.catalog_match_filter_mask`` of the stars
    sharing a distance modulus (``regions["star_dm"]``, that of the track at
    their phi1 rounded to ``dm_step``), at that distance modulus plus
    ``dm_offset``; the stars outside the phi1 range of the regions are not
    selected.

    Parameters
    ----------
    catalog : pandas.DataFrame
        The stars of ``regions``.
    regions : mapping
        From ``stream_regions``.
    isochrone, match_filter, cuts :
        See ``catalog_match_filter_mask``.
    dm_offset : float, optional
        Added to the distance modulus of the track, e.g. to test a distance error.

    Returns
    -------
    selected : numpy.ndarray of bool, shape (len(catalog),)
    """
    star_dm = np.asarray(regions["star_dm"], dtype=float)
    selected = np.zeros(len(catalog), dtype=bool)
    for dm in np.unique(star_dm[np.isfinite(star_dm)]):
        selected |= catalog_match_filter_mask(catalog, float(dm) + dm_offset, isochrone, match_filter, cuts=cuts,
                                              mask=star_dm == dm)
    return selected

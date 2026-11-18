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
3. ``stream_regions``: the on region (within ``on_n_sigma`` times the
   width of the track), the off regions (from ``off_n_sigma[0]`` to
   ``off_n_sigma[1]`` times the width, on both sides), the stars excluded
   (outside the footprint, or on a known object) and the areas.
4. Selections: boolean arrays over the rows of the catalog, as for the
   objects; ``track_match_filter_mask`` is the match filter at the distance
   of the track along it.
5. ``analysis.gc_dwarf_optimization.selection_snr`` and
   ``compare_selections``: the regions have the same keys as
   ``object_regions``.

Geometry. The regions follow the track in phi1 slices of ``slice_deg``, in
the galstreams frame of the track: in each slice, the track is at the phi2
of the track at the centre of the slice (``galstreams_phi2_along_track``)
and at its distance modulus there (``galstreams_dm_along_track``, rounded
to ``dm_step``). A star belongs to a region by its phi2 offset from the
track of its slice; the area of a region is the sum, over the slices, of
the unmasked area of its window in the slice
(``catalogs.survey_footprint.unmasked_area_deg2``): the stars and the areas
see the same stepped regions and the same masks (the footprint from the
stars and the known objects masked as in the searches,
``analysis.build_masks.excluded_healpix_mask``).

The plots (map with the regions; CMD and Hess diagrams with
``plotting.plot_gc_dwarf_optimization``) are in
``plotting.plot_stream_optimization``. The catalog is neither copied nor
modified. The catalog magnitudes are already dereddened.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from streameucliddes.analysis.build_masks import excluded_healpix_mask, overlapped_objects_healpix_mask
from streameucliddes.analysis.candidate_investigation import DEFAULT_OFF_N_SIGMA, DEFAULT_ON_N_SIGMA
from streameucliddes.analysis.catalog_preparation import catalog_match_filter_mask
from streameucliddes.analysis.fit_stream import DEFAULT_SIGMA_GUESS_DEG
from streameucliddes.analysis.gc_dwarf_optimization import DEFAULT_MASKED_OBJECT_TYPES
from streameucliddes.analysis.star_density_maps import star_density_phi12
from streameucliddes.catalogs.galstreams_catalog import (
    galstreams_dm_along_track,
    galstreams_phi2_along_track,
    galstreams_phi12_window,
)
from streameucliddes.catalogs.survey_footprint import (
    footprint_mask_from_stars,
    in_healpix_mask,
    survey_footprint_phi12_mask,
    unmasked_area_deg2,
)
from streameucliddes.stream_frames import phi12_bin_centres, phi12_window_radec_box, radec_to_phi12

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import galstreams
    import pandas as pd
    from numpy.typing import ArrayLike, NDArray

# phi1 slices of the regions: the track phi2 and distance modulus are those at the centre of each slice (deg)
DEFAULT_SLICE_DEG = 0.5
# Distance moduli of the slices rounded to this step: the stars of the slices sharing one go through one filter
DEFAULT_DM_STEP = 0.05
# Footprint from the stars (as object_regions) and resolution of the object masks (pixels of 0.03 deg, as the
# searches with 0.1 deg bins: build_masks.object_mask_nside(0.1))
DEFAULT_FOOTPRINT_NSIDE = 1024
DEFAULT_OBJECT_MASK_NSIDE = 2048
# Window of the regions: beyond the off regions by this margin in phi2 (deg); box read around it: this margin (deg)
WINDOW_MARGIN_DEG = 0.2
BOX_MARGIN_DEG = 0.3
# Maps of a stream: bin size (deg)
DEFAULT_MAP_BIN_DEG = 0.05


# ===================================================
# Field and regions along a track
# ===================================================


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
        along the track plus ``WINDOW_MARGIN_DEG``; the box
        (``stream_frames.phi12_window_radec_box``) is for ``get_euclid_stars``.
    """
    if phi1_range is None:
        phi1_range = galstreams_phi12_window(track, footprint_mask=footprint_mask, nest=nest, padding_phi1_deg=0.0,
                                             padding_phi2_deg=0.0, min_phi2_half_width_deg=0.0)[:2]
    phi1_range = (float(phi1_range[0]), float(phi1_range[1]))
    track_phi2 = galstreams_phi2_along_track(track, np.linspace(*phi1_range, 200)) + phi2_offset_deg
    half_width = off_n_sigma[1] * sigma_deg + WINDOW_MARGIN_DEG
    window = (*phi1_range, float(track_phi2.min() - half_width), float(track_phi2.max() + half_width))
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
    slice_deg: float = DEFAULT_SLICE_DEG,
    dm_step: float = DEFAULT_DM_STEP,
    masked_object_types: Sequence[str] = DEFAULT_MASKED_OBJECT_TYPES,
    object_healpix_mask: NDArray[np.bool_] | None = None,
    footprint_mask: NDArray[np.bool_] | None = None,
    footprint_nest: bool = False,
    phi2_offset_deg: float = 0.0,
    label: str | None = None,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> dict[str, Any]:
    """On and off regions along a stream track: the stars in each and their areas.

    In each phi1 slice of ``slice_deg`` over ``phi1_range``, the track is at
    its phi2 at the centre of the slice; a star is on the stream when its
    phi2 offset from it is at most ``on_n_sigma * sigma_deg``, off when it is
    between ``off_n_sigma[0] * sigma_deg`` and ``off_n_sigma[1] * sigma_deg``,
    on either side. Stars outside the footprint, or on a masked known
    object, are excluded; the areas leave out the same sky.

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
        Half-width of the on region, in units of ``sigma_deg``.
    off_n_sigma : tuple of float, optional
        Inner and outer distances of the off regions to the track, in units of ``sigma_deg``.
    slice_deg : float, optional
        Length of the phi1 slices (deg); the range is cut into equal slices of at most this length.
    dm_step : float, optional
        Rounding step of the distance moduli of the slices.
    masked_object_types : sequence of str, optional
        Types of the known objects masked (``stream_object_healpix_mask``); ignored with ``object_healpix_mask``.
    object_healpix_mask : numpy.ndarray of bool, optional
        Masked objects, RING, from ``stream_object_healpix_mask``: computed once for several catalogs.
    footprint_mask : numpy.ndarray of bool, optional
        HEALPix footprint, True outside. Default: from the stars of ``catalog``
        (``survey_footprint.footprint_mask_from_stars`` at ``DEFAULT_FOOTPRINT_NSIDE``).
    footprint_nest : bool, optional
        Ordering of a given ``footprint_mask``.
    phi2_offset_deg : float, optional
        Shift of the regions from the galstreams track in phi2 (deg), set by hand when
        the stream is not on its galstreams track in the catalog (e.g. Aliqa Uma in DR1: about +0.6 deg in
        the frame of ``galstreams_track_in_own_frame``): the centre of the regions in each slice is the track
        phi2 plus this offset.
    label : str, optional
        Title of the figures. Default: the track name and its distance moduli.
    ra_col, dec_col : str, optional
        Coordinate columns (deg).

    Returns
    -------
    regions : dict
        Per star (arrays of ``len(catalog)``): ``phi1``, ``phi2`` (deg, in
        ``stream_frame``), ``delta_phi2`` (offset from the track of its slice;
        NaN outside ``phi1_range``), ``star_dm`` (distance modulus of its
        slice; NaN outside), ``excluded``, ``on``, ``off_below``,
        ``off_above`` and ``off`` (bool, not excluded). And
        ``on_area_deg2``, ``off_area_deg2`` (the two off regions),
        ``off_below_area_deg2``, ``off_above_area_deg2``,
        ``on_area_fraction``, ``off_area_fraction`` (of the full regions, not
        excluded); ``slice_edges``, ``slice_phi2`` (centre of the regions: track plus offset), ``slice_dm``
        (per slice); ``phi2_offset_deg``;
        ``track``, ``track_name``, ``label``, ``stream_frame``, ``window``,
        ``phi1_range``, ``sigma_deg``, ``on_n_sigma``, ``off_n_sigma``,
        ``excluded_healpix_mask`` (RING), ``footprint_mask``, ``footprint_nest``.

    Raises
    ------
    ValueError
        ``sigma_deg`` not positive, the off regions overlap the on region,
        ``phi1_range`` not increasing, or the track has no distance.
    """
    if not sigma_deg > 0:
        raise ValueError(f"sigma_deg must be > 0, not {sigma_deg}.")
    if not on_n_sigma <= off_n_sigma[0] < off_n_sigma[1]:
        raise ValueError(f"Need on_n_sigma <= off_n_sigma[0] < off_n_sigma[1], not {on_n_sigma}, {off_n_sigma}.")
    if not phi1_range[1] > phi1_range[0]:
        raise ValueError(f"phi1_range must be increasing, not {phi1_range}.")
    frame = track.stream_frame
    ra, dec = catalog[ra_col].to_numpy(dtype=float), catalog[dec_col].to_numpy(dtype=float)
    phi1, phi2 = radec_to_phi12(ra, dec, frame)

    # Slices: track phi2 and distance modulus at their centres
    n_slices = int(np.ceil((phi1_range[1] - phi1_range[0]) / slice_deg - 1e-9))
    slice_edges = np.linspace(phi1_range[0], phi1_range[1], n_slices + 1)
    slice_centres = (slice_edges[:-1] + slice_edges[1:]) / 2
    slice_phi2 = galstreams_phi2_along_track(track, slice_centres) + phi2_offset_deg
    slice_dm = galstreams_dm_along_track(track, slice_centres, extrapolate=True)
    if np.isnan(slice_dm).any():
        raise ValueError(f"Track {track.track_name} has no distance in galstreams.")
    slice_dm = np.round(slice_dm / dm_step) * dm_step
    in_range = (phi1 >= phi1_range[0]) & (phi1 <= phi1_range[1])
    star_slice = np.clip(np.searchsorted(slice_edges, phi1, side="right") - 1, 0, n_slices - 1)
    delta_phi2 = np.where(in_range, phi2 - slice_phi2[star_slice], np.nan)
    star_dm = np.where(in_range, slice_dm[star_slice], np.nan)

    # Excluded sky: outside the footprint, or on a masked object
    if footprint_mask is None:
        footprint_mask, footprint_nest = footprint_mask_from_stars(ra, dec, DEFAULT_FOOTPRINT_NSIDE), False
    if object_healpix_mask is None:
        object_healpix_mask = stream_object_healpix_mask(masked_object_types)
    excluded_mask, _ = excluded_healpix_mask(footprint_mask, object_healpix_mask, nest=footprint_nest)
    excluded = ~np.isfinite(ra) | ~np.isfinite(dec) | in_healpix_mask(ra, dec, excluded_mask, nest=False)

    on_half = on_n_sigma * sigma_deg
    off_inner, off_outer = (n_sigma * sigma_deg for n_sigma in off_n_sigma)
    distance = np.abs(delta_phi2)
    with np.errstate(invalid="ignore"):  # NaN outside phi1_range: in no region
        on = ~excluded & (distance <= on_half)
        off_below = ~excluded & (delta_phi2 <= -off_inner) & (delta_phi2 >= -off_outer)
        off_above = ~excluded & (delta_phi2 >= off_inner) & (delta_phi2 <= off_outer)

    # Areas: the window of each region in each slice, outside the excluded sky and in full
    offsets = {"on": (-on_half, on_half), "off_below": (-off_outer, -off_inner), "off_above": (off_inner, off_outer)}
    areas = {}
    for name, (low, high) in offsets.items():
        kept = full = 0.0
        for (edge_low, edge_high), centre in zip(zip(slice_edges[:-1], slice_edges[1:]), slice_phi2):
            cell = ((edge_low, edge_high), (centre + low, centre + high))
            kept += float(unmasked_area_deg2(frame, *cell, excluded_mask)[0, 0])
            full += float(unmasked_area_deg2(frame, *cell, None)[0, 0])
        areas[name] = (kept, full)
    off_kept = areas["off_below"][0] + areas["off_above"][0]
    off_full = areas["off_below"][1] + areas["off_above"][1]

    if label is None:
        label = f"{track.track_name} (dm {slice_dm.min():.2f} to {slice_dm.max():.2f})"
        if phi2_offset_deg:
            label += f", regions {phi2_offset_deg:+g} deg from the track"
    window = (phi1_range[0], phi1_range[1], float(slice_phi2.min() - off_outer - WINDOW_MARGIN_DEG),
              float(slice_phi2.max() + off_outer + WINDOW_MARGIN_DEG))
    return {
        "track": track, "track_name": track.track_name, "label": label, "stream_frame": frame, "window": window,
        "phi1_range": (float(phi1_range[0]), float(phi1_range[1])), "sigma_deg": float(sigma_deg),
        "phi2_offset_deg": float(phi2_offset_deg),
        "on_n_sigma": float(on_n_sigma), "off_n_sigma": (float(off_n_sigma[0]), float(off_n_sigma[1])),
        "slice_edges": slice_edges, "slice_phi2": slice_phi2, "slice_dm": slice_dm,
        "phi1": phi1, "phi2": phi2, "delta_phi2": delta_phi2, "star_dm": star_dm, "excluded": excluded,
        "on": on, "off_below": off_below, "off_above": off_above, "off": off_below | off_above,
        "on_area_deg2": areas["on"][0], "off_area_deg2": off_kept,
        "off_below_area_deg2": areas["off_below"][0], "off_above_area_deg2": areas["off_above"][0],
        "on_area_fraction": areas["on"][0] / areas["on"][1], "off_area_fraction": off_kept / off_full,
        "excluded_healpix_mask": excluded_mask, "footprint_mask": footprint_mask, "footprint_nest": footprint_nest,
    }


# ===================================================
# Selections and maps
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
    of the slices sharing a distance modulus (``regions["star_dm"]``), at
    that distance modulus plus ``dm_offset``; the stars outside the phi1
    range of the regions are not selected.

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


def stream_density_map(
    regions: Mapping[str, Any],
    selected: ArrayLike | None = None,
    *,
    bin_size_deg: float = DEFAULT_MAP_BIN_DEG,
    smoothing_deg: float = 0.0,
) -> dict[str, Any]:
    """Map of the selected stars along a stream, in its frame.

    ``star_density_maps.star_density_phi12`` of the selected stars not
    excluded, on the grid of the window of the regions; the bins whose
    centre is in the excluded sky (outside the footprint, on a masked
    object) are masked (NaN).

    Parameters
    ----------
    regions : mapping
        From ``stream_regions``.
    selected : array_like of bool, optional
        Selected stars, one per row of the catalog. Default: all.
    bin_size_deg : float, optional
        Bin size (deg).
    smoothing_deg : float, optional
        Gaussian smoothing (deg); 0: none.

    Returns
    -------
    density_map : dict
        The keys of ``gc_dwarf_optimization.object_density_map``, for
        ``plotting.plot_stream_frame_search.plot_stream_frame_search_single_dm``.
    """
    window, frame = regions["window"], regions["stream_frame"]
    stars = ~np.asarray(regions["excluded"], dtype=bool)
    if selected is not None:
        stars &= np.asarray(selected, dtype=bool)
    grid_mask = survey_footprint_phi12_mask(regions["excluded_healpix_mask"], frame, window, bin_size_deg, nest=False)
    phi1_centres, phi2_centres = phi12_bin_centres(window, bin_size_deg)
    density = star_density_phi12(regions["phi1"][stars], regions["phi2"][stars], window, bin_size_deg,
                                 smoothing_deg=smoothing_deg, footprint_mask=grid_mask)
    return {"density": density, "phi1_centres": phi1_centres, "phi2_centres": phi2_centres,
            "bin_size_deg": bin_size_deg, "window": window, "smoothing_deg": smoothing_deg,
            "n_selected": int(stars.sum())}

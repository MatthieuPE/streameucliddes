"""On and off regions around an object or along a track: their stars, areas, SNR, maps and Hess diagrams.

Shared by every study of a signal over the field around it: the tuning of
the selections on known objects (``analysis.gc_dwarf_optimization``) and on
known streams (``analysis.stream_optimization``), and the investigation of
candidates.

A set of regions is defined by one coordinate of the sky, the
``region_coordinate`` (deg): the elliptical radius around an object
(``object_elliptical_radius_deg``), or the phi2 offset from a track,
``phi2 - track_phi2(phi1)`` (``track_band_regions``). Each region is an
interval of it: the on region, and one or more off parts (an annulus
around an object; one band below and one above a track). The stars and the
areas see the same regions and the same excluded sky:

- a star is in a region when its coordinate is in the interval and it is
  not excluded;
- the area of a region is the sum of the small pieces of the window
  (``catalogs.survey_footprint.phi12_area_pieces``, exact solid angles) not
  excluded (their centre), each times its fraction in the region: the part of
  its phi2 height where the coordinate, taken linear across it, is in the
  interval (exact for a band along a track, whose edges need not fall
  between two rows of pieces).

The excluded sky (``excluded_points``) is a HEALPix mask (the footprint,
and the known objects as masked in the searches,
``analysis.build_masks.excluded_healpix_mask``) and/or the exact ellipses
of the masked objects (``masked_objects_in_window``): ellipses around one
object, where a neighbour can be smaller than a HEALPix pixel; the HEALPix
mask of the searches along a stream.

The background expected on the on region is its area times the mean of the
densities of the off parts (``analysis.cmd_match_filter.mean_density_of_off_parts``):
with two parts on each side of a track, a linear gradient of the field
across it cancels whatever area the masks leave in each; with one (an
annulus), its density. ``selection_snr``: SNR = excess / sqrt(background on
the on region) (``compute_snr``).

The regions are a plain dict, one value per star for the masks (the catalog
is neither copied nor modified). Plots: ``plotting.plot_on_off_regions``.
The catalog magnitudes are already dereddened.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from streameucliddes.analysis.build_masks import elliptical_radius_deg, mask_radius_scales
from streameucliddes.analysis.cmd_match_filter import mean_density_of_off_parts, on_off_hess
from streameucliddes.analysis.star_density_maps import star_density_phi12
from streameucliddes.catalogs import overlapped_objects
from streameucliddes.catalogs.survey_footprint import in_healpix_mask, phi12_area_pieces
from streameucliddes.stream_frames import phi12_bin_centres, phi12_to_radec, radec_to_phi12

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any

    import astropy.coordinates as coord
    from numpy.typing import ArrayLike, NDArray

# Bands along a track: half-width of the on band and distances of the off bands, in widths of the stream (user
# choices: 1.4 maximizes the SNR of a Gaussian stream on a flat background; the off bands start well past its wings).
# Shipp et al. 2018 (Sect. 3.3, Table A.1) differ: on +-2w for their Hess diagrams, +-w for their significance, off
# bands of the same 4w width centred at +-4w (2w to 6w), kept as reference only (user choice 2026-11-19)
DEFAULT_ON_N_SIGMA = 1.4
DEFAULT_OFF_N_SIGMA = (3.5, 6)
# Other objects masked around a signal (catalogs.overlapped_objects.OBJECT_TYPES)
DEFAULT_MASKED_OBJECT_TYPES = ("dwarfs", "gc", "galaxies")
# Areas: pieces so that so many fit across the narrowest region, at most AREA_MAX_STEP_DEG (well below the HEALPix
# pixels of the masks), at least so large that at most AREA_MAX_PIECES fit along an axis of the window
AREA_PIECES_ACROSS = 20
AREA_MAX_STEP_DEG = 0.01
AREA_MAX_PIECES = 1000
# Bands along a track: points of the track (and of the outlines) per degree of phi1; window beyond the off bands (deg)
TRACK_POINTS_PER_DEG = 20
DEFAULT_WINDOW_MARGIN_DEG = 0.2
# Maps: bins along phi1 when the regions give no bin size
DEFAULT_MAP_BINS = 60


# ===================================================
# Excluded sky
# ===================================================


def object_elliptical_radius_deg(ra: ArrayLike, dec: ArrayLike, obj: Mapping[str, Any]) -> NDArray[np.float64]:
    """Elliptical radius (deg) of points around an object (``build_masks.elliptical_radius_deg``).

    ``obj`` has ``ra``, ``dec``, ``ellipticity`` and ``position_angle`` (a
    row of ``catalogs.overlapped_objects.get_objects``): a circle when they
    are unknown (NaN).
    """
    ellipticity, major_axis_angle = (float(obj[key]) for key in ("ellipticity", "position_angle"))
    return elliptical_radius_deg(ra, dec, float(obj["ra"]), float(obj["dec"]),
                                 ellipticity if np.isfinite(ellipticity) else 0.0,
                                 major_axis_angle if np.isfinite(major_axis_angle) else 0.0)


def masked_objects_table(object_types: Sequence[str] = DEFAULT_MASKED_OBJECT_TYPES) -> pd.DataFrame:
    """The known objects of some types, with the radius of their mask in the searches.

    ``catalogs.overlapped_objects.get_objects`` plus ``mask_radius_arcmin``,
    ``build_masks.mask_radius_scales`` times the reference radius. Galaxies
    are circles, as in ``build_masks.galaxies_healpix_mask``; objects without
    a mask radius are left out.

    Parameters
    ----------
    object_types : sequence of str, optional
        Among ``"dwarfs"``, ``"gc"``, ``"galaxies"``.

    Returns
    -------
    objects : pandas.DataFrame
    """
    objects = overlapped_objects.get_objects(object_types)
    objects.loc[objects["type"] == "galaxies", "ellipticity"] = np.nan
    scales = np.full(len(objects), np.nan)
    for object_type, rows in objects.groupby("type").indices.items():
        scales[rows] = mask_radius_scales(objects.iloc[rows], object_type)
    objects["mask_radius_arcmin"] = scales * objects["radius_arcmin"].to_numpy(dtype=float)
    return objects[np.isfinite(objects["mask_radius_arcmin"].to_numpy())].reset_index(drop=True)


def masked_objects_in_window(
    frame: coord.BaseCoordinateFrame,
    window: tuple[float, float, float, float],
    object_types: Sequence[str] = DEFAULT_MASKED_OBJECT_TYPES,
) -> pd.DataFrame:
    """The known objects whose mask reaches a phi1/phi2 window (``masked_objects_table``).

    An object is kept when its centre is in the window widened by its mask
    radius (in phi1, divided by the cosine of the largest ``|phi2|`` of the
    window: wide enough everywhere in it).

    Parameters
    ----------
    frame : astropy.coordinates.BaseCoordinateFrame
        Frame of the window.
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg).
    object_types : sequence of str, optional
        See ``masked_objects_table``.

    Returns
    -------
    objects : pandas.DataFrame
        With ``mask_radius_arcmin``; empty without types.
    """
    objects = masked_objects_table(object_types)
    if objects.empty:
        return objects
    phi1, phi2 = radec_to_phi12(objects["ra"].to_numpy(dtype=float), objects["dec"].to_numpy(dtype=float), frame)
    radius = objects["mask_radius_arcmin"].to_numpy() / 60.0
    stretch = 1.0 / np.cos(np.radians(min(89.0, max(abs(window[2]), abs(window[3])))))
    reaches = ((phi1 >= window[0] - radius * stretch) & (phi1 <= window[1] + radius * stretch)
               & (phi2 >= window[2] - radius) & (phi2 <= window[3] + radius))
    return objects[reaches].reset_index(drop=True)


def on_masked_objects(ra: ArrayLike, dec: ArrayLike, masked_objects: pd.DataFrame | None) -> NDArray[np.bool_]:
    """Points inside the mask ellipse of one of ``masked_objects`` (elliptical radius at most ``mask_radius_arcmin``)."""
    ra, dec = np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
    inside = np.zeros(ra.shape, dtype=bool)
    if masked_objects is None:
        return inside
    for _, other in masked_objects.iterrows():
        inside |= object_elliptical_radius_deg(ra, dec, other) <= other["mask_radius_arcmin"] / 60.0
    return inside


def excluded_points(
    ra: ArrayLike,
    dec: ArrayLike,
    excluded_healpix_mask: NDArray[np.bool_] | None = None,
    masked_objects: pd.DataFrame | None = None,
) -> NDArray[np.bool_]:
    """Points in the excluded sky: in the HEALPix mask (RING), on a masked object, or without coordinates.

    Parameters
    ----------
    ra, dec : array_like
        Points (deg).
    excluded_healpix_mask : numpy.ndarray of bool, optional
        True where excluded, RING (footprint OR objects:
        ``analysis.build_masks.excluded_healpix_mask``).
    masked_objects : pandas.DataFrame, optional
        Objects masked by their exact ellipse (``masked_objects_in_window``).

    Returns
    -------
    excluded : numpy.ndarray of bool
    """
    ra, dec = np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
    excluded = ~np.isfinite(ra) | ~np.isfinite(dec)
    if excluded_healpix_mask is not None:
        excluded |= in_healpix_mask(ra, dec, excluded_healpix_mask, nest=False)
    return excluded | on_masked_objects(ra, dec, masked_objects)


# ===================================================
# Regions
# ===================================================


def on_off_regions(
    catalog: pd.DataFrame,
    frame: coord.BaseCoordinateFrame,
    window: tuple[float, float, float, float],
    region_coordinate: Callable[..., NDArray[np.float64]],
    intervals: Mapping[str, tuple[float, float]],
    *,
    excluded_healpix_mask: NDArray[np.bool_] | None = None,
    masked_objects: pd.DataFrame | None = None,
    area_step_deg: float | None = None,
    label: str = "",
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> dict[str, Any]:
    """On and off regions as intervals of one coordinate: the stars in each and their areas.

    The builders (``track_band_regions``, ``gc_dwarf_optimization.object_regions``)
    call it with their coordinate and intervals.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars. Not copied.
    frame : astropy.coordinates.BaseCoordinateFrame
        Frame of the window, the maps and the outlines.
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg), holding every region.
    region_coordinate : callable
        ``region_coordinate(ra, dec, phi1, phi2)`` (deg; arrays): the coordinate
        of the regions, NaN where a point is in no region.
    intervals : mapping
        ``{"on": (low, high), <off part>: (low, high), ...}``: the on region
        and one or more off parts (deg, bounds included), in this order.
    excluded_healpix_mask, masked_objects : optional
        Excluded sky, see ``excluded_points``.
    area_step_deg : float, optional
        Size of the area pieces (deg). Default: ``AREA_PIECES_ACROSS`` across
        the narrowest region, at most ``AREA_MAX_STEP_DEG``, at least
        ``AREA_MAX_PIECES`` along the longer axis of the window.
    label : str, optional
        Title of the figures.
    ra_col, dec_col : str, optional
        Coordinate columns (deg).

    Returns
    -------
    regions : dict
        Per star (arrays of ``len(catalog)``): ``phi1``, ``phi2`` (deg, in
        ``stream_frame``), ``region_coordinate``, ``excluded``, ``on``, each
        off part and ``off`` (their union; bool, not excluded). And
        ``<region>_area_deg2`` and ``<region>_area_fraction`` (not excluded,
        of the full region) of ``on``, each off part and ``off`` (their sum);
        ``off_parts`` (their names), ``intervals``, ``stream_frame``,
        ``window``, ``label``, ``excluded_healpix_mask``, ``masked_objects``,
        ``area_step_deg``.

    Raises
    ------
    ValueError
        No on region or no off part, an interval not increasing, or an off
        part overlapping the on region.
    """
    intervals = {name: (float(low), float(high)) for name, (low, high) in intervals.items()}
    off_parts = tuple(name for name in intervals if name != "on")
    if "on" not in intervals or not off_parts:
        raise ValueError(f"Need an 'on' interval and at least one off part, not {list(intervals)}.")
    on_low, on_high = intervals["on"]
    for name, (low, high) in intervals.items():
        if not high > low:
            raise ValueError(f"The interval of {name} must be increasing, not {(low, high)}.")
        if name != "on" and low < on_high and on_low < high:
            raise ValueError(f"The off region {name} ({low:.3g} to {high:.3g} deg) overlaps the on region "
                             f"({on_low:.3g} to {on_high:.3g} deg).")
    if masked_objects is None:
        masked_objects = masked_objects_in_window(frame, window, ())

    ra, dec = catalog[ra_col].to_numpy(dtype=float), catalog[dec_col].to_numpy(dtype=float)
    phi1, phi2 = radec_to_phi12(ra, dec, frame)
    coordinate = region_coordinate(ra, dec, phi1, phi2)
    excluded = excluded_points(ra, dec, excluded_healpix_mask, masked_objects)

    if area_step_deg is None:
        narrowest = min(high - low for low, high in intervals.values())
        longest = max(window[1] - window[0], window[3] - window[2])
        area_step_deg = max(min(narrowest / AREA_PIECES_ACROSS, AREA_MAX_STEP_DEG), longest / AREA_MAX_PIECES)
    piece_phi1, piece_phi2, piece_area, _ = phi12_area_pieces(window[:2], window[2:], step_deg=area_step_deg)
    piece_ra, piece_dec = phi12_to_radec(piece_phi1, piece_phi2, frame)
    piece_kept = ~excluded_points(piece_ra, piece_dec, excluded_healpix_mask, masked_objects)
    piece_range = _piece_coordinate_range(region_coordinate, frame, window, piece_phi1, piece_phi2)

    regions: dict[str, Any] = {
        "label": label, "stream_frame": frame, "window": tuple(float(value) for value in window),
        "intervals": intervals, "off_parts": off_parts, "phi1": phi1, "phi2": phi2,
        "region_coordinate": coordinate, "excluded": excluded, "excluded_healpix_mask": excluded_healpix_mask,
        "masked_objects": masked_objects, "area_step_deg": float(area_step_deg),
    }
    full_areas = {}
    with np.errstate(invalid="ignore"):  # NaN coordinate: in no region
        for name, (low, high) in intervals.items():
            regions[name] = ~excluded & (coordinate >= low) & (coordinate <= high)
            inside = piece_area * _fraction_inside(*piece_range, low, high)
            regions[f"{name}_area_deg2"] = float(inside[piece_kept].sum())
            full_areas[name] = float(inside.sum())
    regions["off"] = np.logical_or.reduce([regions[part] for part in off_parts])
    regions["off_area_deg2"] = float(sum(regions[f"{part}_area_deg2"] for part in off_parts))
    full_areas["off"] = sum(full_areas[part] for part in off_parts)
    for name, full in full_areas.items():
        regions[f"{name}_area_fraction"] = regions[f"{name}_area_deg2"] / full if full > 0 else np.nan
    return regions


def _piece_coordinate_range(
    region_coordinate: Callable[..., NDArray[np.float64]],
    frame: coord.BaseCoordinateFrame,
    window: tuple[float, float, float, float],
    piece_phi1: NDArray[np.float64],
    piece_phi2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coordinate of the regions at the bottom and top edges of each area piece (at its phi1), lower first."""
    rows = np.unique(piece_phi2)
    half_height = (np.min(np.diff(rows)) if rows.size > 1 else window[3] - window[2]) / 2
    edges = []
    for edge_phi2 in (piece_phi2 - half_height, piece_phi2 + half_height):
        edges.append(region_coordinate(*phi12_to_radec(piece_phi1, edge_phi2, frame), piece_phi1, edge_phi2))
    return np.fmin(*edges), np.fmax(*edges)


def _fraction_inside(
    lower: NDArray[np.float64], upper: NDArray[np.float64], low: float, high: float
) -> NDArray[np.float64]:
    """Fraction of each piece in the interval ``[low, high]`` of the coordinate, ranging from lower to upper over it.

    The coordinate is taken linear across the piece in phi2: exact for the
    phi2 offset from a track, so that an edge of a band between two rows of
    pieces is not rounded to a row. NaN (in no region): 0.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        span = upper - lower
        overlap = np.clip(np.minimum(upper, high) - np.maximum(lower, low), 0.0, None)
        fraction = np.where(span > 0, overlap / span, (lower >= low) & (upper <= high))
    return np.nan_to_num(np.clip(fraction, 0.0, 1.0), nan=0.0)


def track_band_window(
    track_phi2: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    phi1_range: tuple[float, float],
    half_width_deg: float,
    *,
    margin_deg: float = DEFAULT_WINDOW_MARGIN_DEG,
) -> tuple[float, float, float, float]:
    """phi1/phi2 window of the bands within ``half_width_deg`` of a track over ``phi1_range``, plus a margin in phi2."""
    phi2 = track_phi2(_track_phi1_grid(phi1_range))
    return (float(phi1_range[0]), float(phi1_range[1]), float(np.min(phi2) - half_width_deg - margin_deg),
            float(np.max(phi2) + half_width_deg + margin_deg))


def _track_phi1_grid(phi1_range: tuple[float, float]) -> NDArray[np.float64]:
    """phi1 of the points of a track over a range: ``TRACK_POINTS_PER_DEG`` per degree, at least 50."""
    n_points = max(int(np.ceil((phi1_range[1] - phi1_range[0]) * TRACK_POINTS_PER_DEG)) + 1, 50)
    return np.linspace(phi1_range[0], phi1_range[1], n_points)


def track_band_regions(
    catalog: pd.DataFrame,
    frame: coord.BaseCoordinateFrame,
    track_phi2: Callable[[NDArray[np.float64]], NDArray[np.float64]] | float,
    phi1_range: tuple[float, float],
    sigma_deg: float,
    *,
    on_n_sigma: float,
    off_n_sigma: tuple[float, float],
    on_phi2_offset_deg: float = 0.0,
    off_phi2_offsets_deg: tuple[float, float] = (0.0, 0.0),
    window: tuple[float, float, float, float] | None = None,
    window_margin_deg: float = DEFAULT_WINDOW_MARGIN_DEG,
    excluded_healpix_mask: NDArray[np.bool_] | None = None,
    masked_objects: pd.DataFrame | None = None,
    area_step_deg: float | None = None,
    label: str = "",
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> dict[str, Any]:
    """On and off bands along a track: the stars in each and their areas (``on_off_regions``).

    The coordinate is the phi2 offset of a point from the track at its phi1,
    ``phi2 - track_phi2(phi1)``, over ``phi1_range`` (NaN beyond): the bands
    follow the track continuously. The on band is ``|offset| <= on_n_sigma *
    sigma_deg``; the off parts ``off_below`` and ``off_above`` are from
    ``off_n_sigma[0] * sigma_deg`` to ``off_n_sigma[1] * sigma_deg`` on each
    side. A straight track is a constant ``track_phi2``.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars. Not copied.
    frame : astropy.coordinates.BaseCoordinateFrame
        Frame of the track.
    track_phi2 : callable or float
        phi2 of the track (deg) at an array of phi1 (deg), e.g.
        ``lambda phi1: galstreams_phi2_along_track(track, phi1)``; a number for
        a straight track.
    phi1_range : tuple of float
        phi1 range of the bands (deg).
    sigma_deg : float
        Width of the stream in phi2 (deg).
    on_n_sigma : float
        Half-width of the on band, in units of ``sigma_deg``.
    off_n_sigma : tuple of float
        Inner and outer distances of the off bands to the track, in units of ``sigma_deg``.
    on_phi2_offset_deg : float, optional
        phi2 shift of the on band (deg), e.g. off the stream as a null test.
    off_phi2_offsets_deg : tuple of float, optional
        phi2 shifts of the off bands below and above (deg).
    window : tuple of float, optional
        Window of the maps and areas. Default: ``track_band_window`` of the
        off bands, with ``window_margin_deg``.
    excluded_healpix_mask, masked_objects, area_step_deg, label, ra_col, dec_col :
        See ``on_off_regions``.

    Returns
    -------
    regions : dict
        Those of ``on_off_regions`` (off parts ``off_below`` and
        ``off_above``), plus ``delta_phi2`` (the coordinate: offset from the
        track; NaN outside ``phi1_range``), ``phi1_range``, ``sigma_deg``,
        ``on_n_sigma``, ``off_n_sigma``, ``on_phi2_offset_deg``,
        ``off_phi2_offsets_deg``, ``track_phi1`` and ``track_phi2`` (the
        track on a grid of ``phi1_range``) and ``outlines`` (``{"on": [(phi1,
        phi2)], "off": [below, above]}``, closed).

    Raises
    ------
    ValueError
        ``sigma_deg`` not positive, ``phi1_range`` or ``off_n_sigma`` not
        increasing, an off band overlapping the on band.
    """
    if not sigma_deg > 0:
        raise ValueError(f"sigma_deg must be > 0, not {sigma_deg}.")
    if not phi1_range[1] > phi1_range[0]:
        raise ValueError(f"phi1_range must be increasing, not {phi1_range}.")
    if not off_n_sigma[1] > off_n_sigma[0]:
        raise ValueError(f"off_n_sigma must be increasing, not {off_n_sigma}.")
    phi1_range = (float(phi1_range[0]), float(phi1_range[1]))
    if not callable(track_phi2):
        constant_phi2 = float(track_phi2)

        def track_phi2(phi1):
            return np.full(np.shape(phi1), constant_phi2)

    on_half = on_n_sigma * sigma_deg
    off_inner, off_outer = (n_sigma * sigma_deg for n_sigma in off_n_sigma)
    below_offset, above_offset = (float(offset) for offset in off_phi2_offsets_deg)
    intervals = {"on": (-on_half + on_phi2_offset_deg, on_half + on_phi2_offset_deg),
                 "off_below": (-off_outer + below_offset, -off_inner + below_offset),
                 "off_above": (off_inner + above_offset, off_outer + above_offset)}
    if window is None:
        reach = max(abs(bound) for interval in intervals.values() for bound in interval)
        window = track_band_window(track_phi2, phi1_range, reach, margin_deg=window_margin_deg)

    def phi2_offset(ra, dec, phi1, phi2):
        in_range = (phi1 >= phi1_range[0]) & (phi1 <= phi1_range[1])
        return np.where(in_range, phi2 - track_phi2(np.clip(phi1, *phi1_range)), np.nan)

    regions = on_off_regions(catalog, frame, window, phi2_offset, intervals,
                             excluded_healpix_mask=excluded_healpix_mask, masked_objects=masked_objects,
                             area_step_deg=area_step_deg, label=label, ra_col=ra_col, dec_col=dec_col)
    grid_phi1 = _track_phi1_grid(phi1_range)
    grid_phi2 = track_phi2(grid_phi1)
    outlines = {name: _band_outline(grid_phi1, grid_phi2, *intervals[name]) for name in intervals}
    regions.update({
        "delta_phi2": regions["region_coordinate"], "phi1_range": phi1_range, "sigma_deg": float(sigma_deg),
        "on_n_sigma": float(on_n_sigma), "off_n_sigma": (float(off_n_sigma[0]), float(off_n_sigma[1])),
        "on_phi2_offset_deg": float(on_phi2_offset_deg), "off_phi2_offsets_deg": (below_offset, above_offset),
        "track_phi1": grid_phi1, "track_phi2": grid_phi2,
        "outlines": {"on": [outlines["on"]], "off": [outlines["off_below"], outlines["off_above"]]},
    })
    return regions


def _band_outline(
    phi1: NDArray[np.float64], track_phi2: NDArray[np.float64], low: float, high: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Closed outline of the band from ``low`` to ``high`` off a track: lower edge, then upper edge backwards."""
    return (np.concatenate([phi1, phi1[::-1], phi1[:1]]),
            np.concatenate([track_phi2 + low, (track_phi2 + high)[::-1], track_phi2[:1] + low]))


# ===================================================
# SNR of selections
# ===================================================


def compute_snr(n_stream: float, n_background: float) -> float:
    """SNR of ``n_stream`` stream (or excess) stars over ``n_background`` background stars.

    ``n_stream / sqrt(n_background)``; 0 without stream stars, inf without
    background. The SNR of ``selection_snr`` and of the injected streams
    (``analysis.injected_stream_snr.get_stream_snr``).
    """
    if n_stream == 0:
        return 0.0  # No stream star selected, even without background
    if n_background == 0:
        return np.inf  # Avoid division by zero
    return n_stream / np.sqrt(n_background)


def selected_stars(regions: Mapping[str, Any], selected: ArrayLike | None = None) -> NDArray[np.bool_]:
    """The selected stars as a boolean array, one per star of the regions; all by default."""
    return np.ones(len(regions["on"]), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)


def selection_snr(regions: Mapping[str, Any], selected: ArrayLike | None = None) -> dict[str, float]:
    """Excess of the selected stars on the on region over the off parts, and its SNR.

    The background density is the mean of the densities of the off parts
    (``mean_density_of_off_parts``: a part with no area left out); the
    background expected on the on region is its area times it; SNR = excess
    / sqrt(background on the on region) (``compute_snr``). With one off part
    (an annulus), the off count scaled by the ratio of the areas.

    Parameters
    ----------
    regions : mapping
        From a builder of regions: ``on`` and the ``off_parts`` (one value per
        star) and their ``<region>_area_deg2``. Without ``off_parts``: the
        single part ``off``.
    selected : array_like of bool, optional
        Selected stars, one per row of the catalog. Default: all.

    Returns
    -------
    snr : dict
        ``n_on``, ``n_off`` (selected stars in the on region and in all off
        parts), ``n_<part>`` for each off part, ``n_background_on``,
        ``n_signal`` (``n_on - n_background_on``), ``n_signal_error``
        (Poisson, on and off), ``snr``, ``background_density`` (stars/deg²),
        ``n_off_parts`` (in the mean), ``on_area_deg2``, ``off_area_deg2``.
    """
    selected = selected_stars(regions, selected)
    parts = tuple(regions.get("off_parts", ("off",)))
    n_on = int((selected & np.asarray(regions["on"], dtype=bool)).sum())
    counts = {part: int((selected & np.asarray(regions[part], dtype=bool)).sum()) for part in parts}
    areas = [float(regions[f"{part}_area_deg2"]) for part in parts]
    with np.errstate(divide="ignore", invalid="ignore"):
        densities = [count / area if area > 0 else np.nan for count, area in zip(counts.values(), areas)]
        errors = [np.sqrt(count) / area if area > 0 else np.nan for count, area in zip(counts.values(), areas)]
    density, error, n_parts = mean_density_of_off_parts(densities, errors)
    on_area = float(regions["on_area_deg2"])
    n_background = on_area * float(density)
    n_signal = n_on - n_background
    return {
        "n_on": n_on, "n_off": int(sum(counts.values())), **{f"n_{part}": count for part, count in counts.items()},
        "n_background_on": float(n_background), "n_signal": float(n_signal),
        "n_signal_error": float(np.sqrt(n_on + (on_area * float(error)) ** 2)),
        "snr": float(compute_snr(n_signal, n_background)), "background_density": float(density),
        "n_off_parts": int(n_parts), "on_area_deg2": on_area, "off_area_deg2": float(sum(areas)),
    }


def compare_selections(regions: Mapping[str, Any], selections: Mapping[str, ArrayLike]) -> pd.DataFrame:
    """``selection_snr`` of several selections, one row each.

    Parameters
    ----------
    regions : mapping
        From a builder of regions.
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


# ===================================================
# Map and Hess diagrams
# ===================================================


def region_density_map(
    regions: Mapping[str, Any],
    selected: ArrayLike | None = None,
    *,
    bin_size_deg: float | None = None,
    smoothing_deg: float = 0.0,
) -> dict[str, Any]:
    """Map of the selected stars of the regions, in their frame.

    ``star_density_maps.star_density_phi12`` of the selected stars not
    excluded, on the grid of the window of the regions; the bins whose
    centre is in the excluded sky (``excluded_points``) are masked (NaN).

    Parameters
    ----------
    regions : mapping
        From a builder of regions.
    selected : array_like of bool, optional
        Selected stars, one per row of the catalog. Default: all.
    bin_size_deg : float, optional
        Default: the ``map_bin_size_deg`` of the regions, else
        ``DEFAULT_MAP_BINS`` bins along phi1.
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
    if bin_size_deg is None:
        bin_size_deg = regions.get("map_bin_size_deg") or (window[1] - window[0]) / DEFAULT_MAP_BINS
    stars = selected_stars(regions, selected) & ~np.asarray(regions["excluded"], dtype=bool)
    phi1_centres, phi2_centres = phi12_bin_centres(window, bin_size_deg)
    centres_ra, centres_dec = phi12_to_radec(*np.meshgrid(phi1_centres, phi2_centres, indexing="ij"), frame)
    grid_mask = excluded_points(centres_ra, centres_dec, regions["excluded_healpix_mask"], regions["masked_objects"])
    density = star_density_phi12(regions["phi1"][stars], regions["phi2"][stars], window, bin_size_deg,
                                 smoothing_deg=smoothing_deg, footprint_mask=grid_mask)
    return {"density": density, "phi1_centres": phi1_centres, "phi2_centres": phi2_centres,
            "bin_size_deg": bin_size_deg, "window": window, "smoothing_deg": smoothing_deg,
            "n_selected": int(stars.sum())}


def region_on_off_hess(
    catalog: pd.DataFrame,
    mag_cols: Sequence[str],
    regions: Mapping[str, Any],
    selected: ArrayLike | None = None,
    *,
    color_bins: ArrayLike | None = None,
    mag_bins: ArrayLike | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """On, off and on - off Hess diagrams of the regions (``cmd_match_filter.on_off_hess``).

    The off diagram is the mean of those of the off parts, as the background
    of ``selection_snr``.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars, with the ``mag_cols`` columns. Not copied.
    mag_cols : sequence of two str
        Colour ``mag_cols[0] - mag_cols[1]``, magnitude ``mag_cols[0]``.
    regions : mapping
        From a builder of regions.
    selected : array_like of bool, optional
        Stars counted. Default: all.
    color_bins, mag_bins : array_like, optional
        See ``on_off_hess``.

    Returns
    -------
    on_hess, off_hess, difference : dict
    """
    selected = selected_stars(regions, selected)
    off_masks, off_areas = off_part_masks_and_areas(regions, selected)
    return on_off_hess(catalog, mag_cols, selected & regions["on"], regions["on_area_deg2"], off_masks, off_areas,
                       color_bins=color_bins, mag_bins=mag_bins)


def off_part_masks_and_areas(
    regions: Mapping[str, Any], selected: ArrayLike | None = None
) -> tuple[list[NDArray[np.bool_]], list[float]]:
    """Stars (selected) of each off part, and the areas of the parts: the off inputs of ``on_off_hess``."""
    selected = selected_stars(regions, selected)
    parts = tuple(regions.get("off_parts", ("off",)))
    return ([selected & np.asarray(regions[part], dtype=bool) for part in parts],
            [float(regions[f"{part}_area_deg2"]) for part in parts])

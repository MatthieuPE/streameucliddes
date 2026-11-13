"""Known streams from galstreams: tracks, windows and distances.

galstreams (Mateu 2023) holds the tracks of the known Milky Way streams. Each
track (``galstreams.Track6D``) has a ``track`` (ICRS SkyCoord, with distances),
a ``stream_frame`` (gala ``GreatCircleICRSFrame``) and ``InfoFlags``.

The distances are used as galstreams gives them. When galstreams has no
distance track for a stream (second digit of ``InfoFlags`` is 0), its
distances are the reported mean or central value: a single value (dm_min =
dm_max in the selection), or values between two ends.

The tracks of Ibata et al. 2024 (reference ``ibata2024``: 41 of the 141 in
galstreams 1.2.1) have no distance in galstreams: they are all at 1 kpc,
whatever their flag. Their distances are taken as unknown (NaN), and they are
only selected with ``add_ibata2024``.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import astropy.units as u
import healpy as hp
import numpy as np
import pandas as pd
from gala.coordinates.greatcircle import sph_midpoint

from streameucliddes.stream_frames import radec_to_phi12
from streameucliddes.unit_conversions import convert_kpc_to_DM

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:  # imported in load_galstreams only, so that the rest works without it
    import galstreams

LOGGER = logging.getLogger(__name__)

# Cache of the galstreams library (see load_galstreams), next to the catalogs (see
# euclid_catalog.DATA_DIR): the data/ folder of the repository, gitignored.
GALSTREAMS_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "galstreams"

IBATA2024_REFERENCE = "ibata2024"  # galstreams reference of the Ibata et al. 2024 tracks
# galstreams_dm_along_track(extrapolate=True): line fitted to the track points within this
# distance of an end (deg)
EXTRAPOLATION_BASELINE_DEG = 1.0

SELECTION_COLUMNS = [
    "track_name",
    "stream_name",
    "length_in_footprint_deg",
    "dm_min",
    "dm_max",
    "has_distance_track",
]


def load_galstreams(*, verbose: bool = False, use_cache: bool = True) -> galstreams.MWStreams:
    """galstreams library of the Milky Way stream tracks.

    One track per stream, the default one of galstreams (141 tracks in
    galstreams 1.2.1). Building it takes about 15 s. With ``use_cache``, it
    is built once and saved (pickle, about 80 MB) in ``GALSTREAMS_CACHE_DIR``,
    one file per galstreams version, then read back in about 0.1 s. A cache
    that cannot be read is rebuilt. The cache is a pickle: it is only meant
    to hold the files this function writes.

    Parameters
    ----------
    verbose : bool, optional
        Passed to ``galstreams.MWStreams``, when the library is built.
    use_cache : bool, optional
        Read the library from the cache, or build it and save it there.
        False: build it, without reading or writing the cache.

    Returns
    -------
    mws : galstreams.MWStreams
        Dictionary of the tracks (``galstreams.Track6D``), by track name.

    References
    ----------
    Mateu 2023, MNRAS 520, 5225.
    """
    import galstreams

    cache_path = GALSTREAMS_CACHE_DIR / f"mwstreams_galstreams-{galstreams.__version__}.pkl"
    if use_cache and cache_path.exists():
        try:
            with cache_path.open("rb") as file:
                return pickle.load(file)
        except (OSError, EOFError, AttributeError, ImportError, pickle.UnpicklingError) as error:
            LOGGER.warning("galstreams cache %s unreadable (%s): rebuilt.", cache_path, error)

    mws = galstreams.MWStreams(verbose=verbose)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = cache_path.with_suffix(".partial")
        with partial_path.open("wb") as file:
            pickle.dump(mws, file, protocol=pickle.HIGHEST_PROTOCOL)
        partial_path.replace(cache_path)  # a complete file, or none
    return mws


def galstreams_track_length_in_footprint(
    track: galstreams.Track6D,
    footprint_mask: NDArray[np.bool_],
    *,
    nest: bool,
    dm_range: tuple[float | None, float | None] | None = None,
) -> float:
    """Length of a galstreams track inside the survey footprint.

    Sum of the lengths of the track segments (between consecutive track
    points) whose midpoint is in the footprint and, with ``dm_range``, whose
    distance modulus is strictly within that range. A track without
    distances (Ibata et al. 2024) has no length within a ``dm_range``.

    Parameters
    ----------
    track : galstreams.Track6D
        One track of ``load_galstreams()``.
    footprint_mask : numpy.ndarray of bool
        HEALPix mask from
        ``catalogs.survey_footprint.survey_footprint_healpix_mask``, True
        outside.
    nest : bool
        HEALPix ordering of ``footprint_mask``.
    dm_range : tuple of float or None, optional
        ``(dm_min, dm_max)``: only count the segments in this range. None for
        a bound: no bound on that side.

    Returns
    -------
    length_deg : float
        Length in the footprint (deg).
    """
    lengths, kept, _ = _segments_in_footprint(track, footprint_mask, nest=nest, dm_range=dm_range)
    return float(lengths[kept].sum())


def select_galstreams_in_footprint(
    mws: galstreams.MWStreams,
    footprint_mask: NDArray[np.bool_],
    *,
    nest: bool,
    dm_range: tuple[float | None, float | None] = (14.0, None),
    min_length_deg: float = 1.0,
    add_ibata2024: bool = False,
) -> pd.DataFrame:
    """Known streams to search: the tracks crossing the footprint.

    A track is kept when its part inside the footprint, with a distance
    modulus strictly within ``dm_range``, is at least ``min_length_deg`` long
    (see ``galstreams_track_length_in_footprint``). The Ibata et al. 2024
    tracks, without distances, are only kept with ``add_ibata2024``, on their
    length inside the footprint alone.

    Parameters
    ----------
    mws : galstreams.MWStreams
        From ``load_galstreams``.
    footprint_mask : numpy.ndarray of bool
        HEALPix mask, True outside the footprint.
    nest : bool
        HEALPix ordering of ``footprint_mask``.
    dm_range : tuple of float or None, optional
        ``(dm_min, dm_max)`` along the track; None for a bound: no bound on
        that side. Default: dm > 14, no upper bound (user choice: wider than
        the dm 15 to 22 of the maps, so as not to miss a stream).
    min_length_deg : float, optional
        Minimum length in the footprint (deg). Proposal: 1 deg.
    add_ibata2024 : bool, optional
        Also select the Ibata et al. 2024 tracks (no distances: their
        ``dm_min`` and ``dm_max`` are NaN).

    Returns
    -------
    selection : pandas.DataFrame
        One row per kept track, the longest first: ``track_name``,
        ``stream_name``, ``length_in_footprint_deg``; ``dm_min``,
        ``dm_max`` of the part counted in that length; and
        ``has_distance_track``: False when the distance is only a mean value
        or unknown (see the module docstring).
    """
    rows = []
    for track_name, track in mws.items():
        is_ibata2024 = _is_ibata2024(track)
        if is_ibata2024 and not add_ibata2024:
            continue
        lengths, kept, dm = _segments_in_footprint(
            track, footprint_mask, nest=nest, dm_range=None if is_ibata2024 else dm_range
        )
        length = float(lengths[kept].sum())
        if kept.any() and length >= min_length_deg:
            rows.append(
                (
                    track_name,
                    track.stream_name,
                    length,
                    np.min(dm[kept]),
                    np.max(dm[kept]),
                    _has_distance_track(track),
                )
            )
    selection = pd.DataFrame(rows, columns=SELECTION_COLUMNS)
    return selection.sort_values("length_in_footprint_deg", ascending=False, ignore_index=True)


def galstreams_default_tracks(selection: pd.DataFrame, mws: galstreams.MWStreams) -> pd.DataFrame:
    """One track per stream of a selection: its default track, else its longest one.

    The default track of a stream is the one galstreams flags ``On`` in
    ``mws.summary``. When it is not in the selection (e.g. not enough of it
    in the footprint), the longest selected track of the stream is kept.

    Parameters
    ----------
    selection : pandas.DataFrame
        From ``select_galstreams_in_footprint`` (longest track first).
    mws : galstreams.MWStreams
        The library of the selection, from ``load_galstreams``.

    Returns
    -------
    tracks : pandas.DataFrame
        The rows of the tracks kept, in the order of ``selection``.
    """
    is_default = selection["track_name"].map(mws.summary["On"]).fillna(False).astype(bool)
    # the selection is sorted longest first: a stable sort keeps that order among the others
    preferred = selection.assign(is_default=is_default).sort_values("is_default", ascending=False, kind="stable")
    return preferred.drop_duplicates("stream_name").drop(columns="is_default").sort_index()


def galstreams_phi12_window(
    track: galstreams.Track6D,
    *,
    footprint_mask: NDArray[np.bool_] | None = None,
    nest: bool | None = None,
    padding_phi1_deg: float = 5.0,
    padding_phi2_deg: float = 2.0,
    min_phi2_half_width_deg: float = 5.0,
) -> tuple[float, float, float, float]:
    """phi1/phi2 window of the maps of a known stream, in its galstreams frame.

    Around the track, or, with ``footprint_mask``, around its part inside
    the footprint (the segments of ``galstreams_track_length_in_footprint``,
    without distance cut): where there are both data and the track. phi1
    covers it plus ``padding_phi1_deg`` on each side. phi2 is centred on it:
    its extent plus ``padding_phi2_deg`` on each side, and at least
    ``min_phi2_half_width_deg`` on each side of its centre.

    Parameters
    ----------
    track : galstreams.Track6D
        One track of ``load_galstreams()``.
    footprint_mask : numpy.ndarray of bool, optional
        HEALPix mask, True outside the footprint (needs ``nest``).
    nest : bool, optional
        HEALPix ordering of ``footprint_mask``.
    padding_phi1_deg, padding_phi2_deg : float, optional
        Margins around the track (deg).
    min_phi2_half_width_deg : float, optional
        Minimum half-width in phi2 (deg).

    Returns
    -------
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg).

    Raises
    ------
    ValueError
        ``footprint_mask`` without ``nest``, or no part of the track in the
        footprint.

    Notes
    -----
    Adapted from celestial_currents
    ``visualization.py::get_stream_phi12_bounds`` (its phi1 padding is 10
    deg, 5 deg here; same phi2 defaults, but its phi2 range always includes
    -5 to 5 deg, around phi2 = 0, not around the track). Its per-stream
    overrides are left out: pass another window instead.
    """
    phi1, phi2 = _track_phi12(track)
    if footprint_mask is not None:
        if nest is None:
            raise ValueError("A footprint_mask needs nest.")
        _, kept, _ = _segments_in_footprint(track, footprint_mask, nest=nest, dm_range=None)
        if not kept.any():
            raise ValueError(f"No part of the track {track.track_name} is in the footprint.")
        in_footprint = np.zeros(phi1.size, dtype=bool)  # both ends of the segments kept
        in_footprint[:-1] |= kept
        in_footprint[1:] |= kept
        phi1, phi2 = phi1[in_footprint], phi2[in_footprint]
    phi2_centre = (phi2.min() + phi2.max()) / 2
    phi2_half_width = max((phi2.max() - phi2.min()) / 2 + padding_phi2_deg, min_phi2_half_width_deg)
    return (
        float(phi1.min() - padding_phi1_deg),
        float(phi1.max() + padding_phi1_deg),
        float(phi2_centre - phi2_half_width),
        float(phi2_centre + phi2_half_width),
    )


def galstreams_dm_along_track(
    track: galstreams.Track6D, phi1: ArrayLike, *, extrapolate: bool = False
) -> NDArray[np.float64]:
    """Distance modulus of a galstreams track at given phi1.

    Interpolates linearly the distance moduli of the track points in phi1, in
    the track's frame. Outside the phi1 range of the track: NaN, or with
    ``extrapolate`` the line fitted to the track points within
    ``EXTRAPOLATION_BASELINE_DEG`` of the nearest end (to follow a possible
    extension of the stream). NaN for the Ibata et al. 2024 tracks (no
    distances). Constant when galstreams only has a mean distance (see the
    module docstring).

    Parameters
    ----------
    track : galstreams.Track6D
        One track of ``load_galstreams()``.
    phi1 : array_like
        phi1 in the track's ``stream_frame`` (deg).
    extrapolate : bool, optional
        Beyond the ends of the track, extrapolate the distance modulus
        linearly instead of NaN.

    Returns
    -------
    distance_modulus : numpy.ndarray
        Same shape as ``phi1``.
    """
    phi1 = np.asarray(phi1, dtype=float)
    track_phi1, _ = _track_phi12(track)
    order = np.argsort(track_phi1)
    track_phi1, track_dm = track_phi1[order], _track_distance_moduli(track)[order]
    dm = np.interp(phi1, track_phi1, track_dm, left=np.nan, right=np.nan)
    if not extrapolate or np.isnan(track_dm).any():
        return dm
    for end, beyond in ((track_phi1[0], phi1 < track_phi1[0]), (track_phi1[-1], phi1 > track_phi1[-1])):
        near_end = np.abs(track_phi1 - end) <= EXTRAPOLATION_BASELINE_DEG
        if near_end.sum() < 2:  # sparse track: its two points nearest to the end
            near_end = np.argsort(np.abs(track_phi1 - end))[:2]
        slope, intercept = np.polyfit(track_phi1[near_end], track_dm[near_end], 1)
        dm[beyond] = intercept + slope * phi1[beyond]
    return dm


def _is_ibata2024(track: galstreams.Track6D) -> bool:
    """Whether the track is one of Ibata et al. 2024, without distances in galstreams."""
    return str(track.ref) == IBATA2024_REFERENCE


def _has_distance_track(track: galstreams.Track6D) -> bool:
    """Whether galstreams has a distance track, not only a mean or no distance (InfoFlags)."""
    return not _is_ibata2024(track) and str(track.InfoFlags).zfill(4)[1] != "0"


def _track_distance_moduli(track: galstreams.Track6D) -> NDArray[np.float64]:
    """Distance modulus of each track point; NaN for the Ibata et al. 2024 tracks."""
    if _is_ibata2024(track):
        return np.full(len(track.track), np.nan)
    return np.asarray(convert_kpc_to_DM(track.track.distance.to_value(u.kpc)).value)


def _track_phi12(track: galstreams.Track6D) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(phi1, phi2) of the track points in the track's own frame (deg)."""
    return radec_to_phi12(track.track.ra.deg, track.track.dec.deg, track.stream_frame)


def _segments_in_footprint(
    track: galstreams.Track6D,
    footprint_mask: NDArray[np.bool_],
    *,
    nest: bool,
    dm_range: tuple[float | None, float | None] | None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Length (deg), kept flag and distance modulus of each track segment.

    A segment is kept when its midpoint is in the footprint and, with
    ``dm_range``, its distance modulus (mean of its two ends) is strictly
    within that range (None for a bound: no bound on that side; an unknown
    distance modulus, NaN, is never within a range).
    """
    points = track.track
    lengths = points[:-1].separation(points[1:]).deg
    midpoints = sph_midpoint(points[:-1], points[1:])
    footprint_mask = np.asarray(footprint_mask, dtype=bool)
    nside = hp.npix2nside(footprint_mask.size)
    pixels = hp.ang2pix(nside, midpoints.ra.deg, midpoints.dec.deg, lonlat=True, nest=nest)
    dm_points = _track_distance_moduli(track)
    dm = (dm_points[:-1] + dm_points[1:]) / 2
    kept = ~footprint_mask[pixels]
    if dm_range is not None:
        dm_low, dm_high = dm_range
        if dm_low is not None:
            kept &= dm > dm_low
        if dm_high is not None:
            kept &= dm < dm_high
        kept &= ~np.isnan(dm)  # unknown distance: never within a range
    return lengths, kept, dm

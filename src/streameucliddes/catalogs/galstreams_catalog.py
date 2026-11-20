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

The known streams of each Euclid footprint are listed by hand in a CSV file,
``data/known_streams/<footprint>.csv`` (``known_streams_path``, e.g.
``euclid_dr1_south``; local, gitignored with the rest of ``data/``): one row
per stream, ``stream_name`` and ``track_name`` (those of galstreams),
``observed`` (seen in the matched-filter maps: ``yes``, ``no``, ``unclear``,
or empty) and ``status`` (``masked`` or ``not_masked``), any other column
kept (``load_known_streams``). ``not_masked_known_streams`` gives the
``not_masked`` ones of every file, never masked around a candidate
(``analysis.build_masks.galstreams_mask_tracks``); ``observed_known_streams``
the ``yes`` ones, e.g. for ``CandidateInvestigation(always_masked_streams=...)``.
The notebook ``study_mask`` shows them on the maps, to fill the file.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import astropy.coordinates as coord
import astropy.units as u
import healpy as hp
import numpy as np
import pandas as pd
from gala.coordinates.greatcircle import sph_midpoint

from streameucliddes.stream_frames import (
    DEFAULT_MIN_PHI2_HALF_WIDTH_DEG,
    DEFAULT_PADDING_PHI1_DEG,
    DEFAULT_PADDING_PHI2_DEG,
    phi12_window_around,
    radec_to_phi12,
    stream_frame_from_endpoints,
)
from streameucliddes.unit_conversions import convert_DM_to_kpc, convert_kpc_to_DM

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:  # imported in load_galstreams only, so that the rest works without it
    import galstreams

LOGGER = logging.getLogger(__name__)

# Cache of the galstreams library (see load_galstreams), next to the catalogs (see
# euclid_catalog.DATA_DIR): the data/ folder of the repository, gitignored.
GALSTREAMS_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "galstreams"
# The known streams of each footprint, listed by hand (see load_known_streams)
KNOWN_STREAMS_DIR = Path(__file__).resolve().parents[3] / "data" / "known_streams"
KNOWN_STREAM_COLUMNS = ("stream_name", "track_name", "observed", "status")
OBSERVED_VALUES = ("yes", "no", "unclear")  # or empty: not looked at
MASKED_STATUS, NOT_MASKED_STATUS = "masked", "not_masked"

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


def load_galstreams(
    *, verbose: bool = False, use_cache: bool = True, all_tracks: bool = False
) -> galstreams.MWStreams:
    """galstreams library of the Milky Way stream tracks.

    By default one track per stream, the default one of galstreams (141
    tracks in galstreams 1.2.1); with ``all_tracks``, every track, the
    default ones and the others (217 in galstreams 1.2.1: e.g. ``ATLAS-I21``
    and ``ATLAS-I24`` of Ibata et al. 2021 and 2024 besides ``AAU-ATLAS-L21``).
    ``mws.summary`` lists every track either way. Building it takes about 15 s. With ``use_cache``, it
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
    all_tracks : bool, optional
        Every track (galstreams ``implement_Off``), not only the default one
        of each stream; cached in its own file.

    Returns
    -------
    mws : galstreams.MWStreams
        Dictionary of the tracks (``galstreams.Track6D``), by track name.

    References
    ----------
    Mateu 2023, MNRAS 520, 5225.
    """
    import galstreams

    suffix = "_all_tracks" if all_tracks else ""
    cache_path = GALSTREAMS_CACHE_DIR / f"mwstreams_galstreams-{galstreams.__version__}{suffix}.pkl"
    if use_cache and cache_path.exists():
        try:
            with cache_path.open("rb") as file:
                return pickle.load(file)
        except (OSError, EOFError, AttributeError, ImportError, pickle.UnpicklingError) as error:
            LOGGER.warning("galstreams cache %s unreadable (%s): rebuilt.", cache_path, error)

    mws = galstreams.MWStreams(verbose=verbose, implement_Off=all_tracks)
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
    padding_phi1_deg: float = DEFAULT_PADDING_PHI1_DEG,
    padding_phi2_deg: float = DEFAULT_PADDING_PHI2_DEG,
    min_phi2_half_width_deg: float = DEFAULT_MIN_PHI2_HALF_WIDTH_DEG,
) -> tuple[float, float, float, float]:
    """phi1/phi2 window of the maps of a known stream, in its galstreams frame.

    Around the track, or, with ``footprint_mask``, around its part inside
    the footprint (the segments of ``galstreams_track_length_in_footprint``,
    without distance cut): where there are both data and the track
    (``stream_frames.phi12_window_around`` of these points).

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
    ``visualization.py::get_stream_phi12_bounds`` (see
    ``stream_frames.phi12_window_around``). Its per-stream overrides are left
    out: pass another window instead.
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
    return phi12_window_around(
        phi1, phi2, padding_phi1_deg=padding_phi1_deg, padding_phi2_deg=padding_phi2_deg,
        min_phi2_half_width_deg=min_phi2_half_width_deg,
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


def galstreams_phi2_along_track(track: galstreams.Track6D, phi1: ArrayLike) -> NDArray[np.float64]:
    """phi2 of a galstreams track at given phi1, in its own frame.

    Interpolates linearly the phi2 of the track points in phi1; beyond the
    ends of the track, the phi2 of the nearest end.

    Parameters
    ----------
    track : galstreams.Track6D
        One track of ``load_galstreams()``.
    phi1 : array_like
        phi1 in the track's ``stream_frame`` (deg).

    Returns
    -------
    phi2 : numpy.ndarray
        Same shape as ``phi1`` (deg).
    """
    track_phi1, track_phi2 = _track_phi12(track)
    order = np.argsort(track_phi1)
    return np.interp(np.asarray(phi1, dtype=float), track_phi1[order], track_phi2[order])


def galstreams_track_in_own_frame(track: galstreams.Track6D) -> SimpleNamespace:
    """A galstreams track in the great-circle frame through its own two ends.

    galstreams can give several tracks of one stream the same frame (e.g.
    ``AAU-ATLAS-L21`` and ``AAU-AliqaUma-L21``, around one pole): a track
    bent away from that great circle is then tilted in it. In the frame
    through its first and last points (``stream_frames.stream_frame_from_endpoints``),
    its ends are at phi2 = 0 and the regions along it are bands near phi2 = 0.
    The points, distances and flags are unchanged.

    Parameters
    ----------
    track : galstreams.Track6D
        One track of ``load_galstreams()``.

    Returns
    -------
    track : types.SimpleNamespace
        ``track``, ``stream_frame`` (the new frame), ``InfoFlags``,
        ``track_name``, ``stream_name`` and ``ref``: what the functions of
        this module and ``analysis.stream_optimization`` read of a track.
    """
    points = track.track
    first, last = (float(points.ra.deg[0]), float(points.dec.deg[0])), (float(points.ra.deg[-1]),
                                                                       float(points.dec.deg[-1]))
    return SimpleNamespace(track=points, stream_frame=stream_frame_from_endpoints(first, last),
                           InfoFlags=track.InfoFlags, track_name=track.track_name, stream_name=track.stream_name,
                           ref=track.ref)


def track_from_points(
    track_name: str,
    ra: ArrayLike,
    dec: ArrayLike,
    distance_modulus: ArrayLike,
    *,
    stream_name: str | None = None,
    ref: str = "custom",
) -> SimpleNamespace:
    """A track given by its points, as a galstreams track: e.g. a published fit not in galstreams.

    The points go along the stream, in order; their frame is the great
    circle through the first and last ones (as ``galstreams_track_in_own_frame``).
    It has what the functions of this module and
    ``analysis.stream_optimization`` read of a track, with a distance track
    (``InfoFlags`` "1111").

    Parameters
    ----------
    track_name : str
        Name of the track.
    ra, dec : array_like
        Points along the track (deg), in order, at least two.
    distance_modulus : array_like
        Distance modulus at each point.
    stream_name : str, optional
        Name of the stream, e.g. for ``isochrone_parameters.known_stream_isochrone_parameters``.
        Default: ``track_name`` before its last hyphen.
    ref : str, optional
        Reference of the track.

    Returns
    -------
    track : types.SimpleNamespace
        ``track`` (ICRS SkyCoord with distances), ``stream_frame``,
        ``InfoFlags``, ``track_name``, ``stream_name`` and ``ref``.

    Raises
    ------
    ValueError
        Fewer than two points, or arrays of different lengths.
    """
    ra, dec = np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
    distance_modulus = np.asarray(distance_modulus, dtype=float)
    if not ra.size == dec.size == distance_modulus.size or ra.size < 2:
        raise ValueError(f"Need at least two points and as many ra, dec and distance moduli, not {ra.size}, "
                         f"{dec.size}, {distance_modulus.size}.")
    points = coord.SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=convert_DM_to_kpc(distance_modulus))
    return SimpleNamespace(
        track=points, stream_frame=stream_frame_from_endpoints((ra[0], dec[0]), (ra[-1], dec[-1])), InfoFlags="1111",
        track_name=track_name, stream_name=track_name.rsplit("-", 1)[0] if stream_name is None else stream_name,
        ref=ref,
    )


def galstreams_width_phi2(track_name: str, mws: galstreams.MWStreams) -> float:
    """Width of a stream in phi2 from the galstreams summary (``width_phi2``, deg); NaN when unknown.

    Read from ``mws.summary`` (the ``track_width`` of the tracks is about 0
    in galstreams 1.2.1); galstreams gives -1 when unknown.

    Parameters
    ----------
    track_name : str
        A track name of ``mws``.
    mws : galstreams.MWStreams
        From ``load_galstreams``.

    Returns
    -------
    width_deg : float
    """
    width = float(mws.summary.loc[track_name, "width_phi2"])
    return width if width > 0 else np.nan


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


# ===================================================
# The known streams of each footprint, listed by hand
# ===================================================


def known_streams_path(footprint_name: str) -> Path:
    """CSV file of the known streams of a footprint, e.g. ``"euclid_dr1_south"`` (data name and subset)."""
    return KNOWN_STREAMS_DIR / f"{footprint_name}.csv"


def load_known_streams(path: str | Path) -> pd.DataFrame:
    """The known streams of a footprint, as listed by hand.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file, e.g. ``known_streams_path("euclid_dr1_south")``.

    Returns
    -------
    streams : pandas.DataFrame
        The rows of the file: ``KNOWN_STREAM_COLUMNS`` and any other column.

    Raises
    ------
    FileNotFoundError
        No such file.
    ValueError
        A column of ``KNOWN_STREAM_COLUMNS`` missing, an ``observed`` not in
        ``OBSERVED_VALUES`` (nor empty), or a ``status`` neither
        ``masked`` nor ``not_masked``.
    """
    path = Path(path)
    streams = pd.read_csv(path)
    missing = [column for column in KNOWN_STREAM_COLUMNS if column not in streams.columns]
    if missing:
        raise ValueError(f"{path} has no column {missing}; the columns are {KNOWN_STREAM_COLUMNS}.")
    unknown = sorted(set(streams["observed"].dropna()) - set(OBSERVED_VALUES))
    if unknown:
        raise ValueError(f"{path}: 'observed' {unknown} not among {OBSERVED_VALUES} (or empty).")
    unknown = sorted(set(streams["status"].fillna("(empty)")) - {MASKED_STATUS, NOT_MASKED_STATUS})
    if unknown:
        raise ValueError(f"{path}: 'status' {unknown} not among {(MASKED_STATUS, NOT_MASKED_STATUS)}.")
    return streams


def not_masked_known_streams(folder: str | Path | None = None) -> tuple[str, ...] | None:
    """Track and stream names of status ``not_masked`` in the files of every footprint.

    Parameters
    ----------
    folder : str or pathlib.Path, optional
        Default: ``KNOWN_STREAMS_DIR``.

    Returns
    -------
    names : tuple of str or None
        Sorted. None when the folder has no CSV file (then
        ``analysis.build_masks.NEVER_MASKED_STREAMS`` applies).
    """
    return _known_stream_names(folder, "status", NOT_MASKED_STATUS)


def observed_known_streams(folder: str | Path | None = None) -> tuple[str, ...]:
    """Track and stream names observed (``observed`` ``yes``) in the files of every footprint.

    Parameters
    ----------
    folder : str or pathlib.Path, optional
        Default: ``KNOWN_STREAMS_DIR``.

    Returns
    -------
    names : tuple of str
        Sorted; empty without a file.
    """
    return _known_stream_names(folder, "observed", "yes") or ()


def _known_stream_names(folder: str | Path | None, column: str, value: str) -> tuple[str, ...] | None:
    """Track and stream names of the rows whose ``column`` is ``value``, in every file; None without a file."""
    folder = Path(KNOWN_STREAMS_DIR if folder is None else folder)
    paths = sorted(folder.glob("*.csv")) if folder.is_dir() else []
    if not paths:
        return None
    names = set()
    for path in paths:
        streams = load_known_streams(path)
        rows = streams[streams[column] == value]
        names.update(rows["stream_name"])
        names.update(rows["track_name"])
    return tuple(sorted(names))

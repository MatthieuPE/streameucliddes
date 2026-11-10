"""Known streams from galstreams: tracks, windows and distances.

galstreams (Mateu 2023) holds the tracks of the known Milky Way streams. Each
track (``galstreams.Track6D``) has a ``track`` (ICRS SkyCoord, with distances),
a ``stream_frame`` (gala ``GreatCircleICRSFrame``) and ``InfoFlags``.

The distances are used as galstreams gives them. When galstreams has no
distance track for a stream (second digit of ``InfoFlags`` is 0), its
distances are the reported mean or central value: a single value (dm_min =
dm_max in the selection), or values between two ends. The tracks of Ibata
et al. 2024 (41 of the 141 in galstreams 1.2.1) are all at 1 kpc (distance
modulus 10), whatever their flag: no distance, outside the distance moduli of
the searches.
"""

from __future__ import annotations

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

SELECTION_COLUMNS = [
    "track_name",
    "stream_name",
    "length_in_footprint_deg",
    "dm_min",
    "dm_max",
    "has_distance_track",
]


def load_galstreams(*, verbose: bool = False) -> galstreams.MWStreams:
    """galstreams library of the Milky Way stream tracks.

    One track per stream, the default one of galstreams (141 tracks in
    galstreams 1.2.1). Slow (about 20 s): load it once and pass it to the
    other functions.

    Parameters
    ----------
    verbose : bool, optional
        Passed to ``galstreams.MWStreams``.

    Returns
    -------
    mws : galstreams.MWStreams
        Dictionary of the tracks (``galstreams.Track6D``), by track name.

    References
    ----------
    Mateu 2023, MNRAS 520, 5225.
    """
    import galstreams

    return galstreams.MWStreams(verbose=verbose)


def galstreams_track_length_in_footprint(
    track: galstreams.Track6D,
    footprint_mask: NDArray[np.bool_],
    *,
    nest: bool,
    dm_range: tuple[float, float] | None = None,
) -> float:
    """Length of a galstreams track inside the survey footprint.

    Sum of the lengths of the track segments (between consecutive track
    points) whose midpoint is in the footprint and, with ``dm_range``, whose
    distance modulus is strictly within that range.

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
    dm_range : tuple of float, optional
        ``(dm_min, dm_max)``: only count the segments in this range.

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
    dm_range: tuple[float, float] = (15.0, 22.0),
    min_length_deg: float = 1.0,
) -> pd.DataFrame:
    """Known streams to search: the tracks crossing the footprint.

    A track is kept when its part inside the footprint, with a distance
    modulus strictly within ``dm_range``, is at least ``min_length_deg`` long
    (see ``galstreams_track_length_in_footprint``).

    Parameters
    ----------
    mws : galstreams.MWStreams
        From ``load_galstreams``.
    footprint_mask : numpy.ndarray of bool
        HEALPix mask, True outside the footprint.
    nest : bool
        HEALPix ordering of ``footprint_mask``.
    dm_range : tuple of float, optional
        ``(dm_min, dm_max)`` along the track. Default: 15 < dm < 22 (plan,
        Section 1).
    min_length_deg : float, optional
        Minimum length in the footprint (deg). Proposal: 1 deg.

    Returns
    -------
    selection : pandas.DataFrame
        One row per kept track, the longest first: ``track_name``,
        ``stream_name``, ``length_in_footprint_deg``; ``dm_min``,
        ``dm_max`` of the part counted in that length; and
        ``has_distance_track``: False when the distance is only a mean value
        (see the module docstring).
    """
    rows = []
    for track_name, track in mws.items():
        lengths, kept, dm = _segments_in_footprint(track, footprint_mask, nest=nest, dm_range=dm_range)
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


def galstreams_phi12_window(
    track: galstreams.Track6D,
    *,
    padding_phi1_deg: float = 10.0,
    padding_phi2_deg: float = 2.0,
    min_phi2_half_width_deg: float = 5.0,
) -> tuple[float, float, float, float]:
    """phi1/phi2 window of the maps of a known stream, in its galstreams frame.

    phi1 covers the track, plus ``padding_phi1_deg`` on each side. phi2
    covers the track plus ``padding_phi2_deg``, and at least
    ``-min_phi2_half_width_deg`` to ``+min_phi2_half_width_deg``.

    Parameters
    ----------
    track : galstreams.Track6D
        One track of ``load_galstreams()``.
    padding_phi1_deg, padding_phi2_deg : float, optional
        Margins around the track (deg).
    min_phi2_half_width_deg : float, optional
        Minimum half-width in phi2 (deg).

    Returns
    -------
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg).

    Notes
    -----
    Adapted from celestial_currents
    ``visualization.py::get_stream_phi12_bounds`` (same defaults). Its
    per-stream overrides are left out: pass another window instead.
    """
    phi1, phi2 = _track_phi12(track)
    return (
        float(phi1.min() - padding_phi1_deg),
        float(phi1.max() + padding_phi1_deg),
        float(min(phi2.min() - padding_phi2_deg, -min_phi2_half_width_deg)),
        float(max(phi2.max() + padding_phi2_deg, min_phi2_half_width_deg)),
    )


def galstreams_dm_along_track(track: galstreams.Track6D, phi1: ArrayLike) -> NDArray[np.float64]:
    """Distance modulus of a galstreams track at given phi1.

    Interpolates linearly the distance moduli of the track points in phi1, in
    the track's frame. NaN outside the phi1 range of the track (no
    extrapolation). Constant when galstreams only has a mean distance (see
    the module docstring).

    Parameters
    ----------
    track : galstreams.Track6D
        One track of ``load_galstreams()``.
    phi1 : array_like
        phi1 in the track's ``stream_frame`` (deg).

    Returns
    -------
    distance_modulus : numpy.ndarray
        Same shape as ``phi1``.
    """
    phi1 = np.asarray(phi1, dtype=float)
    dm = _track_distance_moduli(track)
    track_phi1, _ = _track_phi12(track)
    order = np.argsort(track_phi1)
    return np.interp(phi1, track_phi1[order], dm[order], left=np.nan, right=np.nan)


def _has_distance_track(track: galstreams.Track6D) -> bool:
    """Whether galstreams has a distance track, not only a mean distance (InfoFlags)."""
    return str(track.InfoFlags).zfill(4)[1] != "0"


def _track_distance_moduli(track: galstreams.Track6D) -> NDArray[np.float64]:
    """Distance modulus of each track point."""
    return np.asarray(convert_kpc_to_DM(track.track.distance.to_value(u.kpc)).value)


def _track_phi12(track: galstreams.Track6D) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(phi1, phi2) of the track points in the track's own frame (deg)."""
    return radec_to_phi12(track.track.ra.deg, track.track.dec.deg, track.stream_frame)


def _segments_in_footprint(
    track: galstreams.Track6D,
    footprint_mask: NDArray[np.bool_],
    *,
    nest: bool,
    dm_range: tuple[float, float] | None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Length (deg), kept flag and distance modulus of each track segment.

    A segment is kept when its midpoint is in the footprint and, with
    ``dm_range``, its distance modulus (mean of its two ends) is strictly
    within that range.
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
        kept &= (dm > dm_range[0]) & (dm < dm_range[1])
    return lengths, kept, dm

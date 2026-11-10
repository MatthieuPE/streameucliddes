"""Known streams from galstreams: tracks, windows and distances.

galstreams (Mateu 2023) holds the tracks of the known Milky Way streams. It
is not installed in the streameuclid environment yet: it is a new dependency
of phase 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:  # galstreams: new dependency of phase 3, not installed yet
    import galstreams


def load_galstreams(*, verbose: bool = False) -> galstreams.MWStreams:
    """galstreams library of the Milky Way stream tracks.

    Slow: load it once and pass it to the other functions.

    Parameters
    ----------
    verbose : bool, optional
        Passed to ``galstreams.MWStreams``.

    Returns
    -------
    mws : galstreams.MWStreams
        Tracks by name. Each has a ``track`` (SkyCoord, with distances) and a
        ``stream_frame``.

    Notes
    -----
    Code to reuse: ``galstreams.MWStreams(verbose=False)``, as in
    celestial_currents ``notebooks/create_projections.ipynb``.

    References
    ----------
    Mateu 2023, MNRAS 520, 5225.
    """
    raise NotImplementedError


def galstreams_track_length_in_footprint(
    track: galstreams.Track6D,
    footprint_mask: NDArray[np.bool_],
    *,
    nest: bool,
    dm_range: tuple[float, float] | None = None,
) -> float:
    """Length of a galstreams track inside the survey footprint.

    Sum of the great-circle lengths of the track segments whose midpoint is
    in the footprint (and, with ``dm_range``, whose distance modulus is in
    that range).

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

    Notes
    -----
    Code to reuse: celestial_currents
    ``visualization.py::_find_overlapping_streams`` (track points inside a
    region).
    """
    raise NotImplementedError


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
    modulus within ``dm_range``, is at least ``min_length_deg`` long.

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
        One row per kept track: ``track_name``, ``stream_name``,
        ``length_in_footprint_deg``, and ``dm_min``, ``dm_max`` of the track
        in the footprint.

    Notes
    -----
    Code to reuse: celestial_currents
    ``visualization.py::_find_overlapping_streams``;
    ``galstreams_track_length_in_footprint``. galstreams can hold several
    tracks of one stream: check which ones ``MWStreams`` loads by default.
    """
    raise NotImplementedError


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
    Code to reuse: celestial_currents
    ``visualization.py::get_stream_phi12_bounds`` (same defaults). Its
    per-stream overrides are left out: pass another window instead.
    """
    raise NotImplementedError


def galstreams_dm_along_track(
    track: galstreams.Track6D, phi1: ArrayLike
) -> NDArray[np.float64]:
    """Distance modulus of a galstreams track at given phi1.

    Interpolates the distance moduli of the track points in phi1, in the
    track's frame. NaN outside the phi1 range of the track (no
    extrapolation), and when the track has no distances.

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

    Notes
    -----
    Code to reuse: none; ``unit_conversions.convert_kpc_to_DM`` for the
    distances of the track.
    """
    raise NotImplementedError

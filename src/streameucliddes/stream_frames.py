"""Stream frames: rotation matrices, and (ra, dec) <-> (phi1, phi2).

A stream frame is a 3x3 rotation matrix ``R`` that maps ICRS Cartesian unit
vectors to the stream frame: ``v_stream = R @ v_icrs``. Its rows are the x, y
and z axes of the stream frame, in ICRS. phi1 = phi2 = 0 is the x axis;
phi2 = 90 deg is the z axis (the pole). Angles are in degrees.

Maps in a stream frame use the phi1/phi2 grid of ``phi12_bin_centres``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.coordinates as coord
import astropy.units as u
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def rotation_matrix_from_astropy_frame(
    frame: coord.BaseCoordinateFrame,
) -> NDArray[np.float64]:
    """Rotation matrix of an astropy stream frame.

    Used for the galstreams frames (``track.stream_frame``), so that the
    known-stream maps use the same phi1/phi2 as galstreams. The points
    (phi1, phi2) = (0, 0), (90, 0) and (0, 90) of the frame, taken to ICRS,
    are the x, y and z axes: the rows of ``R``.

    Parameters
    ----------
    frame : astropy.coordinates.BaseCoordinateFrame
        Stream frame, as an instance: e.g. the ``stream_frame`` of a
        galstreams track (a gala ``GreatCircleICRSFrame``). Any astropy frame
        whose transform to ICRS is a rotation works (e.g. ``Galactic()``).

    Returns
    -------
    rotation_matrix : numpy.ndarray, shape (3, 3)
        ``R`` such that ``R @ v_icrs`` is the Cartesian vector in the stream
        frame.

    References
    ----------
    Mateu 2023, MNRAS 520, 5225 (galstreams).
    """
    axes = coord.UnitSphericalRepresentation(
        lon=[0.0, 90.0, 0.0] * u.deg, lat=[0.0, 0.0, 90.0] * u.deg
    )
    axes_icrs = frame.realize_frame(axes).transform_to(coord.ICRS())
    return np.array(axes_icrs.cartesian.xyz.value).T


def rotation_matrix_from_endpoints(
    ra_1: float, dec_1: float, ra_2: float, dec_2: float
) -> NDArray[np.float64]:
    """Rotation matrix of the great circle through two points.

    The pole is ``v1 x v2`` (normalised), with ``v1`` and ``v2`` the unit
    vectors of the two points. phi1 = phi2 = 0 is their midpoint, ``v1 + v2``
    (normalised). The y axis completes the right-handed frame (``z x x``), so
    phi1 increases from the first point to the second. For a stream or a
    candidate without a galstreams frame.

    Parameters
    ----------
    ra_1, dec_1 : float
        First point (deg).
    ra_2, dec_2 : float
        Second point (deg).

    Returns
    -------
    rotation_matrix : numpy.ndarray, shape (3, 3)
        ``R`` such that ``R @ v_icrs`` is the Cartesian vector in the stream
        frame.

    Raises
    ------
    ValueError
        The two points are the same or antipodal: no unique great circle.

    Notes
    -----
    Adapted from celestial_currents
    ``frame_utils.py::rotation_matrix_from_endpoints``, with floats instead of
    SkyCoord endpoints.
    """
    v1, v2 = _unit_vectors([ra_1, ra_2], [dec_1, dec_2])
    z = np.cross(v1, v2)
    x = v1 + v2
    if np.linalg.norm(z) < 1e-12 or np.linalg.norm(x) < 1e-12:
        raise ValueError(
            f"Points ({ra_1}, {dec_1}) and ({ra_2}, {dec_2}) are the same or "
            "antipodal: they define no unique great circle."
        )
    z = z / np.linalg.norm(z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([x, y, z])


def radec_to_phi12(
    ra: ArrayLike, dec: ArrayLike, rotation_matrix: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(ra, dec) to (phi1, phi2) in a stream frame.

    Parameters
    ----------
    ra, dec : array_like
        ICRS coordinates (deg). Broadcast together.
    rotation_matrix : numpy.ndarray, shape (3, 3)
        Stream frame (see the module docstring).

    Returns
    -------
    phi1 : numpy.ndarray
        Longitude in the stream frame (deg), in [-180, 180). Same shape as
        the broadcast ``ra`` and ``dec``.
    phi2 : numpy.ndarray
        Latitude in the stream frame (deg), in [-90, 90].

    Notes
    -----
    Adapted from celestial_currents ``frame_utils.py::apply_rotation``, with
    ra/dec arrays instead of a SkyCoord.
    """
    vectors = _unit_vectors(ra, dec) @ np.asarray(rotation_matrix).T  # R @ v, for each v
    phi1, phi2 = _lonlat(vectors)
    return (phi1 + 180.0) % 360.0 - 180.0, phi2


def phi12_to_radec(
    phi1: ArrayLike, phi2: ArrayLike, rotation_matrix: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(phi1, phi2) in a stream frame to (ra, dec).

    Parameters
    ----------
    phi1, phi2 : array_like
        Coordinates in the stream frame (deg). Broadcast together.
    rotation_matrix : numpy.ndarray, shape (3, 3)
        Stream frame (see the module docstring).

    Returns
    -------
    ra : numpy.ndarray
        ICRS right ascension (deg), in [0, 360). Same shape as the broadcast
        ``phi1`` and ``phi2``.
    dec : numpy.ndarray
        ICRS declination (deg), in [-90, 90].

    Notes
    -----
    Adapted from celestial_currents ``frame_utils.py::phi12_to_radec``: the
    transpose of ``R`` takes the stream frame back to ICRS.
    """
    vectors = _unit_vectors(phi1, phi2) @ np.asarray(rotation_matrix)  # R.T @ v, for each v
    ra, dec = _lonlat(vectors)
    return ra % 360.0, dec


def _unit_vectors(lon: ArrayLike, lat: ArrayLike) -> NDArray[np.float64]:
    """Cartesian unit vectors of (lon, lat) in degrees: shape (..., 3)."""
    lon, lat = np.broadcast_arrays(np.radians(lon), np.radians(lat))
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


def _lonlat(vectors: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(lon, lat) in degrees of unit vectors of shape (..., 3); lon in (-180, 180]."""
    x, y, z = np.moveaxis(vectors, -1, 0)
    return np.degrees(np.arctan2(y, x)), np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))


def phi12_bin_centres(
    window: tuple[float, float, float, float], bin_size_deg: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bin centres of a phi1/phi2 grid.

    Bins of ``bin_size_deg`` from the minimum of the window, keeping those
    whose centre is inside the window: ``numpy.arange(min + bin_size_deg / 2,
    max, bin_size_deg)`` on each axis. When the bin size divides the window,
    the bins cover it exactly; otherwise the last bin ends within half a bin
    of the maximum.

    Parameters
    ----------
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg).
    bin_size_deg : float
        Bin size (deg), the same in phi1 and phi2.

    Returns
    -------
    phi1_centres, phi2_centres : numpy.ndarray
        Bin centres along each axis (deg). The bin edges are the centres
        minus and plus ``bin_size_deg / 2``.

    Raises
    ------
    ValueError
        The window is narrower than half a bin along an axis: no bin centre
        inside.
    """
    phi1_min, phi1_max, phi2_min, phi2_max = window
    half_bin = bin_size_deg / 2
    phi1_centres = np.arange(phi1_min + half_bin, phi1_max, bin_size_deg)
    phi2_centres = np.arange(phi2_min + half_bin, phi2_max, bin_size_deg)
    if phi1_centres.size == 0 or phi2_centres.size == 0:
        raise ValueError(f"Window {window} narrower than half a bin of {bin_size_deg} deg.")
    return phi1_centres, phi2_centres

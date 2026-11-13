"""Stream frames: (ra, dec) <-> (phi1, phi2), and the phi1/phi2 grid of the maps.

A stream frame is an astropy frame, in practice a gala ``GreatCircleICRSFrame``:
the ``stream_frame`` of a galstreams track, or, for another stream,
``gala.coordinates.GreatCircleICRSFrame.from_endpoints`` (or ``from_pole_ra0``).
The transformations are those of astropy and gala; the functions below only
take and return numpy arrays in degrees, as in the catalogs.

Maps in a stream frame use the phi1/phi2 grid of ``phi12_bin_centres``.

A window is a tuple ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg): the
window of the maps, and also the on region and the off bands of a stream
candidate. ``phi12_window_around`` puts a window around points (a galstreams
track, two endpoints); ``in_phi12_window`` selects stars in a window;
``phi12_window_outline`` gives its outline (to the sky with
``phi12_to_radec``); ``phi12_window_radec_box`` the (ra, dec) box of the
stars to read. A stream candidate gets its frame from two (ra, dec) points
along it (``stream_frame_from_endpoints``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.coordinates as coord
import astropy.units as u
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

# Window around points (phi12_window_around): margins and minimum half-width in phi2 (deg).
# Adapted from celestial_currents visualization.py::get_stream_phi12_bounds (its phi1 padding is 10 deg).
DEFAULT_PADDING_PHI1_DEG = 5.0
DEFAULT_PADDING_PHI2_DEG = 2.0
DEFAULT_MIN_PHI2_HALF_WIDTH_DEG = 5.0


def radec_to_phi12(
    ra: ArrayLike, dec: ArrayLike, stream_frame: coord.BaseCoordinateFrame
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(ra, dec) to (phi1, phi2) in a stream frame.

    Parameters
    ----------
    ra, dec : array_like
        ICRS coordinates (deg). Broadcast together.
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Stream frame, e.g. the ``stream_frame`` of a galstreams track (a gala
        ``GreatCircleICRSFrame``).

    Returns
    -------
    phi1 : numpy.ndarray
        Longitude in the stream frame (deg), in [-180, 180). Same shape as
        the broadcast ``ra`` and ``dec``.
    phi2 : numpy.ndarray
        Latitude in the stream frame (deg), in [-90, 90].
    """
    ra, dec = np.broadcast_arrays(ra, dec)
    stars = coord.SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    in_frame = stars.transform_to(stream_frame).represent_as(coord.UnitSphericalRepresentation)
    return in_frame.lon.wrap_at(180 * u.deg).deg, in_frame.lat.deg


def phi12_to_radec(
    phi1: ArrayLike, phi2: ArrayLike, stream_frame: coord.BaseCoordinateFrame
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(phi1, phi2) in a stream frame to (ra, dec).

    Parameters
    ----------
    phi1, phi2 : array_like
        Coordinates in the stream frame (deg). Broadcast together.
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Stream frame, e.g. the ``stream_frame`` of a galstreams track (a gala
        ``GreatCircleICRSFrame``).

    Returns
    -------
    ra : numpy.ndarray
        ICRS right ascension (deg), in [0, 360). Same shape as the broadcast
        ``phi1`` and ``phi2``.
    dec : numpy.ndarray
        ICRS declination (deg), in [-90, 90].
    """
    phi1, phi2 = np.broadcast_arrays(phi1, phi2)
    positions = coord.UnitSphericalRepresentation(lon=phi1 * u.deg, lat=phi2 * u.deg)
    icrs = stream_frame.realize_frame(positions).transform_to(coord.ICRS())
    return icrs.ra.deg, icrs.dec.deg


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


# ===================================================
# Windows
# ===================================================


def phi12_window_around(
    phi1: ArrayLike,
    phi2: ArrayLike,
    *,
    padding_phi1_deg: float = DEFAULT_PADDING_PHI1_DEG,
    padding_phi2_deg: float = DEFAULT_PADDING_PHI2_DEG,
    min_phi2_half_width_deg: float = DEFAULT_MIN_PHI2_HALF_WIDTH_DEG,
) -> tuple[float, float, float, float]:
    """phi1/phi2 window around points, e.g. a track or two endpoints.

    phi1 covers the points plus ``padding_phi1_deg`` on each side. phi2 is
    centred on them: their extent plus ``padding_phi2_deg`` on each side, and
    at least ``min_phi2_half_width_deg`` on each side of their centre.

    Parameters
    ----------
    phi1, phi2 : array_like
        Points in the stream frame (deg).
    padding_phi1_deg, padding_phi2_deg : float, optional
        Margins around the points (deg).
    min_phi2_half_width_deg : float, optional
        Minimum half-width in phi2 (deg).

    Returns
    -------
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg).

    Notes
    -----
    Adapted from celestial_currents
    ``visualization.py::get_stream_phi12_bounds`` (its phi1 padding is 10
    deg, 5 deg here; same phi2 defaults, but its phi2 range always includes
    -5 to 5 deg, around phi2 = 0, not around the points). Used by
    ``catalogs.galstreams_catalog.galstreams_phi12_window``.
    """
    phi1, phi2 = np.asarray(phi1, dtype=float), np.asarray(phi2, dtype=float)
    phi2_centre = (phi2.min() + phi2.max()) / 2
    phi2_half_width = max((phi2.max() - phi2.min()) / 2 + padding_phi2_deg, min_phi2_half_width_deg)
    return (
        float(phi1.min() - padding_phi1_deg),
        float(phi1.max() + padding_phi1_deg),
        float(phi2_centre - phi2_half_width),
        float(phi2_centre + phi2_half_width),
    )


def in_phi12_window(
    phi1: ArrayLike, phi2: ArrayLike, window: tuple[float, float, float, float]
) -> NDArray[np.bool_]:
    """Points inside a phi1/phi2 window, edges included.

    Parameters
    ----------
    phi1, phi2 : array_like
        Points in the stream frame (deg), e.g. the stars of
        ``PreparedSearch.stars``.
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg), e.g. the on region
        of a stream candidate.

    Returns
    -------
    mask : numpy.ndarray of bool
    """
    raise NotImplementedError


def phi12_window_outline(
    window: tuple[float, float, float, float], *, n_points_per_edge: int = 50
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Outline of a phi1/phi2 window, sampled along its edges.

    Sampled, so that ``phi12_to_radec`` of it is the outline on a sky map
    (the edges are not straight in ra/dec).

    Parameters
    ----------
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg).
    n_points_per_edge : int, optional
        Points along each of the four edges.

    Returns
    -------
    phi1, phi2 : numpy.ndarray
        (deg), a closed line: the last point is the first.
    """
    raise NotImplementedError


def phi12_window_radec_box(
    stream_frame: coord.BaseCoordinateFrame,
    window: tuple[float, float, float, float],
    *,
    margin_deg: float = 0.5,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """(ra, dec) box containing a phi1/phi2 window, to read its stars.

    The outline of the window (``phi12_window_outline``) taken to the sky
    (``phi12_to_radec``); the box is its min/max plus ``margin_deg``. A box
    crossing ra = 0/360 is returned with ``ra_min > ra_max`` and a warning.

    Parameters
    ----------
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Frame of the window.
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg).
    margin_deg : float, optional
        Margin added on each side of the box (deg).

    Returns
    -------
    ra_range, dec_range : tuple of float
        ``(ra_min, ra_max)``, ``(dec_min, dec_max)`` (deg), e.g. for
        ``catalogs.euclid_catalog.get_euclid_stars``.

    Notes
    -----
    Idea of celestial_currents ``frame_utils.phi12_bounds_to_radec_bounds``.
    """
    raise NotImplementedError


# ===================================================
# Frame of a stream candidate, from two endpoints
# ===================================================


def stream_frame_from_endpoints(
    endpoint_1: tuple[float, float], endpoint_2: tuple[float, float]
) -> coord.BaseCoordinateFrame:
    """Great-circle frame through two (ra, dec) points, phi1 along the arc.

    gala ``GreatCircleICRSFrame.from_endpoints``, with its default origin: the
    midpoint of the arc is at (phi1, phi2) = (0, 0), ``endpoint_1`` at
    negative phi1 and ``endpoint_2`` at positive phi1 (checked with gala
    1.10.1), both at phi2 = 0. The great circle on a sky map is
    ``phi12_to_radec(phi1, 0)``.

    Parameters
    ----------
    endpoint_1, endpoint_2 : tuple of float
        (ra, dec) of two points along the stream (deg).

    Returns
    -------
    stream_frame : gala.coordinates.GreatCircleICRSFrame

    Notes
    -----
    celestial_currents ``frame_utils.stream_frame_from_endpoints`` is not
    used: it builds a ``SkyOffsetFrame`` with ``rotation=0``, which ignores
    the pole, so its phi1 is not along the stream.

    References
    ----------
    Price-Whelan 2017, JOSS 2, 388 (gala).
    """
    raise NotImplementedError

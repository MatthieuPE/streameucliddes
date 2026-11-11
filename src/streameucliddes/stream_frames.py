"""Stream frames: (ra, dec) <-> (phi1, phi2), and the phi1/phi2 grid of the maps.

A stream frame is an astropy frame, in practice a gala ``GreatCircleICRSFrame``:
the ``stream_frame`` of a galstreams track, or, for another stream,
``gala.coordinates.GreatCircleICRSFrame.from_endpoints`` (or ``from_pole_ra0``).
The transformations are those of astropy and gala; the functions below only
take and return numpy arrays in degrees, as in the catalogs.

Maps in a stream frame use the phi1/phi2 grid of ``phi12_bin_centres``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.coordinates as coord
import astropy.units as u
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


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

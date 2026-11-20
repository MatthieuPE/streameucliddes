"""Stream tracks fitted in the literature that galstreams does not have.

Aliqa Uma (Li et al. 2021, "Broken into Pieces: ATLAS and Aliqa Uma as One
Single Stream", ApJ 911, 149, arXiv:2006.10763). galstreams' ``AAU-AliqaUma-L21``
is their approximate track from the spectroscopic members (their Eq. 3,
``phi2 = 1.5 - 0.5 ((phi1 - 3) / 10)^2``). Their fitted track is a natural
cubic spline of the stream density in DES (their Sect. 3.3, Fig. 14), published
with its width in their repository (``LI2021_REPOSITORY_URL``,
``stream_params/pub_track_au.fits`` and ``pub_width_au.fits``): it is 0.5 to
0.9 deg above Eq. 3 where it is constrained, as are the Aliqa Uma
spectroscopic members (median +0.5 deg) and the Euclid DR1 excess (+0.6 deg).
Its first and last knots (phi1 = -21 and -10) have uncertainties of 1 deg:
the track is taken between them by default (``LI2021_AU_PHI1_RANGE``).

Both are in the ATLAS frame of Shipp et al. (2019) (rotation matrix in Li et
al. 2021, Appendix A: ``SHIPP2019_AAU_ROTATION``); the distance is their Eq. 2
(``li2021_aau_distance_modulus``, fitted to BHBs over -17 < phi1 < 7 deg).
The files are downloaded once into ``LI2021_DIR`` (gitignored ``data/``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.table import Table
from scipy.interpolate import CubicSpline

from streameucliddes.catalogs.galstreams_catalog import track_from_points
from streameucliddes.catalogs.overlapped_objects import download_file

if TYPE_CHECKING:
    from types import SimpleNamespace

    from numpy.typing import ArrayLike, NDArray

# ATLAS stream frame of Shipp et al. (2019), as printed in Li et al. (2021, Appendix A): (x, y, z) frame = R (x, y, z)
# ICRS, phi1 = atan2(y, x), phi2 = asin(z)
SHIPP2019_AAU_ROTATION = np.array([
    [0.83697865, 0.29481904, -0.4610298],
    [0.51616778, -0.70514011, 0.4861566],
    [0.18176238, 0.64487142, 0.74236331],
])
LI2021_REPOSITORY_URL = "https://raw.githubusercontent.com/s5collab/ATLAS_AliqaUma/master"
LI2021_FILES = {"track": "stream_params/pub_track_au.fits", "width": "stream_params/pub_width_au.fits",
                "members": "data/aau_members_full.fits"}
LI2021_DIR = Path(__file__).resolve().parents[3] / "data" / "li2021_aau"
# phi1 range of the Aliqa Uma track (Shipp 2019 frame, deg): between the end knots, uncertain by 1 deg
LI2021_AU_PHI1_RANGE = (-19.5, -11.5)
LI2021_AU_TRACK_NAME = "AAU-AliqaUma-Li21spline"


def shipp2019_aau_to_radec(phi1: ArrayLike, phi2: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(ra, dec) of points in the ATLAS frame of Shipp et al. 2019 (deg)."""
    phi1, phi2 = np.radians(np.asarray(phi1, dtype=float)), np.radians(np.asarray(phi2, dtype=float))
    frame_xyz = np.stack([np.cos(phi2) * np.cos(phi1), np.cos(phi2) * np.sin(phi1), np.sin(phi2)])
    x, y, z = np.tensordot(SHIPP2019_AAU_ROTATION.T, frame_xyz, axes=1)
    return np.degrees(np.arctan2(y, x)) % 360.0, np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))


def radec_to_shipp2019_aau(ra: ArrayLike, dec: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(phi1, phi2) in the ATLAS frame of Shipp et al. 2019 of (ra, dec) points (deg)."""
    ra, dec = np.radians(np.asarray(ra, dtype=float)), np.radians(np.asarray(dec, dtype=float))
    icrs_xyz = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    x, y, z = np.tensordot(SHIPP2019_AAU_ROTATION, icrs_xyz, axes=1)
    return np.degrees(np.arctan2(y, x)), np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))


def li2021_aau_distance_modulus(phi1: ArrayLike) -> NDArray[np.float64]:
    """Distance modulus along ATLAS-Aliqa Uma, Li et al. 2021 Eq. 2: 16.66 - 0.28 x + 0.045 x^2, x = phi1 / 10 deg."""
    x = np.asarray(phi1, dtype=float) / 10.0
    return 16.66 - 0.28 * x + 0.045 * x**2


def load_li2021_aliqa_uma(data_dir: str | Path | None = None) -> dict[str, Table]:
    """The Li et al. 2021 files: fitted track and width of Aliqa Uma, spectroscopic members of ATLAS-Aliqa Uma.

    Downloaded from ``LI2021_REPOSITORY_URL`` into ``data_dir`` the first time.

    Parameters
    ----------
    data_dir : str or Path, optional
        Default: ``LI2021_DIR``.

    Returns
    -------
    tables : dict of astropy.table.Table
        ``"track"`` (spline knots: ``phi1``, ``phi2_mean``, ``phi2_std``, deg),
        ``"width"`` (``phi1``, ``logwidth_mean``, ``logwidth_std``: natural
        log of the Gaussian sigma in deg), ``"members"`` (the 96 members,
        ``phi1``, ``phi2``, ``stream`` "au" or "atlas"...).
    """
    data_dir = LI2021_DIR if data_dir is None else Path(data_dir)
    tables = {}
    for name, remote in LI2021_FILES.items():
        path = data_dir / Path(remote).name
        if not path.exists():
            download_file(f"{LI2021_REPOSITORY_URL}/{remote}", path)
        tables[name] = Table.read(path)
    return tables


def li2021_aliqa_uma_track(
    phi1_range: tuple[float, float] = LI2021_AU_PHI1_RANGE,
    *,
    step_deg: float = 0.05,
    data_dir: str | Path | None = None,
) -> SimpleNamespace:
    """The fitted track of Aliqa Uma of Li et al. 2021, as a galstreams track.

    The natural cubic spline through their knots (``scipy.interpolate.CubicSpline``,
    ``bc_type="natural"``, as their STAN model), every ``step_deg`` over
    ``phi1_range`` in the Shipp et al. 2019 frame, taken to the sky
    (``shipp2019_aau_to_radec``), with the distance modulus of their Eq. 2;
    ``galstreams_catalog.track_from_points``, in the frame through its ends.

    Parameters
    ----------
    phi1_range : tuple of float, optional
        phi1 range (Shipp 2019 frame, deg).
    step_deg : float, optional
        Step of the points (deg).
    data_dir : str or Path, optional
        See ``load_li2021_aliqa_uma``.

    Returns
    -------
    track : types.SimpleNamespace
        Named ``LI2021_AU_TRACK_NAME``, stream ``"AAU-AliqaUma"``, reference ``"li2021_spline"``.
    """
    knots = load_li2021_aliqa_uma(data_dir)["track"]
    spline = CubicSpline(np.asarray(knots["phi1"], dtype=float), np.asarray(knots["phi2_mean"], dtype=float),
                         bc_type="natural")
    phi1 = np.linspace(*phi1_range, int(round((phi1_range[1] - phi1_range[0]) / step_deg)) + 1)
    ra, dec = shipp2019_aau_to_radec(phi1, spline(phi1))
    return track_from_points(LI2021_AU_TRACK_NAME, ra, dec, li2021_aau_distance_modulus(phi1),
                             stream_name="AAU-AliqaUma", ref="li2021_spline")


def li2021_aliqa_uma_width_deg(
    phi1_range: tuple[float, float] = LI2021_AU_PHI1_RANGE, *, data_dir: str | Path | None = None
) -> float:
    """Gaussian width (sigma, deg) of Aliqa Uma from Li et al. 2021: mean of exp(logwidth) at the knots in ``phi1_range``.

    Their width spline has 4 knots (phi1 = -21, -17.3, -13.7, -10): 0.24 and
    0.30 deg in the default range, 0.27 deg on average.
    """
    width = load_li2021_aliqa_uma(data_dir)["width"]
    phi1 = np.asarray(width["phi1"], dtype=float)
    inside = (phi1 >= phi1_range[0]) & (phi1 <= phi1_range[1])
    return float(np.mean(np.exp(np.asarray(width["logwidth_mean"], dtype=float)[inside])))

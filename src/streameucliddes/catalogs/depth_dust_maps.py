"""Depth and dust maps of the sky, and their values on a phi1/phi2 grid.

To check whether a stream candidate follows a fluctuation of the survey
depth or of the dust (``plotting.plot_candidate_investigation.plot_candidate_depth_dust_maps``):

- depth maps of Euclid, one per band, in :data:`DEPTH_MAP_DIR`: HEALPix FITS
  tables ``euclid_fracdet_map_<band>_nside<nside>.fits`` (RING, explicit
  pixel column), with the columns ``MAGLIM`` (limiting magnitude) and
  ``FRACDET`` (detection fraction of the pixel), read with healpy. Pixels
  outside the survey (``FRACDET`` 0, ``MAGLIM`` UNSEEN) are NaN;
- the E(B-V) of the dust correction of the catalog magnitudes
  (``euclid_catalog.DUST_CORRECTIONS``, same names), read with dustmaps
  (Green 2018, JOSS 3, 695) from the maps of :data:`DUST_MAPS`: ``"SFD98"``,
  Schlegel, Finkbeiner & Davis (1998, ``SFDQuery``); ``"PLANCK"``, the Planck
  2013 E(B-V) (Planck Collaboration: Abergel et al. 2014, ``PlanckQuery``,
  component ``"extragalactic"``, the Euclid ``GAL_EBV`` column).

The values on a phi1/phi2 grid are those at the bin centres
(``stream_frames.phi12_bin_centres``), as ``survey_footprint.survey_footprint_phi12_mask``.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np

from streameucliddes.catalogs.euclid_catalog import DATA_DIR, DEFAULT_DUST_CORRECTION, DUST_CORRECTIONS
from streameucliddes.stream_frames import phi12_bin_centres, phi12_to_radec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import astropy.coordinates as coord
    from numpy.typing import ArrayLike, NDArray

LOGGER = logging.getLogger(__name__)

DEPTH_MAP_DIR = DATA_DIR / "depth_maps" / "euclid"
DEPTH_MAP_PREFIX = "euclid_fracdet_map_"  # file: <prefix><band>_nside<nside>.fits
# Columns of a depth map (field of hp.read_map, after the pixel column), their names and units
DEPTH_QUANTITIES = {"MAGLIM": 0, "FRACDET": 1}
DEPTH_QUANTITY_NAMES = {"MAGLIM": "limiting magnitude", "FRACDET": "detection fraction"}
DEPTH_QUANTITY_UNITS = {"MAGLIM": "mag", "FRACDET": ""}
DEFAULT_DEPTH_QUANTITIES = ("MAGLIM",)
EUCLID_BAND_ORDER = ("VIS", "Y", "J", "H")  # bands with a depth map, in this order (others after, sorted)
# Dust maps of each dust correction (keys of euclid_catalog.DUST_CORRECTIONS): the SFD98 folder
# (SFD_dust_4096_{ngp,sgp}.fits, as scripts/download_euclid_dr1.py), the Planck file (as scripts/compare_dust_maps.py)
DUST_MAPS = {
    "SFD98": DATA_DIR / "dustmaps" / "sfd",
    "PLANCK": DATA_DIR / "dustmaps" / "planck" / "HFI_CompMap_ThermalDustModel_2048_R1.20.fits",
}
DUST_MAP_NAMES = {"SFD98": "SFD98", "PLANCK": "Planck 2013"}  # for the names of the maps
EBV_MAP_KEY = "EBV"  # key of the dust map in depth_dust_phi12_maps


# ===================================================
# Depth maps
# ===================================================


def depth_map_bands(depth_map_dir: str | Path | None = None) -> list[str]:
    """Bands with a depth map in ``depth_map_dir`` (default :data:`DEPTH_MAP_DIR`), upper case.

    In the order of :data:`EUCLID_BAND_ORDER`, then the others sorted.
    """
    names = {path.name[len(DEPTH_MAP_PREFIX):].split("_nside")[0].upper()
             for path in Path(depth_map_dir or DEPTH_MAP_DIR).glob(f"{DEPTH_MAP_PREFIX}*_nside*.fits")}
    return [band for band in EUCLID_BAND_ORDER if band in names] + sorted(names - set(EUCLID_BAND_ORDER))


def depth_map_path(band: str, depth_map_dir: str | Path | None = None) -> Path:
    """Depth map file of ``band`` (any case) in ``depth_map_dir`` (default :data:`DEPTH_MAP_DIR`).

    Raises
    ------
    FileNotFoundError
        No map, or more than one (two nsides), for this band.
    """
    folder = Path(depth_map_dir or DEPTH_MAP_DIR)
    paths = sorted(folder.glob(f"{DEPTH_MAP_PREFIX}{band.lower()}_nside*.fits"))
    if len(paths) != 1:
        found = "no depth map" if not paths else f"{len(paths)} depth maps ({[path.name for path in paths]})"
        raise FileNotFoundError(f"{found} of band {band!r} in {folder}; bands with one: {depth_map_bands(folder)}")
    return paths[0]


def read_depth_map(band: str, quantity: str = "MAGLIM", depth_map_dir: str | Path | None = None) -> NDArray:
    """HEALPix depth map of a band, RING; NaN outside the survey (``FRACDET`` 0 or ``MAGLIM`` UNSEEN).

    Read once per file (cached), read-only.

    Parameters
    ----------
    band : str
        Band, e.g. ``"VIS"`` or ``"H"`` (``depth_map_bands``).
    quantity : str, optional
        Column: ``"MAGLIM"`` or ``"FRACDET"`` (:data:`DEPTH_QUANTITIES`).
    depth_map_dir : str or Path, optional
        Default: :data:`DEPTH_MAP_DIR`.

    Returns
    -------
    values : numpy.ndarray of float
        Shape (12 * nside**2,).

    Raises
    ------
    ValueError
        Unknown quantity.
    """
    if quantity not in DEPTH_QUANTITIES:
        raise ValueError(f"Unknown depth quantity {quantity!r}: choose one of {list(DEPTH_QUANTITIES)}")
    return _read_depth_map(str(depth_map_path(band, depth_map_dir)))[quantity]


@functools.lru_cache(maxsize=4)
def _read_depth_map(path: str) -> dict[str, NDArray]:
    """Both columns of a depth map file, RING, NaN outside the survey, read-only."""
    columns = hp.read_map(path, field=list(DEPTH_QUANTITIES.values()), nest=False, dtype=np.float64)
    values = dict(zip(DEPTH_QUANTITIES, columns))
    outside = ~(values["FRACDET"] > 0) | (values["MAGLIM"] == hp.UNSEEN) | ~np.isfinite(values["MAGLIM"])
    for column in values.values():
        column[outside] = np.nan
        column.setflags(write=False)
    LOGGER.info("Depth map read from %s (nside %d).", path, hp.npix2nside(outside.size))
    return values


def depth_at_radec(
    ra: ArrayLike,
    dec: ArrayLike,
    band: str,
    quantity: str = "MAGLIM",
    depth_map_dir: str | Path | None = None,
) -> NDArray[np.float64]:
    """Value of the depth map of ``band`` at (ra, dec) (deg): that of the HEALPix pixel; NaN outside.

    See ``read_depth_map`` for the arguments. Returns an array of the shape of ``ra``.
    """
    values = read_depth_map(band, quantity, depth_map_dir)
    ra, dec = np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
    result = np.full(ra.shape, np.nan)
    finite = np.isfinite(ra) & np.isfinite(dec)
    result[finite] = values[hp.ang2pix(hp.npix2nside(values.size), ra[finite], dec[finite], lonlat=True)]
    return result


# ===================================================
# Dust maps
# ===================================================


def _dust_map(dust_correction: str) -> Path:
    """Map of :data:`DUST_MAPS` of a dust correction, with a clear error for an unknown name."""
    if dust_correction not in DUST_MAPS:
        raise ValueError(f"Unknown dust correction {dust_correction!r}: choose one of {list(DUST_CORRECTIONS)}")
    return DUST_MAPS[dust_correction]


@functools.lru_cache(maxsize=2)
def _dust_query(dust_correction: str) -> Any:
    """dustmaps query of a dust correction, loaded once (the Planck map: about 0.2 GB in memory)."""
    path = _dust_map(dust_correction)
    if not path.exists():
        raise FileNotFoundError(f"Dust map of {dust_correction!r} not found: {path}")
    if dust_correction == "SFD98":
        from dustmaps.sfd import SFDQuery

        return SFDQuery(map_dir=str(path))
    from dustmaps.planck import PlanckQuery

    return PlanckQuery(map_fname=str(path), component="extragalactic")


def ebv_at_radec(ra: ArrayLike, dec: ArrayLike, dust_correction: str = DEFAULT_DUST_CORRECTION) -> NDArray[np.float64]:
    """E(B-V) (mag) of the dust map of ``dust_correction`` at (ra, dec) (deg); NaN where not finite.

    Parameters
    ----------
    ra, dec : array_like
        Points (deg).
    dust_correction : str, optional
        ``"SFD98"`` or ``"PLANCK"``, as the dust correction of the
        magnitudes (``euclid_catalog.get_euclid_stars``).

    Returns
    -------
    ebv : numpy.ndarray of float
        Of the shape of ``ra``.
    """
    from astropy.coordinates import SkyCoord

    query = _dust_query(dust_correction)
    ra, dec = np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
    result = np.full(ra.shape, np.nan)
    finite = np.isfinite(ra) & np.isfinite(dec)
    if finite.any():
        result[finite] = query(SkyCoord(ra[finite], dec[finite], unit="deg", frame="icrs"))
    return result


# ===================================================
# On a phi1/phi2 grid
# ===================================================


def depth_dust_phi12_maps(
    stream_frame: coord.BaseCoordinateFrame,
    window: tuple[float, float, float, float],
    bin_size_deg: float,
    *,
    bands: Sequence[str] | None = None,
    depth_quantities: Sequence[str] = DEFAULT_DEPTH_QUANTITIES,
    dust_correction: str | None = DEFAULT_DUST_CORRECTION,
    depth_map_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Depth maps of each band and the dust map at the bin centres of a phi1/phi2 grid.

    The grid of the maps of a search in this frame and window
    (``stream_frames.phi12_bin_centres``): the maps can be drawn beside, or
    compared bin by bin with, the star maps.

    Parameters
    ----------
    stream_frame : astropy.coordinates.BaseCoordinateFrame
        Stream frame, e.g. ``cand.stream_frame``.
    window : tuple of float
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg).
    bin_size_deg : float
        Bin size of the grid (deg).
    bands : sequence of str, optional
        Bands of the depth maps. Default: all those with a map
        (``depth_map_bands``); ``()`` for none.
    depth_quantities : sequence of str, optional
        Columns of the depth maps (:data:`DEPTH_QUANTITIES`), one map each.
    dust_correction : str or None, optional
        Dust map (``"SFD98"`` or ``"PLANCK"``); None: no dust map.
    depth_map_dir : str or Path, optional
        Default: :data:`DEPTH_MAP_DIR`.

    Returns
    -------
    maps : dict
        ``{"<QUANTITY>_<band>": map, ..., "EBV": map}`` (depth maps in the
        order of the bands, then of the quantities; the dust map last), each
        map a dict: ``values`` (shape (n_phi1, n_phi2), row = phi1 bin, NaN
        outside the map), ``phi1_centres``, ``phi2_centres``,
        ``bin_size_deg``, ``quantity`` (``"MAGLIM"``, ``"FRACDET"`` or
        ``"EBV"``), ``band`` (None for the dust), ``dust_correction`` (None
        for the depth), ``name`` (e.g. ``"VIS limiting magnitude"``, for a
        title) and ``unit`` (``"mag"``, or ``""``).
    """
    phi1_centres, phi2_centres = phi12_bin_centres(window, bin_size_deg)
    phi1, phi2 = np.meshgrid(phi1_centres, phi2_centres, indexing="ij")
    ra, dec = phi12_to_radec(phi1, phi2, stream_frame)
    grid = {"phi1_centres": phi1_centres, "phi2_centres": phi2_centres, "bin_size_deg": float(bin_size_deg)}
    maps = {}
    for band in depth_map_bands(depth_map_dir) if bands is None else bands:
        for quantity in depth_quantities:
            maps[f"{quantity}_{band}"] = {
                **grid, "values": depth_at_radec(ra, dec, band, quantity, depth_map_dir), "quantity": quantity,
                "band": band, "dust_correction": None, "name": f"{band} {DEPTH_QUANTITY_NAMES[quantity]}",
                "unit": DEPTH_QUANTITY_UNITS[quantity],
            }
    if dust_correction is not None:
        maps[EBV_MAP_KEY] = {
            **grid, "values": ebv_at_radec(ra, dec, dust_correction), "quantity": EBV_MAP_KEY, "band": None,
            "dust_correction": dust_correction, "name": f"$E(B-V)$ {DUST_MAP_NAMES[dust_correction]}", "unit": "mag",
        }
    return maps

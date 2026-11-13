"""Objects that overlap the stream searches: the LMC, the SMC, the dwarf galaxies, the globular
clusters, the bright galaxies.

The Magellanic Clouds, the dwarf galaxies and the globular clusters come
from the Local Volume Database (LVDB; Pace 2024, arXiv:2411.07424), release
``LVDB_VERSION``. Its tables are downloaded once from GitHub and kept in
``LVDB_DIR/<LVDB_VERSION>/<table>.csv``. Only the objects that LVDB confirms
as real systems (``confirmed_real`` = 1) are kept.

The bright galaxies (e.g. M81, M82, M31, NGC 253, not in the LVDB, which
lists their satellites) come from the Siena Galaxy Atlas 2020 (SGA;
Moustakas et al. 2023, ApJS 269, 3): those with ``MAG_LEDA`` below
``SGA_MAG_LEDA_LIMIT`` (16.5, as the Euclid DR1 photometric masks, Euclid
Collaboration: Odier et al. 2026). The full atlas (716 MB) is downloaded
once, its bright galaxies kept in ``SGA_BRIGHT_PATH``, and it is deleted.

Each object comes as one row: ``key`` and ``name``, ``ra``, ``dec`` (deg),
``radius_arcmin`` (semi-major axis of the reference ellipse, arcmin: the
half-light ellipse for the LVDB, ``rhalf``; the 26 mag/arcsec^2 isophote
for the SGA, ``D26 / 2``), ``ellipticity`` (1 - b/a) and ``position_angle``
(of the major axis, deg, north to east) of that ellipse, and
``distance_kpc``; NaN when unknown.

The masks built from them are in ``analysis.build_masks``.
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

LOGGER = logging.getLogger(__name__)

# LVDB release whose tables are used (GitHub tag of apace7/local_volume_database)
LVDB_VERSION = "v1.1.2"
LVDB_URL = "https://raw.githubusercontent.com/apace7/local_volume_database/{version}/data/{table}.csv"
# Downloaded tables, next to the catalogs (see euclid_catalog.DATA_DIR): the data/ folder
# of the repository, gitignored
LVDB_DIR = Path(__file__).resolve().parents[3] / "data" / "lvdb"
DOWNLOAD_TIMEOUT_S = 60

# LVDB keys of the Large and Small Magellanic Clouds, in the dwarf_all table
LMC_KEY, SMC_KEY = "lmc", "smc"
DWARF_TABLE = "dwarf_all"  # Milky Way, M31 and Local Volume dwarf galaxies
# Milky Way globular clusters: Harris catalog, those found since, and the ambiguous
# compact systems (globular cluster or dwarf galaxy)
GLOBULAR_CLUSTER_TABLES = ("gc_harris", "gc_mw_new", "gc_ambiguous")

# Siena Galaxy Atlas 2020: the full atlas, and the file of its bright galaxies kept in data/sga/
SGA_URL = "https://portal.nersc.gov/project/cosmo/data/sga/2020/SGA-2020.fits"
SGA_MAG_LEDA_LIMIT = 16.5
SGA_DIR = Path(__file__).resolve().parents[3] / "data" / "sga"
SGA_BRIGHT_FILE = "SGA-2020_mag_leda_below_16.5.csv"
# Columns kept from the ELLIPSE table of the atlas: identifier, name, position (deg), D26 (diameter
# of the 26 mag/arcsec^2 isophote, arcmin), its position angle (deg, north to east) and axis
# ratio b/a, HyperLEDA magnitude
SGA_COLUMNS = ["SGA_ID", "GALAXY", "RA", "DEC", "D26", "PA", "BA", "MAG_LEDA"]

OBJECT_COLUMNS = ["key", "name", "ra", "dec", "radius_arcmin", "ellipticity", "position_angle", "distance_kpc"]


# ===================================================
# LVDB
# ===================================================


def load_lvdb_table(table: str) -> pd.DataFrame:
    """One table of the LVDB, as it is.

    Read from ``LVDB_DIR/<LVDB_VERSION>/<table>.csv``; downloaded there
    first if it is not there yet (``LVDB_URL``).

    Parameters
    ----------
    table : str
        LVDB table, without ``.csv`` (e.g. ``"dwarf_all"``, ``"gc_harris"``).

    Returns
    -------
    table : pandas.DataFrame
        All the rows and columns of the table.
    """
    path = LVDB_DIR / LVDB_VERSION / f"{table}.csv"
    if not path.exists():
        _download(LVDB_URL.format(version=LVDB_VERSION, table=table), path)
    return pd.read_csv(path, low_memory=False)


def get_lmc() -> pd.DataFrame:
    """The Large Magellanic Cloud (LVDB ``dwarf_all``, key ``LMC_KEY``).

    Returns
    -------
    lmc : pandas.DataFrame
        One row, columns ``OBJECT_COLUMNS``.
    """
    dwarfs = load_lvdb_table(DWARF_TABLE)
    return _lvdb_objects(dwarfs[dwarfs["key"] == LMC_KEY])


def get_smc() -> pd.DataFrame:
    """The Small Magellanic Cloud (LVDB ``dwarf_all``, key ``SMC_KEY``).

    Returns
    -------
    smc : pandas.DataFrame
        One row, columns ``OBJECT_COLUMNS``.
    """
    dwarfs = load_lvdb_table(DWARF_TABLE)
    return _lvdb_objects(dwarfs[dwarfs["key"] == SMC_KEY])


def get_dwarf_galaxies() -> pd.DataFrame:
    """The confirmed dwarf galaxies of the LVDB (``dwarf_all``), but the LMC and the SMC.

    The Magellanic Clouds have their own functions (``get_lmc``, ``get_smc``).

    Returns
    -------
    dwarfs : pandas.DataFrame
        One row per dwarf galaxy, columns ``OBJECT_COLUMNS``.
    """
    dwarfs = load_lvdb_table(DWARF_TABLE)
    return _lvdb_objects(dwarfs[~dwarfs["key"].isin([LMC_KEY, SMC_KEY])])


def get_globular_clusters() -> pd.DataFrame:
    """The confirmed Milky Way globular clusters of the LVDB.

    Those of the tables ``GLOBULAR_CLUSTER_TABLES``: the Harris catalog
    (with the Sagittarius ones), those found since, and the compact systems
    LVDB calls ambiguous (globular cluster or dwarf galaxy).

    Returns
    -------
    clusters : pandas.DataFrame
        One row per cluster, columns ``OBJECT_COLUMNS``.
    """
    tables = [load_lvdb_table(table) for table in GLOBULAR_CLUSTER_TABLES]
    return _lvdb_objects(pd.concat(tables, ignore_index=True))


def _lvdb_objects(table: pd.DataFrame) -> pd.DataFrame:
    """The confirmed objects of an LVDB table, with the columns ``OBJECT_COLUMNS``."""
    confirmed = table[table["confirmed_real"] == 1]
    return pd.DataFrame({
        "key": confirmed["key"].to_numpy(),
        "name": confirmed["name"].to_numpy(),
        "ra": confirmed["ra"].to_numpy(dtype=float),
        "dec": confirmed["dec"].to_numpy(dtype=float),
        "radius_arcmin": confirmed["rhalf"].to_numpy(dtype=float),
        "ellipticity": confirmed["ellipticity"].to_numpy(dtype=float),
        "position_angle": confirmed["position_angle"].to_numpy(dtype=float),
        "distance_kpc": confirmed["distance"].to_numpy(dtype=float),
    })


# ===================================================
# Siena Galaxy Atlas
# ===================================================


def load_sga_bright_galaxies(fits_path: str | Path | None = None) -> pd.DataFrame:
    """The bright galaxies of the Siena Galaxy Atlas 2020: ``MAG_LEDA`` below ``SGA_MAG_LEDA_LIMIT``.

    Read from ``SGA_DIR/SGA_BRIGHT_FILE``. If it is not there yet, it is
    made from the full atlas (``extract_sga_bright_galaxies``): read from
    ``fits_path`` if given, else downloaded from ``SGA_URL`` (716 MB) into
    ``SGA_DIR`` and deleted once the bright galaxies are kept.

    Parameters
    ----------
    fits_path : str or pathlib.Path, optional
        The full atlas (``SGA-2020.fits``), already downloaded: not deleted.

    Returns
    -------
    galaxies : pandas.DataFrame
        One row per galaxy, columns ``SGA_COLUMNS`` (81 833 galaxies in SGA-2020).
    """
    path = SGA_DIR / SGA_BRIGHT_FILE
    if not path.exists():
        if fits_path is not None:
            extract_sga_bright_galaxies(fits_path, path)
        else:
            full_path = SGA_DIR / Path(SGA_URL).name
            _download(SGA_URL, full_path)
            try:
                extract_sga_bright_galaxies(full_path, path)
            finally:
                full_path.unlink()
    return pd.read_csv(path)


def extract_sga_bright_galaxies(fits_path: str | Path, path: str | Path) -> None:
    """Keep the bright galaxies of the full Siena Galaxy Atlas in a CSV file.

    The rows of the ``ELLIPSE`` table with ``MAG_LEDA`` below
    ``SGA_MAG_LEDA_LIMIT``, the columns ``SGA_COLUMNS``.

    Parameters
    ----------
    fits_path : str or pathlib.Path
        The full atlas, ``SGA-2020.fits``.
    path : str or pathlib.Path
        The CSV file written (a complete file, or none).
    """
    path = Path(path)
    with fits.open(fits_path, memmap=True) as hdul:
        table = hdul["ELLIPSE"].data
        bright = np.asarray(table["MAG_LEDA"]) < SGA_MAG_LEDA_LIMIT
        columns = {column: np.asarray(table[column][bright]) for column in SGA_COLUMNS}
    galaxies = pd.DataFrame({  # names without their padding; numbers in the native byte order (FITS: big-endian)
        column: np.char.strip(values.astype(str)) if values.dtype.kind in "SU"
        else values.astype(values.dtype.newbyteorder("="))
        for column, values in columns.items()
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(".partial")
    galaxies.to_csv(partial_path, index=False)
    partial_path.replace(path)
    LOGGER.info("%d SGA galaxies with MAG_LEDA < %g kept in %s.", len(galaxies), SGA_MAG_LEDA_LIMIT, path)


def get_galaxies() -> pd.DataFrame:
    """The bright galaxies of the Siena Galaxy Atlas (``load_sga_bright_galaxies``).

    The reference ellipse is the 26 mag/arcsec^2 isophote of the atlas:
    ``radius_arcmin`` = ``D26 / 2``, ``ellipticity`` = 1 - ``BA``,
    ``position_angle`` = ``PA``. No distance (NaN).

    Returns
    -------
    galaxies : pandas.DataFrame
        One row per galaxy, columns ``OBJECT_COLUMNS``; ``key`` ``"sga_<SGA_ID>"``.
    """
    sga = load_sga_bright_galaxies()
    return pd.DataFrame({
        "key": "sga_" + sga["SGA_ID"].astype(str),
        "name": sga["GALAXY"].to_numpy(),
        "ra": sga["RA"].to_numpy(dtype=float),
        "dec": sga["DEC"].to_numpy(dtype=float),
        "radius_arcmin": sga["D26"].to_numpy(dtype=float) / 2.0,
        "ellipticity": 1.0 - sga["BA"].to_numpy(dtype=float),
        "position_angle": sga["PA"].to_numpy(dtype=float),
        "distance_kpc": np.full(len(sga), np.nan),
    })


def _download(url: str, path: Path) -> None:
    """Download url to path: a complete file, or none."""
    LOGGER.info("Downloading %s to %s.", url, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(".partial")
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as answer, partial_path.open("wb") as file:
        shutil.copyfileobj(answer, file)
    partial_path.replace(path)

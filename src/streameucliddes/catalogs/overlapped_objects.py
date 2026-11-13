"""Objects that overlap the stream searches: the LMC, the SMC, the dwarf galaxies, the globular clusters.

Their positions and sizes come from the Local Volume Database (LVDB; Pace
2024, arXiv:2411.07424), release ``LVDB_VERSION``. Its tables are downloaded
once from GitHub and kept in ``LVDB_DIR/<LVDB_VERSION>/<table>.csv``.

Each object comes as one row: ``key`` and ``name`` (LVDB), ``ra``, ``dec``
(deg), ``rhalf_arcmin`` (half-light radius, LVDB ``rhalf``: the semi-major
axis of the half-light ellipse, in arcmin), ``ellipticity`` (1 - b/a) and
``position_angle`` (of the major axis, deg, north to east) of that ellipse,
and ``distance_kpc``; NaN when unknown. Only the objects that LVDB confirms
as real systems (``confirmed_real`` = 1) are kept.

The masks built from them are in ``analysis.build_masks``.
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

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

OBJECT_COLUMNS = ["key", "name", "ra", "dec", "rhalf_arcmin", "ellipticity", "position_angle", "distance_kpc"]


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
        url = LVDB_URL.format(version=LVDB_VERSION, table=table)
        LOGGER.info("Downloading %s to %s.", url, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = path.with_suffix(".partial")
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as answer, partial_path.open("wb") as file:
            shutil.copyfileobj(answer, file)
        partial_path.replace(path)  # a complete file, or none
    return pd.read_csv(path, low_memory=False)


def get_lmc() -> pd.DataFrame:
    """The Large Magellanic Cloud (LVDB ``dwarf_all``, key ``LMC_KEY``).

    Returns
    -------
    lmc : pandas.DataFrame
        One row, columns ``OBJECT_COLUMNS``.
    """
    dwarfs = load_lvdb_table(DWARF_TABLE)
    return _objects(dwarfs[dwarfs["key"] == LMC_KEY])


def get_smc() -> pd.DataFrame:
    """The Small Magellanic Cloud (LVDB ``dwarf_all``, key ``SMC_KEY``).

    Returns
    -------
    smc : pandas.DataFrame
        One row, columns ``OBJECT_COLUMNS``.
    """
    dwarfs = load_lvdb_table(DWARF_TABLE)
    return _objects(dwarfs[dwarfs["key"] == SMC_KEY])


def get_dwarf_galaxies() -> pd.DataFrame:
    """The confirmed dwarf galaxies of the LVDB (``dwarf_all``), but the LMC and the SMC.

    The Magellanic Clouds have their own functions (``get_lmc``, ``get_smc``).

    Returns
    -------
    dwarfs : pandas.DataFrame
        One row per dwarf galaxy, columns ``OBJECT_COLUMNS``.
    """
    dwarfs = load_lvdb_table(DWARF_TABLE)
    return _objects(dwarfs[~dwarfs["key"].isin([LMC_KEY, SMC_KEY])])


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
    return _objects(pd.concat(tables, ignore_index=True))


def _objects(table: pd.DataFrame) -> pd.DataFrame:
    """The confirmed objects of an LVDB table, with the columns ``OBJECT_COLUMNS``."""
    confirmed = table[table["confirmed_real"] == 1]
    return pd.DataFrame({
        "key": confirmed["key"].to_numpy(),
        "name": confirmed["name"].to_numpy(),
        "ra": confirmed["ra"].to_numpy(dtype=float),
        "dec": confirmed["dec"].to_numpy(dtype=float),
        "rhalf_arcmin": confirmed["rhalf"].to_numpy(dtype=float),
        "ellipticity": confirmed["ellipticity"].to_numpy(dtype=float),
        "position_angle": confirmed["position_angle"].to_numpy(dtype=float),
        "distance_kpc": confirmed["distance"].to_numpy(dtype=float),
    })

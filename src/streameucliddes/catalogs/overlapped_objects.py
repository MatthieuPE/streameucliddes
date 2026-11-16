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
(of the major axis, deg, north to east) of that ellipse, ``distance_kpc``,
``distance_modulus``, ``feh`` ([Fe/H], the LVDB ``metallicity``, its best
value) and ``apparent_magnitude_v`` (V); NaN when unknown (the SGA has none
of the last four). The LVDB ``rhalf`` is the semi-major axis of the
half-light ellipse (LVDB documentation, "Usage & Description of tables").

``get_objects`` gathers several types, with a ``type`` column;
``find_object`` picks one object by name, ``objects_near_distance_modulus``
those near a distance modulus.

The masks built from them are in ``analysis.build_masks``.
"""

from __future__ import annotations

import difflib
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
import pandas as pd
from astropy.io import fits

from streameucliddes.catalogs.euclid_catalog import catalog_point_map
from streameucliddes.unit_conversions import convert_kpc_to_DM

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

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

OBJECT_COLUMNS = [
    "key", "name", "ra", "dec", "radius_arcmin", "ellipticity", "position_angle", "distance_kpc",
    "distance_modulus", "feh", "apparent_magnitude_v",
]
# Types of objects (get_objects), as the keys of the masks of analysis.build_masks
OBJECT_TYPES = ("lmc", "smc", "dwarfs", "gc", "galaxies")


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
    """The confirmed objects of an LVDB table, with the columns ``OBJECT_COLUMNS``.

    The distance modulus from the distance when the LVDB gives none.
    """
    confirmed = table[table["confirmed_real"] == 1]
    distance_kpc = confirmed["distance"].to_numpy(dtype=float)
    distance_modulus = _float_column(confirmed, "distance_modulus")
    from_distance = ~np.isfinite(distance_modulus) & (distance_kpc > 0)
    if from_distance.any():
        distance_modulus[from_distance] = convert_kpc_to_DM(distance_kpc[from_distance]).value
    return pd.DataFrame({
        "key": confirmed["key"].to_numpy(),
        "name": confirmed["name"].to_numpy(),
        "ra": confirmed["ra"].to_numpy(dtype=float),
        "dec": confirmed["dec"].to_numpy(dtype=float),
        "radius_arcmin": confirmed["rhalf"].to_numpy(dtype=float),
        "ellipticity": confirmed["ellipticity"].to_numpy(dtype=float),
        "position_angle": confirmed["position_angle"].to_numpy(dtype=float),
        "distance_kpc": distance_kpc,
        "distance_modulus": distance_modulus,
        "feh": _float_column(confirmed, "metallicity"),
        "apparent_magnitude_v": _float_column(confirmed, "apparent_magnitude_v"),
    })


def _float_column(table: pd.DataFrame, column: str) -> np.ndarray:
    """A column as floats (a new array); NaN when the table does not have it."""
    if column not in table:
        return np.full(len(table), np.nan)
    return pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float, copy=True)


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
        **dict.fromkeys(("distance_kpc", "distance_modulus", "feh", "apparent_magnitude_v"), np.full(len(sga), np.nan)),
    })


# ===================================================
# All the types; one object; objects near a distance modulus
# ===================================================


def get_objects(object_types: Sequence[str] = OBJECT_TYPES) -> pd.DataFrame:
    """Objects of several types, in one table.

    Parameters
    ----------
    object_types : sequence of str, optional
        Among ``OBJECT_TYPES``: ``"lmc"``, ``"smc"``, ``"dwarfs"`` (but the
        Magellanic Clouds), ``"gc"`` (globular clusters), ``"galaxies"``
        (Siena Galaxy Atlas).

    Returns
    -------
    objects : pandas.DataFrame
        Columns ``type``, then ``OBJECT_COLUMNS``.

    Raises
    ------
    ValueError
        A type not in ``OBJECT_TYPES``.
    """
    unknown = set(object_types) - set(OBJECT_TYPES)
    if unknown:
        raise ValueError(f"object types {sorted(unknown)} not in {OBJECT_TYPES}.")
    getters = {"lmc": get_lmc, "smc": get_smc, "dwarfs": get_dwarf_galaxies, "gc": get_globular_clusters,
               "galaxies": get_galaxies}
    tables = [getters[object_type]().assign(type=object_type) for object_type in object_types]
    tables = [table for table in tables if not table.empty] or tables[:1]
    if not tables:
        return pd.DataFrame(columns=["type", *OBJECT_COLUMNS])
    return pd.concat(tables, ignore_index=True)[["type", *OBJECT_COLUMNS]]


def find_object(name: str, object_types: Sequence[str] = ("lmc", "smc", "dwarfs", "gc")) -> pd.Series:
    """One object, by its key or name; case, spaces, underscores and hyphens ignored.

    ``"NGC 1851"``, ``"ngc_1851"`` and ``"ngc1851"`` are the same.

    Parameters
    ----------
    name : str
        Key or name.
    object_types : sequence of str, optional
        Types searched (``get_objects``); by default the LVDB ones.

    Returns
    -------
    object : pandas.Series
        Its row of ``get_objects``.

    Raises
    ------
    KeyError
        No such object; the message lists close names.
    ValueError
        Several objects of different keys have this name.
    """
    objects = get_objects(object_types)
    wanted = simplified_name(name)
    match = objects[(objects["key"].map(simplified_name) == wanted) | (objects["name"].map(simplified_name) == wanted)]
    match = match.drop_duplicates("key")
    if match.empty:
        names = {str(label).lower(): str(label) for label in [*objects["key"], *objects["name"]]}
        close = [names[label] for label in difflib.get_close_matches(name.lower(), list(names), n=5)]
        raise KeyError(f"No object {name!r}. Close keys and names: {close}")
    if len(match) > 1:
        raise ValueError(f"Several objects named {name!r}: keys {match['key'].tolist()}")
    return match.iloc[0]


def simplified_name(name: Any) -> str:
    """Name without case, spaces, underscores and hyphens: "NGC 1851", "ngc_1851" -> "ngc1851"."""
    return str(name).lower().replace(" ", "").replace("_", "").replace("-", "")


def objects_near_distance_modulus(
    distance_modulus: float,
    max_offset: float,
    *,
    object_types: Sequence[str] = ("dwarfs", "gc"),
    catalog: str | None = None,
    data_path: Any = None,
) -> pd.DataFrame:
    """Objects whose distance modulus is within ``max_offset`` of ``distance_modulus``.

    Parameters
    ----------
    distance_modulus : float
        Distance modulus of interest (mag).
    max_offset : float
        Largest ``|object distance modulus - distance_modulus|`` (mag).
    object_types : sequence of str, optional
        See ``get_objects``.
    catalog : str, optional
        Key of ``catalogs.euclid_catalog.CATALOGS``: adds
        ``n_sources_at_centre``, the sources of the catalog in the HEALPix
        pixel of the centre of each object (``catalog_point_map``, nside
        1024 for DR1, about 3.4 arcmin), and ``in_catalog`` (at least one).
    data_path : str or Path, optional
        Folder of the HATS catalog, when it is not the default one.

    Returns
    -------
    objects : pandas.DataFrame
        The columns of ``get_objects``, plus ``dm_offset`` (object minus
        ``distance_modulus``), sorted by ``|dm_offset|``.
    """
    objects = get_objects(object_types)
    objects["dm_offset"] = objects["distance_modulus"] - distance_modulus
    objects = objects[objects["dm_offset"].abs() <= max_offset]
    objects = objects.iloc[np.argsort(objects["dm_offset"].abs().to_numpy(), kind="stable")].reset_index(drop=True)
    if catalog is not None:
        counts = catalog_point_map(catalog, data_path)
        pixels = hp.ang2pix(hp.npix2nside(counts.size), objects["ra"].to_numpy(dtype=float),
                            objects["dec"].to_numpy(dtype=float), nest=True, lonlat=True)
        objects["n_sources_at_centre"] = counts[pixels].astype(int)
        objects["in_catalog"] = objects["n_sources_at_centre"] > 0
    return objects


def _download(url: str, path: Path) -> None:
    """Download url to path: a complete file, or none."""
    LOGGER.info("Downloading %s to %s.", url, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(".partial")
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as answer, partial_path.open("wb") as file:
        shutil.copyfileobj(answer, file)
    partial_path.replace(path)

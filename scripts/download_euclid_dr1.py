#!/usr/bin/env python
"""Download the Euclid DR1 MER final catalog and write it as a local HATS catalog.

The output has the same layout as the Q1 catalog used before (``hats.properties``
+ ``dataset/Norder=*/Dir=*/Npix=*.parquet``), with upper-case column names and no
suffix, so it is read by ``catalogs.euclid_catalog`` (catalog ``"euclid_dr1"``).

Two sources, chosen with ``--source``:

``hats`` (preferred)
    Copy a HATS catalog served by IRSA in the cloud (anonymous S3), keeping only
    :data:`EUCLID_COLUMNS`. Q1 is at ``s3://nasa-irsa-euclid-q1/contributed/q1/
    merged_objects/hats`` (https://irsa.ipac.caltech.edu/cloud_access/). The DR1
    bucket is ``nasa-irsa-euclid-dr1`` (Caltech-IPAC/irsa-tutorials PR #407), but
    on 2026-11-12 it was not yet listable anonymously and no DR1 HATS path was
    published: pass it with ``--hats-url`` once it is.

``tap``
    Query a TAP service (ESA ``https://eas.esac.esa.int/tap-server/tap`` or IRSA
    ``https://irsa.ipac.caltech.edu/TAP``) one MER tile at a time, several tiles at
    once, save one parquet file per tile (resumable: done tiles are skipped), then
    build the HATS catalog with hats-import. The DR1
    wide-survey MER catalog is ``dr1.mer_catalogue_wide_survey`` on ESA TAP
    (:data:`DEFAULT_TABLE`; public, 888 M sources in 8947 tiles, checked on
    2026-11-12). IRSA TAP had only Q1 on that date.

Both sources add ``EBV_SFD98``, the Schlegel, Finkbeiner & Davis (1998) E(B-V) at
the source position, read with ``dustmaps`` (Green 2018, JOSS 3, 695). The
streamobs extinction coefficients apply to SFD98, not to the Euclid ``GAL_EBV``
column (see ``catalogs.euclid_catalog.catalog_columns``).

Examples::

    # check that the columns exist, then exit
    python scripts/download_euclid_dr1.py --source tap --validate-only
    python scripts/download_euclid_dr1.py --source tap --list-tables

    # trial on 20 tiles, then the full download (about 4-5 h, about 220 GB)
    python scripts/download_euclid_dr1.py --source tap --max-tiles 20 --output /tmp/trial
    python scripts/download_euclid_dr1.py --source tap

    # add the VIS template-fitting fluxes (ADDED_COLUMNS) to the downloaded catalog
    python scripts/download_euclid_dr1.py --source tap --add-columns

Requirements: lsdb, hats-import, dask[distributed], pyvo (tap), dustmaps, and the
two SFD98 maps in :data:`SFD_DIR`. ``dustmaps.sfd.fetch()`` gets them from Harvard
Dataverse, which answered 403 on 2026-11-12; the same files are at
``https://github.com/kbarbary/sfddata/raw/master/SFD_dust_4096_{ngp,sgp}.fits``.
"""

from __future__ import annotations

import argparse
import functools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parent / "data" / "euclid_dr1"
CATALOG_NAME = "euclid_dr1_mer"
SFD_DIR = HERE.parent / "data" / "dustmaps" / "sfd"  # SFD_dust_4096_{ngp,sgp}.fits

TAP_URLS = {
    "esa": "https://eas.esac.esa.int/tap-server/tap",
    "irsa": "https://irsa.ipac.caltech.edu/TAP",
}
DEFAULT_TABLE = "dr1.mer_catalogue_wide_survey"  # on ESA TAP

RA_COLUMN = "RIGHT_ASCENSION"
DEC_COLUMN = "DECLINATION"
EBV_COLUMN = "EBV_SFD98"

# ---------------------------------------------------------------- columns ----
# Same Euclid columns as the Q1 x DES Y6 crossmatch
# (data/euclid_q1_des/crossmatch_euclid_q1_x_des_y6.py). Column documentation:
# Q1 MER Final Catalog data product description,
# https://euclid.esac.esa.int/dr/q1/dpdd/merdpd/dpcards/mer_finalcatalog.html

_BANDS_APER = ["VIS", "Y", "J", "H", "NIR_STACK"]
_DECAM = ["U_EXT_DECAM", "G_EXT_DECAM", "R_EXT_DECAM", "I_EXT_DECAM", "Z_EXT_DECAM"]

EUCLID_COLUMNS = [
    # identity + astrometry
    "OBJECT_ID", RA_COLUMN, DEC_COLUMN,
    "SEGMENTATION_MAP_ID", "PARENT_ID", "GAIA_ID", "GAIA_MATCH_QUALITY",
    # detection / quality flags
    "VIS_DET", "DET_QUALITY_FLAG", "SPURIOUS_FLAG", "SPURIOUS_PROB",
    "POINT_LIKE_FLAG", "POINT_LIKE_PROB", "EXTENDED_FLAG", "EXTENDED_PROB",
    # VIS reference photometry
    "FLUX_VIS_PSF", "FLUXERR_VIS_PSF", "FLUX_VIS_SERSIC", "FLUXERR_VIS_SERSIC",
    "FLUX_DETECTION_TOTAL", "FLUXERR_DETECTION_TOTAL",
    # morphology
    "FWHM", "MUMAX_MINUS_MAG", "MAG_STARGAL_SEP", "ELLIPTICITY",
    # extinction (Euclid value, kept for comparison with EBV_SFD98)
    "GAL_EBV", "GAL_EBV_ERR",
]
EUCLID_COLUMNS += [f"FLUX_{b}_2FWHM_APER" for b in _BANDS_APER]
EUCLID_COLUMNS += [f"FLUXERR_{b}_2FWHM_APER" for b in _BANDS_APER]
EUCLID_COLUMNS += [f"FLUX_{b}_TEMPLFIT" for b in ("Y", "J", "H")]
EUCLID_COLUMNS += [f"FLUXERR_{b}_TEMPLFIT" for b in ("Y", "J", "H")]
# VIS template-fitting fluxes. DR1 has no FLUX_VIS_TEMPLFIT: TPHOT measures VIS on the
# VIS image PSF-matched to each other band, so that VIS - Y (J, H) colours use
# FLUX_VIS_TO_Y_TEMPLFIT and FLUX_Y_TEMPLFIT (checked in TAP_SCHEMA on 2026-11-18).
# Added on 2026-11-18 to the catalog downloaded on 2026-11-12 with --add-columns.
VIS_TEMPLFIT_COLUMNS = [f"FLUX_VIS_TO_{b}_TEMPLFIT" for b in ("Y", "J", "H")]
VIS_TEMPLFIT_COLUMNS += [f"FLUXERR_VIS_TO_{b}_TEMPLFIT" for b in ("Y", "J", "H")]
EUCLID_COLUMNS += VIS_TEMPLFIT_COLUMNS
# DECam external photometry: in Q1 and in dr1.mer_catalogue_wide_survey
# (2026-11-12). Kept optional, so that a source without them still works.
OPTIONAL_COLUMNS = [f"FLUX_{b}_TEMPLFIT" for b in _DECAM]
OPTIONAL_COLUMNS += [f"FLUXERR_{b}_TEMPLFIT" for b in _DECAM]


# ------------------------------------------------------------ name mapping ----


def match_columns(available, wanted, prefix="", optional=()):
    """Map the wanted (upper-case) names to the names of the source catalog.

    The match ignores case and an optional prefix (e.g. ``mer_`` in a merged
    catalog), so that a lower-case TAP table or a prefixed HATS catalog work.

    Parameters
    ----------
    available : iterable of str
        Column names of the source.
    wanted : list of str
        Required columns, upper case.
    prefix : str
        Prefix of the source names, ignored in the match.
    optional : list of str
        Columns kept when present, skipped otherwise.

    Returns
    -------
    dict
        ``{source_name: wanted_name}``.
    """
    lookup = {}
    for name in available:
        key = name.upper()
        if prefix and key.startswith(prefix.upper()):
            key = key[len(prefix):]
        lookup.setdefault(key, name)
    missing = [c for c in wanted if c not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} column(s) not in the source: {missing}")
    skipped = [c for c in optional if c not in lookup]
    if skipped:
        print(f"optional columns not in the source, skipped: {skipped}")
    keep = list(wanted) + [c for c in optional if c in lookup]
    return {lookup[c]: c for c in keep}


# -------------------------------------------------------------- extinction ----


@functools.cache
def _sfd_query():
    """dustmaps SFD98 query on the maps of :data:`SFD_DIR`, loaded once."""
    from dustmaps.sfd import SFDQuery

    return SFDQuery(map_dir=str(SFD_DIR))


def sfd_ebv(ra, dec):
    """E(B-V) of the SFD98 map at (ra, dec) in degrees, via dustmaps."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    coords = SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg)
    return _sfd_query()(coords).astype(np.float32)


def add_sfd_ebv(frame):
    """Add the :data:`EBV_COLUMN` column to a DataFrame, in place."""
    if len(frame):
        frame[EBV_COLUMN] = sfd_ebv(frame[RA_COLUMN], frame[DEC_COLUMN])
    else:
        frame[EBV_COLUMN] = pd.Series(dtype=np.float32)
    return frame


# ---------------------------------------------------------------- hats copy ----


def copy_hats(args, output):
    """Copy the selected columns of a cloud HATS catalog to ``output``."""
    import lsdb

    storage = {"anon": True} if args.hats_url.startswith("s3://") else None
    catalog = lsdb.open_catalog(args.hats_url, storage_options=storage)
    mapping = match_columns(
        catalog.all_columns, EUCLID_COLUMNS, args.prefix, OPTIONAL_COLUMNS
    )
    print(f"source: {catalog.hc_structure.catalog_info.total_rows:,} rows, "
          f"{len(mapping)} columns validated")
    if args.validate_only:
        return

    catalog = lsdb.open_catalog(
        args.hats_url, columns=list(mapping), storage_options=storage
    ).rename(columns=mapping)
    if args.where:
        catalog = catalog.query(args.where)
    catalog = catalog.map_partitions(add_sfd_ebv)

    with _dask_client(args):
        t0 = time.time()
        catalog.write_catalog(output, catalog_name=CATALOG_NAME,
                              overwrite=args.overwrite)
        print(f"wrote {output} in {time.time() - t0:,.0f} s")


# --------------------------------------------------------------- tap query ----


# One query per MER tile (``tile_index``, indexed on ESA TAP): no source is in two
# queries, and the download resumes tile by tile. Sync CSV requests, not async jobs:
# on 2026-11-12 a median tile (107 k rows) took 10 s in sync, > 7 min as an async job;
# 8 tiles at once gave 56 k rows/s, 15 at once only 65 k rows/s.

TILE_COLUMN = "tile_index"
FLOAT64_COLUMNS = (RA_COLUMN, DEC_COLUMN)  # the other floating-point columns -> float32
_INT_DTYPES = {"long": "Int64", "int": "Int32", "short": "Int16"}


def column_dtypes(datatypes, mapping):
    """pandas dtypes of the selected columns, from their TAP datatypes.

    Integers are nullable (pandas ``Int64``...), so that ids with nulls (e.g.
    ``GAIA_ID``) keep every digit instead of becoming float64. Floating-point
    columns are float32, except :data:`FLOAT64_COLUMNS`.

    Parameters
    ----------
    datatypes : dict
        ``{source_name: TAP datatype}`` (``TAP_SCHEMA.columns.datatype``).
    mapping : dict
        ``{source_name: wanted_name}``, from :func:`match_columns`.

    Returns
    -------
    dict
        ``{source_name: pandas dtype}``.
    """
    dtypes = {}
    for source, name in mapping.items():
        datatype = datatypes[source]
        if datatype in _INT_DTYPES:
            dtypes[source] = _INT_DTYPES[datatype]
        elif datatype in ("double", "float"):
            dtypes[source] = "float64" if name in FLOAT64_COLUMNS else "float32"
        else:
            raise ValueError(f"{source}: TAP datatype {datatype!r} not handled")
    return dtypes


def _tile_queries(table, mapping, tile, where, dec_edges=()):
    """ADQL queries of one tile: one, or one per declination strip.

    Parameters
    ----------
    table : str
        TAP table.
    mapping : dict
        ``{source_name: wanted_name}``, from :func:`match_columns`.
    tile : int
        Value of :data:`TILE_COLUMN`.
    where : str or None
        Extra ADQL condition.
    dec_edges : sequence of float
        Inner edges of the declination strips (deg), increasing. The first strip
        has no lower bound and the last no upper bound: every source of the tile is
        in exactly one strip. Empty: one query for the whole tile.

    Returns
    -------
    list of str
    """
    dec = next(k for k, v in mapping.items() if v == DEC_COLUMN)
    bounds = [None, *dec_edges, None]
    queries = []
    for low, high in zip(bounds[:-1], bounds[1:]):
        conditions = [f"{TILE_COLUMN} = {tile}"]
        if low is not None:
            conditions.append(f"{dec} >= {float(low):.10f}")
        if high is not None:
            conditions.append(f"{dec} < {float(high):.10f}")
        if where:
            conditions.append(f"({where})")
        queries.append(f"SELECT {', '.join(mapping)} FROM {table} WHERE {' AND '.join(conditions)}")
    return queries


def dec_strip_edges(service, table, mapping, tile, n_strips):
    """Inner edges of ``n_strips`` declination strips of equal width over a tile."""
    dec = next(k for k, v in mapping.items() if v == DEC_COLUMN)
    limits = service.search(
        f"SELECT MIN({dec}) AS dec_min, MAX({dec}) AS dec_max FROM {table} "
        f"WHERE {TILE_COLUMN} = {tile}"
    ).to_table()
    edges = np.linspace(float(limits["dec_min"][0]), float(limits["dec_max"][0]), n_strips + 1)
    return list(edges[1:-1])


def download_tile(url, queries, dtypes, mapping, path, expected=None, retries=3, add_ebv=True):
    """Run sync TAP queries (CSV) of one tile and write the result, with EBV_SFD98, to ``path``.

    The file is written under a temporary name, then renamed: an existing ``path``
    is always complete. A failed attempt (request error, error answer, unreadable
    CSV, or not ``expected`` rows in total) is retried after 30 s, 60 s...

    Parameters
    ----------
    url : str
        TAP service URL (``/sync`` is appended).
    queries : str or list of str
        ADQL query, or queries of parts of the tile (:func:`_tile_queries`), whose
        results are put together.
    dtypes, mapping : dict
        From :func:`column_dtypes` and :func:`match_columns`.
    path : Path
        Output parquet file.
    expected : int, optional
        Number of rows the tile must have (no cut): a different number raises.
    retries : int
        Number of attempts.
    add_ebv : bool
        Add the :data:`EBV_COLUMN` column (False when only columns are added to an
        existing catalog, see :func:`add_columns`).

    Returns
    -------
    int
        Number of rows written.
    """
    import io

    import requests

    def fetch(query):
        params = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
        response = requests.get(f"{url}/sync", params=params, timeout=1800)
        response.raise_for_status()
        if response.content.startswith(b"<"):  # VOTable error instead of CSV
            raise RuntimeError(response.text[:1000])
        return pd.read_csv(io.BytesIO(response.content), dtype=dtypes)

    queries = [queries] if isinstance(queries, str) else list(queries)
    for attempt in range(1, retries + 1):
        try:
            parts = [fetch(query) for query in queries]
            frame = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
            frame = frame.rename(columns=mapping)
            # A transfer cut short can still be a valid CSV: check the number of rows
            if expected is not None and len(frame) != expected:
                raise RuntimeError(f"{len(frame):,} rows instead of {expected:,}")
            break
        except (requests.RequestException, RuntimeError, ValueError):  # ValueError: truncated CSV
            if attempt == retries:
                raise
            time.sleep(30 * attempt)

    if add_ebv:
        add_sfd_ebv(frame)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.rename(path)
    return len(frame)


def tile_list(service, table, path):
    """Tiles of ``table`` with their number of sources, cached in ``path`` (CSV).

    One async query over the whole table (about 1 min for DR1).
    """
    if path.exists():
        return pd.read_csv(path)
    job = service.submit_job(
        f"SELECT {TILE_COLUMN}, COUNT(*) AS n FROM {table} GROUP BY {TILE_COLUMN}",
        maxrec=1_000_000,
    )
    job.run().wait(phases=["COMPLETED", "ERROR", "ABORTED"])
    job.raise_if_error()
    tiles = job.fetch_result().to_table().to_pandas()
    job.delete()
    tiles = tiles.sort_values(TILE_COLUMN).reset_index(drop=True)
    tiles.to_csv(path, index=False)
    return tiles


def query_tap(args, output):
    """Download the catalog tile by tile from TAP, then import it as HATS.

    With ``--add-columns``, add :data:`ADDED_COLUMNS` to the existing catalog instead
    (:func:`add_columns`).
    """
    import pyvo

    service = pyvo.dal.TAPService(TAP_URLS.get(args.tap, args.tap))
    if args.list_tables:
        result = service.search(
            "SELECT schema_name, table_name FROM TAP_SCHEMA.tables "
            "WHERE table_name LIKE '%mer%' OR table_name LIKE '%MER%'"
        )
        print(result.to_table())
        return
    # Column names from TAP_SCHEMA: "SELECT TOP 1 *" fails on ESA DR1, whose VOTable
    # declares an invalid size for calblock_id_list (astropy error E01).
    schema = service.search(
        "SELECT column_name, datatype FROM TAP_SCHEMA.columns "
        f"WHERE table_name = '{args.table}'"
    ).to_table()
    if len(schema) == 0:
        raise SystemExit(f"{args.table} not found (see --list-tables)")
    datatypes = dict(zip(schema["column_name"], schema["datatype"]))
    mapping = match_columns(list(datatypes), EUCLID_COLUMNS, args.prefix, OPTIONAL_COLUMNS)
    dtypes = column_dtypes(datatypes, mapping)
    print(f"{args.table}: {len(mapping)} columns validated")
    if args.validate_only:
        return

    raw_dir = output.parent / f"{output.name}_tap_tiles"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tiles = tile_list(service, args.table, raw_dir / "tiles.csv")
    if args.add_columns:
        add_columns(service, args, datatypes, tiles, raw_dir, output / CATALOG_NAME)
        return
    files = download_tiles(service, args, mapping, dtypes, tiles, raw_dir)
    import_hats([f for f in files.values() if pq.ParquetFile(f).metadata.num_rows],
                output, args)


def download_tiles(service, args, mapping, dtypes, tiles, raw_dir, add_ebv=True):
    """Download the tiles of ``tiles`` to ``raw_dir/tile_<index>.parquet``, several at once.

    Done tiles (existing files) are skipped. Raises ``SystemExit`` if a tile failed:
    running again retries it.

    Parameters
    ----------
    service : pyvo.dal.TAPService
    args : argparse.Namespace
        Command-line options (``table``, ``tap``, ``where``, ``max_tiles``,
        ``parallel_tiles``, ``max_rows_per_query``).
    mapping, dtypes : dict
        From :func:`match_columns` and :func:`column_dtypes`.
    tiles : pd.DataFrame
        :data:`TILE_COLUMN` and ``n`` (number of sources), from :func:`tile_list`.
    raw_dir : Path
        Folder of the tile files.
    add_ebv : bool
        Passed to :func:`download_tile`.

    Returns
    -------
    dict
        ``{tile: file}`` of every tile of ``tiles`` (after ``max_tiles``).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if args.max_tiles:
        tiles = tiles.iloc[: args.max_tiles]
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {t: raw_dir / f"tile_{t}.parquet" for t in tiles[TILE_COLUMN]}
    todo = tiles[[not files[t].exists() for t in tiles[TILE_COLUMN]]]
    print(f"{len(tiles):,} tiles ({tiles['n'].sum():,} sources), "
          f"{len(tiles) - len(todo):,} already done, {args.parallel_tiles} at once")

    # Large tiles in declination strips: ESA cut the answer of the largest DR1 tile
    # (787,520 rows) short three times in a row, while tiles of 330 k rows came whole.
    queries = {}
    for tile, n in zip(todo[TILE_COLUMN], todo["n"]):
        n_strips = int(np.ceil(n / args.max_rows_per_query))
        edges = dec_strip_edges(service, args.table, mapping, tile, n_strips) if n_strips > 1 else ()
        queries[tile] = _tile_queries(args.table, mapping, tile, args.where, edges)
        if n_strips > 1:
            print(f"tile {tile} ({n:,} sources): {n_strips} declination strips")

    url = TAP_URLS.get(args.tap, args.tap)
    t0 = time.time()
    n_todo, n_done, failed = todo["n"].sum(), 0, []
    with ProcessPoolExecutor(args.parallel_tiles) as pool:
        futures = {
            pool.submit(
                download_tile, url, queries[tile], dtypes, mapping, files[tile],
                None if args.where else n, add_ebv=add_ebv,
            ): (tile, n)
            for tile, n in zip(todo[TILE_COLUMN], todo["n"])
        }
        for k, future in enumerate(as_completed(futures), 1):
            tile, n = futures[future]
            n_done += n
            try:
                rows = future.result()
            except Exception as error:  # reported, retried by the next run
                failed.append(tile)
                print(f"[{k}/{len(todo)}] tile {tile} FAILED: {error}", flush=True)
                continue
            rate = n_done / (time.time() - t0)
            print(f"[{k}/{len(todo)}] tile {tile}: {rows:,} rows | {rate:,.0f} sources/s"
                  f" | {(n_todo - n_done) / rate / 3600:.1f} h left", flush=True)

    if failed:
        raise SystemExit(f"{len(failed)} tile(s) failed: {failed}. Run again to retry them.")
    return files


def import_hats(files, output, args):
    """Build a HATS catalog from parquet files with hats-import."""
    from hats_import.catalog.arguments import ImportArguments
    from hats_import.pipeline import pipeline_with_client

    import_args = ImportArguments(
        output_path=output,
        output_artifact_name=CATALOG_NAME,
        input_file_list=[str(f) for f in files],
        file_reader="parquet",
        ra_column=RA_COLUMN,
        dec_column=DEC_COLUMN,
        sort_columns="OBJECT_ID",
        pixel_threshold=args.pixel_threshold,
        resume=True,
    )
    with _dask_client(args) as client:
        pipeline_with_client(import_args, client)
    print(f"wrote {output / CATALOG_NAME}")


# ------------------------------------------------------------- add columns ----


# Columns added to the catalog downloaded on 2026-11-12, for every source. They are
# downloaded tile by tile with OBJECT_ID and the position, which gives the HATS
# partition of each source (the same CSV values, parsed the same way as on
# 2026-11-12, so the same _healpix_29), then joined on OBJECT_ID partition by
# partition. A fresh download has them already (they are in EUCLID_COLUMNS).
ADDED_COLUMNS = VIS_TEMPLFIT_COLUMNS
# Columns downloaded with the new ones; also the join key of the new rows and the
# catalog (see add_columns_to_partition)
KEY_COLUMNS = ["OBJECT_ID", RA_COLUMN, DEC_COLUMN]


def catalog_schema(catalog_dir):
    """Arrow schema of a HATS catalog, from ``dataset/_common_metadata``."""
    return pq.read_schema(Path(catalog_dir) / "dataset" / "_common_metadata")


def partition_ranges(catalog_dir):
    """Partitions of a HATS catalog and their ``_healpix_29`` ranges.

    Returns
    -------
    pixels : list of HealpixPixel
        Partitions, sorted by ``_healpix_29``.
    ranges : np.ndarray
        ``(n, 2)`` ``[start, end)`` of each partition at order 29.
    """
    import hats

    tree = hats.read_hats(catalog_dir).pixel_tree
    return tree.get_healpix_pixels(), tree.to_depth29_ranges()


def split_by_partition(files, catalog_dir, shard_dir, columns):
    """Write the rows of the tile files in one folder per HATS partition of the catalog.

    Each tile file ``tile_<i>.parquet`` gives ``<shard_dir>/<order>_<pixel>/tile_<i>.parquet``
    with :data:`KEY_COLUMNS`, ``_healpix_29`` and ``columns``. A tile already split
    (all its shard files written, flagged by ``<shard_dir>/done/tile_<i>``) is skipped.

    Raises
    ------
    RuntimeError
        If a source is in no partition of the catalog.
    """
    from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN, compute_spatial_index

    pixels, ranges = partition_ranges(catalog_dir)
    done_dir = Path(shard_dir) / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in files if not (done_dir / Path(f).stem).exists()]
    t0 = time.time()
    for k, path in enumerate(files, 1):
        frame = pd.read_parquet(path)
        index = compute_spatial_index(frame[RA_COLUMN].to_numpy(), frame[DEC_COLUMN].to_numpy())
        where = np.searchsorted(ranges[:, 0], index, side="right") - 1
        inside = (where >= 0) & (index < ranges[np.maximum(where, 0), 1])
        if not inside.all():
            raise RuntimeError(f"{path.name}: {np.sum(~inside)} source(s) in no partition")
        frame = frame[[*KEY_COLUMNS, *columns]]
        frame.insert(1, SPATIAL_INDEX_COLUMN, index)
        for w in np.unique(where):
            pixel = pixels[w]
            folder = Path(shard_dir) / f"{pixel.order}_{pixel.pixel}"
            folder.mkdir(exist_ok=True)
            frame[where == w].to_parquet(folder / Path(path).name, index=False)
        (done_dir / Path(path).stem).touch()
        if k % 200 == 0 or k == len(files):
            print(f"split {k}/{len(files)} tiles ({time.time() - t0:,.0f} s)", flush=True)


def add_columns_to_partition(partition_file, shard_files, columns):
    """Add ``columns`` of the shard files to one HATS partition file.

    Rows are joined on :data:`KEY_COLUMNS`, (``OBJECT_ID``, RA, Dec): the positions
    are the same float64 values on both sides (the same ESA CSV, parsed the same
    way). The key is unique in DR1, while OBJECT_ID is not (3 repeats in all the
    partitions, checked on 2026-11-18): e.g. -714456772190573290 is two NIR-only
    sources of adjacent tiles, a few micro-arcsec apart, and -769285311390246657 a
    VIS and a NIR-only source of the same tile. SEGMENTATION_MAP_ID is not unique
    either (347 repeats), and its first digits are not always the tile of the source
    (1,551 sources of tile 102036133 have 102036134). The file is rewritten under a
    temporary name, then renamed. A file that has the columns already is left as it
    is.

    Returns
    -------
    int
        Number of rows of the partition, 0 if it had the columns already.

    Raises
    ------
    RuntimeError
        If the shards do not have exactly the sources of the partition, each once,
        at the same ``_healpix_29``.
    """
    from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

    partition_file = Path(partition_file)
    if set(columns) <= set(pq.read_schema(partition_file).names):
        return 0
    if not shard_files:
        raise RuntimeError(f"{partition_file}: no new rows")
    # ParquetFile, not pq.read_table: read_table takes Norder=/Dir= of the path for
    # partition columns and adds them to the table
    table = pq.ParquetFile(partition_file).read()
    new = pd.concat([pd.read_parquet(f) for f in shard_files], ignore_index=True)
    new = new.set_index(KEY_COLUMNS)
    sources = pd.MultiIndex.from_arrays([table[c].to_numpy() for c in KEY_COLUMNS],
                                        names=KEY_COLUMNS)
    repeated = new.index.duplicated().sum() + sources.duplicated().sum()
    found = sources.isin(new.index)
    if repeated or len(new) != len(sources) or not found.all():
        raise RuntimeError(f"{partition_file}: {len(new):,} new rows for {len(sources):,} "
                           f"sources, {repeated} repeated {tuple(KEY_COLUMNS)}, "
                           f"{(~found).sum()} sources without new row")
    new = new.reindex(sources)
    if not np.array_equal(new[SPATIAL_INDEX_COLUMN].to_numpy(),
                          table[SPATIAL_INDEX_COLUMN].to_numpy()):
        raise RuntimeError(f"{partition_file}: positions differ from the new rows")
    for column in columns:
        table = table.append_column(column, pa.array(new[column].to_numpy(), type=pa.float32()))
    tmp = partition_file.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.rename(partition_file)
    return len(sources)


def merge_columns(catalog_dir, shard_dir, columns, workers=4):
    """Add ``columns`` to every partition of the catalog, then rewrite its metadata files.

    Resumable: the partitions that have the columns already are skipped. ``workers``
    partitions at once (1: in this process).
    """
    from concurrent.futures import ProcessPoolExecutor
    from itertools import repeat

    from hats.io import paths
    from hats.io.parquet_metadata import write_parquet_metadata

    pixels, _ = partition_ranges(catalog_dir)
    partition_files, shards = [], []
    for pixel in pixels:
        folder = Path(shard_dir) / f"{pixel.order}_{pixel.pixel}"
        partition_files.append(paths.pixel_catalog_file(catalog_dir, pixel).path)
        shards.append(sorted(folder.glob("tile_*.parquet")) if folder.exists() else [])
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    t0, rows = time.time(), 0
    try:
        counts = (pool.map if pool else map)(
            add_columns_to_partition, partition_files, shards, repeat(columns))
        for k, count in enumerate(counts, 1):
            rows += count
            if k % 100 == 0 or k == len(pixels):
                print(f"merged {k}/{len(pixels)} partitions, {rows:,} rows "
                      f"({time.time() - t0:,.0f} s)", flush=True)
    finally:
        if pool:
            pool.shutdown()
    total = write_parquet_metadata(catalog_dir)
    print(f"{catalog_dir}: metadata rewritten, {total:,} rows")


def add_columns(service, args, datatypes, tiles, raw_dir, catalog_dir, columns=ADDED_COLUMNS):
    """Download ``columns`` for every source and add them to the HATS catalog ``catalog_dir``.

    Three resumable steps: download (``<raw_dir>/added_columns``), split by partition
    (``<raw_dir>/added_columns_partitions``), merge into the partition files. With
    ``--max-tiles`` only the download runs (trial). The two folders can be deleted
    afterwards.
    """
    if args.where:
        raise SystemExit("--add-columns fills every source: --where is not allowed")
    if set(columns) <= set(catalog_schema(catalog_dir).names):
        print(f"{catalog_dir} has the columns already: {list(columns)}")
        return
    mapping = match_columns(list(datatypes), KEY_COLUMNS + list(columns), args.prefix)
    dtypes = column_dtypes(datatypes, mapping)
    print(f"adding {list(columns)} to {catalog_dir}")
    files = download_tiles(service, args, mapping, dtypes, tiles,
                           raw_dir / "added_columns", add_ebv=False)
    if args.max_tiles:
        print("trial (--max-tiles): the catalog is not changed")
        return
    shard_dir = raw_dir / "added_columns_partitions"
    split_by_partition(files.values(), catalog_dir, shard_dir, columns)
    merge_columns(catalog_dir, shard_dir, columns, args.workers)


# --------------------------------------------------------------------- main ----


class _dask_client:
    """Local dask cluster as a context manager."""

    def __init__(self, args):
        self.args = args

    def __enter__(self):
        from dask.distributed import Client, LocalCluster

        self.cluster = LocalCluster(n_workers=self.args.workers, threads_per_worker=1,
                                    memory_limit=self.args.memory_limit)
        self.client = Client(self.cluster)
        print(f"dask dashboard: {self.client.dashboard_link}")
        return self.client

    def __exit__(self, *exc):
        self.client.close()
        self.cluster.close()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=["hats", "tap"], required=True)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help="output folder (default: data/euclid_dr1)")
    ap.add_argument("--hats-url", help="HATS catalog to copy (--source hats)")
    ap.add_argument("--tap", default="esa",
                    help="'esa', 'irsa' or a TAP URL (--source tap)")
    ap.add_argument("--table", default=DEFAULT_TABLE,
                    help=f"TAP table of the DR1 MER catalog (default: {DEFAULT_TABLE})")
    ap.add_argument("--list-tables", action="store_true",
                    help="list the MER tables of the TAP service, then exit")
    ap.add_argument("--prefix", default="",
                    help="prefix of the source column names, e.g. 'mer_'")
    ap.add_argument("--where", default=None,
                    help="row cut: ADQL in source names for tap (e.g. 'point_like_prob >= 0.5 "
                         "AND point_like_prob <= 1': on ESA, NaN passes 'point_like_prob > 0.5'), "
                         "pandas query in upper-case names for hats")
    ap.add_argument("--parallel-tiles", type=int, default=8,
                    help="tiles downloaded at once (tap; 8 is near ESA's maximum speed)")
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="only the first N tiles (tap), for a trial run")
    ap.add_argument("--max-rows-per-query", type=int, default=300_000,
                    help="larger tiles are queried in declination strips (tap)")
    ap.add_argument("--pixel-threshold", type=int, default=1_000_000,
                    help="max rows per HATS partition")
    ap.add_argument("--workers", type=int, default=4, help="dask workers of the HATS step")
    ap.add_argument("--memory-limit", default="3GB", help="memory per dask worker")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--add-columns", action="store_true",
                    help="add ADDED_COLUMNS (VIS template-fitting fluxes) to the existing "
                         "catalog in --output, for every source (tap)")
    ap.add_argument("--validate-only", action="store_true",
                    help="check the column names, then exit")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.source == "hats":
        if not args.hats_url:
            raise SystemExit("--hats-url is required with --source hats")
        copy_hats(args, args.output)
    else:
        query_tap(args, args.output)


if __name__ == "__main__":
    main()

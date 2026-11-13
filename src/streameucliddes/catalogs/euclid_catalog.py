"""Read the Euclid catalogs, alone or crossmatched with DES Y6 Gold.

Four catalogs, chosen by name (keys of :data:`CATALOGS`):

- ``"euclid_dr1"``: Euclid DR1 wide-survey MER catalog, written by
  ``scripts/download_euclid_dr1.py`` in ``data/euclid_dr1``;
- ``"euclid_q1"``: Euclid Q1 MER catalog, same layout, in ``data/euclid_q1``
  (not downloaded yet);
- ``"euclid_q1_des"``: Euclid Q1 x DES Y6 Gold, written by
  ``data/euclid_q1_des/crossmatch_euclid_q1_x_des_y6.py`` (LSDB nearest neighbour
  within 1 arcsec);
- ``"euclid_dr1_des"``: Euclid DR1 x DES Y6 Gold, same layout, in
  ``data/euclid_dr1_des`` (not built yet).

The catalogs are HATS catalogs, read here with pyarrow: only the requested columns
are loaded, and cuts on catalog columns (and an optional RA/Dec box) are applied
while reading. Euclid-only catalogs have the MER column names; in the crossmatched
catalogs the Euclid columns end with ``_euclid`` and the DES columns with ``_des``.

Harmonised magnitudes follow the streamobs conventions, so the real data can be
compared directly with injected streams:

- columns ``<namespace>_<band>_obs`` / ``<namespace>_<band>_err``, with the
  namespace of the streamobs survey (``euclid_q1``, ``euclid_dr1``, ``des_yr6``);
- the photometry of the streamobs selection functions: Euclid ``FLUX_VIS_PSF`` and
  ``FLUX_{Y,J,H}_TEMPLFIT``, DES ``BDF_MAG_*``;
- Euclid magnitudes dereddened with E(B-V) of Schlegel et al. 1998 (SFD98) and the
  streamobs extinction coefficients; DES magnitudes are already dereddened by DES;
- the same default star selection: ``POINT_LIKE_PROB >= 0.5`` and
  ``SPURIOUS_FLAG == 0`` for Euclid, ``0 <= EXT_XGB <= 1`` for DES.

Two Euclid star selections (:data:`STAR_SELECTIONS`, see :func:`default_star_cuts`):
``"morphology_cuts"`` (default), cuts on the Euclid morphology columns, and
``"xgboost_classifier"``, a cut on the star probability of the XGBoost classifier
trained on Euclid Q1 (:mod:`~streameucliddes.catalogs.star_classifier`), added as
the column ``xgboost_star_prob`` (:func:`add_xgboost_star_prob`).

Column documentation:

- Euclid: MER Final Catalog data product description,
  https://euclid.esac.esa.int/dr/q1/dpdd/merdpd/dpcards/mer_finalcatalog.html
- DES: Y6 Gold paper, Bechtol et al. 2025 (Tables 3-4 and Appendix B),
  https://arxiv.org/abs/2501.05739, and https://des.ncsa.illinois.edu/releases/y6a2/Y6gold
- ``_dist_arcsec``, ``_healpix_29``: added by LSDB / hats-import.
"""

import functools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from streamobs.surveys import Survey

from streameucliddes.catalogs import star_classifier
from streameucliddes.catalogs.catalog_cuts import get_cut_mask
from streameucliddes.unit_conversions import flux_to_mag

# The data/ folder of the repository (gitignored).
DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# euclid_release: streamobs Euclid release (namespace, extinction coefficients,
# saturation); with_des: crossmatched with DES Y6 Gold; data_path: default folder.
CATALOGS = {
    "euclid_dr1": {"euclid_release": "dr1", "with_des": False, "data_path": DATA_DIR / "euclid_dr1"},
    "euclid_q1": {"euclid_release": "q1", "with_des": False, "data_path": DATA_DIR / "euclid_q1"},
    "euclid_q1_des": {"euclid_release": "q1", "with_des": True, "data_path": DATA_DIR / "euclid_q1_des"},
    "euclid_dr1_des": {"euclid_release": "dr1", "with_des": True, "data_path": DATA_DIR / "euclid_dr1_des"},
}

DES_NAMESPACE = "des_yr6"
DES_BANDS = ("g", "r", "i", "z")
# DES magnitudes outside this range are sentinels (-99, 37.5, -9.999e9).
DES_VALID_MAG_RANGE = (0.0, 37.5)

# Rows per batch when open_euclid_catalog reads by batches (batch_selection).
READ_BATCH_SIZE = 1_000_000

# Euclid fluxes and errors per band, in micro-Jansky. There is no FLUX_VIS_TEMPLFIT:
# the template fitting measures the NISP bands with the VIS image as prior, VIS being
# measured by PSF fitting.
EUCLID_FLUX_COLUMNS = {
    "VIS": ("FLUX_VIS_PSF", "FLUXERR_VIS_PSF"),  # VIS PSF-fitting flux
    "Y": ("FLUX_Y_TEMPLFIT", "FLUXERR_Y_TEMPLFIT"),  # NISP Y template-fitting flux
    "J": ("FLUX_J_TEMPLFIT", "FLUXERR_J_TEMPLFIT"),  # NISP J template-fitting flux
    "H": ("FLUX_H_TEMPLFIT", "FLUXERR_H_TEMPLFIT"),  # NISP H template-fitting flux
}
# Euclid columns read by default, besides the fluxes, RA/Dec and E(B-V).
EUCLID_DEFAULT_COLUMNS = [
    "OBJECT_ID",  # Euclid unique source identifier
    "VIS_DET",  # 1: detected in VIS, 0: detected in NIR only
    "DET_QUALITY_FLAG",  # detection-step bitmask (0: no flag)
    "SPURIOUS_FLAG",  # 1: spurious source
    "SPURIOUS_PROB",  # probability of being spurious
    "POINT_LIKE_PROB",  # probability of being point-like (NaN for NIR-only sources)
    "ELLIPTICITY",  # ellipticity of the source (0: circular, 1: linear)
    "MUMAX_MINUS_MAG",  # difference between the peak surface brightness and the total magnitude
    "MAG_STARGAL_SEP",  # star-galaxy separation magnitude (a star classifier feature)
]
# DES columns read by default (crossmatched catalogs), besides the magnitudes.
DES_DEFAULT_COLUMNS = [
    "COADD_OBJECT_ID_des",  # DES unique object identifier
    "_dist_arcsec",  # Euclid - DES separation [arcsec]
    "EXT_XGB_des",  # XGBoost class: 0 high-confidence star, 1 likely star, 2-4 galaxies, -9 no data
    "FLAGS_GOLD_des",  # quality bitmask: fitvd/SExtractor flags, saturation, ... (0: clean)
    "FLAGS_FOREGROUND_des",  # foreground bitmask: bright stars, large galaxies, ... (0: none)
    "FLAGS_FOOTPRINT_des",  # 1: inside the Y6 Gold footprint
]
# Cut format: see catalog_cuts.get_cut_mask.
DES_QUALITY_CUTS = {
    "FLAGS_FOOTPRINT_des": 1,  # inside the Y6 Gold footprint
    "FLAGS_FOREGROUND_des": 0,  # no foreground object
    "FLAGS_GOLD_des": 0,  # no quality flag
}
DES_STAR_CUTS = {"EXT_XGB_des": (0, 1)}  # high-confidence or likely stars

EUCLID_QUALITY_CUTS = {
    "SPURIOUS_FLAG": 0,  # not spurious
    "SPURIOUS_PROB": (None, 0.2),  # not likely spurious
    "DET_QUALITY_FLAG": 0,  # no detection flag
}

# Euclid star selections, see default_star_cuts.
STAR_SELECTIONS = ("morphology_cuts", "xgboost_classifier")
# "morphology_cuts": cuts on the Euclid morphology columns.
MORPHOLOGY_STAR_CUTS = {
    "POINT_LIKE_PROB": (0.5, None),  # point-like
    "ELLIPTICITY": (None, 0.2),  # not elongated
    "MUMAX_MINUS_MAG": (-3.25, -2.7),  # within the expected range
}
# "xgboost_classifier": minimum star probability of the classifier (star_classifier),
# in the column added by add_xgboost_star_prob.
XGBOOST_MIN_STAR_PROB = 0.9
XGBOOST_STAR_PROB_COLUMN = "xgboost_star_prob"

# ===================================================
# Catalog description
# ===================================================


def _catalog_info(catalog):
    """Entry of :data:`CATALOGS`, with a clear error for an unknown name."""
    if catalog not in CATALOGS:
        raise ValueError(f"Unknown catalog {catalog!r}: choose one of {list(CATALOGS)}")
    return CATALOGS[catalog]


@functools.cache
def euclid_survey(catalog):
    """
    streamobs Euclid survey of the catalog's release.

    Its namespace names the magnitude columns, and its extinction coefficients and
    VIS saturation are used here, so they follow any update of streamobs.

    Parameters
    ----------
    catalog : str
        Key of :data:`CATALOGS`.

    Returns
    -------
    streamobs.surveys.Survey
    """
    release = _catalog_info(catalog)["euclid_release"]
    return Survey.load("euclid", release=release, verbose=False)


def euclid_column(catalog, name):
    """Name of the Euclid column ``name`` in ``catalog`` (``_euclid`` suffix if crossmatched)."""
    return f"{name}_euclid" if _catalog_info(catalog)["with_des"] else name


def catalog_columns(catalog):
    """
    Names of the columns with a role in ``catalog``.

    Parameters
    ----------
    catalog : str
        Key of :data:`CATALOGS`.

    Returns
    -------
    dict
        ``ra``, ``dec``: Euclid source barycentre [deg]; ``ebv``: SFD98 E(B-V) (the
        streamobs coefficients are A_band / E(B-V)_SFD, so the Euclid ``GAL_EBV``
        column does not apply); ``euclid_fluxes``: ``{band: (flux, error)}``;
        ``des_magnitudes``: ``{band: (magnitude, error)}``, empty without DES.
    """
    with_des = _catalog_info(catalog)["with_des"]
    return {
        "ra": euclid_column(catalog, "RIGHT_ASCENSION"),
        "dec": euclid_column(catalog, "DECLINATION"),
        # Euclid-only catalogs: added by scripts/download_euclid_dr1.py;
        # crossmatched catalogs: the DES column.
        "ebv": "EBV_SFD98_des" if with_des else "EBV_SFD98",
        "euclid_fluxes": {
            band: (euclid_column(catalog, flux), euclid_column(catalog, error))
            for band, (flux, error) in EUCLID_FLUX_COLUMNS.items()
        },
        # BDF_MAG_<B>_CORRECTED: DES bulge+disk model magnitude, dereddened by DES (SFD98)
        "des_magnitudes": {
            band: (f"BDF_MAG_{band.upper()}_CORRECTED_des", f"BDF_MAG_ERR_{band.upper()}_des")
            for band in DES_BANDS
        } if with_des else {},
    }


def default_columns(catalog):
    """
    Columns read by default: identifiers, positions, classification, quality flags
    and the photometry used by :func:`add_magnitudes`.

    Parameters
    ----------
    catalog : str
        Key of :data:`CATALOGS`.

    Returns
    -------
    list of str
    """
    roles = catalog_columns(catalog)
    columns = [euclid_column(catalog, "OBJECT_ID"), roles["ra"], roles["dec"]]
    columns += [euclid_column(catalog, c) for c in EUCLID_DEFAULT_COLUMNS[1:]]
    if roles["des_magnitudes"]:
        columns += DES_DEFAULT_COLUMNS
    columns.append(roles["ebv"])
    columns += [c for pair in roles["euclid_fluxes"].values() for c in pair]
    columns += [c for pair in roles["des_magnitudes"].values() for c in pair]
    return columns


def default_star_cuts(
    catalog, star_selection="morphology_cuts", min_point_like_prob=None, min_xgboost_star_prob=None,
):
    """
    Star selection cuts (format of :func:`~streameucliddes.catalogs.catalog_cuts.get_cut_mask`).

    Always: Euclid quality cuts (not spurious, no detection flag) and fainter than
    the VIS saturation (brighter, ``POINT_LIKE_PROB`` is undefined); with DES, also
    the DES quality flags and DES high-confidence or likely stars. Then, according
    to ``star_selection``:

    - ``"morphology_cuts"``: the cuts of :data:`MORPHOLOGY_STAR_CUTS` on
      ``POINT_LIKE_PROB``, ``ELLIPTICITY`` and ``MUMAX_MINUS_MAG``;
    - ``"xgboost_classifier"``: star probability of the XGBoost classifier
      (column ``xgboost_star_prob``) at least :data:`XGBOOST_MIN_STAR_PROB`.

    The returned dict can be edited to change any cut, e.g.
    ``cuts[euclid_column(catalog, "ELLIPTICITY")] = (None, 0.3)``.

    Parameters
    ----------
    catalog : str
        Key of :data:`CATALOGS`.
    star_selection : str
        One of :data:`STAR_SELECTIONS`.
    min_point_like_prob : float, optional
        Keep ``POINT_LIKE_PROB >= min_point_like_prob``, whatever ``star_selection``
        (it replaces the ``"morphology_cuts"`` threshold). None: the threshold of
        the selection, no cut with ``"xgboost_classifier"``.
    min_xgboost_star_prob : float, optional
        Keep ``xgboost_star_prob >= min_xgboost_star_prob``, whatever
        ``star_selection``. None: :data:`XGBOOST_MIN_STAR_PROB` with
        ``"xgboost_classifier"``, no cut with ``"morphology_cuts"``.

    Returns
    -------
    dict
        ``get_euclid_stars`` adds the ``xgboost_star_prob`` column when a cut uses it.
    """
    if star_selection not in STAR_SELECTIONS:
        raise ValueError(f"Unknown star_selection {star_selection!r}: choose one of {STAR_SELECTIONS}")
    survey = euclid_survey(catalog)
    cuts = {}
    if _catalog_info(catalog)["with_des"]:
        cuts.update(DES_QUALITY_CUTS)
        cuts.update(DES_STAR_CUTS)

    # Euclid column names without suffix: "_euclid" added for crossmatched catalogs
    cuts.update({euclid_column(catalog, c): spec for c, spec in EUCLID_QUALITY_CUTS.items()})
    cuts[f"{survey.namespace}_VIS_obs"] = (survey.saturation["VIS"], None)  # not saturated

    if star_selection == "morphology_cuts":
        cuts.update({euclid_column(catalog, c): spec for c, spec in MORPHOLOGY_STAR_CUTS.items()})
    else:
        cuts[XGBOOST_STAR_PROB_COLUMN] = (XGBOOST_MIN_STAR_PROB, None)
    if min_point_like_prob is not None:
        cuts[euclid_column(catalog, "POINT_LIKE_PROB")] = (min_point_like_prob, None)
    if min_xgboost_star_prob is not None:
        cuts[XGBOOST_STAR_PROB_COLUMN] = (min_xgboost_star_prob, None)

    return cuts


# ===================================================
# Reading
# ===================================================


def _catalog_dir(data_path):
    """Folder of the HATS catalog (the one with ``hats.properties``) under ``data_path``."""
    data_path = Path(data_path)
    if (data_path / "dataset").is_dir():
        return data_path
    properties = sorted(data_path.rglob("hats.properties"))
    if not properties:
        raise FileNotFoundError(f"No HATS catalog (hats.properties) found under {data_path}")
    return properties[0].parent


def _check_partitions(dataset_dir):
    """Warn when partitions listed in the catalog metadata are missing on disk."""
    metadata_file = dataset_dir / "_metadata"
    if not metadata_file.exists():
        return
    metadata = pq.read_metadata(metadata_file)
    rows = {}
    for i in range(metadata.num_row_groups):
        row_group = metadata.row_group(i)
        path = row_group.column(0).file_path
        rows[path] = rows.get(path, 0) + row_group.num_rows
    missing = {p: n for p, n in rows.items() if not (dataset_dir / p).exists()}
    if missing:
        warnings.warn(
            f"{len(missing)}/{len(rows)} partitions of the catalog are missing in {dataset_dir} "
            f"({sum(missing.values()):,} of {metadata.num_rows:,} rows): {sorted(missing)}"
        )


def _cut_expression(column, spec):
    """pyarrow filter expression of one cut (format of ``get_cut_mask``)."""
    field = ds.field(column)
    if isinstance(spec, tuple):
        low, high = spec
        expression = None
        if low is not None:
            expression = field >= low
        if high is not None:
            expression = field <= high if expression is None else expression & (field <= high)
        return expression
    if isinstance(spec, (list, set)):
        return field.isin(list(spec))
    return field == spec


def _box_expression(ra_column, dec_column, ra_range, dec_range):
    """pyarrow filter expression of an RA/Dec box; ``ra_range`` may cross RA = 0."""
    expressions = []
    if ra_range is not None:
        ra_min, ra_max = ra_range
        ra = ds.field(ra_column)
        if ra_min <= ra_max:
            expressions.append((ra >= ra_min) & (ra <= ra_max))
        else:  # e.g. (350, 10)
            expressions.append((ra >= ra_min) | (ra <= ra_max))
    if dec_range is not None:
        expressions.append(_cut_expression(dec_column, tuple(dec_range)))
    return expressions


def available_columns(catalog, data_path=None):
    """
    List the columns of a catalog.

    Parameters
    ----------
    catalog : str
        Key of :data:`CATALOGS`.
    data_path : str or Path, optional
        Folder containing the HATS catalog; default: the one of :data:`CATALOGS`.

    Returns
    -------
    list of str
    """
    data_path = _catalog_info(catalog)["data_path"] if data_path is None else data_path
    return pq.read_schema(_catalog_dir(data_path) / "dataset" / "_common_metadata").names


def open_euclid_catalog(
    catalog, columns=None, cuts=None, ra_range=None, dec_range=None, data_path=None,
    verbose=True, batch_selection=None,
):
    """
    Read a Euclid catalog.

    Parameters
    ----------
    catalog : str
        Key of :data:`CATALOGS`.
    columns : list of str, "all" or None
        Catalog columns to read. None reads :func:`default_columns`; "all" reads
        every column.
    cuts : dict, optional
        Cuts on catalog columns, applied while reading (format of
        :func:`~streameucliddes.catalogs.catalog_cuts.get_cut_mask`; NaN never
        passes). The cut columns do not need to be in ``columns``.
    ra_range, dec_range : (float, float), optional
        RA/Dec box in degrees (inclusive). ``ra_range`` may cross RA = 0, e.g.
        ``(350, 10)``. Parts of the catalog outside the box are not read. None: no
        limit.
    data_path : str or Path, optional
        Folder containing the HATS catalog; default: the one of :data:`CATALOGS`.
    verbose : bool
        Print the number of rows read.
    batch_selection : callable, optional
        Read the catalog by batches of :data:`READ_BATCH_SIZE` rows and keep only
        ``batch_selection(table)`` of each batch: ``table`` is a pyarrow Table, the
        result a pd.DataFrame (e.g. converted with :func:`_to_pandas`). For a
        selection that the cuts on catalog columns cannot do while reading, e.g.
        the star classifier (:func:`get_euclid_stars`), without having all the
        rows in memory. None: the whole selection is read at once.

    Returns
    -------
    pd.DataFrame
        64-bit integer columns with nulls (e.g. ``GAIA_ID``) are pandas ``Int64``:
        as float64 they would lose digits.
    """
    data_path = _catalog_info(catalog)["data_path"] if data_path is None else data_path
    dataset_dir = _catalog_dir(data_path) / "dataset"
    _check_partitions(dataset_dir)
    dataset = ds.dataset(dataset_dir, format="parquet", partitioning="hive", exclude_invalid_files=True)
    names = [c for c in dataset.schema.names if c not in ("Norder", "Dir", "Npix")]

    if columns is None:
        columns = default_columns(catalog)
    elif columns == "all":
        columns = names
    cuts = cuts or {}
    missing = [c for c in list(columns) + list(cuts) if c not in names]
    if missing:
        raise KeyError(f"Column(s) not in the catalog: {missing}. See available_columns().")

    roles = catalog_columns(catalog)
    expressions = [_cut_expression(column, spec) for column, spec in cuts.items()]
    expressions += _box_expression(roles["ra"], roles["dec"], ra_range, dec_range)
    expression = None
    for cut in expressions:
        expression = cut if expression is None else expression & cut

    if batch_selection is None:
        data = _to_pandas(dataset.to_table(columns=list(columns), filter=expression))
    else:
        scanner = dataset.scanner(columns=list(columns), filter=expression, batch_size=READ_BATCH_SIZE)
        selected = [batch_selection(pa.Table.from_batches([b])) for b in scanner.to_batches() if b.num_rows]
        selected = [s for s in selected if len(s)]  # pandas warns when concatenating empty tables
        if not selected:  # nothing selected: an empty table with the right columns
            selected = [batch_selection(scanner.projected_schema.empty_table())]
        data = pd.concat(selected, ignore_index=True)
    if verbose:
        print(f"Read {len(data):,} rows x {len(data.columns)} columns of {catalog}")
    return data


def _to_pandas(table):
    """
    pandas version of a pyarrow table.

    pyarrow turns an integer column with nulls into float64, which loses the digits
    of 64-bit ids beyond 2**53 (e.g. GAIA_ID): those are kept as nullable Int64.
    """
    data = table.to_pandas()
    for name in table.column_names:
        column = table.column(name)
        if pa.types.is_int64(column.type) and column.null_count:
            data[name] = column.to_pandas(types_mapper={pa.int64(): pd.Int64Dtype()}.get)
    return data


# ===================================================
# Magnitudes
# ===================================================


def magnitude_columns(catalog):
    """
    Columns added by :func:`add_magnitudes` with its default bands.

    Parameters
    ----------
    catalog : str
        Key of :data:`CATALOGS`.

    Returns
    -------
    list of str
        ``ra``, ``dec`` and the ``<namespace>_<band>_obs`` / ``_err`` columns.
    """
    namespace = euclid_survey(catalog).namespace
    columns = ["ra", "dec"]
    for prefix, bands in [(DES_NAMESPACE, catalog_columns(catalog)["des_magnitudes"]),
                          (namespace, EUCLID_FLUX_COLUMNS)]:
        columns += [f"{prefix}_{band}_{kind}" for band in bands for kind in ("obs", "err")]
    return columns


def add_magnitudes(data, catalog, euclid_bands=tuple(EUCLID_FLUX_COLUMNS), des_bands=None):
    """
    Add harmonised, dereddened magnitudes and ``ra``/``dec`` columns, in place.

    Adds ``<euclid namespace>_<band>_obs`` / ``_err`` from the Euclid fluxes
    (non-positive fluxes set to NaN), dereddened with the SFD98 E(B-V) and the
    extinction coefficients of the streamobs Euclid survey, and, for crossmatched
    catalogs, ``des_yr6_<band>_obs`` / ``_err`` from the DES BDF magnitudes
    (sentinels set to NaN). ``ra``/``dec`` are the Euclid positions.

    Parameters
    ----------
    data : pd.DataFrame
        Catalog read with :func:`open_euclid_catalog`.
    catalog : str
        Key of :data:`CATALOGS`.
    euclid_bands : sequence of str
        Euclid bands to convert.
    des_bands : sequence of str, optional
        DES bands to convert; None: all of them with DES, none without.

    Returns
    -------
    pd.DataFrame
        ``data``, with the new columns.
    """
    roles = catalog_columns(catalog)
    if des_bands is None:
        des_bands = tuple(roles["des_magnitudes"])
    elif des_bands and not roles["des_magnitudes"]:
        raise ValueError(f"{catalog} has no DES magnitudes")
    needed = [roles["ra"], roles["dec"]]
    needed += [c for b in des_bands for c in roles["des_magnitudes"][b]]
    if euclid_bands:
        needed += [c for b in euclid_bands for c in roles["euclid_fluxes"][b]] + [roles["ebv"]]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise KeyError(f"Column(s) needed to compute the magnitudes are missing: {missing}")

    data["ra"] = data[roles["ra"]]
    data["dec"] = data[roles["dec"]]

    for band in des_bands:
        mag_col, err_col = roles["des_magnitudes"][band]
        mag = data[mag_col].to_numpy(dtype=float)
        valid = (mag > DES_VALID_MAG_RANGE[0]) & (mag < DES_VALID_MAG_RANGE[1])
        data[f"{DES_NAMESPACE}_{band}_obs"] = np.where(valid, mag, np.nan)
        data[f"{DES_NAMESPACE}_{band}_err"] = np.where(valid, data[err_col], np.nan)

    survey = euclid_survey(catalog)
    if euclid_bands:
        ebv = data[roles["ebv"]].to_numpy(dtype=float)
    for band in euclid_bands:
        flux_col, err_col = roles["euclid_fluxes"][band]
        flux = data[flux_col].to_numpy(dtype=float)
        flux_err = data[err_col].to_numpy(dtype=float)
        valid = flux > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = flux_to_mag(np.where(valid, flux, np.nan) * 1e-6)  # micro-Jy -> Jy
            mag_err = 2.5 / np.log(10) * flux_err / flux
        data[f"{survey.namespace}_{band}_obs"] = mag - survey.coeff_extinc[band] * ebv
        data[f"{survey.namespace}_{band}_err"] = np.where(valid, mag_err, np.nan)

    return data


# ===================================================
# Star classifier
# ===================================================


def add_xgboost_star_prob(data, catalog):
    """
    Add the column ``xgboost_star_prob`` in place: the probability of being a star
    from the XGBoost classifier (:func:`~streameucliddes.catalogs.star_classifier.star_probability`).

    The classifier features are the Euclid columns ``ELLIPTICITY``,
    ``MUMAX_MINUS_MAG``, ``POINT_LIKE_PROB`` and ``MAG_STARGAL_SEP`` (read by
    default). Missing values (e.g. ``POINT_LIKE_PROB`` of NIR-only sources) are
    handled by the classifier.

    Parameters
    ----------
    data : pd.DataFrame
        Catalog read with :func:`open_euclid_catalog`.
    catalog : str
        Key of :data:`CATALOGS`.

    Returns
    -------
    pd.DataFrame
        ``data``, with the new column (float32).
    """
    data[XGBOOST_STAR_PROB_COLUMN] = _xgboost_star_prob(data, catalog)
    return data


def _xgboost_star_prob(data, catalog):
    """Classifier probability of each row of ``data`` (pd.DataFrame or pyarrow Table)."""
    available = data.column_names if isinstance(data, pa.Table) else data.columns
    feature_names = star_classifier.load_star_classifier().feature_names
    columns = {name: euclid_column(catalog, name.upper()) for name in feature_names}
    missing = [c for c in columns.values() if c not in available]
    if missing:
        raise KeyError(f"Column(s) needed by the star classifier are missing: {missing}")
    return star_classifier.star_probability({name: np.asarray(data[c]) for name, c in columns.items()})


def _xgboost_batch_selection(catalog, spec):
    """
    ``batch_selection`` of :func:`open_euclid_catalog` for the classifier: keeps the
    rows whose ``xgboost_star_prob`` passes the cut ``spec``, with that column.

    The classifier runs on the pyarrow table: only the kept rows are converted to
    pandas.
    """
    def select(table):
        probability = _xgboost_star_prob(table, catalog)
        column = XGBOOST_STAR_PROB_COLUMN
        keep = get_cut_mask(pd.DataFrame({column: probability}), {column: spec})
        data = _to_pandas(table.filter(pa.array(keep)))
        data[XGBOOST_STAR_PROB_COLUMN] = probability[keep]
        return data

    return select


# ===================================================
# Stars
# ===================================================


def get_euclid_stars(
    catalog, data=None, cuts=None, columns=None, ra_range=None, dec_range=None,
    data_path=None, verbose=True, subset=None,
):
    """
    Get the stars of a Euclid catalog, with harmonised magnitudes.

    Parameters
    ----------
    catalog : str
        Key of :data:`CATALOGS`.
    data : pd.DataFrame, optional
        Catalog already read with :func:`open_euclid_catalog`. If None, the catalog
        is read, applying the cuts on catalog columns while reading.
    cuts : dict, optional
        Cuts to apply (format of
        :func:`~streameucliddes.catalogs.catalog_cuts.get_cut_mask`); they may use
        the columns added by :func:`add_magnitudes` and ``xgboost_star_prob``
        (:func:`add_xgboost_star_prob`, added only when a cut uses it). None applies
        :func:`default_star_cuts`; ``{}`` applies no cut. Other star selections:
        e.g. ``cuts=default_star_cuts(catalog, "xgboost_classifier", min_xgboost_star_prob=0.95)``.
    columns : list of str, "all" or None
        Catalog columns to read when ``data`` is None (see :func:`open_euclid_catalog`).
    ra_range, dec_range : (float, float), optional
        RA/Dec box in degrees, when ``data`` is None (see :func:`open_euclid_catalog`).
        None: the whole catalog.
    data_path : str or Path, optional
        Folder containing the HATS catalog, when ``data`` is None.
    verbose : bool
        Print the number of selected stars.

    Returns
    -------
    pd.DataFrame
        The selected rows, with the :func:`add_magnitudes` columns.
    """
    cuts = default_star_cuts(catalog) if cuts is None else cuts

    if subset is not None:
        if ra_range is not None or dec_range is not None:
            raise ValueError("subset cannot be used with ra_range or dec_range")
        if subset == "north":
            ra_range = (0, 360)
            dec_range = (0, 90)
        elif subset == "south":
            ra_range = (0, 360)
            dec_range = (-90, 0)
        elif subset == "northwest":
            ra_range = (180, 360)
            dec_range = (0, 90)
        elif subset == "northeast":
            ra_range = (0, 180)
            dec_range = (0, 90)
        elif subset == "southwest":
            ra_range = (180, 360)
            dec_range = (-90, 0)
        elif subset == "southeast":
            ra_range = (0, 180)
            dec_range = (-90, 0)
        else:
            raise ValueError(f"Unknown subset {subset!r}: choose one of "
                             "'north', 'south', 'northwest', 'northeast', "
                             "'southwest', 'southeast'")
    if data is None:
        names = set(available_columns(catalog, data_path))
        derived = magnitude_columns(catalog) + [XGBOOST_STAR_PROB_COLUMN]
        unknown = [c for c in cuts if c not in names and c not in derived]
        if unknown:
            raise KeyError(
                f"Cut column(s) neither in the catalog nor added by add_magnitudes or "
                f"add_xgboost_star_prob: {unknown}"
            )
        # Cuts on catalog columns while reading, the others (magnitudes) after.
        read_cuts = {c: s for c, s in cuts.items() if c in names}
        cuts = {c: s for c, s in cuts.items() if c not in names}
        # Star classifier: its cut is applied while reading, batch by batch, since
        # the rows passing only the cuts on catalog columns may not fit in memory.
        batch_selection = None
        if XGBOOST_STAR_PROB_COLUMN in cuts:
            batch_selection = _xgboost_batch_selection(catalog, cuts[XGBOOST_STAR_PROB_COLUMN])
        data = open_euclid_catalog(
            catalog, columns=columns, cuts=read_cuts, ra_range=ra_range,
            dec_range=dec_range, data_path=data_path, verbose=verbose,
            batch_selection=batch_selection,
        )
        n_input = None
    else:
        n_input = len(data)

    if f"{euclid_survey(catalog).namespace}_VIS_obs" not in data.columns:
        add_magnitudes(data, catalog)
    if XGBOOST_STAR_PROB_COLUMN in cuts and XGBOOST_STAR_PROB_COLUMN not in data.columns:
        add_xgboost_star_prob(data, catalog)

    stars = data[get_cut_mask(data, cuts)].reset_index(drop=True)
    if verbose:
        origin = f" out of {n_input:,}" if n_input is not None else ""
        print(f"Selected {len(stars):,} stars{origin}")
    return stars

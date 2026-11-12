"""Load and prepare the Euclid DR1 MER final catalog (Euclid only, no DES).

The catalog is the HATS catalog written by ``scripts/download_euclid_dr1.py``:
the MER final catalog columns, upper case and without suffix, plus ``EBV_SFD98``.
It is read with pyarrow as in :mod:`streameucliddes.catalogs.euclid_des_catalog`,
whose reading helpers are reused: only the requested columns are loaded, and cuts
on catalog columns are applied while reading.

Harmonised magnitudes follow the streamobs conventions, as for Q1:

- columns ``<namespace>_<band>_obs`` / ``<namespace>_<band>_err``, dereddened with
  ``EBV_SFD98`` and the streamobs extinction coefficients;
- the photometry of the streamobs Euclid selection function: ``FLUX_VIS_PSF`` and
  ``FLUX_{Y,J,H}_TEMPLFIT``;
- the same default star selection: ``POINT_LIKE_PROB > 0.5``, ``SPURIOUS_FLAG == 0``
  and fainter than the VIS saturation.

The streamobs survey is the Euclid DR1 one when streamobs has it, Q1 otherwise
(with a warning): the namespace, extinction coefficients and saturation limit then
stay those of Q1, so that real data and streams injected with streamobs share the
same column names.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from streamobs.surveys import Survey

from streameucliddes.catalogs.catalog_cuts import get_cut_mask
from streameucliddes.catalogs.euclid_des_catalog import (
    _check_partitions,
    _cut_expression,
    _dataset_dir,
)
from streameucliddes.unit_conversions import flux_to_mag

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "euclid_dr1"

# Column documentation: MER Final Catalog data product description. The DR1 page
# was not checked; the Q1 one is
# https://euclid.esac.esa.int/dr/q1/dpdd/merdpd/dpcards/mer_finalcatalog.html


def _load_survey():
    """streamobs Euclid survey: DR1 if streamobs has it, Q1 otherwise."""
    try:
        return Survey.load("euclid", release="dr1", verbose=False)
    except Exception:  # streamobs raises different errors for an unknown release
        warnings.warn(
            "streamobs has no Euclid DR1 survey: using the Q1 one (namespace, "
            "extinction coefficients and VIS saturation of Q1)."
        )
        return Survey.load("euclid", release="q1", verbose=False)


EUCLID_SURVEY = _load_survey()
EUCLID_NAMESPACE = EUCLID_SURVEY.namespace

# Euclid fluxes and errors, in micro-Jansky (see euclid_des_catalog for the choice).
EUCLID_FLUX_COLUMNS = {
    "VIS": ("FLUX_VIS_PSF", "FLUXERR_VIS_PSF"),  # VIS PSF-fitting flux
    "Y": ("FLUX_Y_TEMPLFIT", "FLUXERR_Y_TEMPLFIT"),  # NISP Y template-fitting flux
    "J": ("FLUX_J_TEMPLFIT", "FLUXERR_J_TEMPLFIT"),  # NISP J template-fitting flux
    "H": ("FLUX_H_TEMPLFIT", "FLUXERR_H_TEMPLFIT"),  # NISP H template-fitting flux
}
# E(B-V) of Schlegel et al. 1998, added by scripts/download_euclid_dr1.py: the
# streamobs coefficients are A_band / E(B-V)_SFD. "GAL_EBV" (Euclid value, not SFD)
# can be passed to add_magnitudes instead, but the coefficients then do not apply.
EBV_COLUMN = "EBV_SFD98"
RA_COLUMN = "RIGHT_ASCENSION"  # source barycentre RA [deg]
DEC_COLUMN = "DECLINATION"  # source barycentre Dec [deg]

# Brighter than this, POINT_LIKE_PROB is undefined.
EUCLID_VIS_SATURATION = EUCLID_SURVEY.saturation["VIS"]

DEFAULT_COLUMNS = [
    "OBJECT_ID",  # unique source identifier
    RA_COLUMN,
    DEC_COLUMN,
    "VIS_DET",  # 1: detected in VIS, 0: detected in NIR only
    "DET_QUALITY_FLAG",  # detection-step bitmask (0: no flag)
    "SPURIOUS_FLAG",  # 1: spurious source
    "POINT_LIKE_PROB",  # probability of being point-like (NaN for NIR-only sources)
    EBV_COLUMN,
]
DEFAULT_COLUMNS += [c for pair in EUCLID_FLUX_COLUMNS.values() for c in pair]

# Cut format: see catalog_cuts.get_cut_mask.
DEFAULT_STAR_CUTS = {
    "POINT_LIKE_PROB": (0.5, None),  # point-like
    "SPURIOUS_FLAG": 0,  # not spurious
    f"{EUCLID_NAMESPACE}_VIS_obs": (EUCLID_VIS_SATURATION, None),  # not saturated
}


# ===================================================
# Reading
# ===================================================


def available_columns(data_path=DEFAULT_DATA_PATH):
    """
    List the columns of the Euclid catalog.

    Parameters
    ----------
    data_path : str or Path
        Folder containing the HATS catalog.

    Returns
    -------
    list of str
    """
    return pq.read_schema(_dataset_dir(data_path) / "_common_metadata").names


def open_euclid_data(columns=None, data_path=DEFAULT_DATA_PATH, cuts=None, verbose=True):
    """
    Read the Euclid DR1 MER catalog.

    Parameters
    ----------
    columns : list of str, "all" or None
        Catalog columns to read. None reads :data:`DEFAULT_COLUMNS`; "all" reads
        every column.
    data_path : str or Path
        Folder containing the HATS catalog.
    cuts : dict, optional
        Cuts on catalog columns, applied while reading (format of
        :func:`~streameucliddes.catalogs.catalog_cuts.get_cut_mask`). The cut
        columns do not need to be in ``columns``.
    verbose : bool
        Print the number of rows read.

    Returns
    -------
    pd.DataFrame
    """
    dataset_dir = _dataset_dir(data_path)
    _check_partitions(dataset_dir)
    dataset = ds.dataset(
        dataset_dir, format="parquet", partitioning="hive", exclude_invalid_files=True
    )
    catalog_columns = [
        c for c in dataset.schema.names if c not in ("Norder", "Dir", "Npix")
    ]

    if columns is None:
        columns = DEFAULT_COLUMNS
    elif columns == "all":
        columns = catalog_columns
    cuts = cuts or {}
    missing = [c for c in list(columns) + list(cuts) if c not in catalog_columns]
    if missing:
        raise KeyError(
            f"Column(s) not in the catalog: {missing}. See available_columns()."
        )

    expression = None
    for column, spec in cuts.items():
        cut = _cut_expression(column, spec)
        expression = cut if expression is None else expression & cut

    table = dataset.to_table(columns=list(columns), filter=expression)
    data = table.to_pandas()
    # pyarrow turns an integer column with nulls into float64, which loses the
    # digits of 64-bit ids beyond 2**53 (e.g. GAIA_ID): keep those as nullable Int64.
    for name in table.column_names:
        column = table.column(name)
        if pa.types.is_int64(column.type) and column.null_count:
            data[name] = column.to_pandas(types_mapper={pa.int64(): pd.Int64Dtype()}.get)
    if verbose:
        print(
            f"Read {len(data):,} rows x {len(data.columns)} columns "
            f"from {dataset_dir.parent.name}"
        )
    return data


# ===================================================
# Magnitudes
# ===================================================


def add_magnitudes(data, bands=tuple(EUCLID_FLUX_COLUMNS), ebv_column=EBV_COLUMN):
    """
    Add harmonised, dereddened magnitudes and ``ra``/``dec`` columns, in place.

    Adds ``<namespace>_<band>_obs`` / ``_err`` from the fluxes (non-positive fluxes
    set to NaN), dereddened with ``ebv_column`` and the extinction coefficients of
    the streamobs Euclid survey (``EUCLID_SURVEY.coeff_extinc``).

    Parameters
    ----------
    data : pd.DataFrame
        Catalog read with :func:`open_euclid_data`.
    bands : sequence of str
        Bands to convert.
    ebv_column : str
        E(B-V) column. The coefficients are calibrated on SFD98 (the default).

    Returns
    -------
    pd.DataFrame
        ``data``, with the new columns.
    """
    needed = [RA_COLUMN, DEC_COLUMN, ebv_column]
    needed += [c for b in bands for c in EUCLID_FLUX_COLUMNS[b]]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise KeyError(
            f"Column(s) needed to compute the magnitudes are missing: {missing}"
        )

    data["ra"] = data[RA_COLUMN]
    data["dec"] = data[DEC_COLUMN]
    ebv = data[ebv_column].to_numpy(dtype=float)

    for band in bands:
        flux_col, err_col = EUCLID_FLUX_COLUMNS[band]
        flux = data[flux_col].to_numpy(dtype=float)
        flux_err = data[err_col].to_numpy(dtype=float)
        valid = flux > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = flux_to_mag(np.where(valid, flux, np.nan) * 1e-6)  # micro-Jy -> Jy
            mag_err = 2.5 / np.log(10) * flux_err / flux
        data[f"{EUCLID_NAMESPACE}_{band}_obs"] = (
            mag - EUCLID_SURVEY.coeff_extinc[band] * ebv
        )
        data[f"{EUCLID_NAMESPACE}_{band}_err"] = np.where(valid, mag_err, np.nan)

    return data


# ===================================================
# Cuts
# ===================================================


def get_euclid_stars(
    data=None, cuts=None, columns=None, data_path=DEFAULT_DATA_PATH, verbose=True
):
    """
    Get the stars of the Euclid DR1 catalog, with harmonised magnitudes.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Catalog already read with :func:`open_euclid_data`. If None, the catalog is
        read from ``data_path``, applying the cuts on catalog columns while reading.
    cuts : dict, optional
        Cuts to apply; they may use the columns added by :func:`add_magnitudes`.
        None applies :data:`DEFAULT_STAR_CUTS`; ``{}`` applies no cut.
    columns : list of str, "all" or None
        Catalog columns to read when ``data`` is None (see :func:`open_euclid_data`).
    data_path : str or Path
        Folder containing the HATS catalog, used when ``data`` is None.
    verbose : bool
        Print the number of selected stars.

    Returns
    -------
    pd.DataFrame
        The selected rows, with the :func:`add_magnitudes` columns.
    """
    cuts = DEFAULT_STAR_CUTS if cuts is None else cuts

    if data is None:
        catalog_columns = set(available_columns(data_path))
        read_cuts = {c: s for c, s in cuts.items() if c in catalog_columns}
        cuts = {c: s for c, s in cuts.items() if c not in catalog_columns}
        data = open_euclid_data(
            columns=columns, data_path=data_path, cuts=read_cuts, verbose=verbose
        )
        n_input = None
    else:
        n_input = len(data)

    if f"{EUCLID_NAMESPACE}_VIS_obs" not in data.columns:
        add_magnitudes(data)

    stars = data[get_cut_mask(data, cuts)].reset_index(drop=True)
    if verbose:
        origin = f" out of {n_input:,}" if n_input is not None else ""
        print(f"Selected {len(stars):,} stars{origin}")
    return stars

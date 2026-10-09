"""Load and prepare the Euclid Q1 x DES Y6 Gold crossmatched catalog.

The catalog is the HATS collection written by
``data/euclid_q1_des/crossmatch_euclid_q1_x_des_y6.py`` (LSDB nearest-neighbour
match within 1 arcsec, columns suffixed ``_euclid`` / ``_des``). It is read here
directly with pyarrow: only the requested columns are loaded, and cuts on catalog
columns are applied while reading.

Harmonised magnitudes follow the streamobs conventions, so the real data can be
compared directly with injected streams:

- columns ``<namespace>_<band>_obs`` / ``<namespace>_<band>_err`` with namespaces
  ``des_yr6`` and ``euclid_q1``, dereddened like the streamobs injected magnitudes;
- the same photometry as the streamobs selection functions: DES ``BDF_MAG_*``,
  Euclid ``FLUX_VIS_PSF`` and ``FLUX_{Y,J,H}_TEMPLFIT``;
- the same default star selection: ``0 <= EXT_XGB <= 1`` for DES,
  ``POINT_LIKE_PROB > 0.5`` and ``SPURIOUS_FLAG == 0`` for Euclid.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from streamobs.surveys import Survey

from streameucliddes.utils import fluxToMag

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "euclid_q1_des"

# Column documentation:
# - Euclid (``*_euclid``): Q1 MER Final Catalog data product description,
#   https://euclid.esac.esa.int/dr/q1/dpdd/merdpd/dpcards/mer_finalcatalog.html
# - DES (``*_des``): Y6 Gold paper, Bechtol et al. 2025 (Tables 3-4 and Appendix B),
#   https://arxiv.org/abs/2501.05739, and https://des.ncsa.illinois.edu/releases/y6a2/Y6gold
# - ``_dist_arcsec``, ``_healpix_29``: added by the LSDB crossmatch,
#   see data/euclid_q1_des/crossmatch_euclid_q1_x_des_y6.py

# streamobs Euclid q1 survey: the extinction coefficients and saturation limits are
# read from it, so they follow any update of the streamobs configuration.
EUCLID_SURVEY = Survey.load("euclid", release="q1", verbose=False)
EUCLID_NAMESPACE = EUCLID_SURVEY.namespace
DES_NAMESPACE = "des_yr6"

# Photometry per band, as used for the streamobs Euclid q1 / DES yr6 surveys.
# BDF_MAG_<B>_CORRECTED_des: DES bulge+disk model magnitude, dereddened by DES (SFD98);
# BDF_MAG_ERR_<B>_des: its error.
DES_MAG_COLUMNS = {
    b: (f"BDF_MAG_{b.upper()}_CORRECTED_des", f"BDF_MAG_ERR_{b.upper()}_des")
    for b in "griz"
}
# Euclid fluxes and errors, in micro-Jansky. There is no FLUX_VIS_TEMPLFIT: the template
# fitting measures the other bands using the VIS image as prior, VIS being measured by
# PSF fitting (FLUX_VIS_TO_<band>_TEMPLFIT are VIS fluxes on PSF-matched images, meant
# for colours, and are not in this crossmatch).
EUCLID_FLUX_COLUMNS = {
    "VIS": ("FLUX_VIS_PSF_euclid", "FLUXERR_VIS_PSF_euclid"),  # VIS PSF-fitting flux
    "Y": (
        "FLUX_Y_TEMPLFIT_euclid",
        "FLUXERR_Y_TEMPLFIT_euclid",
    ),  # NISP Y template-fitting flux
    "J": (
        "FLUX_J_TEMPLFIT_euclid",
        "FLUXERR_J_TEMPLFIT_euclid",
    ),  # NISP J template-fitting flux
    "H": (
        "FLUX_H_TEMPLFIT_euclid",
        "FLUXERR_H_TEMPLFIT_euclid",
    ),  # NISP H template-fitting flux
}
# The streamobs extinction coefficients are A_band / E(B-V)_SFD: they apply to the SFD98
# reddening, not to the Euclid GAL_EBV column (Planck map, ~1.45x larger here).
EBV_COLUMN = "EBV_SFD98_des"  # E(B-V) from the Schlegel et al. 1998 map
RA_COLUMN = "RIGHT_ASCENSION_euclid"  # Euclid source barycentre RA [deg]
DEC_COLUMN = "DECLINATION_euclid"  # Euclid source barycentre Dec [deg]

# DES magnitudes outside this range are sentinels (-99, 37.5, -9.999e9).
DES_VALID_MAG_RANGE = (0.0, 37.5)
# Brighter than this, Euclid POINT_LIKE_PROB is undefined.
EUCLID_VIS_SATURATION = EUCLID_SURVEY.saturation["VIS"]

DEFAULT_COLUMNS = [
    "OBJECT_ID_euclid",  # Euclid unique source identifier
    "COADD_OBJECT_ID_des",  # DES unique object identifier
    RA_COLUMN,
    DEC_COLUMN,
    "_dist_arcsec",  # Euclid - DES separation [arcsec]
    # Euclid detection / classification
    "VIS_DET_euclid",  # 1: detected in VIS, 0: detected in NIR only
    "DET_QUALITY_FLAG_euclid",  # detection-step bitmask (0: no flag)
    "SPURIOUS_FLAG_euclid",  # 1: spurious source
    "POINT_LIKE_PROB_euclid",  # probability of being point-like (NaN for NIR-only sources)
    # DES classification / quality
    "EXT_XGB_des",  # XGBoost class: 0 high-confidence star, 1 likely star, 2-4 galaxies, -9 no data
    "FLAGS_GOLD_des",  # quality bitmask: fitvd/SExtractor flags, saturation, ... (0: clean)
    "FLAGS_FOREGROUND_des",  # foreground bitmask: bright stars, large galaxies, ... (0: none)
    "FLAGS_FOOTPRINT_des",  # 1: inside the Y6 Gold footprint
    EBV_COLUMN,
]
DEFAULT_COLUMNS += [c for pair in EUCLID_FLUX_COLUMNS.values() for c in pair]
DEFAULT_COLUMNS += [c for pair in DES_MAG_COLUMNS.values() for c in pair]

# Cut format: {column: spec} with spec a (min, max) tuple (inclusive, None for an
# open bound), a list/set of accepted values, or a single accepted value.
DES_QUALITY_CUTS = {
    "FLAGS_FOOTPRINT_des": 1,  # inside the Y6 Gold footprint
    "FLAGS_FOREGROUND_des": 0,  # no foreground object
    "FLAGS_GOLD_des": 0,  # no quality flag
}
DES_STAR_CUTS = {"EXT_XGB_des": (0, 1)}  # high-confidence or likely stars
EUCLID_STAR_CUTS = {
    "POINT_LIKE_PROB_euclid": (0.5, None),  # point-like
    "SPURIOUS_FLAG_euclid": 0,  # not spurious
    f"{EUCLID_NAMESPACE}_VIS_obs": (
        EUCLID_VIS_SATURATION,
        None,
    ),  # fainter than the VIS saturation
}
DEFAULT_STAR_CUTS = {**DES_QUALITY_CUTS, **DES_STAR_CUTS, **EUCLID_STAR_CUTS}


# ===================================================
# Reading
# ===================================================


def _dataset_dir(data_path):
    """Return the ``dataset`` directory of the HATS catalog found under ``data_path``."""
    data_path = Path(data_path)
    if (data_path / "dataset").is_dir():
        return data_path / "dataset"
    properties = sorted(data_path.rglob("hats.properties"))
    if not properties:
        raise FileNotFoundError(
            f"No HATS catalog (hats.properties) found under {data_path}"
        )
    return properties[0].parent / "dataset"


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


def available_columns(data_path=DEFAULT_DATA_PATH):
    """
    List the columns of the crossmatched catalog.

    Parameters
    ----------
    data_path : str or Path
        Folder containing the HATS catalog.

    Returns
    -------
    list of str
    """
    return pq.read_schema(_dataset_dir(data_path) / "_common_metadata").names


def _cut_mask(values, spec):
    """Boolean mask of ``values`` passing one cut (NaN never passes a range cut)."""
    values = np.asarray(values)
    if isinstance(spec, tuple):
        low, high = spec
        mask = np.ones(len(values), dtype=bool)
        if low is not None:
            mask &= values >= low
        if high is not None:
            mask &= values <= high
        return mask
    if isinstance(spec, (list, set)):
        return np.isin(values, list(spec))
    return values == spec


def _cut_expression(column, spec):
    """pyarrow filter expression equivalent to :func:`_cut_mask`."""
    field = ds.field(column)
    if isinstance(spec, tuple):
        low, high = spec
        expression = None
        if low is not None:
            expression = field >= low
        if high is not None:
            expression = (
                field <= high if expression is None else expression & (field <= high)
            )
        return expression
    if isinstance(spec, (list, set)):
        return field.isin(list(spec))
    return field == spec


def open_eucliddes_data(
    columns=None, data_path=DEFAULT_DATA_PATH, cuts=None, verbose=True
):
    """
    Read the Euclid Q1 x DES Y6 Gold crossmatched catalog.

    Parameters
    ----------
    columns : list of str, "all" or None
        Catalog columns to read. None reads :data:`DEFAULT_COLUMNS` (positions,
        classification, quality flags and the photometry used by
        :func:`add_magnitudes`); "all" reads every column (~130).
    data_path : str or Path
        Folder containing the HATS catalog.
    cuts : dict, optional
        Cuts on catalog columns, applied while reading (see :func:`get_cut_mask`
        for the format). The cut columns do not need to be in ``columns``.
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

    data = dataset.to_table(columns=list(columns), filter=expression).to_pandas()
    if verbose:
        print(
            f"Read {len(data):,} rows x {len(data.columns)} columns from {dataset_dir.parent.name}"
        )
    return data


# ===================================================
# Magnitudes
# ===================================================


def add_magnitudes(
    data, des_bands=tuple(DES_MAG_COLUMNS), euclid_bands=tuple(EUCLID_FLUX_COLUMNS)
):
    """
    Add harmonised, dereddened magnitudes and ``ra``/``dec`` columns, in place.

    Adds ``des_yr6_<band>_obs`` / ``_err`` from the DES BDF magnitudes (sentinels set
    to NaN) and ``euclid_q1_<band>_obs`` / ``_err`` from the Euclid fluxes (non-positive
    fluxes set to NaN), dereddened with ``EBV_SFD98_des`` and the extinction coefficients
    of the streamobs Euclid survey (``EUCLID_SURVEY.coeff_extinc``).
    ``ra``/``dec`` are the Euclid positions.

    Parameters
    ----------
    data : pd.DataFrame
        Catalog read with :func:`open_eucliddes_data`.
    des_bands, euclid_bands : sequence of str
        Bands to convert.

    Returns
    -------
    pd.DataFrame
        ``data``, with the new columns.
    """
    needed = [RA_COLUMN, DEC_COLUMN]
    needed += [c for b in des_bands for c in DES_MAG_COLUMNS[b]]
    if euclid_bands:
        needed += [c for b in euclid_bands for c in EUCLID_FLUX_COLUMNS[b]] + [
            EBV_COLUMN
        ]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise KeyError(
            f"Column(s) needed to compute the magnitudes are missing: {missing}"
        )

    data["ra"] = data[RA_COLUMN]
    data["dec"] = data[DEC_COLUMN]

    for band in des_bands:
        mag_col, err_col = DES_MAG_COLUMNS[band]
        mag = data[mag_col].to_numpy(dtype=float)
        valid = (mag > DES_VALID_MAG_RANGE[0]) & (mag < DES_VALID_MAG_RANGE[1])
        data[f"{DES_NAMESPACE}_{band}_obs"] = np.where(valid, mag, np.nan)
        data[f"{DES_NAMESPACE}_{band}_err"] = np.where(valid, data[err_col], np.nan)

    for band in euclid_bands:
        flux_col, err_col = EUCLID_FLUX_COLUMNS[band]
        flux = data[flux_col].to_numpy(dtype=float)
        flux_err = data[err_col].to_numpy(dtype=float)
        valid = flux > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = fluxToMag(np.where(valid, flux, np.nan) * 1e-6)  # micro-Jy -> Jy
            mag_err = 2.5 / np.log(10) * flux_err / flux
        extinction = EUCLID_SURVEY.coeff_extinc[band] * data[EBV_COLUMN].to_numpy(
            dtype=float
        )
        data[f"{EUCLID_NAMESPACE}_{band}_obs"] = mag - extinction
        data[f"{EUCLID_NAMESPACE}_{band}_err"] = np.where(valid, mag_err, np.nan)

    return data


# ===================================================
# Cuts
# ===================================================


def get_cut_mask(data, cuts):
    """
    Return a boolean mask selecting the rows passing all ``cuts``.

    Parameters
    ----------
    data : pd.DataFrame
    cuts : dict
        ``{column: spec}``. ``spec`` is a ``(min, max)`` tuple (inclusive, None for an
        open bound), a list/set of accepted values, or a single accepted value.
        Example: ``{"EXT_XGB_des": (0, 1), "SPURIOUS_FLAG_euclid": 0,
        "euclid_q1_VIS_obs": (EUCLID_VIS_SATURATION, None)}``.

    Returns
    -------
    np.ndarray of bool
    """
    mask = np.ones(len(data), dtype=bool)
    for column, spec in cuts.items():
        mask &= _cut_mask(data[column], spec)
    return mask


def cut_flow(data, cuts):
    """
    Number of rows left after applying each cut in turn.

    Parameters
    ----------
    data : pd.DataFrame
    cuts : dict
        Same format as :func:`get_cut_mask`.

    Returns
    -------
    pd.DataFrame
        One row per cut: the cut, the rows passing that cut alone, and the rows (and
        fraction of the input) left after it and all previous cuts.
    """
    mask = np.ones(len(data), dtype=bool)
    rows = [{"cut": "input", "n_alone": len(data), "n_remaining": len(data)}]
    for column, spec in cuts.items():
        cut = _cut_mask(data[column], spec)
        mask &= cut
        rows.append(
            {
                "cut": f"{column}: {spec}",
                "n_alone": int(cut.sum()),
                "n_remaining": int(mask.sum()),
            }
        )
    flow = pd.DataFrame(rows)
    flow["fraction_remaining"] = flow["n_remaining"] / max(len(data), 1)
    return flow


def get_eucliddes_stars(
    data=None, cuts=None, columns=None, data_path=DEFAULT_DATA_PATH, verbose=True
):
    """
    Get the stars of the Euclid Q1 x DES Y6 Gold catalog, with harmonised magnitudes.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Catalog already read with :func:`open_eucliddes_data`. If None, the catalog is
        read from ``data_path``, applying the cuts on catalog columns while reading.
    cuts : dict, optional
        Cuts to apply (format of :func:`get_cut_mask`); they may use the columns added
        by :func:`add_magnitudes`. None applies :data:`DEFAULT_STAR_CUTS`; ``{}``
        applies no cut.
    columns : list of str, "all" or None
        Catalog columns to read when ``data`` is None (see :func:`open_eucliddes_data`).
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
        data = open_eucliddes_data(
            columns=columns, data_path=data_path, cuts=read_cuts, verbose=verbose
        )
        n_input = None
    else:
        n_input = len(data)

    if f"{DES_NAMESPACE}_g_obs" not in data.columns:
        add_magnitudes(data)

    stars = data[get_cut_mask(data, cuts)].reset_index(drop=True)
    if verbose:
        origin = f" out of {n_input:,}" if n_input is not None else ""
        print(f"Selected {len(stars):,} stars{origin}")
    return stars

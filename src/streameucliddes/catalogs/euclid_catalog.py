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
- Euclid magnitudes dereddened with the E(B-V) and coefficients of a dust correction
  (:data:`DUST_CORRECTIONS`): ``"SFD98"`` (Schlegel et al. 1998, default) or
  ``"PLANCK"`` (the Euclid ``GAL_EBV`` column, Planck 2013 map, as Gatto et al.);
  DES magnitudes come dereddened by DES (SFD98); with another dust correction, its
  E(B-V) replaces SFD98 with the DES coefficients of streamobs;
- the same default star selection: ``POINT_LIKE_PROB >= 0.5`` and
  ``SPURIOUS_FLAG == 0`` for Euclid, ``0 <= EXT_XGB <= 1`` for DES.

Three Euclid star selections (:data:`STAR_SELECTIONS`, see :func:`default_star_cuts`):
``"morphology_cuts"`` (default), cuts on the Euclid morphology columns;
``"xgboost_classifier"``, a cut on the star probability of the XGBoost classifier
trained on Euclid Q1 (:mod:`~streameucliddes.catalogs.star_classifier`), added as
the column ``xgboost_star_prob`` (:func:`add_xgboost_star_prob`); and
``"gatto_stars_dr1"``, the selection of Gatto et al. (Euclid DR1 dwarf galaxies,
Sect. 2): a loose ``POINT_LIKE_PROB`` cut and a polygon in the dereddened
colour-colour plane, flagged in the column ``in_gatto_color_box``
(:func:`add_gatto_color_box`).

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

import healpy as hp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from matplotlib.path import Path as PolygonPath
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

DES_RELEASE = "yr6"  # streamobs DES release (extinction coefficients, see des_survey)
DES_NAMESPACE = "des_yr6"
DES_BANDS = ("g", "r", "i", "z")
# DES magnitudes outside this range are sentinels (-99, 37.5, -9.999e9).
DES_VALID_MAG_RANGE = (0.0, 37.5)

# Rows per batch when open_euclid_catalog reads by batches (batch_selection).
READ_BATCH_SIZE = 1_000_000

# Euclid fluxes and errors per band, in micro-Jansky. There is no FLUX_VIS_TEMPLFIT:
# TPHOT measures VIS on the VIS image PSF-matched to each other band, so that a
# colour VIS - Y uses FLUX_VIS_TO_Y_TEMPLFIT and FLUX_Y_TEMPLFIT (DR1 only, added by
# scripts/download_euclid_dr1.py --add-columns). "VIS" is the PSF-fitting flux.
VIS_TEMPLFIT_PARTNER_BANDS = ("Y", "J", "H")  # bands of the FLUX_VIS_TO_<band>_TEMPLFIT columns
EUCLID_FLUX_COLUMNS = {
    "VIS": ("FLUX_VIS_PSF", "FLUXERR_VIS_PSF"),  # VIS PSF-fitting flux
    "Y": ("FLUX_Y_TEMPLFIT", "FLUXERR_Y_TEMPLFIT"),  # NISP Y template-fitting flux
    "J": ("FLUX_J_TEMPLFIT", "FLUXERR_J_TEMPLFIT"),  # NISP J template-fitting flux
    "H": ("FLUX_H_TEMPLFIT", "FLUXERR_H_TEMPLFIT"),  # NISP H template-fitting flux
    # VIS template-fitting fluxes, PSF-matched to Y, J, H (DR1 only)
    **{f"VIS_TO_{band}": (f"FLUX_VIS_TO_{band}_TEMPLFIT", f"FLUXERR_VIS_TO_{band}_TEMPLFIT")
       for band in VIS_TEMPLFIT_PARTNER_BANDS},
}
# VIS photometry of a match filter (see vis_band): "psf", the "VIS" magnitudes, or
# "templfit", the "VIS_TO_<band>" magnitudes, <band> being the other band of the colour.
# Default templfit: DR1 only, and only in a colour with Y, J or H; the Q1 catalogs and the
# colours with other bands need "psf" explicitly.
VIS_PHOTOMETRIES = ("psf", "templfit")
DEFAULT_VIS_PHOTOMETRY = "templfit"
# Attribute of the stars read in a box (get_euclid_stars, DataFrame.attrs): (ra_range, dec_range) asked for, so that
# a survey edge inside the box (no star beyond it) is told apart from the edge of the box
# (CandidateInvestigation: a window beyond the stars but inside the box is fine)
LOADED_BOX_ATTR = "loaded_radec_box"
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
STAR_SELECTIONS = ("morphology_cuts", "xgboost_classifier", "gatto_stars_dr1")
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

# "gatto_stars_dr1": selection of Gatto et al., "Pushing back the borders of the
# invisible with Euclid" (Euclid DR1 dwarf galaxies, Sect. 2 and Fig. 1),
# https://pubs.euclid-ec.org/public/dr1/gatto.pdf
GATTO_STAR_CUTS = {
    "POINT_LIKE_PROB": (0.05, None),  # point-like (paper: > 0.05)
}
# Polygon in the dereddened colour-colour plane x = VIS - H, y = Y - H: keeps the
# stellar locus, removes the residual compact galaxies. Contains the main sequence,
# red giant branch and horizontal branch of old isochrones up to [Fe/H] ~ -1.
GATTO_COLOR_BOX = {
    "x_bands": ("VIS", "H"),  # x colour: VIS - H
    "y_bands": ("Y", "H"),  # y colour: Y - H
    "vertices": ((-1.5, -1.5), (-1.5, -0.4), (0.9, 0.8), (1.4, 0.8), (1.4, -0.4), (0.9, -0.4)),
}
# Column added by add_gatto_color_box: True inside GATTO_COLOR_BOX.
GATTO_COLOR_BOX_COLUMN = "in_gatto_color_box"


# Dust correction of the Euclid magnitudes (add_magnitudes), chosen by name: the
# E(B-V) column (Euclid-only catalog name; crossmatched catalogs: see ebv_column)
# and the extinction coefficients A_band / E(B-V) of the Euclid bands.
# - "SFD98": E(B-V) of Schlegel, Finkbeiner & Davis (1998), column added by
#   scripts/download_euclid_dr1.py (crossmatched catalogs: the DES column).
# - "PLANCK": the Euclid GAL_EBV column, the E(B-V) map of Planck Collaboration:
#   Abergel et al. 2014 (A&A 571, A11; HFI_CompMap_ThermalDustModel_2048_R1.20), as
#   in Gatto et al. (Euclid DR1 dwarf galaxies, Sect. 2). GAL_EBV equals dustmaps'
#   PlanckQuery of that map on 20,003 DR1 sources (checked on 2026-11-18).
COEFF_SFD98 = {
    "VIS": 1.790,  # A_VIS / E(B-V)_SFD98
    "Y": 0.879,  # A_Y / E(B-V)_SFD98
    "J": 0.595,  # A_J / E(B-V)_SFD98
    "H": 0.400,  # A_H / E(B-V)_SFD98
}

# Gatto et al. (Sect. 2): from the PARSEC A_band / A_V (Cardelli et al. 1989,
# O'Donnell 1994), R_V = 3.1.
COEFF_PLANCK = {
    "VIS": 2.28,  # A_VIS / E(B-V)_PLANCK
    "Y": 1.14,  # A_Y / E(B-V)_PLANCK
    "J": 0.79,  # A_J / E(B-V)_PLANCK
    "H": 0.51,  # A_H / E(B-V)_PLANCK
}
DUST_CORRECTIONS = {
    "SFD98": {"ebv_column": "EBV_SFD98", "coefficients": COEFF_SFD98},
    "PLANCK": {"ebv_column": "GAL_EBV", "coefficients": COEFF_PLANCK},
}
DEFAULT_DUST_CORRECTION = "PLANCK"


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

    Its namespace names the magnitude columns, and its VIS saturation is used here,
    so it follows any update of streamobs. The extinction coefficients are those of
    :data:`DUST_CORRECTIONS`.

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


@functools.cache
def des_survey():
    """
    streamobs DES survey (release :data:`DES_RELEASE`).

    Its extinction coefficients A_band / E(B-V)_SFD98 are those DES used to deredden
    the ``BDF_MAG_*_CORRECTED`` magnitudes; :func:`add_magnitudes` uses them to
    change the dust correction of the DES bands.

    Returns
    -------
    streamobs.surveys.Survey
    """
    return Survey.load("des", release=DES_RELEASE, verbose=False)


def euclid_column(catalog, name):
    """Name of the Euclid column ``name`` in ``catalog`` (``_euclid`` suffix if crossmatched)."""
    return f"{name}_euclid" if _catalog_info(catalog)["with_des"] else name


def euclid_bands(catalog):
    """
    Euclid bands of ``catalog`` (keys of :data:`EUCLID_FLUX_COLUMNS`).

    The VIS template-fitting bands ``VIS_TO_<band>`` are only in DR1.

    Parameters
    ----------
    catalog : str
        Key of :data:`CATALOGS`.

    Returns
    -------
    tuple of str
    """
    dr1 = _catalog_info(catalog)["euclid_release"] == "dr1"
    return tuple(band for band in EUCLID_FLUX_COLUMNS if dr1 or not band.startswith("VIS_TO_"))


def vis_band(other_band, vis_photometry=DEFAULT_VIS_PHOTOMETRY):
    """
    Euclid band of the VIS magnitude in a colour with ``other_band``.

    Parameters
    ----------
    other_band : str
        The other band of the colour, e.g. ``"H"`` for VIS - H.
    vis_photometry : str
        ``"psf"``: the PSF-fitting magnitude, ``"VIS"``; ``"templfit"``: the
        template-fitting magnitude on the VIS image PSF-matched to ``other_band``,
        ``"VIS_TO_<other_band>"`` (``other_band`` in
        :data:`VIS_TEMPLFIT_PARTNER_BANDS`, DR1 only).

    Returns
    -------
    str
        Key of :data:`EUCLID_FLUX_COLUMNS`; the magnitude column of
        :func:`add_magnitudes` is ``<namespace>_<band>_obs``.
    """
    if vis_photometry not in VIS_PHOTOMETRIES:
        raise ValueError(f"Unknown VIS photometry {vis_photometry!r}: choose one of {VIS_PHOTOMETRIES}")
    if vis_photometry == "psf":
        return "VIS"
    if other_band not in VIS_TEMPLFIT_PARTNER_BANDS:
        raise ValueError(f"No VIS template-fitting magnitude for a colour with {other_band!r}: "
                         f"only with {VIS_TEMPLFIT_PARTNER_BANDS}")
    return f"VIS_TO_{other_band}"


def _extinction_band(band):
    """Band of the extinction coefficient of a Euclid band: ``"VIS_TO_H"`` -> ``"VIS"``."""
    return band.split("_TO_")[0]


def _dust_correction(dust_correction):
    """Entry of :data:`DUST_CORRECTIONS`, with a clear error for an unknown name."""
    if dust_correction not in DUST_CORRECTIONS:
        raise ValueError(f"Unknown dust correction {dust_correction!r}: "
                         f"choose one of {list(DUST_CORRECTIONS)}")
    return DUST_CORRECTIONS[dust_correction]


def ebv_column(catalog, dust_correction=DEFAULT_DUST_CORRECTION):
    """
    Name of the E(B-V) column of a dust correction in ``catalog``.

    Parameters
    ----------
    catalog : str
        Key of :data:`CATALOGS`.
    dust_correction : str
        Key of :data:`DUST_CORRECTIONS`.

    Returns
    -------
    str
        Crossmatched catalogs: the DES column for SFD98 (``_des`` suffix), the Euclid
        column otherwise (``_euclid`` suffix).
    """
    name = _dust_correction(dust_correction)["ebv_column"]
    if dust_correction == "SFD98" and _catalog_info(catalog)["with_des"]:
        return f"{name}_des"
    return euclid_column(catalog, name)


def catalog_columns(catalog, dust_correction=DEFAULT_DUST_CORRECTION):
    """
    Names of the columns with a role in ``catalog``.

    Parameters
    ----------
    catalog : str
        Key of :data:`CATALOGS`.
    dust_correction : str
        Key of :data:`DUST_CORRECTIONS`: gives the ``ebv`` column.

    Returns
    -------
    dict
        ``ra``, ``dec``: Euclid source barycentre [deg]; ``ebv``: E(B-V) of the
        dust correction (:func:`ebv_column`); ``euclid_fluxes``: ``{band: (flux,
        error)}`` of the bands of :func:`euclid_bands`; ``des_magnitudes``: ``{band:
        (magnitude, error)}``, empty without DES.
    """
    with_des = _catalog_info(catalog)["with_des"]
    return {
        "ra": euclid_column(catalog, "RIGHT_ASCENSION"),
        "dec": euclid_column(catalog, "DECLINATION"),
        "ebv": ebv_column(catalog, dust_correction),
        "euclid_fluxes": {
            band: tuple(euclid_column(catalog, column) for column in EUCLID_FLUX_COLUMNS[band])
            for band in euclid_bands(catalog)
        },
        # BDF_MAG_<B>_CORRECTED: DES bulge+disk model magnitude, dereddened by DES (SFD98)
        "des_magnitudes": {
            band: (f"BDF_MAG_{band.upper()}_CORRECTED_des", f"BDF_MAG_ERR_{band.upper()}_des")
            for band in DES_BANDS
        } if with_des else {},
    }


def default_columns(catalog, vis_photometry=DEFAULT_VIS_PHOTOMETRY):
    """
    Columns read by default: identifiers, positions, classification, quality flags
    and the photometry used by :func:`add_magnitudes`, with the E(B-V) of every
    dust correction (:data:`DUST_CORRECTIONS`).

    Parameters
    ----------
    catalog : str
        Key of :data:`CATALOGS`.
    vis_photometry : str
        ``"psf"``: the VIS PSF flux only; ``"templfit"`` (:data:`DEFAULT_VIS_PHOTOMETRY`):
        also the VIS template-fitting fluxes ``FLUX_VIS_TO_<band>_TEMPLFIT`` (DR1 only:
        a Q1 catalog needs ``"psf"``; 6 more columns for every star).

    Returns
    -------
    list of str
    """
    if vis_photometry not in VIS_PHOTOMETRIES:
        raise ValueError(f"Unknown VIS photometry {vis_photometry!r}: choose one of {VIS_PHOTOMETRIES}")
    roles = catalog_columns(catalog)
    if vis_photometry == "templfit" and not any(b.startswith("VIS_TO_") for b in roles["euclid_fluxes"]):
        raise ValueError(f"{catalog} has no VIS template-fitting fluxes (DR1 only)")
    fluxes = {band: pair for band, pair in roles["euclid_fluxes"].items()
              if vis_photometry == "templfit" or not band.startswith("VIS_TO_")}
    columns = [euclid_column(catalog, "OBJECT_ID"), roles["ra"], roles["dec"]]
    columns += [euclid_column(catalog, c) for c in EUCLID_DEFAULT_COLUMNS[1:]]
    if roles["des_magnitudes"]:
        columns += DES_DEFAULT_COLUMNS
    columns += [ebv_column(catalog, name) for name in DUST_CORRECTIONS]
    columns += [c for pair in fluxes.values() for c in pair]
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
      (column ``xgboost_star_prob``) at least :data:`XGBOOST_MIN_STAR_PROB`;
    - ``"gatto_stars_dr1"``: the cuts of :data:`GATTO_STAR_CUTS` on
      ``POINT_LIKE_PROB`` and inside the colour-colour polygon
      :data:`GATTO_COLOR_BOX` (column ``in_gatto_color_box``).

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
        (it replaces the threshold of the selection). None: the threshold of the
        selection, no cut with ``"xgboost_classifier"``.
    min_xgboost_star_prob : float, optional
        Keep ``xgboost_star_prob >= min_xgboost_star_prob``, whatever
        ``star_selection``. None: :data:`XGBOOST_MIN_STAR_PROB` with
        ``"xgboost_classifier"``, no cut with the other selections.

    Returns
    -------
    dict
        ``get_euclid_stars`` adds the ``xgboost_star_prob`` and
        ``in_gatto_color_box`` columns when a cut uses them.
    """
    survey = euclid_survey(catalog)
    cuts = {}
    if _catalog_info(catalog)["with_des"]:
        cuts.update(DES_QUALITY_CUTS)
        cuts.update(DES_STAR_CUTS)

    # Euclid column names without suffix: "_euclid" added for crossmatched catalogs
    cuts.update({euclid_column(catalog, c): spec for c, spec in EUCLID_QUALITY_CUTS.items()})
    cuts[f"{survey.namespace}_VIS_obs"] = (survey.saturation["VIS"], None)  # not saturated

    # Star / galaxy separation
    if star_selection == "morphology_cuts":
        cuts.update({euclid_column(catalog, c): spec for c, spec in MORPHOLOGY_STAR_CUTS.items()})
    elif star_selection == "xgboost_classifier":
        cuts[XGBOOST_STAR_PROB_COLUMN] = (XGBOOST_MIN_STAR_PROB, None)
    elif star_selection == "gatto_stars_dr1":
        cuts.update({euclid_column(catalog, c): spec for c, spec in GATTO_STAR_CUTS.items()})
        cuts[GATTO_COLOR_BOX_COLUMN] = True
    else:
        raise ValueError(f"Unknown star_selection {star_selection!r}: choose one of {STAR_SELECTIONS}")

    # Thresholds given explicitly, whatever the selection
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


def catalog_point_map(catalog, data_path=None):
    """
    Number of sources of a catalog per HEALPix pixel: the ``point_map.fits`` of its HATS folder.

    Written by hats-import, for all the sources of the catalog (not only the
    stars), in NESTED order. Read once per folder (cached). A cheap way to
    tell whether a position is in the catalog, without reading it.

    Parameters
    ----------
    catalog : str
        Key of :data:`CATALOGS`.
    data_path : str or Path, optional
        Folder containing the HATS catalog; default: the one of :data:`CATALOGS`.

    Returns
    -------
    counts : numpy.ndarray
        Sources per pixel, NESTED (read-only: shared between calls).
    """
    data_path = _catalog_info(catalog)["data_path"] if data_path is None else data_path
    return _read_point_map(str(_catalog_dir(data_path) / "point_map.fits"))


@functools.lru_cache(maxsize=2)
def _read_point_map(path):
    """HATS point map of a catalog, NESTED, read-only."""
    counts = hp.read_map(path, nest=True)
    counts = counts.astype(counts.dtype.newbyteorder("="), copy=False)  # FITS: big-endian, pandas needs native
    counts.setflags(write=False)
    return counts


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


def magnitude_columns(catalog, vis_photometry=DEFAULT_VIS_PHOTOMETRY):
    """
    Columns added by :func:`add_magnitudes` with its default bands, to the columns
    of ``default_columns(catalog, vis_photometry)``.

    Parameters
    ----------
    catalog : str
        Key of :data:`CATALOGS`.
    vis_photometry : str
        ``"templfit"``: with the VIS template-fitting magnitudes.

    Returns
    -------
    list of str
        ``ra``, ``dec`` and the ``<namespace>_<band>_obs`` / ``_err`` columns.
    """
    namespace = euclid_survey(catalog).namespace
    bands = [band for band in euclid_bands(catalog)
             if vis_photometry == "templfit" or not band.startswith("VIS_TO_")]
    columns = ["ra", "dec"]
    for prefix, bands in [(DES_NAMESPACE, catalog_columns(catalog)["des_magnitudes"]),
                          (namespace, bands)]:
        columns += [f"{prefix}_{band}_{kind}" for band in bands for kind in ("obs", "err")]
    return columns


def add_magnitudes(
    data, catalog, bands=None, des_bands=None, dust_correction=DEFAULT_DUST_CORRECTION,
):
    """
    Add harmonised, dereddened magnitudes and ``ra``/``dec`` columns, in place.

    Adds ``<euclid namespace>_<band>_obs`` / ``_err`` from the Euclid fluxes
    (non-positive fluxes set to NaN), dereddened with the E(B-V) column and the
    extinction coefficients of ``dust_correction`` (the VIS coefficient for the
    VIS template-fitting bands ``VIS_TO_<band>``), and, for crossmatched catalogs,
    ``des_yr6_<band>_obs`` / ``_err`` from the DES BDF magnitudes (sentinels set to
    NaN). DES dereddened them with SFD98 and the DES coefficients
    (:func:`des_survey`); the E(B-V) of ``dust_correction`` replaces SFD98 with the
    same DES coefficients: ``mag + R_DES * (E(B-V)_SFD98 - E(B-V))``, unchanged for
    ``"SFD98"``. ``ra``/``dec`` are the Euclid positions.

    Parameters
    ----------
    data : pd.DataFrame
        Catalog read with :func:`open_euclid_catalog`.
    catalog : str
        Key of :data:`CATALOGS`.
    bands : sequence of str, optional
        Euclid bands to convert; None: those of :func:`euclid_bands`, the VIS
        template-fitting ones only when their fluxes were read
        (``default_columns(catalog, "templfit")``).
    des_bands : sequence of str, optional
        DES bands to convert; None: all of them with DES, none without.
    dust_correction : str
        Key of :data:`DUST_CORRECTIONS`: ``"SFD98"`` or ``"PLANCK"``.

    Returns
    -------
    pd.DataFrame
        ``data``, with the new columns.
    """
    coefficients = _dust_correction(dust_correction)["coefficients"]
    roles = catalog_columns(catalog, dust_correction)
    if bands is None:
        euclid_bands = tuple(band for band, (flux, _) in roles["euclid_fluxes"].items()
                             if not band.startswith("VIS_TO_") or flux in data.columns)
    else:
        euclid_bands = bands
    if des_bands is None:
        des_bands = tuple(roles["des_magnitudes"])
    elif des_bands and not roles["des_magnitudes"]:
        raise ValueError(f"{catalog} has no DES magnitudes")
    needed = [roles["ra"], roles["dec"]]
    needed += [c for b in des_bands for c in roles["des_magnitudes"][b]]
    if des_bands:
        needed += [ebv_column(catalog, "SFD98"), roles["ebv"]]
    if euclid_bands:
        needed += [c for b in euclid_bands for c in roles["euclid_fluxes"][b]] + [roles["ebv"]]
    missing = list(dict.fromkeys(c for c in needed if c not in data.columns))
    if missing:
        raise KeyError(f"Column(s) needed to compute the magnitudes are missing: {missing}")

    data["ra"] = data[roles["ra"]]
    data["dec"] = data[roles["dec"]]

    if des_bands:
        des_coefficients = des_survey().coeff_extinc
        # E(B-V) of DES's correction minus the chosen one: 0 for SFD98
        ebv_change = (data[ebv_column(catalog, "SFD98")].to_numpy(dtype=float)
                      - data[roles["ebv"]].to_numpy(dtype=float))
    for band in des_bands:
        mag_col, err_col = roles["des_magnitudes"][band]
        mag = data[mag_col].to_numpy(dtype=float)
        valid = (mag > DES_VALID_MAG_RANGE[0]) & (mag < DES_VALID_MAG_RANGE[1])
        mag = mag + des_coefficients[band] * ebv_change
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
        data[f"{survey.namespace}_{band}_obs"] = mag - coefficients[_extinction_band(band)] * ebv
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


# ===================================================
# Colour-colour box
# ===================================================


def in_color_box(data, catalog, color_box=GATTO_COLOR_BOX):
    """
    Whether each source lies inside a polygon of a Euclid colour-colour plane.

    Parameters
    ----------
    data : pd.DataFrame
        With the dereddened Euclid magnitudes of :func:`add_magnitudes`.
    catalog : str
        Key of :data:`CATALOGS`.
    color_box : dict
        ``x_bands`` and ``y_bands``: the (blue, red) Euclid bands of each colour
        (``x = blue - red``); ``vertices``: the (x, y) vertices of the polygon.
        Default: :data:`GATTO_COLOR_BOX`.

    Returns
    -------
    np.ndarray of bool
        False for sources without one of the magnitudes.
    """
    namespace = euclid_survey(catalog).namespace

    def color(bands):
        blue, red = (data[f"{namespace}_{band}_obs"].to_numpy(dtype=float) for band in bands)
        return blue - red

    points = np.column_stack([color(color_box["x_bands"]), color(color_box["y_bands"])])
    inside = PolygonPath(color_box["vertices"]).contains_points(points)
    return inside & np.isfinite(points).all(axis=1)


def add_gatto_color_box(data, catalog):
    """
    Add the column ``in_gatto_color_box`` in place: True inside the colour-colour
    polygon :data:`GATTO_COLOR_BOX` (:func:`in_color_box`).

    Parameters
    ----------
    data : pd.DataFrame
        With the dereddened Euclid magnitudes of :func:`add_magnitudes`.
    catalog : str
        Key of :data:`CATALOGS`.

    Returns
    -------
    pd.DataFrame
        ``data``, with the new column.
    """
    data[GATTO_COLOR_BOX_COLUMN] = in_color_box(data, catalog, GATTO_COLOR_BOX)
    return data


# Columns computed batch by batch while reading (see _batch_selection).
BATCH_COLUMNS = (XGBOOST_STAR_PROB_COLUMN, GATTO_COLOR_BOX_COLUMN)


def _batch_selection(catalog, cuts, dust_correction=DEFAULT_DUST_CORRECTION):
    """
    ``batch_selection`` of :func:`open_euclid_catalog` for the cuts on the columns
    of :data:`BATCH_COLUMNS`: keeps the rows of each batch passing those cuts, with
    the columns.

    The classifier runs on the pyarrow table, so that only the rows it keeps are
    converted to pandas; the colour box needs the magnitudes, added to the batch
    with ``dust_correction``.
    """
    def select(table):
        if XGBOOST_STAR_PROB_COLUMN in cuts:
            probability = _xgboost_star_prob(table, catalog)
            column = XGBOOST_STAR_PROB_COLUMN
            keep = get_cut_mask(pd.DataFrame({column: probability}), {column: cuts[column]})
            table = table.filter(pa.array(keep))
        data = _to_pandas(table)
        if XGBOOST_STAR_PROB_COLUMN in cuts:
            data[XGBOOST_STAR_PROB_COLUMN] = probability[keep]
        if GATTO_COLOR_BOX_COLUMN in cuts:
            add_magnitudes(data, catalog, dust_correction=dust_correction)
            add_gatto_color_box(data, catalog)
            column = GATTO_COLOR_BOX_COLUMN
            data = data[get_cut_mask(data, {column: cuts[column]})].reset_index(drop=True)
        return data

    return select


# ===================================================
# Stars
# ===================================================


def get_euclid_stars(
    catalog, data=None, cuts=None, columns=None, ra_range=None, dec_range=None,
    data_path=None, verbose=True, subset=None, dust_correction=DEFAULT_DUST_CORRECTION,
    vis_photometry=DEFAULT_VIS_PHOTOMETRY,
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
        the columns added by :func:`add_magnitudes`, ``xgboost_star_prob``
        (:func:`add_xgboost_star_prob`) and ``in_gatto_color_box``
        (:func:`add_gatto_color_box`), the last two added only when a cut uses
        them. None applies
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
    dust_correction : str
        Key of :data:`DUST_CORRECTIONS` used by :func:`add_magnitudes`
        (``"SFD98"`` or ``"PLANCK"``); not applied to a ``data`` that has the
        magnitudes already.
    vis_photometry : str
        ``"templfit"``: also read the VIS template-fitting fluxes and add their
        magnitudes ``<namespace>_VIS_TO_<band>_obs`` (DR1), for a match filter
        with ``"vis_photometry": "templfit"``; ``"psf"``: only the PSF VIS (a Q1
        catalog: no VIS template fit). Default: :data:`DEFAULT_VIS_PHOTOMETRY`.
        When ``data`` is None and ``columns`` is None (see :func:`default_columns`).

    Returns
    -------
    pd.DataFrame
        The selected rows, with the :func:`add_magnitudes` columns. When read
        here (``data`` None), ``attrs[LOADED_BOX_ATTR]`` is ``(ra_range,
        dec_range)`` as read (None: no limit), e.g. to tell a survey edge from
        the edge of the box.
    """
    _dust_correction(dust_correction)  # unknown name: fail before reading
    if columns is None and data is None:
        columns = default_columns(catalog, vis_photometry)
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
        derived = magnitude_columns(catalog, vis_photometry) + list(BATCH_COLUMNS)
        unknown = [c for c in cuts if c not in names and c not in derived]
        if unknown:
            raise KeyError(
                f"Cut column(s) neither in the catalog nor added by add_magnitudes, "
                f"add_xgboost_star_prob or add_gatto_color_box: {unknown}"
            )
        # Cuts on catalog columns while reading, the others (magnitudes) after.
        read_cuts = {c: s for c, s in cuts.items() if c in names}
        cuts = {c: s for c, s in cuts.items() if c not in names}
        # Star classifier and colour box: their cuts are applied while reading,
        # batch by batch, since the rows passing only the cuts on catalog columns
        # may not fit in memory.
        batch_selection = None
        if any(c in cuts for c in BATCH_COLUMNS):
            batch_selection = _batch_selection(catalog, cuts, dust_correction)
        data = open_euclid_catalog(
            catalog, columns=columns, cuts=read_cuts, ra_range=ra_range,
            dec_range=dec_range, data_path=data_path, verbose=verbose,
            batch_selection=batch_selection,
        )
        n_input = None
    else:
        n_input = len(data)

    if f"{euclid_survey(catalog).namespace}_VIS_obs" not in data.columns:
        add_magnitudes(data, catalog, dust_correction=dust_correction)
    if XGBOOST_STAR_PROB_COLUMN in cuts and XGBOOST_STAR_PROB_COLUMN not in data.columns:
        add_xgboost_star_prob(data, catalog)
    if GATTO_COLOR_BOX_COLUMN in cuts and GATTO_COLOR_BOX_COLUMN not in data.columns:
        add_gatto_color_box(data, catalog)

    stars = data[get_cut_mask(data, cuts)].reset_index(drop=True)
    if n_input is None:  # read here, in this box
        stars.attrs[LOADED_BOX_ATTR] = (None if ra_range is None else tuple(map(float, ra_range)),
                                        None if dec_range is None else tuple(map(float, dec_range)))
    if verbose:
        origin = f" out of {n_input:,}" if n_input is not None else ""
        print(f"Selected {len(stars):,} stars{origin}")
    return stars

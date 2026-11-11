"""Isochrone parameters of the match filters: age and metallicity.

An isochrone is given as ``{"age": age_gyr, "metallicity": z}``, the
arguments of ``analysis.cmd_match_filter.build_match_filter_polygon``.

The ages and metallicities of the known streams are curated by hand from the
literature in ``stream_populations.csv``, next to this module, each with its
reference. A missing value is left empty (NaN): it is never invented. ``z``,
the metallicity given to ugali, is taken from the paper when it gives it, or
converted from [Fe/H] with ``unit_conversions.convert_FeH_to_z``
(``z_reference`` says which). A stream whose age or metallicity is unknown
uses the default grid of old, metal-poor isochrones.

Spectroscopic [Fe/H] (Li et al. 2022, S5) comes first, then the Pristine
photometric [Fe/H] (Martin et al. 2022). The ages and Z of Shipp et al. 2018
(DES) are a joint photometric fit, degenerate between age and Z, so they are
kept together: they fill the age and ``z`` only of the streams without
[Fe/H] (Ravi, Wambelong), and only where both were free in the fit (their
Table A.1). For the other streams they fitted, they are in ``notes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

POPULATION_TABLE_PATH = Path(__file__).with_name("stream_populations.csv")
INDEX_COLUMN = "galstreams_name"
VALUE_COLUMNS = [  # floats, NaN when unknown
    "age_gyr", "age_err_plus", "age_err_minus", "feh", "feh_err_plus", "feh_err_minus", "z",
]
TEXT_COLUMNS = ["age_reference", "feh_reference", "z_reference", "notes"]  # strings, empty when missing


def default_isochrone_grid(
    ages_gyr: Sequence[float] | None = None,
    metallicities: Sequence[float] | None = None,
) -> list[dict[str, float]]:
    """Default grid of old, metal-poor isochrones.

    For the streams whose age or [Fe/H] is unknown, and for the generic
    search. One isochrone per (age, metallicity) pair.

    Parameters
    ----------
    ages_gyr : sequence of float, optional
        Ages (Gyr). Default: the proposal below.
    metallicities : sequence of float, optional
        Metallicities, as mass fractions Z. Default: the proposal below.

    Returns
    -------
    isochrones : list of dict
        ``{"age": age (Gyr), "metallicity": Z}``, one per isochrone.

    Notes
    -----
    Proposal: the grid of celestial_currents ``configs/delve_dr3_mz_scan.yaml``,
    ages 11 and 12 Gyr and Z = 0.0001 and 0.0004 ([Fe/H] about -2.2 and
    -1.6), so 4 isochrones. Check that streamobs has the Euclid isochrones
    for each (age, Z).
    """
    raise NotImplementedError


def known_stream_isochrone_parameters(
    stream_name: str,
    population_table: pd.DataFrame | None = None,
    default_grid: Sequence[Mapping[str, float]] | None = None,
) -> list[dict[str, float]]:
    """Isochrones of a known stream: its own age and [Fe/H] when known.

    The age and the metallicity Z come from the population table
    (``catalogs/stream_populations.csv``, columns ``age_gyr`` and ``z``; Z is
    given there by the paper or converted from [Fe/H] with
    ``unit_conversions.convert_FeH_to_z``). When the age or Z is missing
    (NaN), it takes every value of the default grid, and the known one is
    kept. Values are never invented.

    Parameters
    ----------
    stream_name : str
        galstreams stream name, as in the ``galstreams_name`` column of the
        population table (one entry per stream, shared by its tracks).
    population_table : pandas.DataFrame, optional
        Output of ``load_stream_population_table``. Default: read with
        ``load_stream_population_table()``; pass it to read the file only once
        for several streams.
    default_grid : sequence of mapping, optional
        Default: ``default_isochrone_grid()``.

    Returns
    -------
    isochrones : list of dict
        ``{"age": age (Gyr), "metallicity": Z}``. One entry when both the age
        and the [Fe/H] are known.

    Notes
    -----
    Code to reuse: ``unit_conversions.convert_FeH_to_z``.

    References
    ----------
    Li et al. 2022, ApJ 928, 30. Martin et al. 2022, MNRAS 516, 5331.
    Shipp et al. 2018, ApJ 862, 114. Bonaca & Price-Whelan 2024,
    arXiv:2405.19410.
    """
    raise NotImplementedError


def load_stream_population_table(path: str | Path | None = None) -> pd.DataFrame:
    """Age and [Fe/H] of the known streams.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        CSV file. Default: ``stream_populations.csv`` next to this module.

    Returns
    -------
    table : pandas.DataFrame
        Indexed by ``galstreams_name`` (the galstreams stream name, shared by
        its tracks). Floats, NaN when unknown: ``age_gyr`` and ``feh`` with
        their upper and lower errors (``*_err_plus``, ``*_err_minus``, as
        given in the papers), and ``z``. Strings, empty when missing:
        ``age_reference``, ``feh_reference``, ``z_reference`` and ``notes``.

    Raises
    ------
    ValueError
        The columns are not those above, a stream appears twice, or an age or
        [Fe/H] is not a number.

    References
    ----------
    Li et al. 2022, ApJ 928, 30 (S5). Martin et al. 2022, MNRAS 516, 5331
    (STREAMFINDER). Shipp et al. 2018, ApJ 862, 114 (DES). Bonaca &
    Price-Whelan 2024, arXiv:2405.19410 (review).
    """
    path = POPULATION_TABLE_PATH if path is None else Path(path)
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = [INDEX_COLUMN, *VALUE_COLUMNS, *TEXT_COLUMNS]
    if list(table.columns) != expected:
        raise ValueError(f"{path}: columns {list(table.columns)}, expected {expected}.")
    duplicated = table[INDEX_COLUMN][table[INDEX_COLUMN].duplicated()].unique()
    if len(duplicated):
        raise ValueError(f"{path}: streams listed more than once: {list(duplicated)}.")

    for column in VALUE_COLUMNS:  # empty cell: NaN; anything else must be a number
        table[column] = pd.to_numeric(table[column].replace("", None), errors="raise").astype(float)
    return table.set_index(INDEX_COLUMN)

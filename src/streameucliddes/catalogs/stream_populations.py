"""Stellar populations of the known streams: age and [Fe/H], with references.

The values are curated by hand from the literature in
``stream_populations.csv``, next to this module, each with its reference. A
missing value is left empty (NaN): it is never invented. ``z``, the
metallicity given to ugali, is taken from the paper when it gives it, or
converted from [Fe/H] with ``unit_conversions.convert_FeH_to_z``
(``z_reference`` says which).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

POPULATION_TABLE_PATH = Path(__file__).with_name("stream_populations.csv")
INDEX_COLUMN = "galstreams_name"
VALUE_COLUMNS = [  # floats, NaN when unknown
    "age_gyr", "age_err_plus", "age_err_minus", "feh", "feh_err_plus", "feh_err_minus", "z",
]
TEXT_COLUMNS = ["age_reference", "feh_reference", "z_reference", "notes"]  # strings, empty when missing


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
    (STREAMFINDER). Bonaca & Price-Whelan 2024, arXiv:2405.19410 (review).
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

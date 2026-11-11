"""Stellar populations of the known streams: age and [Fe/H], with references.

The values are curated by hand from the literature in
``stream_populations.csv``, next to this module. A missing value is left
empty (NaN): it is never invented.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

POPULATION_TABLE_PATH = Path(__file__).with_name("stream_populations.csv")
INDEX_COLUMN = "galstreams_name"
VALUE_COLUMNS = ["age_gyr", "feh"]  # floats, NaN when unknown
TEXT_COLUMNS = ["age_reference", "feh_reference", "notes"]  # strings, empty when missing


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
        its tracks). Columns: ``age_gyr`` and ``feh`` (floats, NaN when
        unknown), ``age_reference``, ``feh_reference`` and ``notes`` (strings,
        empty when missing).

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

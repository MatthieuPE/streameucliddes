"""Stellar populations of the known streams: age and [Fe/H], with references.

The values are curated by hand from the literature in
``stream_populations.csv``, next to this module. A missing value is left
empty (NaN): it is never invented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


def load_stream_population_table(path: str | Path | None = None) -> pd.DataFrame:
    """Age and [Fe/H] of the known streams.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        CSV file. Default: ``stream_populations.csv`` next to this module.

    Returns
    -------
    table : pandas.DataFrame
        Indexed by ``galstreams_name``. Columns: ``age_gyr`` and ``feh`` (NaN
        when unknown), ``age_reference``, ``feh_reference``, ``notes``.

    References
    ----------
    Li et al. 2022, ApJ 928, 30 (S5). Martin et al. 2022, MNRAS 516, 5331
    (STREAMFINDER). Bonaca & Price-Whelan 2024, arXiv:2405.19410 (review).
    """
    raise NotImplementedError

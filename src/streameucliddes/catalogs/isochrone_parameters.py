"""Isochrone parameters of the match filters: age and metallicity.

An isochrone is given as ``{"age": age_gyr, "metallicity": z}``, the
arguments of ``analysis.cmd_match_filter.build_match_filter_polygon``. The
analysis uses the Marigo et al. 2017 isochrones of ugali (PARSEC), so every Z
here is on their scale, that of ``unit_conversions.convert_FeH_to_z``.

The ages and metallicities of the known streams are curated by hand from the
literature in ``stream_populations.csv``, next to this module, each with its
reference. A missing value is left empty (NaN): it is never invented. ``z``,
the metallicity given to ugali, is converted from [Fe/H] with
``unit_conversions.convert_FeH_to_z`` (``z_reference`` says how). A stream whose age or metallicity is unknown
uses the default grid of old, metal-poor isochrones.

Spectroscopic [Fe/H] (Li et al. 2022, S5) comes first, then the Pristine
photometric [Fe/H] (Martin et al. 2022). The ages and Z of Shipp et al. 2018
(DES) are a joint photometric fit, degenerate between age and Z, so they are
kept together: they fill the age and ``z`` only of the streams without
[Fe/H] (Ravi, Wambelong), and only where both were free in the fit (their
Table A.1). For the other streams they fitted, they are in ``notes``. Their Z
is on the Dotter et al. 2008 scale: it is converted to [Fe/H] with ugali's
``Dotter2008.z2feh``, then to the Marigo et al. 2017 Z.

The default isochrone is that of the DES search of Shipp et al. 2018 (and of
streamobs): 13 Gyr, Z = 0.0002 ([Fe/H] = -1.9). On the Marigo et al. 2017
scale, Z = 0.0002 is [Fe/H] = -1.89: the same population.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

POPULATION_TABLE_PATH = Path(__file__).with_name("stream_populations.csv")
INDEX_COLUMN = "galstreams_name"
VALUE_COLUMNS = [  # floats, NaN when unknown
    "age_gyr", "age_err_plus", "age_err_minus", "feh", "feh_err_plus", "feh_err_minus", "z",
]
TEXT_COLUMNS = ["age_reference", "feh_reference", "z_reference", "notes"]  # strings, empty when missing

# Shipp et al. 2018, Sect. 3: matched filter of the DES search, 13 Gyr and Z = 0.0002 ([Fe/H] = -1.9)
DEFAULT_AGES_GYR = (13.0,)
DEFAULT_METALLICITIES = (0.0002,)


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
        Ages (Gyr). Default: ``DEFAULT_AGES_GYR``, 13 Gyr.
    metallicities : sequence of float, optional
        Metallicities, as mass fractions Z (Marigo et al. 2017 scale).
        Default: ``DEFAULT_METALLICITIES``, Z = 0.0002.

    Returns
    -------
    isochrones : list of dict
        ``{"age": age (Gyr), "metallicity": Z}``, one per isochrone, every
        metallicity for the first age, then for the next one.

    Notes
    -----
    The default is the single isochrone of the DES search of Shipp et al.
    2018, Sect. 3. The local ugali Marigo et al. 2017 grid covers 1 to
    13.5 Gyr and Z = 0.0001 to 0.001; outside it, ugali uses the nearest
    isochrone.

    References
    ----------
    Shipp et al. 2018, ApJ 862, 114. Marigo et al. 2017, ApJ 835, 77.
    """
    ages_gyr = DEFAULT_AGES_GYR if ages_gyr is None else ages_gyr
    metallicities = DEFAULT_METALLICITIES if metallicities is None else metallicities
    return [
        {"age": float(age), "metallicity": float(z)}
        for age, z in itertools.product(ages_gyr, metallicities)
    ]


def known_stream_isochrone_parameters(
    stream_name: str,
    population_table: pd.DataFrame | None = None,
    default_grid: Sequence[Mapping[str, float]] | None = None,
) -> list[dict[str, float]]:
    """Isochrones of a known stream: its own age and [Fe/H] when known.

    The age and the metallicity Z come from the population table
    (``catalogs/stream_populations.csv``, columns ``age_gyr`` and ``z``, on
    the Marigo et al. 2017 scale). When the age or Z is missing (NaN), it
    takes every value of the default grid, and the known one is kept. A
    stream not in the table takes the whole default grid. Values are never
    invented.

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
        and Z are known.

    References
    ----------
    Li et al. 2022, ApJ 928, 30. Martin et al. 2022, MNRAS 516, 5331.
    Shipp et al. 2018, ApJ 862, 114. Bonaca & Price-Whelan 2024,
    arXiv:2405.19410.
    """
    population_table = load_stream_population_table() if population_table is None else population_table
    default_grid = default_isochrone_grid() if default_grid is None else default_grid
    if stream_name in population_table.index:
        age, z = population_table.loc[stream_name, ["age_gyr", "z"]]
    else:
        age, z = np.nan, np.nan
    isochrones = []
    for default in default_grid:
        isochrone = {
            "age": float(default["age"] if np.isnan(age) else age),
            "metallicity": float(default["metallicity"] if np.isnan(z) else z),
        }
        if isochrone not in isochrones:  # a known value is shared by the whole grid
            isochrones.append(isochrone)
    return isochrones


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

"""Isochrone parameters of the match filters: isochrone model, age and metallicity.

An isochrone is given as ``{"age": age_gyr, "metallicity": z}``, with an
optional ``"isochrone_model"``: the name of the ugali isochrone model
(``"Marigo2017"``, PARSEC; ``"Dotter2016"``, MIST; ``"Bressan2012"``...),
read with ``unit_conversions.get_isochrone_model`` (by default
``unit_conversions.DEFAULT_ISOCHRONE_MODEL``). These are the arguments of
``analysis.cmd_match_filter.build_match_filter_polygon``. Z is on the scale of
that model: each model has its own relation between [Fe/H] and Z
(``unit_conversions.convert_FeH_to_z``, ``convert_z_to_FeH``, ugali's
relations). The Z of this module, ``DEFAULT_METALLICITIES`` and the
population table, are on the scale of ``Z_SCALE_ISOCHRONE_MODEL``
(Marigo2017), a fixed fact, not a default: for another model they are
converted with the same [Fe/H] (``unit_conversions.convert_metallicity``),
the same population.

ugali takes the isochrone of its grid nearest to the age and Z asked: check
that the grid of the model covers them, for the survey of each band
(``analysis.cmd_match_filter`` warns when it does not).

The ages and metallicities of the known streams are curated by hand from the
literature in ``stream_populations.csv``, next to this module, each with its
reference. A missing value is left empty (NaN): it is never invented. ``z``,
the metallicity given to ugali, is converted from [Fe/H] with
``unit_conversions.convert_FeH_to_z`` for Marigo2017 (``z_reference`` says how). A stream whose age or metallicity is unknown
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

from streameucliddes.unit_conversions import convert_FeH_to_z, convert_metallicity

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from typing import Any


POPULATION_TABLE_PATH = Path(__file__).with_name("stream_populations.csv")
INDEX_COLUMN = "galstreams_name"
VALUE_COLUMNS = [  # floats, NaN when unknown
    "age_gyr", "age_err_plus", "age_err_minus", "feh", "feh_err_plus", "feh_err_minus", "z",
]
TEXT_COLUMNS = ["age_reference", "feh_reference", "z_reference", "notes"]  # strings, empty when missing

# Shipp et al. 2018, Sect. 3: matched filter of the DES search, 13 Gyr and Z = 0.0002 ([Fe/H] = -1.9)
DEFAULT_AGES_GYR = (13.0,)
DEFAULT_METALLICITIES = (0.0002,)  # on the scale of Z_SCALE_ISOCHRONE_MODEL
# Name of the ugali isochrone model on whose scale the Z of this module are (DEFAULT_METALLICITIES, the "z" column of
# the population table): Marigo et al. 2017. A fixed fact of these numbers, not a default to change
Z_SCALE_ISOCHRONE_MODEL = "Marigo2017"


# ===================================================
# Isochrones of an isochrone model
# ===================================================


def _with_isochrone_model(isochrone: dict[str, Any], isochrone_model: str | None) -> dict[str, Any]:
    """The isochrone with its ``"isochrone_model"`` when one is given (none: the default, no key)."""
    return isochrone if isochrone_model is None else {**isochrone, "isochrone_model": isochrone_model}


def with_isochrone_model(isochrones: Iterable[Mapping[str, Any]], isochrone_model: str | None) -> list[dict[str, Any]]:
    """Copies of isochrones, ``isochrone_model`` set on those without their own (their Z read on its scale).

    Used by the searches for the isochrones given (``isochrone``,
    ``default_grid``) with an ``isochrone_model``. None: unchanged copies.
    """
    return [dict(isochrone) if "isochrone_model" in isochrone
            else _with_isochrone_model(dict(isochrone), isochrone_model) for isochrone in isochrones]


# ===================================================
# Isochrones
# ===================================================


def default_isochrone_grid(
    ages_gyr: Sequence[float] | None = None,
    metallicities: Sequence[float] | None = None,
    *,
    isochrone_model: str | None = None,
) -> list[dict[str, Any]]:
    """Default grid of old, metal-poor isochrones.

    For the streams whose age or [Fe/H] is unknown, and for the generic
    search. One isochrone per (age, metallicity) pair.

    Parameters
    ----------
    ages_gyr : sequence of float, optional
        Ages (Gyr). Default: ``DEFAULT_AGES_GYR``, 13 Gyr.
    metallicities : sequence of float, optional
        Metallicities, as mass fractions Z on the scale of the isochrone
        model. Default: ``DEFAULT_METALLICITIES`` (Z = 0.0002 on the scale of
        ``Z_SCALE_ISOCHRONE_MODEL``), converted to the model for the same
        [Fe/H] (``unit_conversions.convert_metallicity``).
    isochrone_model : str, optional
        Name of the ugali isochrone model, added as ``"isochrone_model"``.
        None: none added; the isochrones then use
        ``unit_conversions.get_isochrone_model()``.

    Returns
    -------
    isochrones : list of dict
        ``{"age": age (Gyr), "metallicity": Z}`` (and ``"isochrone_model"``), one per
        isochrone, every metallicity for the first age, then for the next one.

    Notes
    -----
    The default is the single isochrone of the DES search of Shipp et al.
    2018, Sect. 3. The local ugali Marigo et al. 2017 grids (DES, Euclid)
    cover 1 to 13.5 Gyr and Z = 0.0001 to 0.001; the Dotter et al. 2016
    Euclid grid 1 to 13.5 Gyr and Z = 0.00001 to 0.001, its DES grid only 10
    and 12 Gyr, Z = 0.0001 and 0.0002 (2026-11-17). Outside a grid, ugali
    uses the nearest isochrone.

    References
    ----------
    Shipp et al. 2018, ApJ 862, 114. Marigo et al. 2017, ApJ 835, 77.
    """
    ages_gyr = DEFAULT_AGES_GYR if ages_gyr is None else ages_gyr
    if metallicities is None:
        metallicities = [convert_metallicity(z, Z_SCALE_ISOCHRONE_MODEL, isochrone_model)
                         for z in DEFAULT_METALLICITIES]
    return [
        _with_isochrone_model({"age": float(age), "metallicity": float(z)}, isochrone_model)
        for age, z in itertools.product(ages_gyr, metallicities)
    ]


def isochrone_from_feh(
    age: float | None = None,
    *,
    feh: float | None = None,
    metallicity: float | None = None,
    isochrone_model: str | None = None,
) -> dict[str, Any]:
    """One isochrone ``{"age", "metallicity"}``, from [Fe/H] or Z, the defaults for what is unknown.

    Parameters
    ----------
    age : float, optional
        Age (Gyr). None or NaN: ``DEFAULT_AGES_GYR[0]``.
    feh : float, optional
        [Fe/H], converted to Z on the scale of the isochrone model
        (``unit_conversions.convert_FeH_to_z``). None or NaN: unknown.
    metallicity : float, optional
        Z on the scale of the isochrone model, when [Fe/H] is not given. None
        or NaN, and no [Fe/H]: ``DEFAULT_METALLICITIES[0]``, converted to the
        model.
    isochrone_model : str, optional
        Name of the ugali isochrone model, e.g. ``"Dotter2016"``, added as
        ``"isochrone_model"``. None: none added; the model is
        ``unit_conversions.get_isochrone_model()``.

    Returns
    -------
    isochrone : dict
        ``{"age": age (Gyr), "metallicity": Z}`` (and ``"isochrone_model"``).

    Raises
    ------
    ValueError
        Both ``feh`` and ``metallicity`` given.
    """
    known = [value is not None and np.isfinite(value) for value in (age, feh, metallicity)]
    if known[1] and known[2]:
        raise ValueError("Give metallicity (Z) or feh, not both.")
    if known[1]:
        z = convert_FeH_to_z(feh, isochrone_model)
    elif known[2]:
        z = float(metallicity)
    else:
        z = convert_metallicity(DEFAULT_METALLICITIES[0], Z_SCALE_ISOCHRONE_MODEL, isochrone_model)
    age = float(age) if known[0] else float(DEFAULT_AGES_GYR[0])
    return _with_isochrone_model({"age": age, "metallicity": float(z)}, isochrone_model)


def known_stream_isochrone_parameters(
    stream_name: str,
    population_table: pd.DataFrame | None = None,
    default_grid: Sequence[Mapping[str, float]] | None = None,
    *,
    isochrone_model: str | None = None,
) -> list[dict[str, Any]]:
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
        Default: ``default_isochrone_grid(isochrone_model=isochrone_model)``.
    isochrone_model : str, optional
        Name of the ugali isochrone model: the Z of the table (on the scale of
        ``Z_SCALE_ISOCHRONE_MODEL``) converted to it for the same [Fe/H]
        (``unit_conversions.convert_metallicity``), and added as
        ``"isochrone_model"``. None: none added; the model is
        ``unit_conversions.get_isochrone_model()``.

    Returns
    -------
    isochrones : list of dict
        ``{"age": age (Gyr), "metallicity": Z}`` (and ``"isochrone_model"``). One entry
        when both the age and Z are known.

    References
    ----------
    Li et al. 2022, ApJ 928, 30. Martin et al. 2022, MNRAS 516, 5331.
    Shipp et al. 2018, ApJ 862, 114. Bonaca & Price-Whelan 2024,
    arXiv:2405.19410.
    """
    population_table = load_stream_population_table() if population_table is None else population_table
    default_grid = default_isochrone_grid(isochrone_model=isochrone_model) if default_grid is None else default_grid
    if stream_name in population_table.index:
        age, z = population_table.loc[stream_name, ["age_gyr", "z"]]
    else:
        age, z = np.nan, np.nan
    if not np.isnan(z):
        z = convert_metallicity(z, Z_SCALE_ISOCHRONE_MODEL, isochrone_model)
    isochrones = []
    for default in default_grid:
        isochrone = _with_isochrone_model({
            "age": float(default["age"] if np.isnan(age) else age),
            "metallicity": float(default["metallicity"] if np.isnan(z) else z),
        }, isochrone_model)
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

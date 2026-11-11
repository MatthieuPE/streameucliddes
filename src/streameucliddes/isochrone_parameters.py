"""Isochrone parameters of the match filters: age and metallicity.

An isochrone is given as ``{"age": age_gyr, "metallicity": z}``, the
arguments of ``analysis.cmd_match_filter.build_match_filter_polygon``. A
stream whose age or [Fe/H] is unknown uses the default grid of old,
metal-poor isochrones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd


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
    population_table: pd.DataFrame,
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
    population_table : pandas.DataFrame
        Output of ``catalogs.stream_populations.load_stream_population_table``.
        Passed in rather than read here: this core module does not import
        ``catalogs/`` (layering rule).
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
    Bonaca & Price-Whelan 2024, arXiv:2405.19410.
    """
    raise NotImplementedError

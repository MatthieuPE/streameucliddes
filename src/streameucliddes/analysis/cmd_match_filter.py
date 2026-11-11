"""Colour-magnitude match filters: selection of the stars near an isochrone.

Only the selection: the catalog is prepared before (columns checked, cuts,
stream frame) by ``analysis.catalog_preparation``.

A match filter is ``{"name", "reference_band": (namespace, band),
"color_bands": [(namespace, band), ...]}``, with optional ``color_cut`` and
``red_cap_value`` (arguments of ``streamobs.match_filter.build_match_filter``).
A star is selected when it is inside the polygon of every colour
``reference_band - color_band``, the magnitude axis being ``reference_band``.
Magnitudes are the ``<namespace>_<band>_obs`` columns of
``catalogs.euclid_des_catalog`` (the streamobs convention), used as they are:
the Euclid x DES magnitudes are already dereddened.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
from streamobs.match_filter import build_match_filter, is_in_match_filter

from streameucliddes.catalogs.isochrone_parameters import default_isochrone_grid

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import pandas as pd
    from numpy.typing import ArrayLike, NDArray

# Euclid VIS vs VIS - H
DEFAULT_MATCH_FILTER = {
    "name": "default_euclid_vis_h",
    "reference_band": ("euclid_q1", "VIS"),
    "color_bands": [("euclid_q1", "H")],
}
# DES g vs g - r, as the DES search of Shipp et al. 2018
DES_GR_MATCH_FILTER = {
    "name": "des_gr",
    "reference_band": ("des_yr6", "g"),
    "color_bands": [("des_yr6", "r")],
}


@functools.lru_cache(maxsize=1024)
def build_match_filter_polygon(distance_modulus, age, metallicity, band_1, band_2, survey_1, survey_2, color_cut=True, red_cap_value=None):
    """
    build_match_filter polygon, cached: it only depends on these parameters, which
    repeat over a scan (e.g. the same distance modulus for every surface brightness).
    """
    mf_polygon = build_match_filter(
        distance_modulus=distance_modulus,
        age=age,
        metallicity=metallicity,
        band_1=band_1,
        band_2=band_2,
        survey_1=survey_1,
        survey_2=survey_2,
        color_cut=color_cut,
        red_cap_value=red_cap_value
    )
    mf_polygon.setflags(write=False)  # shared between calls
    return mf_polygon


def match_filter_columns(match_filter: Mapping[str, Any] | None = None) -> list[str]:
    """Magnitude columns read by a match filter.

    Parameters
    ----------
    match_filter : mapping, optional
        Match filter (see the module docstring). Default:
        ``DEFAULT_MATCH_FILTER``.

    Returns
    -------
    columns : list of str
        ``<namespace>_<band>_obs`` of the reference band, then of each colour
        band.
    """
    match_filter = DEFAULT_MATCH_FILTER if match_filter is None else match_filter
    bands = [match_filter["reference_band"], *match_filter["color_bands"]]
    return [f"{namespace}_{band}_obs" for namespace, band in bands]


def match_filter_polygons(
    distance_modulus: float,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
) -> list[NDArray[np.float64]]:
    """Polygons of a match filter at one distance modulus, one per colour band.

    Built for one isochrone and cached (``build_match_filter_polygon``).

    Parameters
    ----------
    distance_modulus : float
        Distance modulus of the filter.
    isochrone : mapping, optional
        ``{"age": age (Gyr), "metallicity": Z}``. Default: the first
        isochrone of ``catalogs.isochrone_parameters.default_isochrone_grid``.
    match_filter : mapping, optional
        See the module docstring. Default: ``DEFAULT_MATCH_FILTER``.

    Returns
    -------
    polygons : list of numpy.ndarray, shape (n_vertices, 2)
        In the order of ``match_filter["color_bands"]``. Vertices are
        (colour, magnitude) pairs: colour ``reference_band - color_band``,
        magnitude ``reference_band`` (``streamobs.match_filter.build_match_filter``).
        Read-only: shared between calls.
    """
    isochrone = default_isochrone_grid()[0] if isochrone is None else isochrone
    match_filter = DEFAULT_MATCH_FILTER if match_filter is None else match_filter
    namespace_1, band_1 = match_filter["reference_band"]
    return [
        build_match_filter_polygon(
            float(distance_modulus),
            float(isochrone["age"]),
            float(isochrone["metallicity"]),
            band_1,
            band_2,
            namespace_1.split("_")[0],  # isochrone survey: "des_yr6" -> "des"
            namespace_2.split("_")[0],
            color_cut=match_filter.get("color_cut", True),
            red_cap_value=match_filter.get("red_cap_value", None),
        )
        for namespace_2, band_2 in match_filter["color_bands"]
    ]


def select_stars_in_match_filter(
    catalog: pd.DataFrame,
    distance_modulus: float,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    mask: ArrayLike | None = None,
) -> NDArray[np.bool_]:
    """Stars inside the colour-magnitude match filter at one distance modulus.

    A star is selected when it is inside the polygon of every colour of
    ``match_filter`` (``match_filter_polygons``). Each polygon test only runs
    on the stars still selected.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars, with the magnitude columns of ``match_filter_columns``, e.g.
        the output of ``catalog_preparation.prepare_match_filter_catalog``
        (which applies the cuts).
    distance_modulus : float
        Distance modulus of the filter.
    isochrone : mapping, optional
        ``{"age": age (Gyr), "metallicity": Z}``. Default: the first
        isochrone of ``catalogs.isochrone_parameters.default_isochrone_grid``.
    match_filter : mapping, optional
        See the module docstring. Default: ``DEFAULT_MATCH_FILTER``.
    mask : array_like of bool, optional
        Stars to consider (e.g. the observed ones); the others are not
        selected.

    Returns
    -------
    selected : numpy.ndarray of bool, shape (len(catalog),)
        True for the stars in the filter.

    References
    ----------
    Rockosi et al. 2002, AJ 124, 349. Grillmair 2009, ApJ 693, 1118.
    Shipp et al. 2018, ApJ 862, 114.
    """
    match_filter = DEFAULT_MATCH_FILTER if match_filter is None else match_filter
    reference_column, *color_columns = match_filter_columns(match_filter)
    polygons = match_filter_polygons(distance_modulus, isochrone, match_filter)

    selected = np.ones(len(catalog), dtype=bool) if mask is None else np.array(mask, dtype=bool)
    reference_mag = catalog[reference_column].to_numpy()
    for polygon, color_column in zip(polygons, color_columns):
        idx = np.flatnonzero(selected)
        selected[idx] = is_in_match_filter(
            reference_mag[idx], catalog[color_column].to_numpy()[idx], polygon_vertices=polygon
        )
    return selected

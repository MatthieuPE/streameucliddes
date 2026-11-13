"""Colour-magnitude match filters: selection of the stars near an isochrone.

Only the selection: the catalog is prepared before (columns checked, cuts,
stream frame) by ``analysis.catalog_preparation``.

A match filter is ``{"name", "reference_band": (namespace, band),
"color_bands": [(namespace, band), ...]}``, with optional ``color_cut`` and
``red_cap_value`` (arguments of ``streamobs.match_filter.build_match_filter``).
A star is selected when it is inside the polygon of every colour
``reference_band - color_band``, the magnitude axis being ``reference_band``.
Magnitudes are the ``<namespace>_<band>_obs`` columns of
``catalogs.euclid_catalog`` (the streamobs convention), used as they are:
the catalog magnitudes are already dereddened.

Also the colour-magnitude diagrams of a set of stars, computed here and drawn
by ``plotting.plot_cmd``: 2D histograms of colour ``mag_cols[0] -
mag_cols[1]`` and magnitude ``mag_cols[0]`` (``hess_diagram``), per deg²
for a region of known area, and the on - off difference of two regions
(``difference_hess``, Shipp et al. 2018, Sect. 5); and the isochrone track
in the bands of a match filter (``isochrone_cmd_track``).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from streamobs.match_filter import _isochrone_locus, build_match_filter, is_in_match_filter

from streameucliddes.catalogs.isochrone_parameters import default_isochrone_grid
from streameucliddes.unit_conversions import convert_FeH_to_z

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

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
# ugali isochrone model of the polygons (default of build_match_filter): Marigo et al. 2017 (PARSEC)
ISOCHRONE_MODEL = "Marigo2017"

# Hess diagrams: default bins (mag). Bin sizes of celestial_currents hess.py; ranges of the
# VIS - H CMDs of notebooks/known_stream_search_example.ipynb
DEFAULT_COLOR_BIN = 0.04
DEFAULT_MAG_BIN = 1 / 6
DEFAULT_COLOR_RANGE = (-1.0, 1.5)
DEFAULT_MAG_RANGE = (17.0, 27.0)


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
            _isochrone_survey(namespace_1),
            _isochrone_survey(namespace_2),
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


def isochrone_cmd_track(
    distance_modulus: float,
    age: float,
    metallicity: float | None = None,
    *,
    feh: float | None = None,
    match_filter: Mapping[str, Any] | None = None,
    color_index: int = 0,
) -> dict[str, Any]:
    """Colour and magnitude of an isochrone in the bands of a match filter.

    The isochrone locus of ``streamobs.match_filter._isochrone_locus``, the
    one ``build_match_filter`` uses (ugali, model ``ISOCHRONE_MODEL``), with
    the surveys and bands of the polygons (``match_filter_polygons``): colour
    ``reference_band - color_bands[color_index]``, magnitude
    ``reference_band`` plus ``distance_modulus``. The track and the polygon
    then come from the same model. ``_isochrone_locus`` is private in
    streamobs: check it when streamobs changes.

    Parameters
    ----------
    distance_modulus : float
        Distance modulus (mag).
    age : float
        Age (Gyr).
    metallicity : float, optional
        Z, on the Marigo et al. 2017 scale. Exactly one of ``metallicity``
        and ``feh``.
    feh : float, optional
        [Fe/H], converted to Z with ``unit_conversions.convert_FeH_to_z``
        (Bressan et al. 2012).
    match_filter : mapping, optional
        See the module docstring. Default: ``DEFAULT_MATCH_FILTER``.
    color_index : int, optional
        Colour band: ``match_filter["color_bands"][color_index]``.

    Returns
    -------
    track : dict
        ``{"color", "mag", "age", "metallicity", "feh", "distance_modulus"}``:
        ``color`` and ``mag`` arrays (mag) along the isochrone;
        ``metallicity`` the Z used, ``feh`` the [Fe/H] given (None when Z was
        given).

    Raises
    ------
    ValueError
        Neither or both of ``metallicity`` and ``feh``.

    References
    ----------
    Marigo et al. 2017, ApJ 835, 77; Bressan et al. 2012, MNRAS 427, 127.
    """
    if (metallicity is None) == (feh is None):
        raise ValueError("Give exactly one of metallicity (Z) and feh.")
    z = float(convert_FeH_to_z(feh)) if feh is not None else float(metallicity)
    match_filter = DEFAULT_MATCH_FILTER if match_filter is None else match_filter
    namespace_1, band_1 = match_filter["reference_band"]
    namespace_2, band_2 = match_filter["color_bands"][color_index]
    color, absolute_mag = _isochrone_locus(
        ISOCHRONE_MODEL, float(age), z, band_1, _isochrone_survey(namespace_1), band_2, _isochrone_survey(namespace_2)
    )
    return {
        "color": np.asarray(color, dtype=float), "mag": np.asarray(absolute_mag, dtype=float) + distance_modulus,
        "age": float(age), "metallicity": z, "feh": None if feh is None else float(feh),
        "distance_modulus": float(distance_modulus),
    }


def _isochrone_survey(namespace: str) -> str:
    """ugali survey of a column namespace: "des_yr6" -> "des", "euclid_dr1" -> "euclid"."""
    return namespace.split("_")[0]


def hess_diagram(
    stars: pd.DataFrame,
    mag_cols: Sequence[str],
    *,
    mask: ArrayLike | None = None,
    area_deg2: float | None = None,
    color_bins: ArrayLike | None = None,
    mag_bins: ArrayLike | None = None,
) -> dict[str, Any]:
    """Colour-magnitude 2D histogram of stars: counts, or a Hess diagram per deg².

    Colour ``mag_cols[0] - mag_cols[1]`` and magnitude ``mag_cols[0]``, the
    convention of the match-filter polygons. With ``area_deg2``, the counts
    are divided by it: stars / deg² / bin.

    Parameters
    ----------
    stars : pandas.DataFrame
        Stars with the ``mag_cols`` columns, e.g. ``search.stars`` of
        ``analysis.stream_frame_search.prepare_search``. Not copied.
    mag_cols : sequence of two str
        Magnitude columns, e.g. ``match_filter_columns(match_filter)[0]`` and
        ``[1 + color_index]``.
    mask : array_like of bool, optional
        Stars to count, e.g. those of the on region
        (``stream_frames.in_phi12_window``). Default: all.
    area_deg2 : float, optional
        Area of the region (deg²): density per deg². None: counts only
        (``density`` = ``counts``).
    color_bins, mag_bins : array_like, optional
        Bin edges (mag). Default: ``DEFAULT_COLOR_RANGE`` and
        ``DEFAULT_MAG_RANGE`` every ``DEFAULT_COLOR_BIN`` and
        ``DEFAULT_MAG_BIN``. Give the same bins to the on and off diagrams.

    Returns
    -------
    hess : dict
        ``{"density", "counts", "error", "color_edges", "mag_edges",
        "area_deg2", "n_stars", "color_label", "mag_label"}``: ``density``
        and ``counts`` of shape (n_color, n_mag), ``error`` =
        ``sqrt(counts) / area_deg2`` (``sqrt(counts)`` without area);
        ``n_stars`` the stars inside the bins; labels from the band names.

    Raises
    ------
    ValueError
        ``area_deg2 <= 0``.

    References
    ----------
    Shipp et al. 2018, ApJ 862, 114, Sect. 5.
    """
    if area_deg2 is not None and area_deg2 <= 0:
        raise ValueError(f"area_deg2 must be > 0, not {area_deg2}.")
    mag_column, color_column = mag_cols
    if color_bins is None:
        color_bins = np.arange(DEFAULT_COLOR_RANGE[0], DEFAULT_COLOR_RANGE[1] + DEFAULT_COLOR_BIN / 2, DEFAULT_COLOR_BIN)
    if mag_bins is None:
        mag_bins = np.arange(DEFAULT_MAG_RANGE[0], DEFAULT_MAG_RANGE[1] + DEFAULT_MAG_BIN / 2, DEFAULT_MAG_BIN)
    magnitude = pd.to_numeric(stars[mag_column], errors="coerce").to_numpy(dtype=float)
    other = pd.to_numeric(stars[color_column], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(magnitude) & np.isfinite(other)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    counts, color_edges, mag_edges = np.histogram2d(
        magnitude[keep] - other[keep], magnitude[keep], bins=[np.asarray(color_bins), np.asarray(mag_bins)]
    )
    scale = 1.0 if area_deg2 is None else float(area_deg2)
    return {
        "density": counts / scale, "counts": counts, "error": np.sqrt(counts) / scale,
        "color_edges": color_edges, "mag_edges": mag_edges, "area_deg2": None if area_deg2 is None else scale,
        "n_stars": int(counts.sum()),
        "color_label": f"{_band_name(mag_column)} - {_band_name(color_column)}", "mag_label": _band_name(mag_column),
    }


def difference_hess(on_hess: Mapping[str, Any], off_hess: Mapping[str, Any]) -> dict[str, Any]:
    """On - off Hess diagram.

    Parameters
    ----------
    on_hess, off_hess : mapping
        From ``hess_diagram`` with ``area_deg2``, with the same bin edges.

    Returns
    -------
    hess : dict
        The keys of ``hess_diagram``: ``density`` on - off, ``error`` the two
        errors in quadrature, ``area_deg2`` that of the on region; ``counts``
        the excess stars of the on region per bin, ``density * area_deg2``
        (on counts minus off counts scaled by the ratio of the areas), and
        ``n_stars`` their sum: the excess stars of the on region (float).

    Raises
    ------
    ValueError
        The bin edges differ, or a diagram has no area.
    """
    if on_hess["area_deg2"] is None or off_hess["area_deg2"] is None:
        raise ValueError("Both Hess diagrams need an area_deg2 (densities per deg²).")
    for name in ("color_edges", "mag_edges"):
        if not np.array_equal(on_hess[name], off_hess[name]):
            raise ValueError(f"The on and off Hess diagrams have different {name}.")
    excess = on_hess["counts"] - off_hess["counts"] * on_hess["area_deg2"] / off_hess["area_deg2"]
    return {
        **{key: on_hess[key] for key in ("color_edges", "mag_edges", "area_deg2", "color_label", "mag_label")},
        "density": on_hess["density"] - off_hess["density"], "error": np.hypot(on_hess["error"], off_hess["error"]),
        "counts": excess, "n_stars": float(excess.sum()),
    }


def _band_name(column: str) -> str:
    """Band of a magnitude column: "euclid_dr1_VIS_obs" -> "VIS"."""
    return column.removesuffix("_obs").rsplit("_", 1)[-1]

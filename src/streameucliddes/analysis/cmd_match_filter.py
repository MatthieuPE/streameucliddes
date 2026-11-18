"""Colour-magnitude match filters: selection of the stars near an isochrone.

Only the selection: the catalog is prepared before (columns checked, cuts,
stream frame) by ``analysis.catalog_preparation``.

A match filter is ``{"name", "reference_band": (namespace, band),
"color_bands": [(namespace, band), ...]}``, with optional ``color_cut``,
``red_cap_value``, ``color_spread`` and ``error_multiplier`` (arguments of
``streamobs.match_filter.build_match_filter``; the width of the polygon is
``error_multiplier`` times the photometric error plus ``color_spread``, each
(blue side, red side); absent: those of streamobs, ``(0.05, 0.05)`` mag and
``(2, 2)``; ``DEFAULT_MATCH_FILTER`` sets its own ``error_multiplier`` and
``red_cap_value``), and ``vis_photometry``, the photometry of the Euclid VIS magnitudes:
``"psf"`` (default) or ``"templfit"``, the template-fitting magnitude of VIS
PSF-matched to the other band of each colour (``catalogs.euclid_catalog.vis_band``;
VIS - H uses ``VIS_TO_H``, DR1 only). The isochrone and the polygons do not depend
on it: only the catalog columns do (``match_filter_cmd_columns``). Injected stars
have one simulated VIS, copied to their ``VIS_TO_<band>`` columns
(``injection.inject.add_vis_templfit_magnitudes``).
Its polygons are built for an isochrone ``{"age", "metallicity"}`` with an
optional ``"isochrone_model"``, the name of the ugali isochrone model
(``"Dotter2016"``, ``"Marigo2017"``...; by default
``unit_conversions.DEFAULT_ISOCHRONE_MODEL``; read with
``unit_conversions.get_isochrone_model``; ``catalogs.isochrone_parameters``).
ugali builds from that name a ugali isochrone (``ugali_isochrone``, an
instance of the ugali class of that name).
A star is selected when it is inside the polygon of every colour
``reference_band - color_band``, the magnitude axis being ``reference_band``.
Magnitudes are the ``<namespace>_<band>_obs`` columns of
``catalogs.euclid_catalog`` (the streamobs convention), used as they are:
the catalog magnitudes are already dereddened. The columns of each colour come
from ``match_filter_cmd_columns``.

Also the colour-magnitude diagrams of a set of stars, computed here and drawn
by ``plotting.plot_cmd``: 2D histograms of colour ``mag_cols[0] -
mag_cols[1]`` and magnitude ``mag_cols[0]`` (``hess_diagram``), per deg²
for a region of known area, and the on - off difference of two regions
(``difference_hess``, Shipp et al. 2018, Sect. 5); and the isochrone track
in the bands of a match filter (``isochrone_cmd_track``).
"""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from streamobs.match_filter import _isochrone_locus, build_match_filter, is_in_match_filter
from ugali.isochrone import factory as isochrone_factory

from streameucliddes.catalogs.euclid_catalog import DEFAULT_VIS_PHOTOMETRY, vis_band
from streameucliddes.catalogs.isochrone_parameters import get_isochrone
from streameucliddes.unit_conversions import convert_FeH_to_z, get_isochrone_model

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from numpy.typing import ArrayLike, NDArray

# Euclid VIS vs VIS - H: the default match filter, read through get_match_filter. Its width and red cap: the best SNR
# on Aliqa Uma in Euclid DR1 (notebooks/optimization/stream_optimization.ipynb), error_multiplier 3 instead of the 2
# of streamobs and red_cap_value 0.55 mag of VIS - H instead of color_max - 0.25 = 0.75 (Euclid colours only)
DEFAULT_MATCH_FILTER = {
    "name": "default_euclid_vis_h",
    "reference_band": ("euclid_q1", "VIS"),
    "color_bands": [("euclid_q1", "H")],
    "error_multiplier": (3.0, 3.0),
    "red_cap_value": 0.55,
}
# DES g vs g - r, as the DES search of Shipp et al. 2018
DES_GR_MATCH_FILTER = {
    "name": "des_gr",
    "reference_band": ("des_yr6", "g"),
    "color_bands": [("des_yr6", "r")],
}

# Hess diagrams: default bins (mag). Bin sizes of celestial_currents hess.py; ranges of the
# VIS - H CMDs of notebooks/known_stream_search_example.ipynb
DEFAULT_COLOR_BIN = 0.04
DEFAULT_MAG_BIN = 1 / 6
DEFAULT_COLOR_RANGE = (-1.0, 1.5)
DEFAULT_MAG_RANGE = (17.0, 27.0)
# CMDs of a match filter: the extent of its polygons plus this margin (mag)
DEFAULT_CMD_MARGIN_MAG = 0.5


@functools.lru_cache(maxsize=1024)
def build_match_filter_polygon(distance_modulus, age, metallicity, band_1, band_2, survey_1, survey_2, color_cut=True,
                               red_cap_value=None, isochrone_model=None, color_spread=None, error_multiplier=None):
    """
    build_match_filter polygon, cached: it only depends on these parameters, which
    repeat over a scan (e.g. the same distance modulus for every surface brightness).
    ``isochrone_model``: name of the ugali isochrone model (``"Marigo2017"``, ``"Dotter2016"``...;
    None: ``unit_conversions.get_isochrone_model()``); a warning when its grid has no
    isochrone near ``age`` and ``metallicity`` (``_check_isochrone_grid``).
    ``color_spread`` (mag) and ``error_multiplier``: (blue side, red side) tuples, the
    width of the polygon; None: the streamobs default.
    """
    isochrone_model = get_isochrone_model(isochrone_model)
    for survey, band in {survey_1: band_1, survey_2: band_2}.items():  # one grid per survey
        _check_isochrone_grid(isochrone_model, age, metallicity, survey, band)
    width = {name: list(value) for name, value in
             {"color_spread": color_spread, "error_multiplier": error_multiplier}.items() if value is not None}
    mf_polygon = build_match_filter(
        distance_modulus=distance_modulus,
        age=age,
        metallicity=metallicity,
        band_1=band_1,
        band_2=band_2,
        survey_1=survey_1,
        survey_2=survey_2,
        color_cut=color_cut,
        red_cap_value=red_cap_value,
        isochrone_model=isochrone_model,
        **width,
    )
    mf_polygon.setflags(write=False)  # shared between calls
    return mf_polygon


@functools.lru_cache(maxsize=256)
def _check_isochrone_grid(
    isochrone_model: str, age: float, metallicity: float, survey: str, band: str
) -> tuple[float, float]:
    """Age and Z of the isochrone ugali uses for this request (the nearest of its grid); a warning when off the grid.

    ugali takes, without a warning, the file of its grid nearest to (age,
    Z): far from the request when the grid of the isochrone model for that survey
    does not reach it (e.g. Z below 1e-4 for Marigo2017) or is sparse (the
    local DES Dotter2016 grid: 10 and 12 Gyr, Z 1e-4 and 2e-4). Warns when
    the age or Z asked is farther than one grid step from the value used
    (``_off_grid``): beyond the range of the grid, or between the values of
    a sparse grid; not for the rounding to the nearest grid value. Cached:
    once per request.
    """
    ugali_isochrone = isochrone_factory(isochrone_model, age=age, z=metallicity, survey=survey, band_1=band,
                                        band_2=band)  # an instance of the ugali class named isochrone_model
    used_age, used_z = (float(value) for value in ugali_isochrone.filename2params(ugali_isochrone.get_filename()))
    if _off_grid(age, used_age, ugali_isochrone.agrid) or _off_grid(metallicity, used_z, ugali_isochrone.zgrid):
        warnings.warn(
            f"No {isochrone_model} isochrone near {age:g} Gyr, Z = {metallicity:.3g} for {survey}: ugali uses the "
            f"nearest of its grid, {used_age:g} Gyr, Z = {used_z:.3g} ({ugali_isochrone.get_dirname()}).",
            stacklevel=2,
        )
    return used_age, used_z


def _off_grid(asked: float, used: float, grid_values: ArrayLike) -> bool:
    """Whether ``asked`` is off a grid: more than one grid step from ``used`` (the grid value ugali takes).

    The step is the gap between ``used`` and its nearest other grid value;
    a grid of one value has none: off unless ``asked`` is that value.
    """
    values = np.unique(np.asarray(grid_values, dtype=float))
    others = values[~np.isclose(values, used, rtol=1e-9, atol=0.0)]
    step = np.min(np.abs(others - used)) if others.size else 0.0
    return bool(abs(asked - used) > step * (1 + 1e-9) if step > 0 else not np.isclose(asked, used, rtol=1e-9))


def match_filter_columns(match_filter: Mapping[str, Any] | None = None) -> list[str]:
    """Magnitude columns read by a match filter, each once.

    Parameters
    ----------
    match_filter : mapping, optional
        Match filter (see the module docstring). Default:
        ``get_match_filter()`` (``DEFAULT_MATCH_FILTER``).

    Returns
    -------
    columns : list of str
        The reference column of the first colour (the magnitude axis), the
        colour-band columns, then the reference columns of the other colours
        when they differ (VIS template fits with several colours): see
        ``match_filter_cmd_columns``.
    """
    pairs = match_filter_cmd_columns(match_filter)
    columns = [pairs[0][0], *(color for _, color in pairs), *(reference for reference, _ in pairs[1:])]
    return list(dict.fromkeys(columns))


def _magnitude_column(band: tuple[str, str], other_band: tuple[str, str], vis_photometry: str) -> str:
    """Catalog column of the magnitude of ``band`` in a colour with ``other_band``, both ``(namespace, band)``.

    A Euclid VIS band follows ``vis_photometry`` (``catalogs.euclid_catalog.vis_band``);
    with ``"templfit"``, the other band must be a Euclid band of the same namespace.
    """
    namespace, name = band
    if name == "VIS" and namespace.startswith("euclid"):
        other_namespace, other_name = other_band
        if vis_photometry != DEFAULT_VIS_PHOTOMETRY and other_namespace != namespace:
            raise ValueError(f"VIS {vis_photometry} magnitudes need a colour with a {namespace} band, "
                             f"not {other_band}")
        name = vis_band(other_name, vis_photometry)
    return f"{namespace}_{name}_obs"


def get_match_filter(match_filter: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """The match filter to use: the one given, else ``DEFAULT_MATCH_FILTER``.

    The single place where the default is chosen: functions take
    ``match_filter=None`` and call this. The default is shared: copy it
    before changing it.

    Parameters
    ----------
    match_filter : mapping, optional
        See the module docstring.

    Returns
    -------
    match_filter : mapping
        ``match_filter`` itself, or ``DEFAULT_MATCH_FILTER``.
    """
    return DEFAULT_MATCH_FILTER if match_filter is None else match_filter


def match_filter_cmd_columns(match_filter: Mapping[str, Any] | None = None) -> list[tuple[str, str]]:
    """Magnitude columns of the CMDs of a match filter: (reference band, colour band), one per colour band.

    Colour ``reference - colour band``, magnitude ``reference``: the
    convention of the polygons, and the ``mag_cols`` of ``hess_diagram``.
    ``<namespace>_<band>_obs`` columns, the Euclid VIS one following the
    ``vis_photometry`` of the match filter: with ``"templfit"``, the reference
    VIS of the colour VIS - H is ``<namespace>_VIS_TO_H_obs``.

    Parameters
    ----------
    match_filter : mapping, optional
        Default: ``get_match_filter()`` (``DEFAULT_MATCH_FILTER``).

    Returns
    -------
    pairs : list of tuple of str
    """
    match_filter = get_match_filter(match_filter)
    vis_photometry = match_filter.get("vis_photometry", DEFAULT_VIS_PHOTOMETRY)
    reference = match_filter["reference_band"]
    return [(_magnitude_column(reference, color, vis_photometry), _magnitude_column(color, reference, vis_photometry))
            for color in match_filter["color_bands"]]


def match_filter_cmd_limits(
    distance_modulus: float,
    isochrone: Mapping[str, float] | None = None,
    match_filter: Mapping[str, Any] | None = None,
    *,
    margin_mag: float = DEFAULT_CMD_MARGIN_MAG,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Colour and magnitude ranges of the CMDs of a match filter: the extent of each polygon plus ``margin_mag``.

    Returns
    -------
    limits : list of ((color_min, color_max), (mag_bright, mag_faint))
        One per colour band (``match_filter_polygons``).
    """
    limits = []
    for polygon in match_filter_polygons(distance_modulus, isochrone, match_filter):
        color, mag = polygon[:, 0], polygon[:, 1]
        limits.append(((float(color.min() - margin_mag), float(color.max() + margin_mag)),
                       (float(mag.min() - margin_mag), float(mag.max() + margin_mag))))
    return limits


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
        ``{"age": age (Gyr), "metallicity": Z}``, with an optional
        ``"isochrone_model"`` (name of the ugali isochrone model, read with
        ``unit_conversions.get_isochrone_model``; Z on its scale). Default:
        ``catalogs.isochrone_parameters.get_isochrone()``.
    match_filter : mapping, optional
        See the module docstring. Default: ``get_match_filter()`` (``DEFAULT_MATCH_FILTER``).

    Returns
    -------
    polygons : list of numpy.ndarray, shape (n_vertices, 2)
        In the order of ``match_filter["color_bands"]``. Vertices are
        (colour, magnitude) pairs: colour ``reference_band - color_band``,
        magnitude ``reference_band`` (``streamobs.match_filter.build_match_filter``).
        Read-only: shared between calls.
    """
    isochrone = get_isochrone(isochrone)
    match_filter = get_match_filter(match_filter)
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
            isochrone_model=get_isochrone_model(isochrone),
            color_spread=_width_tuple(match_filter.get("color_spread")),
            error_multiplier=_width_tuple(match_filter.get("error_multiplier")),
        )
        for namespace_2, band_2 in match_filter["color_bands"]
    ]


def _width_tuple(value: ArrayLike | None) -> tuple[float, float] | None:
    """A ``color_spread`` or ``error_multiplier`` of a match filter as a (blue side, red side) tuple (cache key)."""
    if value is None:
        return None
    blue, red = (float(v) for v in value)  # ValueError unless two values
    return blue, red


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
        ``{"age": age (Gyr), "metallicity": Z}``. Default:
        ``catalogs.isochrone_parameters.get_isochrone()``.
    match_filter : mapping, optional
        See the module docstring. Default: ``get_match_filter()`` (``DEFAULT_MATCH_FILTER``).
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
    match_filter = get_match_filter(match_filter)
    polygons = match_filter_polygons(distance_modulus, isochrone, match_filter)

    selected = np.ones(len(catalog), dtype=bool) if mask is None else np.array(mask, dtype=bool)
    for polygon, (reference_column, color_column) in zip(polygons, match_filter_cmd_columns(match_filter)):
        idx = np.flatnonzero(selected)
        selected[idx] = is_in_match_filter(
            catalog[reference_column].to_numpy()[idx], catalog[color_column].to_numpy()[idx],
            polygon_vertices=polygon,
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
    isochrone_model: str | None = None,
) -> dict[str, Any]:
    """Colour and magnitude of an isochrone in the bands of a match filter.

    The isochrone locus of ``streamobs.match_filter._isochrone_locus``, the
    one ``build_match_filter`` uses (the ugali isochrone named ``isochrone_model``), with
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
        Z, on the scale of the isochrone model. Exactly one of
        ``metallicity`` and ``feh``.
    feh : float, optional
        [Fe/H], converted to Z on the scale of the isochrone model
        (``unit_conversions.convert_FeH_to_z``).
    match_filter : mapping, optional
        See the module docstring. Default: ``get_match_filter()`` (``DEFAULT_MATCH_FILTER``).
    color_index : int, optional
        Colour band: ``match_filter["color_bands"][color_index]``.
    isochrone_model : str, optional
        Name of the ugali isochrone model, e.g. ``"Dotter2016"``. None:
        ``unit_conversions.get_isochrone_model()``.

    Returns
    -------
    track : dict
        ``{"color", "mag", "age", "metallicity", "feh", "distance_modulus",
        "isochrone_model"}``: ``color`` and ``mag`` arrays (mag) along the isochrone;
        ``metallicity`` the Z used, ``feh`` the [Fe/H] given (None when Z was
        given), ``isochrone_model`` the name of the isochrone model used.

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
    isochrone_model = get_isochrone_model(isochrone_model)
    z = convert_FeH_to_z(feh, isochrone_model) if feh is not None else float(metallicity)
    match_filter = get_match_filter(match_filter)
    namespace_1, band_1 = match_filter["reference_band"]
    namespace_2, band_2 = match_filter["color_bands"][color_index]
    survey_1, survey_2 = _isochrone_survey(namespace_1), _isochrone_survey(namespace_2)
    for survey, band in {survey_1: band_1, survey_2: band_2}.items():  # one grid per survey
        _check_isochrone_grid(isochrone_model, float(age), z, survey, band)
    color, absolute_mag = _isochrone_locus(isochrone_model, float(age), z, band_1, survey_1, band_2, survey_2)
    return {
        "color": np.asarray(color, dtype=float), "mag": np.asarray(absolute_mag, dtype=float) + distance_modulus,
        "age": float(age), "metallicity": z, "feh": None if feh is None else float(feh),
        "distance_modulus": float(distance_modulus), "isochrone_model": isochrone_model,
    }


def _isochrone_survey(namespace: str) -> str:
    """ugali survey of a column namespace: "des_yr6" -> "des", "euclid_dr1" -> "euclid"."""
    return namespace.split("_")[0]


def default_hess_bins() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Colour and magnitude bin edges (mag) of every Hess diagram without given bins.

    ``DEFAULT_COLOR_RANGE`` every ``DEFAULT_COLOR_BIN`` and ``DEFAULT_MAG_RANGE``
    every ``DEFAULT_MAG_BIN``: the same pixels in every figure, so that diagrams
    of different selections or match filters compare bin by bin.
    """
    color_bins = np.arange(DEFAULT_COLOR_RANGE[0], DEFAULT_COLOR_RANGE[1] + DEFAULT_COLOR_BIN / 2, DEFAULT_COLOR_BIN)
    mag_bins = np.arange(DEFAULT_MAG_RANGE[0], DEFAULT_MAG_RANGE[1] + DEFAULT_MAG_BIN / 2, DEFAULT_MAG_BIN)
    return color_bins, mag_bins


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
        Bin edges (mag). Default: ``default_hess_bins()``, the same for
        every Hess diagram. Give the same bins to the on and off diagrams.

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
    default_color_bins, default_mag_bins = default_hess_bins()
    color_bins = default_color_bins if color_bins is None else color_bins
    mag_bins = default_mag_bins if mag_bins is None else mag_bins
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


def on_off_hess(
    stars: pd.DataFrame,
    mag_cols: Sequence[str],
    on_mask: ArrayLike,
    on_area_deg2: float,
    off_masks: Sequence[ArrayLike],
    off_areas_deg2: Sequence[float],
    *,
    color_bins: ArrayLike | None = None,
    mag_bins: ArrayLike | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """On, off and on - off Hess diagrams of one region against one or two off regions.

    ``hess_diagram`` of the stars of each region over its area. With two off
    regions (e.g. below and above a stream track), the off diagram is the
    mean of theirs (``mean_off_hess``; one with no area is left out); with
    one (e.g. an annulus around an object), its own. Then ``difference_hess``.

    Parameters
    ----------
    stars : pandas.DataFrame
        Stars, with the ``mag_cols`` columns. Not copied.
    mag_cols : sequence of two str
        Colour ``mag_cols[0] - mag_cols[1]``, magnitude ``mag_cols[0]``.
    on_mask : array_like of bool
        Stars of the on region (and of any selection).
    on_area_deg2 : float
        Its area (deg²).
    off_masks : sequence of array_like of bool
        Stars of each off region: one or two.
    off_areas_deg2 : sequence of float
        Their areas (deg²).
    color_bins, mag_bins : array_like, optional
        Bin edges, the same for every diagram. Default: those of ``hess_diagram``.

    Returns
    -------
    on_hess, off_hess, difference : dict
        See ``hess_diagram``, ``mean_off_hess`` and ``difference_hess``.

    Raises
    ------
    ValueError
        Not one or two off regions.
    """
    if len(off_masks) not in (1, 2) or len(off_masks) != len(off_areas_deg2):
        raise ValueError(f"One or two off regions, each with its area: {len(off_masks)} masks, "
                         f"{len(off_areas_deg2)} areas.")
    bins = {"color_bins": color_bins, "mag_bins": mag_bins}
    on_hess = hess_diagram(stars, mag_cols, mask=on_mask, area_deg2=on_area_deg2, **bins)
    off_hess = [hess_diagram(stars, mag_cols, mask=mask, area_deg2=area if area > 0 else None, **bins)
                for mask, area in zip(off_masks, off_areas_deg2)]
    off_hess = off_hess[0] if len(off_hess) == 1 else mean_off_hess(*off_hess)
    return on_hess, off_hess, difference_hess(on_hess, off_hess)


def mean_off_density(
    density_below: ArrayLike, error_below: ArrayLike, density_above: ArrayLike, error_above: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Off density: the mean of the densities of the off regions below and above the track.

    The two off regions lie on each side of the track, at the same distance:
    the mean of their densities cancels a linear gradient of the field across
    the track (along phi2), whatever area the masks leave in each. Their
    stars together over their areas together would not: the region with more
    area would count more. Where one region has no area (its density NaN,
    e.g. a phi1 bin masked on that side), the off density is that of the
    other alone; where neither has, NaN.

    Parameters
    ----------
    density_below, error_below, density_above, error_above : array_like
        Densities and their Poisson errors (stars/deg²) of the off region
        below and above the track, of the same shape (e.g. per bin); NaN
        where the region has no area.

    Returns
    -------
    density, error : numpy.ndarray
        The mean and its error; NaN where neither region has an area.
    n_regions : numpy.ndarray of int
        Number of off regions in the mean: 2, 1 (one has no area) or 0.
    """
    return mean_density_of_off_parts([density_below, density_above], [error_below, error_above])


def mean_density_of_off_parts(
    densities: Sequence[ArrayLike], errors: Sequence[ArrayLike]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Off density: the mean of the densities of any number of off regions (``mean_off_density`` for two).

    One region (an annulus around an object): its density. Regions with no
    area (density NaN) are left out of the mean.

    Parameters
    ----------
    densities, errors : sequence of array_like
        Density and Poisson error (stars/deg²) of each off region, all of the
        same shape; NaN where a region has no area.

    Returns
    -------
    density, error : numpy.ndarray
        The mean and its error; NaN where no region has an area.
    n_regions : numpy.ndarray of int
        Number of off regions in the mean.
    """
    densities = np.stack([np.asarray(density, dtype=float) for density in densities])
    errors = np.stack([np.asarray(error, dtype=float) for error in errors])
    with_area = np.isfinite(densities)
    n_regions = with_area.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(n_regions > 0, np.where(with_area, densities, 0.0).sum(axis=0) / n_regions, np.nan)
        error = np.where(n_regions > 0, np.sqrt(np.where(with_area, errors**2, 0.0).sum(axis=0)) / n_regions, np.nan)
    return density, error, n_regions


def mean_off_hess(below_hess: Mapping[str, Any], above_hess: Mapping[str, Any]) -> dict[str, Any]:
    """Off Hess diagram: the mean of the Hess diagrams of the off regions below and above the track.

    ``mean_off_density`` of their densities, bin by bin.

    Parameters
    ----------
    below_hess, above_hess : mapping
        From ``hess_diagram``, with the same bin edges; ``area_deg2`` None
        for an off region with no area (the other is then used alone).

    Returns
    -------
    hess : dict
        The keys of ``hess_diagram``: ``density`` the mean of the two
        densities and ``error`` its error; ``counts``, ``n_stars`` and
        ``area_deg2`` the sums over the two regions (so ``counts /
        area_deg2`` is not ``density`` when their areas differ); plus
        ``n_off_regions`` (2, or 1 when one has no area).

    Raises
    ------
    ValueError
        The bin edges differ, or neither region has an area.
    """
    for name in ("color_edges", "mag_edges"):
        if not np.array_equal(below_hess[name], above_hess[name]):
            raise ValueError(f"The Hess diagrams of the off regions have different {name}.")
    with_area = [hess for hess in (below_hess, above_hess) if hess["area_deg2"]]
    if not with_area:
        raise ValueError("Neither off region has an area.")
    arrays = [(hess["density"], hess["error"]) if hess["area_deg2"] else (np.full_like(hess["density"], np.nan),) * 2
              for hess in (below_hess, above_hess)]
    density, error, _ = mean_off_density(*arrays[0], *arrays[1])
    return {
        **{key: below_hess[key] for key in ("color_edges", "mag_edges", "color_label", "mag_label")},
        "density": density, "error": error,
        "counts": below_hess["counts"] + above_hess["counts"], "n_stars": below_hess["n_stars"] + above_hess["n_stars"],
        "area_deg2": float(sum(hess["area_deg2"] for hess in with_area)), "n_off_regions": len(with_area),
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
        (for one off diagram: on counts minus off counts scaled by the ratio
        of the areas; also right for ``mean_off_hess``), and ``n_stars``
        their sum: the excess stars of the on region (float).

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
    excess = (on_hess["density"] - off_hess["density"]) * on_hess["area_deg2"]
    return {
        **{key: on_hess[key] for key in ("color_edges", "mag_edges", "area_deg2", "color_label", "mag_label")},
        "density": on_hess["density"] - off_hess["density"], "error": np.hypot(on_hess["error"], off_hess["error"]),
        "counts": excess, "n_stars": float(excess.sum()),
    }


def _band_name(column: str) -> str:
    """Band of a magnitude column: "euclid_dr1_VIS_obs" -> "VIS", "euclid_dr1_VIS_TO_H_obs" -> "VIS"."""
    return column.removesuffix("_obs").split("_TO_")[0].rsplit("_", 1)[-1]

"""Colour-magnitude match filters."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from streamobs.match_filter import build_match_filter

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray


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


def select_stars_in_match_filter(
    catalog: pd.DataFrame,
    distance_modulus: float,
    match_filter: Mapping[str, Any],
    isochrone: Mapping[str, float],
    *,
    cuts: Mapping[str, Any] | None = None,
    flag_cuts: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """Stars inside the colour-magnitude match filter at one distance modulus.

    A star is selected when it passes the cuts and is inside the polygon of
    every colour of ``match_filter``, built for one isochrone. The cuts run
    first: the polygon test, the slow part, only runs on the stars left.
    Magnitudes are used as they are: the Euclid x DES magnitudes are already
    dereddened.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars, with the ``<namespace>_<band>_obs`` magnitude columns of
        ``catalogs.euclid_des_catalog``.
    distance_modulus : float
        Distance modulus of the filter.
    match_filter : mapping
        ``{"name", "reference_band": (namespace, band), "color_bands":
        [(namespace, band), ...]}``, with optional ``color_cut`` and
        ``red_cap_value``: the format of
        ``injected_stream_snr.DEFAULT_MATCH_FILTERS``.
    isochrone : mapping
        One isochrone, ``{"age": age (Gyr), "metallicity": Z}`` (see
        ``catalogs.isochrone_parameters``).
    cuts : mapping, optional
        Cuts of ``catalogs.catalog_cuts.get_cut_mask``.
    flag_cuts : numpy.ndarray of bool, optional
        Precomputed mask of ``cuts``, as in ``generic_search_single_dm``
        (computed once for a scan over distance moduli).

    Returns
    -------
    selected : numpy.ndarray of bool, shape (len(catalog),)
        True for the stars in the filter.

    Notes
    -----
    Code to reuse: ``build_match_filter_polygon`` and
    ``streamobs.match_filter.is_in_match_filter``, as in
    ``injected_stream_snr._count_selected`` (which counts the stars instead).
    Alternative for DES g - r: celestial_currents
    ``search/matched_filter.py::get_filter_splines_from_color_mag`` and
    ``filter_data``.

    References
    ----------
    Rockosi et al. 2002, AJ 124, 349. Grillmair 2009, ApJ 693, 1118.
    Shipp et al. 2018, ApJ 862, 114.
    """
    raise NotImplementedError

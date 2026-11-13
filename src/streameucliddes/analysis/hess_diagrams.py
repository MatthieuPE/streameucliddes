"""Hess diagrams: colour-magnitude densities of the stars of a region, per deg².

A Hess diagram is a 2D histogram of colour ``reference_band -
color_bands[color_index]`` and magnitude ``reference_band`` of a match
filter (``analysis.cmd_match_filter.match_filter_columns``), divided by the
area of the region: stars / deg² / bin. The on - off difference
(``difference_hess``) brings out the population of a stream over the field.

Hess diagrams are plain dicts, with their bin edges, so that on and off can
be compared bin by bin.

References
----------
Shipp et al. 2018, ApJ 862, 114, Sect. 5 (background-subtracted Hess diagrams).

Notes
-----
Bin sizes from celestial_currents ``hess.py`` (colour bins of 0.04 mag).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd
    from numpy.typing import ArrayLike

DEFAULT_COLOR_BIN = 0.04  # mag, celestial_currents hess.py
DEFAULT_MAG_BIN = 1 / 6  # mag
DEFAULT_COLOR_RANGE = (-1.0, 1.5)  # VIS - H, as the CMDs of notebooks/known_stream_search_example.ipynb
DEFAULT_MAG_RANGE = (17.0, 27.0)  # VIS


def hess_diagram(
    stars: pd.DataFrame,
    mask: ArrayLike,
    area_deg2: float,
    *,
    match_filter: dict[str, Any] | None = None,
    color_index: int = 0,
    color_bins: ArrayLike | None = None,
    mag_bins: ArrayLike | None = None,
) -> dict[str, Any]:
    """Hess diagram of the stars of a region, in stars / deg² / bin.

    Parameters
    ----------
    stars : pandas.DataFrame
        Stars with the magnitude columns of the match filter, e.g.
        ``search.stars`` of ``analysis.stream_frame_search.prepare_search``.
        Not copied: ``mask`` selects the region.
    mask : array_like of bool
        Stars of the region, e.g. ``analysis.stream_regions.on_stream_mask``.
    area_deg2 : float
        Area of the region (deg²), e.g. from ``region_area_deg2``.
    match_filter : dict, optional
        Gives the bands (see ``analysis.cmd_match_filter``). Default:
        ``cmd_match_filter.DEFAULT_MATCH_FILTER``.
    color_index : int, optional
        Colour band of the match filter: ``match_filter["color_bands"][color_index]``.
    color_bins, mag_bins : array_like, optional
        Bin edges (mag). Default: ``DEFAULT_COLOR_RANGE`` and
        ``DEFAULT_MAG_RANGE`` every ``DEFAULT_COLOR_BIN`` and
        ``DEFAULT_MAG_BIN``. Give the same bins to the on and off diagrams.

    Returns
    -------
    hess : dict
        ``{"density", "counts", "error", "color_edges", "mag_edges",
        "area_deg2", "n_stars", "color_label", "mag_label"}``: ``density``
        (stars / deg² / bin) and ``counts`` of shape (n_color, n_mag),
        ``error`` = ``sqrt(counts) / area_deg2``; ``n_stars`` the stars of the
        region inside the bins.

    Raises
    ------
    ValueError
        ``area_deg2 <= 0``.
    """
    raise NotImplementedError


def difference_hess(on_hess: dict[str, Any], off_hess: dict[str, Any]) -> dict[str, Any]:
    """On - off Hess diagram.

    Parameters
    ----------
    on_hess, off_hess : dict
        From ``hess_diagram``, with the same bin edges.

    Returns
    -------
    hess : dict
        The keys of ``hess_diagram``: ``density`` on - off, ``error`` the two
        errors in quadrature, ``counts`` and ``n_stars`` on - off,
        ``area_deg2`` that of the on region.

    Raises
    ------
    ValueError
        The bin edges differ.
    """
    raise NotImplementedError

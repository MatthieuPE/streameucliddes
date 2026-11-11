"""Preparation of the star catalogs for the searches, before the match filter.

Columns checked, magnitude columns resolved (dust-corrected or not), cuts,
stream-frame coordinates and window. The selection itself is in
``analysis.cmd_match_filter``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from streameucliddes.analysis.cmd_match_filter import DEFAULT_MATCH_FILTER, match_filter_columns
from streameucliddes.catalogs import dust_correction
from streameucliddes.catalogs.catalog_cuts import get_cut_mask
from streameucliddes.stream_frames import radec_to_phi12

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any

    import astropy.coordinates as coord
    import pandas as pd

# Default cut on the errors of the match-filter bands (as the SNR forecasts of injected_stream_snr)
DEFAULT_ERR_CUT = (None, 0.2)


# ===================================================
# Magnitude columns and dust correction (generic search)
# ===================================================


def resolve_mag_columns(data, mag_cols, use_dust_corrected):
    """Swap in '<col>_corrected' columns when available and requested."""
    resolved = []
    for col in mag_cols:
        corrected = f"{col}_corrected"
        if use_dust_corrected and corrected in data.columns:
            resolved.append(corrected)
        else:
            if use_dust_corrected:
                print(
                    f"Warning: requested dust-corrected column '{corrected}' not found in data; using '{col}' instead."
                )
            resolved.append(col)
    return resolved


def ensure_dust_corrected(
    data, mag_cols, ra_col, dec_col, dust_correction_kwargs=None, verbose=False
):
    """Compute missing '<col>_corrected' columns in place via dust_correction.correct_dust_map."""
    missing = [col for col in mag_cols if f"{col}_corrected" not in data.columns]
    if not missing:
        return

    if verbose:
        print(f"Applying dust correction for: {missing}")
    dust_correction.correct_dust_map(
        data,
        mag_cols=list(mag_cols),
        pos_columns=[ra_col, dec_col],
        **dict(dust_correction_kwargs or {}),
    )


# ===================================================
# Match-filter catalog
# ===================================================


def default_error_cuts(match_filter: Mapping[str, Any] | None = None) -> dict[str, tuple]:
    """Default cuts of a match filter: errors <= 0.2 on each of its bands.

    Parameters
    ----------
    match_filter : mapping, optional
        See ``cmd_match_filter``. Default: ``cmd_match_filter.DEFAULT_MATCH_FILTER``.

    Returns
    -------
    cuts : dict
        ``{"<namespace>_<band>_err": DEFAULT_ERR_CUT}`` for the reference band
        and each colour band (format of ``catalogs.catalog_cuts.get_cut_mask``).
    """
    match_filter = DEFAULT_MATCH_FILTER if match_filter is None else match_filter
    bands = [match_filter["reference_band"], *match_filter["color_bands"]]
    return {f"{namespace}_{band}_err": DEFAULT_ERR_CUT for namespace, band in bands}


def check_columns(catalog: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a KeyError naming every column missing from the catalog.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars.
    columns : iterable of str
        Columns needed.

    Raises
    ------
    KeyError
        Some columns are missing; the message lists them all.
    """
    missing = [column for column in dict.fromkeys(columns) if column not in catalog.columns]
    if missing:
        raise KeyError(f"columns missing from the catalog: {missing}")


def prepare_match_filter_catalog(
    catalog: pd.DataFrame,
    match_filter: Mapping[str, Any] | None = None,
    *,
    cuts: Mapping[str, Any] | None = None,
    stream_frame: coord.BaseCoordinateFrame | None = None,
    window: tuple[float, float, float, float] | None = None,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> pd.DataFrame:
    """Stars ready for a match filter: columns checked, cuts applied, phi1/phi2.

    Done once for a scan, then reused for every distance modulus and
    isochrone. Keeps only the columns needed: the magnitudes of the match
    filter, the coordinates when a stream frame is given, and the cut
    columns.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars, e.g. ``catalogs.euclid_des_catalog.get_eucliddes_stars()``.
    match_filter : mapping, optional
        See ``cmd_match_filter``. Default: ``cmd_match_filter.DEFAULT_MATCH_FILTER``.
    cuts : mapping, optional
        Format of ``catalogs.catalog_cuts.get_cut_mask``. Default:
        ``default_error_cuts(match_filter)``; ``{}`` for no cut.
    stream_frame : astropy.coordinates.BaseCoordinateFrame, optional
        Adds the ``phi1`` and ``phi2`` columns (deg) in this frame
        (``stream_frames.radec_to_phi12``).
    window : tuple of float, optional
        ``(phi1_min, phi1_max, phi2_min, phi2_max)`` (deg, inclusive): keeps
        only the stars inside. Needs ``stream_frame``.
    ra_col, dec_col : str, optional
        Coordinate columns (deg), read when ``stream_frame`` is given.

    Returns
    -------
    stars : pandas.DataFrame
        The stars passing the cuts (and inside the window), with the index of
        ``catalog``.

    Raises
    ------
    KeyError
        Columns missing from the catalog (all of them listed).
    ValueError
        ``window`` without ``stream_frame``.
    """
    if window is not None and stream_frame is None:
        raise ValueError("a window needs a stream_frame")
    match_filter = DEFAULT_MATCH_FILTER if match_filter is None else match_filter
    cuts = default_error_cuts(match_filter) if cuts is None else cuts

    columns = match_filter_columns(match_filter)
    if stream_frame is not None:
        columns += [ra_col, dec_col]
    check_columns(catalog, [*columns, *cuts])

    stars = catalog.loc[get_cut_mask(catalog, cuts), list(dict.fromkeys([*columns, *cuts]))]
    if stream_frame is not None:
        phi1, phi2 = radec_to_phi12(stars[ra_col].to_numpy(), stars[dec_col].to_numpy(), stream_frame)
        stars = stars.assign(phi1=phi1, phi2=phi2)
        if window is not None:
            phi1_min, phi1_max, phi2_min, phi2_max = window
            inside = (phi1 >= phi1_min) & (phi1 <= phi1_max) & (phi2 >= phi2_min) & (phi2 <= phi2_max)
            stars = stars.loc[inside]
    return stars

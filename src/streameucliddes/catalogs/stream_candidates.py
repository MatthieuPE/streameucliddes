"""List of the stream candidates to investigate, in a CSV file.

One row per candidate: an integer ``id``, the (ra, dec) of two points along
the stream (``endpoint_1_ra``, ``endpoint_1_dec``, ``endpoint_2_ra``,
``endpoint_2_dec``, deg) and the distance modulus where it shows best by eye
on the maps (``best_dm_by_eye``): the inputs of
``analysis.candidate_investigation.CandidateInvestigation``. ``best_dm``, the
distance modulus of its largest SNR, is computed by
``analysis.candidate_investigation.update_candidate_best_dm`` (empty until
then); ``candidate_best_dm`` gives it, else the one by eye.
The optional column ``status`` (``STATUS_COLUMN``, added by hand) rates
each candidate ``strong``, ``good`` or ``bad`` (``CANDIDATE_STATUSES``);
``load_candidates`` leaves the bad ones out unless
``include_bad_candidates`` (so do the masks of the other candidates and the
plots, which read the list through it), empty: not bad.
The file is in ``data/candidates/`` (local, gitignored with the rest of
``data/``). Columns added by hand are kept when a candidate is added.

``load_candidates`` reads the list, ``get_candidate`` gives one by its id,
``add_candidate`` adds one and writes the file (``CandidateInvestigation.from_candidate_id``
starts the investigation of a listed one); ``plotting.plot_candidate_investigation.plot_candidates`` draws them on a
sky map, and ``EndpointPicker`` gives the endpoints from clicks on the map.
``write_candidate_width`` writes the width fit of a candidate in the fits
file (``CANDIDATE_FITS_PATH``, ``candidate_fits.csv``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd


if TYPE_CHECKING:
    from collections.abc import Sequence

# Next to the catalogs (see euclid_catalog.DATA_DIR): the data/ folder of the repository, gitignored.
CANDIDATES_DIR = Path(__file__).resolve().parents[3] / "data" / "candidates"
DEFAULT_CANDIDATES_PATH = CANDIDATES_DIR / "candidates_info.csv"
CANDIDATE_COLUMNS = ("id", "endpoint_1_ra", "endpoint_1_dec", "endpoint_2_ra", "endpoint_2_dec", "best_dm_by_eye")
# Distance modulus of the largest SNR of a candidate, computed (update_candidate_best_dm): NaN until then
BEST_DM_COLUMN = "best_dm"
# Optional column, added by hand: Gaussian sigma in phi2 of a candidate (deg), the width of its mask when another
# candidate is investigated (analysis.build_masks.candidate_mask_tracks)
CANDIDATE_WIDTH_COLUMN = "sigma_deg"
# Optional column, added by hand: how good a candidate looks, one of CANDIDATE_STATUSES (empty: not rated). The bad
# ones are left out of the list (load_candidates), of the masks and of the plots unless include_bad_candidates
STATUS_COLUMN = "status"
CANDIDATE_STATUSES = ("strong", "good", "bad")
BAD_STATUS = "bad"
# Fitted values of the candidates (mu, age, z, N with their errors, the width): one row per "candidate" id
CANDIDATE_FITS_PATH = CANDIDATES_DIR / "candidate_fits.csv"
# Width fit (CandidateInvestigation.fit_width) written to the fits file by write_candidate_width: sigma and its error
WIDTH_FIT_COLUMNS = ("sigma_deg", "sigma_deg_err")


def load_candidates(
    path: str | Path = DEFAULT_CANDIDATES_PATH,
    *,
    include_bad_candidates: bool = False,
) -> pd.DataFrame:
    """The candidates of the CSV file, without the bad ones unless ``include_bad_candidates``.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        CSV file. Missing or empty: no candidate.
    include_bad_candidates : bool, optional
        Keep the candidates whose ``STATUS_COLUMN`` is ``BAD_STATUS``
        (always kept without the column).

    Returns
    -------
    candidates : pandas.DataFrame
        ``CANDIDATE_COLUMNS``, ``BEST_DM_COLUMN`` (NaN where not computed,
        all NaN when the file has no such column), then any column added by
        hand; ``id`` as integers. Empty, with these columns, when there is
        no candidate. The index is that of the rows of the file.

    Raises
    ------
    ValueError
        A column of ``CANDIDATE_COLUMNS`` missing from the file (a file of
        the former format, ``best_dm`` by eye and no ``best_dm_by_eye``: a
        message to rename it), or a ``STATUS_COLUMN`` value not in
        ``CANDIDATE_STATUSES``.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return _empty_candidates()
    candidates = pd.read_csv(path)
    missing = [column for column in CANDIDATE_COLUMNS if column not in candidates.columns]
    if missing == ["best_dm_by_eye"] and BEST_DM_COLUMN in candidates.columns:
        raise ValueError(f"{path} has a column {BEST_DM_COLUMN!r} and no 'best_dm_by_eye': the distance modulus by "
                         f"eye is now 'best_dm_by_eye' (rename the column), {BEST_DM_COLUMN!r} the one of the "
                         "largest SNR (update_candidate_best_dm).")
    if missing:
        raise ValueError(f"{path} has no column {missing}; the columns are {CANDIDATE_COLUMNS}.")
    candidates["id"] = candidates["id"].astype(int)
    if BEST_DM_COLUMN not in candidates.columns:
        candidates[BEST_DM_COLUMN] = np.nan
    if STATUS_COLUMN in candidates.columns:
        status = candidates[STATUS_COLUMN]
        unknown = sorted(set(status.dropna()) - set(CANDIDATE_STATUSES))
        if unknown:
            raise ValueError(f"{path}: {STATUS_COLUMN!r} {unknown} of no known status; the statuses are "
                             f"{CANDIDATE_STATUSES} (or empty).")
        if not include_bad_candidates:
            candidates = candidates[status != BAD_STATUS]
    return _ordered(candidates)


def add_candidate(
    endpoint_1: Sequence[float],
    endpoint_2: Sequence[float],
    best_dm_by_eye: float,
    *,
    candidate_id: int | None = None,
    path: str | Path = DEFAULT_CANDIDATES_PATH,
) -> pd.DataFrame:
    """Add a candidate to the CSV file (created if needed) and return the list.

    Parameters
    ----------
    endpoint_1, endpoint_2 : sequence of float
        (ra, dec) of two points along the stream (deg), e.g.
        ``EndpointPicker.endpoints``. ra is written in [0, 360).
    best_dm_by_eye : float
        Distance modulus where the candidate shows best, by eye on the maps.
        ``BEST_DM_COLUMN`` is left empty (``update_candidate_best_dm``).
    candidate_id : int, optional
        Default: one more than the largest ``id`` of the file (1 for the
        first candidate, bad ones included).
    path : str or pathlib.Path, optional
        CSV file.

    Returns
    -------
    candidates : pandas.DataFrame
        The list with the new candidate, as written (the bad candidates
        included; the new one without a status).

    Raises
    ------
    ValueError
        ``candidate_id`` already in the file, a dec outside [-90, 90], or
        the two endpoints at the same position.
    """
    (ra_1, dec_1), (ra_2, dec_2) = (_radec(endpoint) for endpoint in (endpoint_1, endpoint_2))
    if np.isclose(ra_1, ra_2) and np.isclose(dec_1, dec_2):
        raise ValueError(f"The two endpoints are at the same position ({ra_1:g}, {dec_1:g}) deg.")
    path = Path(path)
    candidates = load_candidates(path, include_bad_candidates=True)  # written back: the bad ones kept
    if candidate_id is None:
        candidate_id = int(candidates["id"].max()) + 1 if len(candidates) else 1
    elif int(candidate_id) in set(candidates["id"]):
        raise ValueError(f"A candidate with id {candidate_id} is already in {path}.")
    row = pd.DataFrame([{
        "id": int(candidate_id), "endpoint_1_ra": ra_1, "endpoint_1_dec": dec_1, "endpoint_2_ra": ra_2,
        "endpoint_2_dec": dec_2, "best_dm_by_eye": float(best_dm_by_eye), BEST_DM_COLUMN: np.nan,
    }])
    candidates = row if candidates.empty else pd.concat([candidates, row], ignore_index=True)
    return write_candidates(candidates, path)


def write_candidates(candidates: pd.DataFrame, path: str | Path = DEFAULT_CANDIDATES_PATH) -> pd.DataFrame:
    """Write the list of candidates (``CANDIDATE_COLUMNS``, ``BEST_DM_COLUMN``, then the others) and return it.

    Parameters
    ----------
    candidates : pandas.DataFrame
        E.g. ``load_candidates`` with values changed.
    path : str or pathlib.Path, optional
        CSV file (its folder created if needed).

    Returns
    -------
    candidates : pandas.DataFrame
        As written.
    """
    candidates = _ordered(candidates)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(path, index=False)
    return candidates


def write_candidate_width(
    candidate_id: int,
    width_fit: dict,
    path: str | Path = CANDIDATE_FITS_PATH,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Write the width of a candidate (``WIDTH_FIT_COLUMNS``) in its row of the fits file and return the file.

    The other columns and rows are kept; the columns are added if needed,
    and a row for a candidate not yet in the file (its other values empty).
    A width already in the file is kept (nothing written) unless ``force``.

    Parameters
    ----------
    candidate_id : int
        Its ``candidate`` id in the file.
    width_fit : dict
        ``CandidateInvestigation.width_fit`` (``fit_stream_width``):
        ``sigma`` and ``errors["sigma"]``, deg.
    path : str or pathlib.Path, optional
        CSV file of the fits, created if needed.
    force : bool, optional
        Replace a width already in the file.

    Returns
    -------
    fits : pandas.DataFrame
        The file as written (as it was when its width is kept).
    """
    path = Path(path)
    fits = pd.read_csv(path) if path.exists() else pd.DataFrame({"candidate": pd.Series(dtype=int)})
    rows = fits["candidate"] == int(candidate_id)
    if not rows.any():
        fits = pd.concat([fits, pd.DataFrame({"candidate": [int(candidate_id)]})], ignore_index=True)
        rows = fits["candidate"] == int(candidate_id)
    sigma_column, error_column = WIDTH_FIT_COLUMNS
    if not force and sigma_column in fits and fits.loc[rows, sigma_column].notna().any():
        return fits
    fits.loc[rows, sigma_column] = float(width_fit["sigma"])
    fits.loc[rows, error_column] = float(width_fit["errors"]["sigma"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.to_csv(path, index=False)
    return fits


def candidate_best_dm(candidate: pd.Series) -> float:
    """Distance modulus of a candidate (a row of ``load_candidates``): ``BEST_DM_COLUMN`` when computed, else by eye."""
    best_dm = candidate.get(BEST_DM_COLUMN, np.nan)
    return float(candidate["best_dm_by_eye"] if pd.isna(best_dm) else best_dm)


def get_candidate(candidate_id: int, path: str | Path = DEFAULT_CANDIDATES_PATH) -> pd.Series:
    """The candidate of an id, a row of ``load_candidates``, a bad one too (asked for by its id).

    Parameters
    ----------
    candidate_id : int
        Its ``id`` in the file.
    path : str or pathlib.Path, optional
        CSV file.

    Returns
    -------
    candidate : pandas.Series
        Its columns: ``CANDIDATE_COLUMNS``, ``BEST_DM_COLUMN`` and any added by hand.

    Raises
    ------
    ValueError
        No candidate with this id (the ids of the file are listed).
    """
    candidates = load_candidates(path, include_bad_candidates=True)
    rows = candidates[candidates["id"] == int(candidate_id)]
    if rows.empty:
        raise ValueError(f"No candidate with id {candidate_id} in {path}; the ids are {candidates['id'].tolist()}.")
    return rows.iloc[0]


def candidate_endpoints(candidate: pd.Series) -> tuple[tuple[float, float], tuple[float, float]]:
    """(ra, dec) of the two endpoints of a candidate (a row of ``load_candidates``), in deg."""
    return ((float(candidate["endpoint_1_ra"]), float(candidate["endpoint_1_dec"])),
            (float(candidate["endpoint_2_ra"]), float(candidate["endpoint_2_dec"])))


def _radec(endpoint: Sequence[float]) -> tuple[float, float]:
    """(ra in [0, 360), dec) of an endpoint; ValueError for a dec outside [-90, 90]."""
    ra, dec = (float(value) for value in endpoint)
    if not -90.0 <= dec <= 90.0:
        raise ValueError(f"dec must be in [-90, 90] deg, not {dec}.")
    return ra % 360.0, dec


def _ordered(candidates: pd.DataFrame) -> pd.DataFrame:
    """The columns in the order of the file: ``CANDIDATE_COLUMNS``, ``BEST_DM_COLUMN``, then the others."""
    first = [*CANDIDATE_COLUMNS, BEST_DM_COLUMN]
    return candidates[[*first, *(column for column in candidates.columns if column not in first)]]


def _empty_candidates() -> pd.DataFrame:
    """No candidate, with the columns of the file."""
    return pd.DataFrame({column: pd.Series(dtype=int if column == "id" else float)
                         for column in (*CANDIDATE_COLUMNS, BEST_DM_COLUMN)})

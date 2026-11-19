"""List of the stream candidates to investigate, in a CSV file.

One row per candidate: an integer ``id``, the (ra, dec) of two points along
the stream (``endpoint_1_ra``, ``endpoint_1_dec``, ``endpoint_2_ra``,
``endpoint_2_dec``, deg) and the distance modulus where it shows best
(``best_dm``): the inputs of ``analysis.candidate_investigation.CandidateInvestigation``.
The file is in ``data/candidates/`` (local, gitignored with the rest of
``data/``). Columns added by hand are kept when a candidate is added.

``load_candidates`` reads the list, ``get_candidate`` gives one by its id,
``add_candidate`` adds one and writes the file (``CandidateInvestigation.from_candidate_id``
starts the investigation of a listed one); ``plotting.plot_candidate_investigation.plot_candidates`` draws them on a
sky map, and ``EndpointPicker`` gives the endpoints from clicks on the map.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from streameucliddes.stream_frames import stream_frame_from_endpoints

if TYPE_CHECKING:
    from collections.abc import Sequence

# Next to the catalogs (see euclid_catalog.DATA_DIR): the data/ folder of the repository, gitignored.
CANDIDATES_DIR = Path(__file__).resolve().parents[3] / "data" / "candidates"
DEFAULT_CANDIDATES_PATH = CANDIDATES_DIR / "candidates_info.csv"
CANDIDATE_COLUMNS = ("id", "endpoint_1_ra", "endpoint_1_dec", "endpoint_2_ra", "endpoint_2_dec", "best_dm")
# Optional column, added by hand: Gaussian sigma in phi2 of a candidate (deg), the width of its mask when another
# candidate is investigated (candidate_mask_tracks)
CANDIDATE_WIDTH_COLUMN = "sigma_deg"
# Half-width of the mask of a candidate without a width (deg): narrower than for the known streams of unknown width
# (analysis.build_masks.DEFAULT_STREAM_MASK_HALF_WIDTH_DEG), the candidates being thin (user choice, 2026-11-19)
CANDIDATE_MASK_HALF_WIDTH_DEG = 0.3


def load_candidates(path: str | Path = DEFAULT_CANDIDATES_PATH) -> pd.DataFrame:
    """The candidates of the CSV file.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        CSV file. Missing or empty: no candidate.

    Returns
    -------
    candidates : pandas.DataFrame
        ``CANDIDATE_COLUMNS`` first, then any column added by hand; ``id``
        as integers. Empty, with ``CANDIDATE_COLUMNS``, when there is no
        candidate.

    Raises
    ------
    ValueError
        A column of ``CANDIDATE_COLUMNS`` missing from the file.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return _empty_candidates()
    candidates = pd.read_csv(path)
    missing = [column for column in CANDIDATE_COLUMNS if column not in candidates.columns]
    if missing:
        raise ValueError(f"{path} has no column {missing}; the columns are {CANDIDATE_COLUMNS}.")
    candidates["id"] = candidates["id"].astype(int)
    extra = [column for column in candidates.columns if column not in CANDIDATE_COLUMNS]
    return candidates[[*CANDIDATE_COLUMNS, *extra]]


def add_candidate(
    endpoint_1: Sequence[float],
    endpoint_2: Sequence[float],
    best_dm: float,
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
    best_dm : float
        Distance modulus where the candidate shows best.
    candidate_id : int, optional
        Default: one more than the largest ``id`` of the file (1 for the
        first candidate).
    path : str or pathlib.Path, optional
        CSV file.

    Returns
    -------
    candidates : pandas.DataFrame
        The list with the new candidate, as written.

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
    candidates = load_candidates(path)
    if candidate_id is None:
        candidate_id = int(candidates["id"].max()) + 1 if len(candidates) else 1
    elif int(candidate_id) in set(candidates["id"]):
        raise ValueError(f"A candidate with id {candidate_id} is already in {path}.")
    row = pd.DataFrame([{
        "id": int(candidate_id), "endpoint_1_ra": ra_1, "endpoint_1_dec": dec_1, "endpoint_2_ra": ra_2,
        "endpoint_2_dec": dec_2, "best_dm": float(best_dm),
    }])
    candidates = row if candidates.empty else pd.concat([candidates, row], ignore_index=True)
    candidates = candidates[[*CANDIDATE_COLUMNS, *(c for c in candidates.columns if c not in CANDIDATE_COLUMNS)]]
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(path, index=False)
    return candidates


def get_candidate(candidate_id: int, path: str | Path = DEFAULT_CANDIDATES_PATH) -> pd.Series:
    """The candidate of an id, a row of ``load_candidates``.

    Parameters
    ----------
    candidate_id : int
        Its ``id`` in the file.
    path : str or pathlib.Path, optional
        CSV file.

    Returns
    -------
    candidate : pandas.Series
        Its columns: ``CANDIDATE_COLUMNS`` and any added by hand.

    Raises
    ------
    ValueError
        No candidate with this id (the ids of the file are listed).
    """
    candidates = load_candidates(path)
    rows = candidates[candidates["id"] == int(candidate_id)]
    if rows.empty:
        raise ValueError(f"No candidate with id {candidate_id} in {path}; the ids are {candidates['id'].tolist()}.")
    return rows.iloc[0]


def candidate_endpoints(candidate: pd.Series) -> tuple[tuple[float, float], tuple[float, float]]:
    """(ra, dec) of the two endpoints of a candidate (a row of ``load_candidates``), in deg."""
    return ((float(candidate["endpoint_1_ra"]), float(candidate["endpoint_1_dec"])),
            (float(candidate["endpoint_2_ra"]), float(candidate["endpoint_2_dec"])))


def candidate_mask_tracks(
    candidates: pd.DataFrame,
    *,
    exclude_ids: Sequence[int] = (),
    default_half_width_deg: float = CANDIDATE_MASK_HALF_WIDTH_DEG,
) -> list[dict]:
    """The listed candidates as tracks to mask (``analysis.build_masks.streams_healpix_mask``), but some.

    Each candidate is the great circle between its endpoints
    (``stream_frames.stream_frame_from_endpoints``), its width the optional
    column ``CANDIDATE_WIDTH_COLUMN`` (its Gaussian sigma in phi2, added by
    hand; NaN when absent or empty: unknown).

    Parameters
    ----------
    candidates : pandas.DataFrame
        From ``load_candidates``.
    exclude_ids : sequence of int, optional
        Candidates left out, e.g. the one investigated.
    default_half_width_deg : float, optional
        Half-width of the mask of a candidate without a width (deg), in each
        track (``default_half_width_deg``, used by ``streams_healpix_mask``).

    Returns
    -------
    tracks : list of dict
        One per candidate kept: ``name`` and ``stream_name``
        (``"candidate_<id>"``), ``ra``, ``dec`` (deg, its two endpoints),
        ``stream_frame``, ``width_deg``, ``dm_min`` and ``dm_max`` (its
        ``best_dm``) and ``default_half_width_deg``.
    """
    excluded = {int(candidate_id) for candidate_id in exclude_ids}
    tracks = []
    for _, candidate in candidates.iterrows():
        if int(candidate["id"]) in excluded:
            continue
        endpoint_1, endpoint_2 = candidate_endpoints(candidate)
        name = f"candidate_{int(candidate['id'])}"
        width = float(candidate[CANDIDATE_WIDTH_COLUMN]) if CANDIDATE_WIDTH_COLUMN in candidate.index else np.nan
        tracks.append({"name": name, "stream_name": name, "ra": np.array([endpoint_1[0], endpoint_2[0]]),
                       "dec": np.array([endpoint_1[1], endpoint_2[1]]),
                       "stream_frame": stream_frame_from_endpoints(endpoint_1, endpoint_2), "width_deg": width,
                       "dm_min": float(candidate["best_dm"]), "dm_max": float(candidate["best_dm"]),
                       "default_half_width_deg": float(default_half_width_deg)})
    return tracks


def _radec(endpoint: Sequence[float]) -> tuple[float, float]:
    """(ra in [0, 360), dec) of an endpoint; ValueError for a dec outside [-90, 90]."""
    ra, dec = (float(value) for value in endpoint)
    if not -90.0 <= dec <= 90.0:
        raise ValueError(f"dec must be in [-90, 90] deg, not {dec}.")
    return ra % 360.0, dec


def _empty_candidates() -> pd.DataFrame:
    """No candidate, with the columns of the file."""
    return pd.DataFrame({column: pd.Series(dtype=int if column == "id" else float) for column in CANDIDATE_COLUMNS})

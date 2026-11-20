"""Tests of catalogs/stream_candidates.py: the CSV list of the stream candidates, in a temporary file."""

import numpy as np
import pandas as pd
import pytest

from streameucliddes.catalogs.stream_candidates import (
    BEST_DM_COLUMN,
    CANDIDATE_COLUMNS,
    STATUS_COLUMN,
    add_candidate,
    candidate_best_dm,
    candidate_endpoints,
    get_candidate,
    load_candidates,
    write_candidates,
)

pytestmark = pytest.mark.stream_candidates


# A missing file and an empty one (as created by hand): no candidate, with the columns of the file
@pytest.mark.parametrize("create", [False, True])
def test_load_candidates_without_file(tmp_path, create):
    path = tmp_path / "candidates.csv"
    if create:
        path.touch()
    candidates = load_candidates(path)
    assert candidates.empty and tuple(candidates.columns) == (*CANDIDATE_COLUMNS, BEST_DM_COLUMN)


# Ids 1, 2, ... in the order added; ra in [0, 360); read back as written; endpoints of a row
def test_add_candidates_and_read_back(tmp_path):
    path = tmp_path / "candidates" / "list.csv"  # folder created
    add_candidate((45.5, -45.0), (44.0, -51.0), 17.0, path=path)
    written = add_candidate((-3.0, -40.5), (70.0, -44.5), 17.5, path=path)
    candidates = load_candidates(path)
    assert list(candidates["id"]) == [1, 2] and candidates["id"].dtype.kind == "i"
    assert candidates.loc[1, "endpoint_1_ra"] == pytest.approx(357.0)
    pd.testing.assert_frame_equal(candidates, written)
    assert candidate_endpoints(candidates.iloc[0]) == ((45.5, -45.0), (44.0, -51.0))
    assert add_candidate((10.0, 0.0), (12.0, 1.0), 16.0, candidate_id=10, path=path)["id"].iloc[-1] == 10
    assert add_candidate((10.0, 0.0), (12.0, 1.0), 16.0, path=path)["id"].iloc[-1] == 11


# A column added by hand is kept, after the columns of the file, empty for the new candidate
def test_add_candidate_keeps_extra_columns(tmp_path):
    path = tmp_path / "candidates.csv"
    add_candidate((45.5, -45.0), (44.0, -51.0), 17.0, path=path)
    table = pd.read_csv(path)
    table["comment"] = ["faint"]
    table.to_csv(path, index=False)
    candidates = add_candidate((10.0, 0.0), (12.0, 1.0), 16.0, path=path)
    assert tuple(candidates.columns) == (*CANDIDATE_COLUMNS, BEST_DM_COLUMN, "comment")
    assert candidates["comment"].iloc[0] == "faint" and pd.isna(candidates["comment"].iloc[1])


# An id already used, a dec outside [-90, 90], the same point twice, a file missing a column: ValueError
def test_add_candidate_errors(tmp_path):
    path = tmp_path / "candidates.csv"
    add_candidate((45.5, -45.0), (44.0, -51.0), 17.0, path=path)
    with pytest.raises(ValueError, match="already"):
        add_candidate((10.0, 0.0), (12.0, 1.0), 16.0, candidate_id=1, path=path)
    with pytest.raises(ValueError, match="dec"):
        add_candidate((10.0, -95.0), (12.0, 1.0), 16.0, path=path)
    with pytest.raises(ValueError, match="same position"):
        add_candidate((10.0, 1.0), (370.0, 1.0), 16.0, path=path)
    pd.read_csv(path).drop(columns="endpoint_2_dec").to_csv(path, index=False)
    with pytest.raises(ValueError, match="endpoint_2_dec"):
        load_candidates(path)


# The former format (best_dm by eye, no best_dm_by_eye): ValueError asking to rename the column; a file without
# best_dm: read with an empty one
def test_load_candidates_columns_of_best_dm(tmp_path):
    path = tmp_path / "candidates.csv"
    add_candidate((45.5, -45.0), (44.0, -51.0), 17.0, path=path)
    table = pd.read_csv(path)
    table.drop(columns=BEST_DM_COLUMN).to_csv(path, index=False)
    assert load_candidates(path)[BEST_DM_COLUMN].isna().all()
    table.drop(columns=BEST_DM_COLUMN).rename(columns={"best_dm_by_eye": "best_dm"}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="rename"):
        load_candidates(path)


# The distance modulus of a candidate: best_dm once computed, else the one by eye; write_candidates keeps a value set
def test_candidate_best_dm(tmp_path):
    path = tmp_path / "candidates.csv"
    add_candidate((45.5, -45.0), (44.0, -51.0), 17.0, path=path)
    candidates = add_candidate((77.0, -40.5), (70.0, -44.5), 17.5, path=path)
    assert candidates[BEST_DM_COLUMN].isna().all()
    assert [candidate_best_dm(row) for _, row in candidates.iterrows()] == [17.0, 17.5]
    candidates.loc[1, BEST_DM_COLUMN] = 17.7
    write_candidates(candidates, path)
    assert [candidate_best_dm(row) for _, row in load_candidates(path).iterrows()] == [17.0, 17.7]
    assert np.isnan(load_candidates(path)[BEST_DM_COLUMN].iloc[0])


# The row of an id; an id not in the file: ValueError listing the ids
def test_get_candidate(tmp_path):
    path = tmp_path / "candidates.csv"
    add_candidate((45.5, -45.0), (44.0, -51.0), 17.0, path=path)
    add_candidate((77.0, -40.5), (70.0, -44.5), 17.5, path=path)
    candidate = get_candidate(2, path)
    assert candidate["id"] == 2 and candidate["best_dm_by_eye"] == 17.5
    assert candidate_endpoints(candidate) == ((77.0, -40.5), (70.0, -44.5))
    with pytest.raises(ValueError, match=r"\[1, 2\]"):
        get_candidate(3, path)


def _list_with_statuses(path, statuses):
    """A list of len(statuses) candidates (ids 1, 2, ...) with these statuses (None: empty)."""
    for index, _ in enumerate(statuses):
        candidates = add_candidate((45.0 + index, -45.0), (44.0 + index, -51.0), 17.0, path=path)
    candidates[STATUS_COLUMN] = statuses
    write_candidates(candidates, path)


# The bad candidates left out unless include_bad_candidates, an empty status kept; a file without the column: all
# kept; an unknown status: ValueError
def test_load_candidates_status(tmp_path):
    path = tmp_path / "candidates.csv"
    _list_with_statuses(path, ["strong", "bad", "good", None])
    assert load_candidates(path)["id"].tolist() == [1, 3, 4]
    assert load_candidates(path, include_bad_candidates=True)["id"].tolist() == [1, 2, 3, 4]
    pd.read_csv(path).drop(columns=STATUS_COLUMN).to_csv(path, index=False)
    assert load_candidates(path)["id"].tolist() == [1, 2, 3, 4]
    _list_with_statuses(tmp_path / "unknown.csv", ["strong", "maybe"])
    with pytest.raises(ValueError, match="maybe"):
        load_candidates(tmp_path / "unknown.csv")


# Writing the file keeps the bad candidates: add_candidate after them (next id), get_candidate finds one by its id
def test_bad_candidates_kept_in_the_file(tmp_path):
    path = tmp_path / "candidates.csv"
    _list_with_statuses(path, ["strong", "bad"])
    candidates = add_candidate((10.0, 0.0), (12.0, 1.0), 16.0, path=path)
    assert candidates["id"].tolist() == [1, 2, 3] and candidates[STATUS_COLUMN].tolist()[:2] == ["strong", "bad"]
    assert pd.isna(candidates[STATUS_COLUMN].iloc[2])
    assert get_candidate(2, path)[STATUS_COLUMN] == "bad"

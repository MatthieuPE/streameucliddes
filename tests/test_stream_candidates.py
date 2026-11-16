"""Tests of catalogs/stream_candidates.py: the CSV list of the stream candidates, in a temporary file."""

import pandas as pd
import pytest

from streameucliddes.catalogs.stream_candidates import (
    CANDIDATE_COLUMNS,
    add_candidate,
    candidate_endpoints,
    load_candidates,
)

pytestmark = pytest.mark.stream_candidates


# A missing file and an empty one (as created by hand): no candidate, with the columns of the file
@pytest.mark.parametrize("create", [False, True])
def test_load_candidates_without_file(tmp_path, create):
    path = tmp_path / "candidates.csv"
    if create:
        path.touch()
    candidates = load_candidates(path)
    assert candidates.empty and tuple(candidates.columns) == CANDIDATE_COLUMNS


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
    assert tuple(candidates.columns) == (*CANDIDATE_COLUMNS, "comment")
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
    pd.read_csv(path).drop(columns="best_dm").to_csv(path, index=False)
    with pytest.raises(ValueError, match="best_dm"):
        load_candidates(path)

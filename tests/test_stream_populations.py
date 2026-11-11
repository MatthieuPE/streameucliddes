"""Tests of catalogs/stream_populations.py."""

import numpy as np
import pytest

from streameucliddes.catalogs.stream_populations import load_stream_population_table

pytestmark = pytest.mark.stream_populations

HEADER = "galstreams_name,age_gyr,feh,age_reference,feh_reference,notes\n"
COLUMNS = ["age_gyr", "feh", "age_reference", "feh_reference", "notes"]


def _write(tmp_path, rows, header=HEADER):
    path = tmp_path / "populations.csv"
    path.write_text(header + "".join(row + "\n" for row in rows))
    return path


# The table shipped with the package loads, with its columns (no values curated yet)
def test_shipped_table():
    table = load_stream_population_table()
    assert table.index.name == "galstreams_name"
    assert list(table.columns) == COLUMNS


# Values as floats, NaN when empty (never invented); references as strings, empty when missing
def test_values_and_missing_values(tmp_path):
    path = _write(tmp_path, [
        "Phoenix,,-2.70,,li2022,",
        "AAU-ATLAS,12.5,-2.22,somebody2020,li2022,a note",
    ])
    table = load_stream_population_table(path)
    assert list(table.index) == ["Phoenix", "AAU-ATLAS"]
    assert np.isnan(table.loc["Phoenix", "age_gyr"])
    assert table.loc["Phoenix", "feh"] == pytest.approx(-2.70)
    assert table.loc["AAU-ATLAS", "age_gyr"] == pytest.approx(12.5)
    assert table["age_gyr"].dtype == table["feh"].dtype == float
    assert table.loc["Phoenix", "age_reference"] == ""
    assert table.loc["AAU-ATLAS", "notes"] == "a note"


# Errors: other columns, a stream twice, a value that is not a number
@pytest.mark.parametrize(
    "header, rows, message",
    [
        ("galstreams_name,age,feh\n", ["Phoenix,,-2.7"], "columns"),
        (HEADER, ["Phoenix,,-2.7,,li2022,", "Phoenix,,-2.6,,other,"], "more than once"),
        (HEADER, ["Phoenix,old,-2.7,,li2022,"], "old"),
    ],
)
def test_invalid_tables(tmp_path, header, rows, message):
    with pytest.raises(ValueError, match=message):
        load_stream_population_table(_write(tmp_path, rows, header))

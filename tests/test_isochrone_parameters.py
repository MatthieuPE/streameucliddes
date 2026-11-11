"""Tests of catalogs/isochrone_parameters.py."""

import numpy as np
import pytest

from streameucliddes.catalogs.isochrone_parameters import (
    default_isochrone_grid,
    known_stream_isochrone_parameters,
    load_stream_population_table,
)

pytestmark = pytest.mark.isochrone_parameters

COLUMNS = [
    "age_gyr", "age_err_plus", "age_err_minus", "feh", "feh_err_plus", "feh_err_minus", "z",
    "age_reference", "feh_reference", "z_reference", "notes",
]
HEADER = ",".join(["galstreams_name", *COLUMNS]) + "\n"


def _write(tmp_path, rows, header=HEADER):
    path = tmp_path / "populations.csv"
    path.write_text(header + "".join(row + "\n" for row in rows))
    return path


# The table shipped with the package loads, with its columns
def test_shipped_table():
    table = load_stream_population_table()
    assert table.index.name == "galstreams_name"
    assert list(table.columns) == COLUMNS


# Shipped table: every stream has a Z, and every value its reference
def test_shipped_table_references():
    table = load_stream_population_table()
    assert table["z"].notna().all() and (table["z_reference"] != "").all()
    assert (table.loc[table["age_gyr"].notna(), "age_reference"] != "").all()
    assert (table.loc[table["feh"].notna(), "feh_reference"] != "").all()
    assert table.loc["Ravi", "age_gyr"] == pytest.approx(13.5)  # Shipp et al. 2018, no [Fe/H]
    assert table.loc["Ravi", "z"] == pytest.approx(0.0002699)  # Shipp Z = 0.0003 (Dotter), on the Marigo scale
    assert np.isnan(table.loc["AAU-ATLAS", "age_gyr"])  # spectroscopic Z, default ages


# Values as floats, NaN when empty (never invented); references as strings, empty when missing
def test_values_and_missing_values(tmp_path):
    path = _write(tmp_path, [
        "Phoenix,,,,-2.70,0.05,0.05,0.0001,,li2022,converted,",
        "AAU-ATLAS,12.5,1.0,0.5,-2.22,0.02,0.02,0.0001,somebody2020,li2022,converted,a note",
    ])
    table = load_stream_population_table(path)
    assert list(table.index) == ["Phoenix", "AAU-ATLAS"]
    assert np.isnan(table.loc["Phoenix", "age_gyr"])
    assert table.loc["Phoenix", "feh"] == pytest.approx(-2.70)
    assert table.loc["AAU-ATLAS", "age_gyr"] == pytest.approx(12.5)
    assert table.loc["AAU-ATLAS", "age_err_minus"] == pytest.approx(0.5)
    assert np.isnan(table.loc["Phoenix", "age_err_plus"])
    assert (table[COLUMNS[:7]].dtypes == float).all()
    assert table.loc["Phoenix", "age_reference"] == ""
    assert table.loc["AAU-ATLAS", "notes"] == "a note"


# Errors: other columns, a stream twice, a value that is not a number
@pytest.mark.parametrize(
    "header, rows, message",
    [
        ("galstreams_name,age,feh\n", ["Phoenix,,-2.7"], "columns"),
        (HEADER, ["Phoenix,,,,-2.7,,,,,li2022,,", "Phoenix,,,,-2.6,,,,,other,,"], "more than once"),
        (HEADER, ["Phoenix,old,,,-2.7,,,,,li2022,,"], "old"),
    ],
)
def test_invalid_tables(tmp_path, header, rows, message):
    with pytest.raises(ValueError, match=message):
        load_stream_population_table(_write(tmp_path, rows, header))


# ===================================================
# Test default_isochrone_grid
# ===================================================


# Default: the DES search isochrone of Shipp et al. 2018, 13 Gyr and Z = 0.0002
def test_default_isochrone_grid():
    assert default_isochrone_grid() == [{"age": 13.0, "metallicity": 0.0002}]


# One isochrone per (age, Z) pair, every Z for the first age, then for the next one
def test_default_isochrone_grid_pairs():
    grid = default_isochrone_grid(ages_gyr=[11.0, 12], metallicities=np.array([0.0001, 0.0004]))
    assert grid == [
        {"age": 11.0, "metallicity": 0.0001}, {"age": 11.0, "metallicity": 0.0004},
        {"age": 12.0, "metallicity": 0.0001}, {"age": 12.0, "metallicity": 0.0004},
    ]
    assert all(type(value) is float for isochrone in grid for value in isochrone.values())
    assert default_isochrone_grid(ages_gyr=[12.0]) == [{"age": 12.0, "metallicity": 0.0002}]


# ===================================================
# Test known_stream_isochrone_parameters
# ===================================================


@pytest.fixture
def population_table(tmp_path):
    return load_stream_population_table(_write(tmp_path, [
        "Both,12.5,,,-2.2,0.1,0.1,0.0001,somebody,somebody,converted,",
        "NoAge,,,,-2.2,0.1,0.1,0.0001,,somebody,converted,",
        "NoZ,12.5,,,,,,,somebody,,,",
    ]))


GRID = [
    {"age": 11.0, "metallicity": 0.0001}, {"age": 11.0, "metallicity": 0.0004},
    {"age": 12.0, "metallicity": 0.0001}, {"age": 12.0, "metallicity": 0.0004},
]


# Age and Z known: one isochrone. Age unknown: each age of the grid, with the known Z.
# Z unknown: each Z of the grid, with the known age. Stream not in the table: the grid
@pytest.mark.parametrize(
    "stream_name, expected",
    [
        ("Both", [{"age": 12.5, "metallicity": 0.0001}]),
        ("NoAge", [{"age": 11.0, "metallicity": 0.0001}, {"age": 12.0, "metallicity": 0.0001}]),
        ("NoZ", [{"age": 12.5, "metallicity": 0.0001}, {"age": 12.5, "metallicity": 0.0004}]),
        ("Unknown", GRID),
    ],
)
def test_known_stream_isochrone_parameters(population_table, stream_name, expected):
    assert known_stream_isochrone_parameters(stream_name, population_table, GRID) == expected


# Defaults: the shipped table and the default grid
def test_known_stream_isochrone_parameters_defaults():
    table = load_stream_population_table()
    ravi = known_stream_isochrone_parameters("Ravi")  # Shipp et al. 2018 age and Z
    assert ravi == [{"age": 13.5, "metallicity": table.loc["Ravi", "z"]}]
    atlas = known_stream_isochrone_parameters("AAU-ATLAS")  # spectroscopic Z, default age
    assert atlas == [{"age": 13.0, "metallicity": table.loc["AAU-ATLAS", "z"]}]
    assert known_stream_isochrone_parameters("Unknown") == default_isochrone_grid()

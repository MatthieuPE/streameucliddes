"""Tests of catalogs/isochrone_parameters.py."""

import numpy as np
import pytest

from streameucliddes.catalogs.isochrone_parameters import (
    DEFAULT_AGES_GYR,
    DEFAULT_METALLICITIES,
    Z_SCALE_ISOCHRONE_MODEL,
    default_isochrone_grid,
    isochrone_from_feh,
    known_stream_isochrone_parameters,
    load_stream_population_table,
    with_isochrone_model,
)
from streameucliddes.unit_conversions import convert_FeH_to_z, convert_metallicity

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


# ===================================================
# Test isochrone_from_feh
# ===================================================


# Z from [Fe/H], else the given Z, else the default; the default age when unknown (None or NaN)
def test_isochrone_from_feh():
    assert isochrone_from_feh(11.0, feh=-2.0) == {"age": 11.0, "metallicity": pytest.approx(convert_FeH_to_z(-2.0))}
    assert isochrone_from_feh(metallicity=0.0004) == {"age": DEFAULT_AGES_GYR[0], "metallicity": 0.0004}
    default = {"age": DEFAULT_AGES_GYR[0], "metallicity": DEFAULT_METALLICITIES[0]}
    assert isochrone_from_feh() == default
    assert isochrone_from_feh(np.nan, feh=np.nan, metallicity=np.nan) == default
    with pytest.raises(ValueError, match="not both"):
        isochrone_from_feh(feh=-2.0, metallicity=0.0004)


# ===================================================
# Test the isochrones of an isochrone model
# ===================================================


# Set on the isochrones without their own isochrone model; copies
def test_with_isochrone_model():
    given = [{"age": 13.0, "metallicity": 0.0002},
             {"age": 12.0, "metallicity": 0.0001, "isochrone_model": "Bressan2012"}]
    assert with_isochrone_model(given, "Dotter2016") == [{**given[0], "isochrone_model": "Dotter2016"}, given[1]]
    assert with_isochrone_model(given, None) == given and with_isochrone_model(given, None)[0] is not given[0]


# With an isochrone model: [Fe/H] on its scale, the default Z converted (same [Fe/H]), given Z kept,
# "isochrone_model" added; without: no "isochrone_model" key, as before
def test_isochrones_of_a_model():
    assert isochrone_from_feh(12.0, feh=-2.0, isochrone_model="Dotter2016") == {
        "age": 12.0, "metallicity": pytest.approx(convert_FeH_to_z(-2.0, "Dotter2016")),
        "isochrone_model": "Dotter2016"}
    assert isochrone_from_feh(metallicity=0.0003, isochrone_model="Dotter2016")["metallicity"] == 0.0003
    default_z = convert_metallicity(DEFAULT_METALLICITIES[0], Z_SCALE_ISOCHRONE_MODEL, "Dotter2016")
    assert isochrone_from_feh(isochrone_model="Dotter2016")["metallicity"] == pytest.approx(default_z)
    assert default_isochrone_grid(isochrone_model="Dotter2016") == [
        {"age": DEFAULT_AGES_GYR[0], "metallicity": pytest.approx(default_z), "isochrone_model": "Dotter2016"}]
    assert default_isochrone_grid([12.0], [0.0003], isochrone_model="Dotter2016") == [
        {"age": 12.0, "metallicity": 0.0003, "isochrone_model": "Dotter2016"}]
    assert "isochrone_model" not in default_isochrone_grid()[0]
    assert "isochrone_model" not in isochrone_from_feh(feh=-2.0)


# A known stream: the Z of the table (on the scale of Z_SCALE_ISOCHRONE_MODEL), or of the default grid when unknown,
# converted to the isochrone model; the ages unchanged
@pytest.mark.parametrize("stream", ["Both", "NoZ"])
def test_known_stream_isochrones_of_a_model(population_table, stream):
    marigo = known_stream_isochrone_parameters(stream, population_table)
    dotter = known_stream_isochrone_parameters(stream, population_table, isochrone_model="Dotter2016")
    assert [isochrone["age"] for isochrone in dotter] == [isochrone["age"] for isochrone in marigo]
    for m, d in zip(marigo, dotter):
        assert d["isochrone_model"] == "Dotter2016"
        assert d["metallicity"] == pytest.approx(
            convert_metallicity(m["metallicity"], Z_SCALE_ISOCHRONE_MODEL, "Dotter2016"))

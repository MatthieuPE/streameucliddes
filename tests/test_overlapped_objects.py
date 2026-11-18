"""Tests of catalogs/overlapped_objects.py, on small mock LVDB tables and a mock Siena Galaxy Atlas
(no download)."""

import io

import healpy as hp
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from streameucliddes.catalogs import overlapped_objects
from streameucliddes.catalogs.overlapped_objects import (
    GLOBULAR_CLUSTER_TABLES,
    OBJECT_COLUMNS,
    SGA_COLUMNS,
    OBJECT_TYPES,
    add_in_catalog,
    extract_sga_bright_galaxies,
    find_object,
    get_dwarf_galaxies,
    get_galaxies,
    get_globular_clusters,
    get_lmc,
    get_objects,
    get_smc,
    load_lvdb_table,
    load_sga_bright_galaxies,
    make_object,
    objects_near_distance_modulus,
)

pytestmark = pytest.mark.overlapped_objects


def _table(keys, confirmed, rhalf=None):
    """An LVDB-like table: the columns read, plus one the functions do not return."""
    n = len(keys)
    return pd.DataFrame({
        "key": keys, "ra": np.linspace(10.0, 20.0, n), "dec": np.linspace(-60.0, -50.0, n),
        "name": [key.upper() for key in keys], "host": "mw", "confirmed_real": confirmed,
        "rhalf": np.linspace(1.0, 2.0, n) if rhalf is None else rhalf, "ellipticity": np.linspace(0.1, 0.5, n),
        "position_angle": np.linspace(10.0, 170.0, n), "distance": np.linspace(30.0, 90.0, n),
        "apparent_magnitude_v": np.linspace(8.0, 16.0, n), "distance_modulus": np.linspace(17.0, 19.0, n),
        "metallicity": np.linspace(-2.5, -1.0, n), "age": np.linspace(10.0, 13.0, n),
    })


@pytest.fixture
def lvdb_dir(tmp_path, monkeypatch):
    """Mock LVDB tables in a temporary LVDB_DIR: a dwarf table and the globular cluster ones."""
    monkeypatch.setattr(overlapped_objects, "LVDB_DIR", tmp_path)
    version_dir = tmp_path / overlapped_objects.LVDB_VERSION
    version_dir.mkdir()
    _table(["lmc", "fornax_1", "smc", "candidate"], [1, 1, 1, 0], rhalf=[192.9, 19.9, 59.4, np.nan]).to_csv(
        version_dir / f"{overlapped_objects.DWARF_TABLE}.csv", index=False
    )
    for number, table in enumerate(GLOBULAR_CLUSTER_TABLES):
        _table([f"{table}_a", f"{table}_b"], [1, 0 if number else 1]).to_csv(version_dir / f"{table}.csv", index=False)
    return tmp_path


# ===================================================
# Test load_lvdb_table
# ===================================================


# A table in the cache is read as it is, without a download
def test_load_cached(lvdb_dir, monkeypatch):
    monkeypatch.setattr(overlapped_objects.urllib.request, "urlopen", _no_download)
    table = load_lvdb_table(overlapped_objects.DWARF_TABLE)
    assert list(table["key"]) == ["lmc", "fornax_1", "smc", "candidate"]
    assert "host" in table.columns


# A missing table is downloaded once (URL of the pinned release), written whole, then read from the cache
def test_load_downloads_once(tmp_path, monkeypatch):
    monkeypatch.setattr(overlapped_objects, "LVDB_DIR", tmp_path)
    content = _table(["a", "b"], [1, 1]).to_csv(index=False).encode()
    urls = []

    def fake_urlopen(url, timeout):
        urls.append(url)
        return io.BytesIO(content)

    monkeypatch.setattr(overlapped_objects.urllib.request, "urlopen", fake_urlopen)
    first = load_lvdb_table("gc_harris")
    second = load_lvdb_table("gc_harris")
    assert urls == [overlapped_objects.LVDB_URL.format(version=overlapped_objects.LVDB_VERSION, table="gc_harris")]
    pd.testing.assert_frame_equal(first, second)
    assert list(first["key"]) == ["a", "b"]
    assert sorted(path.name for path in (tmp_path / overlapped_objects.LVDB_VERSION).iterdir()) == ["gc_harris.csv"]


def _no_download(*args, **kwargs):
    raise AssertionError("no download expected")


# ===================================================
# Test get_lmc, get_smc, get_dwarf_galaxies, get_globular_clusters
# ===================================================


# The LMC and the SMC alone; the dwarfs: the others confirmed (the Clouds and candidates out)
def test_magellanic_clouds_and_dwarfs(lvdb_dir):
    for get_cloud, key, rhalf in ((get_lmc, "lmc", 192.9), (get_smc, "smc", 59.4)):
        cloud = get_cloud()
        assert list(cloud.columns) == OBJECT_COLUMNS
        assert list(cloud["key"]) == [key] and list(cloud["radius_arcmin"]) == [rhalf]
    dwarfs = get_dwarf_galaxies()
    assert list(dwarfs.columns) == OBJECT_COLUMNS
    assert list(dwarfs["key"]) == ["fornax_1"]


# The globular clusters: the confirmed ones of every table, positions, radii, distances, V, [Fe/H] and age as in LVDB
def test_globular_clusters(lvdb_dir):
    clusters = get_globular_clusters()
    expected = [f"{GLOBULAR_CLUSTER_TABLES[0]}_a", f"{GLOBULAR_CLUSTER_TABLES[0]}_b"]
    expected += [f"{table}_a" for table in GLOBULAR_CLUSTER_TABLES[1:]]
    assert list(clusters["key"]) == expected
    source = _table(["x", "y"], [1, 1])
    np.testing.assert_array_equal(
        clusters[["ra", "dec", "radius_arcmin", "ellipticity", "position_angle", "distance_kpc",
                  "apparent_magnitude_v", "distance_modulus", "feh", "age_gyr"]].to_numpy()[:2],
        source[["ra", "dec", "rhalf", "ellipticity", "position_angle", "distance", "apparent_magnitude_v",
                "distance_modulus", "metallicity", "age"]].to_numpy(),
    )


# ===================================================
# Test the Siena Galaxy Atlas: extract_sga_bright_galaxies, load_sga_bright_galaxies, get_galaxies
# ===================================================

# Mock atlas: MAG_LEDA on both sides of the limit (exactly at it: left out)
SGA_MAGNITUDES = [8.0, 16.4, overlapped_objects.SGA_MAG_LEDA_LIMIT, 18.0]


def _write_sga_fits(path):
    """A mock SGA-2020.fits: an ELLIPSE table with the kept columns (big-endian, padded names) and one more."""
    n = len(SGA_MAGNITUDES)
    columns = [
        fits.Column(name="SGA_ID", format="K", array=np.arange(n) + 10),
        fits.Column(name="GALAXY", format="20A", array=np.array([f"NGC{i:04d}" for i in range(n)])),
        fits.Column(name="RA", format="D", array=np.linspace(10.0, 40.0, n)),
        fits.Column(name="DEC", format="D", array=np.linspace(-30.0, 60.0, n)),
        fits.Column(name="D26", format="E", array=np.linspace(1.0, 30.0, n)),
        fits.Column(name="PA", format="E", array=np.linspace(0.0, 150.0, n)),
        fits.Column(name="BA", format="E", array=np.linspace(0.2, 0.9, n)),
        fits.Column(name="MAG_LEDA", format="E", array=np.array(SGA_MAGNITUDES)),
        fits.Column(name="Z_LEDA", format="E", array=np.zeros(n)),
    ]
    table = fits.BinTableHDU.from_columns(columns, name="ELLIPSE")
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)


@pytest.fixture
def sga_dir(tmp_path, monkeypatch):
    """An empty SGA_DIR: no file of bright galaxies yet."""
    monkeypatch.setattr(overlapped_objects, "SGA_DIR", tmp_path / "sga")
    return tmp_path / "sga"


# The bright galaxies (MAG_LEDA below the limit), the columns SGA_COLUMNS, names without padding
def test_extract_sga_bright_galaxies(tmp_path):
    _write_sga_fits(tmp_path / "SGA-2020.fits")
    extract_sga_bright_galaxies(tmp_path / "SGA-2020.fits", tmp_path / "bright.csv")
    bright = pd.read_csv(tmp_path / "bright.csv")
    assert list(bright.columns) == SGA_COLUMNS
    assert list(bright["SGA_ID"]) == [10, 11] and list(bright["GALAXY"]) == ["NGC0000", "NGC0001"]
    np.testing.assert_allclose(bright["MAG_LEDA"], [8.0, 16.4], rtol=1e-6)


# From an atlas already downloaded: the file of bright galaxies made, the atlas kept; then read
# from that file
def test_load_from_fits_path(tmp_path, sga_dir, monkeypatch):
    _write_sga_fits(tmp_path / "SGA-2020.fits")
    first = load_sga_bright_galaxies(fits_path=tmp_path / "SGA-2020.fits")
    assert (tmp_path / "SGA-2020.fits").exists()
    assert (sga_dir / overlapped_objects.SGA_BRIGHT_FILE).exists()
    monkeypatch.setattr(overlapped_objects, "extract_sga_bright_galaxies", _no_download)
    pd.testing.assert_frame_equal(load_sga_bright_galaxies(), first)


# Without it: the atlas downloaded once (SGA_URL), its bright galaxies kept, the atlas deleted
def test_load_downloads_and_deletes(tmp_path, sga_dir, monkeypatch):
    _write_sga_fits(tmp_path / "SGA-2020.fits")
    content = (tmp_path / "SGA-2020.fits").read_bytes()
    urls = []

    def fake_urlopen(url, timeout):
        urls.append(url)
        return io.BytesIO(content)

    monkeypatch.setattr(overlapped_objects.urllib.request, "urlopen", fake_urlopen)
    bright = load_sga_bright_galaxies()
    load_sga_bright_galaxies()
    assert urls == [overlapped_objects.SGA_URL]
    assert len(bright) == 2
    assert sorted(path.name for path in sga_dir.iterdir()) == [overlapped_objects.SGA_BRIGHT_FILE]


# The galaxies as objects: the D26 isophote (radius D26 / 2, ellipticity 1 - BA, PA), no distance
def test_get_galaxies(tmp_path, sga_dir):
    _write_sga_fits(tmp_path / "SGA-2020.fits")
    sga = load_sga_bright_galaxies(fits_path=tmp_path / "SGA-2020.fits")
    galaxies = get_galaxies()
    assert list(galaxies.columns) == OBJECT_COLUMNS
    assert list(galaxies["key"]) == ["sga_10", "sga_11"] and list(galaxies["name"]) == list(sga["GALAXY"])
    np.testing.assert_allclose(galaxies["radius_arcmin"], sga["D26"] / 2)
    np.testing.assert_allclose(galaxies["ellipticity"], 1 - sga["BA"])
    np.testing.assert_allclose(galaxies["position_angle"], sga["PA"])
    assert galaxies["distance_kpc"].isna().all()


# An LVDB table without V, [Fe/H], age nor distance modulus: V, [Fe/H] and age unknown (NaN), the distance modulus
# from the distance
def test_without_optional_columns(lvdb_dir):
    path = lvdb_dir / overlapped_objects.LVDB_VERSION / f"{overlapped_objects.DWARF_TABLE}.csv"
    table = pd.read_csv(path)
    table.drop(columns=["apparent_magnitude_v", "metallicity", "age", "distance_modulus"]).to_csv(path, index=False)
    dwarfs = get_dwarf_galaxies()
    assert dwarfs[["apparent_magnitude_v", "feh", "age_gyr"]].isna().all().all()
    distance_kpc = table.set_index("key").loc["fornax_1", "distance"]
    assert dwarfs["distance_modulus"].tolist() == pytest.approx([5 * np.log10(distance_kpc * 1e3) - 5])


# ===================================================
# Test get_objects, find_object, objects_near_distance_modulus
# ===================================================


# The objects of the types asked, with their type; an unknown type refused
def test_get_objects(lvdb_dir):
    objects = get_objects(("lmc", "dwarfs", "gc"))
    assert list(objects.columns) == ["type", *OBJECT_COLUMNS]
    assert objects.groupby("type", sort=False).size().to_dict() == {
        "lmc": 1, "dwarfs": 1, "gc": len(get_globular_clusters())}
    assert get_objects(("galaxies",)).empty  # the session's empty atlas
    assert set(OBJECT_TYPES) == {"lmc", "smc", "dwarfs", "gc", "galaxies"}
    with pytest.raises(ValueError, match="sagittarius"):
        get_objects(("sagittarius",))


# By key or name, whatever the case, spaces, underscores and hyphens; an unknown name lists close ones
@pytest.mark.parametrize("name", ["fornax_1", "FORNAX_1", "Fornax-1", "fornax1"])
def test_find_object(lvdb_dir, name):
    assert find_object(name)["key"] == "fornax_1"


def test_find_object_unknown(lvdb_dir):
    with pytest.raises(KeyError, match="(?i)fornax_1"):
        find_object("fornax_2")
    with pytest.raises(KeyError):
        find_object("candidate")  # not confirmed


# Within max_offset, sorted by |offset|; in_catalog from the point map of the catalog
def test_objects_near_distance_modulus(lvdb_dir, monkeypatch):
    near = objects_near_distance_modulus(17.0, 0.5)
    expected = [f"{table}_a" for table in GLOBULAR_CLUSTER_TABLES]  # their first rows: distance modulus 17
    assert near["key"].tolist() == expected
    assert (near["dm_offset"] == 0.0).all() and "in_catalog" not in near
    assert objects_near_distance_modulus(17.4, 0.5, object_types=("dwarfs",))["key"].tolist() == ["fornax_1"]

    nside = 16
    counts = np.zeros(hp.nside2npix(nside))
    first = near.iloc[0]
    counts[hp.ang2pix(nside, first["ra"], first["dec"], nest=True, lonlat=True)] = 7
    monkeypatch.setattr(overlapped_objects, "catalog_point_map", lambda catalog, data_path=None: counts)
    near = objects_near_distance_modulus(17.0, 0.5, catalog="euclid_dr1")
    assert near["n_sources_at_centre"].tolist() == [7, 7, 7]  # the same position in every mock table
    assert near["in_catalog"].all()

    # Any table: an object elsewhere has no source; the input is not changed
    away = pd.DataFrame({"ra": [first["ra"], first["ra"] + 90.0], "dec": [first["dec"], first["dec"]]})
    checked = add_in_catalog(away, "euclid_dr1")
    assert checked["n_sources_at_centre"].tolist() == [7, 0] and checked["in_catalog"].tolist() == [True, False]
    assert list(away.columns) == ["ra", "dec"]


# An object given by hand: a row of get_objects, unknown values NaN, the distance from the distance modulus
def test_make_object():
    obj = make_object("Ursa Minor II", 196.1542, 75.5724, 2.25, ellipticity=0.2, position_angle=162.0,
                      distance_modulus=16.5, feh=-2.2)
    assert list(obj.index) == ["type", *OBJECT_COLUMNS]
    assert (obj["type"], obj["key"], obj["name"]) == ("dwarfs", "custom_ursaminorii", "Ursa Minor II")
    assert obj["distance_kpc"] == pytest.approx(10 ** (16.5 / 5 + 1) / 1e3)
    assert np.isnan(obj["age_gyr"]) and np.isnan(obj["apparent_magnitude_v"])
    assert np.isnan(make_object("x", 0.0, 0.0, 1.0)["distance_kpc"])
    assert make_object("x", 0.0, 0.0, 1.0, object_type="gc", key="my_key")[["type", "key"]].tolist() == ["gc", "my_key"]
    with pytest.raises(ValueError, match="not in"):
        make_object("x", 0.0, 0.0, 1.0, object_type="star")

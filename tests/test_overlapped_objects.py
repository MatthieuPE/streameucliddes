"""Tests of catalogs/overlapped_objects.py, on small mock LVDB tables and a mock Siena Galaxy Atlas
(no download)."""

import io

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from streameucliddes.catalogs import overlapped_objects
from streameucliddes.catalogs.overlapped_objects import (
    GLOBULAR_CLUSTER_TABLES,
    OBJECT_COLUMNS,
    SGA_COLUMNS,
    extract_sga_bright_galaxies,
    get_dwarf_galaxies,
    get_galaxies,
    get_globular_clusters,
    get_lmc,
    get_smc,
    load_lvdb_table,
    load_sga_bright_galaxies,
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


# The globular clusters: the confirmed ones of every table, positions, radii and distances as in LVDB
def test_globular_clusters(lvdb_dir):
    clusters = get_globular_clusters()
    expected = [f"{GLOBULAR_CLUSTER_TABLES[0]}_a", f"{GLOBULAR_CLUSTER_TABLES[0]}_b"]
    expected += [f"{table}_a" for table in GLOBULAR_CLUSTER_TABLES[1:]]
    assert list(clusters["key"]) == expected
    source = _table(["x", "y"], [1, 1])
    np.testing.assert_array_equal(
        clusters[["ra", "dec", "radius_arcmin", "ellipticity", "position_angle", "distance_kpc"]].to_numpy()[:2],
        source[["ra", "dec", "rhalf", "ellipticity", "position_angle", "distance"]].to_numpy(),
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

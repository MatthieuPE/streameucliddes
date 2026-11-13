"""Tests of catalogs/overlapped_objects.py, on small mock LVDB tables (no download)."""

import io

import numpy as np
import pandas as pd
import pytest

from streameucliddes.catalogs import overlapped_objects
from streameucliddes.catalogs.overlapped_objects import (
    GLOBULAR_CLUSTER_TABLES,
    OBJECT_COLUMNS,
    get_dwarf_galaxies,
    get_globular_clusters,
    get_lmc,
    get_smc,
    load_lvdb_table,
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
        assert list(cloud["key"]) == [key] and list(cloud["rhalf_arcmin"]) == [rhalf]
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
        clusters[["ra", "dec", "rhalf_arcmin", "ellipticity", "position_angle", "distance_kpc"]].to_numpy()[:2],
        source[["ra", "dec", "rhalf", "ellipticity", "position_angle", "distance"]].to_numpy(),
    )

"""Tests of workflows/run_generic_search.py, with uniform mock stars."""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from streameucliddes.analysis.generic_stream_search import generic_search_dm_scan, prepare_generic_search
from streameucliddes.workflows.run_generic_search import save_generic_search_healpix_maps

pytestmark = pytest.mark.run_generic_search

NSIDE = 32


@pytest.fixture(scope="module")
def search():
    """Uniform stars (Euclid VIS, H) on ra -15 to 15, dec -30 to -10."""
    rng = np.random.default_rng(7)
    n = 100_000
    vis = rng.uniform(18.0, 26.0, n)
    catalog = pd.DataFrame({
        "ra": rng.uniform(-15.0, 15.0, n) % 360.0, "dec": rng.uniform(-30.0, -10.0, n),
        "euclid_q1_VIS_obs": vis, "euclid_q1_H_obs": vis - rng.uniform(-1.0, 1.5, n),
        "euclid_q1_VIS_err": np.full(n, 0.02), "euclid_q1_H_err": np.full(n, 0.02),
    })
    return prepare_generic_search(catalog, nside=NSIDE, distance_moduli=[16.0, 16.5, 17.0])


def _read(path):
    return pd.read_parquet(path), json.loads(pq.read_schema(path).metadata[b"streameucliddes"])


# ===================================================
# Test save_generic_search_healpix_maps
# ===================================================


# One table: the pixels inside the footprint, in the ordering of the maps (RING), and one
# column per distance modulus; the parameters in the file metadata; from the generator
def test_save_maps(search, tmp_path):
    results = list(generic_search_dm_scan(search=search, smoothing_deg=2.0))
    path = save_generic_search_healpix_maps(
        generic_search_dm_scan(search=search, smoothing_deg=2.0), tmp_path / "new_folder" / "maps.parquet",
        metadata={"run": "test", "n": np.int64(3)},
    )
    assert path == tmp_path / "new_folder" / "maps.parquet" and path.exists()
    table, metadata = _read(path)
    inside = ~search.footprint_mask
    assert list(table.columns) == [f"hpx{NSIDE}", "16.00", "16.50", "17.00"]
    np.testing.assert_array_equal(table[f"hpx{NSIDE}"], np.flatnonzero(inside))
    for dm, result in results:
        np.testing.assert_array_equal(table[f"{dm:.2f}"], result["residual"].data[inside])
    assert (metadata["map_key"], metadata["nside"], metadata["nest"]) == ("residual", NSIDE, False)
    assert metadata["isochrone"] == search.isochrones[0]
    assert metadata["match_filter"]["name"] == search.match_filter["name"]
    assert (metadata["smoothing_deg"], metadata["polynomial_degree"]) == (2.0, results[0][1]["polynomial_degree"])
    assert (metadata["run"], metadata["n"]) == ("test", 3)


# Another map, from a mapping {distance modulus: result}
def test_save_maps_density(search, tmp_path):
    results = dict(generic_search_dm_scan(search=search, smoothing_deg=0.0))
    table, metadata = _read(save_generic_search_healpix_maps(results, tmp_path / "density.parquet", map_key="density"))
    np.testing.assert_array_equal(table["16.50"], results[16.5]["density"].compressed())
    assert metadata["map_key"] == "density"


# One file per isochrone and parameters: mixed results, a distance modulus twice, or no
# result, are refused
def test_save_maps_errors(search, tmp_path):
    (_, first), = generic_search_dm_scan(search=search, distance_moduli=[17.0])
    (_, other_isochrone), = generic_search_dm_scan(
        search=search, distance_moduli=[17.0], isochrone={"age": 11.0, "metallicity": 0.0005}
    )
    (_, other_smoothing), = generic_search_dm_scan(search=search, distance_moduli=[17.0], smoothing_deg=1.0)
    with pytest.raises(ValueError, match="different isochrone"):
        save_generic_search_healpix_maps([(16.0, first), (17.0, other_isochrone)], tmp_path / "mixed.parquet")
    with pytest.raises(ValueError, match="different smoothing_deg"):
        save_generic_search_healpix_maps([(16.0, first), (17.0, other_smoothing)], tmp_path / "mixed.parquet")
    with pytest.raises(ValueError, match="twice"):
        save_generic_search_healpix_maps([(17.0, first), (17.0, first)], tmp_path / "twice.parquet")
    with pytest.raises(ValueError, match="No result"):
        save_generic_search_healpix_maps([], tmp_path / "empty.parquet")

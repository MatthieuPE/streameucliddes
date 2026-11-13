"""Tests of workflows/run_generic_search.py, with uniform mock stars."""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from streameucliddes.analysis.generic_stream_search import generic_search_dm_scan, prepare_generic_search
from PIL import Image

from streameucliddes.catalogs.galstreams_catalog import (
    galstreams_default_tracks,
    load_galstreams,
    select_galstreams_in_footprint,
)
from streameucliddes.workflows import run_generic_search as run_generic_search_module
from streameucliddes.workflows.run_generic_search import run_generic_search, save_generic_search_healpix_maps

pytestmark = pytest.mark.run_generic_search

NSIDE = 32


@pytest.fixture(scope="module")
def catalog():
    """Uniform stars (Euclid VIS, H) on ra -15 to 15, dec -30 to -10."""
    rng = np.random.default_rng(7)
    n = 100_000
    vis = rng.uniform(18.0, 26.0, n)
    return pd.DataFrame({
        "ra": rng.uniform(-15.0, 15.0, n) % 360.0, "dec": rng.uniform(-30.0, -10.0, n),
        "euclid_q1_VIS_obs": vis, "euclid_q1_H_obs": vis - rng.uniform(-1.0, 1.5, n),
        "euclid_q1_VIS_err": np.full(n, 0.02), "euclid_q1_H_err": np.full(n, 0.02),
    })


@pytest.fixture(scope="module")
def search(catalog):
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


# ===================================================
# Test run_generic_search
# ===================================================


def _gif_frames(path):
    with Image.open(path) as gif:
        return gif.n_frames


# For each isochrone, in output_dir/<name>/age<age>_z<Z>/: one four-panel frame per distance
# modulus, in their order, the gif of these frames only (not those of an earlier run); the
# same for the corrected maps with the known tracks; and the maps file, equal to the scan
def test_run_outputs(search, tmp_path):
    isochrone = search.isochrones[0]
    folder = tmp_path / "test_run" / f"age{isochrone['age']:g}_z{isochrone['metallicity']:g}"
    (folder / "frames").mkdir(parents=True)
    Image.new("RGB", (20, 10)).save(folder / "frames" / "test_run_dm21.00.png")  # an earlier run
    summary = run_generic_search(search=search, smoothing_deg=2.0, output_dir=tmp_path, name="test_run")

    outputs, = summary["isochrones"]
    assert summary["name"] == "test_run"
    assert outputs["isochrone"] == isochrone and outputs["distance_moduli"] == search.distance_moduli
    assert [path.name for path in outputs["figure_paths"]] == [
        "test_run_dm16.00.png", "test_run_dm16.50.png", "test_run_dm17.00.png"
    ]
    assert all(path.exists() for path in outputs["figure_paths"])
    assert outputs["gif_path"] == folder / "test_run.gif" and _gif_frames(outputs["gif_path"]) == 3
    assert [path.name for path in outputs["residual_figure_paths"]] == [
        "test_run_residual_dm16.00.png", "test_run_residual_dm16.50.png", "test_run_residual_dm17.00.png"
    ]
    assert all(path.parent == folder / "frames_residual" for path in outputs["residual_figure_paths"])
    assert outputs["residual_gif_path"] == folder / "test_run_residual.gif"
    assert _gif_frames(outputs["residual_gif_path"]) == 3

    assert outputs["maps_path"] == folder / "test_run_residual.parquet"
    table, metadata = _read(outputs["maps_path"])
    assert metadata["name"] == "test_run" and metadata["map_key"] == "residual"
    for dm, result in generic_search_dm_scan(search=search, smoothing_deg=2.0):
        np.testing.assert_array_equal(table[f"{dm:.2f}"], result["residual"].compressed())


# From the catalog: the search is prepared (prepare_generic_search), one folder per
# isochrone; another map in the file
def test_run_from_catalog(catalog, tmp_path):
    grid = [{"age": 12.0, "metallicity": 0.0002}, {"age": 13.0, "metallicity": 0.0003}]
    summary = run_generic_search(
        catalog, nside=NSIDE, default_grid=grid, distance_moduli=[17.0], smoothing_deg=2.0,
        map_key="density", known_tracks=False, output_dir=tmp_path,
    )
    assert [outputs["isochrone"] for outputs in summary["isochrones"]] == grid
    for outputs, isochrone in zip(summary["isochrones"], grid):
        folder = tmp_path / "generic" / f"age{isochrone['age']:g}_z{isochrone['metallicity']:g}"
        assert outputs["maps_path"] == folder / "generic_density.parquet" and outputs["maps_path"].exists()
        assert _gif_frames(outputs["gif_path"]) == 1


# The known tracks: the galstreams tracks crossing the footprint, one per stream, with their
# stream names and whole tracks
def test_known_tracks(search):
    mws = load_galstreams()
    tracks = run_generic_search_module._known_tracks(search.footprint_mask, mws)
    selection = select_galstreams_in_footprint(mws, search.footprint_mask, nest=False)
    expected = galstreams_default_tracks(selection, mws)
    assert len(tracks) > 0  # e.g. SGP-S, Molonglo in ra -15 to 15, dec -30 to -10
    assert [name for name, _, _ in tracks] == list(expected["stream_name"])
    for (_, ra, dec), track_name in zip(tracks, expected["track_name"]):
        np.testing.assert_array_equal(ra, mws[track_name].track.ra.deg)
        np.testing.assert_array_equal(dec, mws[track_name].track.dec.deg)


def test_run_errors(catalog, search, tmp_path):
    with pytest.raises(ValueError, match="catalog"):
        run_generic_search(catalog, search=search, output_dir=tmp_path)
    with pytest.raises(ValueError, match="nside"):
        run_generic_search(search=search, nside=64, output_dir=tmp_path)
    with pytest.raises(ValueError, match="Give a catalog"):
        run_generic_search(output_dir=tmp_path)

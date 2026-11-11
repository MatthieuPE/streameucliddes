"""Tests of workflows/run_known_stream_search.py, with uniform mock stars."""

import json
from types import SimpleNamespace

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pandas as pd
import pytest

from streameucliddes.analysis.known_stream_search import (
    known_stream_search_dm_along_track,
    known_stream_search_dm_scan,
    prepare_search,
)
from streameucliddes.stream_frames import phi12_bin_centres, phi12_to_radec
from streameucliddes.workflows.run_known_stream_search import (
    run_all_known_stream_searches,
    run_known_stream_search,
    save_known_stream_search_maps,
)

pytestmark = pytest.mark.run_known_stream_search

WINDOW = (-10.0, 10.0, -3.0, 3.0)
BIN = 0.5


@pytest.fixture
def search(gc_frame):
    """Prepared search on uniform mock stars (Euclid VIS, H), the footprint at phi1 < 8."""
    rng = np.random.default_rng(5)
    n = 20_000
    ra, dec = phi12_to_radec(rng.uniform(-11.0, 11.0, n), rng.uniform(-4.0, 4.0, n), gc_frame)
    vis = rng.uniform(18.0, 26.0, n)
    catalog = pd.DataFrame({
        "ra": ra, "dec": dec, "euclid_q1_VIS_obs": vis, "euclid_q1_H_obs": vis - rng.uniform(-1.0, 1.5, n),
        "euclid_q1_VIS_err": np.full(n, 0.02), "euclid_q1_H_err": np.full(n, 0.02),
    })
    phi1_centres, phi2_centres = phi12_bin_centres(WINDOW, BIN)
    footprint = np.broadcast_to(phi1_centres[:, None] > 8.0, (phi1_centres.size, phi2_centres.size))
    return prepare_search(catalog, gc_frame, WINDOW, bin_size_deg=BIN, footprint_mask=footprint)


def _track(frame):
    """galstreams-like track along phi2 = 0 of frame, phi1 -6 to 6, at 20 kpc."""
    phi1 = np.linspace(-6.0, 6.0, 121)
    points = coord.SkyCoord(
        phi1=phi1 * u.deg, phi2=np.zeros(phi1.size) * u.deg, distance=np.full(phi1.size, 20.0) * u.kpc, frame=frame
    ).transform_to(coord.ICRS())
    return SimpleNamespace(
        track=points, stream_frame=frame, InfoFlags="1111", ref="synthetic", stream_name="Mock", track_name="Mock-T00"
    )


def _read(path):
    with np.load(path) as file:
        arrays = {name: file[name] for name in file.files}
    return arrays, json.loads(arrays.pop("metadata").item())


# Phase 2 stubs: the functions still to write
def test_import():
    assert callable(run_known_stream_search) and callable(run_all_known_stream_searches)


# ===================================================
# Test save_known_stream_search_maps
# ===================================================


# One file: the maps stacked along the distance modulus (NaN outside the footprint), the
# distance moduli, the bins, and the parameters as JSON; from the generator of the scan
def test_save_maps(search, tmp_path):
    distance_moduli = [16.5, 17.0, 17.5]
    results = list(known_stream_search_dm_scan(distance_moduli=distance_moduli, search=search))
    path = save_known_stream_search_maps(
        known_stream_search_dm_scan(distance_moduli=distance_moduli, search=search),  # a generator
        tmp_path / "new_folder" / "maps.npz", metadata={"run": "test", "n": np.int64(3)},
    )
    assert path == tmp_path / "new_folder" / "maps.npz" and path.exists()
    arrays, metadata = _read(path)
    assert set(arrays) == {"density", "distance_modulus", "phi1", "phi2"}
    assert arrays["density"].shape == (3, search.phi1_centres.size, search.phi2_centres.size)
    for density, (_, result) in zip(arrays["density"], results):
        np.testing.assert_array_equal(density, result["density"].filled(np.nan))
    assert np.isnan(arrays["density"][:, search.footprint_mask]).all()
    np.testing.assert_array_equal(arrays["distance_modulus"], distance_moduli)
    np.testing.assert_array_equal(arrays["phi1"], search.phi1_centres)
    np.testing.assert_array_equal(arrays["phi2"], search.phi2_centres)
    assert metadata["window"] == list(WINDOW) and metadata["bin_size_deg"] == BIN
    assert metadata["isochrone"] == search.isochrones[0]
    assert metadata["match_filter"]["name"] == search.match_filter["name"]
    assert metadata["track_name"] is None
    assert (metadata["run"], metadata["n"]) == ("test", 3)


# Maps following the track: the offsets and the distance modulus of each phi1 bin
def test_save_maps_along_track(search, gc_frame, tmp_path):
    track = _track(gc_frame)
    results = list(known_stream_search_dm_along_track(
        stars=search.stars, track=track, window=WINDOW, bin_size_deg=BIN, dm_offsets=(-0.5, 0.5),
    ))
    arrays, metadata = _read(save_known_stream_search_maps(results, tmp_path / "along.npz"))
    assert "distance_modulus" not in arrays
    np.testing.assert_array_equal(arrays["dm_offset"], [-0.5, 0.5])
    np.testing.assert_allclose(arrays["dm_along_phi1"], [result["dm_along_phi1"] for _, result in results])
    assert metadata["track_name"] == "Mock-T00"


# One file per isochrone and grid: mixed results, or none, are refused
def test_save_maps_errors(search, tmp_path):
    two_isochrones = [
        (17.0, result) for _, result in known_stream_search_dm_scan(distance_moduli=[17.0], search=search)
    ] + [
        (17.0, result) for _, result in known_stream_search_dm_scan(
            distance_moduli=[17.0], isochrone={"age": 11.0, "metallicity": 0.0005}, search=search
        )
    ]
    with pytest.raises(ValueError, match="different isochrone"):
        save_known_stream_search_maps(two_isochrones, tmp_path / "mixed.npz")
    with pytest.raises(ValueError, match="No result"):
        save_known_stream_search_maps([], tmp_path / "empty.npz")

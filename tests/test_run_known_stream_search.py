"""Tests of workflows/run_known_stream_search.py, with uniform mock stars."""

import json
from types import SimpleNamespace

import astropy.coordinates as coord
import astropy.units as u
import healpy as hp
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from streameucliddes.analysis.known_stream_search import (
    known_stream_search_dm_along_track,
    known_stream_search_dm_scan,
    prepare_search,
)
from streameucliddes.catalogs.isochrone_parameters import known_stream_isochrone_parameters
from streameucliddes.catalogs.survey_footprint import survey_footprint_healpix_mask
from streameucliddes.stream_frames import phi12_bin_centres, phi12_to_radec
from streameucliddes.unit_conversions import convert_DM_to_kpc
from streameucliddes.workflows import run_known_stream_search as workflow
from streameucliddes.workflows.run_known_stream_search import (
    run_all_known_stream_searches,
    run_known_stream_search,
    save_known_stream_search_maps,
)

pytestmark = pytest.mark.run_known_stream_search

WINDOW = (-10.0, 10.0, -3.0, 3.0)
BIN = 0.5


@pytest.fixture(scope="module")
def catalog(gc_frame):
    """Uniform mock stars (Euclid VIS, H) around the window."""
    rng = np.random.default_rng(5)
    n = 20_000
    ra, dec = phi12_to_radec(rng.uniform(-11.0, 11.0, n), rng.uniform(-4.0, 4.0, n), gc_frame)
    vis = rng.uniform(18.0, 26.0, n)
    return pd.DataFrame({
        "ra": ra, "dec": dec, "euclid_q1_VIS_obs": vis, "euclid_q1_H_obs": vis - rng.uniform(-1.0, 1.5, n),
        "euclid_q1_VIS_err": np.full(n, 0.02), "euclid_q1_H_err": np.full(n, 0.02),
    })


@pytest.fixture
def search(catalog, gc_frame):
    """Prepared search on the mock stars, the footprint at phi1 < 8."""
    phi1_centres, phi2_centres = phi12_bin_centres(WINDOW, BIN)
    footprint = np.broadcast_to(phi1_centres[:, None] > 8.0, (phi1_centres.size, phi2_centres.size))
    return prepare_search(catalog, gc_frame, WINDOW, bin_size_deg=BIN, footprint_mask=footprint)


def _track(frame, phi1_range=(-6.0, 6.0), dm=None, ref="synthetic", stream_name="Mock", track_name="Mock-T00"):
    """galstreams-like track along phi2 = 0 of frame, at 20 kpc (or at the distance modulus dm)."""
    phi1 = np.linspace(*phi1_range, 121)
    distance = np.full(phi1.size, 20.0) * u.kpc if dm is None else convert_DM_to_kpc(np.full(phi1.size, dm))
    points = coord.SkyCoord(
        phi1=phi1 * u.deg, phi2=np.zeros(phi1.size) * u.deg, distance=distance, frame=frame
    ).transform_to(coord.ICRS())
    return SimpleNamespace(
        track=points, stream_frame=frame, InfoFlags="1111", ref=ref, stream_name=stream_name, track_name=track_name
    )


class _MockMWStreams(dict):
    """galstreams.MWStreams-like: tracks by track name, and the summary table (stream, default track)."""

    def __init__(self, tracks, on):
        super().__init__({track.track_name: track for track in tracks})
        self.summary = pd.DataFrame({
            "TrackName": [track.track_name for track in tracks],
            "Name": [track.stream_name for track in tracks],
            "On": [track.track_name in on for track in tracks],
        }).set_index("TrackName", drop=False)


@pytest.fixture(scope="module")
def mock_mws(gc_frame):
    """Mock: default track and a longer one. MockTwo: its default track out of the dm range
    (dm < 14), two others, the longest last. NoDistance: an Ibata et al. 2024 track."""
    return _MockMWStreams(
        [
            _track(gc_frame),
            _track(gc_frame, phi1_range=(-9.0, 9.0), track_name="Mock-T01"),
            _track(gc_frame, dm=13.0, stream_name="MockTwo", track_name="MockTwo-T00"),
            _track(gc_frame, phi1_range=(-3.0, 3.0), stream_name="MockTwo", track_name="MockTwo-T01"),
            _track(gc_frame, phi1_range=(-5.0, 5.0), stream_name="MockTwo", track_name="MockTwo-T02"),
            _track(gc_frame, ref="ibata2024", stream_name="NoDistance", track_name="NoDistance-I24"),
        ],
        on=["Mock-T00", "MockTwo-T00", "NoDistance-I24"],
    )


# Whole sky (RING): every track is in the footprint
SKY = np.zeros(hp.nside2npix(8), dtype=bool)


def _read(path):
    with np.load(path) as file:
        arrays = {name: file[name] for name in file.files}
    return arrays, json.loads(arrays.pop("metadata").item())


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


# ===================================================
# Test run_known_stream_search
# ===================================================


def _gif_frames(path):
    with Image.open(path) as gif:
        return gif.n_frames


# For each isochrone, in output_dir/<name>/age<age>_z<Z>/: one map and one CMD frame per
# distance modulus, in the order of the distance moduli, their gifs, and the maps file
# equal to the scan
def test_run_outputs(search, tmp_path):
    distance_moduli = [9.5, 16.5, 17.0]  # 9.5: the frame names still sort as the distance moduli
    summary = run_known_stream_search(search=search, distance_moduli=distance_moduli, output_dir=tmp_path, name="Mock")
    isochrone = search.isochrones[0]
    folder = tmp_path / "Mock" / f"age{isochrone['age']:g}_z{isochrone['metallicity']:g}"
    assert summary["name"] == "Mock" and summary["track_name"] is None and summary["window"] == search.window
    outputs, = summary["isochrones"]
    assert outputs["isochrone"] == isochrone and outputs["distance_moduli"] == distance_moduli
    for paths, subfolder in ((outputs["figure_paths"], "maps"), (outputs["cmd_figure_paths"], "cmd")):
        assert [path.parent for path in paths] == [folder / subfolder] * 3
        assert all(path.exists() for path in paths)
        assert sorted(paths) == paths  # the order of the gif
    assert outputs["gif_path"] == folder / "Mock.gif" and _gif_frames(outputs["gif_path"]) == 3
    assert outputs["cmd_gif_path"] == folder / "Mock_cmd.gif" and _gif_frames(outputs["cmd_gif_path"]) == 3

    arrays, metadata = _read(outputs["maps_path"])
    assert outputs["maps_path"] == folder / "Mock_maps.npz" and metadata["name"] == "Mock"
    np.testing.assert_array_equal(arrays["distance_modulus"], distance_moduli)
    for density, (_, result) in zip(arrays["density"], known_stream_search_dm_scan(
        distance_moduli=distance_moduli, search=search,
    )):
        np.testing.assert_array_equal(density, result["density"].filled(np.nan))


# From a galstreams track name alone: the search is prepared (prepare_search), the outputs
# are named after the track, one folder per isochrone of the stream
def test_run_stream_name(catalog, mock_mws, tmp_path):
    grid = [{"age": 12.0, "metallicity": 0.0002}, {"age": 13.0, "metallicity": 0.0003}]
    summary = run_known_stream_search(
        catalog, stream_name="Mock-T00", mws=mock_mws, default_grid=grid, distance_moduli=[17.0],
        bin_size_deg=BIN, output_dir=tmp_path,
    )
    prepared = prepare_search(catalog, stream_name="Mock-T00", mws=mock_mws, default_grid=grid, bin_size_deg=BIN)
    assert summary["name"] == summary["track_name"] == "Mock-T00"
    assert summary["window"] == prepared.window
    isochrones = known_stream_isochrone_parameters("Mock", default_grid=grid)
    assert [outputs["isochrone"] for outputs in summary["isochrones"]] == isochrones == grid
    for outputs, isochrone in zip(summary["isochrones"], isochrones):
        folder = tmp_path / "Mock-T00" / f"age{isochrone['age']:g}_z{isochrone['metallicity']:g}"
        assert outputs["maps_path"] == folder / "Mock-T00_maps.npz" and outputs["maps_path"].exists()


# The given isochrone only
def test_run_isochrone(search, tmp_path):
    isochrone = {"age": 11.0, "metallicity": 0.0005}
    summary = run_known_stream_search(
        search=search, distance_moduli=[17.0], isochrone=isochrone, output_dir=tmp_path, name="Mock",
    )
    assert [outputs["isochrone"] for outputs in summary["isochrones"]] == [isochrone]
    assert (tmp_path / "Mock" / "age11_z0.0005" / "Mock_maps.npz").exists()


def test_run_errors(catalog, search, gc_frame, tmp_path):
    with pytest.raises(ValueError, match="catalog"):
        run_known_stream_search(catalog, search=search, output_dir=tmp_path, name="Mock")
    with pytest.raises(ValueError, match="bin_size_deg"):
        run_known_stream_search(search=search, bin_size_deg=1.0, output_dir=tmp_path, name="Mock")
    with pytest.raises(ValueError, match="name"):  # a search without a track
        run_known_stream_search(search=search, distance_moduli=[17.0], output_dir=tmp_path)


# ===================================================
# Test run_all_known_stream_searches
# ===================================================


@pytest.fixture
def runs(monkeypatch):
    """run_known_stream_search replaced: records its calls, returns two isochrones."""
    calls = []

    def fake_run(catalog, *, stream_name, **kwargs):
        calls.append({"catalog": catalog, "stream_name": stream_name, **kwargs})
        return {"isochrones": [
            {"isochrone": {"age": age, "metallicity": 0.0002}, "gif_path": f"{stream_name}_{age}.gif",
             "cmd_gif_path": f"{stream_name}_{age}_cmd.gif", "maps_path": f"{stream_name}_{age}.npz"}
            for age in (12.0, 13.0)
        ]}

    monkeypatch.setattr(workflow, "run_known_stream_search", fake_run)
    return calls


# One track per stream: its default track, else its longest selected one; the Ibata et al.
# 2024 tracks only with add_ibata2024 (given to the selection and to the runs). One row
# per track and isochrone, with the selection columns and the outputs
def test_run_all_tracks(catalog, mock_mws, runs):
    table = run_all_known_stream_searches(catalog, mws=mock_mws, footprint_mask=SKY, nest=False, smoothing_deg=0.5)
    assert [call["stream_name"] for call in runs] == ["Mock-T00", "MockTwo-T02"]
    for call in runs:
        assert call["catalog"] is catalog and call["mws"] is mock_mws and call["smoothing_deg"] == 0.5
        assert call["footprint_mask"] is SKY and call["nest"] is False
    assert list(table["track_name"]) == ["Mock-T00", "Mock-T00", "MockTwo-T02", "MockTwo-T02"]
    assert list(table["age"]) == [12.0, 13.0, 12.0, 13.0]
    assert list(table["maps_path"]) == ["Mock-T00_12.0.npz", "Mock-T00_13.0.npz",
                                        "MockTwo-T02_12.0.npz", "MockTwo-T02_13.0.npz"]
    assert {"stream_name", "length_in_footprint_deg", "dm_min", "dm_max", "gif_path", "cmd_gif_path"} <= set(table)

    runs.clear()
    run_all_known_stream_searches(catalog, mws=mock_mws, footprint_mask=SKY, nest=False, add_ibata2024=True)
    assert "NoDistance-I24" in [call["stream_name"] for call in runs]
    assert all(call["add_ibata2024"] is True for call in runs)


# The given tracks, any track of a stream; not selected: error
def test_run_all_track_names(catalog, mock_mws, runs):
    table = run_all_known_stream_searches(
        catalog, track_names=["Mock-T01"], mws=mock_mws, footprint_mask=SKY, nest=False
    )
    assert [call["stream_name"] for call in runs] == ["Mock-T01"] and set(table["track_name"]) == {"Mock-T01"}
    with pytest.raises(ValueError, match="MockTwo-T00"):  # dm 13: out of the default dm range
        run_all_known_stream_searches(catalog, track_names=["MockTwo-T00"], mws=mock_mws, footprint_mask=SKY, nest=False)


# Default footprint: from the star counts of the catalog, RING
def test_run_all_default_footprint(catalog, mock_mws, runs):
    run_all_known_stream_searches(catalog, mws=mock_mws, footprint_nside=16)
    expected = survey_footprint_healpix_mask(catalog["ra"].to_numpy(), catalog["dec"].to_numpy(), 16, nest=False)
    np.testing.assert_array_equal(runs[0]["footprint_mask"], expected)
    assert runs[0]["nest"] is False
    with pytest.raises(ValueError, match="nest"):
        run_all_known_stream_searches(catalog, mws=mock_mws, footprint_mask=SKY)


# The whole run, on one track
def test_run_all_end_to_end(catalog, mock_mws, tmp_path):
    table = run_all_known_stream_searches(
        catalog, track_names=["Mock-T00"], mws=mock_mws, footprint_mask=SKY, nest=False,
        default_grid=[{"age": 13.0, "metallicity": 0.0002}], distance_moduli=[17.0], bin_size_deg=BIN,
        output_dir=tmp_path,
    )
    row, = table.itertuples()
    assert (row.track_name, row.age, row.metallicity) == ("Mock-T00", 13.0, 0.0002)
    assert row.maps_path == tmp_path / "Mock-T00" / "age13_z0.0002" / "Mock-T00_maps.npz"
    assert row.maps_path.exists() and row.gif_path.exists() and row.cmd_gif_path.exists()

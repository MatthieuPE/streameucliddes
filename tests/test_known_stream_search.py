"""Tests of analysis/known_stream_search.py, with a mock stream on a uniform background.

The background is uniform in the window and in colour-magnitude space (Euclid VIS and
H, the bands of the default match filter); the stream stars are at phi2 = 0, inside
the match filter of the default isochrone at their distance modulus (drawn in its
polygon). galstreams is replaced by a small mock library of synthetic tracks.
"""

from types import SimpleNamespace

import astropy.coordinates as coord
import astropy.units as u
import healpy as hp
import numpy as np
import pandas as pd
import pytest
from streamobs.match_filter import is_in_match_filter

from streameucliddes.analysis.catalog_preparation import prepare_match_filter_catalog
from streameucliddes.analysis.cmd_match_filter import (
    DEFAULT_MATCH_FILTER,
    build_match_filter_polygon,
    select_stars_in_match_filter,
)
from streameucliddes.analysis.known_stream_search import (
    DEFAULT_DISTANCE_MODULI,
    known_stream_search_dm_along_track,
    known_stream_search_dm_scan,
    known_stream_search_single_dm,
)
from streameucliddes.analysis.star_density_maps import star_density_phi12
from streameucliddes.catalogs import isochrone_parameters
from streameucliddes.catalogs.galstreams_catalog import galstreams_dm_along_track, galstreams_phi12_window
from streameucliddes.catalogs.isochrone_parameters import default_isochrone_grid
from streameucliddes.catalogs.survey_footprint import survey_footprint_phi12_mask
from streameucliddes.stream_frames import phi12_bin_centres, phi12_to_radec
from streameucliddes.unit_conversions import convert_DM_to_kpc

pytestmark = pytest.mark.known_stream_search

WINDOW = (-10.0, 10.0, -3.0, 3.0)
BIN = 0.2
SMOOTHING = 0.4
STREAM_DM = 17.0
ISOCHRONE = default_isochrone_grid()[0]
VIS, H = "euclid_q1_VIS", "euclid_q1_H"


def _cmd_in_polygon(rng, n, dm):
    """n (VIS, H) inside the default match filter of the default isochrone at dm."""
    polygon = build_match_filter_polygon(dm, ISOCHRONE["age"], ISOCHRONE["metallicity"], "VIS", "H", "euclid", "euclid")
    vis, h = np.empty(0), np.empty(0)
    while vis.size < n:
        vis_try = rng.uniform(18.0, 26.0, 20 * n)
        h_try = vis_try - rng.uniform(-1.0, 1.5, 20 * n)
        inside = is_in_match_filter(vis_try, h_try, polygon_vertices=polygon)
        vis, h = np.append(vis, vis_try[inside]), np.append(h, h_try[inside])
    return vis[:n], h[:n]


def _catalog(rng, frame, stream_dm_of_phi1, n_background=60_000, n_stream=3000):
    """Background + stream stars, with ra, dec and Euclid VIS, H (and errors)."""
    phi1 = rng.uniform(WINDOW[0] - 1, WINDOW[1] + 1, n_background)
    phi2 = rng.uniform(WINDOW[2] - 1, WINDOW[3] + 1, n_background)
    vis = rng.uniform(18.0, 26.0, n_background)
    h = vis - rng.uniform(-1.0, 1.5, n_background)

    # Stream: 12 groups along phi1, each at the distance modulus of its centre
    group_edges = np.linspace(-8.0, 8.0, 13)
    stream_phi1, stream_vis, stream_h = [], [], []
    for low, high in zip(group_edges[:-1], group_edges[1:]):
        n = n_stream // 12
        stream_phi1.append(rng.uniform(low, high, n))
        group_vis, group_h = _cmd_in_polygon(rng, n, stream_dm_of_phi1((low + high) / 2))
        stream_vis.append(group_vis)
        stream_h.append(group_h)
    stream_phi1 = np.concatenate(stream_phi1)
    phi1 = np.concatenate([phi1, stream_phi1])
    phi2 = np.concatenate([phi2, rng.normal(0.0, 0.2, stream_phi1.size)])
    vis = np.concatenate([vis, *stream_vis])
    h = np.concatenate([h, *stream_h])

    ra, dec = phi12_to_radec(phi1, phi2, frame)
    return pd.DataFrame({
        "ra": ra, "dec": dec, f"{VIS}_obs": vis, f"{H}_obs": h,
        f"{VIS}_err": np.full(vis.size, 0.02), f"{H}_err": np.full(vis.size, 0.02),
        "is_stream": np.arange(vis.size) >= n_background,
    })


@pytest.fixture(scope="module")
def stream_catalog(gc_frame):
    return _catalog(np.random.default_rng(1), gc_frame, lambda phi1: STREAM_DM)


def _track(frame, dm_of_phi1, phi1_range=(-6.0, 6.0), ref="synthetic", stream_name="Mock", track_name="Mock-T00"):
    """galstreams-like track along phi2 = 0 of frame, with the distance of dm_of_phi1."""
    phi1 = np.linspace(*phi1_range, 121)
    distance = convert_DM_to_kpc(dm_of_phi1(phi1))
    points = coord.SkyCoord(
        phi1=phi1 * u.deg, phi2=np.zeros(phi1.size) * u.deg, distance=distance, frame=frame
    ).transform_to(coord.ICRS())
    return SimpleNamespace(
        track=points, stream_frame=frame, InfoFlags="1111", ref=ref, stream_name=stream_name, track_name=track_name
    )


class _MockMWStreams(dict):
    """galstreams.MWStreams-like: tracks by track name, the default ('On') ones listed."""

    def __init__(self, tracks, on):
        super().__init__({track.track_name: track for track in tracks})
        self.on = set(on)

    def get_track_names_for_stream(self, StreamName, On_only=False):
        # As galstreams: any stream whose name contains StreamName
        return [name for name, track in self.items()
                if StreamName in track.stream_name and (name in self.on or not On_only)]


@pytest.fixture(scope="module")
def mock_mws(gc_frame):
    constant = lambda phi1: np.full(np.shape(phi1), STREAM_DM)
    return _MockMWStreams(
        [
            _track(gc_frame, constant),  # default track of "Mock"
            _track(gc_frame, lambda phi1: np.full(np.shape(phi1), 18.0), track_name="Mock-T01"),
            _track(gc_frame, lambda phi1: np.full(np.shape(phi1), 15.0), stream_name="MockTwo", track_name="MockTwo-T00"),
            _track(gc_frame, constant, ref="ibata2024", stream_name="NoDistance", track_name="NoDistance-I24"),
        ],
        on=["Mock-T00", "MockTwo-T00", "NoDistance-I24"],
    )


def _population_table(age_gyr=np.nan, z=np.nan):
    """Population table with the stream Mock."""
    return pd.DataFrame({"age_gyr": [age_gyr], "z": [z]}, index=pd.Index(["Mock"], name="galstreams_name"))


def _stream_contrast(result):
    """Mean density in the stream (|phi2| < 0.3, |phi1| < 7) over the background (|phi2| > 1.5)."""
    density = result["density"]
    phi1, phi2 = np.meshgrid(result["phi1_centres"], result["phi2_centres"], indexing="ij")
    in_stream = (np.abs(phi2) < 0.3) & (np.abs(phi1) < 7.0)
    return density[in_stream].mean() / density[np.abs(phi2) > 1.5].mean()


# ===================================================
# Test known_stream_search_single_dm
# ===================================================


# The map is the shared preparation, selection and density map, with its metadata
def test_single_dm_is_the_pipeline(stream_catalog, gc_frame):
    result = known_stream_search_single_dm(
        stream_catalog, gc_frame, WINDOW, STREAM_DM, bin_size_deg=BIN, smoothing_deg=SMOOTHING
    )
    stars = prepare_match_filter_catalog(stream_catalog, stream_frame=gc_frame, window=WINDOW)
    selected = select_stars_in_match_filter(stars, STREAM_DM, ISOCHRONE)
    expected = star_density_phi12(stars["phi1"][selected], stars["phi2"][selected], WINDOW, BIN, smoothing_deg=SMOOTHING)
    np.testing.assert_array_equal(result["density"], expected)

    assert result["n_stars"] == len(stars) and result["n_selected"] == selected.sum()
    for key, centres in zip(("phi1_centres", "phi2_centres"), phi12_bin_centres(WINDOW, BIN)):
        np.testing.assert_array_equal(result[key], centres)
    assert result["distance_modulus"] == STREAM_DM
    assert result["distance_kpc"] == pytest.approx(25.12, abs=0.01)
    assert result["isochrone"] == ISOCHRONE and result["match_filter"] == DEFAULT_MATCH_FILTER
    assert (result["window"], result["bin_size_deg"], result["smoothing_deg"]) == (WINDOW, BIN, SMOOTHING)
    assert result["track_name"] is None  # no galstreams track


# The stream stands out at its distance modulus
def test_single_dm_finds_the_stream(stream_catalog, gc_frame):
    result = known_stream_search_single_dm(stream_catalog, gc_frame, WINDOW, STREAM_DM, bin_size_deg=BIN)
    assert _stream_contrast(result) > 3.0


# Prepared stars give the same map as the catalog; one of them is needed, and a frame
# and a window with the catalog
def test_single_dm_prepared_stars(stream_catalog, gc_frame):
    stars = prepare_match_filter_catalog(stream_catalog, stream_frame=gc_frame, window=WINDOW)
    from_catalog = known_stream_search_single_dm(stream_catalog, gc_frame, WINDOW, STREAM_DM)
    from_stars = known_stream_search_single_dm(None, None, WINDOW, STREAM_DM, stars=stars)
    np.testing.assert_array_equal(from_stars["density"], from_catalog["density"])
    with pytest.raises(ValueError, match="prepared stars"):
        known_stream_search_single_dm(None, gc_frame, WINDOW, STREAM_DM)
    with pytest.raises(ValueError, match="stream_frame"):
        known_stream_search_single_dm(stream_catalog, None, WINDOW, STREAM_DM)
    with pytest.raises(ValueError, match="window"):
        known_stream_search_single_dm(stream_catalog, gc_frame, None, STREAM_DM)
    with pytest.raises(ValueError, match="distance_modulus"):
        known_stream_search_single_dm(stream_catalog, gc_frame, WINDOW)


# Bins outside the footprint are masked; a HEALPix footprint is put on the grid
def test_single_dm_footprint(stream_catalog, gc_frame):
    phi1_centres, phi2_centres = phi12_bin_centres(WINDOW, BIN)
    footprint_mask = np.broadcast_to(phi1_centres[:, None] > 5.0, (phi1_centres.size, phi2_centres.size))
    result = known_stream_search_single_dm(
        stream_catalog, gc_frame, WINDOW, STREAM_DM, bin_size_deg=BIN, footprint_mask=footprint_mask
    )
    np.testing.assert_array_equal(result["density"].mask, footprint_mask)

    nside = 64
    _, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    healpix_mask = dec < -25.0  # True outside
    result = known_stream_search_single_dm(
        stream_catalog, gc_frame, WINDOW, STREAM_DM, bin_size_deg=BIN, footprint_mask=healpix_mask, nest=False
    )
    expected = survey_footprint_phi12_mask(healpix_mask, gc_frame, WINDOW, BIN, nest=False)
    assert 0 < expected.sum() < expected.size
    np.testing.assert_array_equal(result["density"].mask, expected)
    with pytest.raises(ValueError, match="nest"):
        known_stream_search_single_dm(stream_catalog, gc_frame, WINDOW, STREAM_DM, footprint_mask=healpix_mask)


# ===================================================
# Test known_stream_search_dm_scan
# ===================================================


# One map per distance modulus, in order, each that of single_dm; the stream stands out
# most at its distance modulus
def test_dm_scan(stream_catalog, gc_frame):
    distance_moduli = [14.5, 17.0, 20.0]
    results = list(known_stream_search_dm_scan(stream_catalog, gc_frame, WINDOW, distance_moduli, bin_size_deg=BIN))
    assert [dm for dm, _ in results] == distance_moduli
    for dm, result in results:
        single = known_stream_search_single_dm(stream_catalog, gc_frame, WINDOW, dm, bin_size_deg=BIN)
        assert result.keys() == single.keys()
        np.testing.assert_array_equal(result["density"], single["density"])
    contrasts = [_stream_contrast(result) for _, result in results]
    assert contrasts[1] > max(contrasts[0], contrasts[2]) + 1.0


# Processes give the same maps, in the same order
def test_dm_scan_workers(stream_catalog, gc_frame):
    distance_moduli = [17.0, 15.0, 19.0]
    serial = list(known_stream_search_dm_scan(stream_catalog, gc_frame, WINDOW, distance_moduli))
    parallel = list(known_stream_search_dm_scan(stream_catalog, gc_frame, WINDOW, distance_moduli, n_workers=2))
    assert [dm for dm, _ in parallel] == distance_moduli
    for (_, expected), (_, result) in zip(serial, parallel):
        np.testing.assert_array_equal(result["density"], expected["density"])
        assert result["n_selected"] == expected["n_selected"]


# ===================================================
# Test the stream name: values from galstreams and the population table
# ===================================================


# A galstreams stream name alone: its default track gives the frame, the window and the
# distance moduli (track +- 1 mag), the population table the isochrones
def test_dm_scan_stream_name(stream_catalog, gc_frame, mock_mws):
    population_table = _population_table(age_gyr=12.0, z=0.0001)
    results = list(known_stream_search_dm_scan(
        stream_catalog, stream_name="Mock", mws=mock_mws, population_table=population_table, bin_size_deg=BIN
    ))
    track = mock_mws["Mock-T00"]
    assert [dm for dm, _ in results] == list(np.round(np.arange(16.0, 18.05, 0.1), 1))
    _, result = results[10]  # dm 17
    assert result["window"] == galstreams_phi12_window(track)
    assert result["isochrone"] == {"age": 12.0, "metallicity": 0.0001}
    assert result["track_name"] == "Mock-T00"
    explicit = known_stream_search_single_dm(
        stream_catalog, gc_frame, galstreams_phi12_window(track), 17.0, {"age": 12.0, "metallicity": 0.0001},
        bin_size_deg=BIN,
    )
    np.testing.assert_array_equal(result["density"], explicit["density"])


# The given values are kept: window, distance moduli, isochrone
def test_dm_scan_stream_name_given_values(stream_catalog, mock_mws):
    results = list(known_stream_search_dm_scan(
        stream_catalog, window=WINDOW, distance_moduli=[16.5, 17.5], isochrone=ISOCHRONE,
        stream_name="Mock", mws=mock_mws, population_table=_population_table(12.0, 0.0001), bin_size_deg=BIN,
    ))
    assert [dm for dm, _ in results] == [16.5, 17.5]
    assert all(result["window"] == WINDOW and result["isochrone"] == ISOCHRONE for _, result in results)


# Every isochrone of the stream: every distance modulus for the first, then the next
def test_dm_scan_stream_name_isochrones(stream_catalog, mock_mws, monkeypatch):
    monkeypatch.setattr(isochrone_parameters, "DEFAULT_AGES_GYR", (11.0, 13.0))
    results = list(known_stream_search_dm_scan(
        stream_catalog, distance_moduli=[16.5, 17.5], stream_name="Mock", mws=mock_mws,
        population_table=_population_table(z=0.0001), bin_size_deg=BIN,  # age unknown: the default ages
    ))
    assert [(dm, result["isochrone"]["age"]) for dm, result in results] == [
        (16.5, 11.0), (17.5, 11.0), (16.5, 13.0), (17.5, 13.0)
    ]
    assert all(result["isochrone"]["metallicity"] == 0.0001 for _, result in results)


# Track names work too, and stream names are exact: "Mock" is neither MockTwo nor Mock-T01
def test_stream_name_resolution(stream_catalog, mock_mws):
    kwargs = {"mws": mock_mws, "population_table": _population_table(), "bin_size_deg": BIN}
    assert known_stream_search_single_dm(stream_catalog, stream_name="Mock", **kwargs)["track_name"] == "Mock-T00"
    by_track = known_stream_search_single_dm(stream_catalog, stream_name="Mock-T01", **kwargs)
    assert by_track["track_name"] == "Mock-T01" and by_track["distance_modulus"] == 18.0  # its median distance
    assert known_stream_search_single_dm(stream_catalog, stream_name="MockTwo", **kwargs)["distance_modulus"] == 15.0


# A stream not in galstreams: with its frame and window, the default isochrones; else an error
def test_stream_name_not_in_galstreams(stream_catalog, gc_frame, mock_mws):
    result = known_stream_search_single_dm(
        stream_catalog, gc_frame, WINDOW, STREAM_DM, stream_name="NewStream", mws=mock_mws, bin_size_deg=BIN
    )
    assert result["track_name"] is None and result["isochrone"] == ISOCHRONE
    with pytest.raises(ValueError, match="not in galstreams"):
        known_stream_search_single_dm(stream_catalog, stream_name="NewStream", mws=mock_mws)


# A track without distance (Ibata et al. 2024): the default grid of distance moduli, 15 to 22
def test_dm_scan_stream_name_no_distance(stream_catalog, mock_mws):
    distance_moduli = [dm for dm, _ in known_stream_search_dm_scan(
        stream_catalog, stream_name="NoDistance", mws=mock_mws, bin_size_deg=1.0, smoothing_deg=0.0
    )]
    assert distance_moduli == list(DEFAULT_DISTANCE_MODULI)
    assert (distance_moduli[0], distance_moduli[-1], len(distance_moduli)) == (15.0, 22.0, 71)


# ===================================================
# Test known_stream_search_dm_along_track
# ===================================================


# A track at one distance: the map of single_dm at that distance (offset 0)
def test_dm_along_track_constant(stream_catalog, gc_frame):
    track = _track(gc_frame, lambda phi1: np.full(np.shape(phi1), STREAM_DM))
    (dm_offset, result), = known_stream_search_dm_along_track(stream_catalog, track, None, WINDOW, bin_size_deg=BIN)
    single = known_stream_search_single_dm(stream_catalog, gc_frame, WINDOW, STREAM_DM, bin_size_deg=BIN)
    assert dm_offset == 0.0
    np.testing.assert_allclose(result["dm_along_phi1"], STREAM_DM)
    np.testing.assert_array_equal(result["density"], single["density"])
    assert result["n_selected"] == single["n_selected"]


# The track of a galstreams stream name, as the track itself
def test_dm_along_track_stream_name(stream_catalog, mock_mws):
    (_, by_name), = known_stream_search_dm_along_track(
        stream_catalog, stream_name="Mock", mws=mock_mws, population_table=_population_table(), bin_size_deg=BIN
    )
    (_, by_track), = known_stream_search_dm_along_track(
        stream_catalog, mock_mws["Mock-T00"], population_table=_population_table(), bin_size_deg=BIN
    )
    np.testing.assert_array_equal(by_name["density"], by_track["density"])
    assert by_name["track_name"] == "Mock-T00"


# Distance moduli along phi1: those of galstreams_dm_along_track, held at the ends of the
# track, rounded, plus each offset
def test_dm_along_track_distance_moduli(stream_catalog, gc_frame):
    track = _track(gc_frame, lambda phi1: 17.0 + 0.1 * phi1)  # 16.4 to 17.6 from phi1 -6 to 6
    offsets = (-0.5, 0.0, 0.5)
    results = list(known_stream_search_dm_along_track(
        stream_catalog, track, None, WINDOW, dm_offsets=offsets, dm_step=0.1, bin_size_deg=BIN,
        population_table=_population_table(),
    ))
    assert [dm_offset for dm_offset, _ in results] == list(offsets)
    phi1_centres = results[0][1]["phi1_centres"]
    expected = np.round(galstreams_dm_along_track(track, phi1_centres, hold_ends=True) / 0.1) * 0.1
    np.testing.assert_allclose(expected[[0, -1]], [16.4, 17.6])  # held at the ends
    for dm_offset, result in results:
        np.testing.assert_allclose(result["dm_along_phi1"], expected + dm_offset)


# A stream whose distance changes along phi1 is found better by following its track
# than at one distance modulus
def test_dm_along_track_follows_the_stream(gc_frame):
    dm_of_phi1 = lambda phi1: 17.0 + 0.2 * np.asarray(phi1)  # 15.4 to 18.6 along the stream
    catalog = _catalog(np.random.default_rng(2), gc_frame, dm_of_phi1)
    track = _track(gc_frame, dm_of_phi1, phi1_range=(-8.0, 8.0))
    (_, along), = known_stream_search_dm_along_track(
        catalog, track, None, WINDOW, bin_size_deg=BIN, population_table=_population_table()
    )
    fixed = known_stream_search_single_dm(catalog, gc_frame, WINDOW, 17.0, bin_size_deg=BIN)

    stars = prepare_match_filter_catalog(catalog, stream_frame=gc_frame, window=WINDOW)
    n_stream = catalog.loc[stars.index, "is_stream"].sum()
    assert _stream_contrast(along) > _stream_contrast(fixed) + 0.5
    assert along["n_selected"] - fixed["n_selected"] > 0.2 * n_stream


# No track, or a track without distance: nothing to follow
def test_dm_along_track_errors(stream_catalog, gc_frame, mock_mws):
    with pytest.raises(ValueError, match="track"):
        next(known_stream_search_dm_along_track(stream_catalog, None, gc_frame, WINDOW))
    with pytest.raises(ValueError, match="no distance"):
        next(known_stream_search_dm_along_track(stream_catalog, stream_name="NoDistance", mws=mock_mws))

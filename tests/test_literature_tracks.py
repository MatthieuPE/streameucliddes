"""Tests of catalogs/literature_tracks.py and galstreams_catalog.track_from_points, with mock Li et al. 2021 files."""

import numpy as np
import pytest
from astropy.table import Table

from streameucliddes.catalogs import literature_tracks
from streameucliddes.catalogs.galstreams_catalog import galstreams_dm_along_track, track_from_points
from streameucliddes.catalogs.literature_tracks import (
    li2021_aau_distance_modulus,
    li2021_aliqa_uma_track,
    li2021_aliqa_uma_width_deg,
    load_li2021_aliqa_uma,
    radec_to_shipp2019_aau,
    shipp2019_aau_to_radec,
)
from streameucliddes.stream_frames import radec_to_phi12

pytestmark = pytest.mark.literature_tracks

KNOTS_PHI1 = np.array([-21.0, -18.25, -15.5, -12.75, -10.0])
KNOTS_PHI2 = np.array([-0.2, -0.3, 0.4, 1.2, 0.5])
WIDTH_PHI1 = np.array([-21.0, -17.0, -14.0, -10.0])
WIDTH_LOG = np.log([0.3, 0.2, 0.4, 0.3])


@pytest.fixture
def li2021_dir(tmp_path):
    """Mock files of the Li et al. 2021 repository, in a folder: no download."""
    Table({"phi1": KNOTS_PHI1, "phi2_mean": KNOTS_PHI2, "phi2_std": np.full(5, 0.1)}).write(tmp_path / "pub_track_au.fits")
    Table({"phi1": WIDTH_PHI1, "logwidth_mean": WIDTH_LOG, "logwidth_std": np.full(4, 0.1)}).write(
        tmp_path / "pub_width_au.fits")
    Table({"phi1": [-15.0], "phi2": [0.3], "stream": ["au"]}).write(tmp_path / "aau_members_full.fits")
    return tmp_path


# The Shipp 2019 frame: a rotation (round trip), and the published matrix's pole (its third row)
def test_shipp2019_frame():
    phi1, phi2 = np.array([-20.0, -15.0, 0.0, 10.0]), np.array([-1.0, 0.5, 0.0, 2.0])
    # the published matrix has 8 decimals: orthogonal to 1e-8, round trip to well below a milliarcsecond
    np.testing.assert_allclose(radec_to_shipp2019_aau(*shipp2019_aau_to_radec(phi1, phi2)), (phi1, phi2), atol=1e-5)
    pole_ra, pole_dec = 74.2590674, 47.93312435  # galstreams' AAU pole: the same great circle, to 15 arcsec
    assert radec_to_shipp2019_aau(pole_ra, pole_dec)[1] == pytest.approx(90.0, abs=0.01)


def test_distance_modulus():
    assert li2021_aau_distance_modulus(0.0) == pytest.approx(16.66)
    assert li2021_aau_distance_modulus(-20.0) == pytest.approx(16.66 + 0.56 + 0.18)


# The files are read from the folder, downloaded only when missing
def test_load(li2021_dir, monkeypatch):
    monkeypatch.setattr(literature_tracks, "download_file", lambda url, path: pytest.fail(f"downloaded {url}"))
    tables = load_li2021_aliqa_uma(li2021_dir)
    assert set(tables) == {"track", "width", "members"} and len(tables["track"]) == 5

    downloads = []
    (li2021_dir / "pub_width_au.fits").rename(li2021_dir / "kept.fits")

    def fake_download(url, path):
        downloads.append(url)
        (li2021_dir / "kept.fits").rename(path)

    monkeypatch.setattr(literature_tracks, "download_file", fake_download)
    load_li2021_aliqa_uma(li2021_dir)
    assert downloads == [f"{literature_tracks.LI2021_REPOSITORY_URL}/stream_params/pub_width_au.fits"]


# The track: through the knots in the Shipp 2019 frame, a natural spline between them, the distance of Eq. 2
def test_track(li2021_dir):
    track = li2021_aliqa_uma_track((-18.25, -12.75), step_deg=0.05, data_dir=li2021_dir)
    phi1, phi2 = radec_to_shipp2019_aau(track.track.ra.deg, track.track.dec.deg)
    assert phi1[[0, -1]] == pytest.approx([-18.25, -12.75], abs=1e-6)
    assert phi2[[0, -1]] == pytest.approx([-0.3, 1.2], abs=1e-6)  # knots
    assert np.interp(-15.5, phi1, phi2) == pytest.approx(0.4, abs=1e-4)
    dm = galstreams_dm_along_track(track, radec_to_phi12(track.track.ra.deg[50], track.track.dec.deg[50],
                                                         track.stream_frame)[0])
    assert dm == pytest.approx(li2021_aau_distance_modulus(phi1[50]), abs=1e-3)
    assert (track.track_name, track.stream_name, track.ref) == (
        literature_tracks.LI2021_AU_TRACK_NAME, "AAU-AliqaUma", "li2021_spline")


# The width: mean sigma at the knots in the range
def test_width(li2021_dir):
    assert li2021_aliqa_uma_width_deg((-19.5, -11.5), data_dir=li2021_dir) == pytest.approx(0.3)
    assert li2021_aliqa_uma_width_deg((-22.0, -9.0), data_dir=li2021_dir) == pytest.approx(0.3)


# A track from points: in the frame through its ends, with their distances; checked inputs
def test_track_from_points():
    ra, dec, dm = np.array([30.0, 32.0, 34.0]), np.array([-30.0, -31.0, -32.5]), np.array([17.0, 17.2, 17.4])
    track = track_from_points("Test-T00", ra, dec, dm)
    phi1, phi2 = radec_to_phi12(ra, dec, track.stream_frame)
    assert phi2[[0, -1]] == pytest.approx([0.0, 0.0], abs=1e-6) and phi1[-1] > phi1[0]
    np.testing.assert_allclose(galstreams_dm_along_track(track, phi1), dm, atol=1e-6)
    assert (track.stream_name, track.ref, track.InfoFlags) == ("Test", "custom", "1111")
    with pytest.raises(ValueError, match="at least two"):
        track_from_points("Test-T00", ra[:1], dec[:1], dm[:1])
    with pytest.raises(ValueError, match="as many"):
        track_from_points("Test-T00", ra, dec, dm[:2])

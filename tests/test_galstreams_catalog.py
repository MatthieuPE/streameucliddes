"""Tests of catalogs/galstreams_catalog.py.

Synthetic tracks, built along the gala frame of conftest.py with known distances,
give exact expectations. The real galstreams library (loaded once, about 20 s)
checks the validation streams of the plan (ATLAS, Phoenix, Turranburra).
"""

from types import SimpleNamespace

import astropy.coordinates as coord
import astropy.units as u
import healpy as hp
import numpy as np
import pytest

from streameucliddes.catalogs.galstreams_catalog import (
    SELECTION_COLUMNS,
    galstreams_dm_along_track,
    galstreams_phi12_window,
    galstreams_track_length_in_footprint,
    load_galstreams,
    select_galstreams_in_footprint,
)
from streameucliddes.catalogs.survey_footprint import survey_footprint_healpix_mask
from streameucliddes.stream_frames import phi12_to_radec, radec_to_phi12
from streameucliddes.unit_conversions import convert_kpc_to_DM

pytestmark = pytest.mark.galstreams_catalog

NSIDE = 256  # pixels of 0.23 deg


def _dm(distance_kpc):
    return float(convert_kpc_to_DM(distance_kpc).value)


def _synthetic_track(frame, phi1_range=(-10.0, 10.0), phi2_range=(0.0, 0.0),
                     distance_kpc=(20.0, 30.0), info_flags="1111", name="Synthetic-T00"):
    """Track along ``frame``, points every 0.01 deg, phi2 and distance linear in phi1."""
    phi1 = np.linspace(*phi1_range, round((phi1_range[1] - phi1_range[0]) / 0.01) + 1)
    points = coord.SkyCoord(
        phi1=phi1 * u.deg,
        phi2=np.linspace(*phi2_range, phi1.size) * u.deg,
        distance=np.linspace(*distance_kpc, phi1.size) * u.kpc,
        frame=frame,
    ).transform_to(coord.ICRS())
    return SimpleNamespace(
        track=points, stream_frame=frame, InfoFlags=info_flags,
        track_name=name, stream_name=name.rsplit("-", 1)[0],
    )


def _footprint_in_frame(frame, inside):
    """HEALPix mask (True outside) of the pixels whose centre (phi1, phi2) satisfies ``inside``."""
    ra, dec = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    phi1, phi2 = radec_to_phi12(ra, dec, frame)
    return ~inside(phi1, phi2)


# ===================================================
# Test galstreams_track_length_in_footprint
# ===================================================


# Whole sky: the length of the track; half of it in the footprint: half of the length
def test_track_length(gc_frame):
    track = _synthetic_track(gc_frame)
    whole_sky = np.zeros(hp.nside2npix(NSIDE), dtype=bool)
    assert galstreams_track_length_in_footprint(track, whole_sky, nest=False) == pytest.approx(20.0)

    phi1_negative = _footprint_in_frame(gc_frame, lambda phi1, phi2: phi1 < 0)
    length = galstreams_track_length_in_footprint(track, phi1_negative, nest=False)
    assert length == pytest.approx(10.0, abs=0.25)  # pixel size at the edge


# Only the part with a distance modulus in dm_range counts: here the far half (> 25 kpc)
def test_track_length_in_dm_range(gc_frame):
    track = _synthetic_track(gc_frame)  # 20 kpc at phi1 = -10 to 30 kpc at phi1 = 10
    whole_sky = np.zeros(hp.nside2npix(NSIDE), dtype=bool)
    length = galstreams_track_length_in_footprint(track, whole_sky, nest=False, dm_range=(_dm(25.0), 22.0))
    assert length == pytest.approx(10.0, abs=0.02)


# ===================================================
# Test galstreams_dm_along_track
# ===================================================


# The distance modulus of the track points, interpolated in phi1; NaN outside the track
# (the ends themselves can fall just outside, by rounding: the track has been to ICRS and back)
def test_dm_along_track(gc_frame):
    track = _synthetic_track(gc_frame)  # 20 kpc at phi1 = -10 to 30 kpc at phi1 = 10
    dm = galstreams_dm_along_track(track, [-9.0, 0.0, 9.0])
    np.testing.assert_allclose(dm, [_dm(20.5), _dm(25.0), _dm(29.5)], atol=1e-6)
    assert np.isnan(galstreams_dm_along_track(track, [-10.5, 10.5])).all()
    assert galstreams_dm_along_track(track, np.zeros((3, 4))).shape == (3, 4)


# A track with only a mean distance (InfoFlags distance bit 0): constant distance modulus
def test_dm_along_track_mean_distance(gc_frame):
    track = _synthetic_track(gc_frame, distance_kpc=(17.5, 17.5), info_flags="1011")
    np.testing.assert_allclose(galstreams_dm_along_track(track, [-9.0, 0.0, 9.0]), _dm(17.5))


# ===================================================
# Test galstreams_phi12_window
# ===================================================


# The track plus the padding, and at least +/- min_phi2_half_width_deg in phi2
def test_phi12_window(gc_frame):
    track = _synthetic_track(gc_frame, phi2_range=(-1.0, 4.0))
    window = galstreams_phi12_window(track)
    np.testing.assert_allclose(window, (-20.0, 20.0, -5.0, 6.0), atol=1e-6)
    window = galstreams_phi12_window(track, padding_phi1_deg=1.0, padding_phi2_deg=0.5, min_phi2_half_width_deg=0.0)
    np.testing.assert_allclose(window, (-11.0, 11.0, -1.5, 4.5), atol=1e-6)


# ===================================================
# Test select_galstreams_in_footprint
# ===================================================


# Kept: tracks crossing the footprint over min_length_deg at 15 < dm < 22, the longest
# first. Left out: a track outside the footprint, one at 1 kpc (dm = 10), a short one
def test_select_galstreams_in_footprint(gc_frame):
    mws = {
        "Long-T00": _synthetic_track(gc_frame, name="Long-T00"),
        "Mean-T00": _synthetic_track(gc_frame, phi1_range=(0.0, 5.0), distance_kpc=(17.5, 17.5),
                                     info_flags="1011", name="Mean-T00"),
        "Outside-T00": _synthetic_track(gc_frame, phi1_range=(100.0, 110.0), name="Outside-T00"),
        "Near-T00": _synthetic_track(gc_frame, distance_kpc=(1.0, 1.0), info_flags="1000", name="Near-T00"),
        "Short-T00": _synthetic_track(gc_frame, phi1_range=(0.0, 0.5), name="Short-T00"),
    }
    footprint_mask = _footprint_in_frame(gc_frame, lambda phi1, phi2: (np.abs(phi1) < 15) & (np.abs(phi2) < 5))
    selection = select_galstreams_in_footprint(mws, footprint_mask, nest=False)

    assert list(selection.columns) == SELECTION_COLUMNS
    assert list(selection["track_name"]) == ["Long-T00", "Mean-T00"]
    assert list(selection["stream_name"]) == ["Long", "Mean"]
    np.testing.assert_allclose(selection["length_in_footprint_deg"], [20.0, 5.0], atol=1e-6)
    np.testing.assert_allclose(selection["dm_min"], [_dm(20.0), _dm(17.5)], atol=1e-3)
    np.testing.assert_allclose(selection["dm_max"], [_dm(30.0), _dm(17.5)], atol=1e-3)
    assert list(selection["has_distance_track"]) == [True, False]


# No track crossing the footprint: an empty table, with its columns
def test_select_galstreams_nothing(gc_frame):
    mws = {"Outside-T00": _synthetic_track(gc_frame, phi1_range=(100.0, 110.0), name="Outside-T00")}
    footprint_mask = _footprint_in_frame(gc_frame, lambda phi1, phi2: np.abs(phi1) < 15)
    selection = select_galstreams_in_footprint(mws, footprint_mask, nest=False, min_length_deg=0.0)
    assert selection.empty
    assert list(selection.columns) == SELECTION_COLUMNS


# ===================================================
# Test with the galstreams library
# ===================================================


@pytest.fixture(scope="module")
def mws():
    pytest.importorskip("galstreams")
    return load_galstreams()


# The validation streams of the plan are there, at their literature distances
# (Phoenix 17.5 kpc, Turranburra 27.5 kpc: Shipp et al. 2018), selected on the whole sky
def test_galstreams_validation_streams(mws):
    whole_sky = np.zeros(hp.nside2npix(64), dtype=bool)
    selection = select_galstreams_in_footprint(mws, whole_sky, nest=False).set_index("track_name")
    for name, (dm_min, dm_max) in {
        "AAU-ATLAS-L21": (16.3, 17.1),
        "Phoenix-S19": (_dm(17.5) - 0.01, _dm(17.5) + 0.01),
        "Turranburra-S19": (_dm(27.5) - 0.01, _dm(27.5) + 0.01),
    }.items():
        assert name in selection.index
        assert dm_min < selection.loc[name, "dm_min"] <= selection.loc[name, "dm_max"] < dm_max
        assert selection.loc[name, "length_in_footprint_deg"] == pytest.approx(mws[name].length.deg, rel=1e-3)


# A footprint from stars sampled uniformly around the ATLAS track gives back ATLAS,
# over its whole length
def test_galstreams_footprint_from_stars_around_atlas(mws, rng):
    atlas = mws["AAU-ATLAS-L21"]
    phi1_min, phi1_max, _, _ = galstreams_phi12_window(atlas, padding_phi1_deg=2.0)
    phi1 = rng.uniform(phi1_min, phi1_max, 200_000)
    phi2 = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-3.0)), np.sin(np.radians(3.0)), phi1.size)))
    ra, dec = phi12_to_radec(phi1, phi2, atlas.stream_frame)  # about 12 stars per pixel

    footprint_mask = survey_footprint_healpix_mask(ra, dec, NSIDE, nest=False)
    selection = select_galstreams_in_footprint(mws, footprint_mask, nest=False).set_index("track_name")
    assert "AAU-ATLAS-L21" in selection.index
    assert selection.loc["AAU-ATLAS-L21", "length_in_footprint_deg"] == pytest.approx(atlas.length.deg, rel=0.01)

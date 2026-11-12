"""Tests of catalogs/survey_footprint.py.

Footprint maps are written to a temporary FOOTPRINT_DIR. The stream frames
are ICRS itself (phi1 = ra, phi2 = dec) and the gala great-circle frame of
conftest.py.
"""

import tracemalloc

import astropy.coordinates as coord
import healpy as hp
import numpy as np
import pytest

from streameucliddes.catalogs import survey_footprint
from streameucliddes.catalogs.survey_footprint import (
    convert_footprint_healpix_mask,
    survey_footprint_healpix_mask,
    survey_footprint_phi12_mask,
)
from streameucliddes.stream_frames import phi12_bin_centres, phi12_to_radec

pytestmark = pytest.mark.survey_footprint

NSIDE = 16


def _reorder_to(values, nest):
    """RING values in the requested ordering."""
    return hp.reorder(values, r2n=True) if nest else values


def _reorder_to_nest(values_nest, nest):
    """NESTED values in the requested ordering."""
    return values_nest if nest else hp.reorder(values_nest, n2r=True).astype(bool)


@pytest.fixture
def footprint_dir(tmp_path, monkeypatch):
    """Empty FOOTPRINT_DIR in a temporary folder."""
    monkeypatch.setattr(survey_footprint, "FOOTPRINT_DIR", tmp_path)
    return tmp_path


def _write_map(path, values, nest):
    hp.write_map(path, values, nest=nest, overwrite=True, dtype=np.float64)


# ===================================================
# Test survey_footprint_healpix_mask: from the star counts
# ===================================================


# A pixel is in the footprint when it holds at least min_count stars
@pytest.mark.parametrize("nest", [False, True])
def test_mask_from_star_counts(nest):
    ra, dec = hp.pix2ang(NSIDE, [3, 3, 10, 500], lonlat=True, nest=nest)  # 2 stars in pixel 3
    mask = survey_footprint_healpix_mask(ra, dec, NSIDE, nest=nest)
    assert mask.shape == (hp.nside2npix(NSIDE),)
    assert mask.dtype == bool
    np.testing.assert_array_equal(np.flatnonzero(~mask), [3, 10, 500])

    mask = survey_footprint_healpix_mask(ra, dec, NSIDE, nest=nest, min_count=2)
    np.testing.assert_array_equal(np.flatnonzero(~mask), [3])


# Min_count across chunks: the stars of a pixel counted in several chunks add up
def test_mask_from_star_counts_by_chunks(monkeypatch):
    ra, dec = hp.pix2ang(NSIDE, [3, 10, 3, 500, 10, 3], lonlat=True, nest=False)
    expected = survey_footprint_healpix_mask(ra, dec, NSIDE, nest=False, min_count=2)
    monkeypatch.setattr(survey_footprint, "CHUNK_SIZE", 2)
    np.testing.assert_array_equal(survey_footprint_healpix_mask(ra, dec, NSIDE, nest=False, min_count=2), expected)
    np.testing.assert_array_equal(np.flatnonzero(~expected), [3, 10])


# Peak memory: the counts per pixel and the temporaries of a chunk, not of the catalog (no
# copy of ra, dec)
def test_mask_from_star_counts_memory(rng, monkeypatch):
    chunk_size = 50_000
    monkeypatch.setattr(survey_footprint, "CHUNK_SIZE", chunk_size)
    n = 1_000_000
    ra, dec = rng.uniform(0.0, 360.0, n), rng.uniform(-90.0, 90.0, n)
    tracemalloc.start()
    try:
        survey_footprint_healpix_mask(ra, dec, NSIDE, nest=False)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 60 * chunk_size + 20 * hp.nside2npix(NSIDE) < 8 * n


# Stars with a non-finite position are left out
def test_non_finite_positions_left_out():
    ra = np.array([10.0, np.nan, 30.0])
    dec = np.array([20.0, 5.0, np.inf])
    mask = survey_footprint_healpix_mask(ra, dec, NSIDE, nest=False)
    np.testing.assert_array_equal(
        np.flatnonzero(~mask), [hp.ang2pix(NSIDE, 10.0, 20.0, lonlat=True)]
    )


# ===================================================
# Test survey_footprint_healpix_mask: from a footprint map
# ===================================================


# The footprint map is used when there is one: positive values are inside,
# whatever the ordering of the map and of the mask
@pytest.mark.parametrize("map_nest", [False, True])
@pytest.mark.parametrize("nest", [False, True])
def test_mask_from_footprint_map(footprint_dir, map_nest, nest):
    inside = np.zeros(hp.nside2npix(NSIDE), dtype=bool)  # RING
    inside[100:300] = True
    values = np.where(inside, 0.7, hp.UNSEEN)  # e.g. a coverage fraction
    _write_map(footprint_dir / "mysurvey.fits", _reorder_to(values, map_nest), map_nest)

    no_stars = np.array([])
    mask = survey_footprint_healpix_mask(no_stars, no_stars, NSIDE, nest=nest, survey_name="mysurvey")
    expected = _reorder_to((~inside).astype(float), nest).astype(bool)
    np.testing.assert_array_equal(mask, expected)


# At a coarser nside, a pixel is in the footprint when at least half of its sub-pixels are
def test_footprint_map_at_coarser_nside(footprint_dir):
    values = np.zeros(hp.nside2npix(2 * NSIDE))
    # NESTED: the sub-pixels of pixel p at NSIDE are 4p to 4p + 3 at 2 * NSIDE
    values[0:1] = 1  # pixel 0: 1 sub-pixel of 4 inside
    values[4:6] = 1  # pixel 1: 2 of 4
    values[8:12] = 1  # pixel 2: 4 of 4
    _write_map(footprint_dir / "fine.fits", values, nest=True)

    mask = survey_footprint_healpix_mask([], [], NSIDE, nest=True, survey_name="fine")
    np.testing.assert_array_equal(np.flatnonzero(~mask), [1, 2])


# At a finer nside, the sub-pixels take the value of their pixel
def test_footprint_map_at_finer_nside(footprint_dir):
    values = np.zeros(hp.nside2npix(NSIDE))
    values[[5, 7]] = 1
    _write_map(footprint_dir / "coarse.fits", values, nest=True)

    mask = survey_footprint_healpix_mask([], [], 2 * NSIDE, nest=True, survey_name="coarse")
    np.testing.assert_array_equal(np.flatnonzero(~mask), np.r_[20:24, 28:32])


# Without a footprint map for the survey, the mask is computed from the stars
def test_survey_without_footprint_map(footprint_dir):
    ra, dec = hp.pix2ang(NSIDE, [42], lonlat=True)
    mask = survey_footprint_healpix_mask(ra, dec, NSIDE, nest=False, survey_name="missing")
    np.testing.assert_array_equal(mask, survey_footprint_healpix_mask(ra, dec, NSIDE, nest=False))


# ===================================================
# Test convert_footprint_healpix_mask
# ===================================================


def _ring_footprint():
    """RING mask at NSIDE: inside, the pixels of dec -60 to -20 and ra 0 to 90."""
    ra, dec = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    return ~((dec > -60.0) & (dec < -20.0) & (ra < 90.0))


# Same nside: the mask in the ordering asked (both ways), a copy
@pytest.mark.parametrize("nest, nest_out", [(False, False), (False, True), (True, False), (True, True)])
def test_convert_ordering(nest, nest_out):
    ring = _ring_footprint()
    given = _reorder_to(ring, nest).astype(bool)
    converted = convert_footprint_healpix_mask(given, NSIDE, nest=nest, nest_out=nest_out)
    assert converted.dtype == bool and converted is not given
    np.testing.assert_array_equal(converted, _reorder_to(ring, nest_out).astype(bool))


# Coarser: a pixel is outside when any of its sub-pixels is (fully covered pixels only);
# finer: the sub-pixels of a pixel take its value
@pytest.mark.parametrize("nest", [False, True])
def test_convert_nside(nest):
    fine = np.zeros(hp.nside2npix(2 * NSIDE), dtype=bool)  # NESTED: the 4 sub-pixels of p are 4p..4p+3
    fine[[4 * 10, 4 * 11, 4 * 11 + 1, 4 * 11 + 2, 4 * 11 + 3]] = True  # pixel 10: one quarter out
    coarse = convert_footprint_healpix_mask(_reorder_to_nest(fine, nest), NSIDE, nest=nest, nest_out=True)
    np.testing.assert_array_equal(np.flatnonzero(coarse), [10, 11])

    back = convert_footprint_healpix_mask(coarse, 2 * NSIDE, nest=True, nest_out=True)
    np.testing.assert_array_equal(np.flatnonzero(back), np.arange(4 * 10, 4 * 12))


# ===================================================
# Test survey_footprint_phi12_mask
# ===================================================


@pytest.fixture
def icrs_frame():
    """ICRS as the stream frame: phi1 = ra, phi2 = dec."""
    return coord.ICRS()


# A bin is outside the footprint when its centre falls in a masked pixel
@pytest.mark.parametrize("nest", [False, True])
def test_phi12_mask(icrs_frame, nest):
    nside = 256  # pixels of 0.23 deg, smaller than the half bin
    _, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True, nest=nest)
    footprint_mask = dec < 0  # footprint: the northern hemisphere
    window = (10.0, 30.0, -5.0, 5.0)
    mask = survey_footprint_phi12_mask(footprint_mask, icrs_frame, window, 1.0, nest=nest)

    assert mask.shape == (20, 10)  # row = phi1 bin
    # phi2 bin centres -4.5 to 4.5: the first 5 bins are south of the equator
    np.testing.assert_array_equal(mask, np.broadcast_to(np.arange(10) < 5, (20, 10)))


# The grid of stream_frames.phi12_bin_centres, also when the bin size does not divide the window
def test_phi12_grid_shape(icrs_frame):
    footprint_mask = np.zeros(hp.nside2npix(NSIDE), dtype=bool)
    window = (0.0, 1.0, -0.15, 0.15)
    mask = survey_footprint_phi12_mask(footprint_mask, icrs_frame, window, 0.1, nest=False)
    assert mask.shape == (10, 3)
    assert not mask.any()


# A window narrower than half a bin is an error
def test_phi12_window_narrower_than_a_bin(icrs_frame):
    footprint_mask = np.zeros(hp.nside2npix(NSIDE), dtype=bool)
    with pytest.raises(ValueError, match="narrower than half a bin"):
        survey_footprint_phi12_mask(footprint_mask, icrs_frame, (0.0, 0.04, 0.0, 1.0), 0.1, nest=False)


# ===================================================
# Test the footprint from uniformly sampled stars
# ===================================================


def _uniform_on_sphere(rng, n, lon_range, lat_range):
    """n positions uniform on the sphere, in a lon/lat box (deg)."""
    lon = rng.uniform(*lon_range, n)
    sin_lat = rng.uniform(*np.sin(np.radians(lat_range)), n)
    return lon, np.degrees(np.arcsin(sin_lat))


# Stars sampled uniformly inside a footprint give it back. Outside, no pixel holds a
# star: the footprint from the stars lies inside the true one. Inside, a pixel is only
# missed when no star fell in it: probability exp(-n_per_pixel), about 1e-5 here.
def test_footprint_from_uniform_stars(rng, footprint, nside):
    true_mask = ~footprint  # conftest.py: footprint is True inside
    # Box around the footprint of conftest.py; about 40% of its stars fall inside
    ra, dec = _uniform_on_sphere(rng, 15_000, (30.0, 70.0), (-40.0, -10.0))
    inside = footprint[hp.ang2pix(nside, ra, dec, lonlat=True)]
    ra, dec = ra[inside], dec[inside]
    n_per_pixel = inside.sum() / footprint.sum()
    assert n_per_pixel > 10

    mask = survey_footprint_healpix_mask(ra, dec, nside, nest=False)
    assert not np.any(~mask & true_mask), "no pixel outside the footprint should hold a star"
    assert np.mean(mask[footprint]) < 0.01


# The same in a stream frame: stars sampled uniformly in a phi1/phi2 box give the box
# back on the phi1/phi2 grid. The HEALPix pixels (0.23 deg) are smaller than half a bin
# (0.5 deg): a bin outside the box never falls in a pixel holding a star.
def test_phi12_footprint_from_uniform_stars(rng, gc_frame):
    nside = 256
    box = (-10.0, 10.0, -3.0, 3.0)  # footprint: |phi1| < 10, |phi2| < 3 (deg)
    phi1, phi2 = _uniform_on_sphere(rng, 25_000, box[:2], box[2:])  # about 11 per pixel
    ra, dec = phi12_to_radec(phi1, phi2, gc_frame)

    footprint_mask = survey_footprint_healpix_mask(ra, dec, nside, nest=False)
    window = (-15.0, 15.0, -5.0, 5.0)
    mask = survey_footprint_phi12_mask(footprint_mask, gc_frame, window, 1.0, nest=False)

    phi1_centres, phi2_centres = phi12_bin_centres(window, 1.0)
    phi1_grid, phi2_grid = np.meshgrid(phi1_centres, phi2_centres, indexing="ij")
    in_box = (np.abs(phi1_grid) < box[1]) & (np.abs(phi2_grid) < box[3])
    assert mask.shape == in_box.shape == (30, 10)
    assert mask[~in_box].all()
    assert np.mean(mask[in_box]) < 0.01

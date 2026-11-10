"""Tests of catalogs/survey_footprint.py.

Footprint maps are written to a temporary FOOTPRINT_DIR. The phi1/phi2 grid is
tested in the frame aligned with ICRS (phi1 = ra, phi2 = dec), with a stand-in
for stream_frames.phi12_to_radec.
"""

import healpy as hp
import numpy as np
import pytest

from streameucliddes.catalogs import survey_footprint
from streameucliddes.catalogs.survey_footprint import (
    survey_footprint_healpix_mask,
    survey_footprint_phi12_mask,
)

pytestmark = pytest.mark.survey_footprint

NSIDE = 16


def _reorder_to(values, nest):
    """RING values in the requested ordering."""
    return hp.reorder(values, r2n=True) if nest else values


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
    mask = survey_footprint_healpix_mask(no_stars, no_stars, NSIDE, nest=nest, survey="mysurvey")
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

    mask = survey_footprint_healpix_mask([], [], NSIDE, nest=True, survey="fine")
    np.testing.assert_array_equal(np.flatnonzero(~mask), [1, 2])


# At a finer nside, the sub-pixels take the value of their pixel
def test_footprint_map_at_finer_nside(footprint_dir):
    values = np.zeros(hp.nside2npix(NSIDE))
    values[[5, 7]] = 1
    _write_map(footprint_dir / "coarse.fits", values, nest=True)

    mask = survey_footprint_healpix_mask([], [], 2 * NSIDE, nest=True, survey="coarse")
    np.testing.assert_array_equal(np.flatnonzero(~mask), np.r_[20:24, 28:32])


# Without a footprint map for the survey, the mask is computed from the stars
def test_survey_without_footprint_map(footprint_dir):
    ra, dec = hp.pix2ang(NSIDE, [42], lonlat=True)
    mask = survey_footprint_healpix_mask(ra, dec, NSIDE, nest=False, survey="missing")
    np.testing.assert_array_equal(mask, survey_footprint_healpix_mask(ra, dec, NSIDE, nest=False))


# ===================================================
# Test survey_footprint_phi12_mask
# ===================================================


def _phi12_to_radec_icrs(phi1, phi2, rotation_matrix):
    """phi12_to_radec in the frame aligned with ICRS (stand-in until stream_frames is written)."""
    return np.mod(phi1, 360.0), np.asarray(phi2)


@pytest.fixture
def icrs_frame(monkeypatch):
    """Rotation matrix of the frame aligned with ICRS, with its phi12_to_radec."""
    monkeypatch.setattr(survey_footprint, "phi12_to_radec", _phi12_to_radec_icrs)
    return np.eye(3)


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


# round((max - min) / bin_size) bins per axis, also when the bin size does not divide the window
def test_phi12_grid_shape(icrs_frame):
    footprint_mask = np.zeros(hp.nside2npix(NSIDE), dtype=bool)
    window = (0.0, 1.0, -0.15, 0.15)
    mask = survey_footprint_phi12_mask(footprint_mask, icrs_frame, window, 0.1, nest=False)
    assert mask.shape == (10, 3)
    assert not mask.any()


# A window narrower than one bin is an error
def test_phi12_window_narrower_than_a_bin(icrs_frame):
    footprint_mask = np.zeros(hp.nside2npix(NSIDE), dtype=bool)
    with pytest.raises(ValueError, match="narrower than one bin"):
        survey_footprint_phi12_mask(footprint_mask, icrs_frame, (0.0, 0.04, 0.0, 1.0), 0.1, nest=False)

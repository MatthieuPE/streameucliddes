"""Tests of analysis/build_masks.py, with mock objects (no LVDB download)."""

import astropy.units as u
import healpy as hp
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

from streameucliddes.analysis.build_masks import (
    DEFAULT_RADIUS_SCALES,
    OBJECT_TYPES,
    objects_healpix_mask,
    overlapped_objects_healpix_mask,
)
from streameucliddes.catalogs import overlapped_objects

pytestmark = pytest.mark.build_masks

NSIDE = 128  # pixels of 0.46 deg


def _objects(ra, dec, rhalf_arcmin, ellipticity=np.nan, position_angle=np.nan):
    """Objects with the columns of catalogs.overlapped_objects."""
    ra, dec, rhalf_arcmin, ellipticity, position_angle = (
        np.atleast_1d(value).astype(float) for value in (ra, dec, rhalf_arcmin, ellipticity, position_angle)
    )
    n = ra.size
    return pd.DataFrame({
        "key": [f"object_{i}" for i in range(n)], "name": "", "ra": ra, "dec": dec, "rhalf_arcmin": rhalf_arcmin,
        "ellipticity": np.broadcast_to(ellipticity, n), "position_angle": np.broadcast_to(position_angle, n),
        "distance_kpc": 50.0,
    })


def _expected_mask(objects, nside, radius_scale, nest=False):
    """Pixels whose centre is in an ellipse (SkyCoord separation and position angle), and the centre pixels."""
    lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=nest, lonlat=True)
    pixels = SkyCoord(lon * u.deg, lat * u.deg)
    mask = np.zeros(lon.size, dtype=bool)
    for row in objects.itertuples():
        centre = SkyCoord(row.ra * u.deg, row.dec * u.deg)
        angle = centre.position_angle(pixels) - np.nan_to_num(row.position_angle) * u.deg
        separation = centre.separation(pixels).deg
        a = radius_scale * row.rhalf_arcmin / 60.0
        b = a * (1.0 - np.nan_to_num(row.ellipticity))
        mask |= (separation * np.cos(angle) / a) ** 2 + (separation * np.sin(angle) / b) ** 2 <= 1.0
        mask[hp.ang2pix(nside, row.ra, row.dec, lonlat=True, nest=nest)] = True
    return mask


# ===================================================
# Test objects_healpix_mask
# ===================================================


# The pixels whose centre is in an ellipse (semi-major axis radius_scale * rhalf, ellipticity, major axis
# at the position angle), as computed with SkyCoord; objects across ra = 0, near the pole, large and
# elongated; a circle when the ellipticity is unknown
@pytest.mark.parametrize("radius_scale", [2.0, 4.0])
def test_ellipses(radius_scale):
    objects = _objects([0.5, 359.0, 80.0, 200.0], [-30.0, -40.0, -69.0, -85.0], [60.0, 30.0, 190.0, 45.0],
                       ellipticity=[0.5, np.nan, 0.16, 0.3], position_angle=[30.0, np.nan, 227.0, 100.0])
    mask = objects_healpix_mask(objects, NSIDE, nest=False, radius_scale=radius_scale)
    np.testing.assert_array_equal(mask, _expected_mask(objects, NSIDE, radius_scale))
    assert mask.sum() > objects_healpix_mask(objects, NSIDE, nest=False, radius_scale=radius_scale / 2).sum()


# Orientation: position angle 0, the major axis north-south; 90, east-west
def test_orientation():
    nside = 512
    a_deg, ellipticity = 4.0, 0.5  # semi-minor axis 2 deg
    north = hp.ang2pix(nside, 60.0, -30.0 + 3.0, lonlat=True)
    east = SkyCoord(60.0 * u.deg, -30.0 * u.deg).directional_offset_by(90.0 * u.deg, 3.0 * u.deg)
    east = hp.ang2pix(nside, east.ra.deg, east.dec.deg, lonlat=True)
    for angle, inside, outside in [(0.0, north, east), (90.0, east, north)]:
        mask = objects_healpix_mask(_objects(60.0, -30.0, a_deg * 60.0, ellipticity, angle), nside, nest=False,
                                    radius_scale=1.0)
        assert mask[inside] and not mask[outside]


# An object smaller than a pixel masks its pixel; one without a radius is not masked; no object, no mask
def test_small_unknown_and_none():
    small = objects_healpix_mask(_objects(30.0, -20.0, 0.5), NSIDE, nest=False)
    assert np.flatnonzero(small).tolist() == [hp.ang2pix(NSIDE, 30.0, -20.0, lonlat=True)]
    assert not objects_healpix_mask(_objects(30.0, -20.0, np.nan), NSIDE, nest=False).any()
    assert not objects_healpix_mask(_objects([], [], []), NSIDE, nest=False).any()


# NESTED: the RING mask reordered
def test_nest():
    objects = _objects([10.0, 50.0], [-30.0, -50.0], [40.0, 100.0], ellipticity=[0.4, np.nan],
                       position_angle=[60.0, np.nan])
    ring = objects_healpix_mask(objects, NSIDE, nest=False)
    np.testing.assert_array_equal(objects_healpix_mask(objects, NSIDE, nest=True), hp.reorder(ring, r2n=True))


# ===================================================
# Test overlapped_objects_healpix_mask
# ===================================================


@pytest.fixture
def mock_objects(monkeypatch):
    """One object per type, apart on the sky, instead of the LVDB."""
    objects = {"lmc": _objects(80.0, -69.0, 190.0, 0.16, 227.0), "smc": _objects(16.0, -72.0, 60.0),
               "dwarfs": _objects(40.0, -34.0, 20.0), "gc": _objects(10.0, -10.0, 5.0)}
    monkeypatch.setattr(overlapped_objects, "get_lmc", lambda: objects["lmc"])
    monkeypatch.setattr(overlapped_objects, "get_smc", lambda: objects["smc"])
    monkeypatch.setattr(overlapped_objects, "get_dwarf_galaxies", lambda: objects["dwarfs"])
    monkeypatch.setattr(overlapped_objects, "get_globular_clusters", lambda: objects["gc"])
    return objects


# Each flag adds the mask of its type, with its default radius scale; none: nothing masked
def test_union_of_types(mock_objects):
    single = {object_type: objects_healpix_mask(objects, NSIDE, nest=False,
                                                radius_scale=DEFAULT_RADIUS_SCALES[object_type])
              for object_type, objects in mock_objects.items()}
    flags = {object_type: f"mask_{object_type}" for object_type in OBJECT_TYPES}
    for object_type, flag in flags.items():
        np.testing.assert_array_equal(overlapped_objects_healpix_mask(NSIDE, nest=False, **{flag: True}),
                                      single[object_type])
    every = overlapped_objects_healpix_mask(NSIDE, nest=False, **dict.fromkeys(flags.values(), True))
    np.testing.assert_array_equal(every, single["lmc"] | single["smc"] | single["dwarfs"] | single["gc"])
    assert not overlapped_objects_healpix_mask(NSIDE, nest=False).any()
    assert set(DEFAULT_RADIUS_SCALES) == set(OBJECT_TYPES)


# radius_scale: one value for every type, or per type (the others at their defaults); unknown type refused
def test_radius_scale(mock_objects):
    flags = {f"mask_{object_type}": True for object_type in OBJECT_TYPES}
    scale = 1.5
    expected = np.zeros(hp.nside2npix(NSIDE), dtype=bool)
    for objects in mock_objects.values():
        expected |= objects_healpix_mask(objects, NSIDE, nest=False, radius_scale=scale)
    np.testing.assert_array_equal(overlapped_objects_healpix_mask(NSIDE, nest=False, radius_scale=scale, **flags),
                                  expected)

    per_type = overlapped_objects_healpix_mask(NSIDE, nest=False, radius_scale={"lmc": scale}, **flags)
    expected = objects_healpix_mask(mock_objects["lmc"], NSIDE, nest=False, radius_scale=scale)
    for object_type in ("smc", "dwarfs", "gc"):
        expected |= objects_healpix_mask(mock_objects[object_type], NSIDE, nest=False,
                                         radius_scale=DEFAULT_RADIUS_SCALES[object_type])
    np.testing.assert_array_equal(per_type, expected)

    with pytest.raises(ValueError, match="sagittarius"):
        overlapped_objects_healpix_mask(NSIDE, nest=False, radius_scale={"sagittarius": 1.0})

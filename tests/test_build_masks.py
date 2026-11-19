"""Tests of analysis/build_masks.py, with mock objects (no LVDB download)."""

import astropy.units as u
import healpy as hp
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

from streameucliddes.analysis.build_masks import (
    DEFAULT_RADIUS_SCALE,
    DEFAULT_STREAM_MASK_HALF_WIDTH_DEG,
    STREAM_MASK_MAX_DM_DIFFERENCE,
    STREAM_MASK_MAX_WIDTH_DEG,
    STREAM_MASK_N_SIGMA,
    DEFAULT_RADIUS_SCALES,
    GC_PROFILE_SLOPE,
    GC_REFERENCE_RADIUS_SCALE,
    GC_REFERENCE_SURFACE_BRIGHTNESS,
    OBJECT_TYPES,
    ellipse_outline_radec,
    elliptical_radius_deg,
    excluded_healpix_mask,
    fraction_on_stream_band,
    galaxies_healpix_mask,
    globular_cluster_radius_scales,
    globular_clusters_healpix_mask,
    mask_radius_scales,
    object_mask_nside,
    objects_healpix_mask,
    overlapped_objects_healpix_mask,
    stream_mask_half_width_deg,
    streams_healpix_mask,
    streams_near_distance_modulus,
    streams_not_wider_than,
)
from streameucliddes.catalogs import overlapped_objects
from streameucliddes.stream_frames import phi12_to_radec, radec_to_phi12, stream_frame_from_endpoints

pytestmark = pytest.mark.build_masks

NSIDE = 128  # pixels of 0.46 deg


def _objects(ra, dec, radius_arcmin, ellipticity=np.nan, position_angle=np.nan, v_mag=np.nan):
    """Objects with the columns of catalogs.overlapped_objects."""
    ra, dec, radius_arcmin, ellipticity, position_angle, v_mag = (
        np.atleast_1d(value).astype(float) for value in (ra, dec, radius_arcmin, ellipticity, position_angle, v_mag)
    )
    n = ra.size
    return pd.DataFrame({
        "key": [f"object_{i}" for i in range(n)], "name": "", "ra": ra, "dec": dec, "radius_arcmin": radius_arcmin,
        "ellipticity": np.broadcast_to(ellipticity, n), "position_angle": np.broadcast_to(position_angle, n),
        "distance_kpc": 50.0, "apparent_magnitude_v": np.broadcast_to(v_mag, n),
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
        a = radius_scale * row.radius_arcmin / 60.0
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


# Without the centre pixel: an object smaller than a pixel masks nothing; a larger one, the same
# pixels (their centres in its ellipse, the centre pixel among them)
def test_no_centre_pixel():
    assert not objects_healpix_mask(_objects(30.0, -20.0, 0.5), NSIDE, nest=False, mask_centre_pixel=False).any()
    large = _objects(30.0, -20.0, 120.0, 0.3, 45.0)
    np.testing.assert_array_equal(objects_healpix_mask(large, NSIDE, nest=False, mask_centre_pixel=False),
                                  objects_healpix_mask(large, NSIDE, nest=False))


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
    """Objects of each type, apart on the sky, instead of the LVDB and the atlas; galaxies: an elongated
    large one, and one smaller than a pixel."""
    objects = {"lmc": _objects(80.0, -69.0, 190.0, 0.16, 227.0), "smc": _objects(16.0, -72.0, 60.0),
               "dwarfs": _objects(40.0, -34.0, 20.0),
               "gc": _objects([10.0, 20.0], [-10.0, -10.0], [5.0, 5.0], v_mag=[6.0, 15.0]),
               "galaxies": _objects([120.0, 150.0], [20.0, 30.0], [15.0, 0.2], [0.75, 0.0], [30.0, 0.0])}
    monkeypatch.setattr(overlapped_objects, "get_lmc", lambda: objects["lmc"])
    monkeypatch.setattr(overlapped_objects, "get_smc", lambda: objects["smc"])
    monkeypatch.setattr(overlapped_objects, "get_dwarf_galaxies", lambda: objects["dwarfs"])
    monkeypatch.setattr(overlapped_objects, "get_globular_clusters", lambda: objects["gc"])
    monkeypatch.setattr(overlapped_objects, "get_galaxies", lambda: objects["galaxies"])
    return objects


def _type_mask(object_type, objects, radius_scale):
    """The mask of one type by its rules: galaxies as circles, without the centre pixel."""
    if object_type == "galaxies":
        return objects_healpix_mask(objects.assign(ellipticity=np.nan), NSIDE, nest=False, radius_scale=radius_scale,
                                    mask_centre_pixel=False)
    return objects_healpix_mask(objects, NSIDE, nest=False, radius_scale=radius_scale)


# The galaxies: circles of radius_scale * D26/2, not their ellipses; the small one masks nothing
def test_galaxies_circles(mock_objects):
    galaxies = mock_objects["galaxies"]
    mask = galaxies_healpix_mask(NSIDE, nest=False, radius_scale=4.0)
    circles = _expected_mask(galaxies.iloc[:1].assign(ellipticity=np.nan), NSIDE, 4.0)
    np.testing.assert_array_equal(mask, circles)
    assert mask.sum() > objects_healpix_mask(galaxies.iloc[:1], NSIDE, nest=False, radius_scale=4.0).sum()


# Each flag adds the mask of its type, with its default radius scale; none: nothing masked
def test_union_of_types(mock_objects):
    single = {object_type: _type_mask(object_type, objects, mask_radius_scales(objects, object_type))
              for object_type, objects in mock_objects.items()}
    assert all(single[object_type].any() for object_type in OBJECT_TYPES)
    flags = {object_type: f"mask_{object_type}" for object_type in OBJECT_TYPES}
    for object_type, flag in flags.items():
        np.testing.assert_array_equal(overlapped_objects_healpix_mask(NSIDE, nest=False, **{flag: True}),
                                      single[object_type])
    every = overlapped_objects_healpix_mask(NSIDE, nest=False, **dict.fromkeys(flags.values(), True))
    np.testing.assert_array_equal(every, np.logical_or.reduce(list(single.values())))
    assert not overlapped_objects_healpix_mask(NSIDE, nest=False).any()
    assert set(DEFAULT_RADIUS_SCALES) == set(OBJECT_TYPES)


# radius_scale: one value for every type, or per type (the others at their defaults); unknown type refused
def test_radius_scale(mock_objects):
    flags = {f"mask_{object_type}": True for object_type in OBJECT_TYPES}
    scale = 1.5
    expected = np.zeros(hp.nside2npix(NSIDE), dtype=bool)
    for object_type, objects in mock_objects.items():
        expected |= _type_mask(object_type, objects, scale)
    np.testing.assert_array_equal(overlapped_objects_healpix_mask(NSIDE, nest=False, radius_scale=scale, **flags),
                                  expected)

    per_type = overlapped_objects_healpix_mask(NSIDE, nest=False, radius_scale={"lmc": scale}, **flags)
    expected = _type_mask("lmc", mock_objects["lmc"], scale)
    for object_type in ("smc", "dwarfs", "gc", "galaxies"):
        objects = mock_objects[object_type]
        expected |= _type_mask(object_type, objects, mask_radius_scales(objects, object_type))
    np.testing.assert_array_equal(per_type, expected)
    # None: the default of the type
    np.testing.assert_array_equal(
        overlapped_objects_healpix_mask(NSIDE, nest=False, radius_scale={"gc": None, "lmc": scale}, **flags), per_type)

    with pytest.raises(ValueError, match="sagittarius"):
        overlapped_objects_healpix_mask(NSIDE, nest=False, radius_scale={"sagittarius": 1.0})


# ===================================================
# Test object_mask_nside
# ===================================================


# The object masks: pixels at most half a bin, the coarsest such nside
@pytest.mark.parametrize("bin_size_deg", [0.05, 0.1, 0.2, 1.0])
def test_object_mask_nside(bin_size_deg):
    nside = object_mask_nside(bin_size_deg)
    assert np.degrees(hp.nside2resol(nside)) <= bin_size_deg / 2 < np.degrees(hp.nside2resol(nside // 2))


# ===================================================
# Test excluded_healpix_mask
# ===================================================


def _footprint(nside, nest):
    """Footprint: True (outside) where dec < -30 deg."""
    _, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True, nest=nest)
    return dec < -30.0


# The footprint at a coarse nside, either ordering: the same sky excluded at the object nside, in RING;
# OR the objects
@pytest.mark.parametrize("nest", [False, True])
def test_excluded_mask_upgrades_footprint_or_objects(nest):
    coarse, fine = 16, 64
    objects = np.zeros(hp.nside2npix(fine), dtype=bool)
    objects[hp.query_disc(fine, hp.ang2vec(40.0, 10.0, lonlat=True), np.radians(5.0))] = True
    mask, nside = excluded_healpix_mask(_footprint(coarse, nest), objects, nest=nest)
    assert nside == fine and mask.size == objects.size
    parent = hp.ang2pix(coarse, *hp.pix2ang(fine, np.arange(mask.size), lonlat=True), lonlat=True, nest=nest)
    np.testing.assert_array_equal(mask, _footprint(coarse, nest)[parent] | objects)


# No object mask: the footprint alone, in RING at its own nside
def test_excluded_mask_without_objects():
    mask, nside = excluded_healpix_mask(_footprint(16, True), None, nest=True)
    assert nside == 16
    np.testing.assert_array_equal(mask, _footprint(16, False))


# Object mask coarser than the footprint: ValueError
def test_excluded_mask_rejects_coarser_objects():
    with pytest.raises(ValueError, match="coarser"):
        excluded_healpix_mask(_footprint(64, False), np.zeros(hp.nside2npix(16), dtype=bool), nest=False)


# ===================================================
# Test the scales of the globular clusters
# ===================================================


def _surface_brightness(v_mag, radius_arcmin):
    """Mean surface brightness within the half-light radius (mag/arcmin**2)."""
    return v_mag + 2.5 * np.log10(2 * np.pi * radius_arcmin**2)


# The reference scale at the reference surface brightness; larger for brighter clusters, as a power law of slope
# GC_PROFILE_SLOPE; at least DEFAULT_RADIUS_SCALE; DEFAULT_RADIUS_SCALE when V is unknown or not given
def test_globular_cluster_radius_scales():
    radius = 0.5
    v_reference = GC_REFERENCE_SURFACE_BRIGHTNESS - 2.5 * np.log10(2 * np.pi * radius**2)
    clusters = pd.DataFrame({"apparent_magnitude_v": [v_reference, v_reference - 1.0, 25.0, np.nan],
                             "radius_arcmin": radius})
    scales = globular_cluster_radius_scales(clusters)
    assert scales[0] == pytest.approx(GC_REFERENCE_RADIUS_SCALE)
    assert scales[1] / scales[0] == pytest.approx(10 ** (0.4 / GC_PROFILE_SLOPE))
    assert scales[2:].tolist() == [DEFAULT_RADIUS_SCALE, DEFAULT_RADIUS_SCALE]
    np.testing.assert_array_equal(globular_cluster_radius_scales(clusters[["radius_arcmin"]]), DEFAULT_RADIUS_SCALE)
    # same V, larger cluster: fainter surface brightness, smaller scale (in half-light radii)
    assert globular_cluster_radius_scales(clusters.assign(radius_arcmin=1.0))[0] < scales[0]
    # an ellipse of half the area (b/a = 0.5): brighter by 2.5 log10(2), as the circle of V 0.75 mag brighter
    flattened = globular_cluster_radius_scales(clusters.assign(ellipticity=0.5))[0]
    circle = globular_cluster_radius_scales(clusters.assign(apparent_magnitude_v=v_reference - 2.5 * np.log10(2)))[0]
    assert flattened == pytest.approx(circle)


# One scale per object: the given one, the default of the type, or per cluster
def test_mask_radius_scales():
    clusters = _objects([10.0, 20.0], [-10.0, -10.0], [1.0, 1.0], v_mag=[6.0, 15.0])
    np.testing.assert_array_equal(mask_radius_scales(clusters, "gc", 2.5), [2.5, 2.5])
    np.testing.assert_array_equal(mask_radius_scales(clusters, "gc"), globular_cluster_radius_scales(clusters))
    for object_type in ("lmc", "smc", "dwarfs", "galaxies"):
        np.testing.assert_array_equal(mask_radius_scales(clusters, object_type), DEFAULT_RADIUS_SCALES[object_type])
    with pytest.raises(ValueError, match="sagittarius"):
        mask_radius_scales(clusters, "sagittarius")


# The bright cluster is masked over its own scale, larger than DEFAULT_RADIUS_SCALE; the faint one over
# DEFAULT_RADIUS_SCALE; a given scale for both
def test_globular_clusters_mask(mock_objects):
    clusters = mock_objects["gc"]
    scales = globular_cluster_radius_scales(clusters)
    assert scales[0] > DEFAULT_RADIUS_SCALE == scales[1]
    expected = (_expected_mask(clusters.iloc[:1], NSIDE, scales[0])
                | _expected_mask(clusters.iloc[1:], NSIDE, scales[1]))
    np.testing.assert_array_equal(globular_clusters_healpix_mask(NSIDE, nest=False), expected)
    np.testing.assert_array_equal(globular_clusters_healpix_mask(NSIDE, nest=False, radius_scale=3.0),
                                  _expected_mask(clusters, NSIDE, 3.0))


# The outline: every point at elliptical radius semi_major_deg; the major-axis ends at that separation
@pytest.mark.parametrize("centre", [(60.0, -30.0), (0.1, 80.0)])
def test_ellipse_outline_radec(centre):
    ra, dec = ellipse_outline_radec(*centre, 2.0, 0.4, 30.0, n_points=72)
    assert ra.size == 73 and ra[0] == pytest.approx(ra[-1]) and dec[0] == pytest.approx(dec[-1])
    np.testing.assert_allclose(elliptical_radius_deg(ra, dec, *centre, 0.4, 30.0), 2.0, rtol=1e-9)
    separation = SkyCoord(centre[0] * u.deg, centre[1] * u.deg).separation(SkyCoord(ra * u.deg, dec * u.deg)).deg
    assert separation.max() == pytest.approx(2.0) and separation.min() == pytest.approx(2.0 * 0.6)


# ===================================================
# Test streams_healpix_mask
# ===================================================

STREAM_NSIDE = 512  # pixels of 0.11 deg


def _straight_track(width_deg, n_points):
    """Track of n_points along phi2 = 0 of a frame, from phi1 = -3 to 3 deg."""
    frame = stream_frame_from_endpoints((40.0, -30.0), (50.0, -32.0))
    ra, dec = phi12_to_radec(np.linspace(-3.0, 3.0, n_points), np.zeros(n_points), frame)
    return {"name": "Straight-T00", "stream_name": "Straight", "ra": ra, "dec": dec, "stream_frame": frame,
            "width_deg": width_deg}


# n_sigma times the width; the default half-width when the width is unknown; the defaults of the module
def test_stream_mask_half_width():
    assert stream_mask_half_width_deg(0.2, n_sigma=2.0, default_half_width_deg=0.7) == pytest.approx(0.4)
    for unknown in (np.nan, -1.0, 0.0):
        assert stream_mask_half_width_deg(unknown, n_sigma=2.0, default_half_width_deg=0.7) == 0.7
    assert stream_mask_half_width_deg(0.2) == pytest.approx(STREAM_MASK_N_SIGMA * 0.2)
    assert stream_mask_half_width_deg(np.nan) == DEFAULT_STREAM_MASK_HALF_WIDTH_DEG


# A band of half-width n_sigma * width along the track, between its points too (two points or many), ends rounded:
# nothing beyond a half-width from the ends; no track: nothing masked
@pytest.mark.parametrize("n_points", [2, 50])
def test_streams_healpix_mask_band(n_points):
    track = _straight_track(0.2, n_points)
    mask = streams_healpix_mask([track], STREAM_NSIDE, nest=False, n_sigma=2.0, default_half_width_deg=5.0)
    ra, dec = hp.pix2ang(STREAM_NSIDE, np.arange(hp.nside2npix(STREAM_NSIDE)), lonlat=True)
    phi1, phi2 = radec_to_phi12(ra, dec, track["stream_frame"])
    along = np.abs(phi1) <= 3.0
    assert mask[along & (np.abs(phi2) <= 0.39)].all()
    assert not mask[along & (np.abs(phi2) > 0.4)].any()
    assert mask[(np.abs(phi1) <= 3.35) & (np.abs(phi2) <= 0.05)].all()
    assert not mask[np.abs(phi1) > 3.4].any()
    assert not streams_healpix_mask([], STREAM_NSIDE, nest=False).any()


# Unknown width: the default half-width; NESTED: the RING mask reordered
def test_streams_healpix_mask_unknown_width_and_nest():
    track = _straight_track(np.nan, 2)
    ring = streams_healpix_mask([track], STREAM_NSIDE, nest=False, n_sigma=2.0, default_half_width_deg=0.3)
    np.testing.assert_array_equal(ring, streams_healpix_mask([_straight_track(0.15, 2)], STREAM_NSIDE, nest=False,
                                                             n_sigma=2.0, default_half_width_deg=5.0))
    nested = streams_healpix_mask([track], STREAM_NSIDE, nest=True, n_sigma=2.0, default_half_width_deg=0.3)
    np.testing.assert_array_equal(hp.reorder(nested, n2r=True), ring)


# Kept: tracks whose distance moduli come within max_dm_difference (bounds included), and those of unknown
# distance; every track without a distance modulus for the candidate
def test_streams_near_distance_modulus():
    tracks = [{"name": "near", "dm_min": 16.0, "dm_max": 16.5}, {"name": "edge", "dm_min": 18.5, "dm_max": 18.5},
              {"name": "far", "dm_min": 19.0, "dm_max": 20.0}, {"name": "unknown", "dm_min": np.nan, "dm_max": np.nan}]
    names = [track["name"] for track in streams_near_distance_modulus(tracks, 17.0, max_dm_difference=1.5)]
    assert names == ["near", "edge", "unknown"]
    names = [track["name"] for track in streams_near_distance_modulus(tracks, 17.0, max_dm_difference=0.4)]
    assert names == ["unknown"]
    assert streams_near_distance_modulus(tracks, None) == tracks
    assert len(streams_near_distance_modulus(tracks, 17.0)) == len(
        streams_near_distance_modulus(tracks, 17.0, max_dm_difference=STREAM_MASK_MAX_DM_DIFFERENCE))


# Kept: tracks not wider than max_width_deg (bound included) and those of unknown width
def test_streams_not_wider_than():
    tracks = [{"name": "narrow", "width_deg": 0.3}, {"name": "edge", "width_deg": 1.0},
              {"name": "broad", "width_deg": 4.4}, {"name": "unknown", "width_deg": np.nan}]
    assert [track["name"] for track in streams_not_wider_than(tracks, 1.0)] == ["narrow", "edge", "unknown"]
    assert [track["name"] for track in streams_not_wider_than(tracks, 0.2)] == ["unknown"]
    assert streams_not_wider_than(tracks) == streams_not_wider_than(tracks, STREAM_MASK_MAX_WIDTH_DEG)


# Points within the half-width of the track (between its two points too) are on its band; beyond, not; no point: 0
def test_fraction_on_stream_band():
    track = _straight_track(0.2, 2)
    phi1 = np.array([-2.0, 0.0, 1.0, 2.5])
    inside = phi12_to_radec(phi1, np.full(4, 0.38), track["stream_frame"])
    outside = phi12_to_radec(phi1, np.full(4, -0.42), track["stream_frame"])
    kwargs = {"n_sigma": 2.0, "default_half_width_deg": 5.0}
    assert fraction_on_stream_band(*inside, track, **kwargs) == 1.0
    assert fraction_on_stream_band(*outside, track, **kwargs) == 0.0
    ra, dec = (np.concatenate(values) for values in zip(inside, outside))
    assert fraction_on_stream_band(ra, dec, track, **kwargs) == 0.5
    assert fraction_on_stream_band([], [], track, **kwargs) == 0.0


# A track's own default_half_width_deg replaces the argument when its width is unknown, in the mask and the fraction
def test_track_default_half_width():
    track = {**_straight_track(np.nan, 2), "default_half_width_deg": 0.3}
    expected = streams_healpix_mask([_straight_track(0.15, 2)], STREAM_NSIDE, nest=False, n_sigma=2.0)
    np.testing.assert_array_equal(streams_healpix_mask([track], STREAM_NSIDE, nest=False, n_sigma=2.0,
                                                       default_half_width_deg=5.0), expected)
    on_band = phi12_to_radec(np.array([0.0]), np.array([0.5]), track["stream_frame"])
    assert fraction_on_stream_band(*on_band, track, default_half_width_deg=5.0) == 0.0

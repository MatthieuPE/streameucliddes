"""Tests of analysis/gc_dwarf_optimization.py, with mock objects and uniform mock stars."""

import astropy.units as u
import healpy as hp
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord, angular_separation

from streameucliddes.analysis.build_masks import elliptical_radius_deg, mask_radius_scales
from streameucliddes.analysis.gc_dwarf_optimization import (
    object_field_radius_deg,
    object_field_window,
    object_radec_box,
    object_regions,
)
from streameucliddes.analysis.on_off_regions import region_density_map
from streameucliddes.catalogs import overlapped_objects
from streameucliddes.stream_frames import frame_centred_on_point, phi12_bin_centres, phi12_to_radec, radec_to_phi12

pytestmark = pytest.mark.gc_dwarf_optimization

# The target: an elongated dwarf galaxy; a globular cluster in its off annulus, along the major axis
CENTRE = (50.0, -30.0)
RHALF_ARCMIN, ELLIPTICITY, POSITION_ANGLE = 6.0, 0.5, 30.0
ON_RADII, OFF_RADII = (0.0, 2.0), (5.0, 10.0)
NEAR_GC_OFFSET_ARCMIN, NEAR_GC_RHALF_ARCMIN = 45.0, 0.5


def _object(key, ra, dec, radius_arcmin, ellipticity=np.nan, position_angle=np.nan, v_mag=np.nan):
    """One object with the columns of catalogs.overlapped_objects.get_objects."""
    return {"key": key, "name": key.upper(), "ra": ra, "dec": dec, "radius_arcmin": radius_arcmin,
            "ellipticity": ellipticity, "position_angle": position_angle, "distance_kpc": 100.0,
            "distance_modulus": 20.0, "feh": -2.0, "age_gyr": 12.0, "apparent_magnitude_v": v_mag}


def _near_gc_position():
    """(ra, dec) of the globular cluster in the off annulus of the target."""
    position = SkyCoord(CENTRE[0] * u.deg, CENTRE[1] * u.deg).directional_offset_by(
        POSITION_ANGLE * u.deg, NEAR_GC_OFFSET_ARCMIN * u.arcmin)
    return position.ra.deg, position.dec.deg


@pytest.fixture
def target(monkeypatch):
    """The target dwarf, and the cluster near it, as the only objects."""
    dwarf = _object("dwarf_a", *CENTRE, RHALF_ARCMIN, ELLIPTICITY, POSITION_ANGLE)
    cluster = _object("gc_near", *_near_gc_position(), NEAR_GC_RHALF_ARCMIN)
    monkeypatch.setattr(overlapped_objects, "get_dwarf_galaxies", lambda: pd.DataFrame([dwarf]))
    monkeypatch.setattr(overlapped_objects, "get_globular_clusters", lambda: pd.DataFrame([cluster]))
    return pd.Series(dwarf)


@pytest.fixture
def stars():
    """Stars uniform on the sphere within 1.2 deg of the target."""
    rng = np.random.default_rng(1)
    n = 200_000
    separation = np.degrees(np.arccos(1.0 - rng.random(n) * (1.0 - np.cos(np.radians(1.2)))))
    sky = SkyCoord(CENTRE[0] * u.deg, CENTRE[1] * u.deg).directional_offset_by(
        rng.uniform(0.0, 360.0, n) * u.deg, separation * u.deg)
    return pd.DataFrame({"ra": sky.ra.deg, "dec": sky.dec.deg})


def _no_footprint_hole():
    return np.zeros(hp.nside2npix(64), dtype=bool)


def _regions(stars, target, **kwargs):
    kwargs = {"on_radii": ON_RADII, "off_radii": OFF_RADII, "masked_object_types": ("dwarfs", "gc"),
              "footprint_mask": _no_footprint_hole(), **kwargs}
    return object_regions(stars, target, **kwargs)


def _ellipse_area(radius_rhalf):
    """Area of the ellipse of the target of semi-major axis radius_rhalf half-light radii (small: flat)."""
    return np.pi * (radius_rhalf * RHALF_ARCMIN / 60.0) ** 2 * (1 - ELLIPTICITY)


# ===================================================
# Field
# ===================================================


# The box holds every point within the radius, also across ra = 0 and near a pole
@pytest.mark.parametrize("centre", [CENTRE, (0.2, -60.0), (359.5, 40.0), (10.0, 88.5)])
def test_object_radec_box(centre):
    obj = pd.Series({"ra": centre[0], "dec": centre[1], "radius_arcmin": 6.0})
    radius = object_field_radius_deg(obj, OFF_RADII)
    assert radius == pytest.approx(OFF_RADII[1] * 0.1)
    (ra_min, ra_max), (dec_min, dec_max) = object_radec_box(obj, radius, margin_deg=0.0)
    edge = SkyCoord(centre[0] * u.deg, centre[1] * u.deg).directional_offset_by(
        np.linspace(0.0, 360.0, 73) * u.deg, radius * 0.999 * u.deg)
    ra, dec = edge.ra.deg, edge.dec.deg
    in_ra = (ra >= ra_min) & (ra <= ra_max) if ra_min <= ra_max else (ra >= ra_min) | (ra <= ra_max)
    assert in_ra.all() and ((dec >= dec_min) & (dec <= dec_max)).all()


# Every point within the radius is inside the window of the field, in the frame of the object
def test_object_field_window():
    frame = frame_centred_on_point(*CENTRE)
    edge = SkyCoord(CENTRE[0] * u.deg, CENTRE[1] * u.deg).directional_offset_by(
        np.linspace(0.0, 360.0, 73) * u.deg, 0.999 * u.deg)
    phi1, phi2 = radec_to_phi12(edge.ra.deg, edge.dec.deg, frame)
    phi1_min, phi1_max, phi2_min, phi2_max = object_field_window(1.0)
    assert ((phi1 >= phi1_min) & (phi1 <= phi1_max) & (phi2 >= phi2_min) & (phi2 <= phi2_max)).all()


# ===================================================
# Regions
# ===================================================


# On and off stars from their elliptical radius; the stars on the other object excluded; areas of the ellipses,
# less the mask of the other object; phi1/phi2 in the frame centred on the target
def test_object_regions(stars, target):
    regions = _regions(stars, target)
    half_light_deg = RHALF_ARCMIN / 60.0
    radius = elliptical_radius_deg(stars["ra"], stars["dec"], *CENTRE, ELLIPTICITY, POSITION_ANGLE) / half_light_deg
    np.testing.assert_allclose(regions["elliptical_radius"], radius)
    gc_ra, gc_dec = _near_gc_position()
    near_gc = pd.DataFrame([{"radius_arcmin": NEAR_GC_RHALF_ARCMIN}])
    mask_radius_deg = mask_radius_scales(near_gc, "gc")[0] * NEAR_GC_RHALF_ARCMIN / 60.0
    on_gc = np.degrees(angular_separation(*np.radians([gc_ra, gc_dec]),
                                          *np.radians([stars["ra"].to_numpy(), stars["dec"].to_numpy()])))
    on_gc = on_gc <= mask_radius_deg
    assert on_gc.sum() > 0
    np.testing.assert_array_equal(regions["excluded"], on_gc)
    np.testing.assert_array_equal(regions["on"], (radius <= ON_RADII[1]) & ~on_gc)
    np.testing.assert_array_equal(regions["off"], (radius >= OFF_RADII[0]) & (radius <= OFF_RADII[1]) & ~on_gc)
    assert regions["masked_objects"]["key"].tolist() == ["gc_near"]  # the target itself not masked
    phi1, phi2 = radec_to_phi12(stars["ra"], stars["dec"], frame_centred_on_point(*CENTRE))
    np.testing.assert_allclose(regions["phi1"], phi1)
    np.testing.assert_allclose(regions["phi2"], phi2)

    gc_area = np.pi * mask_radius_deg**2
    off_full = _ellipse_area(OFF_RADII[1]) - _ellipse_area(OFF_RADII[0])
    assert regions["on_area_deg2"] == pytest.approx(_ellipse_area(ON_RADII[1]), rel=0.01)
    assert regions["off_area_deg2"] == pytest.approx(off_full - gc_area, rel=0.01)
    assert regions["on_area_fraction"] == pytest.approx(1.0)
    assert regions["off_area_fraction"] == pytest.approx(1.0 - gc_area / off_full, rel=0.01)
    # uniform stars: the same density on and off, within the Poisson noise
    on_density = regions["on"].sum() / regions["on_area_deg2"]
    off_density = regions["off"].sum() / regions["off_area_deg2"]
    assert abs(on_density - off_density) < 5 * on_density / np.sqrt(regions["on"].sum())


# A large field (5 deg): the areas those of the circles on the sphere, 2 pi (1 - cos r); a flat annulus from 3 to
# 5 deg is off by 9e-4; pieces fine enough for the edge effects to be smaller (100 across the on region)
def test_object_regions_large_field(stars):
    obj = pd.Series(_object("big", *CENTRE, 30.0))  # a circle of 30 arcmin
    regions = object_regions(stars, obj, on_radii=(0.0, 4.0), off_radii=(6.0, 10.0), masked_object_types=(),
                             footprint_mask=_no_footprint_hole(), area_step_deg=0.02)

    def cap(radius_deg):
        return 2 * np.pi * (1 - np.cos(np.radians(radius_deg))) * np.degrees(1.0) ** 2

    assert regions["on_area_deg2"] == pytest.approx(cap(2.0), rel=5e-4)  # the edge of 100 pieces across
    assert regions["off_area_deg2"] == pytest.approx(cap(5.0) - cap(3.0), rel=1e-4)
    flat = np.pi * (5.0**2 - 3.0**2)
    assert abs(regions["off_area_deg2"] / flat - 1) > 5e-4  # the flat area is off by 9e-4


# An on annulus: no star, no area inside its inner radius
def test_object_regions_on_annulus(stars, target):
    regions = _regions(stars, target, on_radii=(1.0, 2.0))
    assert (regions["elliptical_radius"][regions["on"]] >= 1.0).all()
    assert regions["on_area_deg2"] == pytest.approx(_ellipse_area(2.0) - _ellipse_area(1.0), rel=0.01)


# A footprint with about the west half of the field out: its stars excluded; the area fractions those of the
# uniform stars kept (within their Poisson noise), whatever the shape of the pixels
def test_object_regions_footprint(stars, target):
    nside = 1024
    lon, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    footprint_mask = lon < CENTRE[0]
    regions = _regions(stars, target, footprint_mask=footprint_mask, masked_object_types=())
    full = _regions(stars, target, masked_object_types=())
    pixels = hp.ang2pix(nside, stars["ra"].to_numpy(), stars["dec"].to_numpy(), lonlat=True)
    np.testing.assert_array_equal(regions["excluded"], footprint_mask[pixels])
    assert regions["masked_objects"].empty
    for region in ("on", "off"):
        n_full = full[region].sum()
        kept = regions[region].sum() / n_full
        assert 0.3 < kept < 0.7
        assert regions[f"{region}_area_fraction"] == pytest.approx(kept, abs=4 / np.sqrt(n_full))
        assert regions[f"{region}_area_deg2"] == pytest.approx(
            regions[f"{region}_area_fraction"] * full[f"{region}_area_deg2"])


# Default footprint from the stars: all the field is in it
def test_object_regions_default_footprint(stars, target):
    regions = object_regions(stars, target, on_radii=ON_RADII, off_radii=OFF_RADII, masked_object_types=())
    assert regions["on_area_fraction"] == pytest.approx(1.0)
    assert regions["off_area_fraction"] == pytest.approx(1.0)


# The object is never masked: its entry in the atlas (centre 20 arcsec away, as large) and an entry of the same
# name are recognised; a small background galaxy inside it, and a large galaxy whose mask covers its centre, are
# masked, the latter with a warning
def test_object_regions_same_object(stars, target, monkeypatch):
    centre = SkyCoord(CENTRE[0] * u.deg, CENTRE[1] * u.deg)
    near = centre.directional_offset_by(0.0 * u.deg, 20.0 * u.arcsec)
    inside = centre.directional_offset_by(120.0 * u.deg, 4.0 * u.arcmin)
    galaxies = pd.DataFrame([
        _object("sga_1", near.ra.deg, near.dec.deg, 4.0),  # the dwarf in the atlas
        _object("sga_2", inside.ra.deg, inside.dec.deg, 0.3),  # a small background galaxy in the dwarf
        {**_object("sga_3", CENTRE[0] + 0.3, CENTRE[1] - 0.25, 1.0), "name": "Dwarf-A"},  # same name, elsewhere
    ])
    monkeypatch.setattr(overlapped_objects, "get_galaxies", lambda: galaxies)
    regions = _regions(stars, target, masked_object_types=("galaxies",))
    assert sorted(regions["same_object_entries"]["key"]) == ["sga_1", "sga_3"]
    assert regions["masked_objects"]["key"].tolist() == ["sga_2"]
    galaxy_mask_area = np.pi * (mask_radius_scales(galaxies.iloc[1:2], "galaxies")[0] * 0.3 / 60.0) ** 2
    assert regions["on_area_fraction"] == pytest.approx(1 - galaxy_mask_area / _ellipse_area(ON_RADII[1]), rel=0.01)

    host = centre.directional_offset_by(90.0 * u.deg, 30.0 * u.arcmin)
    monkeypatch.setattr(overlapped_objects, "get_galaxies", lambda: pd.DataFrame([_object("sga_4", host.ra.deg,
                                                                                         host.dec.deg, 8.0)]))
    with pytest.warns(UserWarning, match="SGA_4"):
        regions = _regions(stars, target, masked_object_types=("galaxies",))
    assert regions["on_area_fraction"] == 0.0


def test_object_regions_errors(stars, target):
    with pytest.raises(ValueError, match="on_radii"):
        _regions(stars, target, on_radii=(0.0, 6.0))  # overlaps the off annulus
    no_radius = target.copy()
    no_radius["radius_arcmin"] = np.nan
    with pytest.raises(ValueError, match="half-light"):
        _regions(stars, no_radius)


# The map: the stars not excluded in the window, binned; the bins whose centre is on the masked cluster masked;
# by default 60 bins across the field
def test_object_density_map(stars, target):
    regions = _regions(stars, target)
    bin_size = 0.02
    density_map = region_density_map(regions, bin_size_deg=bin_size)
    phi1_centres, phi2_centres = phi12_bin_centres(regions["window"], bin_size)
    np.testing.assert_array_equal(density_map["phi1_centres"], phi1_centres)
    density = density_map["density"]
    assert density.shape == (phi1_centres.size, phi2_centres.size)
    ra, dec = phi12_to_radec(*np.meshgrid(phi1_centres, phi2_centres, indexing="ij"), regions["stream_frame"])
    near_gc = np.degrees(angular_separation(*np.radians(_near_gc_position()), np.radians(ra), np.radians(dec)))
    near_gc_object = pd.DataFrame([{"radius_arcmin": NEAR_GC_RHALF_ARCMIN}])
    on_gc = near_gc <= mask_radius_scales(near_gc_object, "gc")[0] * NEAR_GC_RHALF_ARCMIN / 60.0
    assert on_gc.any()
    np.testing.assert_array_equal(np.ma.getmaskarray(density), on_gc)
    half = bin_size / 2
    in_bins = ((regions["phi1"] >= phi1_centres[0] - half) & (regions["phi1"] < phi1_centres[-1] + half)
               & (regions["phi2"] >= phi2_centres[0] - half) & (regions["phi2"] < phi2_centres[-1] + half))
    counted = (in_bins & ~regions["excluded"]).sum()
    # stars of the masked bins not counted: those of the bins around the cluster's mask
    assert density.sum() <= counted and density.sum() > 0.99 * counted
    selected = regions["on"]
    assert region_density_map(regions, selected, bin_size_deg=bin_size)["density"].sum() == regions["on"].sum()
    assert region_density_map(regions)["bin_size_deg"] == pytest.approx(2 * OFF_RADII[1] * RHALF_ARCMIN / 60.0 / 60)

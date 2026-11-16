"""Tests of analysis/gc_dwarf_optimization.py, with a mock LVDB and uniform mock stars."""

import inspect

import astropy.units as u
import healpy as hp
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord, angular_separation

from streameucliddes.analysis import gc_dwarf_optimization
from streameucliddes.analysis.build_masks import DEFAULT_RADIUS_SCALES, elliptical_radius_deg
from streameucliddes.analysis.catalog_preparation import prepare_match_filter_catalog
from streameucliddes.analysis.cmd_match_filter import match_filter_polygons, select_stars_in_match_filter
from streameucliddes.analysis.gc_dwarf_optimization import (
    compare_selections,
    get_object,
    known_objects,
    match_filter_cmd_bins,
    match_filter_color_mag_pairs,
    object_field_radius_deg,
    object_hess_diagrams,
    object_isochrone,
    object_radec_box,
    object_regions,
    objects_near_distance_modulus,
    selection_snr,
    stars_in_match_filter,
    stars_passing_cuts,
)
from streameucliddes.catalogs import overlapped_objects
from streameucliddes.catalogs.isochrone_parameters import default_isochrone_grid
from streameucliddes.unit_conversions import convert_FeH_to_z

pytestmark = pytest.mark.gc_dwarf_optimization

LVDB_COLUMNS = ["key", "name", "ra", "dec", "rhalf", "ellipticity", "position_angle", "distance", "distance_modulus",
                "metallicity", "confirmed_real"]
# The target: an elongated dwarf galaxy; a globular cluster in its off annulus, along the major axis
CENTRE = (50.0, -30.0)
RHALF_ARCMIN, ELLIPTICITY, POSITION_ANGLE = 6.0, 0.5, 30.0
ON_RADII, OFF_RADII = (0.0, 2.0), (5.0, 10.0)
NEAR_GC_OFFSET_ARCMIN, NEAR_GC_RHALF_ARCMIN = 45.0, 0.5
VIS, H = "euclid_q1_VIS", "euclid_q1_H"
VIS_H = {"name": "vis_h", "reference_band": ("euclid_q1", "VIS"), "color_bands": [("euclid_q1", "H")]}


def _near_gc_position():
    """(ra, dec) of the globular cluster in the off annulus of the target."""
    centre = SkyCoord(CENTRE[0] * u.deg, CENTRE[1] * u.deg)
    position = centre.directional_offset_by(POSITION_ANGLE * u.deg, NEAR_GC_OFFSET_ARCMIN * u.arcmin)
    return position.ra.deg, position.dec.deg


@pytest.fixture
def lvdb(tmp_path, monkeypatch):
    """Mock LVDB tables: the target dwarf, the LMC, a dwarf without [Fe/H] nor distance modulus, two clusters."""
    gc_ra, gc_dec = _near_gc_position()
    dwarfs = pd.DataFrame([
        ["dwarf_a", "Dwarf A", *CENTRE, RHALF_ARCMIN, ELLIPTICITY, POSITION_ANGLE, 100.0, 20.0, -2.0, 1],
        ["lmc", "LMC", 80.9, -69.8, 200.0, 0.1, 230.0, 50.0, 18.5, -0.5, 1],
        ["dwarf_b", "Dwarf B", 200.0, 10.0, 3.0, np.nan, np.nan, 250.0, np.nan, np.nan, 1],
        ["dwarf_fake", "Dwarf Fake", 10.0, 10.0, 3.0, np.nan, np.nan, 100.0, 20.0, np.nan, 0],
    ], columns=LVDB_COLUMNS)
    clusters = pd.DataFrame([
        ["ngc_1", "NGC 1", 120.0, -40.0, 1.0, np.nan, np.nan, 10.0, 15.0, -1.5, 1],
        ["gc_near", "GC Near", gc_ra, gc_dec, NEAR_GC_RHALF_ARCMIN, np.nan, np.nan, 20.0, 16.5, -1.0, 1],
    ], columns=LVDB_COLUMNS)
    folder = tmp_path / overlapped_objects.LVDB_VERSION
    folder.mkdir()
    tables = {overlapped_objects.DWARF_TABLE: dwarfs, "gc_harris": clusters}
    for table in (overlapped_objects.DWARF_TABLE, *overlapped_objects.GLOBULAR_CLUSTER_TABLES):
        tables.get(table, pd.DataFrame(columns=LVDB_COLUMNS)).to_csv(folder / f"{table}.csv", index=False)
    monkeypatch.setattr(overlapped_objects, "LVDB_DIR", tmp_path)


def _uniform_stars(rng, centre, radius_deg, n):
    """Stars uniform on the sphere within radius_deg of centre, with VIS and H magnitudes."""
    separation = np.degrees(np.arccos(1.0 - rng.random(n) * (1.0 - np.cos(np.radians(radius_deg)))))
    angle = rng.uniform(0.0, 360.0, n)
    stars = SkyCoord(centre[0] * u.deg, centre[1] * u.deg).directional_offset_by(angle * u.deg, separation * u.deg)
    vis = rng.uniform(18.0, 27.0, n)
    return pd.DataFrame({
        "ra": stars.ra.deg, "dec": stars.dec.deg, f"{VIS}_obs": vis, f"{H}_obs": vis - rng.uniform(-1.0, 1.5, n),
        f"{VIS}_err": rng.uniform(0.0, 0.3, n), f"{H}_err": rng.uniform(0.0, 0.3, n),
    })


@pytest.fixture
def stars():
    """Uniform stars over the field of the target and beyond."""
    return _uniform_stars(np.random.default_rng(1), CENTRE, 1.2, 200_000)


@pytest.fixture
def target(lvdb):
    return get_object("dwarf_a")


def _no_footprint_hole():
    return np.zeros(hp.nside2npix(64), dtype=bool)


def _regions(stars, target, **kwargs):
    kwargs = {"on_radii": ON_RADII, "off_radii": OFF_RADII, "masked_object_types": ("dwarfs", "gc"),
              "footprint_mask": _no_footprint_hole(), **kwargs}
    return object_regions(stars, target, **kwargs)


# ===================================================
# Known objects
# ===================================================


# Distance modulus and [Fe/H] from the LVDB, the distance modulus from the distance when missing; only confirmed
# objects, of the types asked
def test_known_objects(lvdb):
    objects = known_objects(("dwarfs", "gc")).set_index("key")
    assert sorted(objects.index) == ["dwarf_a", "dwarf_b", "gc_near", "ngc_1"]
    assert objects.loc["dwarf_a", ["type", "distance_modulus", "feh"]].tolist() == ["dwarfs", 20.0, -2.0]
    assert objects.loc["ngc_1", "type"] == "gc"
    assert objects.loc["dwarf_b", "distance_modulus"] == pytest.approx(5 * np.log10(250.0e3) - 5)
    assert np.isnan(objects.loc["dwarf_b", "feh"])
    assert known_objects(("lmc",))["key"].tolist() == ["lmc"]
    with pytest.raises(ValueError, match="not in"):
        known_objects(("galaxies",))


# Within max_offset, sorted by |offset|; in_catalog from the point map of the catalog
def test_objects_near_distance_modulus(lvdb, monkeypatch):
    near = objects_near_distance_modulus(16.0, 2.0)
    assert near["key"].tolist() == ["gc_near", "ngc_1"]  # offsets 0.5 and -1.0
    assert near["dm_offset"].tolist() == [0.5, -1.0]
    assert "in_catalog" not in near

    nside = 16
    counts = np.zeros(hp.nside2npix(nside))
    counts[hp.ang2pix(nside, 120.0, -40.0, nest=True, lonlat=True)] = 7  # only NGC 1 in the catalog
    monkeypatch.setattr(gc_dwarf_optimization, "catalog_point_map", lambda catalog, data_path=None: counts)
    near = objects_near_distance_modulus(16.0, 2.0, catalog="euclid_dr1").set_index("key")
    assert near["n_sources_at_centre"].to_dict() == {"gc_near": 0, "ngc_1": 7}
    assert near["in_catalog"].to_dict() == {"gc_near": False, "ngc_1": True}


# By key or name, whatever the case, spaces, underscores and hyphens
@pytest.mark.parametrize("name", ["dwarf_a", "Dwarf A", "DWARF-A", "dwarfa"])
def test_get_object(lvdb, name):
    assert get_object(name)["key"] == "dwarf_a"


def test_get_object_unknown(lvdb):
    with pytest.raises(KeyError, match="Dwarf A"):  # a close name
        get_object("Dwarf C")
    with pytest.raises(KeyError):
        get_object("dwarf_fake")  # not confirmed


# Z from [Fe/H]; the default grid when [Fe/H] is unknown
def test_object_isochrone(lvdb):
    default = default_isochrone_grid()[0]
    assert object_isochrone(get_object("dwarf_a")) == {
        "age": default["age"], "metallicity": pytest.approx(float(convert_FeH_to_z(-2.0)))}
    assert object_isochrone(get_object("dwarf_b"), age=11.0) == {"age": 11.0, "metallicity": default["metallicity"]}


# The box holds every point within the radius, also across ra = 0
@pytest.mark.parametrize("centre", [CENTRE, (0.2, -60.0), (359.5, 40.0)])
def test_object_radec_box(centre):
    obj = pd.Series({"ra": centre[0], "dec": centre[1], "radius_arcmin": 6.0})
    radius = object_field_radius_deg(obj, OFF_RADII)
    assert radius == pytest.approx(OFF_RADII[1] * 0.1)
    (ra_min, ra_max), (dec_min, dec_max) = object_radec_box(obj, radius, margin_deg=0.0)
    angles = np.linspace(0.0, 360.0, 73)
    edge = SkyCoord(centre[0] * u.deg, centre[1] * u.deg).directional_offset_by(angles * u.deg, radius * 0.999 * u.deg)
    ra, dec = edge.ra.deg, edge.dec.deg
    in_ra = (ra >= ra_min) & (ra <= ra_max) if ra_min <= ra_max else (ra >= ra_min) | (ra <= ra_max)
    assert in_ra.all() and ((dec >= dec_min) & (dec <= dec_max)).all()


def test_object_radec_box_near_pole():
    obj = pd.Series({"ra": 10.0, "dec": -89.5, "radius_arcmin": 6.0})
    ra_range, dec_range = object_radec_box(obj, 1.0, margin_deg=0.0)
    assert ra_range == (0.0, 360.0)
    assert dec_range == (-90.0, -88.5)


# ===================================================
# Regions
# ===================================================


# Elliptical radius: the separation for a circle; along the major axis the separation, along the minor axis the
# separation / (1 - ellipticity)
def test_elliptical_radius_deg():
    centre = SkyCoord(CENTRE[0] * u.deg, CENTRE[1] * u.deg)
    separation = 0.3
    points = centre.directional_offset_by([POSITION_ANGLE, POSITION_ANGLE + 90.0, 77.0] * u.deg, separation * u.deg)
    circle = elliptical_radius_deg(points.ra.deg, points.dec.deg, *CENTRE, 0.0, POSITION_ANGLE)
    assert circle == pytest.approx(separation)
    ellipse = elliptical_radius_deg(points.ra.deg, points.dec.deg, *CENTRE, ELLIPTICITY, POSITION_ANGLE)
    assert ellipse[:2] == pytest.approx([separation, separation / (1 - ELLIPTICITY)])


# On and off stars from their elliptical radius; the stars on the other object excluded; areas of the ellipses,
# less the mask of the other object
def test_object_regions(stars, target):
    regions = _regions(stars, target)
    half_light_deg = RHALF_ARCMIN / 60.0
    radius = elliptical_radius_deg(stars["ra"], stars["dec"], *CENTRE, ELLIPTICITY, POSITION_ANGLE) / half_light_deg
    np.testing.assert_allclose(regions["elliptical_radius"], radius)
    gc_ra, gc_dec = _near_gc_position()
    mask_radius_deg = DEFAULT_RADIUS_SCALES["gc"] * NEAR_GC_RHALF_ARCMIN / 60.0
    on_gc = np.degrees(angular_separation(*np.radians([gc_ra, gc_dec]),
                                          *np.radians([stars["ra"].to_numpy(), stars["dec"].to_numpy()])))
    on_gc = on_gc <= mask_radius_deg
    assert on_gc.sum() > 0
    np.testing.assert_array_equal(regions["excluded"], on_gc)
    np.testing.assert_array_equal(regions["on"], (radius <= ON_RADII[1]) & ~on_gc)
    np.testing.assert_array_equal(regions["off"], (radius >= OFF_RADII[0]) & (radius <= OFF_RADII[1]) & ~on_gc)
    assert regions["masked_objects"]["key"].tolist() == ["gc_near"]  # the target itself not masked

    def ellipse_area(radius_rhalf):
        return np.pi * (radius_rhalf * half_light_deg) ** 2 * (1 - ELLIPTICITY)

    gc_area = np.pi * mask_radius_deg**2
    assert regions["on_area_deg2"] == pytest.approx(ellipse_area(ON_RADII[1]), rel=0.01)
    assert regions["off_area_deg2"] == pytest.approx(ellipse_area(OFF_RADII[1]) - ellipse_area(OFF_RADII[0]) - gc_area,
                                                     rel=0.01)
    assert regions["on_area_fraction"] == pytest.approx(1.0)
    assert regions["off_area_fraction"] == pytest.approx(1.0 - gc_area / (ellipse_area(OFF_RADII[1])
                                                                          - ellipse_area(OFF_RADII[0])), rel=0.01)
    # uniform stars: the same density on and off, within the Poisson noise
    on_density = regions["on"].sum() / regions["on_area_deg2"]
    off_density = regions["off"].sum() / regions["off_area_deg2"]
    assert abs(on_density - off_density) < 5 * on_density / np.sqrt(regions["on"].sum())


# An on annulus: no star, no area inside its inner radius
def test_object_regions_on_annulus(stars, target):
    regions = _regions(stars, target, on_radii=(1.0, 2.0))
    assert (regions["elliptical_radius"][regions["on"]] >= 1.0).all()
    full = np.pi * (2.0 * RHALF_ARCMIN / 60.0) ** 2 * (1 - ELLIPTICITY)
    assert regions["on_area_deg2"] == pytest.approx(full * (1 - 0.25), rel=0.01)


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


def test_object_regions_errors(stars, target, lvdb):
    with pytest.raises(ValueError, match="on_radii"):
        _regions(stars, target, on_radii=(0.0, 6.0))  # overlaps the off annulus
    no_radius = target.copy()
    no_radius["radius_arcmin"] = np.nan
    with pytest.raises(ValueError, match="half-light"):
        _regions(stars, no_radius)


# ===================================================
# Selections and SNR
# ===================================================


# Background on the object: the off count scaled by the areas
def test_selection_snr():
    on = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    off = ~on
    regions = {"on": on, "off": off, "on_area_deg2": 1.0, "off_area_deg2": 7.0}
    snr = selection_snr(regions)
    assert snr["n_on"] == 5 and snr["n_off"] == 7
    assert snr["n_background_on"] == pytest.approx(1.0)
    assert snr["n_signal"] == pytest.approx(4.0)
    assert snr["snr"] == pytest.approx(4.0)
    assert snr["n_signal_error"] == pytest.approx(np.sqrt(5 + 7 / 49))
    assert snr["background_density"] == pytest.approx(1.0)
    selected = np.zeros(12, dtype=bool)
    selected[:2] = True
    assert selection_snr(regions, selected)["snr"] == np.inf  # no background
    assert selection_snr(regions, np.zeros(12, dtype=bool))["snr"] == 0.0

    table = compare_selections(regions, {"all": np.ones(12, dtype=bool), "two": selected})
    assert table.index.tolist() == ["all", "two"]
    assert table.loc["two", "n_on"] == 2


# The selection of the searches (prepare_match_filter_catalog then select_stars_in_match_filter), as a mask
def test_stars_in_match_filter(stars):
    cuts = {f"{VIS}_err": (None, 0.25)}
    prepared = prepare_match_filter_catalog(stars, VIS_H, cuts=cuts)
    expected = np.zeros(len(stars), dtype=bool)
    expected[stars.index.get_indexer(prepared.index)] = select_stars_in_match_filter(prepared, 18.0, None, VIS_H)
    selected = stars_in_match_filter(stars, 18.0, None, VIS_H, cuts=cuts)
    np.testing.assert_array_equal(selected, expected)
    assert selected.sum() > 0
    np.testing.assert_array_equal(stars_passing_cuts(stars, VIS_H, cuts),
                                  prepare_match_filter_catalog(stars, VIS_H, cuts=cuts, as_mask=True))
    bright = (stars[f"{VIS}_obs"] < 22.0).to_numpy()
    np.testing.assert_array_equal(stars_in_match_filter(stars, 18.0, None, VIS_H, cuts=cuts, mask=bright),
                                  expected & bright)


# ===================================================
# Colour-magnitude diagrams
# ===================================================


def test_match_filter_cmd_bins():
    three_bands = {**VIS_H, "color_bands": [("euclid_q1", "H"), ("euclid_q1", "J")]}
    assert match_filter_color_mag_pairs(three_bands) == [(f"{VIS}_obs", f"{H}_obs"), (f"{VIS}_obs", "euclid_q1_J_obs")]
    margin = inspect.signature(match_filter_cmd_bins).parameters["margin_mag"].default
    for (color_edges, mag_edges), polygon in zip(match_filter_cmd_bins(18.0, None, three_bands),
                                                  match_filter_polygons(18.0, None, three_bands)):
        assert color_edges[0] <= polygon[:, 0].min() - margin + 1e-9 and color_edges[-1] >= polygon[:, 0].max()
        assert mag_edges[0] <= polygon[:, 1].min() - margin + 1e-9 and mag_edges[-1] >= polygon[:, 1].max()


# On and off Hess diagrams over their areas; the difference sums to the excess of selection_snr (stars in the bins)
def test_object_hess_diagrams(stars, target):
    regions = _regions(stars, target)
    pair = (f"{VIS}_obs", f"{H}_obs")
    color_bins, mag_bins = np.linspace(-2.0, 3.0, 26), np.linspace(17.0, 28.0, 23)  # every star inside
    shown = (stars[f"{VIS}_obs"] < 25.0).to_numpy()
    on_hess, off_hess, difference = object_hess_diagrams(stars, regions, pair, stars=shown, color_bins=color_bins,
                                                         mag_bins=mag_bins)
    snr = selection_snr(regions, shown)
    assert on_hess["n_stars"] == snr["n_on"] and off_hess["n_stars"] == snr["n_off"]
    assert on_hess["area_deg2"] == regions["on_area_deg2"]
    assert difference["n_stars"] == pytest.approx(snr["n_signal"])

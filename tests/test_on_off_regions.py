"""Tests of analysis/on_off_regions.py.

Stars uniform on the sphere in the gala frame of conftest.py, and bands along a
straight or a tilted track (phi2 linear in phi1): their areas are exact integrals,
their counts in proportion.
"""

import healpy as hp
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import angular_separation
from scipy.integrate import quad

from streameucliddes.analysis.cmd_match_filter import on_off_hess
from streameucliddes.analysis.on_off_regions import (
    compare_selections,
    excluded_points,
    masked_objects_in_window,
    region_density_map,
    region_on_off_hess,
    selection_snr,
    track_band_regions,
)
from streameucliddes.catalogs import overlapped_objects
from streameucliddes.catalogs.survey_footprint import in_healpix_mask
from streameucliddes.stream_frames import phi12_bin_centres, phi12_to_radec, radec_to_phi12

pytestmark = pytest.mark.on_off_regions

SIGMA, ON_N_SIGMA, OFF_N_SIGMA = 0.2, 1.5, (3.0, 5.0)
PHI1_RANGE = (-4.0, 4.0)
SLOPE = 0.1  # tilted track: phi2 = SLOPE * phi1
N_STARS = 300_000
VIS, H = "euclid_q1_VIS_obs", "euclid_q1_H_obs"
# A masked object on the track, inside the on band: a circle of MASK_RADIUS_ARCMIN
OBJECT_PHI1, MASK_RADIUS_ARCMIN = 1.0, 12.0


def tilted_track(phi1):
    return SLOPE * np.asarray(phi1, dtype=float)


@pytest.fixture(scope="module")
def catalog(gc_frame):
    """Stars uniform on the sphere over phi1 -6 to 6, phi2 -3 to 3 deg, with magnitudes."""
    rng = np.random.default_rng(5)
    phi1 = rng.uniform(-6.0, 6.0, N_STARS)
    phi2 = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-3.0)), np.sin(np.radians(3.0)), N_STARS)))
    ra, dec = phi12_to_radec(phi1, phi2, gc_frame)
    vis = rng.uniform(18.0, 26.0, N_STARS)
    return pd.DataFrame({"ra": ra, "dec": dec, VIS: vis, H: vis - rng.uniform(-1.0, 1.5, N_STARS)})


def _bands(catalog, frame, track_phi2=tilted_track, **kwargs):
    return track_band_regions(catalog, frame, track_phi2, PHI1_RANGE, SIGMA, on_n_sigma=ON_N_SIGMA,
                              off_n_sigma=OFF_N_SIGMA, **kwargs)


@pytest.fixture(scope="module")
def bands(catalog, gc_frame):
    return _bands(catalog, gc_frame)


def _band_area_deg2(track_phi2, low, high):
    """Exact area of the band from low to high off a track over PHI1_RANGE: the integral of
    dphi1 (sin phi2_top - sin phi2_bottom), in deg²."""
    def strip(phi1):
        centre = float(track_phi2(phi1))
        return np.degrees(np.sin(np.radians(centre + high)) - np.sin(np.radians(centre + low)))
    return quad(strip, *PHI1_RANGE)[0]


def _masked_object(frame):
    """One object on the tilted track, its mask a circle of MASK_RADIUS_ARCMIN."""
    ra, dec = phi12_to_radec(np.array([OBJECT_PHI1]), tilted_track([OBJECT_PHI1]), frame)
    return pd.DataFrame({"key": ["obj"], "name": ["OBJ"], "type": ["dwarfs"], "ra": ra, "dec": dec,
                         "radius_arcmin": [MASK_RADIUS_ARCMIN / 4], "ellipticity": [np.nan],
                         "position_angle": [np.nan], "mask_radius_arcmin": [MASK_RADIUS_ARCMIN]})


# ===================================================
# Bands along a track
# ===================================================


# Each star by its phi2 offset from the track at its phi1; nothing beyond the phi1 range; the off parts on each side
@pytest.mark.parametrize("track_phi2", [0.3, tilted_track])
def test_track_band_membership(catalog, gc_frame, track_phi2):
    regions = _bands(catalog, gc_frame, track_phi2)
    phi1, phi2 = regions["phi1"], regions["phi2"]
    in_range = (phi1 >= PHI1_RANGE[0]) & (phi1 <= PHI1_RANGE[1])
    centre = tilted_track(phi1) if callable(track_phi2) else np.full(phi1.size, track_phi2)
    delta = regions["delta_phi2"]
    assert np.isnan(delta[~in_range]).all()
    np.testing.assert_allclose(delta[in_range], (phi2 - centre)[in_range])

    distance = np.abs(np.nan_to_num(delta, nan=99.0))
    np.testing.assert_array_equal(regions["on"], distance <= ON_N_SIGMA * SIGMA)
    off = (distance >= OFF_N_SIGMA[0] * SIGMA) & (distance <= OFF_N_SIGMA[1] * SIGMA)
    np.testing.assert_array_equal(regions["off"], off)
    np.testing.assert_array_equal(regions["off"], regions["off_below"] | regions["off_above"])
    assert (delta[regions["off_below"]] < 0).all() and (delta[regions["off_above"]] > 0).all()
    assert regions["off_parts"] == ("off_below", "off_above") and not regions["excluded"].any()


# Areas: the exact integrals along the tilted track; uniform stars in proportion; SNR of a uniform field about 0
def test_track_band_areas(bands):
    on_half = ON_N_SIGMA * SIGMA
    inner, outer = (n_sigma * SIGMA for n_sigma in OFF_N_SIGMA)
    assert bands["on_area_deg2"] == pytest.approx(_band_area_deg2(tilted_track, -on_half, on_half), rel=2e-3)
    assert bands["off_below_area_deg2"] == pytest.approx(_band_area_deg2(tilted_track, -outer, -inner), rel=2e-3)
    assert bands["off_above_area_deg2"] == pytest.approx(_band_area_deg2(tilted_track, inner, outer), rel=2e-3)
    assert bands["off_area_deg2"] == pytest.approx(bands["off_below_area_deg2"] + bands["off_above_area_deg2"])
    for name in ("on", "off_below", "off_above", "off"):
        assert bands[f"{name}_area_fraction"] == pytest.approx(1.0)
    density_on = bands["on"].sum() / bands["on_area_deg2"]
    assert bands["off"].sum() / bands["off_area_deg2"] == pytest.approx(density_on, rel=0.05)
    assert abs(selection_snr(bands)["snr"]) < 5


# A straight track: the band edges along the rows of the area pieces, between two rows; each piece counted for its
# fraction in the band, the areas exact (counting the pieces by their centre would be off by up to a row)
def test_track_band_areas_straight(catalog, gc_frame):
    regions = _bands(catalog, gc_frame, 0.3, area_step_deg=0.04)
    on_half = ON_N_SIGMA * SIGMA
    inner, outer = (n_sigma * SIGMA for n_sigma in OFF_N_SIGMA)
    straight = lambda phi1: 0.3  # noqa: E731
    assert regions["on_area_deg2"] == pytest.approx(_band_area_deg2(straight, -on_half, on_half), rel=1e-4)
    assert regions["off_below_area_deg2"] == pytest.approx(_band_area_deg2(straight, -outer, -inner), rel=1e-4)


# The window holds the off bands along the track; the outlines are closed and follow the track
def test_track_band_window_and_outlines(bands):
    outer = OFF_N_SIGMA[1] * SIGMA
    window = bands["window"]
    assert window[:2] == pytest.approx(PHI1_RANGE)
    assert window[2] < tilted_track(PHI1_RANGE[0]) - outer and window[3] > tilted_track(PHI1_RANGE[1]) + outer
    (on_phi1, on_phi2), = bands["outlines"]["on"]
    assert (on_phi1[0], on_phi2[0]) == (on_phi1[-1], on_phi2[-1])
    np.testing.assert_allclose(np.abs(on_phi2 - tilted_track(on_phi1)), ON_N_SIGMA * SIGMA)
    assert len(bands["outlines"]["off"]) == 2
    np.testing.assert_allclose(bands["track_phi2"], tilted_track(bands["track_phi1"]))


# Offsets move the on band and each off band, not the track
def test_track_band_offsets(catalog, gc_frame, bands):
    shifted = _bands(catalog, gc_frame, on_phi2_offset_deg=0.1, off_phi2_offsets_deg=(-0.2, 0.3))
    on_half, (inner, outer) = ON_N_SIGMA * SIGMA, (n_sigma * SIGMA for n_sigma in OFF_N_SIGMA)
    assert shifted["intervals"]["on"] == pytest.approx((-on_half + 0.1, on_half + 0.1))
    assert shifted["intervals"]["off_below"] == pytest.approx((-outer - 0.2, -inner - 0.2))
    assert shifted["intervals"]["off_above"] == pytest.approx((inner + 0.3, outer + 0.3))
    np.testing.assert_allclose(shifted["delta_phi2"], bands["delta_phi2"])
    delta = np.nan_to_num(shifted["delta_phi2"], nan=99.0)
    np.testing.assert_array_equal(shifted["on"], np.abs(delta - 0.1) <= on_half)


def test_track_band_errors(catalog, gc_frame):
    with pytest.raises(ValueError, match="sigma_deg"):
        track_band_regions(catalog, gc_frame, 0.0, PHI1_RANGE, 0.0, on_n_sigma=1.0, off_n_sigma=(2.0, 3.0))
    with pytest.raises(ValueError, match="phi1_range"):
        track_band_regions(catalog, gc_frame, 0.0, (1.0, -1.0), SIGMA, on_n_sigma=1.0, off_n_sigma=(2.0, 3.0))
    with pytest.raises(ValueError, match="off_n_sigma"):
        track_band_regions(catalog, gc_frame, 0.0, PHI1_RANGE, SIGMA, on_n_sigma=1.0, off_n_sigma=(3.0, 2.0))
    with pytest.raises(ValueError, match="overlaps the on region"):
        track_band_regions(catalog, gc_frame, 0.0, PHI1_RANGE, SIGMA, on_n_sigma=1.0, off_n_sigma=(2.0, 3.0),
                           on_phi2_offset_deg=0.3)


# ===================================================
# Excluded sky
# ===================================================


# A HEALPix hole over phi1 > 0: its stars excluded, half the areas left
def test_excluded_healpix_mask(catalog, gc_frame, bands):
    nside = 256
    ra, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    hole = radec_to_phi12(ra, dec, gc_frame)[0] > 0.0
    regions = _bands(catalog, gc_frame, excluded_healpix_mask=hole)
    np.testing.assert_array_equal(regions["excluded"], in_healpix_mask(catalog["ra"], catalog["dec"], hole,
                                                                       nest=False))
    for name in ("on", "off_below", "off_above"):
        assert regions[f"{name}_area_fraction"] == pytest.approx(0.5, abs=0.03)
        assert regions[f"{name}_area_deg2"] == pytest.approx(regions[f"{name}_area_fraction"]
                                                             * bands[f"{name}_area_deg2"])


# An object masked by its exact circle, inside the on band: its stars excluded, its area out of the on area
def test_excluded_ellipses(catalog, gc_frame, bands):
    masked = _masked_object(gc_frame)
    regions = _bands(catalog, gc_frame, masked_objects=masked)
    separation = np.degrees(angular_separation(*np.radians([masked["ra"][0], masked["dec"][0]]),
                                               *np.radians([catalog["ra"].to_numpy(), catalog["dec"].to_numpy()])))
    np.testing.assert_array_equal(regions["excluded"], separation <= MASK_RADIUS_ARCMIN / 60.0)
    np.testing.assert_array_equal(regions["excluded"], excluded_points(catalog["ra"], catalog["dec"], None, masked))
    disc = np.pi * (MASK_RADIUS_ARCMIN / 60.0) ** 2
    assert regions["on_area_deg2"] == pytest.approx(bands["on_area_deg2"] - disc, rel=2e-3)
    assert regions["off_area_deg2"] == pytest.approx(bands["off_area_deg2"])


# The objects whose mask reaches a window: inside it, or outside by less than their mask radius
def test_masked_objects_in_window(gc_frame, monkeypatch):
    window = (-2.0, 2.0, -1.0, 1.0)
    rhalf_arcmin = 6.0  # dwarfs masked to 4 half-light radii (build_masks.DEFAULT_RADIUS_SCALE): 0.4 deg
    phi1, phi2 = np.array([0.0, 2.3, 0.0, 3.0]), np.array([0.0, 0.0, 1.5, 0.0])
    ra, dec = phi12_to_radec(phi1, phi2, gc_frame)
    dwarfs = pd.DataFrame({"key": list("abcd"), "name": list("ABCD"), "ra": ra, "dec": dec,
                           "radius_arcmin": rhalf_arcmin, "ellipticity": np.nan, "position_angle": np.nan,
                           **{column: np.nan for column in ("distance_kpc", "distance_modulus", "feh", "age_gyr",
                                                            "apparent_magnitude_v")}})
    monkeypatch.setattr(overlapped_objects, "get_dwarf_galaxies", lambda: dwarfs)
    objects = masked_objects_in_window(gc_frame, window, ("dwarfs",))
    assert objects["key"].tolist() == ["a", "b"]  # c 0.5 deg above, d 1 deg beyond: out of reach
    assert objects["mask_radius_arcmin"].tolist() == pytest.approx([4 * rhalf_arcmin] * 2)
    assert masked_objects_in_window(gc_frame, window, ()).empty


# ===================================================
# SNR of selections
# ===================================================


# One off part (an annulus): the off count scaled by the areas
def test_selection_snr_one_part():
    on = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    regions = {"on": on, "off": ~on, "on_area_deg2": 1.0, "off_area_deg2": 7.0, "off_parts": ("off",)}
    snr = selection_snr(regions)
    assert (snr["n_on"], snr["n_off"], snr["n_off_parts"]) == (5, 7, 1)
    assert snr["n_background_on"] == pytest.approx(1.0)
    assert snr["n_signal"] == pytest.approx(4.0) and snr["snr"] == pytest.approx(4.0)
    assert snr["n_signal_error"] == pytest.approx(np.sqrt(5 + 7 / 49))
    assert snr["background_density"] == pytest.approx(1.0)
    selected = np.zeros(12, dtype=bool)
    selected[:2] = True
    assert selection_snr(regions, selected)["snr"] == np.inf  # no background
    assert selection_snr(regions, np.zeros(12, dtype=bool))["snr"] == 0.0
    assert selection_snr({key: value for key, value in regions.items() if key != "off_parts"})["snr"] == snr["snr"]


def _two_part_regions(above_area=1.0):
    """15 stars on (0.5 deg²); 20 below (2 deg²: density 10) and 30 above (above_area: density 30)."""
    labels = np.repeat(["on", "below", "above"], [15, 20, 30])
    return {"on": labels == "on", "off_below": labels == "below", "off_above": labels == "above",
            "off": labels != "on", "off_parts": ("off_below", "off_above"), "on_area_deg2": 0.5,
            "off_below_area_deg2": 2.0, "off_above_area_deg2": above_area, "off_area_deg2": 2.0 + above_area}


# Two off parts: the mean of their densities (20, not the pooled 50 / 3), so that a gradient across the track cancels
def test_selection_snr_two_parts():
    snr = selection_snr(_two_part_regions())
    assert snr["background_density"] == pytest.approx(20.0)
    assert snr["n_background_on"] == pytest.approx(10.0) and snr["n_signal"] == pytest.approx(5.0)
    assert snr["snr"] == pytest.approx(5.0 / np.sqrt(10.0))
    assert snr["n_signal_error"] == pytest.approx(np.sqrt(15 + (0.5 * np.sqrt(20 / 4 + 30) / 2) ** 2))
    assert (snr["n_off_below"], snr["n_off_above"], snr["n_off"], snr["n_off_parts"]) == (20, 30, 50, 2)
    alone = selection_snr(_two_part_regions(above_area=0.0))  # a part with no area: left out
    assert alone["background_density"] == pytest.approx(10.0) and alone["n_off_parts"] == 1

    table = compare_selections(_two_part_regions(), {"all": np.ones(65, dtype=bool), "on": np.arange(65) < 15})
    assert table.index.tolist() == ["all", "on"]
    assert table.loc["on", "n_on"] == 15 and table.loc["on", "n_off"] == 0


# ===================================================
# Map and Hess diagrams
# ===================================================


# The map: the stars not excluded, binned; the bins whose centre is on the masked object masked
def test_region_density_map(catalog, gc_frame):
    masked = _masked_object(gc_frame)
    regions = _bands(catalog, gc_frame, masked_objects=masked)
    bin_size = 0.05
    density_map = region_density_map(regions, bin_size_deg=bin_size)
    phi1_centres, phi2_centres = phi12_bin_centres(regions["window"], bin_size)
    assert density_map["density"].shape == (phi1_centres.size, phi2_centres.size)
    ra, dec = phi12_to_radec(*np.meshgrid(phi1_centres, phi2_centres, indexing="ij"), gc_frame)
    on_object = np.degrees(angular_separation(*np.radians([masked["ra"][0], masked["dec"][0]]), np.radians(ra),
                                              np.radians(dec))) <= MASK_RADIUS_ARCMIN / 60.0
    assert on_object.any()
    np.testing.assert_array_equal(np.ma.getmaskarray(density_map["density"]), on_object)
    assert density_map["n_selected"] == (~regions["excluded"]).sum()
    assert region_density_map(regions, regions["on"], bin_size_deg=bin_size)["n_selected"] == regions["on"].sum()
    assert region_density_map(regions)["bin_size_deg"] == pytest.approx((PHI1_RANGE[1] - PHI1_RANGE[0]) / 60)


# Hess diagrams: the off diagram the mean of those of the off parts, as the background of the SNR
def test_region_on_off_hess(catalog, bands):
    selected = (catalog[VIS] < 24.0).to_numpy()
    on_hess, off_hess, difference = region_on_off_hess(catalog, (VIS, H), bands, selected)
    expected = on_off_hess(catalog, (VIS, H), selected & bands["on"], bands["on_area_deg2"],
                           [selected & bands["off_below"], selected & bands["off_above"]],
                           [bands["off_below_area_deg2"], bands["off_above_area_deg2"]])
    for result, reference in zip((on_hess, off_hess, difference), expected):
        np.testing.assert_allclose(result["density"], reference["density"])

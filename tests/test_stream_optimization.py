"""Tests of analysis/stream_optimization.py and plotting/plot_stream_optimization.py.

A synthetic track along the gala frame of conftest.py (phi2 and distance linear in phi1)
and stars uniform on the sphere around it: exact areas, and counts in proportion.
"""

import matplotlib

matplotlib.use("Agg")

from types import SimpleNamespace

import astropy.coordinates as coord
import astropy.units as u
import healpy as hp
import numpy as np
import pandas as pd
import pytest

from streameucliddes.analysis import stream_optimization
from streameucliddes.analysis.gc_dwarf_optimization import selection_snr
from streameucliddes.analysis.stream_optimization import (
    stream_density_map,
    stream_field,
    stream_regions,
    track_match_filter_mask,
)
from streameucliddes.catalogs.galstreams_catalog import galstreams_dm_along_track, galstreams_phi2_along_track
from streameucliddes.plotting.plot_gc_dwarf_optimization import plot_object_hess
from streameucliddes.plotting.plot_stream_optimization import plot_stream_map
from streameucliddes.stream_frames import phi12_to_radec, radec_to_phi12

pytestmark = pytest.mark.stream_optimization

PHI1_ENDS, PHI2_ENDS, DISTANCE_KPC = (-5.0, 5.0), (-0.5, 0.5), (20.0, 25.0)
SIGMA = 0.2
NSIDE = 256  # footprints of the tests: pixels of 0.23 deg
N_STARS = 400_000
VIS, H = "euclid_q1_VIS", "euclid_q1_H"
MATCH_FILTER = {"name": "vis_h", "reference_band": ("euclid_q1", "VIS"), "color_bands": [("euclid_q1", "H")]}
ISOCHRONE = {"age": 12.0, "metallicity": 0.0003}


@pytest.fixture(scope="module")
def track(gc_frame):
    """Synthetic galstreams-like track: phi2 and distance linear in phi1, points every 0.01 deg."""
    phi1 = np.linspace(*PHI1_ENDS, 1001)
    points = coord.SkyCoord(phi1=phi1 * u.deg, phi2=np.linspace(*PHI2_ENDS, phi1.size) * u.deg,
                            distance=np.linspace(*DISTANCE_KPC, phi1.size) * u.kpc,
                            frame=gc_frame).transform_to(coord.ICRS())
    return SimpleNamespace(track=points, stream_frame=gc_frame, InfoFlags="1111", track_name="Synthetic-T00",
                           stream_name="Synthetic", ref="synthetic")


@pytest.fixture(scope="module")
def catalog(gc_frame):
    """Stars uniform on the sphere over phi1 -6 to 6, phi2 -3 to 3 deg; magnitudes for the match filter."""
    rng = np.random.default_rng(3)
    phi1 = rng.uniform(-6.0, 6.0, N_STARS)
    phi2 = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-3.0)), np.sin(np.radians(3.0)), N_STARS)))
    ra, dec = phi12_to_radec(phi1, phi2, gc_frame)
    vis = rng.uniform(18.0, 26.0, N_STARS)
    return pd.DataFrame({"ra": ra, "dec": dec, f"{VIS}_obs": vis, f"{H}_obs": vis - rng.uniform(-1.0, 1.5, N_STARS),
                         f"{VIS}_err": 0.02, f"{H}_err": 0.02})


def _no_footprint_hole():
    return np.zeros(hp.nside2npix(NSIDE), dtype=bool)


def _regions(catalog, track, **kwargs):
    kwargs = {"phi1_range": PHI1_ENDS, "sigma_deg": SIGMA, "masked_object_types": (),
              "footprint_mask": _no_footprint_hole(), **kwargs}
    return stream_regions(catalog, track, **kwargs)


def _band_area_deg2(edges, centres, low, high):
    """Exact area of a band following the slices: sum of dphi1 (sin phi2_top - sin phi2_bottom)."""
    widths = np.radians(np.diff(edges))
    return float(np.degrees(np.degrees(np.sum(widths * (np.sin(np.radians(centres + high))
                                                        - np.sin(np.radians(centres + low)))))))


# ===================================================
# Field
# ===================================================


# The phi1 range of the track (in the footprint when given); the window covers its off regions; the box its outline
def test_stream_field(track, gc_frame):
    field = stream_field(track, sigma_deg=SIGMA)
    assert field["phi1_range"] == pytest.approx(PHI1_ENDS, abs=1e-6)
    off_outer = stream_optimization.DEFAULT_OFF_N_SIGMA[1] * SIGMA
    assert field["window"][2] < PHI2_ENDS[0] - off_outer and field["window"][3] > PHI2_ENDS[1] + off_outer
    (ra_min, ra_max), (dec_min, dec_max) = field["ra_range"], field["dec_range"]
    ra, dec = track.track.ra.deg, track.track.dec.deg
    assert (dec >= dec_min).all() and (dec <= dec_max).all() and (ra >= ra_min).all() and (ra <= ra_max).all()

    ra_pixels, dec_pixels = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    footprint = radec_to_phi12(ra_pixels, dec_pixels, gc_frame)[0] > 0.0  # True outside: phi1 > 0
    half = stream_field(track, sigma_deg=SIGMA, footprint_mask=footprint, nest=False)
    assert half["phi1_range"][0] == pytest.approx(PHI1_ENDS[0], abs=1e-6)
    assert half["phi1_range"][1] == pytest.approx(0.0, abs=0.3)  # the track points around the edge pixels


# ===================================================
# Regions
# ===================================================


# Slices: the track phi2 and rounded distance modulus at their centres; each star by its offset from its slice
def test_stream_regions_slices(catalog, track):
    regions = _regions(catalog, track, slice_deg=0.5, dm_step=0.05)
    edges = regions["slice_edges"]
    assert edges.size == 21 and edges[[0, -1]] == pytest.approx(PHI1_ENDS)
    centres = (edges[:-1] + edges[1:]) / 2
    np.testing.assert_allclose(regions["slice_phi2"], galstreams_phi2_along_track(track, centres))
    np.testing.assert_allclose(regions["slice_dm"],
                               np.round(galstreams_dm_along_track(track, centres) / 0.05) * 0.05)

    phi1, delta = regions["phi1"], regions["delta_phi2"]
    in_range = (phi1 >= PHI1_ENDS[0]) & (phi1 <= PHI1_ENDS[1])
    assert np.isnan(delta[~in_range]).all() and np.isnan(regions["star_dm"][~in_range]).all()
    star_slice = np.clip(np.searchsorted(edges, phi1[in_range], side="right") - 1, 0, edges.size - 2)
    np.testing.assert_allclose(delta[in_range], regions["phi2"][in_range] - regions["slice_phi2"][star_slice])

    on_half = regions["on_n_sigma"] * SIGMA
    off_inner, off_outer = (n * SIGMA for n in regions["off_n_sigma"])
    np.testing.assert_array_equal(regions["on"], np.abs(np.nan_to_num(delta, nan=99.0)) <= on_half)
    distance = np.abs(np.nan_to_num(delta, nan=99.0))
    np.testing.assert_array_equal(regions["off"], (distance >= off_inner) & (distance <= off_outer))
    assert not (regions["on"] & regions["off"]).any() and not (regions["off_below"] & regions["off_above"]).any()
    assert (delta[regions["off_below"]] < 0).all() and (delta[regions["off_above"]] > 0).all()


# Areas exact without a mask; uniform stars in proportion to them; SNR of a uniform field about 0
def test_stream_regions_areas(catalog, track):
    regions = _regions(catalog, track)
    edges, centres = regions["slice_edges"], regions["slice_phi2"]
    on_half = regions["on_n_sigma"] * SIGMA
    off_inner, off_outer = (n * SIGMA for n in regions["off_n_sigma"])
    assert regions["on_area_deg2"] == pytest.approx(_band_area_deg2(edges, centres, -on_half, on_half), rel=1e-3)
    off = (_band_area_deg2(edges, centres, -off_outer, -off_inner)
           + _band_area_deg2(edges, centres, off_inner, off_outer))
    assert regions["off_area_deg2"] == pytest.approx(off, rel=1e-3)
    assert regions["on_area_fraction"] == pytest.approx(1.0) and regions["off_area_fraction"] == pytest.approx(1.0)

    density_on = regions["on"].sum() / regions["on_area_deg2"]
    density_off = regions["off"].sum() / regions["off_area_deg2"]
    assert density_on == pytest.approx(density_off, rel=0.05)
    assert abs(selection_snr(regions)["snr"]) < 5


# A footprint hole over half of the regions: their stars excluded, half of their areas left
def test_stream_regions_footprint_hole(catalog, track, gc_frame):
    ra, dec = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    footprint = radec_to_phi12(ra, dec, gc_frame)[0] > 0.0  # True outside: phi1 > 0
    regions = _regions(catalog, track, footprint_mask=footprint)
    assert regions["on_area_fraction"] == pytest.approx(0.5, abs=0.03)
    assert regions["off_area_fraction"] == pytest.approx(0.5, abs=0.03)
    pixel_deg = np.degrees(hp.nside2resol(NSIDE))
    assert (regions["phi1"][regions["on"] | regions["off"]] < pixel_deg).all()


# An offset moves the regions, not the track: the same stars as the regions of a track shifted by it, the same areas
# (the track is far from the poles of the frame), and the field window shifted too
def test_stream_regions_phi2_offset(catalog, track):
    offset = 0.6
    regions = _regions(catalog, track)
    shifted = _regions(catalog, track, phi2_offset_deg=offset)
    np.testing.assert_allclose(shifted["slice_phi2"], regions["slice_phi2"] + offset)
    np.testing.assert_allclose(shifted["delta_phi2"], regions["delta_phi2"] - offset)
    on_half = regions["on_n_sigma"] * SIGMA
    np.testing.assert_array_equal(shifted["on"], np.abs(np.nan_to_num(regions["delta_phi2"] - offset, nan=99.0)) <= on_half)
    assert shifted["on_area_deg2"] == pytest.approx(regions["on_area_deg2"], rel=0.01)
    assert shifted["phi2_offset_deg"] == offset and "+0.6 deg" in shifted["label"]
    field, field_shifted = stream_field(track, sigma_deg=SIGMA), stream_field(track, sigma_deg=SIGMA, phi2_offset_deg=offset)
    assert np.array(field_shifted["window"][2:]) == pytest.approx(np.array(field["window"][2:]) + offset)

def test_stream_regions_errors(catalog, track):
    with pytest.raises(ValueError, match="sigma_deg"):
        _regions(catalog, track, sigma_deg=0.0)
    with pytest.raises(ValueError, match="off_n_sigma"):
        _regions(catalog, track, on_n_sigma=4.0, off_n_sigma=(3.0, 6.0))
    with pytest.raises(ValueError, match="phi1_range"):
        _regions(catalog, track, phi1_range=(1.0, -1.0))
    no_distance = SimpleNamespace(**{**vars(track), "ref": "ibata2024"})
    with pytest.raises(ValueError, match="no distance"):
        _regions(catalog, no_distance)


# ===================================================
# Match filter along the track and map
# ===================================================


# One match filter per distance modulus of the slices, at that distance plus the offset; nothing outside the range
def test_track_match_filter_mask(catalog, track, monkeypatch):
    regions = _regions(catalog, track)
    calls = []

    def fake_mask(stars, distance_modulus, isochrone, match_filter, *, cuts=None, mask=None):
        calls.append((distance_modulus, np.asarray(mask)))
        return np.asarray(mask)

    monkeypatch.setattr(stream_optimization, "catalog_match_filter_mask", fake_mask)
    selected = track_match_filter_mask(catalog, regions, ISOCHRONE, MATCH_FILTER, dm_offset=0.3)
    distinct = np.unique(regions["slice_dm"])
    assert sorted(dm for dm, _ in calls) == pytest.approx(sorted(distinct + 0.3))
    groups = np.array([mask for _, mask in calls])
    assert (groups.sum(axis=0) <= 1).all()  # each star in one group at most
    np.testing.assert_array_equal(selected, np.isfinite(regions["star_dm"]))


# The real match filter: stars inside it at the distance of their slice
def test_track_match_filter_mask_real(catalog, track):
    regions = _regions(catalog, track)
    selected = track_match_filter_mask(catalog, regions, ISOCHRONE, MATCH_FILTER)
    assert 0 < selected.sum() < np.isfinite(regions["star_dm"]).sum()
    assert not selected[~np.isfinite(regions["star_dm"])].any()


def test_stream_density_map(catalog, track):
    regions = _regions(catalog, track)
    density_map = stream_density_map(regions, regions["on"], bin_size_deg=0.1)
    assert density_map["n_selected"] == regions["on"].sum()
    assert density_map["density"].shape == (density_map["phi1_centres"].size, density_map["phi2_centres"].size)


# ===================================================
# Plots
# ===================================================


# Map: the edges of the on (2) and off (4) regions slice by slice, and the track; Hess diagrams titled by the label
def test_plot_stream_map_and_hess(catalog, track):
    regions = _regions(catalog, track)
    fig, ax = plot_stream_map(regions, smoothing_deg=0.1)
    assert len(ax.patches) == 6
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert {"on", "off", "Synthetic-T00"} <= set(labels)
    fig, _ = plot_object_hess(catalog, regions, float(np.mean(regions["slice_dm"])), ISOCHRONE, MATCH_FILTER)
    assert "Synthetic" in fig._suptitle.get_text()

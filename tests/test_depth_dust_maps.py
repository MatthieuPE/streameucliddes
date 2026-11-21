"""Tests of catalogs/depth_dust_maps.py: synthetic depth maps and a fake dust query (conftest.py,
synthetic_depth_dust_maps); the real dust maps of data/dustmaps only when they are there."""

import healpy as hp
import numpy as np
import pytest

from streameucliddes.catalogs import depth_dust_maps
from streameucliddes.catalogs.depth_dust_maps import (
    DEPTH_QUANTITIES,
    DUST_MAPS,
    EBV_MAP_KEY,
    depth_at_radec,
    depth_dust_phi12_maps,
    depth_map_bands,
    depth_map_path,
    ebv_at_radec,
    read_depth_map,
)
from streameucliddes.catalogs.euclid_catalog import DUST_CORRECTIONS
from streameucliddes.stream_frames import phi12_bin_centres, phi12_to_radec, stream_frame_from_endpoints

from .conftest import DEPTH_TEST_NSIDE, DEPTH_TEST_RA_MAX, synthetic_ebv, synthetic_maglim

pytestmark = pytest.mark.depth_dust_maps


# One dust map per dust correction of the catalog magnitudes, under the same names
def test_dust_maps_of_every_dust_correction():
    assert set(DUST_MAPS) == set(DUST_CORRECTIONS)


# Bands found from the file names, VIS first; one file per band; an unknown band says which exist
def test_depth_map_bands_and_path(synthetic_depth_dust_maps):
    assert depth_map_bands() == ["VIS", "H"]
    assert depth_map_path("vis").name.endswith(f"vis_nside{DEPTH_TEST_NSIDE}.fits")
    assert depth_map_path("H") == depth_map_path("h")
    with pytest.raises(FileNotFoundError, match=r"no depth map of band 'J'.*\['VIS', 'H'\]"):
        depth_map_path("J")
    with pytest.raises(ValueError, match="Unknown depth quantity"):
        read_depth_map("VIS", "DEPTH")


# The values of the covered pixels, NaN elsewhere (pixels missing from the explicit table); read-only
def test_read_depth_map(synthetic_depth_dust_maps):
    maglim = read_depth_map("VIS")
    pixel_ra, pixel_dec = hp.pix2ang(DEPTH_TEST_NSIDE, np.arange(maglim.size), lonlat=True)
    covered = pixel_ra < DEPTH_TEST_RA_MAX
    assert maglim[covered] == pytest.approx(synthetic_maglim("VIS", pixel_ra, pixel_dec)[covered], abs=1e-5)
    assert np.isnan(maglim[~covered]).all()
    fracdet = read_depth_map("H", "FRACDET")
    assert fracdet[covered] == pytest.approx(0.5) and np.isnan(fracdet[~covered]).all()
    assert not maglim.flags.writeable


# The value of the pixel of each point; NaN outside the map and for points without coordinates
def test_depth_at_radec(synthetic_depth_dust_maps):
    ra, dec = np.array([[10.0, 200.0], [np.nan, 40.0]]), np.array([[-40.0, -40.0], [-40.0, 20.0]])
    values = depth_at_radec(ra, dec, "H")
    assert values.shape == (2, 2)
    pixel_ra, pixel_dec = hp.pix2ang(DEPTH_TEST_NSIDE, hp.ang2pix(DEPTH_TEST_NSIDE, 10.0, -40.0, lonlat=True),
                                     lonlat=True)
    assert values[0, 0] == pytest.approx(synthetic_maglim("H", pixel_ra, pixel_dec), abs=1e-5)
    assert np.isnan(values[0, 1]) and np.isnan(values[1, 0]) and np.isfinite(values[1, 1])


# The query of the dust correction at each finite point; an unknown dust correction is refused
def test_ebv_at_radec(synthetic_depth_dust_maps):
    ra, dec = np.array([70.0, 72.0, np.nan]), np.array([-40.0, -41.0, -40.0])
    ebv = ebv_at_radec(ra, dec, "SFD98")
    assert ebv[:2] == pytest.approx(synthetic_ebv(ra[:2], dec[:2])) and np.isnan(ebv[2])


def test_ebv_at_radec_unknown_dust_correction():
    with pytest.raises(ValueError, match="Unknown dust correction"):
        ebv_at_radec([70.0], [-40.0], "SFD11")


# The grid of phi12_bin_centres; each map the values at the bin centres; the depth maps of each band and quantity,
# then the dust map; no dust map with dust_correction=None, only the bands given
def test_depth_dust_phi12_maps(synthetic_depth_dust_maps, candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    window, bin_size_deg = (-6.0, 6.0, -3.0, 3.0), 0.25
    maps = depth_dust_phi12_maps(frame, window, bin_size_deg, depth_quantities=tuple(DEPTH_QUANTITIES),
                                 dust_correction="PLANCK")
    assert list(maps) == ["MAGLIM_VIS", "FRACDET_VIS", "MAGLIM_H", "FRACDET_H", EBV_MAP_KEY]
    phi1_centres, phi2_centres = phi12_bin_centres(window, bin_size_deg)
    ra, dec = phi12_to_radec(*np.meshgrid(phi1_centres, phi2_centres, indexing="ij"), frame)
    for name, map_values in maps.items():
        assert map_values["phi1_centres"] == pytest.approx(phi1_centres)
        assert map_values["phi2_centres"] == pytest.approx(phi2_centres)
        assert map_values["values"].shape == (phi1_centres.size, phi2_centres.size)
        assert map_values["bin_size_deg"] == bin_size_deg
    assert maps["MAGLIM_H"]["values"] == pytest.approx(depth_at_radec(ra, dec, "H"), nan_ok=True)
    pixel_ra, _ = hp.pix2ang(DEPTH_TEST_NSIDE, hp.ang2pix(DEPTH_TEST_NSIDE, ra, dec, lonlat=True), lonlat=True)
    outside = pixel_ra >= DEPTH_TEST_RA_MAX  # bins in a pixel not in the depth map
    assert outside.any() and (~outside).any()
    assert (np.isnan(maps["MAGLIM_VIS"]["values"]) == outside).all()
    assert maps[EBV_MAP_KEY]["values"] == pytest.approx(synthetic_ebv(ra, dec))
    assert maps[EBV_MAP_KEY]["dust_correction"] == "PLANCK" and maps[EBV_MAP_KEY]["unit"] == "mag"
    assert maps["MAGLIM_VIS"]["name"] == "VIS limiting magnitude" and maps["MAGLIM_VIS"]["band"] == "VIS"
    only_h = depth_dust_phi12_maps(frame, window, bin_size_deg, bands=["H"], dust_correction=None)
    assert list(only_h) == ["MAGLIM_H"]


# The real dust maps of data/dustmaps, when there: E(B-V) of a few hundredths at high latitude
@pytest.mark.parametrize("dust_correction", sorted(DUST_MAPS))
def test_real_dust_maps(dust_correction):
    if not DUST_MAPS[dust_correction].exists():
        pytest.skip(f"no {dust_correction} map in {DUST_MAPS[dust_correction]}")
    pytest.importorskip("dustmaps")
    ebv = ebv_at_radec(np.array([45.0, 60.0]), np.array([-48.0, -40.0]), dust_correction)
    assert ((ebv > 0.0) & (ebv < 0.1)).all()
    depth_dust_maps._dust_query.cache_clear()  # free the map

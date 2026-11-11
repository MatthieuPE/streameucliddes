"""Tests of analysis/star_density_maps.py (phi1/phi2 maps), with uniform mock stars."""

import numpy as np
import pytest

from streameucliddes.analysis.star_density_maps import star_density_phi12
from streameucliddes.stream_frames import phi12_bin_centres

pytestmark = pytest.mark.star_density_maps

WINDOW = (-10.0, 10.0, -4.0, 4.0)
BIN = 0.2
DENSITY = 200.0  # stars per deg^2: 8 per bin


@pytest.fixture
def uniform_stars(rng):
    """Stars uniform over the window and 2 deg beyond it."""
    phi1_range, phi2_range = (WINDOW[0] - 2, WINDOW[1] + 2), (WINDOW[2] - 2, WINDOW[3] + 2)
    n = rng.poisson(DENSITY * np.ptp(phi1_range) * np.ptp(phi2_range))
    return rng.uniform(*phi1_range, n), rng.uniform(*phi2_range, n)


# ===================================================
# Test star_density_phi12
# ===================================================


# No smoothing: the star counts per bin, on the bins of phi12_bin_centres; stars
# outside the window, or with NaN coordinates, are not counted
def test_counts(uniform_stars):
    phi1, phi2 = uniform_stars
    density = star_density_phi12(np.append(phi1, np.nan), np.append(phi2, 0.0), WINDOW, BIN)
    phi1_centres, phi2_centres = phi12_bin_centres(WINDOW, BIN)
    assert density.shape == (phi1_centres.size, phi2_centres.size)
    assert not density.mask.any()

    i = np.floor((phi1 - WINDOW[0]) / BIN).astype(int)
    j = np.floor((phi2 - WINDOW[2]) / BIN).astype(int)
    inside = (i >= 0) & (i < phi1_centres.size) & (j >= 0) & (j < phi2_centres.size)
    expected = np.zeros(density.shape)
    np.add.at(expected, (i[inside], j[inside]), 1)
    np.testing.assert_array_equal(density, expected)


# Smoothing: a uniform density stays uniform, at the window edges too (edge correction)
def test_smoothing_uniform(uniform_stars):
    density = star_density_phi12(*uniform_stars, WINDOW, BIN, smoothing_deg=0.6)
    expected = DENSITY * BIN**2
    edges = np.concatenate([density[0], density[-1], density[:, 0], density[:, -1]])
    assert np.mean(density) == pytest.approx(expected, rel=0.03)
    assert np.mean(edges) == pytest.approx(expected, rel=0.05)  # not halved at the edges
    assert np.std(density) < np.sqrt(expected)  # smoothed: below the Poisson noise of a bin


# Smoothing keeps a narrow overdensity at its place
def test_smoothing_peak(rng):
    phi1, phi2 = rng.normal(2.0, 0.05, 2000), rng.normal(-1.0, 0.05, 2000)
    density = star_density_phi12(phi1, phi2, WINDOW, BIN, smoothing_deg=0.5)
    phi1_centres, phi2_centres = phi12_bin_centres(WINDOW, BIN)
    i, j = np.unravel_index(np.argmax(density), density.shape)
    assert phi1_centres[i] == pytest.approx(2.0, abs=BIN) and phi2_centres[j] == pytest.approx(-1.0, abs=BIN)


# Footprint: masked outside, its stars not counted, and the density does not drop at its edge
def test_footprint(uniform_stars):
    phi1_centres, phi2_centres = phi12_bin_centres(WINDOW, BIN)
    footprint_mask = np.broadcast_to(phi2_centres > 0, (phi1_centres.size, phi2_centres.size))
    density = star_density_phi12(*uniform_stars, WINDOW, BIN, smoothing_deg=0.6, footprint_mask=footprint_mask)
    np.testing.assert_array_equal(density.mask, footprint_mask)
    assert np.isnan(density.data[footprint_mask]).all()

    expected = DENSITY * BIN**2
    edge = density[:, phi2_centres < 0][:, -1]  # last row of bins inside the footprint
    assert np.mean(edge) == pytest.approx(expected, rel=0.05)

    raw = star_density_phi12(*uniform_stars, WINDOW, BIN, footprint_mask=footprint_mask)
    assert raw.sum() == np.sum(star_density_phi12(*uniform_stars, WINDOW, BIN)[~footprint_mask])


# A footprint of another shape is refused
def test_footprint_shape(uniform_stars):
    with pytest.raises(ValueError, match="shape"):
        star_density_phi12(*uniform_stars, WINDOW, BIN, footprint_mask=np.zeros((3, 3), dtype=bool))

"""Tests of analysis/fit_stream.py: the width fit, on phi2 profiles of synthetic stars
(``analysis.candidate_investigation.phi12_density_profile``)."""

import numpy as np
import pytest

from streameucliddes.analysis.candidate_investigation import phi12_density_profile
from streameucliddes.analysis.fit_stream import DEFAULT_FIT_HALF_WIDTH_DEG, fit_stream_width, stream_width_model
from streameucliddes.catalogs.survey_footprint import unmasked_area_deg2
from streameucliddes.stream_frames import stream_frame_from_endpoints

pytestmark = pytest.mark.fit_stream


def _uniform_in_window(rng, window, n):
    """n points uniform on the sphere in a phi1/phi2 window."""
    phi1 = rng.uniform(window[0], window[1], n)
    phi2 = np.degrees(np.arcsin(rng.uniform(*np.sin(np.radians(window[2:])), n)))
    return phi1, phi2


# Uniform stars + a Gaussian stream of known centre and sigma: both recovered within 3 errors
def test_fit_recovers_centre_and_sigma(rng, candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    window = (-3.0, 3.0, -4.0, 4.0)
    phi1, phi2 = _uniform_in_window(rng, window, 20_000)
    centre, sigma = 0.4, 0.3
    phi1 = np.append(phi1, rng.uniform(*window[:2], 3000))
    phi2 = np.append(phi2, rng.normal(centre, sigma, 3000))
    profile = phi12_density_profile(phi1, phi2, [window], frame, None, axis="phi2", bin_deg=0.1)
    fit = fit_stream_width(profile, sigma_guess=0.5)
    assert abs(fit["phi2_centre"] - centre) < 3 * fit["errors"]["phi2_centre"]
    assert abs(fit["sigma"] - sigma) < 3 * fit["errors"]["sigma"]
    assert fit["model"].shape == profile["centres"].shape and 0.5 < fit["chi2"] / fit["n_dof"] < 1.5


# A field density with a gradient in phi2 (40 % over the 10 deg window) and a stream, at the statistics of
# the Euclid DR1 candidate (about 8000 field stars, a stream about as dense as the field on its track): the fit
# near the track recovers the stream; the default range is the first guess +- DEFAULT_FIT_HALF_WIDTH_DEG.
# (With far more stars, the slope left in the fitted range biases the centre beyond the errors.)
def test_fit_near_the_track_with_a_gradient(rng, candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    window = (-3.0, 3.0, -5.0, 5.0)
    phi1, phi2 = _uniform_in_window(rng, window, 10_000)
    keep = rng.uniform(0.0, 1.0, phi2.size) < 0.8 + 0.032 * phi2  # density rising with phi2: 0.64 to 0.96
    phi1, phi2 = phi1[keep], phi2[keep]
    centre, sigma = 0.2, 0.3
    phi1 = np.append(phi1, rng.uniform(*window[:2], 400))
    phi2 = np.append(phi2, rng.normal(centre, sigma, 400))
    profile = phi12_density_profile(phi1, phi2, [window], frame, None, axis="phi2", bin_deg=0.1)
    fit = fit_stream_width(profile)
    assert fit["phi2_fit_range"] == (-DEFAULT_FIT_HALF_WIDTH_DEG, DEFAULT_FIT_HALF_WIDTH_DEG)
    assert abs(fit["phi2_centre"] - centre) < 3 * fit["errors"]["phi2_centre"]
    assert abs(fit["sigma"] - sigma) < 3 * fit["errors"]["sigma"]
    narrow = fit_stream_width(profile, phi2_fit_range=(-1.0, 1.5))
    assert narrow["phi2_fit_range"] == (-1.0, 1.5) and narrow["n_dof"] == 25 - 4
    np.testing.assert_allclose(stream_width_model(fit, profile["centres"]), fit["model"])


# Many stars, a field gradient left in the fitted range: a linear background recovers the stream (a
# constant one shifts its centre beyond the errors, see fit_stream_width); the background coefficients
# are those of a polynomial in phi2 - background_reference, constant first
def test_fit_linear_background(rng, candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    window = (-3.0, 3.0, -5.0, 5.0)
    phi1, phi2 = _uniform_in_window(rng, window, 60_000)
    keep = rng.uniform(0.0, 1.0, phi2.size) < 0.8 + 0.032 * phi2
    phi1, phi2 = phi1[keep], phi2[keep]
    centre, sigma = 0.2, 0.3
    phi1 = np.append(phi1, rng.uniform(*window[:2], 3000))
    phi2 = np.append(phi2, rng.normal(centre, sigma, 3000))
    profile = phi12_density_profile(phi1, phi2, [window], frame, None, axis="phi2", bin_deg=0.1)
    fit = fit_stream_width(profile, background_degree=1)
    assert abs(fit["phi2_centre"] - centre) < 3 * fit["errors"]["phi2_centre"]
    assert abs(fit["sigma"] - sigma) < 3 * fit["errors"]["sigma"]
    assert fit["background_degree"] == 1 and len(fit["background_coefficients"]) == 2
    assert fit["background_reference"] == 0.0 and fit["background"] == fit["background_coefficients"][0]
    field = 60_000 / unmasked_area_deg2(frame, window[:2], window[2:], None)[0, 0]  # stars/deg² before thinning
    assert fit["background_coefficients"][1] == pytest.approx(0.032 * field, rel=0.3)
    assert len(fit_stream_width(profile)["background_coefficients"]) == 1
    with pytest.raises(ValueError, match="background_degree"):
        fit_stream_width(profile, background_degree=-1)


# A profile along phi1: ValueError
def test_fit_rejects_phi1_profile(rng, candidate_endpoints):
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    phi1, phi2 = _uniform_in_window(rng, (-3.0, 3.0, -1.0, 1.0), 1000)
    profile = phi12_density_profile(phi1, phi2, [(-3.0, 3.0, -1.0, 1.0)], frame, None, axis="phi1", bin_deg=0.5)
    with pytest.raises(ValueError, match="along phi2"):
        fit_stream_width(profile)

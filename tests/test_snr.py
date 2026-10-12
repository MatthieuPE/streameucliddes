"""Tests of the stream SNR estimate (data_manipulation.get_stream_snr).

The Euclid x DES stars are replaced by a mock background with the same columns as
get_eucliddes_stars(), uniform on the sky around the default stream and uniform in
colour-magnitude space, so the tests neither need the real catalog nor depend on it.
"""

import copy

import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc
import numpy as np
import pandas as pd
import pytest

from streameucliddes.analysis import data_manipulation

SEED = 42

N_BACKGROUND = 20_000
RA_RANGE = (50.0, 72.0)  # deg, around the default stream (ra 55 -> 67)
DEC_RANGE = (-53.0, -43.0)  # deg, around the default stream (dec -49 -> -47)
MAG_G_RANGE = (17.0, 27.5)

MATCH_FILTER_LIST = [
    {
        "name": "default_des_gr",
        "reference_band": ("des_yr6", "g"),
        "color_bands": [("des_yr6", "r")],
    },
    {
        "name": "des_gri",
        "reference_band": ("des_yr6", "g"),
        "color_bands": [("des_yr6", "r"), ("des_yr6", "i")],
    },
    {
        "name": "eucliddes_VISg",
        "reference_band": ("euclid_q1", "VIS"),
        "color_bands": [("des_yr6", "g")],
    },
]
DEFAULT_MATCH_FILTER = MATCH_FILTER_LIST[0]["name"]


def _mag_err(mag):
    """Photometric error growing with magnitude (0.01 at mag 23, 0.25 at mag 26.5)."""
    return 0.01 * 10 ** (0.4 * (mag - 23.0))


@pytest.fixture
def mock_background(monkeypatch):
    """Replace the Euclid x DES stars used by get_stream_snr with a mock catalog."""
    rng = np.random.default_rng(SEED)
    n = N_BACKGROUND
    sin_dec = np.sin(np.radians(DEC_RANGE))

    g = rng.uniform(*MAG_G_RANGE, n)
    r = g - rng.uniform(-0.5, 1.5, n)
    i = r - rng.uniform(-0.5, 1.0, n)
    vis = r - rng.uniform(-0.3, 0.7, n)
    y = vis - rng.uniform(0.0, 1.0, n)
    j = y - rng.uniform(-0.2, 0.6, n)
    mags = {
        "des_yr6_g": g,
        "des_yr6_r": r,
        "des_yr6_i": i,
        "euclid_q1_VIS": vis,
        "euclid_q1_Y": y,
        "euclid_q1_J": j,
    }

    columns = {
        "ra": rng.uniform(*RA_RANGE, n),
        "dec": np.degrees(np.arcsin(rng.uniform(*sin_dec, n))),
    }
    for band, mag in mags.items():
        columns[f"{band}_obs"] = mag
        columns[f"{band}_err"] = _mag_err(mag)
    background = pd.DataFrame(columns)

    monkeypatch.setattr(data_manipulation, "STARS_BACKGROUND", background)
    return background


def _snr(match_filter=DEFAULT_MATCH_FILTER, **kwargs):
    """SNR of the stream for one match filter of MATCH_FILTER_LIST."""
    snr_results, _ = data_manipulation.get_stream_snr(
        seed=SEED, match_filter_list=MATCH_FILTER_LIST, **kwargs
    )
    return snr_results[match_filter]


# ===================================================
# Test _compute_SNR
# ===================================================


# SNR = number of stream stars / sqrt(number of background stars), within the mask
def test_compute_snr():
    data = pd.DataFrame({"x": np.arange(10)})
    flag_stream = np.arange(10) < 2  # 2 stream stars, 8 background stars

    snr = data_manipulation._compute_SNR(data, flag_stream=flag_stream)
    assert snr == pytest.approx(2 / np.sqrt(8))

    mask = np.arange(10) % 2 == 0  # keeps 1 stream star and 4 background stars
    snr = data_manipulation._compute_SNR(data, mask=mask, flag_stream=flag_stream)
    assert snr == pytest.approx(1 / np.sqrt(4))

    # No background left: infinite SNR rather than a division by zero
    snr = data_manipulation._compute_SNR(
        data, mask=flag_stream, flag_stream=flag_stream
    )
    assert snr == np.inf

    with pytest.raises(ValueError):
        data_manipulation._compute_SNR(data)


# ===================================================
# Test get_stream_snr
# ===================================================


@pytest.mark.usefixtures("mock_background")
def test_update_kwargs_snr():
    """Verify that giving new kwargs updates the stream_config correctly."""
    default_config = copy.deepcopy(data_manipulation.STREAM_CONFIG)
    end1 = coord.SkyCoord(ra=66 * u.deg, dec=-47 * u.deg)
    end2 = coord.SkyCoord(ra=56 * u.deg, dec=-49 * u.deg)
    gc_frame = gc.GreatCircleICRSFrame.from_endpoints(end1, end2)

    _, stream_config = data_manipulation.get_stream_snr(
        seed=SEED,
        surface_brightness=32.0,
        stream_width=0.3,
        distance_modulus={"center": {"type": "Constant", "value": 17.5}},
        isochrone={"age": 10.0},
        gc_frame=gc_frame,
    )

    # Top-level values are replaced
    assert stream_config["surface_brightness"] == 32.0
    assert stream_config["stream_width"] == 0.3
    assert stream_config["gc_frame"] is gc_frame
    # Nested dictionaries are updated key by key, the other keys keep their default
    assert stream_config["distance_modulus"]["center"]["value"] == 17.5
    assert (
        stream_config["distance_modulus"]["spread"]
        == default_config["distance_modulus"]["spread"]
    )
    assert stream_config["isochrone"]["age"] == 10.0
    assert stream_config["isochrone"]["z"] == default_config["isochrone"]["z"]
    # Values not given keep their default
    assert stream_config["stream_length"] == default_config["stream_length"]

    # The update does not leak into the next call
    _, next_config = data_manipulation.get_stream_snr(seed=SEED)
    assert {key: next_config[key] for key in default_config} == default_config


@pytest.mark.usefixtures("mock_background")
def test_snr_surface_brightness():
    """Verify that the SNR logic works as expected: gradient with surface brightness."""
    # 3 different surface brightnesses should leads to larger SNR for the brightest stream, and smaller SNR for the faintest stream.
    snr = [_snr(surface_brightness=sb) for sb in (32.0, 33.5, 35.0)]

    assert snr[0] > snr[1] > snr[2] > 0


@pytest.mark.usefixtures("mock_background")
def test_snr_distance():
    """Verify that the SNR logic works as expected: gradient with distance modulus."""
    # 3 different distance moduli should leads to larger SNR for the closest stream, and smaller SNR for the farthest stream.
    # At a fixed number of stars: at fixed surface brightness instead, the stream holds
    # more stars when farther (luminosity ~ d^2) and the SNR increases with distance.
    snr = [
        _snr(
            N=5000,
            surface_brightness=None,
            distance_modulus={"center": {"type": "Constant", "value": dm}},
        )
        for dm in (16.0, 17.0, 18.0)
    ]

    assert snr[0] > snr[1] > snr[2] > 0


@pytest.mark.usefixtures("mock_background")
def test_snr_different_filters():
    """Verify that the SNR logic works as expected: different match filters."""
    # 2 different match filters should leads to different SNR values.
    snr_results, _ = data_manipulation.get_stream_snr(
        seed=SEED, match_filter_list=MATCH_FILTER_LIST
    )

    assert list(snr_results) == [mf["name"] for mf in MATCH_FILTER_LIST]
    for snr in snr_results.values():
        assert np.isfinite(snr) and snr > 0
    assert len(set(snr_results.values())) == len(MATCH_FILTER_LIST)

    # Each match filter is applied on its own: its SNR does not depend on the
    # other match filters of the list (e.g. on their error cuts)
    for match_filter in MATCH_FILTER_LIST:
        alone, _ = data_manipulation.get_stream_snr(
            seed=SEED, match_filter_list=[match_filter]
        )
        assert alone[match_filter["name"]] == snr_results[match_filter["name"]]


@pytest.mark.usefixtures("mock_background")
def test_cuts_update():
    """Verify that the cuts are updated correctly when new kwargs are provided."""
    # A phi2 window away from the stream (width 0.2 deg) replaces the default
    # one around it: no stream star is left
    cuts = {"phi2": (2.0, 3.0)}
    given_cuts = copy.deepcopy(cuts)
    assert _snr(cuts=cuts) == 0
    # The default cuts are added without modifying the given dictionary
    assert cuts == given_cuts

    # A given error cut replaces the default one (err <= 0.2)
    assert _snr(cuts={"des_yr6_g_err": (None, 0.02)}) != _snr()

import numpy as np
import pandas as pd
import pytest

from streameucliddes.injection import generate, inject, inject_utils

ISOCHRONE = {
    "name": "Marigo2017",
    "age": 12.0,  # Gyr
    "z": 0.0006,  # metallicity
    "surveys": ["des", "euclid"],
}

ISOCHRONE_SINGlE_SURVEY = {
    "name": "Marigo2017",
    "survey": "des",
    "age": 12.0,  # Gyr
    "z": 0.0006,  # metallicity
    "band_1": "g",
    "band_2": "r",
    "band_1_detection": True,
}

DISTANCE_MODULUS = {
    "center": {"type": "Constant", "value": 16.8},
    "spread": {"type": "Constant", "value": 0.0},
}

STREAM_CONFIG = {
    "stream_length": 10.0,  # degrees
    "stream_width": 0.5,  # degrees
    "N": 1000,  # Number of stars
    "surface_brightness": None,
    "isochrone": ISOCHRONE,
    "distance_modulus": DISTANCE_MODULUS,
}

STREAM_CONFIG_SINGLE_SURVEY = STREAM_CONFIG.copy()
STREAM_CONFIG_SINGLE_SURVEY["isochrone"] = ISOCHRONE_SINGlE_SURVEY


# ===================================================
# Test injection utils functions
# ===================================================
# Test convert_N_SurfaceBrightness function
def test_convert_N_SurfaceBrightness():
    N = 1000
    mag_bounds = (None, 24)
    stream_length = 10.0  # degrees
    stream_width = 0.5  # degrees
    isochrone = ISOCHRONE
    distance_modulus = DISTANCE_MODULUS

    surface_brightness = inject_utils.convert_N_to_SurfaceBrightness(
        N,
        mag_bounds=mag_bounds,
        stream_length=stream_length,
        stream_width=stream_width,
        isochrone=isochrone,
        band="r",
        verbose=True,
        distance_modulus=distance_modulus,
    )

    assert isinstance(
        surface_brightness, float
    ), "Surface brightness should be a float."
    assert (
        surface_brightness > 32
    ), "Surface brightness should be less than 32 mag/arcsec^2 for this test case."

    surface_brightness_2 = inject_utils.convert_N_to_SurfaceBrightness(
        N * 2,
        mag_bounds=mag_bounds,
        stream_length=stream_length,
        stream_width=stream_width,
        isochrone=isochrone,
        distance_modulus=distance_modulus,
        band="r",
        verbose=True,
    )

    assert (
        surface_brightness_2 < surface_brightness
    ), "Surface brightness should decrease with increasing N."


# ===================================================
# Test generate functions
# ===================================================


def test_generate_uniform_stream():

    catalog = generate.generate_uniform_stream(
        STREAM_CONFIG_SINGLE_SURVEY, seed=42, complete=True
    )

    assert catalog is not None
    assert len(catalog["phi1"]) == STREAM_CONFIG_SINGLE_SURVEY["N"]
    assert len(catalog["phi2"]) == STREAM_CONFIG_SINGLE_SURVEY["N"]
    assert len(catalog) == STREAM_CONFIG_SINGLE_SURVEY["N"]

    config_stream_sb = STREAM_CONFIG_SINGLE_SURVEY.copy()
    config_stream_sb["N"] = None
    config_stream_sb["surface_brightness"] = 32.0
    catalog_sb = generate.generate_uniform_stream(config_stream_sb, seed=42)
    assert (
        config_stream_sb["surface_brightness"] == 32.0
    ), "The stream config should not be modified by the generation."
    assert catalog_sb is not None
    assert len(catalog_sb["phi1"]) > 0
    assert len(catalog_sb["phi2"]) > 0
    assert len(catalog_sb) > 0

    N_sb = len(catalog_sb["phi1"])
    config_stream_N = STREAM_CONFIG_SINGLE_SURVEY.copy()
    config_stream_N["N"] = N_sb
    config_stream_N["surface_brightness"] = None
    catalog_N = generate.generate_uniform_stream(config_stream_N, seed=42)
    assert (
        len(catalog_N["phi1"]) == N_sb
    ), "Catalog generated with N should have the same number of stars as catalog generated with surface brightness."

    # Verify output is full
    needed_columns = ["phi1", "phi2", "des_g_true", "des_r_true", "mass", "dist"]
    for col in needed_columns:
        assert col in catalog, f"Column {col} is missing from the generated catalog."

    # Verify that it does not contain useless columns
    optional_col = []
    assert all(
        (col in needed_columns) | (col in optional_col) for col in catalog.keys()
    ), "Catalog contains unexpected columns."

    # Verify if it contains nan values
    for col in catalog.keys():
        assert not np.any(np.isnan(catalog[col])), f"Column {col} contains NaN values."


# ===================================================
# Test injection functions
# ===================================================


def verify_injected_catalog(injected_catalog, metadata):
    assert injected_catalog is not None, "Injected catalog should not be None."
    assert metadata is not None, "Metadata should not be None."
    assert "gc_frame" in metadata, "Metadata should contain 'gc_frame'."
    assert "seed" in metadata, "Metadata should contain 'seed'."
    assert isinstance(
        metadata["seed"], (int, type(None))
    ), "'seed' in metadata should be an int or None."


def test_inject_stream():
    # Test injection with a generated catalog
    catalog = generate.generate_uniform_stream(STREAM_CONFIG, seed=42)
    injected_catalog, metadata = inject.inject_stream(
        stream_catalog=catalog, seed=42, stream_config=STREAM_CONFIG
    )

    verify_injected_catalog(injected_catalog, metadata)

    # Test injection with the stream config only (catalog generated internally)
    injected_catalog_2, metadata_2 = inject.inject_stream(
        stream_config=STREAM_CONFIG, seed=42
    )
    verify_injected_catalog(injected_catalog_2, metadata_2)

    # Compare the two injected catalogs
    assert len(injected_catalog) == len(
        injected_catalog_2
    ), "Injected catalogs should have the same number of stars when using the same seed."
    # *_obs columns mix floats and the "BAD_MAG" string, hence a pandas comparison
    pd.testing.assert_frame_equal(injected_catalog, injected_catalog_2)
    assert metadata["seed"] == metadata_2["seed"], "Metadata seeds should match."
    assert (
        metadata["gc_frame"] == metadata_2["gc_frame"]
    ), "Metadata gc_frame should match."


def test_inject_stream_output_content():
    survey = [
        {"survey": "des", "release": "yr6"},
        {"survey": "euclid", "release": "q1"},
    ]
    bands = {"des_yr6": ["g", "r", "i"], "euclid_q1": ["VIS", "Y", "J"]}
    injected_catalog, metadata = inject.inject_stream(
        stream_config=STREAM_CONFIG, seed=42, survey=survey, bands=bands
    )

    # Verify that the injected catalog contains the expected columns for each survey
    expected_columns = ["phi1", "phi2"]
    for survey_spec in survey:
        survey_name = survey_spec["survey"]
        release = survey_spec["release"]
        namespace = f"{survey_name}_{release}"
        for band in bands[namespace]:
            expected_columns.append(f"{namespace}_{band}_obs")
            expected_columns.append(f"{namespace}_{band}_err")
            expected_columns.append(f"{survey_name}_{band}_true")
        expected_columns.append(f"{namespace}_flag_observed")

    for col in expected_columns:
        assert (
            col in injected_catalog
        ), f"Expected column {col} is missing from the injected catalog."

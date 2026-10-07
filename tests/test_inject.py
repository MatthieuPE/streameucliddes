import pytest
import numpy as np
from streameucliddes.injection import inject_utils
from streameucliddes.injection import generate


ISOCHRONE = {
            "name": "Marigo2017",
            "survey": "lsst",
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

STREAM_PARAMETERS = {
    "stream_length": 10.0,  # degrees
    "stream_width": 0.5,    # degrees
    "N": 1000,              # Number of stars
    "surface_brightness": None,
    "isochrone": ISOCHRONE,
    "distance_modulus": DISTANCE_MODULUS,
}

# ===================================================
# Test injection utils functions
# ===================================================
# Test convert_N_SurfaceBrightness function
def test_convert_N_SurfaceBrightness():
    N = 1000
    mag_bounds = (None, 24)
    stream_length = 10.0  # degrees
    stream_width = 0.5    # degrees
    isochrone = ISOCHRONE
    distance_modulus = DISTANCE_MODULUS
    
    surface_brightness = inject_utils.convert_N_to_SurfaceBrightness(N, mag_bounds=mag_bounds, stream_length=stream_length, stream_width=stream_width, isochrone=isochrone, band='r', verbose=True,distance_modulus=distance_modulus)
    
    assert isinstance(surface_brightness, float), "Surface brightness should be a float."
    assert surface_brightness > 32, "Surface brightness should be less than 32 mag/arcsec^2 for this test case."

    surface_brightness_2 = inject_utils.convert_N_to_SurfaceBrightness(N*2, mag_bounds=mag_bounds, stream_length=stream_length, stream_width=stream_width,  isochrone=isochrone, distance_modulus=distance_modulus, band='r', verbose=True)

    assert surface_brightness_2 < surface_brightness, "Surface brightness should decrease with increasing N."



# ===================================================
# Test generate functions
# ===================================================

    
def test_generate_uniform_stream():

    catalog = generate.generate_uniform_stream(STREAM_PARAMETERS, seed=42)

    assert catalog is not None
    assert len(catalog['phi1']) == STREAM_PARAMETERS['N']
    assert len(catalog['phi2']) == STREAM_PARAMETERS['N']
    assert len(catalog)==STREAM_PARAMETERS['N']

    config_stream_sb = STREAM_PARAMETERS.copy()
    config_stream_sb["N"] = None
    config_stream_sb["surface_brightness"] = 32.0
    catalog_sb = generate.generate_uniform_stream(config_stream_sb, seed=42)
    assert catalog_sb is not None
    assert len(catalog_sb['phi1']) > 0
    assert len(catalog_sb['phi2']) > 0
    assert len(catalog_sb) > 0

    N_sb = len(catalog_sb['phi1'])
    config_stream_N = STREAM_PARAMETERS.copy()
    config_stream_N["N"] = N_sb
    config_stream_N["surface_brightness"] = None
    catalog_N = generate.generate_uniform_stream(config_stream_N, seed=42)
    assert len(catalog_N['phi1']) == N_sb, "Catalog generated with N should have the same number of stars as catalog generated with surface brightness."

    # Verify output is full
    needed_columns = ['phi1', 'phi2', 'lsst_g_true', 'lsst_r_true', 'mass', "dist"]
    for col in needed_columns:
        assert col in catalog, f"Column {col} is missing from the generated catalog."

    # Verify that it does not contain useless columns
    optional_col = []
    assert all((col in needed_columns)|(col in optional_col) for col in catalog.keys()), "Catalog contains unexpected columns."

    # Verify if it contains nan values
    for col in catalog.keys():
        assert not np.any(np.isnan(catalog[col])), f"Column {col} contains NaN values."
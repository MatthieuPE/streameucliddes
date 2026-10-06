import pytest
import numpy as np
from streameucliddes.injection import inject_utils


# ===================================================
# Test injection utils functions
# ===================================================

# Test convert_N_SurfaceBrightness function
def test_convert_N_SurfaceBrightness():
    N = 1000
    mag_bounds = (None, 24)
    stream_length = 10.0  # degrees
    stream_width = 0.5    # degrees
    isochrone_params = {
        "isochrone": {
            "name": "Marigo2017",
            "survey": "lsst",
            "age": 12.0,  # Gyr
            "z": 0.0006,  # metallicity
            "band_1": "g",
        "band_2": "r",
        "band_1_detection": True,
    },
    'distance_modulus': {'center': {'value': 15.0}}
    }
    
    surface_brightness = inject_utils.convert_N_SurfaceBrightness(N, mag_bounds=mag_bounds, stream_length=stream_length, stream_width=stream_width, isochrone_params=isochrone_params, band='r', verbose=True)
    
    assert isinstance(surface_brightness, float), "Surface brightness should be a float."
    assert surface_brightness > 32, "Surface brightness should be less than 32 mag/arcsec^2 for this test case."

    surface_brightness_2 = inject_utils.convert_N_SurfaceBrightness(N*2, mag_bounds=mag_bounds, stream_length=stream_length, stream_width=stream_width, isochrone_params=isochrone_params, band='r', verbose=True)

    assert surface_brightness_2 < surface_brightness, "Surface brightness should decrease with increasing N."

    
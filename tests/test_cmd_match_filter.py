import numpy as np
import pytest
from streamobs.match_filter import build_match_filter

from streameucliddes.analysis import cmd_match_filter

pytestmark = pytest.mark.cmd_match_filter


# ===================================================
# Test build_match_filter_polygon
# ===================================================


# The polygon is build_match_filter's, built once for the same parameters
def test_match_filter_polygon_cache():
    params = {
        "distance_modulus": 16.8,
        "age": 12.0,
        "metallicity": 0.0006,
        "band_1": "g",
        "band_2": "r",
        "survey_1": "des",
        "survey_2": "des",
    }
    polygon = cmd_match_filter.build_match_filter_polygon(*params.values())
    np.testing.assert_array_equal(polygon, build_match_filter(**params))

    assert cmd_match_filter.build_match_filter_polygon(*params.values()) is polygon
    assert not polygon.flags.writeable  # shared between calls

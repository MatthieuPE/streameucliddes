import numpy as np
import ugali.isochrone
from ugali.isochrone.model import Isochrone

from streameucliddes import utils

ISOCHRONE = {"name": "Marigo2017", "survey": "des", "age": 12.0, "z": 0.0006}


# ===================================================
# Test cache_isochrone_grids
# ===================================================


# The cached grid is the one ugali finds by scanning the isochrone directory
def test_isochrone_grid_cache():
    iso = ugali.isochrone.factory(**ISOCHRONE)
    uncached_grid = Isochrone.create_grid.__wrapped__(iso)
    for cached, uncached in zip(iso.grid, uncached_grid):
        np.testing.assert_array_equal(cached, uncached)

    # One scan per directory, whatever the age; each isochrone has its own copy
    other = ugali.isochrone.factory(**{**ISOCHRONE, "age": 10.0})
    keys = [key for key in utils._ISOCHRONE_GRIDS if key[1] == iso.get_dirname()]
    assert len(keys) == 1
    assert not np.shares_memory(iso.grid[0], other.grid[0])

    # Enabling the cache again does not wrap it twice
    utils.cache_isochrone_grids()
    assert not hasattr(Isochrone.create_grid.__wrapped__, "__wrapped__")

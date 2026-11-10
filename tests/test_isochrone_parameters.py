import pytest

from streameucliddes.isochrone_parameters import (
    default_isochrone_grid,
    known_stream_isochrone_parameters,
)

pytestmark = pytest.mark.isochrone_parameters


# Phase 2 stubs: the module imports, with its functions
def test_import():
    stubs = (
        default_isochrone_grid,
        known_stream_isochrone_parameters,
    )
    assert all(callable(stub) for stub in stubs)

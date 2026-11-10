import pytest

from streameucliddes.stream_frames import (
    phi12_to_radec,
    radec_to_phi12,
    rotation_matrix_from_astropy_frame,
    rotation_matrix_from_endpoints,
)

pytestmark = pytest.mark.stream_frames


# Phase 2 stubs: the module imports, with its functions
def test_import():
    stubs = (
        rotation_matrix_from_astropy_frame,
        rotation_matrix_from_endpoints,
        radec_to_phi12,
        phi12_to_radec,
    )
    assert all(callable(stub) for stub in stubs)

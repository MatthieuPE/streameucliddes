import pytest

from streameucliddes.catalogs.survey_footprint import (
    survey_footprint_healpix_mask,
    survey_footprint_phi12_mask,
)

pytestmark = pytest.mark.survey_footprint


# Phase 2 stubs: the module imports, with its functions
def test_import():
    stubs = (
        survey_footprint_healpix_mask,
        survey_footprint_phi12_mask,
    )
    assert all(callable(stub) for stub in stubs)

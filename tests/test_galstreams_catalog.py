import pytest

from streameucliddes.catalogs.galstreams_catalog import (
    galstreams_dm_along_track,
    galstreams_phi12_window,
    galstreams_track_length_in_footprint,
    load_galstreams,
    select_galstreams_in_footprint,
)

pytestmark = pytest.mark.galstreams_catalog


# Phase 2 stubs: the module imports, with its functions
def test_import():
    stubs = (
        load_galstreams,
        galstreams_track_length_in_footprint,
        select_galstreams_in_footprint,
        galstreams_phi12_window,
        galstreams_dm_along_track,
    )
    assert all(callable(stub) for stub in stubs)

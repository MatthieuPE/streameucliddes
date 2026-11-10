import pytest

from streameucliddes.analysis.known_stream_search import (
    known_stream_search_dm_along_track,
    known_stream_search_dm_scan,
    known_stream_search_single_dm,
)

pytestmark = pytest.mark.known_stream_search


# Phase 2 stubs: the module imports, with its functions
def test_import():
    stubs = (
        known_stream_search_single_dm,
        known_stream_search_dm_scan,
        known_stream_search_dm_along_track,
    )
    assert all(callable(stub) for stub in stubs)

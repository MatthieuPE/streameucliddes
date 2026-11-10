import pytest

from streameucliddes.workflows.run_known_stream_search import (
    run_all_known_stream_searches,
    run_known_stream_search,
    save_known_stream_search_maps,
)

pytestmark = pytest.mark.run_known_stream_search


# Phase 2 stubs: the module imports, with its functions
def test_import():
    stubs = (
        run_known_stream_search,
        run_all_known_stream_searches,
        save_known_stream_search_maps,
    )
    assert all(callable(stub) for stub in stubs)

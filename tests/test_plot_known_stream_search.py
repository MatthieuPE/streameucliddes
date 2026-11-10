import pytest

from streameucliddes.plotting.plot_known_stream_search import (
    plot_galstreams_selection,
    plot_known_stream_search_single_dm,
)

pytestmark = pytest.mark.plot_known_stream_search


# Phase 2 stubs: the module imports, with its functions
def test_import():
    stubs = (
        plot_known_stream_search_single_dm,
        plot_galstreams_selection,
    )
    assert all(callable(stub) for stub in stubs)

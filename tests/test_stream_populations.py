import pytest

from streameucliddes.catalogs.stream_populations import (
    load_stream_population_table,
)

pytestmark = pytest.mark.stream_populations


# Phase 2 stubs: the module imports, with its functions
def test_import():
    stubs = (
        load_stream_population_table,
    )
    assert all(callable(stub) for stub in stubs)

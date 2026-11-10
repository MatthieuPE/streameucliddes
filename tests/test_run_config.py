import pytest

from streameucliddes.workflows.run_config import (
    RunConfig,
)

pytestmark = pytest.mark.run_config


# Phase 2 stubs: the module imports, with its functions
def test_import():
    stubs = (
        RunConfig,
        RunConfig.from_yaml,
    )
    assert all(callable(stub) for stub in stubs)

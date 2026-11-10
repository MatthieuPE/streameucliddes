"""Smoke test: the package imports and exposes its version."""
import pytest

import streameucliddes

pytestmark = pytest.mark.package


def test_version():
    assert streameucliddes.__version__ == "0.1.0"

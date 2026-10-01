"""Smoke test: the package imports and exposes its version."""

import streameucliddes


def test_version():
    assert streameucliddes.__version__ == "0.1.0"

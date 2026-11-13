"""Tests of analysis/hess_diagrams.py, on synthetic stars."""

import pytest

pytestmark = pytest.mark.hess_diagrams


@pytest.mark.skip(reason="stub")
def test_single_star_density():
    """One star in a 2 deg^2 region: density 0.5 in one bin, 0 elsewhere."""


@pytest.mark.skip(reason="stub")
def test_density_counts_error():
    """density = counts / area, error = sqrt(counts) / area."""


@pytest.mark.skip(reason="stub")
def test_difference_consistent_with_zero():
    """On and off from the same uniform population, areas 1 and 3: |diff| / error ~ N(0, 1)."""


@pytest.mark.skip(reason="stub")
def test_difference_rejects_different_bins():
    """Different bin edges: ValueError."""


@pytest.mark.skip(reason="stub")
def test_hess_rejects_non_positive_area():
    """area_deg2 <= 0: ValueError."""

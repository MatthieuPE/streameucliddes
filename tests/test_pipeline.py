import healpy as hp
import numpy as np

from streameucliddes.analysis import pipeline

CUTS = {"mag_g": (None, 22.0)}


def _find_stream_kwargs(nside, **kwargs):
    """find_stream() kwargs for the mock catalog (see conftest.py)."""
    return {
        "use_dust_corrected": False,  # the mock catalog has no dust map
        "nside": nside,
        "ra_col": "ra",
        "dec_col": "dec",
        **kwargs,
    }


# ===================================================
# Test find_stream
# ===================================================


# Smoke test: the full pipeline runs on the mock catalog (see conftest.py)
def test_pipeline_runs_without_error(sample_sky_catalog, nside):
    results = pipeline.find_stream(
        sample_sky_catalog,
        match_filter_parameters={"distance_modulus": 16.0},
        **_find_stream_kwargs(nside),
    )

    assert results["n_selected"] > 0
    for key in ("data_map", "fit_map", "residual_map"):
        assert results[key].size == hp.nside2npix(nside)


# The cuts are applied on top of the match filter, and stored in the results
def test_pipeline_applies_cuts(sample_sky_catalog, nside):
    match_filter_parameters = {"distance_modulus": 16.0}
    no_cuts = pipeline.find_stream(
        sample_sky_catalog, match_filter_parameters, **_find_stream_kwargs(nside)
    )
    with_cuts = pipeline.find_stream(
        sample_sky_catalog,
        match_filter_parameters,
        **_find_stream_kwargs(nside, cuts=CUTS),
    )

    expected = no_cuts["selection_flag"] & (sample_sky_catalog["mag_g"] <= 22.0)
    np.testing.assert_array_equal(with_cuts["selection_flag"], expected)
    assert 0 < with_cuts["n_selected"] < no_cuts["n_selected"]
    assert no_cuts["cuts"] == {}
    assert with_cuts["cuts"] == CUTS


# ===================================================
# Test scan_find_stream
# ===================================================


# The cuts precomputed once by the scan select the same stars as find_stream
def test_scan_applies_same_cuts_as_find_stream(sample_sky_catalog, nside):
    find_stream_kwargs = _find_stream_kwargs(nside, cuts=CUTS)
    scan = pipeline.scan_find_stream(
        sample_sky_catalog,
        [16.0],
        match_filter_parameters={},
        find_stream_kwargs=find_stream_kwargs,
        verbose=False,
    )
    single = pipeline.find_stream(
        sample_sky_catalog, {"distance_modulus": 16.0}, **find_stream_kwargs
    )

    assert scan[16.0]["n_selected"] == single["n_selected"]

import healpy as hp

from streameucliddes.analysis import pipeline

# ===================================================
# Test find_stream
# ===================================================


# Smoke test: the full pipeline runs on the mock catalog (see conftest.py)
def test_pipeline_runs_without_error(sample_sky_catalog, nside):
    results = pipeline.find_stream(
        sample_sky_catalog,
        match_filter_parameters={"distance_modulus": 16.0},
        use_dust_corrected=False,  # the mock catalog has no dust map
        nside=nside,
        ra_col="ra",
        dec_col="dec",
    )

    assert results["n_selected"] > 0
    for key in ("data_map", "fit_map", "residual_map"):
        assert results[key].size == hp.nside2npix(nside)

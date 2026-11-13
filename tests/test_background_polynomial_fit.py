import healpy as hp
import numpy as np
import pytest

from streameucliddes.analysis import background_polynomial_fit

pytestmark = pytest.mark.background_polynomial_fit

# The mock sky (projection, footprint, polynomial background) is defined in conftest.py.


def fit_background(data, degree, fit_log_space, **kwargs):
    """Run the polyfit2d fit and return the background map, with masked pixels as NaN."""
    _, _, prepared = background_polynomial_fit.fit_background_model_polyfit2d(
        data,
        model_type={"polynomial": {"degree": degree}},
        fit_log_space=fit_log_space,
        **kwargs,
    )
    return np.ma.filled(prepared["background_map"], np.nan)


# ===================================================
# Test fit_background_model_polyfit2d
# ===================================================


# A density that is exactly a polynomial of the fitted degree must be recovered
@pytest.mark.parametrize("fit_log_space", [False, True], ids=["linear", "log"])
@pytest.mark.parametrize("degree", [0, 1, 2, 3, 5])
def test_polynomial_density_is_recovered(
    proj, nside, in_extent, background_map, degree, fit_log_space
):
    truth = background_map(degree, log_space=fit_log_space)

    bkg = fit_background(truth, degree, fit_log_space, proj=proj, nside=nside)

    np.testing.assert_allclose(bkg[in_extent], truth[in_extent], rtol=1e-6)


def hide_outside(density, footprint, how):
    """Mark pixels outside the footprint as unobserved, the way a survey map may encode it."""
    data = density.copy()
    if how == "nan":
        data[~footprint] = np.nan
        return data, {}
    if how == "unseen":
        data[~footprint] = hp.UNSEEN
        return data, {}
    if how == "masked_array":
        return np.ma.array(data, mask=~footprint), {}
    if how == "mask_kwarg":  # sparse map storing unobserved pixels as 0
        data[~footprint] = 0.0
        return data, {"mask": ~footprint}
    raise ValueError(how)


# Unobserved pixels must not enter the fit, nor bias it at the footprint edges
@pytest.mark.parametrize("how", ["nan", "unseen", "masked_array", "mask_kwarg"])
def test_partial_footprint_is_recovered(proj, nside, footprint, background_map, how):
    degree = 3
    truth = background_map(degree, log_space=True)
    data, kwargs = hide_outside(truth, footprint, how)

    bkg = fit_background(
        data, degree, fit_log_space=True, proj=proj, nside=nside, **kwargs
    )

    # The footprint includes its outer and inner (hole) edges.
    np.testing.assert_allclose(bkg[footprint], truth[footprint], rtol=1e-6)


# Memory: only the unmasked pixels are projected and evaluated. On a small footprint the
# fit costs a few full-sky maps (input, mask, background), about 35 bytes per sky pixel;
# computing the coordinates of the whole sky cost about 200
def test_fit_memory_small_footprint():
    import tracemalloc

    nside = 256
    lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    outside = ~((((lon + 15.0) % 360.0) < 30.0) & (lat > -30.0) & (lat < -10.0))  # across ra = 0
    density = np.ma.array(np.exp(1.0 + 0.01 * lat), mask=outside)
    tracemalloc.start()
    try:
        bkg = fit_background(density, 5, True, proj=background_polynomial_fit.SimpleLonLatProj(), nside=nside,
                             mask=outside)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 60 * hp.nside2npix(nside)
    assert np.isnan(bkg[outside]).all() and np.isfinite(bkg[~outside]).all()

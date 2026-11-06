import healpy as hp
import numpy as np
import pytest

from streameucliddes.analysis import fit

# The mock sky (projection, footprint, polynomial background) is defined in conftest.py.


def fit_background(data, degree, fit_log_space, **kwargs):
    """Run the polyfit2d fit and return the background map, with masked pixels as NaN."""
    _, _, prepared = fit.fit_background_model_polyfit2d(
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

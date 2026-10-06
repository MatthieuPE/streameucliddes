import healpy as hp
import numpy as np
import pytest
from numpy.polynomial import polynomial

from streameucliddes.analysis import fit


# ===================================================
# Mock sky: a polynomial density over a small patch
# ===================================================

NSIDE = 64
LON0, LAT0 = 50.0, -25.0   # patch centre (deg)
HALF_WIDTH = (20.0, 15.0)  # patch half-size in projected x, y (deg)


class PlateCarree:
    """Minimal projection centred on the patch: x = lon - LON0 (wrapped), y = lat - LAT0."""

    def ang2xy(self, lon, lat, lonlat=True):
        x = (np.asarray(lon, dtype=float) - LON0 + 180.0) % 360.0 - 180.0
        y = np.asarray(lat, dtype=float) - LAT0
        return x, y

    def get_extent(self):
        return (-HALF_WIDTH[0], HALF_WIDTH[0], -HALF_WIDTH[1], HALF_WIDTH[1])


PROJ = PlateCarree()
X, Y = PROJ.ang2xy(*hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True))
IN_EXTENT = (np.abs(X) <= HALF_WIDTH[0]) & (np.abs(Y) <= HALF_WIDTH[1])


def polynomial_map(degree, log_space, seed=0):
    """HEALPix map following P(u, v), or exp(P) if log_space, inside the extent; 0 elsewhere.

    u, v are the projected coordinates rescaled to [-1, 1] over the extent. Higher-order
    coefficients are damped so that P stays within (2, 8), i.e. the density stays positive.
    """
    rng = np.random.default_rng(seed)
    order = np.add.outer(np.arange(degree + 1), np.arange(degree + 1))
    coeffs = rng.uniform(-1.0, 1.0, order.shape) * 0.5**order
    coeffs[0, 0] = 5.0

    p = polynomial.polyval2d(X[IN_EXTENT] / HALF_WIDTH[0], Y[IN_EXTENT] / HALF_WIDTH[1], coeffs)
    density = np.zeros(X.size)
    density[IN_EXTENT] = np.exp(p) if log_space else p
    return density


def fit_background(data, degree, fit_log_space, **kwargs):
    """Run the polyfit2d fit and return the background map, with masked pixels as NaN."""
    _, _, prepared = fit.fit_background_model_polyfit2d(
        data,
        model_type={"polynomial": {"degree": degree}},
        proj=PROJ,
        nside=NSIDE,
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
def test_polynomial_density_is_recovered(degree, fit_log_space):
    truth = polynomial_map(degree, log_space=fit_log_space)

    bkg = fit_background(truth, degree, fit_log_space)

    np.testing.assert_allclose(bkg[IN_EXTENT], truth[IN_EXTENT], rtol=1e-6)


def irregular_footprint():
    """Disc inside the extent with a hole (e.g. a star mask): unobserved pixels fall inside the fit box."""
    disc = hp.query_disc(NSIDE, hp.ang2vec(LON0, LAT0, lonlat=True), np.radians(12.0))
    hole = hp.query_disc(NSIDE, hp.ang2vec(LON0 + 5.0, LAT0 + 3.0, lonlat=True), np.radians(3.0))
    footprint = np.zeros(X.size, dtype=bool)
    footprint[disc] = True
    footprint[hole] = False
    return footprint


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
def test_partial_footprint_is_recovered(how):
    degree = 3
    truth = polynomial_map(degree, log_space=True)
    footprint = irregular_footprint()
    data, kwargs = hide_outside(truth, footprint, how)

    bkg = fit_background(data, degree, fit_log_space=True, **kwargs)

    # The footprint includes its outer and inner (hole) edges.
    np.testing.assert_allclose(bkg[footprint], truth[footprint], rtol=1e-6)

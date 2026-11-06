"""Unit conversions: AB magnitude and flux, distance modulus and kpc, [Fe/H] and Z."""

import astropy.units as u
import numpy as np
from astropy.coordinates import Distance

def magToFlux(mag):
    """
    Convert from AB magnitude to flux.

    Parameters
    ----------
    mag : float or np.ndarray
        AB magnitude(s).

    Returns
    -------
    float or np.ndarray
        Flux in Janskys (Jy).
    """
    return 3631.0 * 10 ** (-0.4 * mag)


def fluxToMag(flux):
    """
    Convert from flux to AB magnitude.

    Parameters
    ----------
    flux : float or np.ndarray
        Flux in Janskys (Jy).

    Returns
    -------
    float or np.ndarray
        AB magnitude(s).
    """
    return -2.5 * np.log10(flux / 3631.0)


def convert_DM_to_kpc(distance_modulus):
    return Distance(distmod=distance_modulus).to(u.kpc)


def convert_kpc_to_DM(distance_kpc):
    return Distance(distance_kpc, unit=u.kpc).distmod


def convert_FeH_to_z(feh):
    # Function coming from github.com/DarkEnergySurvey/ugali/blob/e4aa0a26e9d245d489ccb6b4cdb149386fdcb45b/ugali/isochrone/parsec.py#L190
    # Taken from Table 3 and Section 3 of Bressan et al. 2012
    # Confirmed in Section 2.1 of Marigo et al. 2017
    Y_p = 0.2485  # Primordial He abundance
    c = 1.78  # He enrichment ratio

    Z_solar = 0.01524  # Solar metal abundance
    Y_solar = 0.2485  # Solar He abundance
    X_solar = 1 - Y_solar - Z_solar

    return (1 - Y_p) / ((1 + c) + X_solar / Z_solar * 10 ** (-feh))

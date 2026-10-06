import numpy as np


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

"""Unit conversions: AB magnitude and flux, distance modulus and kpc, [Fe/H] and Z.

[Fe/H] and Z: the relation depends on the isochrone model (its solar and
primordial abundances): [Fe/H] = -2 is Z = 1.55e-4 for Marigo2017, 1.49e-4
for Dotter2016. The conversions use ugali's relations (the ``feh2z`` and
``z2feh`` of its isochrone classes), for the isochrone model given:

- ``isochrone_model``: the name of a ugali isochrone model, the name given
  to ugali (``ugali.isochrone.factory(isochrone_model, ...)``), e.g.
  ``"Marigo2017"``, ``"Dotter2016"``;
- a ugali isochrone (``ugali_isochrone``) is what ugali builds from it, an
  instance of the ugali class of that name (``ugali_isochrone_class``, e.g.
  ``ugali.isochrone.Marigo2017``).

Functions take ``isochrone_model=None`` and resolve it with
``get_isochrone_model``: the one place where the default,
``DEFAULT_ISOCHRONE_MODEL``, is chosen.
"""

from collections.abc import Mapping

import astropy.units as u
import numpy as np
import ugali.isochrone
from astropy.coordinates import Distance

# Name of the ugali isochrone model used by default: Dotter et al. 2016 (MIST), a higher SNR than Marigo2017 (PARSEC,
# that of streamobs.match_filter.build_match_filter) on Aliqa Uma in Euclid DR1 VIS - H
# (notebooks/optimization/stream_optimization.ipynb). Read through get_isochrone_model
DEFAULT_ISOCHRONE_MODEL = "Dotter2016"

def mag_to_flux(mag):
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


def flux_to_mag(flux):
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


def get_isochrone_model(isochrone=None):
    """
    Name of the ugali isochrone model to use.

    The single place where the default is chosen: functions take
    ``isochrone_model=None`` and call this.

    Parameters
    ----------
    isochrone : str, mapping or None
        A model name (returned as it is), an isochrone ``{"age",
        "metallicity"}`` (its ``"isochrone_model"``), or None.

    Returns
    -------
    str
        The name, ``DEFAULT_ISOCHRONE_MODEL`` when none is given.
    """
    if isinstance(isochrone, Mapping):
        isochrone = isochrone.get("isochrone_model")
    return DEFAULT_ISOCHRONE_MODEL if isochrone is None else str(isochrone)


def _ugali_isochrone_class(isochrone_model=None):
    """ugali class of a model name (``get_isochrone_model``); ValueError for an unknown name."""
    isochrone_model = get_isochrone_model(isochrone_model)
    ugali_isochrone_class = getattr(ugali.isochrone, isochrone_model, None)
    if not isinstance(ugali_isochrone_class, type) or not hasattr(ugali_isochrone_class, "feh2z"):
        classes = {name: getattr(ugali.isochrone, name) for name in dir(ugali.isochrone)}
        known = [name for name, value in classes.items() if isinstance(value, type) and hasattr(value, "feh2z")]
        raise ValueError(f"Unknown ugali isochrone model {isochrone_model!r}: one of {known}.")
    return ugali_isochrone_class


def convert_FeH_to_z(feh, isochrone_model=None):
    """
    Convert [Fe/H] to the metal mass fraction Z of an isochrone model.

    ugali's relation for that model (``feh2z`` of its class): for
    Marigo2017 and Bressan2012 (PARSEC), Table 3 and Sect. 3 of Bressan et
    al. 2012, confirmed in Sect. 2.1 of Marigo et al. 2017; for Dotter2016
    (MIST), Sect. 3.1 of Choi et al. 2016.

    Parameters
    ----------
    feh : float or array_like
        [Fe/H].
    isochrone_model : str, optional
        Name of the ugali isochrone model. Default: ``get_isochrone_model()``.

    Returns
    -------
    float or np.ndarray
        Z, on the scale of the model.

    Raises
    ------
    ValueError
        Unknown isochrone model.
    """
    z = _ugali_isochrone_class(isochrone_model).feh2z(np.asarray(feh, dtype=float))
    return z if np.ndim(feh) else float(z)


def convert_z_to_FeH(z, isochrone_model=None):
    """
    Convert the metal mass fraction Z of an isochrone model to [Fe/H] (``z2feh`` of its ugali class).

    Parameters and errors: see ``convert_FeH_to_z``.
    """
    feh = _ugali_isochrone_class(isochrone_model).z2feh(np.asarray(z, dtype=float))
    return feh if np.ndim(z) else float(feh)


def convert_metallicity(z, from_isochrone_model, to_isochrone_model):
    """
    Convert a Z from the scale of one isochrone model to that of another, for the same [Fe/H].

    ``convert_FeH_to_z(convert_z_to_FeH(z, from_isochrone_model), to_isochrone_model)``;
    unchanged for the same model. E.g. Z = 0.0002 for Marigo2017 ([Fe/H] =
    -1.89) is Z = 0.000192 for Dotter2016.

    Parameters
    ----------
    z : float or array_like
        Z, on the scale of ``from_isochrone_model``.
    from_isochrone_model, to_isochrone_model : str or None
        Names of the ugali isochrone models (None: ``get_isochrone_model()``).

    Returns
    -------
    float or np.ndarray
    """
    if get_isochrone_model(from_isochrone_model) == get_isochrone_model(to_isochrone_model):
        return z if np.ndim(z) else float(z)
    return convert_FeH_to_z(convert_z_to_FeH(z, from_isochrone_model), to_isochrone_model)

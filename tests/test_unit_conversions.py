"""Tests of unit_conversions.py: [Fe/H] and Z per isochrone model, and the isochrone model to use."""

import numpy as np
import pytest
import ugali.isochrone

from streameucliddes.unit_conversions import (
    DEFAULT_ISOCHRONE_MODEL,
    convert_FeH_to_z,
    convert_metallicity,
    convert_z_to_FeH,
    get_isochrone_model,
)

pytestmark = pytest.mark.unit_conversions


# The name to use: the one given, that of an isochrone, else the default
def test_get_isochrone_model():
    assert get_isochrone_model() == get_isochrone_model(None) == DEFAULT_ISOCHRONE_MODEL
    assert get_isochrone_model({"age": 13.0}) == DEFAULT_ISOCHRONE_MODEL
    assert get_isochrone_model({"age": 13.0, "isochrone_model": "Dotter2016"}) == "Dotter2016"
    assert get_isochrone_model("Dotter2016") == "Dotter2016"


# Each isochrone model its own [Fe/H] <-> Z: that of its ugali class; arrays and floats
@pytest.mark.parametrize("isochrone_model", ["Marigo2017", "Dotter2016", "Dotter2008", "Bressan2012"])
def test_convert_FeH_to_z(isochrone_model):
    feh = np.array([-3.0, -2.0, -1.0])
    z = convert_FeH_to_z(feh, isochrone_model)
    np.testing.assert_allclose(z, getattr(ugali.isochrone, isochrone_model).feh2z(feh))
    np.testing.assert_allclose(convert_z_to_FeH(z, isochrone_model), feh)
    assert isinstance(convert_FeH_to_z(-2.0, isochrone_model), float)


# The default model: the PARSEC relation of Bressan et al. 2012 (Table 3, Sect. 3), that of the Z of
# catalogs/stream_populations.csv; another model, another Z; an unknown model refused
def test_convert_FeH_to_z_scales():
    feh = np.array([-2.5, -1.0])
    y_p, c, z_solar, y_solar = 0.2485, 1.78, 0.01524, 0.2485
    bressan = (1 - y_p) / ((1 + c) + (1 - y_solar - z_solar) / z_solar * 10 ** (-feh))
    np.testing.assert_allclose(convert_FeH_to_z(feh, "Marigo2017"), bressan)
    assert convert_FeH_to_z(-2.0, "Dotter2016") != pytest.approx(convert_FeH_to_z(-2.0, "Marigo2017"), rel=1e-2)
    with pytest.raises(ValueError, match="Dotter2016"):
        convert_FeH_to_z(-2.0, "Dotter2017")


# The same [Fe/H] on another scale; the same model (None: the default one): unchanged
def test_convert_metallicity():
    z = convert_metallicity(0.0002, "Marigo2017", "Dotter2016")
    assert convert_z_to_FeH(z, "Dotter2016") == pytest.approx(convert_z_to_FeH(0.0002, "Marigo2017"))
    assert convert_metallicity(0.0002, None, DEFAULT_ISOCHRONE_MODEL) == 0.0002
    assert convert_metallicity(0.0002, "Dotter2016", "Dotter2016") == 0.0002


# Round trips, for every isochrone model: Z -> [Fe/H] -> Z and [Fe/H] -> Z -> [Fe/H] give back what was given,
# for arrays and floats, over the Z of the ugali grids (1e-5 to 1e-3) and beyond (up to solar)
ISOCHRONE_MODELS = ["Marigo2017", "Dotter2016", "Dotter2008", "Bressan2012"]


@pytest.mark.parametrize("isochrone_model", ISOCHRONE_MODELS)
def test_round_trip_z_feh_z(isochrone_model):
    z = np.array([1e-5, 1e-4, 2e-4, 1e-3, 0.015])
    np.testing.assert_allclose(convert_FeH_to_z(convert_z_to_FeH(z, isochrone_model), isochrone_model), z, rtol=1e-12)
    assert convert_FeH_to_z(convert_z_to_FeH(2e-4, isochrone_model), isochrone_model) == pytest.approx(2e-4, rel=1e-12)
    feh = np.array([-4.0, -2.5, -1.0, 0.0])
    np.testing.assert_allclose(convert_z_to_FeH(convert_FeH_to_z(feh, isochrone_model), isochrone_model), feh,
                               rtol=0, atol=1e-12)


# Z to another isochrone model and back: the same Z
@pytest.mark.parametrize("isochrone_model", ISOCHRONE_MODELS)
def test_round_trip_convert_metallicity(isochrone_model):
    z = np.array([1e-5, 2e-4, 1e-3])
    there = convert_metallicity(z, DEFAULT_ISOCHRONE_MODEL, isochrone_model)
    np.testing.assert_allclose(convert_metallicity(there, isochrone_model, DEFAULT_ISOCHRONE_MODEL), z, rtol=1e-12)

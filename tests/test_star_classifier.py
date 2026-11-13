"""Tests of catalogs/star_classifier.py (XGBoost star / galaxy classifier)."""

import numpy as np
import pandas as pd
import pytest

xgboost = pytest.importorskip("xgboost")

from streameucliddes.catalogs import star_classifier  # noqa: E402

pytestmark = pytest.mark.star_classifier

FEATURES = {"ellipticity", "mumax_minus_mag", "point_like_prob", "mag_stargal_sep"}


@pytest.fixture
def features(rng):
    """Random features, in the ranges of the Euclid catalogs, with some missing values."""
    n = 500
    values = {
        "ellipticity": rng.uniform(0.0, 1.0, n),
        "mumax_minus_mag": rng.uniform(-4.0, -1.0, n),
        "point_like_prob": rng.uniform(0.0, 1.0, n),
        "mag_stargal_sep": rng.uniform(18.0, 26.0, n),
    }
    values["point_like_prob"][::9] = np.nan  # NIR-only sources
    return values


def _model_matrix(features):
    """(n, 4) float32 array of the features, in the order of the model."""
    names = star_classifier.load_star_classifier().feature_names
    return np.column_stack([features[name] for name in names]).astype(np.float32)


def test_model_features():
    assert set(star_classifier.load_star_classifier().feature_names) == FEATURES


def test_probability_matches_booster(features):
    model = star_classifier.load_star_classifier()
    expected = model.predict(xgboost.DMatrix(_model_matrix(features), feature_names=model.feature_names))
    probability = star_classifier.star_probability(features)
    assert probability.dtype == np.float32
    np.testing.assert_allclose(probability, expected, rtol=1e-6)
    assert ((probability >= 0) & (probability <= 1)).all()
    assert probability.min() < 0.5 < probability.max()  # the features give both classes


def test_probability_matches_xgbclassifier(features):
    """Same values as the scikit-learn interface used to train the model."""
    pytest.importorskip("sklearn")
    classifier = xgboost.XGBClassifier()
    classifier.load_model(star_classifier.STAR_CLASSIFIER_MODEL)
    names = classifier.get_booster().feature_names
    expected = classifier.predict_proba(pd.DataFrame({n: features[n] for n in names}))[:, 1]
    np.testing.assert_allclose(star_classifier.star_probability(features), expected, rtol=1e-6)


def test_probability_any_feature_order(features):
    reversed_order = dict(reversed(list(features.items())))
    np.testing.assert_array_equal(
        star_classifier.star_probability(reversed_order), star_classifier.star_probability(features)
    )


def test_probability_by_chunks(features):
    np.testing.assert_array_equal(
        star_classifier.star_probability(features, chunk_size=7), star_classifier.star_probability(features)
    )


def test_infinite_features_are_missing(features):
    with_inf = {name: values.copy() for name, values in features.items()}
    with_nan = {name: values.copy() for name, values in features.items()}
    with_inf["mumax_minus_mag"][::5] = np.inf
    with_inf["mag_stargal_sep"][::7] = -np.inf
    with_nan["mumax_minus_mag"][::5] = np.nan
    with_nan["mag_stargal_sep"][::7] = np.nan
    np.testing.assert_array_equal(
        star_classifier.star_probability(with_inf), star_classifier.star_probability(with_nan)
    )


def test_missing_feature(features):
    del features["mag_stargal_sep"]
    with pytest.raises(KeyError, match="mag_stargal_sep"):
        star_classifier.star_probability(features)

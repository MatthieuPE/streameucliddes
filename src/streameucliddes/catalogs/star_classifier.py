"""Star / galaxy classifier: an XGBoost model trained on Euclid Q1.

The model (``star_galaxy_xgb_q1.json``, next to this module) gives the probability
that a source is a star from four Euclid MER morphology columns, named in the
model in lower case: ``ellipticity``, ``mumax_minus_mag``, ``point_like_prob`` and
``mag_stargal_sep``. :func:`star_probability` takes them by name.
"""

import functools
from pathlib import Path

import numpy as np
import xgboost

STAR_CLASSIFIER_MODEL = Path(__file__).with_name("star_galaxy_xgb_q1.json")


@functools.cache
def load_star_classifier(model_path=STAR_CLASSIFIER_MODEL):
    """
    Load the XGBoost star / galaxy classifier (once per path).

    Parameters
    ----------
    model_path : str or Path
        XGBoost model file (JSON); default: the Euclid Q1 model.

    Returns
    -------
    xgboost.Booster
        Its ``feature_names`` are the features, in the order of the model.
    """
    return xgboost.Booster(model_file=str(model_path))


def star_probability(features, model_path=STAR_CLASSIFIER_MODEL, chunk_size=1_000_000):
    """
    Probability that each source is a star, from the classifier.

    Same values as ``XGBClassifier.predict_proba(features)[:, 1]`` with the model
    loaded in an ``XGBClassifier``. Infinite feature values (a few in the Euclid
    catalogs, e.g. ``MUMAX_MINUS_MAG``) are treated as missing, like NaN: XGBoost
    rejects infinities.

    Parameters
    ----------
    features : dict or pd.DataFrame
        One array per feature of the model (``load_star_classifier().feature_names``),
        under that name.
    model_path : str or Path
        XGBoost model file; default: the Euclid Q1 model.
    chunk_size : int
        Number of sources classified at once (bounds the temporary arrays).

    Returns
    -------
    np.ndarray of float32
    """
    model = load_star_classifier(model_path)
    missing = [name for name in model.feature_names if name not in features]
    if missing:
        raise KeyError(f"Feature(s) of the star classifier missing: {missing}")
    columns = [np.asarray(features[name]) for name in model.feature_names]
    n_sources = len(columns[0])
    probability = np.empty(n_sources, dtype=np.float32)
    for start in range(0, n_sources, chunk_size):
        stop = start + chunk_size
        chunk = np.column_stack([c[start:stop] for c in columns]).astype(np.float32, copy=False)
        chunk[np.isinf(chunk)] = np.nan  # missing
        probability[start:stop] = model.inplace_predict(chunk)
    return probability

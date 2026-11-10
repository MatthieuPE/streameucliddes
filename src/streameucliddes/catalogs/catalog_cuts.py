import numpy as np
import pandas as pd


# ===================================================
# Cuts
# ===================================================


def _cut_mask(values, spec):
    """Boolean mask of ``values`` passing one cut (NaN never passes a range cut)."""
    values = np.asarray(values)
    if isinstance(spec, tuple):
        low, high = spec
        mask = np.ones(len(values), dtype=bool)
        if low is not None:
            mask &= values >= low
        if high is not None:
            mask &= values <= high
        return mask
    if isinstance(spec, (list, set)):
        return np.isin(values, list(spec))
    return values == spec


def get_cut_mask(data, cuts):
    """
    Return a boolean mask selecting the rows passing all ``cuts``.

    Parameters
    ----------
    data : pd.DataFrame
    cuts : dict
        ``{column: spec}``. ``spec`` is a ``(min, max)`` tuple (inclusive, None for an
        open bound), a list/set of accepted values, or a single accepted value.
        Example: ``{"EXT_XGB_des": (0, 1), "SPURIOUS_FLAG_euclid": 0,
        "euclid_q1_VIS_obs": (EUCLID_VIS_SATURATION, None)}``.

    Returns
    -------
    np.ndarray of bool
    """
    mask = np.ones(len(data), dtype=bool)
    for column, spec in cuts.items():
        mask &= _cut_mask(data[column], spec)
    return mask


def cut_flow(data, cuts):
    """
    Number of rows left after applying each cut in turn.

    Parameters
    ----------
    data : pd.DataFrame
    cuts : dict
        Same format as :func:`get_cut_mask`.

    Returns
    -------
    pd.DataFrame
        One row per cut: the cut, the rows passing that cut alone, and the rows (and
        fraction of the input) left after it and all previous cuts.
    """
    mask = np.ones(len(data), dtype=bool)
    rows = [{"cut": "input", "n_alone": len(data), "n_remaining": len(data)}]
    for column, spec in cuts.items():
        cut = _cut_mask(data[column], spec)
        mask &= cut
        rows.append(
            {
                "cut": f"{column}: {spec}",
                "n_alone": int(cut.sum()),
                "n_remaining": int(mask.sum()),
            }
        )
    flow = pd.DataFrame(rows)
    flow["fraction_remaining"] = flow["n_remaining"] / max(len(data), 1)
    return flow
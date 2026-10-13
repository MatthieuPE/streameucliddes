import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from streameucliddes.plotting import plotting


def _scan_results(names):
    """scan_snr_grid-like results: SNR growing with brightness, one SNR = inf cell."""
    surface_brightness = np.linspace(32.0, 36.0, 5)
    distance_modulus = np.linspace(16.0, 20.0, 4)
    results = {}
    for k, name in enumerate(names):
        snr = (k + 1) * 10 ** (0.4 * (36.0 - surface_brightness))[:, None] * np.ones(4)
        snr[0, -1] = np.inf  # stream stars but no background star
        results[name] = {
            "snr": snr,
            "surface_brightness": surface_brightness,
            "distance_modulus": distance_modulus,
        }
    return results


# One color per match filter, a legend entry and a contour label for each
def test_plot_snr_contours():
    results = _scan_results(["des_gr", "cross_gH"])
    fig, ax = plotting.plot_snr_contours(results, snr_level=5)

    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["des$\\_$gr", "cross$\\_$gH"]
    colors = [line.get_color() for line in legend.get_lines()]
    assert colors == plotting.CATEGORICAL_COLORS[:2]
    assert len(ax.texts) == 2  # direct labels
    assert ax.yaxis_inverted()  # brighter surface brightness up
    plt.close(fig)

    # More match filters than distinct colors
    with pytest.raises(ValueError):
        plotting.plot_snr_contours(_scan_results([f"mf{i}" for i in range(7)]))
    plt.close("all")


# One panel per match filter, with the cells without background star hatched
def test_plot_snr_maps():
    results = _scan_results(["des_gr", "cross_gH"])
    fig, axes = plotting.plot_snr_maps(results, snr_level=5)

    assert [ax.get_title() for ax in axes] == ["des$\\_$gr", "cross$\\_$gH"]
    for ax in axes:
        assert len(ax.patches) == 1  # the SNR = inf cell
    plt.close(fig)

"""Colour-magnitude diagrams."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from streameucliddes.plotting import plot_style  # noqa: F401  (sets the figure style)

def plot_CMD(
    data,
    fig=None,
    ax=None,
    mag_cols=["mag_g", "mag_r"],
    ylim=(15, 28),
    xlim=(0.1, 1.2),
    xlabel=rf"$g-r$",
    ylabel=rf"$g$",
    color=None,
    cmap="cividis",
    norm="log",
    bins_colors=None,
    bins_mag=None,
):
    """
    To plot 2D hist in the CMD

    data must be data frame of dict.
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    mag1 = pd.to_numeric(data[mag_cols[0]], errors="coerce")
    mag2 = pd.to_numeric(data[mag_cols[1]], errors="coerce")

    if bins_colors is None:
        bins_colors = np.arange(xlim[0], xlim[1], 0.01)
    if bins_mag is None:
        bins_mag = np.arange(ylim[0], ylim[1], 0.1)

    counts, xedges, yedges = np.histogram2d(
        mag1 - mag2, mag1, bins=[bins_colors, bins_mag]
    )
    if norm is None:
        label = "N"
    elif norm == "log":
        norm = LogNorm()
        label = "log10(N)"
    else:
        raise ValueError(f"Invalid norm: {norm}. Use None or 'log'.")

    im = ax.imshow(
        counts.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap,
        norm=norm,
    )
    # plt.colorbar(im, ax=ax, label=label)
    # plt.colorbar(im, ax=ax)
    cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label(label)

    # cbar.ax.yaxis.set_label_coords(0.5, 1.05)
    # cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label, loc="top")

    ax.set_xlabel(f"{mag_cols[0]} - {mag_cols[1]}")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel(mag_cols[0])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ax.get_legend():
        ax.legend()

    ax.invert_yaxis()
    return fig, ax

"""Figures of the injected-stream SNR study (analysis.injected_stream_snr)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from streameucliddes.plotting.plot_style import CATEGORICAL_COLORS, N_SERIES_MAX, escape_underscores

SNR_AXIS_LABELS = {
    "surface_brightness": r"Surface brightness [mag arcsec$^{-2}$]",
    "distance_modulus": "Distance modulus",
    "stream_width": "Stream width [deg]",
    "stream_length": "Stream length [deg]",
}
INK = "0.15"  # Text and neutral lines
NO_BACKGROUND_HATCH = "////"


def _snr_grid_axes(grid):
    """(y values, x values, y name, x name) of a scan_snr_grid result: snr[i, j] at (y[i], x[j])."""
    y_name, x_name = [key for key in grid if key != "snr"]
    return grid[y_name], grid[x_name], y_name, x_name


def _defined_snr(snr):
    """SNR with NaN where undefined: stream stars but no background star (SNR = inf)."""
    return np.where(np.isfinite(snr), snr, np.nan)


def _isolated_point(vertices, other_vertices, x_range, y_range, margin=0.06):
    """
    Vertex of a contour farthest from the other contours (in axes fractions), away
    from the frame: where its label cannot be mistaken for another contour's.
    """
    def to_axes(xy):
        return np.column_stack(
            [(xy[:, 0] - x_range[0]) / np.ptp(x_range), (xy[:, 1] - y_range[0]) / np.ptp(y_range)]
        )

    points = to_axes(vertices)
    inside = np.all((points > margin) & (points < 1 - margin), axis=1)
    if not inside.any():
        inside[:] = True
    if len(other_vertices) == 0:
        distance = np.ones(len(points))
    else:
        others = to_axes(other_vertices)
        distance = np.min(np.linalg.norm(points[:, None] - others[None], axis=2), axis=1)
    return vertices[np.argmax(np.where(inside, distance, -1))]


def _format_snr_axes(ax, x_name, y_name):
    """Axis labels; surface brightness brighter up; distance in kpc on top."""
    ax.set_xlabel(SNR_AXIS_LABELS.get(x_name, x_name))
    ax.set_ylabel(SNR_AXIS_LABELS.get(y_name, y_name))
    if y_name == "surface_brightness" and not ax.yaxis_inverted():
        ax.invert_yaxis()
    if x_name == "distance_modulus":
        ax.tick_params(axis="x", top=False)
        top = ax.secondary_xaxis(
            "top",
            functions=(
                lambda dm: 10 ** (np.asarray(dm) / 5 - 2),
                lambda d: 5 * (np.log10(np.clip(d, 1e-3, None)) + 2),
            ),
        )
        top.set_xticks([5, 10, 20, 50, 100, 200, 500])
        top.set_xlabel("Distance [kpc]")


def plot_snr_contours(results, snr_level=5, ax=None, colors=None, label_on_contours=False, names_to_plot=None):
    """
    Contour SNR = ``snr_level`` of each match filter of a scan_snr_grid result.

    One color per match filter, in the fixed order of CATEGORICAL_COLORS, with a
    legend and a label on each contour. Cells with stream stars but no background
    star (SNR = inf) are left out: the SNR is not defined there (see plot_snr_maps).

    Parameters
    ----------
    results : dict
        Output of injected_stream_snr.scan_snr_grid.
    snr_level : float
        SNR of the contours.
    ax : matplotlib.axes.Axes, optional
    colors : dict, optional
        ``{match filter name: color}``. Default: CATEGORICAL_COLORS in the order of
        ``results``, so pass it to keep a filter's color when changing the list.

    Returns
    -------
    fig, ax
    """
    if names_to_plot is not None:
        results = {name: grid for name, grid in results.items() if name in names_to_plot}
    
    if colors is None:
        if len(results) > N_SERIES_MAX:
            raise ValueError(
                f"{len(results)} match filters: at most {N_SERIES_MAX} can be told apart "
                "by color, plot fewer or pass `colors`."
            )
        colors = dict(zip(results, CATEGORICAL_COLORS))
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
    else:
        fig = ax.figure

    handles, contours = [], {}
    for name, grid in results.items():
        y, x, y_name, x_name = _snr_grid_axes(grid)
        contours[name] = ax.contour(
            x, y, _defined_snr(grid["snr"]), levels=[snr_level],
            colors=[colors[name]], linewidths=2,
        )
        handles.append(Line2D([], [], color=colors[name], lw=2, label=escape_underscores(name)))

    if label_on_contours:
        # Each label where its contour is farthest from the others
        vertices = {name: contour.get_paths()[0].vertices for name, contour in contours.items()}
        for name, contour in contours.items():
            if len(vertices[name]) == 0:
                continue  # SNR never reaches snr_level: only in the legend
            others = [v for other, v in vertices.items() if other != name and len(v)]
            position = _isolated_point(
                vertices[name], np.concatenate(others) if others else [], (x[0], x[-1]), (y[0], y[-1])
            )
            ax.clabel(
                contour, fmt={snr_level: escape_underscores(name)}, manual=[position],
                fontsize=7, colors=INK, inline_spacing=2,
            )

    _format_snr_axes(ax, x_name, y_name)
    ax.legend(handles=handles, title=f"SNR = {snr_level:g}", loc="lower left", frameon=False)
    return fig, ax


def plot_snr_maps(results, snr_level=5, cmap="Blues", vmin=1.0, vmax=None):
    """
    SNR map of each match filter of a scan_snr_grid result, side by side.

    Shared logarithmic color scale (SNR below ``vmin`` shown as ``vmin``), the
    contour SNR = ``snr_level`` in black, and hatched cells for stream stars without
    any background star (SNR = inf).

    Returns
    -------
    fig, axes
    """
    names = list(results)
    snrs = [results[name]["snr"] for name in names]
    if vmax is None:
        vmax = max(snr[np.isfinite(snr)].max() for snr in snrs)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        1, len(names), sharex=True, sharey=True, squeeze=False,
        figsize=(2.6 * len(names) + 1.0, 3.2), constrained_layout=True,
    )
    axes = axes[0]
    for ax, name, snr in zip(axes, names, snrs):
        y, x, y_name, x_name = _snr_grid_axes(results[name])
        mesh = ax.pcolormesh(
            x, y, np.clip(_defined_snr(snr), vmin, None),
            cmap=cmap, norm=norm, shading="nearest",
        )
        # Hatch the cells with stream stars but no background star
        dx, dy = np.gradient(x), np.gradient(y)
        for i, j in zip(*np.nonzero(np.isposinf(snr))):
            ax.add_patch(Rectangle(
                (x[j] - dx[j] / 2, y[i] - dy[i] / 2), dx[j], dy[i],
                fill=False, hatch=NO_BACKGROUND_HATCH, edgecolor="0.6", linewidth=0,
            ))
        ax.contour(x, y, _defined_snr(snr), levels=[snr_level], colors=INK, linewidths=1.5)
        ax.set_title(escape_underscores(name))
        _format_snr_axes(ax, x_name, y_name)
        if ax is not axes[0]:
            ax.set_ylabel("")

    fig.colorbar(mesh, ax=axes, label="SNR")
    handles = [Line2D([], [], color=INK, lw=1.5, label=f"SNR = {snr_level:g}")]
    if any(np.isposinf(snr).any() for snr in snrs):
        handles.append(Patch(facecolor="none", edgecolor="0.6", hatch=NO_BACKGROUND_HATCH,
                             label="No background star (SNR = inf)"))
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False)
    return fig, axes

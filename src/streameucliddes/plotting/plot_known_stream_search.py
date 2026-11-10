"""Figures of the known-stream search (analysis.known_stream_search)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import numpy as np
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:  # galstreams: new dependency of phase 3, not installed yet
    import galstreams


def plot_known_stream_search_single_dm(
    result: Mapping[str, Any],
    *,
    track_phi12: tuple[ArrayLike, ArrayLike] | None = None,
    other_tracks: Sequence[tuple[str, ArrayLike, ArrayLike]] = (),
    ax: Axes | None = None,
    vmax: float | None = None,
    cmap: str = "gray_r",
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Map of a known-stream search at one distance modulus.

    ``imshow`` of the density on the phi1/phi2 grid, phi1 increasing to the
    left, with the track of the stream and the other galstreams tracks in the
    window. Does not save the figure: the workflows do.

    Parameters
    ----------
    result : mapping
        Output of ``known_stream_search_single_dm``.
    track_phi12 : tuple of array_like, optional
        (phi1, phi2) of the stream track (deg), drawn as a line.
    other_tracks : sequence of (label, phi1, phi2), optional
        Other tracks in the window (deg), labelled.
    ax : matplotlib.axes.Axes, optional
        Axes to draw in. Default: a new figure.
    vmax : float, optional
        Top of the colour scale. Default: a high percentile of the density.
    cmap : str, optional
        Colour map.
    title : str, optional
        Default: the distance modulus and the distance.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Notes
    -----
    Code to reuse: celestial_currents
    ``visualization.py::plot_custom_stream_at_mu``, rewritten with ``imshow``
    on the grid and without saving; ``_find_overlapping_streams`` for the
    other tracks.
    """
    raise NotImplementedError


def plot_galstreams_selection(
    selection: pd.DataFrame,
    mws: galstreams.MWStreams,
    footprint_mask: NDArray[np.bool_],
    *,
    nest: bool,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Sky map of the footprint, with the known streams selected for the search.

    The footprint (skyproj), with the tracks of ``selection`` drawn and
    labelled.

    Parameters
    ----------
    selection : pandas.DataFrame
        Output of ``catalogs.galstreams_catalog.select_galstreams_in_footprint``.
    mws : galstreams.MWStreams
        From ``catalogs.galstreams_catalog.load_galstreams``.
    footprint_mask : numpy.ndarray of bool
        HEALPix mask, True outside the footprint.
    nest : bool
        HEALPix ordering of ``footprint_mask``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw in. Default: a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Notes
    -----
    Code to reuse: ``plot_sky_density.plot_2D_density_from_hp`` for the
    footprint; celestial_currents ``visualization.py::_find_overlapping_streams``.
    """
    raise NotImplementedError

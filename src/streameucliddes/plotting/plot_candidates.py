"""Stream candidates on a sky map: the listed ones, and new ones picked by clicking.

- ``plot_candidates`` draws the candidates of the CSV file
  (``catalogs.stream_candidates``) over a skyproj sky map, e.g. the
  corrected map of ``plotting.plot_generic_search.plot_generic_search_residual_map``:
  the great-circle arc between the two endpoints of each, its endpoints and
  its id;
- ``EndpointPicker`` gives the (ra, dec) of two clicks on such a map: the
  endpoints of a new candidate. It needs an interactive backend
  (``%matplotlib widget`` in a notebook, from ipympl).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.patheffects import withStroke

from streameucliddes.catalogs.stream_candidates import (
    DEFAULT_CANDIDATES_PATH,
    candidate_endpoints,
    load_candidates,
)
from streameucliddes.plotting import plot_style  # noqa: F401  (sets the figure style)
from streameucliddes.plotting.plot_stream_frame_search import draw_stream_track_on_sky
from streameucliddes.plotting.plot_style import CATEGORICAL_COLORS
from streameucliddes.stream_frames import phi12_to_radec, radec_to_phi12, stream_frame_from_endpoints

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    import pandas as pd
    from matplotlib.artist import Artist
    from matplotlib.backend_bases import MouseEvent
    from numpy.typing import ArrayLike, NDArray

CANDIDATE_COLOR = CATEGORICAL_COLORS[1]  # listed candidates (the known streams: CATEGORICAL_COLORS[0])
PICK_COLOR = CATEGORICAL_COLORS[4]  # the endpoints being picked
CANDIDATE_ARC_STYLE = {"color": CANDIDATE_COLOR, "lw": 1.5, "ls": "-", "alpha": 0.9}
CANDIDATE_ENDPOINT_STYLE = {"ls": "none", "marker": "o", "ms": 4, "color": CANDIDATE_COLOR}
PICK_ARC_STYLE = {"color": PICK_COLOR, "lw": 1.5, "ls": "-", "alpha": 0.9}
PICK_ENDPOINT_STYLE = {"ls": "none", "marker": "o", "ms": 7, "mfc": "none", "mec": PICK_COLOR, "mew": 1.5}
ARC_POINTS = 100
LABEL_FONTSIZE = 9  # ids of the candidates, outlined in white to stand out on the map
LEFT_BUTTON, RIGHT_BUTTON = 1, 3
# Press and release closer than this: a click (an endpoint), not a drag (zoom or pan). Below matplotlib's
# 5 pixels, under which the zoom tool does not zoom: a click never zooms.
CLICK_TOLERANCE_PIXELS = 4.0


def endpoints_arc_radec(
    endpoint_1: ArrayLike, endpoint_2: ArrayLike, n_points: int = ARC_POINTS
) -> tuple[NDArray, NDArray]:
    """(ra, dec) of the great-circle arc between two endpoints (deg).

    The phi2 = 0 line of their frame (``stream_frames.stream_frame_from_endpoints``),
    between their phi1.

    Parameters
    ----------
    endpoint_1, endpoint_2 : array_like
        (ra, dec) (deg).
    n_points : int, optional
        Number of points.

    Returns
    -------
    ra, dec : numpy.ndarray
        (deg).
    """
    frame = stream_frame_from_endpoints(tuple(endpoint_1), tuple(endpoint_2))
    phi1, _ = radec_to_phi12([endpoint_1[0], endpoint_2[0]], [endpoint_1[1], endpoint_2[1]], frame)
    arc_phi1 = np.linspace(phi1[0], phi1[1], n_points)
    return phi12_to_radec(arc_phi1, np.zeros(n_points), frame)


def plot_candidates(
    ax: Any,
    candidates: pd.DataFrame | None = None,
    *,
    path: str | Path = DEFAULT_CANDIDATES_PATH,
    label_ids: bool = True,
    arc_style: dict[str, Any] | None = None,
    endpoint_style: dict[str, Any] | None = None,
) -> list[Artist]:
    """Overplot the listed candidates on a sky map: arc between the endpoints, endpoints, id.

    Parameters
    ----------
    ax : skyproj.SkyAxes
        Axes of a sky map, e.g. from ``plot_generic_search_residual_map``;
        its ``plot`` and ``text`` take (ra, dec) in degrees.
    candidates : pandas.DataFrame, optional
        From ``catalogs.stream_candidates.load_candidates``. Default: those
        of the file ``path``.
    path : str or pathlib.Path, optional
        CSV file read when ``candidates`` is not given.
    label_ids : bool, optional
        Write the id of each candidate next to its first endpoint.
    arc_style, endpoint_style : dict, optional
        Arguments of ``ax.plot`` updating ``CANDIDATE_ARC_STYLE`` and
        ``CANDIDATE_ENDPOINT_STYLE``.

    Returns
    -------
    artists : list of matplotlib.artist.Artist
        The arcs, endpoints and labels added; empty when there is no
        candidate.
    """
    if candidates is None:
        candidates = load_candidates(path)
    arc_style = {**CANDIDATE_ARC_STYLE, **(arc_style or {})}
    endpoint_style = {**CANDIDATE_ENDPOINT_STYLE, **(endpoint_style or {})}
    artists: list[Artist] = []
    for index, (_, candidate) in enumerate(candidates.iterrows()):
        endpoint_1, endpoint_2 = candidate_endpoints(candidate)
        artists += draw_stream_track_on_sky(ax, *endpoints_arc_radec(endpoint_1, endpoint_2),
                                            label="candidates" if index == 0 else None, **arc_style)
        artists += ax.plot([endpoint_1[0], endpoint_2[0]], [endpoint_1[1], endpoint_2[1]], **endpoint_style)
        if label_ids:
            artists.append(ax.text(*endpoint_1, f" {int(candidate['id'])}", color=arc_style["color"],
                                   fontsize=LABEL_FONTSIZE, fontweight="bold", ha="left", va="bottom",
                                   path_effects=[withStroke(linewidth=2.5, foreground="white")]))
    return artists


def sky_axes_to_radec(ax: Any, x: ArrayLike, y: ArrayLike) -> tuple[NDArray, NDArray]:
    """(ra, dec) of points of a skyproj map given in its data coordinates (deg).

    The inverse of the projection of the map (``ax.projection``, as skyproj
    does for the coordinates shown under the cursor). Points outside the sky
    (beyond the edge of the projection) are NaN.

    Parameters
    ----------
    ax : skyproj.SkyAxes
        Axes of a sky map.
    x, y : array_like
        Data coordinates, e.g. ``event.xdata`` and ``event.ydata`` of a click.

    Returns
    -------
    ra, dec : numpy.ndarray
        (deg), ra in [0, 360).
    """
    x, y = np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
    lonlat = ax.projection.transform_points(x, y, inverse=True)
    back = ax.projection.transform_points(lonlat[:, 0], lonlat[:, 1])  # outside the sky: not the same point back
    outside = ~(np.isclose(back[:, 0], x) & np.isclose(back[:, 1], y))
    ra, dec = lonlat[:, 0] % 360.0, lonlat[:, 1].copy()
    ra[outside], dec[outside] = np.nan, np.nan
    return ra, dec


class EndpointPicker:
    """Two clicks on a sky map: the endpoints of a new candidate.

    Left click: an endpoint (marked); the second one draws the great-circle
    arc between them; a third click starts a new pair. Right click: removes
    the last endpoint. A click is a press and a release at the same place
    (within ``CLICK_TOLERANCE_PIXELS``), so it works with the zoom or pan
    tool of the toolbar on: a drag zooms or pans and adds no endpoint.
    Needs an interactive backend: ``%matplotlib widget`` (ipympl) in a
    notebook.

    Parameters
    ----------
    ax : skyproj.SkyAxes
        Axes of a sky map, e.g. from ``plot_generic_search_residual_map``.

    Attributes
    ----------
    points : list of tuple of float
        (ra, dec) of the endpoints clicked (deg), at most two.

    Examples
    --------
    >>> fig, ax = plot_generic_search_residual_map(result)  # doctest: +SKIP
    >>> picker = EndpointPicker(ax)  # doctest: +SKIP
    >>> # two clicks on the map, then:
    >>> add_candidate(*picker.endpoints, best_dm=17.0)  # doctest: +SKIP
    """

    def __init__(self, ax: Any) -> None:
        self.ax = ax
        self.points: list[tuple[float, float]] = []
        self._artists: list[Artist] = []
        self._press: tuple[float, float, Any] | None = None  # (x, y) in pixels and button of the last press
        canvas = ax.figure.canvas
        self._connections = [canvas.mpl_connect("button_press_event", self._on_press),
                             canvas.mpl_connect("button_release_event", self._on_release)]

    @property
    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The two endpoints clicked, (ra, dec) each (deg).

        Raises
        ------
        ValueError
            Fewer than two endpoints clicked.
        """
        if len(self.points) < 2:
            raise ValueError(f"{len(self.points)} endpoint(s) clicked: click two points on the map.")
        return self.points[0], self.points[1]

    def add_point(self, ra: float, dec: float) -> None:
        """Add an endpoint, as a left click at (ra, dec) does (a third one starts a new pair)."""
        if len(self.points) == 2:
            self.points = []
        self.points.append((float(ra) % 360.0, float(dec)))
        self._redraw()

    def remove_last_point(self) -> None:
        """Remove the last endpoint, as a right click does."""
        if self.points:
            self.points.pop()
            self._redraw()

    def disconnect(self) -> None:
        """Stop listening to the clicks; the marks stay."""
        for connection in self._connections:
            self.ax.figure.canvas.mpl_disconnect(connection)

    def _on_press(self, event: MouseEvent) -> None:
        """Keep where a press in the map starts: a click if released at the same place."""
        self._press = (event.x, event.y, event.button) if event.inaxes is self.ax else None

    def _on_release(self, event: MouseEvent) -> None:
        """A click (press and release at the same place): left adds an endpoint, right removes the last one."""
        press, self._press = self._press, None
        if press is None or event.inaxes is not self.ax or event.xdata is None or event.button != press[2]:
            return
        if np.hypot(event.x - press[0], event.y - press[1]) > CLICK_TOLERANCE_PIXELS:  # a drag: zoom or pan
            return
        if event.button == RIGHT_BUTTON:
            self.remove_last_point()
        elif event.button == LEFT_BUTTON:
            ra, dec = sky_axes_to_radec(self.ax, event.xdata, event.ydata)
            if np.isfinite(ra[0]):
                self.add_point(ra[0], dec[0])

    def _redraw(self) -> None:
        """Marks of the endpoints, and their arc when there are two."""
        for artist in self._artists:
            artist.remove()
        self._artists = []
        if self.points:
            ra, dec = zip(*self.points)
            self._artists += self.ax.plot(ra, dec, **PICK_ENDPOINT_STYLE)
        if len(self.points) == 2:
            self._artists += draw_stream_track_on_sky(self.ax, *endpoints_arc_radec(*self.points), **PICK_ARC_STYLE)
        self.ax.figure.canvas.draw_idle()

"""Run configurations of the searches: dictionaries of their arguments.

``KNOWN_STREAM_SEARCH_CONFIG`` holds the arguments of
``workflows.run_known_stream_search.run_all_known_stream_searches``, with
their default values. Copy it, update what you need, and pass it on, e.g. in
a notebook::

    config = {**KNOWN_STREAM_SEARCH_CONFIG, "n_workers": 8, "smoothing_deg": 0.5}
    table = run_all_known_stream_searches(stars, **config)

A YAML file (``configs/``) is optional: ``RunConfig.from_yaml`` reads the
values it changes, for the scripts.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from streameucliddes.analysis.cmd_match_filter import DEFAULT_MATCH_FILTER
from streameucliddes.analysis.known_stream_search import (
    DEFAULT_BIN_SIZE_DEG,
    DEFAULT_SMOOTHING_DEG,
    SELECTION_ARGUMENTS,
)
from streameucliddes.workflows.run_known_stream_search import (
    DEFAULT_FOOTPRINT_NSIDE,
    DEFAULT_OUTPUT_DIR,
    FIGURE_DPI,
    FRAME_DURATION_MS,
)

if TYPE_CHECKING:
    from pathlib import Path

# Arguments of run_all_known_stream_searches (and of the functions it calls), by default
KNOWN_STREAM_SEARCH_CONFIG = {
    # Match filter (see analysis.cmd_match_filter): Euclid VIS vs VIS - H
    "match_filter": copy.deepcopy(DEFAULT_MATCH_FILTER),
    # None: errors <= 0.2 on the bands of the match filter (catalog_preparation.default_error_cuts)
    "cuts": None,
    # None: catalogs.isochrone_parameters.default_isochrone_grid(), for the age and Z not known
    "default_grid": None,
    # None: the track's distance moduli in the footprint +- 1 mag, every 0.1 (prepare_search)
    "distance_moduli": None,
    "bin_size_deg": DEFAULT_BIN_SIZE_DEG,
    "smoothing_deg": DEFAULT_SMOOTHING_DEG,
    # Default footprint: from the star counts of the catalog
    "footprint_nside": DEFAULT_FOOTPRINT_NSIDE,
    "output_dir": DEFAULT_OUTPUT_DIR,
    "n_workers": 1,
    "frame_duration_ms": FRAME_DURATION_MS,
    "dpi": FIGURE_DPI,
}
# Other arguments a configuration may set: the tracks, and those of the track selection
# (select_galstreams_in_footprint: dm_range, min_length_deg, add_ibata2024)
KNOWN_STREAM_SEARCH_OTHER_ARGUMENTS = ("track_names", *SELECTION_ARGUMENTS)


@dataclass
class RunConfig:
    """Configuration of a search run, read from YAML with ``from_yaml``.

    Attributes
    ----------
    search : str
        ``"known_stream"`` (the generic search: later).
    kwargs : dict
        Arguments of the run: ``KNOWN_STREAM_SEARCH_CONFIG`` updated by the
        file. ``run_all_known_stream_searches(stars, **config.kwargs)``.

    Notes
    -----
    Adapted from celestial_currents ``config.py::RunConfig`` (``from_yaml``
    with ``yaml.safe_load``).
    """

    search: str = "known_stream"
    kwargs: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(KNOWN_STREAM_SEARCH_CONFIG))

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Read a run configuration from a YAML file.

        The file gives ``search`` and the values that change: its other keys
        are those of ``KNOWN_STREAM_SEARCH_CONFIG`` or
        ``KNOWN_STREAM_SEARCH_OTHER_ARGUMENTS``. ``distance_moduli`` may be
        ``{start, stop, step}`` (stop included). An empty file: the defaults.

        Parameters
        ----------
        path : str or pathlib.Path
            YAML file, e.g. ``configs/known_stream_search.yaml``.

        Returns
        -------
        config : RunConfig
            The configuration.

        Raises
        ------
        ValueError
            A search other than ``"known_stream"``, or keys of no argument.
        """
        import yaml

        with open(path) as file:
            values = yaml.safe_load(file) or {}
        search = values.pop("search", "known_stream")
        if search != "known_stream":
            raise ValueError(f"search {search!r}: only 'known_stream' for now")
        unknown = sorted(set(values) - set(KNOWN_STREAM_SEARCH_CONFIG) - set(KNOWN_STREAM_SEARCH_OTHER_ARGUMENTS))
        if unknown:
            raise ValueError(
                f"Keys {unknown} are not arguments of the run: "
                f"{sorted(KNOWN_STREAM_SEARCH_CONFIG) + list(KNOWN_STREAM_SEARCH_OTHER_ARGUMENTS)}"
            )
        if isinstance(values.get("distance_moduli"), dict):
            grid = values["distance_moduli"]
            values["distance_moduli"] = np.arange(grid["start"], grid["stop"] + grid["step"] / 2, grid["step"])
        return cls(search=search, kwargs={**copy.deepcopy(KNOWN_STREAM_SEARCH_CONFIG), **values})

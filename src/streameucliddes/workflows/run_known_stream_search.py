"""Known-stream search of galstreams streams: maps per isochrone over the dm grid, figures, videos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any

    import pandas as pd
    from numpy.typing import NDArray

    from streameucliddes.workflows.run_config import RunConfig

if TYPE_CHECKING:  # galstreams: new dependency of phase 3, not installed yet
    import galstreams


def run_known_stream_search(
    catalog: pd.DataFrame,
    track_name: str,
    config: RunConfig,
    *,
    mws: galstreams.MWStreams,
    population_table: pd.DataFrame,
    footprint_mask: NDArray[np.bool_],
    nest: bool,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Known-stream search of one galstreams track: one set of maps per isochrone.

    Uses the track's ``stream_frame`` (a gala ``GreatCircleICRSFrame``) as
    the stream frame, and builds the window
    (``catalogs.galstreams_catalog.galstreams_phi12_window``), the footprint
    on the grid (``catalogs.survey_footprint.survey_footprint_phi12_mask``)
    and the isochrones of the stream
    (``catalogs.isochrone_parameters.known_stream_isochrone_parameters``).

    Then, for each isochrone: the maps over the grid of distance moduli, one
    at a time (``analysis.known_stream_search.known_stream_search_dm_scan``),
    drawn and saved
    (``plotting.plot_known_stream_search.plot_known_stream_search_single_dm``);
    the gif (``plotting.make_videos.make_gif_from_plots``); and the maps in
    one file
    (``save_known_stream_search_maps``).

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars, e.g. ``catalogs.euclid_des_catalog.get_eucliddes_stars()``.
    track_name : str
        galstreams track, as the ``track_name`` column of
        ``catalogs.galstreams_catalog.select_galstreams_in_footprint``.
    config : RunConfig
        Known-stream search configuration.
    mws : galstreams.MWStreams
        From ``catalogs.galstreams_catalog.load_galstreams``.
    population_table : pandas.DataFrame
        From ``catalogs.isochrone_parameters.load_stream_population_table``.
    footprint_mask : numpy.ndarray of bool
        HEALPix mask of the footprint, True outside.
    nest : bool
        HEALPix ordering of ``footprint_mask``.
    output_dir : str or pathlib.Path
        The outputs of each isochrone go to
        ``output_dir/<track_name>/age<age>_z<Z>/``.

    Returns
    -------
    summary : dict
        ``track_name``, ``window``, and ``isochrones``: one dict per
        isochrone, with ``isochrone``, ``figure_paths``, ``gif_path``
        and ``maps_path``.

    Notes
    -----
    Code to reuse: ``workflows.run_generic_search.run_generic_search`` (loop
    over the scan, figures); the sweep cells of celestial_currents
    ``notebooks/candidate_investigations/full_search/template.ipynb``.
    """
    raise NotImplementedError


def run_all_known_stream_searches(
    catalog: pd.DataFrame,
    config: RunConfig,
    *,
    nest: bool,
    output_dir: str | Path,
    mws: galstreams.MWStreams | None = None,
    population_table: pd.DataFrame | None = None,
    footprint_mask: NDArray[np.bool_] | None = None,
) -> pd.DataFrame:
    """Known-stream search of every galstreams track crossing the footprint.

    Selects the tracks (``catalogs.galstreams_catalog.select_galstreams_in_footprint``,
    or those of ``config.streams``), then runs ``run_known_stream_search`` for
    each.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars, e.g. ``catalogs.euclid_des_catalog.get_eucliddes_stars()``.
    config : RunConfig
        Known-stream search configuration.
    nest : bool
        HEALPix ordering of ``footprint_mask``.
    output_dir : str or pathlib.Path
        One subdirectory per track.
    mws : galstreams.MWStreams, optional
        Default: ``catalogs.galstreams_catalog.load_galstreams()``.
    population_table : pandas.DataFrame, optional
        Default: ``catalogs.isochrone_parameters.load_stream_population_table()``.
    footprint_mask : numpy.ndarray of bool, optional
        Default: ``catalogs.survey_footprint.survey_footprint_healpix_mask``
        of the catalog, at ``config.maps["footprint_nside"]``.

    Returns
    -------
    table : pandas.DataFrame
        The selection, with one row per track and isochrone, and the output
        paths of ``run_known_stream_search``.
    """
    raise NotImplementedError


def save_known_stream_search_maps(
    results: Iterable[tuple[float, Mapping[str, Any]]],
    output_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save the maps of a known-stream search, one per distance modulus, in one file.

    For one stream and one isochrone. Compressed numpy file (``.npz``)
    holding ``density``, the maps stacked along the distance modulus, of
    shape (n_dm, n_phi1, n_phi2), NaN outside the footprint;
    ``distance_modulus``; the ``phi1`` and ``phi2`` bin centres; and
    ``metadata``, a JSON string: track, window, bin size, smoothing,
    isochrone, match filter, plus the ``metadata`` given. The phi1/phi2 maps
    are small: they all fit in memory. Maps following the track
    (``known_stream_search_dm_along_track``) are saved the same way, with
    ``dm_offset`` and ``dm_along_phi1`` (n_offset, n_phi1) instead of
    ``distance_modulus``.

    Read it back with ``numpy.load`` and ``json.loads(file["metadata"].item())``.

    Parameters
    ----------
    results : iterable of (distance_modulus, result)
        As yielded by ``analysis.known_stream_search.known_stream_search_dm_scan``
        (or ``(dm_offset, result)`` by ``known_stream_search_dm_along_track``),
        for one isochrone, on one grid.
    output_path : str or pathlib.Path
        File to write (its folder is created).
    metadata : mapping, optional
        Added to the file, e.g. the name of the run. JSON-serialisable.

    Returns
    -------
    output_path : pathlib.Path
        The file written.

    Raises
    ------
    ValueError
        No result, or results of several isochrones, match filters or grids.

    Notes
    -----
    Adapted from the output block of celestial_currents
    ``scripts/run_matched_filter_maps.py::main`` (all the maps of a run in
    one file, with their axes).
    """
    densities, keys, dm_along_phi1, first = [], [], [], None
    for key, result in results:  # one map at a time: a generator is not kept
        if first is None:
            first = result
        else:
            for name in SAME_IN_A_MAPS_FILE:
                if not _same(result[name], first[name]):
                    raise ValueError(
                        f"Results with different {name}: save one file per isochrone and grid "
                        f"({first[name]!r} and {result[name]!r})."
                    )
        densities.append(np.ma.filled(np.ma.asarray(result["density"], dtype=float), np.nan))
        keys.append(float(key))
        if "dm_along_phi1" in result:
            dm_along_phi1.append(np.asarray(result["dm_along_phi1"], dtype=float))
    if first is None:
        raise ValueError("No result to save.")

    along_track = "dm_along_phi1" in first
    info = {
        "track_name": first.get("track_name"),
        "window": list(first["window"]),
        "bin_size_deg": first["bin_size_deg"],
        "smoothing_deg": first["smoothing_deg"],
        "isochrone": dict(first["isochrone"]),
        "match_filter": first["match_filter"],
        **dict(metadata or {}),
    }
    arrays = {
        "density": np.stack(densities),
        "phi1": np.asarray(first["phi1_centres"], dtype=float),
        "phi2": np.asarray(first["phi2_centres"], dtype=float),
        "metadata": np.array(json.dumps(info, default=_json_value)),
    }
    if along_track:
        arrays.update(dm_offset=np.array(keys), dm_along_phi1=np.stack(dm_along_phi1))
    else:
        arrays["distance_modulus"] = np.array(keys)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:  # a file object: numpy does not append .npz
        np.savez_compressed(file, **arrays)
    return output_path


# Parameters shared by every map of a maps file
SAME_IN_A_MAPS_FILE = ("isochrone", "match_filter", "window", "bin_size_deg", "smoothing_deg", "track_name")


def _json_value(value):
    """numpy values in JSON: their Python value (arrays as lists), else their text."""
    return value.tolist() if hasattr(value, "tolist") else str(value)


def _same(value, other):
    """Equal values, tuples and lists alike (e.g. a window or a match filter)."""
    return json.dumps(value, sort_keys=True, default=_json_value) == json.dumps(other, sort_keys=True, default=_json_value)

"""Known-stream search of galstreams streams: maps over the dm grid, figures, videos, cubes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path
    from typing import Any

    import numpy as np
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
    """Known-stream search of one galstreams track.

    Builds the stream frame (``stream_frames.rotation_matrix_from_astropy_frame``
    of the track's ``stream_frame``), the window
    (``catalogs.galstreams_catalog.galstreams_phi12_window``), the isochrones
    (``isochrone_parameters.known_stream_isochrone_parameters``) and the
    footprint on the grid
    (``catalogs.survey_footprint.survey_footprint_phi12_mask``). Then, one
    distance modulus at a time
    (``analysis.known_stream_search.known_stream_search_dm_scan``), draws and
    saves the map (``plotting.plot_known_stream_search.plot_known_stream_search_single_dm``).
    Finally, makes the videos (``plotting.make_videos.make_gif_from_plots``
    and ``make_mp4``) and saves the cube (``save_known_stream_search_cube``).

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
        From ``catalogs.stream_populations.load_stream_population_table``.
    footprint_mask : numpy.ndarray of bool
        HEALPix mask of the footprint, True outside.
    nest : bool
        HEALPix ordering of ``footprint_mask``.
    output_dir : str or pathlib.Path
        The outputs go to ``output_dir/<track_name>/``.

    Returns
    -------
    summary : dict
        ``track_name``, ``window``, ``isochrones``, ``figure_paths``,
        ``gif_path``, ``mp4_path`` and ``cube_path``.

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
        Default: ``catalogs.stream_populations.load_stream_population_table()``.
    footprint_mask : numpy.ndarray of bool, optional
        Default: ``catalogs.survey_footprint.survey_footprint_healpix_mask``
        of the catalog, at ``config.maps["footprint_nside"]``.

    Returns
    -------
    table : pandas.DataFrame
        The selection, one row per track, with the output paths of
        ``run_known_stream_search``.
    """
    raise NotImplementedError


def save_known_stream_search_cube(
    results: Iterable[tuple[float, Mapping[str, Any]]],
    output_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save the maps of a known-stream search, one per distance modulus, as one cube.

    Compressed numpy file (``.npz``, proposal): ``density`` of shape (n_dm,
    n_phi1, n_phi2), NaN outside the footprint; ``distance_modulus``; the
    ``phi1`` and ``phi2`` bin centres; and the metadata (stream, window,
    isochrones, match filter) as JSON. The phi1/phi2 maps are small, so the
    whole cube fits in memory.

    Parameters
    ----------
    results : iterable of (distance_modulus, result)
        As yielded by ``analysis.known_stream_search.known_stream_search_dm_scan``.
    output_path : str or pathlib.Path
        File to write.
    metadata : mapping, optional
        Added to the file, e.g. the track name.

    Returns
    -------
    output_path : pathlib.Path
        The file written.

    Notes
    -----
    Code to reuse: the output block of celestial_currents
    ``scripts/run_matched_filter_maps.py::main`` (one cube per run, with its
    axes).
    """
    raise NotImplementedError

"""Known-stream search of galstreams streams: maps per isochrone over the dm grid, figures, gifs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from streameucliddes.analysis.cmd_match_filter import match_filter_columns, match_filter_polygons
from streameucliddes.analysis.known_stream_search import (
    DEFAULT_SMOOTHING_DEG,
    SELECTION_ARGUMENTS,
    known_stream_search_dm_scan,
    prepare_search,
)
from streameucliddes.catalogs.galstreams_catalog import (
    galstreams_default_tracks,
    load_galstreams,
    select_galstreams_in_footprint,
)
from streameucliddes.catalogs.survey_footprint import survey_footprint_healpix_mask
from streameucliddes.plotting.make_videos import make_gif_from_plots
from streameucliddes.plotting.plot_cmd import MATCH_FILTER_MARGIN_MAG, plot_match_filter
from streameucliddes.plotting.plot_known_stream_search import plot_known_stream_search_single_dm
from streameucliddes.stream_frames import radec_to_phi12

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from typing import Any

    import astropy.coordinates as coord
    import galstreams
    from numpy.typing import NDArray

    from streameucliddes.analysis.known_stream_search import PreparedSearch

DEFAULT_OUTPUT_DIR = "figures/outputs/known_stream_search"  # gitignored
FRAME_DURATION_MS = 300  # gifs: time per distance modulus
FIGURE_DPI = 110
# HEALPix footprint from the star counts (catalogs.survey_footprint), about 0.1 deg pixels
DEFAULT_FOOTPRINT_NSIDE = 512


def run_known_stream_search(
    catalog: pd.DataFrame | None = None,
    stream_frame: coord.BaseCoordinateFrame | None = None,
    window: tuple[float, float, float, float] | None = None,
    *,
    stream_name: str | None = None,
    search: PreparedSearch | None = None,
    distance_moduli: Iterable[float] | None = None,
    isochrone: Mapping[str, float] | None = None,
    smoothing_deg: float = DEFAULT_SMOOTHING_DEG,
    n_workers: int = 1,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    name: str | None = None,
    frame_duration_ms: int = FRAME_DURATION_MS,
    dpi: int = FIGURE_DPI,
    **prepare_kwargs: Any,
) -> dict[str, Any]:
    """Known-stream search of one stream, with its figures, gifs and maps file.

    The search is prepared once
    (``analysis.known_stream_search.prepare_search``: from the galstreams
    ``stream_name`` alone, the values not given come from galstreams), or
    given (``search``). Then, for each isochrone, in
    ``output_dir/<name>/age<age>_z<Z>/``:

    - the maps over the distance moduli
      (``analysis.known_stream_search.known_stream_search_dm_scan``), drawn
      (``plotting.plot_known_stream_search.plot_known_stream_search_single_dm``)
      in ``maps/``, and their gif ``<name>.gif``
      (``plotting.make_videos.make_gif_from_plots``, from the frames of this
      run only);
    - the match filter on the CMD of the stars of the window at each distance
      modulus (``plotting.plot_cmd.plot_match_filter``, fixed colour and
      magnitude ranges: those of the filters of the scan) in ``cmd/``, and
      their gif ``<name>_cmd.gif``;
    - the maps in one file, ``<name>_maps.npz``
      (``save_known_stream_search_maps``).

    Every figure is closed once saved.

    Parameters
    ----------
    catalog, stream_frame, window
        See ``prepare_search``. Not with ``search``.
    stream_name : str, optional
        galstreams stream or track name, or the name of another stream (see
        ``prepare_search``). Not with ``search``.
    search : PreparedSearch, optional
        Prepared search, e.g. from a notebook.
    distance_moduli : iterable of float, optional
        Default: those of the search.
    isochrone : mapping, optional
        The one isochrone ``{"age", "metallicity"}``. Default: those of the
        search.
    smoothing_deg : float, optional
        Gaussian smoothing of the maps (deg).
    n_workers : int, optional
        Processes over the distance moduli.
    output_dir : str or pathlib.Path, optional
        Directory of the outputs of every stream.
    name : str, optional
        Name of the stream folder and files. Default: the galstreams track
        name, else ``stream_name``.
    frame_duration_ms : int, optional
        Time per distance modulus in the gifs (ms).
    dpi : int, optional
        Resolution of the frames.
    **prepare_kwargs
        Other arguments of ``prepare_search`` (``match_filter``, ``cuts``,
        ``mws``, ``population_table``, ``default_grid``, ``bin_size_deg``,
        ``footprint_mask``, ``nest``, the selection arguments). Not with
        ``search``.

    Returns
    -------
    summary : dict
        ``name``, ``track_name``, ``window``, and ``isochrones``: one dict
        per isochrone, with ``isochrone``, ``distance_moduli``,
        ``figure_paths``, ``cmd_figure_paths``, ``gif_path``,
        ``cmd_gif_path`` and ``maps_path``.

    Raises
    ------
    ValueError
        Preparation arguments with ``search``, or no name for the outputs.

    Notes
    -----
    Adapted from ``workflows.run_generic_search.run_generic_search`` (loop
    over the scan, figures, gif) and the sweep cells of celestial_currents
    ``notebooks/candidate_investigations/full_search/template.ipynb``.
    """
    if search is None:
        search = prepare_search(catalog, stream_frame, window, stream_name=stream_name, **prepare_kwargs)
    else:
        given = [label for label, value in (("catalog", catalog), ("stream_frame", stream_frame),
                                            ("window", window), ("stream_name", stream_name)) if value is not None]
        if given or prepare_kwargs:
            raise ValueError(f"{given + list(prepare_kwargs)} prepare a search: not with search")
    name = name or search.track_name or stream_name
    if not name:
        raise ValueError("Give a name for the outputs of this stream.")
    distance_moduli = search.distance_moduli if distance_moduli is None else [float(dm) for dm in distance_moduli]
    isochrones = search.isochrones if isochrone is None else [dict(isochrone)]
    track_phi12 = None
    if search.track is not None:
        track_phi12 = radec_to_phi12(search.track.track.ra.deg, search.track.track.dec.deg, search.stream_frame)

    summaries = []
    for isochrone in isochrones:
        folder = Path(output_dir) / name / f"age{isochrone['age']:g}_z{isochrone['metallicity']:g}"
        (folder / "maps").mkdir(parents=True, exist_ok=True)
        (folder / "cmd").mkdir(exist_ok=True)
        color_lim, mag_lim = _cmd_ranges(search, distance_moduli, isochrone)

        results, figure_paths, cmd_figure_paths = [], [], []
        for dm, result in known_stream_search_dm_scan(
            distance_moduli=distance_moduli, isochrone=isochrone, search=search,
            smoothing_deg=smoothing_deg, n_workers=n_workers,
        ):
            results.append((dm, result))
            frame = f"{name}_dm{dm:05.2f}.png"  # 05.2f: the names sort as the distance moduli (gif from a pattern)
            fig, _ = plot_known_stream_search_single_dm(result, track_phi12=track_phi12)
            figure_paths.append(_save(fig, folder / "maps" / frame, dpi))
            fig, _ = plot_match_filter(dm, isochrone, search.match_filter, stars=search.stars,
                                       color_lim=color_lim, mag_lim=mag_lim)
            cmd_figure_paths.append(_save(fig, folder / "cmd" / frame, dpi))

        # From the frames of this run only: not those of an earlier run in the folder
        gif_path, cmd_gif_path = folder / f"{name}.gif", folder / f"{name}_cmd.gif"
        make_gif_from_plots(gif_path, pattern=figure_paths, duration=frame_duration_ms)
        make_gif_from_plots(cmd_gif_path, pattern=cmd_figure_paths, duration=frame_duration_ms)
        maps_path = save_known_stream_search_maps(results, folder / f"{name}_maps.npz", metadata={"name": name})
        summaries.append({
            "isochrone": dict(isochrone), "distance_moduli": list(distance_moduli),
            "figure_paths": figure_paths, "cmd_figure_paths": cmd_figure_paths,
            "gif_path": gif_path, "cmd_gif_path": cmd_gif_path, "maps_path": maps_path,
        })
    return {"name": name, "track_name": search.track_name, "window": search.window, "isochrones": summaries}


def run_all_known_stream_searches(
    catalog: pd.DataFrame,
    *,
    track_names: Sequence[str] | None = None,
    mws: galstreams.MWStreams | None = None,
    footprint_mask: NDArray[np.bool_] | None = None,
    nest: bool | None = None,
    footprint_nside: int = DEFAULT_FOOTPRINT_NSIDE,
    **run_kwargs: Any,
) -> pd.DataFrame:
    """Known-stream search of every galstreams stream crossing the footprint.

    Selects the tracks crossing the footprint
    (``catalogs.galstreams_catalog.select_galstreams_in_footprint``), keeps
    one per stream (``catalogs.galstreams_catalog.galstreams_default_tracks``:
    its galstreams default track, else its longest selected track); or the
    ``track_names`` given. Then runs
    ``run_known_stream_search`` for each, from the full catalog: only the
    stars of its window are kept, one stream at a time.

    ``workflows.run_config.KNOWN_STREAM_SEARCH_CONFIG`` holds the arguments
    of a run, to copy and update:
    ``run_all_known_stream_searches(catalog, **config)``.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Stars, with ``ra``, ``dec`` (deg), e.g.
        ``catalogs.euclid_des_catalog.get_eucliddes_stars()``.
    track_names : sequence of str, optional
        Tracks to search, among the selected ones (any track of a stream).
    mws : galstreams.MWStreams, optional
        Default: ``catalogs.galstreams_catalog.load_galstreams()``.
    footprint_mask : numpy.ndarray of bool, optional
        HEALPix mask, True outside the footprint (needs ``nest``). Default:
        ``catalogs.survey_footprint.survey_footprint_healpix_mask`` of the
        catalog at ``footprint_nside``, RING.
    nest : bool, optional
        HEALPix ordering of ``footprint_mask``.
    footprint_nside : int, optional
        nside of the default footprint.
    **run_kwargs
        Arguments of ``run_known_stream_search`` (and of ``prepare_search``),
        the same for every stream. The selection arguments (``dm_range``,
        ``min_length_deg``, ``add_ibata2024``) also go to the selection.

    Returns
    -------
    table : pandas.DataFrame
        The selection rows of the tracks searched, one row per track and
        isochrone, with ``age``, ``metallicity``, ``gif_path``,
        ``cmd_gif_path`` and ``maps_path``.

    Raises
    ------
    ValueError
        ``footprint_mask`` without ``nest``; ``track_names`` not in the
        selection.
    """
    mws = load_galstreams() if mws is None else mws
    if footprint_mask is None:
        nest = False if nest is None else nest
        footprint_mask = survey_footprint_healpix_mask(
            catalog["ra"].to_numpy(), catalog["dec"].to_numpy(), footprint_nside, nest=nest
        )
    elif nest is None:
        raise ValueError("A footprint_mask needs nest.")
    selection_kwargs = {key: run_kwargs[key] for key in SELECTION_ARGUMENTS if key in run_kwargs}
    selection = select_galstreams_in_footprint(mws, footprint_mask, nest=nest, **selection_kwargs)
    if track_names is None:
        tracks = galstreams_default_tracks(selection, mws)
    else:
        missing = sorted(set(track_names) - set(selection["track_name"]))
        if missing:
            raise ValueError(f"Tracks not selected by select_galstreams_in_footprint: {missing}")
        tracks = selection[selection["track_name"].isin(track_names)]

    rows = []
    for _, track in tracks.iterrows():
        summary = run_known_stream_search(
            catalog, stream_name=track["track_name"], mws=mws, footprint_mask=footprint_mask, nest=nest, **run_kwargs
        )
        for outputs in summary["isochrones"]:
            rows.append({
                **track.to_dict(),
                "age": outputs["isochrone"]["age"], "metallicity": outputs["isochrone"]["metallicity"],
                "gif_path": outputs["gif_path"], "cmd_gif_path": outputs["cmd_gif_path"],
                "maps_path": outputs["maps_path"],
            })
    return pd.DataFrame(rows, columns=[*selection.columns, "age", "metallicity", "gif_path", "cmd_gif_path", "maps_path"])


def _cmd_ranges(
    search: PreparedSearch, distance_moduli: Sequence[float], isochrone: Mapping[str, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Colour and magnitude ranges of the CMD frames of a scan: the same for every frame.

    Those of the polygons over the scan plus ``plotting.plot_cmd.MATCH_FILTER_MARGIN_MAG``;
    the faint end at most the faintest star of the window plus that margin.
    """
    vertices = np.concatenate([
        polygon for dm in distance_moduli for polygon in match_filter_polygons(dm, isochrone, search.match_filter)
    ])
    margin = MATCH_FILTER_MARGIN_MAG
    color_lim = (float(vertices[:, 0].min() - margin), float(vertices[:, 0].max() + margin))
    faint = vertices[:, 1].max() + margin
    reference_mag = search.stars[match_filter_columns(search.match_filter)[0]].to_numpy()
    if np.isfinite(reference_mag).any():
        faint = min(faint, np.nanmax(reference_mag) + margin)
    return color_lim, (float(vertices[:, 1].min() - margin), float(faint))


def _save(fig, path: Path, dpi: int) -> Path:
    """Save a figure and close it."""
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


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

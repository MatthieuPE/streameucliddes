"""Generic stream search over the footprint: maps per isochrone over the dm grid, figures, gifs."""

from __future__ import annotations

import gc
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from streameucliddes.analysis.generic_stream_search import (
    CMD_SAMPLE_SIZE,
    DEFAULT_POLYNOMIAL_DEGREE,
    DEFAULT_SMOOTHING_DEG,
    generic_search_cmd_stars,
    generic_search_dm_scan,
    prepare_generic_search,
)
from streameucliddes.catalogs.galstreams_catalog import (
    galstreams_default_tracks,
    load_galstreams,
    select_galstreams_in_footprint,
)
from streameucliddes.plotting.make_videos import make_gif_from_plots
from streameucliddes.plotting.plot_generic_search import (
    plot_generic_search_residual_map,
    plot_generic_search_result,
)
from streameucliddes.workflows.run_known_stream_search import (
    FIGURE_DPI,
    FRAME_DURATION_MS,
    _cmd_ranges,
    _json_value,
    _same,
    _save,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import galstreams

    from streameucliddes.analysis.generic_stream_search import PreparedGenericSearch
    from typing import Any

DEFAULT_OUTPUT_DIR = "figures/outputs/generic_search"  # gitignored
DEFAULT_NAME = "generic"


def run_generic_search(
    catalog: pd.DataFrame | None = None,
    *,
    search: PreparedGenericSearch | None = None,
    distance_moduli: Iterable[float] | None = None,
    isochrone: Mapping[str, float] | None = None,
    smoothing_deg: float = DEFAULT_SMOOTHING_DEG,
    polynomial_degree: int = DEFAULT_POLYNOMIAL_DEGREE,
    n_workers: int = 1,
    map_key: str = "residual",
    known_tracks: bool = True,
    mws: galstreams.MWStreams | None = None,
    cmd_sample_size: int = CMD_SAMPLE_SIZE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    name: str = DEFAULT_NAME,
    frame_duration_ms: int = FRAME_DURATION_MS,
    dpi: int = FIGURE_DPI,
    **prepare_kwargs: Any,
) -> dict[str, Any]:
    """Generic search over the footprint, with its figures, gif and maps file, per isochrone.

    The search is prepared once
    (``analysis.generic_stream_search.prepare_generic_search``), or given
    (``search``). Then, for each isochrone, in
    ``output_dir/<name>/age<age>_z<Z>/``:

    - one figure per distance modulus
      (``plotting.plot_generic_search.plot_generic_search_result``: CMD
      selection, map before correction, fitted background, corrected map) in
      ``frames/``, and their gif ``<name>.gif``
      (``plotting.make_videos.make_gif_from_plots``, from the frames of this
      run only);
    - one figure per distance modulus of the corrected map alone, with the
      known stream tracks
      (``plotting.plot_generic_search.plot_generic_search_residual_map``) in
      ``frames_residual/``, and their gif ``<name>_residual.gif``;
    - the map ``map_key`` of every distance modulus in one file,
      ``<name>_<map_key>.parquet`` (``save_generic_search_healpix_maps``).

    The known stream tracks are the galstreams tracks crossing the footprint
    (``catalogs.galstreams_catalog.select_galstreams_in_footprint``, with its
    defaults), one per stream
    (``catalogs.galstreams_catalog.galstreams_default_tracks``).

    The maps are computed one at a time
    (``analysis.generic_stream_search.generic_search_dm_scan``): each is
    drawn, written to the file, then dropped. The CMD panel shows at most
    ``cmd_sample_size`` stars (``generic_search_cmd_stars``), with ranges
    fixed over the scan.

    Parameters
    ----------
    catalog : pandas.DataFrame, optional
        Stars: prepared here. Not with ``search``.
    search : PreparedGenericSearch, optional
        The search, prepared once, e.g. in a notebook.
    distance_moduli : iterable of float, optional
        Default: those of the search.
    isochrone : mapping, optional
        The one isochrone ``{"age", "metallicity"}``. Default: those of the
        search.
    smoothing_deg, polynomial_degree
        See ``generic_search_single_dm``.
    n_workers : int, optional
        Processes over the maps (see ``generic_search_dm_scan``).
    map_key : str, optional
        Map saved in the Parquet file: ``"residual"``, ``"density"`` or
        ``"background"``.
    known_tracks : bool, optional
        Draw the known stream tracks on the corrected maps.
    mws : galstreams.MWStreams, optional
        Default: ``catalogs.galstreams_catalog.load_galstreams()``, when
        ``known_tracks``.
    cmd_sample_size : int, optional
        Maximum number of stars of the CMD panel.
    output_dir : str or pathlib.Path, optional
        Directory of the outputs.
    name : str, optional
        Name of the run folder and files.
    frame_duration_ms : int, optional
        Time per distance modulus in the gif (ms).
    dpi : int, optional
        Resolution of the frames.
    **prepare_kwargs
        Arguments of ``prepare_generic_search`` (``match_filter``, ``cuts``,
        ``default_grid``, ``nside``, ``footprint_mask``, ``nest``, ...). Not
        with ``search``.

    Returns
    -------
    summary : dict
        ``name``, and ``isochrones``: one dict per isochrone, with
        ``isochrone``, ``distance_moduli``, ``figure_paths``, ``gif_path``,
        ``residual_figure_paths``, ``residual_gif_path`` and ``maps_path``.

    Raises
    ------
    ValueError
        Preparation arguments with ``search``.

    Notes
    -----
    Adapted from the old ``run_generic_search`` (loop over the scan, figures,
    gif) and ``workflows.run_known_stream_search.run_known_stream_search``.
    """
    if search is None:
        if catalog is None:
            raise ValueError("Give a catalog or a prepared search.")
        search = prepare_generic_search(catalog, **prepare_kwargs)
    elif catalog is not None or prepare_kwargs:
        given = (["catalog"] if catalog is not None else []) + list(prepare_kwargs)
        raise ValueError(f"{given} prepare a search: not with search")
    distance_moduli = search.distance_moduli if distance_moduli is None else [float(dm) for dm in distance_moduli]
    isochrones = search.isochrones if isochrone is None else [dict(isochrone)]
    cmd_stars = generic_search_cmd_stars(search, cmd_sample_size)
    tracks = _known_tracks(search.footprint_mask, mws) if known_tracks else []

    summaries = []
    for isochrone in isochrones:
        folder = Path(output_dir) / name / f"age{isochrone['age']:g}_z{isochrone['metallicity']:g}"
        (folder / "frames").mkdir(parents=True, exist_ok=True)
        (folder / "frames_residual").mkdir(exist_ok=True)
        color_lim, mag_lim = _cmd_ranges(cmd_stars, search.match_filter, distance_moduli, isochrone)
        figure_paths, residual_figure_paths = [], []

        def drawn(scan: Iterator[tuple[float, dict]]) -> Iterator[tuple[float, dict]]:
            """The scan, each map drawn and saved before it goes on to the maps file."""
            for dm, result in scan:
                fig, _ = plot_generic_search_result(result, cmd_stars=cmd_stars, color_lim=color_lim, mag_lim=mag_lim)
                # 05.2f: the names sort as the distance moduli
                figure_paths.append(_save(fig, folder / "frames" / f"{name}_dm{dm:05.2f}.png", dpi))
                fig, _ = plot_generic_search_residual_map(result, tracks=tracks)
                residual_figure_paths.append(
                    _save(fig, folder / "frames_residual" / f"{name}_residual_dm{dm:05.2f}.png", dpi)
                )
                yield dm, result
                gc.collect()  # the maps of this distance modulus, before the next one

        maps_path = save_generic_search_healpix_maps(
            drawn(generic_search_dm_scan(
                distance_moduli=distance_moduli, isochrone=isochrone, search=search,
                smoothing_deg=smoothing_deg, polynomial_degree=polynomial_degree, n_workers=n_workers,
            )),
            folder / f"{name}_{map_key}.parquet", map_key=map_key, metadata={"name": name},
        )
        gif_path = folder / f"{name}.gif"
        make_gif_from_plots(gif_path, pattern=figure_paths, duration=frame_duration_ms)
        residual_gif_path = folder / f"{name}_residual.gif"
        make_gif_from_plots(residual_gif_path, pattern=residual_figure_paths, duration=frame_duration_ms)
        summaries.append({
            "isochrone": dict(isochrone), "distance_moduli": list(distance_moduli),
            "figure_paths": figure_paths, "gif_path": gif_path,
            "residual_figure_paths": residual_figure_paths, "residual_gif_path": residual_gif_path,
            "maps_path": maps_path,
        })
    return {"name": name, "isochrones": summaries}


def _known_tracks(
    footprint_mask: np.ndarray, mws: galstreams.MWStreams | None
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(stream name, ra, dec) of the galstreams tracks crossing a RING footprint, one per stream."""
    mws = load_galstreams() if mws is None else mws
    selection = select_galstreams_in_footprint(mws, footprint_mask, nest=False)
    return [
        (row.stream_name, mws[row.track_name].track.ra.deg, mws[row.track_name].track.dec.deg)
        for row in galstreams_default_tracks(selection, mws).itertuples()
    ]


def save_generic_search_healpix_maps(
    results: Mapping[float, Mapping[str, Any]] | Iterable[tuple[float, Mapping[str, Any]]],
    output_path: str | Path,
    *,
    map_key: str = "residual",
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save the HEALPix maps of a generic search, one per distance modulus, in one file.

    For one isochrone. Parquet table in the format of the celestial_currents
    outputs: a column ``hpx<nside>`` of the pixels inside the footprint, in
    the ordering of the maps (RING for ``generic_search_single_dm``: no
    conversion), and one column per distance modulus, named by its value
    with two decimals (e.g. ``"16.50"``). The parameters are in the file
    metadata (key ``"streameucliddes"``, JSON): ``map_key``, ``nside``,
    ``nest``, isochrone, match filter, smoothing, polynomial degree, plus the
    ``metadata`` given.

    Read it back with ``pandas.read_parquet`` and
    ``json.loads(pyarrow.parquet.read_schema(path).metadata[b"streameucliddes"])``.

    Parameters
    ----------
    results : mapping or iterable of (distance_modulus, result)
        Results of ``analysis.generic_stream_search.generic_search_dm_scan``
        (or ``generic_search_single_dm``), for one isochrone: read one at a
        time, only the map ``map_key`` of each is kept.
    output_path : str or pathlib.Path
        Parquet file to write (its folder is created).
    map_key : str, optional
        Map to save: ``"residual"``, ``"density"`` or ``"background"``.
    metadata : mapping, optional
        Added to the file metadata. JSON-serialisable.

    Returns
    -------
    output_path : pathlib.Path
        The file written.

    Raises
    ------
    ValueError
        No result, or results of several isochrones, footprints or
        parameters.

    Notes
    -----
    Adapted from the output block of celestial_currents
    ``scripts/run_matched_filter_maps.py::main`` (one Parquet table per
    isochrone, a column of pixels and one per distance modulus); there in
    NESTED, here in the ordering of the maps.
    """
    items = results.items() if isinstance(results, Mapping) else results
    columns, first = {}, None
    for dm, result in items:  # one map at a time: a generator is not kept
        if first is None:
            first = result
            inside = ~np.asarray(result["footprint_mask"], dtype=bool)
            columns[f"hpx{result['nside']}"] = np.flatnonzero(inside)
        else:
            for name in SAME_IN_A_MAPS_FILE:
                if not _same(result[name], first[name]):
                    raise ValueError(
                        f"Results with different {name}: save one file per isochrone "
                        f"({first[name]!r} and {result[name]!r})."
                    )
            if not np.array_equal(result["footprint_mask"], first["footprint_mask"]):
                raise ValueError("Results with different footprints.")
        name = f"{float(dm):.2f}"
        if name in columns:
            raise ValueError(f"Distance modulus {name} twice.")
        columns[name] = np.ma.getdata(result[map_key])[inside].astype(float)
    if first is None:
        raise ValueError("No result to save.")

    info = {
        "map_key": map_key,
        "nside": first["nside"],
        "nest": first["nest"],
        "isochrone": dict(first["isochrone"]),
        "match_filter": first["match_filter"],
        "smoothing_deg": first["smoothing_deg"],
        "polynomial_degree": first["polynomial_degree"],
        **dict(metadata or {}),
    }
    table = pa.Table.from_pandas(pd.DataFrame(columns, copy=False), preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"streameucliddes": json.dumps(info, default=_json_value).encode()}
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    return output_path


# Parameters shared by every map of a maps file
SAME_IN_A_MAPS_FILE = ("isochrone", "match_filter", "nside", "nest", "smoothing_deg", "polynomial_degree")

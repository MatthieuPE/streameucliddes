"""Generic stream search over a grid of distance moduli: figures, gif and results."""

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

from streameucliddes.analysis.generic_stream_search import generic_search_dm_scan
from streameucliddes.plotting.make_videos import make_gif
from streameucliddes.plotting.plot_generic_search import plot_generic_search_result

from streameucliddes.workflows.run_known_stream_search import _json_value, _same

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any


def run_generic_search(
    data,
    distance_modulus_array,
    match_filter_parameters,
    find_stream_kwargs=None,
    plot=False,
    save_figures=False,
    fig_folder="../../figures",
    plot_kwargs=None,
    make_gif_flag=False,
    gif_kwargs=None,
    keep_results=False,
    verbose=True,
):
    """
    Run generic_search_single_dm() over a grid of distance moduli, optionally plotting
    (and saving) each frame and building a gif from the saved figures.

    The scan itself (dust correction and cuts computed once, then one search per
    distance modulus) is analysis.generic_stream_search.generic_search_dm_scan.

    Parameters
    ----------
    data : pandas.DataFrame
    distance_modulus_array : iterable of float
    match_filter_parameters : dict
        Base kwargs for match_filter.build_match_filter; 'distance_modulus'
        is overridden for each value in `distance_modulus_array`.
    find_stream_kwargs : dict, optional
        Extra kwargs forwarded to generic_search_single_dm() (e.g. mag_cols, cuts,
        use_dust_corrected, nside...).
    plot : bool
        If True, display each frame (via plot_generic_search_result, show=True).
    save_figures : bool
        If True, save each frame to `fig_folder`.
    plot_kwargs : dict, optional
        Extra kwargs forwarded to plot_generic_search_result().
    make_gif_flag : bool
        If True, build a gif from `fig_folder` once the scan is done
        (requires save_figures=True).
    gif_kwargs : dict, optional
        Extra kwargs forwarded to make_gif().
    keep_results : bool
        If True, retain the full generic_search_single_dm() output (HEALPix maps, etc.)
        for every distance modulus. Default False: only a lightweight summary
        ('distance_kpc', 'n_selected', 'fname') is kept per distance modulus,
        and the heavy per-frame result is dropped as soon as it has been
        plotted/saved -- important when scanning many distance moduli over a
        large catalog, since retaining every frame's HEALPix maps would
        otherwise grow without bound.

    Returns
    -------
    results_by_dm : dict[float, dict]
        generic_search_single_dm() output (or the lightweight summary, see `keep_results`)
        keyed by distance modulus.
    """
    plot_kwargs = dict(plot_kwargs or {})
    gif_kwargs = dict(gif_kwargs or {})

    results_by_dm = {}
    for dm, results in generic_search_dm_scan(
        data,
        distance_modulus_array,
        match_filter_parameters,
        find_stream_kwargs=find_stream_kwargs,
        verbose=verbose,
    ):
        fname_path = None
        if plot or save_figures:
            _, _, fname_path = plot_generic_search_result(
                results,
                save=save_figures,
                show=plot,
                fig_folder=fig_folder,
                **plot_kwargs,
            )

        if keep_results:
            results_by_dm[dm] = results
        else:
            results_by_dm[dm] = {
                "distance_kpc": results["distance_kpc"],
                "n_selected": results["n_selected"],
                "fname": fname_path,
            }
            del results

        # Release any lingering matplotlib/array reference cycles before the next frame.
        gc.collect()

    if make_gif_flag:
        make_gif(figures_folder=fig_folder, **gif_kwargs)

    return results_by_dm


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

"""Run configurations of the searches, read from YAML (``configs/``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class RunConfig:
    """Configuration of a generic or a known-stream search run.

    Read from a YAML file of ``configs/`` with ``from_yaml``.

    Attributes
    ----------
    search : str
        ``"generic"`` or ``"known_stream"``.
    distance_moduli : dict
        ``{"start", "stop", "step"}`` of the grid. Plan: 15 to 22.
    output_dir : str
        Directory of the figures, videos and map files.
    n_workers : int
        Processes over the distance moduli.
    match_filter_parameters : dict
        Generic search: arguments of ``streamobs.match_filter.build_match_filter``
        (age, metallicity, surveys, bands).
    search_kwargs : dict
        Generic search: the ``find_stream_kwargs`` of
        ``workflows.run_generic_search.run_generic_search`` (magnitude and
        coordinate columns, nside, smoothing, cuts, polynomial degree). Its
        ``use_dust_corrected`` must be False: the Euclid x DES magnitudes are
        already dereddened.
    match_filter : dict
        Known-stream search: see
        ``analysis.cmd_match_filter.select_stars_in_match_filter``.
    isochrones : dict
        Known-stream search: default grid, ``{"ages_gyr", "metallicities"}``
        (see ``isochrone_parameters.default_isochrone_grid``). One set of maps
        per isochrone.
    cuts : dict
        Known-stream search: see ``catalogs.catalog_cuts.get_cut_mask``.
    maps : dict
        Known-stream search: ``bin_size_deg`` and ``smoothing_deg`` of the
        phi1/phi2 maps, ``footprint_nside`` of the HEALPix footprint.
    streams : list of str or None
        Known-stream search: galstreams tracks to search. None: every track of
        ``catalogs.galstreams_catalog.select_galstreams_in_footprint``.

    Notes
    -----
    Code to reuse: celestial_currents ``config.py::RunConfig`` (dataclasses
    filled by ``from_yaml``).
    """

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Read a run configuration from a YAML file.

        Parameters
        ----------
        path : str or pathlib.Path
            YAML file, e.g. ``configs/known_stream_search.yaml``.

        Returns
        -------
        config : RunConfig
            The configuration.

        Notes
        -----
        Code to reuse: celestial_currents ``config.py::RunConfig.from_yaml``.
        Needs pyyaml (new dependency of phase 3).
        """
        raise NotImplementedError

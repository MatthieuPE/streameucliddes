#!/usr/bin/env python3
"""Generic stream search over the survey footprint, from a YAML configuration.

Usage
-----
python scripts/run_generic_search.py configs/generic_search.yaml
python scripts/run_generic_search.py configs/generic_search.yaml --workers 8 \\
    --output-dir figures/outputs/generic_search

Reads the configuration (``workflows.run_config.RunConfig.from_yaml``) and the
Euclid x DES stars (``catalogs.euclid_des_catalog.get_eucliddes_stars``), runs
``workflows.run_generic_search.run_generic_search`` over the grid of distance
moduli, and saves the maps in one file
(``workflows.run_generic_search.save_generic_search_healpix_maps``).

Code to reuse: celestial_currents ``scripts/run_matched_filter_maps.py``
(argparse, logging, output directory).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the generic search from the command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments. Default: ``sys.argv[1:]``.

    Returns
    -------
    status : int
        Exit status.
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())

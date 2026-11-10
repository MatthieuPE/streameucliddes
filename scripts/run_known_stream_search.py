#!/usr/bin/env python3
"""Known-stream search of the galstreams streams, from a YAML configuration.

Usage
-----
python scripts/run_known_stream_search.py configs/known_stream_search.yaml
python scripts/run_known_stream_search.py configs/known_stream_search.yaml \\
    --track <galstreams track name> --workers 8 --output-dir figures/outputs/known_stream_search

Reads the configuration (``workflows.run_config.RunConfig.from_yaml``) and the
Euclid x DES stars (``catalogs.euclid_des_catalog.get_eucliddes_stars``), then
runs ``workflows.run_known_stream_search.run_known_stream_search`` for the
given tracks, or ``run_all_known_stream_searches`` for every track crossing
the footprint.

Code to reuse: celestial_currents ``scripts/run_matched_filter_maps.py``
(argparse, logging, output directory).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the known-stream search from the command line.

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

"""Patches of ugali, applied when streameucliddes is imported."""

import functools
import glob
import os

from ugali.isochrone.model import Isochrone

# (isochrone class, directory, file prefix) -> grid of available (age, metallicity)
_ISOCHRONE_GRIDS = {}


def cache_isochrone_grids():
    """
    Cache the grid of (age, metallicity) available for the ugali isochrones.

    Each time an isochrone is built, ugali scans its whole directory (~11k files) to
    find the available (age, metallicity): ~30 ms of the ~37 ms of the build, for
    each of the many isochrones built by the injection and the match filters. The
    scan only depends on the directory, so it is done once per isochrone class and
    directory. The isochrones are still built as new objects: nothing else is shared.

    Patches ``Isochrone.create_grid`` for every user of ugali (streamobs included);
    called when streameucliddes is imported.
    """
    create_grid = Isochrone.create_grid
    if hasattr(create_grid, "__wrapped__"):
        return  # already cached

    @functools.wraps(create_grid)
    def cached_create_grid(self, abins=None, zbins=None):
        if abins is not None or zbins is not None:
            return create_grid(self, abins, zbins)
        key = (type(self), self.get_dirname(), self._prefix)
        if key not in _ISOCHRONE_GRIDS:
            grid = create_grid(self)
            pattern = os.path.join(self.get_dirname(), f"{self._prefix}_*.dat")
            if next(glob.iglob(pattern), None) is None:
                # No isochrone file: ugali falls back on this isochrone's own (age,
                # metallicity). Not cached, so that files downloaded later are seen.
                return grid
            _ISOCHRONE_GRIDS[key] = grid
        return tuple(values.copy() for values in _ISOCHRONE_GRIDS[key])

    Isochrone.create_grid = cached_create_grid

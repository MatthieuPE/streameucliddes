"""Dust correction of the magnitudes.

correct_dust_map belongs here. analysis.generic_stream_search calls it, but it was
never defined in this package (see docs/dev/package_structure.md): calling it
raises AttributeError, as data_manipulation.correct_dust_map did."""

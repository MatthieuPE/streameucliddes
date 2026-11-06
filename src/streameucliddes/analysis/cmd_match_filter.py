"""Colour-magnitude match filters."""

import functools

from streamobs.match_filter import build_match_filter

@functools.lru_cache(maxsize=1024)
def _match_filter_polygon(distance_modulus, age, metallicity, band_1, band_2, survey_1, survey_2, color_cut=True, red_cap_value=None):
    """
    build_match_filter polygon, cached: it only depends on these parameters, which
    repeat over a scan (e.g. the same distance modulus for every surface brightness).
    """
    mf_polygon = build_match_filter(
        distance_modulus=distance_modulus,
        age=age,
        metallicity=metallicity,
        band_1=band_1,
        band_2=band_2,
        survey_1=survey_1,
        survey_2=survey_2,
        color_cut=color_cut,
        red_cap_value=red_cap_value
    )
    mf_polygon.setflags(write=False)  # shared between calls
    return mf_polygon

"""SNR of mock streams injected in the Euclid x DES stars."""

import copy
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc
import logging

from streameucliddes.analysis.catalog_preparation import default_error_cuts, prepare_match_filter_catalog
from streameucliddes.analysis.cmd_match_filter import DEFAULT_MATCH_FILTER, select_stars_in_match_filter
from streameucliddes.catalogs.catalog_cuts import cut_flow
from streameucliddes.catalogs.euclid_des_catalog import get_eucliddes_stars
from streameucliddes.injection import inject

# Euclid x DES stars used as background by get_stream_snr: read on first use (see
# _get_background), not at import
STARS_BACKGROUND = None
# Frame of the phi1/phi2 columns of STARS_BACKGROUND (None: not computed yet)
_BACKGROUND_FRAME = None


END1 = coord.SkyCoord(ra=67 * u.deg, dec=-47 * u.deg, frame="icrs")
END2 = coord.SkyCoord(ra=55 * u.deg, dec=-49 * u.deg, frame="icrs")
GC_FRAME_DEFAULT = gc.GreatCircleICRSFrame.from_endpoints(END1, END2)

STREAM_CONFIG = {
    "stream_length": 7.0,  # degrees
    "stream_width": 0.2,  # degrees
    "N": None,  # Number of stars, used when surface_brightness is None
    "surface_brightness": 33.5,
    "distance_modulus":{
    "center": {"type": "Constant", "value": 16.8},
    "spread": {"type": "Constant", "value": 0.0},
},
    "isochrone": {
        "name": "Marigo2017",
        "age": 12.0,  # Gyr
        "z": 0.0006,  # metallicity
        "surveys": ["des", "euclid"],
    }
}

DEFAULT_MATCH_FILTERS = [DEFAULT_MATCH_FILTER]
# Default phi2 window: +- 1.4 stream_width, maximizes the SNR of a gaussian signal on
# a uniform background
PHI2_WIDTH_FACTOR = 1.4


################################
########### SNR ##############
################################

def get_stream_snr(seed=None, reload_data=False, match_filter_list=None, **kwargs):
    """
    Estimate the SNR of a mock stream injected in the Euclid x DES stars.

    SNR = N_stream / sqrt(N_background), counting the stars passing the cuts and
    inside the match filter.

    Parameters
    ----------
    seed : int, optional
        Seed of the stream generation and injection.
    reload_data : bool
        Read the Euclid x DES stars again instead of using the cached ones.
    match_filter_list : list of dict, optional
        ``{"name": ..., "reference_band": (namespace, band), "color_bands":
        [(namespace, band), ...]}``; the stars must be inside the polygon of every
        colour. Default: DEFAULT_MATCH_FILTERS (DES g vs g - r).
    **kwargs
        - keys of STREAM_CONFIG (e.g. ``surface_brightness``, ``N``,
          ``distance_modulus``): update the stream configuration, nested
          dictionaries key by key;
        - ``gc_frame``: frame of the stream (default: GC_FRAME_DEFAULT);
        - ``cuts``: cuts of get_cut_mask, replacing the default ones on the same
          columns (errors of the match-filter bands <= 0.2, ``|phi2|`` <= 1.4
          stream_width);
        - anything else is passed to inject.inject_stream (e.g. ``survey``,
          ``bands``).

    Returns
    -------
    snr_results : dict
        SNR for each match filter name.
    stream_config : dict
        Stream configuration used, with its ``gc_frame``.
    """
    if match_filter_list is None:
        match_filter_list = DEFAULT_MATCH_FILTERS
    # kwargs is local to this call: popping does not modify the caller's arguments
    cuts = kwargs.pop("cuts", {})
    gc_frame = kwargs.pop("gc_frame", GC_FRAME_DEFAULT)

    # Update stream configuration with the kwargs among its keys
    stream_config = copy.deepcopy(STREAM_CONFIG)
    for key in STREAM_CONFIG:
        if key in kwargs:
            value = kwargs.pop(key)
            if isinstance(stream_config[key], dict):
                stream_config[key].update(value)
            else:
                stream_config[key] = value
            logging.info(f"Updated stream_config[{key}] with {value}")

    # The remaining kwargs are injection options
    injected_catalog, _ = inject.inject_stream(
        stream_config=stream_config, seed=seed,
        detection_mag_cut=[], # if we want to avoid applying snr cuts
        gc_frame=gc_frame,
        **kwargs
    )
    stream_config["gc_frame"] = gc_frame

    # Stream stars observed by every injected survey
    flag_observed = np.logical_and.reduce(
        [
            injected_catalog[col].to_numpy(dtype=bool)
            for col in injected_catalog.columns
            if col.endswith("_flag_observed")
        ]
    )
    stars_background = _get_background(gc_frame, reload_data=reload_data)

    distance_modulus = stream_config["distance_modulus"]["center"]["value"]
    isochrone = {"age": stream_config["isochrone"]["age"], "metallicity": stream_config["isochrone"]["z"]}
    phi2_max = PHI2_WIDTH_FACTOR * stream_config["stream_width"]

    snr_results = {}
    for match_filter in match_filter_list:
        logging.info(f"Applying match filter: {match_filter}")
        # Default cuts (errors of the match-filter bands), replaced by the given ones on the same columns
        mf_cuts = {**default_error_cuts(match_filter), "phi2": (-phi2_max, phi2_max), **cuts}

        n_stream = _count_selected(
            injected_catalog.loc[flag_observed], mf_cuts, distance_modulus, isochrone, match_filter
        )
        n_background = _count_selected(stars_background, mf_cuts, distance_modulus, isochrone, match_filter)
        logging.info(
            f"{match_filter['name']}: {n_stream} stream stars, {n_background} background stars"
        )
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Background cut flow:\n{cut_flow(stars_background, mf_cuts)}")

        snr_results[match_filter["name"]] = _compute_SNR(n_stream, n_background)

    return snr_results, stream_config


def _get_background(gc_frame, reload_data=False):
    """
    Euclid x DES stars, read once and cached in STARS_BACKGROUND, with phi1/phi2 in
    ``gc_frame``.

    phi1/phi2 are written in place, so that the cached catalog is never copied, and
    only recomputed when ``gc_frame`` differs from the frame they were computed in.
    """
    global STARS_BACKGROUND, _BACKGROUND_FRAME
    if STARS_BACKGROUND is None or reload_data:
        STARS_BACKGROUND = get_eucliddes_stars()
        _BACKGROUND_FRAME = None

    if _BACKGROUND_FRAME is None or not gc_frame.is_equivalent_frame(_BACKGROUND_FRAME):
        stream_coord = coord.SkyCoord(
            ra=STARS_BACKGROUND["ra"].to_numpy() * u.deg,
            dec=STARS_BACKGROUND["dec"].to_numpy() * u.deg,
            frame="icrs",
        ).transform_to(gc_frame)
        STARS_BACKGROUND["phi1"] = stream_coord.phi1.deg
        STARS_BACKGROUND["phi2"] = stream_coord.phi2.deg
        _BACKGROUND_FRAME = gc_frame
    return STARS_BACKGROUND


def _count_selected(catalog, cuts, distance_modulus, isochrone, match_filter):
    """
    Number of stars of ``catalog`` passing ``cuts`` and inside the match filter, with
    the preparation and selection of the searches.
    """
    stars = prepare_match_filter_catalog(catalog, match_filter, cuts=cuts)
    return int(select_stars_in_match_filter(stars, distance_modulus, isochrone, match_filter).sum())

def _compute_SNR(n_stream, n_background):
    """SNR of ``n_stream`` stream stars over ``n_background`` background stars."""
    if n_stream == 0:
        return 0.0  # No stream star selected, even without background
    if n_background == 0:
        return np.inf  # Avoid division by zero
    return n_stream / np.sqrt(n_background)


def scan_snr_grid(axes=("surface_brightness", "distance_modulus"), axis_values=None, seed=None, **kwargs):
    """
    Scan the SNR of a mock stream over a grid of values for two axes.

    Parameters
    ----------
    axes : tuple of str
        Names of the two axes to scan, among the keys of STREAM_CONFIG (default:
        ("surface_brightness", "distance_modulus")). A distance_modulus value is the
        constant center of the distance modulus distribution.
    axis_values : dict, optional
        Values to scan for each axis, replacing the default ranges
        (surface_brightness: 31 -> 34.5, distance_modulus: 15 -> 20, 5 values each).
    seed : int, optional
        Seed of the stream generation and injection.
    **kwargs
        Additional keyword arguments passed to get_stream_snr.

    Returns
    -------
    results : dict
        For each match filter name, ``{"snr": 2D array, axes[0]: 1D array, axes[1]:
        1D array}``, with ``snr[i, j]`` the SNR at ``axes[0][i]`` and ``axes[1][j]``
        (e.g. ``plt.pcolormesh(res[axes[1]], res[axes[0]], res["snr"])``).
    """
    axis_values = {
        "surface_brightness": np.linspace(31.0, 34.5, 5),
        "distance_modulus": np.linspace(15.0, 20.0, 5),
        **(axis_values or {}),
    }
    values_0 = np.asarray(axis_values[axes[0]], dtype=float)
    values_1 = np.asarray(axis_values[axes[1]], dtype=float)

    results = {}
    for i, value_0 in enumerate(values_0):
        for j, value_1 in enumerate(values_1):
            grid_kwargs = {
                axes[0]: _axis_kwarg(axes[0], value_0),
                axes[1]: _axis_kwarg(axes[1], value_1),
            }
            snr_results, _ = get_stream_snr(seed=seed, **{**kwargs, **grid_kwargs})
            for name, snr in snr_results.items():
                if name not in results:
                    results[name] = {
                        "snr": np.full((len(values_0), len(values_1)), np.nan),
                        axes[0]: values_0,
                        axes[1]: values_1,
                    }
                results[name]["snr"][i, j] = snr

    return results


def _axis_kwarg(axis, value):
    """get_stream_snr kwarg setting ``axis`` of the stream configuration to ``value``."""
    if axis == "distance_modulus":
        return {"center": {"type": "Constant", "value": value}}
    return value

import copy
import functools
import os
import healpy as hp
import numpy as np
import pandas as pd
import skyproj
import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc
import logging

from streamobs.match_filter import build_match_filter, is_in_match_filter


from streameucliddes.analysis.euclid_des_preparation import get_eucliddes_stars
from streameucliddes.analysis.utils import cut_flow, get_cut_mask
from streameucliddes.injection import generate, inject


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

DEFAULT_MATCH_FILTERS = [
    {
        "name": "default_des_gr",
        "reference_band": ("des_yr6", "g"),
        "color_bands": [("des_yr6", "r")],
    },
]
DEFAULT_ERR_CUT = (None, 0.2)  # Default cut on the errors of the match-filter bands
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
          columns (errors of the match-filter bands <= 0.2, |phi2| <= 1.4
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
    age = stream_config["isochrone"]["age"]
    metallicity = stream_config["isochrone"]["z"]
    phi2_max = PHI2_WIDTH_FACTOR * stream_config["stream_width"]

    snr_results = {}
    for match_filter in match_filter_list:
        logging.info(f"Applying match filter: {match_filter}")
        namespace_1, band_1 = match_filter["reference_band"]

        # (band_1 column, band_2 column, polygon) for each colour of the match filter
        polygons = []
        for namespace_2, band_2 in match_filter["color_bands"]:
            mf_polygon = _match_filter_polygon(
                distance_modulus,
                age,
                metallicity,
                band_1,
                band_2,
                namespace_1.split("_")[0],  # isochrone survey: "des_yr6" -> "des"
                namespace_2.split("_")[0],
            )
            polygons.append(
                (f"{namespace_1}_{band_1}_obs", f"{namespace_2}_{band_2}_obs", mf_polygon)
            )

        # Default cuts, replaced by the given ones on the same columns
        mf_cuts = {
            f"{namespace}_{band}_err": DEFAULT_ERR_CUT
            for namespace, band in [match_filter["reference_band"], *match_filter["color_bands"]]
        }
        mf_cuts["phi2"] = (-phi2_max, phi2_max)
        mf_cuts.update(cuts)

        n_stream = _count_selected(injected_catalog, mf_cuts, polygons, mask=flag_observed)
        n_background = _count_selected(stars_background, mf_cuts, polygons)
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


def _count_selected(catalog, cuts, polygons, mask=None):
    """
    Number of stars of ``catalog`` passing ``cuts`` (and ``mask``) and inside every
    match-filter polygon of ``polygons`` (``(band_1 column, band_2 column, vertices)``).

    The cuts are applied first: the polygon test, the slow part, only runs on the
    stars left.
    """
    selected = get_cut_mask(catalog, cuts)
    if mask is not None:
        selected &= mask
    for band_1_col, band_2_col, mf_polygon in polygons:
        idx = np.flatnonzero(selected)
        selected[idx] = is_in_match_filter(
            catalog[band_1_col].to_numpy()[idx],
            catalog[band_2_col].to_numpy()[idx],
            polygon_vertices=mf_polygon,
        )
    return int(selected.sum())


@functools.lru_cache(maxsize=1024)
def _match_filter_polygon(distance_modulus, age, metallicity, band_1, band_2, survey_1, survey_2):
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
    )
    mf_polygon.setflags(write=False)  # shared between calls
    return mf_polygon


def _compute_SNR(n_stream, n_background):
    """SNR of ``n_stream`` stream stars over ``n_background`` background stars."""
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


################################
########### Others ############
################################

def get_healpy_data_from_table(
    data,
    columns=("ra", "dec"),
    nside=512,
    smoothing=False,
    smoothing_scale_degrees=0.3,
    footprint_mask=None,
):
    """
    Build HEALPix map from catalog with optional smoothing and footprint masking.

    Unobserved pixels are masked to prevent smoothing from bleeding values into the
    unobserved footprint. Returns a masked array where masked pixels indicate regions
    outside the survey footprint.

    Parameters
    ----------
    data : table
        Catalog with coordinate columns
    columns : tuple of str
        Column names for ra/dec (default: ("ra", "dec"))
    nside : int
        HEALPix resolution (default: 512)
    smoothing : bool
        Whether to apply Gaussian smoothing (default: False)
    smoothing_scale_degrees : float
        Gaussian smoothing width in degrees (default: 0.3)
    footprint_mask : array-like of bool, optional
        Pre-computed footprint mask where True = unobserved (masked), False = observed.
        If None (default), unobserved pixels are auto-detected as those with count=0.
        Provide this when zero-count pixels are physical (e.g., from deep observations)
        rather than indicating missing data.

    Returns
    -------
    hp_map : np.ma.MaskedArray
        HEALPix count map with unobserved pixels masked
    lon, lat : np.ndarray
        Longitude and latitude of HEALPix pixel centers (degrees)
    npix : int
        Total number of HEALPix pixels at this resolution
    """
    # Read coordinates and keep only valid values
    ra = np.asarray(data[columns[0]], dtype=float)
    dec = np.asarray(data[columns[1]], dtype=float)
    valid = np.isfinite(ra) & np.isfinite(dec) & (dec >= -90.0) & (dec <= 90.0)

    ra = ra[valid]
    dec = dec[valid]

    # Build HEALPix count map
    npix = hp.nside2npix(nside)
    pix_idx = hp.ang2pix(nside, ra, dec, lonlat=True, nest=False)
    hp_map = np.bincount(pix_idx, minlength=npix).astype(np.float32)
    pix = np.arange(npix)
    lon, lat = hp.pix2ang(nside, pix, lonlat=True)

    # Determine footprint mask
    if footprint_mask is None:
        # Auto-detect: pixels that received no observations
        footprint_mask = hp_map == 0
    else:
        # Use provided mask; coerce to boolean array
        footprint_mask = np.asarray(footprint_mask, dtype=bool)
        if footprint_mask.shape != (npix,):
            raise ValueError(
                f"footprint_mask must have shape ({npix},), got {footprint_mask.shape}"
            )

    # Apply smoothing if requested
    if smoothing:
        smoothing_scale_radian = 2 * np.pi / 360.0 * smoothing_scale_degrees
        # To smooth safely: fill masked (unobserved) pixels with a benign value before smoothing,
        # then restore the mask afterwards. healpy.smoothing does not handle NaN gracefully.
        hp_map_temp = hp_map.copy().astype(float)

        # Fill unobserved pixels with 0 (not NaN) before smoothing
        hp_map_temp[footprint_mask] = 0.0

        # Apply smoothing on the filled array
        hp_map_smooth = hp.sphtfunc.smoothing(
            hp_map_temp, sigma=np.radians(smoothing_scale_degrees)
        )

        # Return as masked array with unobserved region masked
        hp_map = np.ma.array(hp_map_smooth, mask=footprint_mask, fill_value=np.nan)
    else:
        # Return as masked array even without smoothing for consistency
        hp_map = np.ma.array(hp_map, mask=footprint_mask)

    return hp_map, lon, lat, npix
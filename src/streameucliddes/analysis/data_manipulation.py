import copy
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
from streameucliddes.analysis.utils import get_cut_mask
from streameucliddes.injection import generate, inject


STARS_BACKGROUND = get_eucliddes_stars()


END1 = coord.SkyCoord(ra=67 * u.deg, dec=-47 * u.deg, frame="icrs")
END2 = coord.SkyCoord(ra=55 * u.deg, dec=-49 * u.deg, frame="icrs")
GC_FRAME_DEFAULT = gc.GreatCircleICRSFrame.from_endpoints(END1, END2)

stream_coord = coord.SkyCoord(
ra=STARS_BACKGROUND["ra"].to_numpy() * u.deg,
dec=STARS_BACKGROUND["dec"].to_numpy() * u.deg,
frame="icrs",
).transform_to(GC_FRAME_DEFAULT)

STARS_BACKGROUND["phi1"] = stream_coord.phi1.deg
STARS_BACKGROUND["phi2"] = stream_coord.phi2.deg

STREAM_CONFIG = {
    "stream_length": 7.0,  # degrees
    "stream_width": 0.2,  # degrees
    #"N": 10000,  # Number of stars
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


################################
########### SNR ##############
################################

def get_stream_snr(seed=None, reload_data=False,match_filter_list=[{"name": "default_des_gr",
                                                                     "reference_band": ("des_yr6", "g"),
                                                                       "color_bands": [("des_yr6", "r")] }], **kwargs):
    if reload_data:
        stars_background = get_eucliddes_stars()
    else:
        stars_background = STARS_BACKGROUND


    # Update stream configuration with any additional parameters
    stream_config = copy.deepcopy(STREAM_CONFIG)
    # update each of the nested dictionaries in stream_config with kwargs if they exist
    for key in stream_config:
        if isinstance(stream_config[key], dict):
            stream_config[key].update(kwargs.get(key, {}))
            logging.info(f"Updated stream_config[{key}] with {kwargs.get(key, {})}")
        else:
            if key in kwargs:
                stream_config[key] = kwargs[key]
                logging.info(f"Updated stream_config[{key}] with {kwargs[key]}")

    # Update GC frame if provided in kwargs
    if not "gc_frame" in kwargs:
        kwargs["gc_frame"] = GC_FRAME_DEFAULT

    injected_catalog, metadata = inject.inject_stream(
        stream_config=stream_config, seed=seed, 
        detection_mag_cut=[], # if we want to avoid applying snr cuts
        **kwargs
    )
    gc_frame = metadata["gc_frame"]

    flag_obs_euclid = injected_catalog['euclid_q1_flag_observed']
    flag_obs_des = injected_catalog['des_yr6_flag_observed']
    flag_obs_cross = flag_obs_euclid & flag_obs_des
    observed_stream_both = injected_catalog[flag_obs_cross]

    observed_stream_both['is_stream'] = True
    stars_background['is_stream'] = False

    # Add phi1 and phi2 columns to the background stars if a different GC frame is used
    if gc_frame != GC_FRAME_DEFAULT:
        stream_coord = coord.SkyCoord(
        ra=stars_background["ra"].to_numpy() * u.deg,
        dec=stars_background["dec"].to_numpy() * u.deg,
        frame="icrs",
        ).transform_to(gc_frame)

        stars_background["phi1"] = stream_coord.phi1.deg
        stars_background["phi2"] = stream_coord.phi2.deg


    total_stars = pd.concat([observed_stream_both, stars_background])


    snr_results = {}
    for match_filter in match_filter_list:
        logging.info(f"Applying match filter: {match_filter}")

        DISTANCE_MODULUS = stream_config["distance_modulus"]["center"]["value"]
        AGE = stream_config["isochrone"]["age"]
        METALLICITY = stream_config["isochrone"]["z"]

        survey_1, band_1 = match_filter["reference_band"]
        band_1_col = f"{survey_1}_{band_1}_obs"
        band_1_err = f"{survey_1}_{band_1}_err"

        flag_mf = np.ones(len(total_stars), dtype=bool)  # Initialize flag for match filter
        for survey_2, band_2 in match_filter["color_bands"]:
            mf_polygon = build_match_filter(
            distance_modulus=DISTANCE_MODULUS,
            age=AGE,
            metallicity=METALLICITY,
            band_1=band_1,
            band_2=band_2,
            survey_1=survey_1,
            survey_2=survey_2,
            )

            band_2_col = f"{survey_2}_{band_2}_obs"

            # Selecting stars in the stream and background catalogs that are inside the match filter polygon
            flag_sub_mf = is_in_match_filter(
                total_stars[band_1_col],
                total_stars[band_2_col],
                polygon_vertices=mf_polygon,
            )

            flag_mf &= flag_sub_mf  # Combine flags for all color bands in the match filter

        cuts = kwargs.get("cuts", {})
        if band_1_err not in cuts:
            cuts[band_1_err] = (None, 0.2)  # Default error cut for band_1 if not provided
        for survey_2, band_2 in match_filter["color_bands"]:
            band_2_err = f"{survey_2}_{band_2}_err"
            if band_2_err not in cuts:
                cuts[band_2_err] = (None, 0.2)  # Default error cut for band_2 if not provided
        if "phi2" not in cuts:
            # Default cut for phi2 based on stream width if not provided. 
            # 1.4 factor to maximize SNR of gaussian signal on uniform background
            cuts["phi2"] = (-1.4*stream_config["stream_width"], 1.4*stream_config["stream_width"])
        mask_cuts = get_cut_mask(total_stars, cuts)
        flag_stream = total_stars['is_stream']

        logging.debug(f"{cut_flow(total_stars, cuts)} print to be improved")

        snr = _compute_SNR(total_stars, mask=mask_cuts&flag_mf, flag_stream=flag_stream)
        snr_results[match_filter["name"]] = snr

    return snr_results, stream_config


def _compute_SNR(data, mask=None, flag_stream=None):
    if mask is None:
        mask = np.ones(len(data), dtype=bool)
    if flag_stream is None:
        raise ValueError("flag_stream must be provided to compute SNR.")
    # Compute SNR for the selected data
    signal = len(data[mask & flag_stream])
    noise = len(data[mask & ~flag_stream])
    if noise == 0:
        return np.inf  # Avoid division by zero
    return signal / np.sqrt(noise)





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
# Generation of stream injection catalogs using streamobs

import numpy as np
from .inject_utils import convert_N_to_SurfaceBrightness, convert_SurfaceBrightness_to_N
from streamobs.model import  StreamModel
import pandas as pd


def generate_stream_catalog(method="uniform", stream_config=None, **kwargs):
    """
    Generate a mock catalog of stars for a stream using the streamobs package.

    Parameters
    ----------
    method : str
        The method to use for generating the stream. Options are "uniform" or "gaussian".
    stream_config : dict
        The stream configuration (the ``stream`` section consumed by
        :class:`~streamobs.model.StreamModel`).

    Returns
    -------
    catalog : np.ndarray
        A mock catalog of stars for the stream.
    """

    if method == "uniform":
        return generate_uniform_stream(stream_config, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

def generate_uniform_stream(
    stream_config,
    seed=None,
    rng=None,
    complete=False,
):
    if rng is None:
        rng = np.random.default_rng(seed)


    length = stream_config.get("stream_length", 20.0)  # Length of the stream in degrees
    width = stream_config.get("stream_width", 0.2)  # Width of the stream in degrees
    N = stream_config.get("N", None)  # Number of stars
    if N is None:
        surface_brightness = stream_config.get("surface_brightness", None)
        if surface_brightness is None:
            raise ValueError("Either N or surface_brightness must be provided.")
        N = convert_SurfaceBrightness_to_N(surface_brightness, **stream_config)

    phi1 = rng.uniform(-length / 2, length / 2, N)  # Uniform distribution along the stream length
    phi2 = rng.normal(0, width, N)
    catalog = pd.DataFrame({"phi1": phi1, "phi2": phi2})

    if complete:
        if "surveys" in stream_config["isochrone"]:
            raise ValueError("The 'surveys' key in the isochrone configuration is not supported for complete catalog generation. Please remove it from the isochrone configuration. Or use directly the injection method.")
        streammod = StreamModel(stream_config)

        data = streammod.complete_catalog(
            catalog,
            rng=rng,
        )
        return data
    return catalog
# Generation of stream injection catalogs using streamobs

import numpy as np
from .inject_utils import convert_N_to_SurfaceBrightness, convert_SurfaceBrightness_to_N
from streamobs.model import  StreamModel


def generate_stream_catalog(method="uniform", stream_parameters=None, **kwargs):
    """
    Generate a mock catalog of stars for a stream using the streamobs package.

    Parameters
    ----------
    method : str
        The method to use for generating the stream. Options are "uniform" or "gaussian".
    stream_parameters : dict
        A dictionary containing the parameters for the stream.

    Returns
    -------
    catalog : np.ndarray
        A mock catalog of stars for the stream.
    """

    if method == "uniform":
        return generate_uniform_stream(stream_parameters, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

def generate_uniform_stream(
    stream_parameters,
    seed=None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(seed)


    length = stream_parameters.get("stream_length", 20.0)  # Length of the stream in degrees
    width = stream_parameters.get("stream_width", 0.2)  # Width of the stream in degrees
    N = stream_parameters.get("N", None)  # Number of stars
    if N is None:
        surface_brightness = stream_parameters.pop("surface_brightness", None)
        if surface_brightness is None:
            raise ValueError("Either N or surface_brightness must be provided.")
        N = convert_SurfaceBrightness_to_N(surface_brightness, **stream_parameters)

    phi1 = rng.uniform(-length / 2, length / 2, N)  # Uniform distribution along the stream length
    phi2 = rng.normal(0, width, N)
    catalog = {"phi1": phi1, "phi2": phi2}

    streammod = StreamModel(stream_parameters)

    data = streammod.complete_catalog(
        catalog,
        rng=rng,
    )

    return data
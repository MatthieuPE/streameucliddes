import numpy as np
import ugali.isochrone

from streameucliddes.utils import fluxToMag, magToFlux


def _set_isochrone_for_ugali(isochrone, verbose=False):
    """
    Set the survey for the ugali isochrone.

    """
    # Set survey for ugali isochrone
    iso = isochrone.copy()
    if "surveys" in isochrone:
        if isinstance(isochrone["surveys"], dict):
            surveys = list(isochrone["surveys"].keys())
        elif isinstance(isochrone["surveys"], list):
            surveys = isochrone["surveys"]
        else:
            raise ValueError(
                "The 'surveys' key in the isochrone must be a list or a dict."
            )
        if len(surveys) > 0:
            iso["survey"] = surveys[0]  # Return the first survey found
            if verbose:
                print(f"Setting survey for ugali isochrone to {iso['survey']}.")
            iso.pop("surveys", None)  # Remove the surveys key to avoid confusion

    return iso


def convert_N_to_SurfaceBrightness(
    N,
    mag_bounds=(None, 24),
    surface=None,
    stream_length=None,
    stream_width=None,
    isochrone_config_path=None,
    band="r",
    isochrone=None,
    verbose=False,
    distance_modulus=None,
    **kwargs,
):
    if isochrone is None:
        if isochrone_config_path is None:
            raise ValueError(
                "Either isochrone_config_path or isochrone must be provided."
            )
        import yaml

        with open(isochrone_config_path, "r") as f:
            isochrone = yaml.safe_load(f)

    isochrone = _set_isochrone_for_ugali(isochrone)

    isochrone_ugali = ugali.isochrone.factory(**isochrone)
    distance_modulus = (
        distance_modulus["center"]["value"]
        if distance_modulus is not None
        else isochrone.get("distance_modulus", {})["center"]["value"]
    )
    isochrone_ugali.distance_modulus = distance_modulus

    # Surface estimation
    if surface is None:
        if stream_length is None or stream_width is None:
            raise ValueError(
                "If surface is not provided, stream_length and stream_width must be provided to estimate surface."
            )
        sigma_rad = np.deg2rad(stream_width)
        length_rad = np.deg2rad(stream_length)

        surface_sr = (
            length_rad * 2 * np.sin(sigma_rad)
        )  # Surface in steradians. Sinus because of sphere integration
        # It's equivalent when the width is small, since sin(sigma) ~ sigma

        surface_deg2 = surface_sr * (180 / np.pi) ** 2  # Convert surface to deg^2

        surface = surface_deg2 * (3600) ** 2  # Convert surface to arcsec^2
        if verbose:
            print(
                f"Estimated surface of the stream is {surface_deg2:.2f}deg^2 (length={stream_length}, width={stream_width})"
            )

    N_in_surface = int(
        N * 0.68
    )  # 68% of stars are within 1 sigma of the gaussian profile of the stream
    tot_magnitude = convert_N_to_luminosity(
        N_in_surface,
        isochrone_ugali=isochrone_ugali,
        mag_bounds=mag_bounds,
        verbose=verbose,
        band=band,
        **kwargs,
    )
    surface_brightness = tot_magnitude + 2.5 * np.log10(
        surface
    )  # Surface brightness in mag/arcdeg^2

    if verbose:
        print(
            f"Estimated surface brightness is {surface_brightness:.2f} mag/arcsec^2 for N={N}, N_in_surface={N_in_surface}, tot_magnitude={tot_magnitude:.2f}, surface={surface:.2f} arcsec^2"
        )

    return surface_brightness


def convert_N_to_luminosity(
    N, isochrone_ugali, mag_bounds=(None, 24), verbose=False, band="r", **kwargs
):
    mass_init, mass_pdf, mass_act, mag_g, mag_r = isochrone_ugali.sample(
        mass_steps=10000
    )
    mag_g, mag_r = (
        mag_g + isochrone_ugali.distance_modulus,
        mag_r + isochrone_ugali.distance_modulus,
    )
    if band == "g":
        mag1 = mag_g
    elif band == "r":
        mag1 = mag_r
    else:
        raise ValueError(f"Band '{band}' not recognized. Use 'g' or 'r'.")

    mask = np.ones_like(mag1, dtype=bool)
    if mag_bounds[0] is not None:
        mask &= mag1 > mag_bounds[0]
    if mag_bounds[1] is not None:
        mask &= mag1 < mag_bounds[1]
    mass_pdf_norm = mass_pdf / np.sum(
        mass_pdf
    )  # Normalize the mass pdf to get a proper probability distribution function
    number_of_stars_per_mass_bin = mass_pdf_norm * N

    # number_of_stars_per_mass_bin = mass_pdf * N # Number of stars corresponding to each mass step, given the pdf and total N
    number_of_stars_per_mass_bin_within_mag_bounds = number_of_stars_per_mass_bin[
        mask
    ]  # Number of stars corresponding to each mass step, but only for the mass steps that have mag within bounds

    # Select only the magnitudes corresponding to the mass steps that have mag within bounds
    mag_sel = mag1[mask]
    lum = magToFlux(
        mag_sel
    )  # Luminosity corresponding to the magnitude of each mass step

    L_tot = np.sum(number_of_stars_per_mass_bin_within_mag_bounds * lum)
    M_tot = fluxToMag(L_tot)

    return M_tot


def convert_SurfaceBrightness_to_N(
    target_surface_brightness,
    mag_bounds=(None, 24),
    surface=None,
    stream_length=None,
    stream_width=None,
    isochrone_config_path=None,
    band="r",
    isochrone=None,
    n_bracket=(10, 1e8),
    verbose=False,
    distance_modulus=None,
    **kwargs,
):
    """
    Numeric inverse of convert_N_SurfaceBrightness: the number of stars N whose
    stream (with the given isochrone/surface parameters) has the requested
    surface brightness.

    convert_N_SurfaceBrightness(N, ...) is monotonic in N (more stars -> more
    total flux -> brighter/smaller surface brightness), so this brackets and
    solves for N in log10(N) space with scipy.optimize.brentq, calling
    convert_N_SurfaceBrightness itself (same isochrone-loading path, mag_bounds,
    and surface as the forward function -- nothing here is duplicated).

    Parameters
    ----------
    target_surface_brightness : float
        Desired surface brightness, mag/arcsec^2 (same convention as
        convert_N_SurfaceBrightness's return value).
    mag_bounds, surface, stream_length, stream_width, isochrone_config_path,
    band, isochrone, verbose : see convert_N_SurfaceBrightness.
    n_bracket : tuple (N_min, N_max)
        Bracket searched in log10(N) space. Widen this if brentq raises a
        sign-mismatch error (the target surface brightness falls outside what
        this bracket can produce).

    Returns
    -------
    float
        N (number of stars) giving `target_surface_brightness`.
    """
    from scipy.optimize import brentq

    def _residual(log10_N):
        N = 10**log10_N
        sb = convert_N_to_SurfaceBrightness(
            N,
            mag_bounds=mag_bounds,
            surface=surface,
            stream_length=stream_length,
            stream_width=stream_width,
            isochrone_config_path=isochrone_config_path,
            band=band,
            isochrone=isochrone,
            verbose=False,
            distance_modulus=distance_modulus,
        )
        return sb - target_surface_brightness

    log10_N = brentq(_residual, np.log10(n_bracket[0]), np.log10(n_bracket[1]))
    N = 10**log10_N

    if verbose:
        print(
            f"N={N:.1f} gives surface brightness {target_surface_brightness:.2f} mag/arcsec^2"
        )

    return int(N)

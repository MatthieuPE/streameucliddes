"""Mock sky shared by the tests: a projection, a survey footprint, a polynomial
background density and a stream, plus fixtures sampling star positions from them."""

import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc
import healpy as hp
import numpy as np
import pandas as pd
import pytest
from numpy.polynomial import polynomial

NSIDE = 64
LON0, LAT0 = 50.0, -25.0  # patch centre (deg)
HALF_WIDTH = (20.0, 15.0)  # patch half-size in projected x, y (deg)

N_BACKGROUND = 100_000
BACKGROUND_DEGREE = 5

N_STREAM = 10_000
STREAM_ENDS = (
    (42.0, -20.0),
    (58.0, -30.0),
)  # (ra, dec) of the stream ends (deg), inside the footprint
STREAM_WIDTH = 0.2  # deg

MAG_G_RANGE = (18.0, 25.0)  # g magnitude
COLOR_RANGE = (-0.5, 1.5)  # g - r colour


# ===================================================
# Randomness
# ===================================================


@pytest.fixture(scope="session")
def seed():
    """Seed of every random draw in the mock sky, for reproducibility."""
    return 42


@pytest.fixture
def rng(seed):
    """Fresh seeded generator per test, so samples don't depend on the test order."""
    return np.random.default_rng(seed)


# ===================================================
# Sky geometry: projection, fit extent, footprint
# ===================================================


class PlateCarree:
    """Minimal projection centred on the patch: x = lon - LON0 (wrapped), y = lat - LAT0."""

    def ang2xy(self, lon, lat, lonlat=True):
        x = (np.asarray(lon, dtype=float) - LON0 + 180.0) % 360.0 - 180.0
        y = np.asarray(lat, dtype=float) - LAT0
        return x, y

    def get_extent(self):
        return (-HALF_WIDTH[0], HALF_WIDTH[0], -HALF_WIDTH[1], HALF_WIDTH[1])


PROJ = PlateCarree()


def _pixel_lonlat():
    return hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)


def _in_extent(lon, lat):
    x, y = PROJ.ang2xy(lon, lat)
    return (np.abs(x) <= HALF_WIDTH[0]) & (np.abs(y) <= HALF_WIDTH[1])


@pytest.fixture(scope="session")
def nside():
    return NSIDE


@pytest.fixture(scope="session")
def proj():
    """Projection passed to the fitter (`proj` argument)."""
    return PROJ


@pytest.fixture(scope="session")
def in_extent():
    """HEALPix mask of the pixels inside the projection extent, where the background is defined."""
    return _in_extent(*_pixel_lonlat())


@pytest.fixture(scope="session")
def footprint():
    """HEALPix mask of the survey footprint: a disc inside the extent with a hole (e.g. a star mask).

    Unobserved pixels thus fall inside the fit box, around the disc and in the hole.
    """
    disc = hp.query_disc(NSIDE, hp.ang2vec(LON0, LAT0, lonlat=True), np.radians(12.0))
    hole = hp.query_disc(
        NSIDE, hp.ang2vec(LON0 + 5.0, LAT0 + 3.0, lonlat=True), np.radians(3.0)
    )
    footprint = np.zeros(hp.nside2npix(NSIDE), dtype=bool)
    footprint[disc] = True
    footprint[hole] = False
    return footprint


# ===================================================
# Background: polynomial density over the extent
# ===================================================


def _background_density(lon, lat, coeffs, log_space):
    """P(u, v), or exp(P) if log_space, inside the extent; 0 elsewhere.

    u, v are the projected coordinates rescaled to [-1, 1] over the extent.
    """
    x, y = PROJ.ang2xy(lon, lat)
    inside = _in_extent(lon, lat)
    p = polynomial.polyval2d(
        x[inside] / HALF_WIDTH[0], y[inside] / HALF_WIDTH[1], coeffs
    )
    density = np.zeros(x.shape)
    density[inside] = np.exp(p) if log_space else p
    return density


@pytest.fixture(scope="session")
def background_coeffs(seed):
    """Factory: coefficients of P(u, v) for a given degree.

    They are damped with the order so that P stays within (2, 8) for |u|, |v| <= 1.
    """

    def make(degree):
        rng = np.random.default_rng(seed)
        order = np.add.outer(np.arange(degree + 1), np.arange(degree + 1))
        coeffs = rng.uniform(-1.0, 1.0, order.shape) * 0.5**order
        coeffs[0, 0] = 5.0
        return coeffs

    return make


@pytest.fixture(scope="session")
def background_map(background_coeffs):
    """Factory: exact background density at the pixel centres, for a polynomial of the given degree."""

    def make(degree, log_space):
        return _background_density(
            *_pixel_lonlat(), background_coeffs(degree), log_space
        )

    return make


# ===================================================
# Position sampling: background, stream, and both
# ===================================================


@pytest.fixture(scope="session")
def gc_frame():
    """Great-circle frame along the stream, with phi1 = 0 halfway between its ends."""
    (ra1, dec1), (ra2, dec2) = STREAM_ENDS
    end1 = coord.SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    end2 = coord.SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
    return gc.GreatCircleICRSFrame.from_endpoints(end1, end2)


def _draw_in_footprint(footprint, n_stars, propose):
    """Collect n_stars draws from `propose(n) -> (columns, accepted)`, keeping those inside the footprint."""
    batches, n_kept = [], 0
    while n_kept < n_stars:
        columns, accepted = propose(n_stars)
        keep = (
            accepted
            & footprint[hp.ang2pix(NSIDE, columns["ra"], columns["dec"], lonlat=True)]
        )
        batches.append(pd.DataFrame(columns)[keep])
        n_kept += int(keep.sum())
    return pd.concat(batches, ignore_index=True).iloc[:n_stars]


@pytest.fixture
def sample_background_positions(rng, footprint, background_coeffs):
    """N_BACKGROUND stars following the (linear) polynomial background density, inside the footprint."""
    coeffs = background_coeffs(BACKGROUND_DEGREE)
    density_max = np.abs(coeffs).sum()  # bound on |P(u, v)| for |u|, |v| <= 1
    sin_dec = np.sin(np.radians([LAT0 - HALF_WIDTH[1], LAT0 + HALF_WIDTH[1]]))

    def propose(n):
        # Uniform on the sphere over the extent, then rejection sampling of the density.
        ra = rng.uniform(LON0 - HALF_WIDTH[0], LON0 + HALF_WIDTH[0], n)
        dec = np.degrees(np.arcsin(rng.uniform(*sin_dec, n)))
        accepted = rng.uniform(0.0, density_max, n) < _background_density(
            ra, dec, coeffs, log_space=False
        )
        return {"ra": ra, "dec": dec}, accepted

    return _draw_in_footprint(footprint, N_BACKGROUND, propose)


@pytest.fixture
def sample_stream_positions(rng, footprint, gc_frame):
    """N_STREAM stars uniform along the great circle between STREAM_ENDS, Gaussian across it, inside the footprint."""
    (ra1, dec1), (ra2, dec2) = STREAM_ENDS
    ends = coord.SkyCoord(ra=[ra1, ra2] * u.deg, dec=[dec1, dec2] * u.deg).transform_to(
        gc_frame
    )
    phi1_range = np.sort(ends.phi1.wrap_at(180 * u.deg).deg)

    def propose(n):
        phi1 = rng.uniform(*phi1_range, n)
        phi2 = rng.normal(0.0, STREAM_WIDTH, n)
        stars = coord.SkyCoord(
            phi1=phi1 * u.deg, phi2=phi2 * u.deg, frame=gc_frame
        ).transform_to(coord.ICRS())
        return {
            "ra": stars.ra.deg,
            "dec": stars.dec.deg,
            "phi1": phi1,
            "phi2": phi2,
        }, np.ones(n, dtype=bool)

    return _draw_in_footprint(footprint, N_STREAM, propose)


@pytest.fixture
def sample_sky_positions(sample_background_positions, sample_stream_positions):
    """Observed catalog: background and stream stars together, flagged by `is_stream`."""
    return pd.concat(
        [
            sample_background_positions.assign(is_stream=False),
            sample_stream_positions.assign(is_stream=True),
        ],
        ignore_index=True,
    )


# ===================================================
# Magnitude sampling, and the full mock catalog
# ===================================================


@pytest.fixture
def sample_uniform_magnitudes(rng):
    """Factory: (mag_g, mag_r) of n_stars, uniform in g magnitude and in g - r colour."""

    def sample(n_stars):
        mag_g = rng.uniform(*MAG_G_RANGE, n_stars)
        color = rng.uniform(*COLOR_RANGE, n_stars)
        return pd.DataFrame({"mag_g": mag_g, "mag_r": mag_g - color})

    return sample


@pytest.fixture
def sample_sky_catalog(sample_sky_positions, sample_uniform_magnitudes):
    """Mock survey catalog: background and stream positions, with magnitudes."""
    return sample_sky_positions.join(
        sample_uniform_magnitudes(len(sample_sky_positions))
    )

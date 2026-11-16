"""Mock sky shared by the tests: a projection, a survey footprint, a polynomial
background density and a stream, plus fixtures sampling star positions from them.

Every test runs with an empty mock LVDB and Siena Galaxy Atlas, and no download
(``empty_lvdb``): the searches mask their objects by default; the tests of the masks give
their own objects."""

import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc
import healpy as hp
import numpy as np
import pandas as pd
import pytest
from numpy.polynomial import polynomial

from streameucliddes.catalogs import overlapped_objects

LVDB_COLUMNS = ["key", "name", "ra", "dec", "rhalf", "ellipticity", "position_angle", "distance", "confirmed_real"]


@pytest.fixture(scope="session", autouse=True)
def empty_lvdb(tmp_path_factory):
    """No LVDB object, no SGA galaxy, and no download, unless a test sets its own.

    Session-wide, so that the fixtures of a module or of the session see it too.
    """
    lvdb_dir = tmp_path_factory.mktemp("lvdb")
    (lvdb_dir / overlapped_objects.LVDB_VERSION).mkdir()
    for table in (overlapped_objects.DWARF_TABLE, *overlapped_objects.GLOBULAR_CLUSTER_TABLES):
        pd.DataFrame(columns=LVDB_COLUMNS).to_csv(lvdb_dir / overlapped_objects.LVDB_VERSION / f"{table}.csv",
                                                  index=False)
    sga_dir = tmp_path_factory.mktemp("sga")
    pd.DataFrame(columns=overlapped_objects.SGA_COLUMNS).to_csv(sga_dir / overlapped_objects.SGA_BRIGHT_FILE,
                                                                index=False)

    def no_download(url, *args, **kwargs):
        raise AssertionError(f"download in a test: {url}")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(overlapped_objects, "LVDB_DIR", lvdb_dir)
        patch.setattr(overlapped_objects, "SGA_DIR", sga_dir)
        patch.setattr(overlapped_objects.urllib.request, "urlopen", no_download)
        yield

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


# ===================================================
# A stream candidate (analysis.candidate_investigation)
# ===================================================

CANDIDATE_NSIDE = 64  # coarse maps: fast tests
NO_OBJECT_MASKS = {"mask_lmc": False, "mask_smc": False, "mask_dwarfs": False, "mask_gc": False, "mask_galaxies": False}


@pytest.fixture(scope="session")
def candidate_endpoints():
    """(ra, dec) of the endpoints of the Euclid DR1 candidate of the example notebook (deg)."""
    return (77.0, -40.0), (70.0, -43.0)


@pytest.fixture(scope="session")
def no_object_masks():
    """prepare_search / prepare_generic_search arguments masking no object."""
    return dict(NO_OBJECT_MASKS)


@pytest.fixture(scope="session")
def candidate_catalog():
    """Uniform stars in an (ra, dec) box around the candidate, with the columns of the default match filter."""
    rng = np.random.default_rng(3)
    n = 50_000
    vis = rng.uniform(18.0, 26.0, n)
    return pd.DataFrame({
        "ra": rng.uniform(58.0, 90.0, n),
        "dec": np.degrees(np.arcsin(rng.uniform(*np.sin(np.radians([-52.0, -30.0])), n))),
        "euclid_q1_VIS_obs": vis, "euclid_q1_H_obs": vis - rng.uniform(-1.0, 1.5, n),
        "euclid_q1_VIS_err": np.full(n, 0.02), "euclid_q1_H_err": np.full(n, 0.02),
    })


@pytest.fixture
def candidate_with_maps(candidate_endpoints, candidate_catalog):
    """A CandidateInvestigation with its sky map, prepared stars and zoom map, at dm 17 (no object masked)."""
    from streameucliddes.analysis.candidate_investigation import CandidateInvestigation

    cand = CandidateInvestigation("test", *candidate_endpoints, prepare_kwargs={"bin_size_deg": 0.2, **NO_OBJECT_MASKS})
    cand.compute_sky_map(candidate_catalog, 17.0, nside=CANDIDATE_NSIDE, **NO_OBJECT_MASKS)
    return cand.prepare().compute_zoom_map()


CANDIDATE_STREAM = {"distance_modulus": 17.0, "phi2_centre": 0.3, "sigma": 0.25, "n_stars": 1500}


@pytest.fixture(scope="session")
def candidate_stream():
    """Parameters of the stream of candidate_stream_catalog."""
    return dict(CANDIDATE_STREAM)


@pytest.fixture(scope="session")
def candidate_stream_catalog(candidate_endpoints, candidate_catalog):
    """candidate_catalog plus a stream (CANDIDATE_STREAM): phi1 uniform between the endpoints, phi2 Gaussian,
    magnitudes inside the default match filter of the default isochrone at its distance modulus."""
    from streamobs.match_filter import is_in_match_filter

    from streameucliddes.analysis.cmd_match_filter import match_filter_polygons
    from streameucliddes.stream_frames import phi12_to_radec, radec_to_phi12, stream_frame_from_endpoints

    rng = np.random.default_rng(5)
    n = CANDIDATE_STREAM["n_stars"]
    frame = stream_frame_from_endpoints(*candidate_endpoints)
    (ra_1, dec_1), (ra_2, dec_2) = candidate_endpoints
    phi1_ends, _ = radec_to_phi12([ra_1, ra_2], [dec_1, dec_2], frame)
    ra, dec = phi12_to_radec(rng.uniform(*phi1_ends, n),
                             rng.normal(CANDIDATE_STREAM["phi2_centre"], CANDIDATE_STREAM["sigma"], n), frame)
    (polygon,) = match_filter_polygons(CANDIDATE_STREAM["distance_modulus"])
    vis, h = np.empty(0), np.empty(0)
    while vis.size < n:
        vis_try = rng.uniform(18.0, 26.0, 20 * n)
        h_try = vis_try - rng.uniform(-1.0, 1.5, 20 * n)
        inside = is_in_match_filter(vis_try, h_try, polygon_vertices=polygon)
        vis, h = np.append(vis, vis_try[inside]), np.append(h, h_try[inside])
    stream = pd.DataFrame({
        "ra": ra, "dec": dec, "euclid_q1_VIS_obs": vis[:n], "euclid_q1_H_obs": h[:n],
        "euclid_q1_VIS_err": np.full(n, 0.02), "euclid_q1_H_err": np.full(n, 0.02),
    })
    return pd.concat([candidate_catalog, stream], ignore_index=True)

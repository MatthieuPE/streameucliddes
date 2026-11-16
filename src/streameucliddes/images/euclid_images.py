"""Euclid DR1 MER images at a sky position: cutouts, eummy colour images, figures.

The MER mosaics are 0.1 arcsec per pixel: a stream 5 deg long would be
~180,000 pixels at full resolution, in tiles of ~1.5 GB per band. So the
images are small cutouts (a few arcmin) at chosen positions, e.g. along the
track of a stream candidate and in its off region at the same phi1
(``candidate_image_positions``), to check what the stars counted in the match
filter look like.

Steps, for one position:

1. ``mer_cutouts_for_eummy``: VIS, Y, J and H cutouts of one MER tile (IRSA
   SIA, ``MER_COLLECTION``), read with fsspec (only the pixels of the
   cutout), saved under their original MER file names;
2. ``make_color_image``: the ``eummy`` command on them (``pipx install
   eummy``, or a venv with ``~/.local/bin/eummy``), a colour TIFF;
3. ``read_color_image`` and ``cutout_wcs``, then ``plot_color_image``: the
   image with its sky coordinates and stars of a catalog marked.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from streameucliddes.analysis.candidate_investigation import region_windows
from streameucliddes.plotting import plot_style  # noqa: F401  (sets the figure style)
from streameucliddes.plotting.plot_style import CATEGORICAL_COLORS
from streameucliddes.stream_frames import phi12_to_radec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray

    from streameucliddes.analysis.candidate_investigation import CandidateInvestigation

MER_COLLECTION = "euclid_dr1_wide_DpdMerBksMosaic"  # IRSA SIA collection of the DR1 wide MER mosaics
EUMMY_BANDS = ("VIS", "Y", "J", "H")  # energy_bandpassname of the four MER mosaics eummy combines
# File names of the MER mosaics, as eummy finds them (one per band, in the order of EUMMY_BANDS)
MER_FILE_PATTERNS = ("EUC_MER_BGSUB-MOSAIC-VIS*.fits", "EUC_MER_BGSUB-MOSAIC-NIR-Y*.fits",
                     "EUC_MER_BGSUB-MOSAIC-NIR-J*.fits", "EUC_MER_BGSUB-MOSAIC-NIR-H*.fits")
DEFAULT_CUTOUT_SIZE_ARCMIN = 3.0
DEFAULT_IMAGE_STEP_DEG = 1.0  # spacing of the image positions along the track
COLOR_IMAGE_NAME = "color.tif"  # TIFF written by make_color_image, in the folder of the cutouts
EUMMY_FALLBACK_PATH = Path("~/.local/bin/eummy").expanduser()  # where pipx installs it
IMAGE_FIGSIZE = (8.0, 8.0)
STAR_MARKER_SIZE = 80


# ===================================================
# Cutouts and colour image
# ===================================================


def mer_cutouts_for_eummy(
    ra: float, dec: float, size_arcmin: float, outdir: str | os.PathLike, *, overwrite: bool = False
) -> list[Path]:
    """Save VIS, Y, J and H cutouts of one MER tile under their original MER file names.

    The tile is the VIS mosaic whose centre is closest to (ra, dec); the
    NISP mosaics are those of the same tile. Each cutout is read with fsspec
    (only its pixels) and saved with the header of its mosaic, its WCS
    updated. Positions near a tile edge give partial cutouts (NaN outside).

    Parameters
    ----------
    ra, dec : float
        Centre of the cutouts (deg).
    size_arcmin : float
        Side of the cutouts (arcmin).
    outdir : str or path
        Folder of the cutouts, created if needed.
    overwrite : bool, optional
        Download again when ``outdir`` already holds one cutout per band.
        Default: keep them, no query.

    Returns
    -------
    paths : list of pathlib.Path
        The four cutouts, in the order of ``EUMMY_BANDS``.

    Raises
    ------
    ValueError
        No VIS mosaic at this position, or a band missing for its tile.
    """
    outdir = Path(outdir)
    existing = [sorted(outdir.glob(pattern)) for pattern in MER_FILE_PATTERNS]
    if not overwrite and all(len(paths) == 1 for paths in existing):
        return [paths[0] for paths in existing]
    from astroquery.ipac.irsa import Irsa  # only for a download

    coord = SkyCoord(ra * u.deg, dec * u.deg)
    tab = Irsa.query_sia(pos=(coord, 1 * u.arcsec), collection=MER_COLLECTION)
    names = np.char.upper(np.asarray(tab["energy_bandpassname"]).astype(str))
    science = np.asarray(tab["dataproduct_subtype"]).astype(str) == "science"
    tile_ra_all, tile_dec_all = np.asarray(tab["s_ra"], float), np.asarray(tab["s_dec"], float)

    # One tile for all bands: the VIS tile whose centre is closest
    vis = np.where(science & (names == "VIS"))[0]
    if len(vis) == 0:
        raise ValueError(f"No VIS mosaic of {MER_COLLECTION} at (ra, dec) = ({ra}, {dec}).")
    distances = SkyCoord(tile_ra_all[vis] * u.deg, tile_dec_all[vis] * u.deg).separation(coord)
    tile_ra, tile_dec = tile_ra_all[vis[np.argmin(distances)]], tile_dec_all[vis[np.argmin(distances)]]

    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for band in EUMMY_BANDS:
        index = np.where(science & (names == band))[0]
        if len(index) == 0:
            raise ValueError(f"No {band} mosaic of {MER_COLLECTION} at (ra, dec) = ({ra}, {dec}).")
        offsets = np.hypot((tile_ra_all[index] - tile_ra) * np.cos(np.radians(tile_dec)), tile_dec_all[index] - tile_dec)
        url = str(tab["access_url"][index[np.argmin(offsets)]])  # same tile as VIS
        with fits.open(url, use_fsspec=True) as hdul:
            hdu = hdul[0]
            cut = Cutout2D(hdu.section, position=coord, size=size_arcmin * u.arcmin, wcs=WCS(hdu.header),
                           mode="partial", fill_value=np.nan)
            header = hdu.header.copy()
        header.update(cut.wcs.to_header())
        path = outdir / os.path.basename(url.split("?")[0])
        fits.writeto(path, np.asarray(cut.data, np.float32), header, overwrite=True)
        written.append(path)
    return written


def make_color_image(
    directory: str | os.PathLike, *, output: str = COLOR_IMAGE_NAME, eummy_args: Sequence[str] = ()
) -> Path:
    """Colour image of the four MER cutouts of a folder, with the ``eummy`` command.

    ``eummy --path <directory> --output <output>``; the folder is given as
    an absolute path (eummy joins a relative one twice).

    Parameters
    ----------
    directory : str or path
        Folder with one cutout per band (``mer_cutouts_for_eummy``).
    output : str, optional
        Name of the TIFF, in ``directory``.
    eummy_args : sequence of str, optional
        More options of eummy (``eummy --help``), e.g. ``("--contrast", "1.2")``.

    Returns
    -------
    path : pathlib.Path
        The TIFF: uint16 RGB, north up (the FITS rows flipped).

    Raises
    ------
    FileNotFoundError
        eummy not installed.
    RuntimeError
        eummy failed or wrote no image (its output in the message).
    """
    eummy = shutil.which("eummy") or (str(EUMMY_FALLBACK_PATH) if EUMMY_FALLBACK_PATH.exists() else None)
    if eummy is None:
        raise FileNotFoundError("eummy not found: pipx install eummy (or pip install it in its own venv and link "
                                f"{EUMMY_FALLBACK_PATH}).")
    directory = Path(directory).resolve()
    env = dict(os.environ, NUMEXPR_MAX_THREADS=str(os.cpu_count()), NUMEXPR_NUM_THREADS=str(os.cpu_count()))
    result = subprocess.run([eummy, "--path", str(directory), "--output", output, *eummy_args],
                            capture_output=True, text=True, env=env)
    path = directory / output
    if result.returncode != 0 or not path.exists():
        raise RuntimeError(f"eummy failed on {directory} (exit {result.returncode}):\n"
                           f"{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
    return path


def read_color_image(path: str | os.PathLike) -> NDArray[np.float64]:
    """Colour TIFF of ``make_color_image`` as floats in [0, 1], in the row order of the FITS cutouts.

    eummy writes north up (row 0 at the top); the rows are flipped back, so
    that the image draws with ``origin="lower"`` and the WCS of the cutouts
    (``cutout_wcs``).

    Parameters
    ----------
    path : str or path
        The TIFF.

    Returns
    -------
    image : numpy.ndarray
        Shape (ny, nx, 3), RGB.
    """
    image = tifffile.imread(path)
    if np.issubdtype(image.dtype, np.integer):
        image = image / np.iinfo(image.dtype).max  # 8- or 16-bit -> 0..1
    return np.asarray(image[::-1], dtype=float)


def cutout_wcs(directory: str | os.PathLike) -> WCS:
    """Celestial WCS of the VIS cutout of a folder (``mer_cutouts_for_eummy``): that of the colour image.

    Raises
    ------
    FileNotFoundError
        No single VIS cutout in the folder.
    """
    paths = sorted(Path(directory).glob(MER_FILE_PATTERNS[0]))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one VIS cutout ({MER_FILE_PATTERNS[0]}) in {directory}, found {len(paths)}.")
    return WCS(fits.getheader(paths[0])).celestial


# ===================================================
# Positions of a stream candidate
# ===================================================


def candidate_image_positions(
    cand: CandidateInvestigation, step_deg: float = DEFAULT_IMAGE_STEP_DEG, *, off_region: str = "above"
) -> pd.DataFrame:
    """Image positions along the track of a candidate, and in an off region at the same phi1.

    Over ``cand.regions["phi1_range"]``, at the centres of steps of about
    ``step_deg``; on: phi2 of the centre of the on region; off: phi2 of the
    centre of one off region (``region_windows``, offsets included).

    Parameters
    ----------
    cand : CandidateInvestigation
        With its regions (``define_regions``).
    step_deg : float, optional
        Spacing along phi1 (deg).
    off_region : {"above", "below"}, optional
        Off region of the comparison images.

    Returns
    -------
    positions : pandas.DataFrame
        Columns ``region`` ("on" or "off"), ``phi1``, ``phi2``, ``ra``,
        ``dec`` (deg); the on positions first.

    Raises
    ------
    ValueError
        No regions, ``step_deg <= 0``, or an unknown ``off_region``.
    """
    if cand.regions is None:
        raise ValueError("No regions: call cand.define_regions() first.")
    if step_deg <= 0:
        raise ValueError(f"step_deg must be > 0, not {step_deg}.")
    sides = {"below": 0, "above": 1}  # the off regions below and above the track (region_windows)
    if off_region not in sides:
        raise ValueError(f"off_region must be one of {sorted(sides)}, not {off_region!r}.")
    windows = region_windows(cand.regions)
    phi1_min, phi1_max = cand.regions["phi1_range"]
    n_steps = max(1, round((phi1_max - phi1_min) / step_deg))
    phi1 = phi1_min + (np.arange(n_steps) + 0.5) * (phi1_max - phi1_min) / n_steps
    rows = []
    for region, window in (("on", windows["on"]), ("off", windows["off"][sides[off_region]])):
        phi2 = np.full(n_steps, 0.5 * (window[2] + window[3]))
        ra, dec = phi12_to_radec(phi1, phi2, cand.stream_frame)
        rows.append(pd.DataFrame({"region": region, "phi1": phi1, "phi2": phi2, "ra": ra, "dec": dec}))
    return pd.concat(rows, ignore_index=True)


# ===================================================
# Figure
# ===================================================


def plot_color_image(
    image: NDArray[np.float64],
    wcs: WCS,
    *,
    marked_stars: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Colour image with its sky coordinates, and stars of a catalog circled.

    Parameters
    ----------
    image : numpy.ndarray
        From ``read_color_image`` (FITS row order).
    wcs : astropy.wcs.WCS
        Its WCS (``cutout_wcs``).
    marked_stars : mapping, optional
        ``{label: (ra, dec)}`` (deg); only the stars inside the image are
        drawn, one colour per label, their number in the legend.
    ax : matplotlib.axes.Axes, optional
        Axes with ``projection=wcs``. Default: a new figure.
    title : str, optional
        Title of the axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig = plt.figure(figsize=IMAGE_FIGSIZE, layout="constrained")
        ax = fig.add_subplot(projection=wcs)
    else:
        fig = ax.figure
    ax.imshow(image, origin="lower", interpolation="nearest")
    ny, nx = image.shape[:2]
    for (label, (ra, dec)), color in zip((marked_stars or {}).items(), CATEGORICAL_COLORS):
        ra, dec = np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
        x, y = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        inside = (x > -0.5) & (x < nx - 0.5) & (y > -0.5) & (y < ny - 0.5)
        ax.scatter(x[inside], y[inside], s=STAR_MARKER_SIZE, facecolors="none", edgecolors=color, linewidths=1.0,
                   label=f"{label} ({inside.sum()})")
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    if hasattr(ax, "coords"):  # WCS axes: decimal degrees, without the degree sign (not in the font of plot_style)
        for coord in ax.coords:
            coord.set_format_unit(u.deg, decimal=True, show_decimal_unit=False)
    ax.set_xlabel("ra [deg]")
    ax.set_ylabel("dec [deg]")
    if marked_stars:
        ax.legend(loc="upper right", fontsize="small", framealpha=0.8)
    if title is not None:
        ax.set_title(title)
    return fig, ax


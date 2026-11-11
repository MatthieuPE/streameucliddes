"""Videos made from saved figures."""

from __future__ import annotations

import glob
import os
import re
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


def make_gif(
    figures_folder="../../figures",
    pattern="streamfind_dm*.png",
    gif_name="streamfind.gif",
    fps=2,
    sort_numeric=True,
    dither=True,
):
    """
    Make a gif from several figures saved in `figures_folder`.

    Each frame is quantized to a 256-color palette with Pillow's octree
    quantizer (`Image.Quantize.FASTOCTREE`) plus Floyd-Steinberg dithering by
    default. GIF only supports 256 colors per frame; imageio's default writer
    quantizes without dithering, which visibly flattens/banding continuous
    colormaps (e.g. the density maps here). Dithering trades that banding for
    a bit of grain, which reads as noticeably more detail preserved.

    Parameters
    ----------
    pattern : str
        Glob pattern (relative to figures_folder) matching frame filenames.
    sort_numeric : bool
        If True, order frames by the first number found in each filename
        (e.g. the distance modulus) instead of lexicographic order.
    dither : bool
        If True (default), apply Floyd-Steinberg dithering during palette
        quantization. Set False for flatter, more compressible frames at the
        cost of visible color banding.

    Returns
    -------
    gif_path : str
    """
    files = glob.glob(os.path.join(figures_folder, pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in '{figures_folder}'."
        )

    if sort_numeric:

        def _sort_key(path):
            match = re.search(r"[-+]?\d*\.?\d+", os.path.basename(path))
            return float(match.group()) if match else path

        files = sorted(files, key=_sort_key)
    else:
        files = sorted(files)

    dither_mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    frames = [
        Image.open(f)
        .convert("RGB")
        .quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=dither_mode)
        for f in files
    ]

    gif_path = os.path.join(figures_folder, gif_name)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        disposal=2,
    )
    print(f"{gif_name} created ({len(files)} frames) at {gif_path}")
    return gif_path


def make_gif_from_plots(
    output_path: str | Path, pattern: str = "./plots/*.png", duration: int = 500
) -> None:
    """Animated GIF from the PNG images matching a glob pattern.

    Frames in the sorted order of their file names, each shown ``duration``
    ms; the GIF loops. All new gifs use this function (user choice), not
    ``make_gif``.

    Parameters
    ----------
    output_path : str or pathlib.Path
        GIF file to write.
    pattern : str, optional
        Glob pattern of the frames.
    duration : int, optional
        Time per frame (ms).

    Raises
    ------
    FileNotFoundError
        No image matches ``pattern``.

    Notes
    -----
    Copied from celestial_currents ``visualization.py::make_gif_from_plots``;
    the paths may also be ``pathlib.Path``.
    """
    image_files = sorted(glob.glob(str(pattern)))
    if not image_files:
        raise FileNotFoundError(
            f"No images found with pattern: {pattern}"
        )
    images = [Image.open(f).convert("RGB") for f in image_files]
    images[0].save(
        str(output_path),
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )
    print(f"GIF saved to {output_path}")


def make_mp4(
    output_path: str | Path, pattern: str = "./plots/*.png", duration: int = 500
) -> None:
    """MP4 video from the PNG images matching a glob pattern.

    Same interface as ``make_gif_from_plots``: frames in the sorted order of
    their file names, each shown ``duration`` ms.

    Parameters
    ----------
    output_path : str or pathlib.Path
        MP4 file to write.
    pattern : str, optional
        Glob pattern of the frames.
    duration : int, optional
        Time per frame (ms): 1000 / ``duration`` frames per second.

    Raises
    ------
    FileNotFoundError
        No image matches ``pattern``.

    Notes
    -----
    Code to reuse: none. ``imageio`` with the ``imageio-ffmpeg`` plugin (new
    dependencies of phase 3).
    """
    raise NotImplementedError

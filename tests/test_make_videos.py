"""Tests of plotting/make_videos.py, with small synthetic frames."""

import pytest
from PIL import Image

from streameucliddes.plotting.make_videos import make_gif_from_plots

pytestmark = pytest.mark.make_videos

# One plain colour per frame, in the order of the file names
COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


@pytest.fixture
def frames(tmp_path):
    # Written in reverse order: the gif follows the sorted file names, not the writing order
    for index, colour in reversed(list(enumerate(COLOURS))):
        Image.new("RGB", (20, 10), colour).save(tmp_path / f"frame_dm{15 + index:.2f}.png")
    return tmp_path


def _gif_frames(path):
    """Colours (of the central pixel) and durations of the frames of a gif."""
    colours, durations = [], []
    with Image.open(path) as gif:
        for index in range(gif.n_frames):
            gif.seek(index)
            durations.append(gif.info["duration"])
            colours.append(gif.convert("RGB").getpixel((10, 5)))
        loop = gif.info.get("loop")
    return colours, durations, loop


# ===================================================
# Test make_gif_from_plots
# ===================================================


# Every frame matching the pattern, in the sorted order of the names, each shown `duration`
# ms; the gif loops. Paths may be pathlib.Path
def test_make_gif_from_plots(frames):
    output = frames / "scan.gif"
    make_gif_from_plots(output, pattern=frames / "frame_dm*.png", duration=300)
    colours, durations, loop = _gif_frames(output)
    assert colours == COLOURS
    assert durations == [300] * len(COLOURS)
    assert loop == 0  # loops forever


# Only the files matching the pattern
def test_make_gif_from_plots_pattern(frames):
    Image.new("RGB", (20, 10), (255, 255, 255)).save(frames / "other.png")
    output = frames / "scan.gif"
    make_gif_from_plots(str(output), pattern=str(frames / "frame_*.png"))
    colours, _, _ = _gif_frames(output)
    assert colours == COLOURS


# No frame: FileNotFoundError, no gif written
def test_make_gif_from_plots_no_frame(tmp_path):
    output = tmp_path / "scan.gif"
    with pytest.raises(FileNotFoundError, match="No images found"):
        make_gif_from_plots(output, pattern=tmp_path / "*.png")
    assert not output.exists()

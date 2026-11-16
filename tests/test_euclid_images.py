"""Tests of images/euclid_images.py, offline: synthetic MER mosaics on disk instead of IRSA."""

import io
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from streameucliddes.analysis.candidate_investigation import CandidateInvestigation, region_windows, stream_regions
from streameucliddes.images import euclid_images
from streameucliddes.images.euclid_images import (
    EUMMY_BANDS,
    candidate_image_positions,
    cutout_wcs,
    make_color_image,
    match_filter_star_positions,
    mer_cutouts_for_eummy,
    plot_color_image,
    read_color_image,
)
from streameucliddes.stream_frames import in_phi12_window, radec_to_phi12

pytestmark = pytest.mark.euclid_images

PIXEL_DEG = 0.1 / 3600  # MER pixel
FILE_BANDS = {"VIS": "VIS", "Y": "NIR-Y", "J": "NIR-J", "H": "NIR-H"}  # band -> its name in the MER file names
SOURCE_PIXEL = (50, 250)  # (x, y), 0-based, of the bright source of the synthetic images


def _wcs(ra, dec, n):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    wcs.wcs.cdelt = [-PIXEL_DEG, PIXEL_DEG]
    return wcs


def _write_mer_images(directory, ra, dec, n, tile="TILE102000001", source=False):
    """One MER-named FITS image per band, n x n pixels centred on (ra, dec), noise (and one bright source)."""
    rng = np.random.default_rng(1)
    y, x = np.mgrid[:n, :n]
    paths = {}
    for band, name in FILE_BANDS.items():
        data = rng.normal(0.0, 1.0, (n, n))
        if source:
            data += 500.0 * np.exp(-((x - SOURCE_PIXEL[0]) ** 2 + (y - SOURCE_PIXEL[1]) ** 2) / 8.0)
        header = _wcs(ra, dec, n).to_header()
        header["MAGZERO"] = 30.0
        path = Path(directory) / f"EUC_MER_BGSUB-MOSAIC-{name}_{tile}-ABC_20260101T000000.000000Z_00.00.fits"
        fits.writeto(path, data.astype(np.float32), header)
        paths[band] = path
    return paths


# IRSA: two tiles, the cutout of the closer one in each band, under its MER file name, the WCS centred on the position,
# the header kept; a second call reuses the files without a query
def test_mer_cutouts_for_eummy(tmp_path, monkeypatch):
    from astroquery.ipac.irsa import Irsa

    (tmp_path / "mosaics").mkdir()
    near = _write_mer_images(tmp_path / "mosaics", 45.5, -45.0, 600, tile="TILE102000001")
    far = _write_mer_images(tmp_path / "mosaics", 45.52, -45.0, 600, tile="TILE102000002")
    rows = [(band, "science", 45.5, -45.0, str(near[band])) for band in EUMMY_BANDS]
    rows += [(band, "science", 45.52, -45.0, str(far[band])) for band in EUMMY_BANDS]
    rows += [("VIS", "weight", 45.5, -45.0, "unused.fits")]
    table = Table(rows=rows, names=("energy_bandpassname", "dataproduct_subtype", "s_ra", "s_dec", "access_url"))
    monkeypatch.setattr(Irsa, "query_sia", lambda **kwargs: table)
    ra, dec, size_arcmin = 45.501, -45.001, 0.5
    paths = mer_cutouts_for_eummy(ra, dec, size_arcmin, tmp_path / "cutouts", source="irsa")
    assert [path.name for path in paths] == [near[band].name for band in EUMMY_BANDS]
    n = round(size_arcmin / 60 / PIXEL_DEG)
    for path in paths:
        data, header = fits.getdata(path, header=True)
        assert data.shape == (n, n) and header["MAGZERO"] == 30.0
        centre = WCS(header).pixel_to_world((n - 1) / 2, (n - 1) / 2)
        assert centre.separation(SkyCoord(ra * u.deg, dec * u.deg)) < 2 * PIXEL_DEG * u.deg

    def no_query(**kwargs):
        raise AssertionError("queried again")

    monkeypatch.setattr(Irsa, "query_sia", no_query)
    assert mer_cutouts_for_eummy(ra, dec, size_arcmin, tmp_path / "cutouts", source="irsa") == paths


# No VIS mosaic at the position: ValueError
def test_mer_cutouts_for_eummy_no_vis(tmp_path, monkeypatch):
    from astroquery.ipac.irsa import Irsa

    table = Table(rows=[("H", "science", 45.5, -45.0, "h.fits")],
                  names=("energy_bandpassname", "dataproduct_subtype", "s_ra", "s_dec", "access_url"))
    monkeypatch.setattr(Irsa, "query_sia", lambda **kwargs: table)
    with pytest.raises(ValueError, match="VIS"):
        mer_cutouts_for_eummy(45.5, -45.0, 0.5, tmp_path, source="irsa")


# ESA: the mosaics of the closer tile in each band, cut by the cutout service (a circle of radius half the side),
# saved under their MER file names in the order of the bands; a second call reuses the files without a query
def test_mer_cutouts_for_eummy_esa(tmp_path, monkeypatch):
    rows = []
    for tile, tile_ra in ((102000001, 45.5), (102000002, 45.52)):
        for band, filter_name in zip(EUMMY_BANDS, euclid_images.ESA_FILTERS):
            name = f"EUC_MER_BGSUB-MOSAIC-{FILE_BANDS[band]}_TILE{tile}-ABC_20260101T000000.000000Z_00.00.fits"
            rows.append({"file_name": name, "file_path": f"/data/{tile}", "filter_name": filter_name,
                         "tile_index": tile, "instrument_name": "VIS" if band == "VIS" else "NISP",
                         "ra": tile_ra, "dec": -45.0})
    table = pd.DataFrame(rows)
    monkeypatch.setattr(euclid_images, "_esa_mosaic_table", lambda ra, dec: table)
    fits_bytes = io.BytesIO()
    fits.PrimaryHDU(np.zeros((3, 3), dtype=np.float32), header=_wcs(45.5, -45.0, 3).to_header()).writeto(fits_bytes)
    requests_made = []

    class Response:
        content = fits_bytes.getvalue()

        def raise_for_status(self):
            pass

    def fake_get(url, params, timeout):
        requests_made.append((url, params))
        return Response()

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    paths = mer_cutouts_for_eummy(45.501, -45.001, 3.0, tmp_path / "cutouts")
    assert [path.name for path in paths] == list(table["file_name"][:4])  # tile 102000001, the closer one
    assert all(url == euclid_images.ESA_CUTOUT_URL for url, _ in requests_made)
    first = requests_made[0][1]
    assert first["filepath"] == f"/data/102000001/{table['file_name'][0]}" and first["obsid"] == "102000001"
    assert first["collection"] == "VIS" and first["POS"] == "CIRCLE,45.501,-45.001,0.025000"
    assert all(path.read_bytes() == Response.content for path in paths)
    assert mer_cutouts_for_eummy(45.501, -45.001, 3.0, tmp_path / "cutouts") == paths
    assert len(requests_made) == 4


# ESA: no VIS mosaic, or no FITS returned (an error page): ValueError; an unknown source: ValueError
def test_mer_cutouts_for_eummy_esa_errors(tmp_path, monkeypatch):
    import requests

    with pytest.raises(ValueError, match="source"):
        mer_cutouts_for_eummy(45.5, -45.0, 3.0, tmp_path, source="mast")
    empty = pd.DataFrame(columns=["file_name", "file_path", "filter_name", "tile_index", "instrument_name", "ra", "dec"])
    monkeypatch.setattr(euclid_images, "_esa_mosaic_table", lambda ra, dec: empty)
    with pytest.raises(ValueError, match="No VIS"):
        mer_cutouts_for_eummy(45.5, -45.0, 3.0, tmp_path)
    table = pd.DataFrame([{"file_name": f"{band}.fits", "file_path": "/data", "filter_name": filter_name,
                           "tile_index": 1, "instrument_name": "VIS", "ra": 45.5, "dec": -45.0}
                          for band, filter_name in zip(EUMMY_BANDS, euclid_images.ESA_FILTERS)])
    monkeypatch.setattr(euclid_images, "_esa_mosaic_table", lambda ra, dec: table)

    class ErrorPage:
        content = b"<html>Service unavailable</html>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda url, params, timeout: ErrorPage())
    with pytest.raises(ValueError, match="no FITS"):
        mer_cutouts_for_eummy(45.5, -45.0, 3.0, tmp_path)


# The TIFF of eummy is north up: read_color_image gives it back in the FITS row order, in [0, 1]
def test_read_color_image(tmp_path):
    rgb = np.zeros((4, 3, 3), dtype=np.uint16)
    rgb[0, 1] = 65535  # top row of the TIFF = last FITS row
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "color.tif", rgb)
    image = read_color_image(tmp_path / "color.tif")
    assert image.shape == (4, 3, 3) and image.max() == 1.0
    assert np.unravel_index(image.sum(-1).argmax(), (4, 3)) == (3, 1)


# eummy on synthetic cutouts: a TIFF of their shape, the source at its FITS pixel once read; the WCS is that of VIS
@pytest.mark.skipif(not (euclid_images.shutil.which("eummy") or euclid_images.EUMMY_FALLBACK_PATH.exists()),
                    reason="eummy not installed")
def test_make_color_image(tmp_path):
    paths = _write_mer_images(tmp_path, 45.5, -45.0, 300, source=True)
    path = make_color_image(tmp_path)
    assert path == tmp_path.resolve() / euclid_images.COLOR_IMAGE_NAME
    image = read_color_image(path)
    assert image.shape == (300, 300, 3)
    row, column = np.unravel_index(image.sum(-1).argmax(), image.shape[:2])
    assert (column, row) == SOURCE_PIXEL
    assert cutout_wcs(tmp_path).to_header() == WCS(fits.getheader(paths["VIS"])).celestial.to_header()


# eummy on a folder without cutouts: RuntimeError with its output; no single VIS cutout: FileNotFoundError
@pytest.mark.skipif(not (euclid_images.shutil.which("eummy") or euclid_images.EUMMY_FALLBACK_PATH.exists()),
                    reason="eummy not installed")
def test_make_color_image_errors(tmp_path):
    with pytest.raises(RuntimeError, match="eummy failed"):
        make_color_image(tmp_path)
    with pytest.raises(FileNotFoundError, match="VIS"):
        cutout_wcs(tmp_path)


# Positions at the centres of steps over phi1_range, on the centre of the on region and of the chosen off region
# (offsets included), with their (ra, dec); errors
def test_candidate_image_positions(candidate_endpoints):
    cand = CandidateInvestigation("test", *candidate_endpoints)
    with pytest.raises(ValueError, match="define_regions"):
        candidate_image_positions(cand)
    cand.regions = stream_regions((-2.0, 2.0), 0.1, 0.3, on_n_sigma=1.5, off_n_sigma=(2.5, 4.0),
                                  on_phi2_offset_deg=0.2)
    windows = region_windows(cand.regions)
    for off_region, index in (("above", 1), ("below", 0)):
        positions = candidate_image_positions(cand, 1.0, off_region=off_region)
        on, off = positions[positions["region"] == "on"], positions[positions["region"] == "off"]
        np.testing.assert_allclose(on["phi1"], [-1.5, -0.5, 0.5, 1.5])
        np.testing.assert_allclose(off["phi1"], on["phi1"])
        assert np.allclose(on["phi2"], 0.5 * (windows["on"][2] + windows["on"][3]))
        assert np.allclose(off["phi2"], 0.5 * (windows["off"][index][2] + windows["off"][index][3]))
        phi1, phi2 = radec_to_phi12(positions["ra"], positions["dec"], cand.stream_frame)
        np.testing.assert_allclose(phi1, positions["phi1"], atol=1e-8)
        np.testing.assert_allclose(phi2, positions["phi2"], atol=1e-8)
    assert len(candidate_image_positions(cand, 10.0)) == 2  # one step at least
    with pytest.raises(ValueError, match="off_region"):
        candidate_image_positions(cand, off_region="left")
    with pytest.raises(ValueError, match="step_deg"):
        candidate_image_positions(cand, 0.0)


# The on-region stars in the match filter: n of them, sorted by phi1, spread over the whole list (first and last
# kept), with their magnitude and colour; all of them when fewer; errors
def test_match_filter_star_positions(candidate_endpoints, candidate_stream_catalog, candidate_stream, no_object_masks):
    cand = CandidateInvestigation("test", *candidate_endpoints, distance_modulus=candidate_stream["distance_modulus"],
                                  prepare_kwargs={"bin_size_deg": 0.2, **no_object_masks})
    with pytest.raises(ValueError, match="define_regions"):
        match_filter_star_positions(cand)
    cand.prepare(candidate_stream_catalog).fit_width().define_regions()
    stars = cand.search.stars
    on = cand.stars_in_match_filter() & in_phi12_window(stars["phi1"], stars["phi2"], region_windows(cand.regions)["on"])
    all_on = stars[on].sort_values("phi1")
    picked = match_filter_star_positions(cand, 10)
    assert len(picked) == 10 and picked["phi1"].is_monotonic_increasing
    assert set(picked["index"]) <= set(all_on.index)
    assert picked["index"].iloc[0] == all_on.index[0] and picked["index"].iloc[-1] == all_on.index[-1]
    row = stars.loc[picked["index"].iloc[0]]
    assert picked["ra"].iloc[0] == row["ra"] and picked["mag"].iloc[0] == row["euclid_q1_VIS_obs"]
    assert picked["color"].iloc[0] == pytest.approx(row["euclid_q1_VIS_obs"] - row["euclid_q1_H_obs"])
    assert len(match_filter_star_positions(cand, 10 * on.sum())) == on.sum()
    with pytest.raises(ValueError, match="n_stars"):
        match_filter_star_positions(cand, 0)


# The image on WCS axes; only the marked stars inside it drawn, at their pixel, counted in the legend
def test_plot_color_image():
    n = 100
    wcs = _wcs(45.5, -45.0, n)
    inside = wcs.pixel_to_world(20, 70)
    outside = wcs.pixel_to_world(-50, 50)
    stars = {"match filter": ([inside.ra.deg, outside.ra.deg], [inside.dec.deg, outside.dec.deg])}
    fig, ax = plot_color_image(np.zeros((n, n, 3)), wcs, marked_stars=stars, title="test")
    (points,) = ax.collections
    np.testing.assert_allclose(points.get_offsets(), [[20, 70]], atol=1e-6)
    assert ax.get_legend().get_texts()[0].get_text() == "match filter (1)" and ax.get_title() == "test"
    plt.close(fig)

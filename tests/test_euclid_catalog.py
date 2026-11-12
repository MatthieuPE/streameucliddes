"""Tests of catalogs/euclid_catalog.py and scripts/download_euclid_dr1.py, with mock catalogs."""

import functools
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("streamobs")

from streameucliddes.catalogs import euclid_catalog  # noqa: E402
from streameucliddes.catalogs.catalog_cuts import get_cut_mask  # noqa: E402

pytestmark = pytest.mark.euclid_catalog

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_euclid_dr1.py"


@pytest.fixture(scope="module")
def download():
    """The download script, imported as a module."""
    spec = importlib.util.spec_from_file_location("download_euclid_dr1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CATALOG_NAMES = ["euclid_dr1", "euclid_q1_des"]  # Euclid only, crossmatched with DES


def _mock_catalog(catalog, path, rng, n=2000):
    """Write a one-partition HATS-like catalog with the default columns of ``catalog``."""
    columns = euclid_catalog.catalog_columns(catalog)
    column = functools.partial(euclid_catalog.euclid_column, catalog)
    data = {c: rng.uniform(0.0, 1.0, n) for c in euclid_catalog.default_columns(catalog)}
    data[column("OBJECT_ID")] = np.arange(n)
    data[columns["ra"]] = rng.uniform(0.0, 360.0, n)
    data[columns["dec"]] = rng.uniform(-60.0, -30.0, n)
    data[column("SPURIOUS_FLAG")] = rng.integers(0, 2, n)
    data[column("POINT_LIKE_PROB")][::7] = np.nan  # NIR-only sources
    for flux_col, err_col in columns["euclid_fluxes"].values():
        data[flux_col] = rng.uniform(-1.0, 100.0, n)  # micro-Jy, some non-positive
        data[err_col] = rng.uniform(0.01, 1.0, n)
    data[columns["ebv"]] = np.full(n, 0.1)
    for mag_col, err_col in columns["des_magnitudes"].values():
        data[mag_col] = rng.uniform(15.0, 40.0, n)  # some above the valid range
        data[mag_col][::11] = -99.0  # sentinel
    if columns["des_magnitudes"]:
        for flag in euclid_catalog.DES_QUALITY_CUTS:
            data[flag] = rng.integers(0, 2, n)
        data["EXT_XGB_des"] = rng.integers(-1, 5, n)
    # 64-bit ids with nulls; 2**62 + i is not exactly representable as float64
    data[column("GAIA_ID")] = pd.array([2**62 + i if i % 3 else None for i in range(n)], dtype="Int64")
    table = pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False)
    dataset = path / catalog / "dataset"
    (dataset / "Norder=0" / "Dir=0").mkdir(parents=True)
    pq.write_table(table, dataset / "Norder=0" / "Dir=0" / "Npix=0.parquet")
    pq.write_metadata(table.schema, dataset / "_common_metadata")
    (path / catalog / "hats.properties").write_text(f"obs_collection={catalog}\n")
    return pd.DataFrame(data)


@pytest.fixture(params=CATALOG_NAMES)
def catalog_dir(request, tmp_path, rng):
    """(catalog name, folder, mock data) for each kind of catalog."""
    return request.param, tmp_path, _mock_catalog(request.param, tmp_path, rng)


def test_unknown_catalog():
    with pytest.raises(ValueError, match="euclid_dr1"):
        euclid_catalog.default_columns("euclid_dr2")


def test_column_names():
    dr1 = euclid_catalog.default_columns("euclid_dr1")
    crossmatch = euclid_catalog.default_columns("euclid_q1_des")
    assert not any(c.endswith(("_euclid", "_des")) for c in dr1)
    assert all(c.endswith(("_euclid", "_des")) or c.startswith("_") for c in crossmatch)
    assert euclid_catalog.catalog_columns("euclid_dr1")["des_magnitudes"] == {}


def test_open_with_cuts(catalog_dir):
    catalog, path, data = catalog_dir
    spurious = euclid_catalog.euclid_column(catalog, "SPURIOUS_FLAG")
    read = euclid_catalog.open_euclid_catalog(catalog, data_path=path, cuts={spurious: 0}, verbose=False)
    assert list(read.columns) == euclid_catalog.default_columns(catalog)
    assert len(read) == (data[spurious] == 0).sum()


@pytest.mark.parametrize("ra_range", [(10.0, 50.0), (350.0, 10.0)])
def test_open_sky_box(catalog_dir, ra_range):
    catalog, path, data = catalog_dir
    columns = euclid_catalog.catalog_columns(catalog)
    dec_range = (-50.0, -40.0)
    read = euclid_catalog.open_euclid_catalog(
        catalog, data_path=path, ra_range=ra_range, dec_range=dec_range, verbose=False
    )
    ra, dec = data[columns["ra"]], data[columns["dec"]]
    in_ra = (ra >= ra_range[0]) & (ra <= ra_range[1]) if ra_range[0] <= ra_range[1] else (
        (ra >= ra_range[0]) | (ra <= ra_range[1]))
    expected = in_ra & dec.between(*dec_range)
    assert expected.sum() > 0
    assert sorted(read[euclid_catalog.euclid_column(catalog, "OBJECT_ID")]) == sorted(
        data.loc[expected, euclid_catalog.euclid_column(catalog, "OBJECT_ID")])


def test_open_keeps_64bit_ids_with_nulls(catalog_dir):
    catalog, path, data = catalog_dir
    object_id, gaia_id = (euclid_catalog.euclid_column(catalog, c) for c in ("OBJECT_ID", "GAIA_ID"))
    read = euclid_catalog.open_euclid_catalog(
        catalog, columns=[object_id, gaia_id], data_path=path, verbose=False
    )
    assert read[object_id].dtype == np.int64  # no null: plain int64
    pd.testing.assert_series_equal(read[gaia_id], data[gaia_id], check_index=False)


def test_open_unknown_column(catalog_dir):
    catalog, path, _ = catalog_dir
    with pytest.raises(KeyError):
        euclid_catalog.open_euclid_catalog(catalog, columns=["NOT_A_COLUMN"], data_path=path)


def test_magnitudes(catalog_dir):
    catalog, path, data = catalog_dir
    columns = euclid_catalog.catalog_columns(catalog)
    survey = euclid_catalog.euclid_survey(catalog)
    read = euclid_catalog.add_magnitudes(
        euclid_catalog.open_euclid_catalog(catalog, data_path=path, verbose=False), catalog
    )
    flux = data[columns["euclid_fluxes"]["VIS"][0]].to_numpy()
    expected = -2.5 * np.log10(flux * 1e-6 / 3631.0) - survey.coeff_extinc["VIS"] * 0.1
    vis = read[f"{survey.namespace}_VIS_obs"]
    np.testing.assert_allclose(vis[flux > 0], expected[flux > 0], rtol=1e-6)
    assert vis[flux <= 0].isna().all()
    np.testing.assert_array_equal(read["ra"], data[columns["ra"]])
    for band, (mag_col, _) in columns["des_magnitudes"].items():
        mag = data[mag_col].to_numpy()
        valid = (mag > euclid_catalog.DES_VALID_MAG_RANGE[0]) & (mag < euclid_catalog.DES_VALID_MAG_RANGE[1])
        des = read[f"{euclid_catalog.DES_NAMESPACE}_{band}_obs"].to_numpy()
        np.testing.assert_array_equal(des[valid], mag[valid])
        assert np.isnan(des[~valid]).all()
    if not columns["des_magnitudes"]:
        assert not any(c.startswith(euclid_catalog.DES_NAMESPACE) for c in read.columns)


def test_stars(catalog_dir):
    catalog, path, data = catalog_dir
    cuts = euclid_catalog.default_star_cuts(catalog)
    stars = euclid_catalog.get_euclid_stars(catalog, data_path=path, verbose=False)
    all_rows = euclid_catalog.add_magnitudes(data.copy(), catalog)
    assert 0 < len(stars) == get_cut_mask(all_rows, cuts).sum()
    assert get_cut_mask(stars, cuts).all()


def test_magnitude_columns(catalog_dir):
    catalog, _, data = catalog_dir
    added = set(euclid_catalog.add_magnitudes(data.copy(), catalog).columns) - set(data.columns)
    assert added == set(euclid_catalog.magnitude_columns(catalog))


def test_stars_default_cuts_no_warning(catalog_dir):
    catalog, path, _ = catalog_dir
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        euclid_catalog.get_euclid_stars(catalog, data_path=path, verbose=False)


def test_stars_unknown_cut_column(catalog_dir):
    catalog, path, _ = catalog_dir
    with pytest.raises(KeyError, match="NOT_A_COLUMN"):
        euclid_catalog.get_euclid_stars(catalog, cuts={"NOT_A_COLUMN": 0}, data_path=path)


def test_match_columns(download):
    available = ["mer_object_id", "mer_right_ascension", "mer_flux_g_ext_decam_templfit"]
    mapping = download.match_columns(
        available, ["OBJECT_ID", "RIGHT_ASCENSION"], prefix="mer_",
        optional=["FLUX_G_EXT_DECAM_TEMPLFIT", "FLUX_U_EXT_DECAM_TEMPLFIT"],
    )
    assert mapping == {
        "mer_object_id": "OBJECT_ID",
        "mer_right_ascension": "RIGHT_ASCENSION",
        "mer_flux_g_ext_decam_templfit": "FLUX_G_EXT_DECAM_TEMPLFIT",
    }
    with pytest.raises(ValueError):
        download.match_columns(available, ["GAL_EBV"])


def test_column_dtypes(download):
    mapping = {"gaia_id": "GAIA_ID", "vis_det": "VIS_DET",
               "right_ascension": "RIGHT_ASCENSION", "flux_vis_psf": "FLUX_VIS_PSF"}
    datatypes = {"gaia_id": "long", "vis_det": "short",
                 "right_ascension": "double", "flux_vis_psf": "double"}
    assert download.column_dtypes(datatypes, mapping) == {
        "gaia_id": "Int64", "vis_det": "Int16",
        "right_ascension": "float64", "flux_vis_psf": "float32",
    }
    with pytest.raises(ValueError):
        download.column_dtypes({"gaia_id": "char"}, {"gaia_id": "GAIA_ID"})


def test_tile_query(download):
    mapping = {"object_id": "OBJECT_ID", "right_ascension": "RIGHT_ASCENSION"}
    assert download._tile_query("dr1.t", mapping, 12, None) == (
        "SELECT object_id, right_ascension FROM dr1.t WHERE tile_index = 12"
    )
    assert download._tile_query("dr1.t", mapping, 12, "point_like_prob > 0.5").endswith(
        "WHERE tile_index = 12 AND (point_like_prob > 0.5)"
    )


@pytest.fixture
def tap_csv(download, monkeypatch):
    """A fake sync TAP answer (CSV, a null id) and no SFD map (E(B-V) = 0)."""
    big_id = 2**62 + 1  # not exactly representable as float64
    text = (f"gaia_id,right_ascension,declination,flux_vis_psf\n"
            f"{big_id},10.5,-30.5,1.5\n,11.5,-31.5,\n")

    class Response:
        content = text.encode()

        def raise_for_status(self):
            pass

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(download, "_sfd_query", lambda: (lambda coords: np.zeros(len(coords))))
    mapping = {"gaia_id": "GAIA_ID", "right_ascension": "RIGHT_ASCENSION",
               "declination": "DECLINATION", "flux_vis_psf": "FLUX_VIS_PSF"}
    dtypes = {"gaia_id": "Int64", "right_ascension": "float64", "declination": "float64",
              "flux_vis_psf": "float32"}
    return mapping, dtypes, big_id


def test_download_tile(download, tap_csv, tmp_path):
    mapping, dtypes, big_id = tap_csv
    path = tmp_path / "tile_1.parquet"
    rows = download.download_tile("https://tap", "query", dtypes, mapping, path, expected=2)
    frame = pd.read_parquet(path)
    assert rows == 2
    assert list(frame.columns) == [*mapping.values(), "EBV_SFD98"]
    assert frame["GAIA_ID"].iloc[0] == big_id and pd.isna(frame["GAIA_ID"].iloc[1])
    assert frame["FLUX_VIS_PSF"].dtype == np.float32
    assert frame["RIGHT_ASCENSION"].dtype == np.float64


def test_download_tile_wrong_row_count(download, tap_csv, tmp_path):
    mapping, dtypes, _ = tap_csv
    path = tmp_path / "tile_1.parquet"
    with pytest.raises(RuntimeError):
        download.download_tile("https://tap", "query", dtypes, mapping, path, expected=3)
    assert not path.exists()

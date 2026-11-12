"""Tests of catalogs/euclid_catalog.py and scripts/download_euclid_dr1.py, with a mock catalog."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("streamobs")

from streameucliddes.catalogs import euclid_catalog  # noqa: E402

pytestmark = pytest.mark.euclid_catalog

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_euclid_dr1.py"


@pytest.fixture(scope="module")
def download():
    """The download script, imported as a module."""
    spec = importlib.util.spec_from_file_location("download_euclid_dr1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def catalog_dir(tmp_path, rng):
    """A one-partition HATS-like catalog with the default columns."""
    n = 1000
    data = {c: rng.uniform(0.0, 1.0, n) for c in euclid_catalog.DEFAULT_COLUMNS}
    data["OBJECT_ID"] = np.arange(n)
    data["SPURIOUS_FLAG"] = rng.integers(0, 2, n)
    for flux_col, err_col in euclid_catalog.EUCLID_FLUX_COLUMNS.values():
        data[flux_col] = rng.uniform(-1.0, 100.0, n)  # micro-Jy, some non-positive
        data[err_col] = rng.uniform(0.01, 1.0, n)
    data["EBV_SFD98"] = np.full(n, 0.1)
    # 64-bit ids with nulls; 2**62 + i is not exactly representable as float64
    data["GAIA_ID"] = pd.array([2**62 + i if i % 3 else None for i in range(n)], dtype="Int64")
    table = pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False)
    dataset = tmp_path / "euclid_dr1_mer" / "dataset"
    (dataset / "Norder=0" / "Dir=0").mkdir(parents=True)
    pq.write_table(table, dataset / "Norder=0" / "Dir=0" / "Npix=0.parquet")
    pq.write_metadata(table.schema, dataset / "_common_metadata")
    (tmp_path / "euclid_dr1_mer" / "hats.properties").write_text("obs_collection=euclid_dr1_mer\n")
    return tmp_path, pd.DataFrame(data)


def test_open_with_cuts(catalog_dir):
    path, data = catalog_dir
    read = euclid_catalog.open_euclid_data(data_path=path, cuts={"SPURIOUS_FLAG": 0}, verbose=False)
    assert list(read.columns) == euclid_catalog.DEFAULT_COLUMNS
    assert len(read) == (data["SPURIOUS_FLAG"] == 0).sum()


def test_open_keeps_64bit_ids_with_nulls(catalog_dir):
    path, data = catalog_dir
    read = euclid_catalog.open_euclid_data(
        columns=["OBJECT_ID", "GAIA_ID"], data_path=path, verbose=False
    )
    assert read["OBJECT_ID"].dtype == np.int64  # no null: plain int64
    pd.testing.assert_series_equal(read["GAIA_ID"], data["GAIA_ID"].rename("GAIA_ID"),
                                   check_index=False)


def test_open_unknown_column(catalog_dir):
    with pytest.raises(KeyError):
        euclid_catalog.open_euclid_data(columns=["NOT_A_COLUMN"], data_path=catalog_dir[0])


def test_magnitudes(catalog_dir):
    path, data = catalog_dir
    read = euclid_catalog.add_magnitudes(euclid_catalog.open_euclid_data(data_path=path, verbose=False))
    ns = euclid_catalog.EUCLID_NAMESPACE
    flux = data["FLUX_VIS_PSF"].to_numpy()
    expected = -2.5 * np.log10(flux * 1e-6 / 3631.0) - euclid_catalog.EUCLID_SURVEY.coeff_extinc["VIS"] * 0.1
    np.testing.assert_allclose(read[f"{ns}_VIS_obs"][flux > 0], expected[flux > 0])
    assert read[f"{ns}_VIS_obs"][flux <= 0].isna().all()
    np.testing.assert_array_equal(read["ra"], data["RIGHT_ASCENSION"])


def test_stars(catalog_dir):
    stars = euclid_catalog.get_euclid_stars(data_path=catalog_dir[0], verbose=False)
    assert (stars["POINT_LIKE_PROB"] >= 0.5).all()
    assert (stars["SPURIOUS_FLAG"] == 0).all()
    assert (stars[f"{euclid_catalog.EUCLID_NAMESPACE}_VIS_obs"] >= euclid_catalog.EUCLID_VIS_SATURATION).all()


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

import zipfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from deeprvat.deeprvat.associate import combine_burden_chunks_

script_dir = Path(__file__).resolve().parent
tests_data_dir = script_dir / "test_data" / "associate"


def open_zarr(zarr_path: Path):
    zarr_data = zarr.open(zarr_path.as_posix(), mode="r")
    return zarr_data


def unzip_data(zip_path, out_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_path)

        return out_path


@pytest.fixture
def chunks_data(request, tmp_path) -> Path:
    zipped_chunks_path = Path(request.param)
    chunks_unpacked_path = tmp_path / "burdens"
    unzip_data(zip_path=zipped_chunks_path, out_path=chunks_unpacked_path)

    yield chunks_unpacked_path


@pytest.fixture
def expected_array(request, tmp_path) -> Path:
    zipped_expected_path = Path(request.param)
    expected_data_unpacked_path = tmp_path / "expected"
    unzip_data(zip_path=zipped_expected_path, out_path=expected_data_unpacked_path)

    yield expected_data_unpacked_path


@pytest.mark.parametrize(
    "n_chunks, skip_burdens, overwrite, chunks_data, expected_array",
    [
        (
            2,
            False,
            False,
            tests_data_dir / "combine_burden_chunks/input/chunks.zip",
            tests_data_dir / "combine_burden_chunks/expected/burdens.zip",
        ),
        (
            2,
            True,
            False,
            tests_data_dir / "combine_burden_chunks/input/chunks.zip",
            tests_data_dir / "combine_burden_chunks/expected/burdens.zip",
        ),
        (
            2,
            False,
            True,
            tests_data_dir / "combine_burden_chunks/input/chunks.zip",
            tests_data_dir / "combine_burden_chunks/expected/burdens.zip",
        ),
        (
            2,
            True,
            True,
            tests_data_dir / "combine_burden_chunks/input/chunks.zip",
            tests_data_dir / "combine_burden_chunks/expected/burdens.zip",
        ),
    ],
    indirect=["chunks_data", "expected_array"],
)
def test_combine_burden_chunks_data_same(
    n_chunks,
    skip_burdens,
    overwrite,
    tmp_path,
    chunks_data,
    expected_array,
):

    combine_burden_chunks_(
        n_chunks=n_chunks,
        burdens_chunks_dir=chunks_data,
        skip_burdens=skip_burdens,
        overwrite=overwrite,
        result_dir=tmp_path,
    )

    expected_files = ["burdens.zarr", "x.zarr", "y.zarr", "sample_ids.zarr"]
    for expected_file in expected_files:

        if expected_file == "burdens.zarr" and skip_burdens:
            continue

        expected_burdens_data = open_zarr(zarr_path=expected_array / expected_file)
        written_burdens_data = open_zarr(zarr_path=tmp_path / expected_file)

        assert np.array_equal(expected_burdens_data, written_burdens_data)


@pytest.mark.parametrize(
    "n_chunks, skip_burdens, overwrite, chunks_data",
    [
        (
            2,
            False,
            False,
            tests_data_dir / "combine_burden_chunks/input/chunks.zip",
        ),
        (
            2,
            True,
            False,
            tests_data_dir / "combine_burden_chunks/input/chunks.zip",
        ),
        (
            2,
            False,
            True,
            tests_data_dir / "combine_burden_chunks/input/chunks.zip",
        ),
        (
            2,
            True,
            True,
            tests_data_dir / "combine_burden_chunks/input/chunks.zip",
        ),
    ],
    indirect=["chunks_data"],
)
def test_combine_burden_chunks_file_exists(
    n_chunks, skip_burdens, overwrite, tmp_path, chunks_data
):

    combine_burden_chunks_(
        n_chunks=n_chunks,
        burdens_chunks_dir=chunks_data,
        skip_burdens=skip_burdens,
        overwrite=overwrite,
        result_dir=tmp_path,
    )

    if not skip_burdens:
        assert (tmp_path / "burdens.zarr").exists()
    else:
        assert not (tmp_path / "burdens.zarr").exists()
    assert (tmp_path / "x.zarr").exists()
    assert (tmp_path / "y.zarr").exists()
    assert (tmp_path / "sample_ids.zarr").exists()

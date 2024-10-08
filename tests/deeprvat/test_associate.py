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
    chunks_unpacked_path = tmp_path / "chunks"
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
            n_chunks,
            False,
            False,
            tests_data_dir / f"combine_burden_chunks/input/chunks_{n_chunks}.zip",
            tests_data_dir / f"combine_burden_chunks/expected/burdens_{n_chunks}.zip",
        )
        for n_chunks in range(2, 6)
    ]
    + [
        (
            n_chunks,
            True,
            True,
            tests_data_dir / f"combine_burden_chunks/input/chunks_{n_chunks}.zip",
            tests_data_dir / f"combine_burden_chunks/expected/burdens_{n_chunks}.zip",
        )
        for n_chunks in range(2, 6)
    ]
    + [
        (
            n_chunks,
            True,
            False,
            tests_data_dir / f"combine_burden_chunks/input/chunks_{n_chunks}.zip",
            tests_data_dir / f"combine_burden_chunks/expected/burdens_{n_chunks}.zip",
        )
        for n_chunks in range(2, 6)
    ]
    + [
        (
            n_chunks,
            False,
            True,
            tests_data_dir / f"combine_burden_chunks/input/chunks_{n_chunks}.zip",
            tests_data_dir / f"combine_burden_chunks/expected/burdens_{n_chunks}.zip",
        )
        for n_chunks in range(2, 6)
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

    zarr_files = ["sample_ids.zarr", "burdens.zarr"]
    if skip_burdens:
        zarr_files.remove("burdens.zarr")

    for zarr_file in zarr_files:

        expected_data = open_zarr(zarr_path=(expected_array / zarr_file))
        written_data = open_zarr(zarr_path=(tmp_path / zarr_file))
        expected_data_arr, written_data_arr = expected_data[:], written_data[:]
        assert written_data_arr.dtype == expected_data.dtype
        assert expected_data_arr.shape == written_data_arr.shape
        if zarr_file == "sample_ids.zarr":
            assert np.array_equal(expected_data_arr, written_data_arr, equal_nan=False)
        else:
            assert np.array_equal(expected_data_arr, written_data_arr, equal_nan=True)

        # No more than 10% zeros
        nr_zeros = np.count_nonzero(written_data_arr == 0)
        zero_percentage = nr_zeros / len(written_data_arr)
        assert zero_percentage < 0.1


@pytest.mark.parametrize(
    "n_chunks, skip_burdens, overwrite, chunks_data",
    [
        (
            n_chunks,
            False,
            False,
            tests_data_dir / f"combine_burden_chunks/input/chunks_{n_chunks}.zip",
        )
        for n_chunks in range(2, 6)
    ]
    + [
        (
            n_chunks,
            True,
            True,
            tests_data_dir / f"combine_burden_chunks/input/chunks_{n_chunks}.zip",
        )
        for n_chunks in range(2, 6)
    ]
    + [
        (
            n_chunks,
            True,
            False,
            tests_data_dir / f"combine_burden_chunks/input/chunks_{n_chunks}.zip",
        )
        for n_chunks in range(2, 6)
    ]
    + [
        (
            n_chunks,
            False,
            True,
            tests_data_dir / f"combine_burden_chunks/input/chunks_{n_chunks}.zip",
        )
        for n_chunks in range(2, 6)
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
    assert (tmp_path / "sample_ids.zarr").exists()

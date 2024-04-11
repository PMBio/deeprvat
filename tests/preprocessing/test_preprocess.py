import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from deeprvat.preprocessing.preprocess import cli as preprocess_cli
from click.testing import CliRunner
from pathlib import Path
import h5py
import pytest

script_dir = Path(__file__).resolve().parent
tests_data_dir = script_dir / "test_data"


def load_h5_archive(h5_path):
    with h5py.File(h5_path, "r") as f:
        written_samples = f["samples"][:]
        written_variant_matrix = f["variant_matrix"][:]
        written_genotype_matrix = f["genotype_matrix"][:]

        return written_samples, written_variant_matrix, written_genotype_matrix


@pytest.mark.parametrize(
    "test_data_name_dir, extra_cli_params, input_h5, result_h5",
    [

        (
            "filter_calls_variants_samples_minimal_split",
            [
                "--chromosomes",
                "1,2",
                "--exclude-calls",
                f"{(tests_data_dir / 'process_and_combine_sparse_gt/filter_calls_variants_samples_minimal_split/input/qc/calls/').as_posix()}",
                "--exclude-samples",
                f"{(tests_data_dir / 'process_and_combine_sparse_gt/filter_calls_variants_samples_minimal_split/input/qc/samples/').as_posix()}",
                "--exclude-variants",
                f"{(tests_data_dir / 'process_and_combine_sparse_gt/filter_calls_variants_samples_minimal_split/input/qc/variants/').as_posix()}",
            ],
            [
                "genotypes_chr1.h5",
                "genotypes_chr2.h5",
            ],
            "genotypes.h5",
        ),
    ],
)
def test_process_and_combine_sparse_gt(test_data_name_dir, extra_cli_params,
                                       input_h5, result_h5, tmp_path):
    cli_runner = CliRunner()

    current_test_data_dir = (tests_data_dir / "process_and_combine_sparse_gt" /
                             test_data_name_dir)

    test_data_input_dir = current_test_data_dir / "input"

    variant_file = test_data_input_dir / "variants.parquet"
    samples_file = test_data_input_dir / "samples_chr.csv"
    sparse_gt_dir = test_data_input_dir / "sparse_gt"

    preprocessed_dir = tmp_path / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=False)

    out_file_base = preprocessed_dir / "genotypes"
    expected_array_archive = current_test_data_dir / "expected/expected_data.npz"
    combined_output_h5 = preprocessed_dir / result_h5

    cli_parameters_process = [
        "process-sparse-gt",
        *extra_cli_params,
        "--chunksize",
        "3",
        variant_file.as_posix(),
        samples_file.as_posix(),
        sparse_gt_dir.as_posix(),
        out_file_base.as_posix(),
    ]

    result_process = cli_runner.invoke(preprocess_cli,
                                       cli_parameters_process,
                                       catch_exceptions=False)
    assert result_process.exit_code == 0

    cli_parameters_combine = [
        "combine-genotypes",
        "--chunksize",
        3,
        *[(preprocessed_dir / h5f).as_posix() for h5f in input_h5],
        combined_output_h5.as_posix(),
    ]

    result_combine = cli_runner.invoke(preprocess_cli,
                                       cli_parameters_combine,
                                       catch_exceptions=False)
    assert result_combine.exit_code == 0

    written_samples, written_variant_matrix, written_genotype_matrix = load_h5_archive(
        h5_path=combined_output_h5)

    expected_data = np.load(expected_array_archive.as_posix(),
                            allow_pickle=True)

#    assert np.array_equal(written_variant_matrix,
#                          expected_data["variant_matrix"])
    assert np.array_equal(written_genotype_matrix,
                          expected_data["genotype_matrix"])
    assert np.array_equal(written_samples, expected_data["samples"])

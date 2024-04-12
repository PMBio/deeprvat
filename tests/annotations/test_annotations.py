from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from deeprvat.annotations.annotations import cli as annotations_cli

script_dir = Path(__file__).resolve().parent
tests_data_dir = script_dir / "test_data"


@pytest.mark.parametrize(
    "test_data_name_dir, in_variants, out_variants, expected_out_variants",
    [
        (
                "process_annotations_small",
                "in_variants.parquet",
                "out_variants.parquet",
                "expected_variants.parquet",
        ),
    ],
)
def test_process_annotations(
        test_data_name_dir, in_variants, out_variants, expected_out_variants, tmp_path
):
    cli_runner = CliRunner()

    current_test_data_dir = tests_data_dir / "process_annotations" / test_data_name_dir

    in_variants_file = current_test_data_dir / in_variants
    out_variants_file = tmp_path / out_variants
    expected_out_variants_file = current_test_data_dir / expected_out_variants

    cli_parameters = [
        "process-annotations",
        in_variants_file.as_posix(),
        out_variants_file.as_posix(),
    ]

    result = cli_runner.invoke(annotations_cli, cli_parameters)
    assert result.exit_code == 0

    written_results = pd.read_parquet(out_variants_file)

    expected_variants_data = pd.read_parquet(expected_out_variants_file)
    assert pd.testing.assert_frame_equal(written_results, expected_variants_data)

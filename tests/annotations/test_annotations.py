from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from pandas.testing import assert_frame_equal
from deeprvat.annotations.annotations import cli as annotations_cli

script_dir = Path(__file__).resolve().parent
tests_data_dir = script_dir / "test_data"


@pytest.mark.parametrize(
    "test_data_name_dir, deepsea_scores_1, deepsea_scores_2, out_scores, expected_out_scores",
    [
        (
                "concatenate_deepseascores_small",
                "deepsea_scores_1.tsv",
                "deepsea_scores_2.tsv",
                "out_scores.parquet",
                "expected.parquet",
        ),
    ],
)
def test_concatenate_deepsea(
        test_data_name_dir, deepsea_scores_1, deepsea_scores_2, out_scores, expected_out_scores, tmp_path
):
    cli_runner = CliRunner()

    current_test_data_dir = tests_data_dir / "concatenate_deepsea" / test_data_name_dir

    deepsea_score_file_1 = current_test_data_dir / "input" / deepsea_scores_1
    deepsea_score_file_2 = current_test_data_dir / "input" / deepsea_scores_2
    out_out_scores_file = tmp_path / out_scores
    expected_out_scores_file = current_test_data_dir / "expected" / expected_out_scores

    cli_parameters = [
        "concatenate-deepsea",
        ",".join([deepsea_score_file_1.as_posix(), deepsea_score_file_2.as_posix()]),
        out_out_scores_file.as_posix(),"8"
    ]

    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0

    written_results = pd.read_parquet(out_out_scores_file)

    expected_scores = pd.read_parquet(expected_out_scores_file)
    assert_frame_equal(written_results, expected_scores, check_exact = False)

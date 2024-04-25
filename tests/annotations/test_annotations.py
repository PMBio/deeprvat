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
        (
                "concatenate_deepseascores_medium",
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
    out_scores_file = tmp_path / out_scores
    expected_out_scores_file = current_test_data_dir / "expected" / expected_out_scores

    cli_parameters = [
        "concatenate-deepsea",
        ",".join([deepsea_score_file_1.as_posix(), deepsea_score_file_2.as_posix()]),
        out_scores_file.as_posix(),
        "8"
    ]

    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0

    written_results = pd.read_parquet(out_scores_file)

    expected_scores = pd.read_parquet(expected_out_scores_file)
    assert_frame_equal(written_results, expected_scores, check_exact = False)





@pytest.mark.parametrize(
    "test_data_name_dir, deapseascores_file, variant_file, out_df, expected_out_df",
    [
        (
                "add_ids_small",
                "deepseascores.parquet",
                "variants.parquet",
                "out_df.parquet",
                "expected.parquet",
        ),
        (
                "add_ids_medium",
                "deepseascores.parquet",
                "variants.parquet",
                "out_df.parquet",
                "expected.parquet",
        ),
    ],
)
def test_add_ids_dask(
        test_data_name_dir, deapseascores_file, variant_file, out_df, expected_out_df, tmp_path
):
    cli_runner = CliRunner()

    current_test_data_dir = tests_data_dir / "add_ids_dask" / test_data_name_dir

    deepsea_score_file = current_test_data_dir / "input" / deapseascores_file
    variant_path = current_test_data_dir / "input" / variant_file
    out_scores_file = tmp_path / out_df
    expected_out_scores_file = current_test_data_dir / "expected" / expected_out_df

    cli_parameters = [
        'add-ids-dask',
        deepsea_score_file.as_posix(), 
        variant_path.as_posix(),
        out_scores_file.as_posix()
    ]

    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0

    written_results = pd.read_parquet(out_scores_file)

    expected_scores = pd.read_parquet(expected_out_scores_file)
    assert_frame_equal(written_results, expected_scores, check_exact = False)





@pytest.mark.parametrize(
    "test_data_name_dir, deapseascores_file, pca_file, mean_sds_file, expected_out_df",
    [
        (
                "deepsea_pca_small",
                "deepseascores.parquet",
                "deepsea_pca/pca.npy",
                "deepsea_pca/mean_sds.parquet",
                "deepsea_pca.parquet",
        ),
        (
                "deepsea_pca_small",
                "deepseascores.parquet",
                "deepsea_pca/pca.npy",
                "{tmp_path}/mean_sds.parquet",
                "deepsea_pca.parquet",
        ),
        (
                "deepsea_pca_small",
                "deepseascores.parquet",
                "{tmp_path}/pca.npy",
                "deepsea_pca/mean_sds.parquet",
                "deepsea_pca.parquet",
        ),
        (
                "deepsea_pca_small",
                "deepseascores.parquet",
                "{tmp_path}/pca.npy",
                "{tmp_path}/mean_sds.parquet",
                "deepsea_pca.parquet",
        ),
    ],
)
def test_deepsea_pca(
        test_data_name_dir, deapseascores_file, pca_file, mean_sds_file, expected_out_df, tmp_path
):
    cli_runner = CliRunner()

    current_test_data_dir = tests_data_dir / "deepsea_pca" / test_data_name_dir

    deepsea_score_file = current_test_data_dir / "input" / deapseascores_file
    pca_file = current_test_data_dir / pca_file.format(tmp_path = tmp_path)
    mean_sds_file = current_test_data_dir / mean_sds_file.format(tmp_path = tmp_path)
    expected_out_scores_file = current_test_data_dir / "expected" / expected_out_df

    cli_parameters = [
        'deepsea-pca',
        deepsea_score_file.as_posix(), 
        pca_file.as_posix(),
        mean_sds_file.as_posix(),
        tmp_path.as_posix()
    ]

    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0

    written_results = pd.read_parquet(tmp_path / expected_out_df)

    expected_scores = pd.read_parquet(expected_out_scores_file)
    assert_frame_equal(written_results, expected_scores, check_exact = False)


@pytest.mark.parametrize(
    "test_data_name_dir, expected, hg2_output, k5_output, parclip_output, variants, vcf, vep_output, vep_header_line",
    [
        (
            "merge_annotations_small",
            "merged_annotations_expected.parquet",
            "test_hg2_deepripe.csv.gz",
            "test_k5_deepripe.csv.gz",
            "test_parclip.csv.gz",
            "variants.parquet",
            "test.vcf",
            "test_vep.tsv",
            "49",

        ),

    ]
)
def test_merge_annotations(
    test_data_name_dir, expected, hg2_output, k5_output, parclip_output, variants, vcf, vep_output, vep_header_line, tmp_path
    ):
    current_test_data_dir = tests_data_dir / 'merge_annotations'/ test_data_name_dir
    expected_path = current_test_data_dir / 'expected' / expected
    hg2_deepripe_path = current_test_data_dir / 'input' / hg2_output
    k5_deepripe_path = current_test_data_dir / 'input' / k5_output
    parclip_deepripe_path = current_test_data_dir / 'input' / parclip_output
    vcf_path = current_test_data_dir / 'input' / vcf
    vep_output_path = current_test_data_dir / 'input' / vep_output
    variants_path = current_test_data_dir / 'input' / variants
    output_path = tmp_path / 'out_merged.parquet'
    cli_runner = CliRunner()

    cli_parameters = ["merge-annotations",
                      vep_header_line,
                      vep_output_path.as_posix(),
                      parclip_deepripe_path.as_posix(),
                      hg2_deepripe_path.as_posix(),
                      k5_deepripe_path.as_posix(),
                      variants_path.as_posix(), 
                      vcf_path.as_posix(),
                      output_path.as_posix(),
                      ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(written_results[expected_results.columns], expected_results, check_exact = False)
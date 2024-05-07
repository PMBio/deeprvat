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
        (
            "merge_annotations_mixedIDs",
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


@pytest.mark.parametrize(
    "test_data_name_dir, expected, merged_files",
    [
        (
            "concatenate_annotations_small",
            "expected.parquet",
            "chr3test_merged.parquet,chr4test_merged.parquet",
        ),
    ]
)
def test_concatenate_annotations(
    test_data_name_dir, expected, merged_files, tmp_path
    ):
    current_test_data_dir = tests_data_dir / 'concatenate_annotations'/ test_data_name_dir
    expected_path = current_test_data_dir / 'expected' / expected
    merged_files  = [(current_test_data_dir/'input'/ file).as_posix() for file in merged_files.split(',')]
    output_path = tmp_path / 'out.parquet'
    cli_runner = CliRunner()

    cli_parameters = ["concat-annotations",
                      ",".join(merged_files),
                      output_path.as_posix(),
                      ]
    
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(written_results[expected_results.columns], expected_results, check_exact = False)


@pytest.mark.parametrize(
    "test_data_name_dir, annotations, deepSEA_scores, annotation_columns_yaml_file, expected",
    [
        (
            "merge_deepsea_pcas_small",
            "vep_deepripe.parquet",
            "all_variants.wID.deepSea.parquet",
            "annotation_colnames_filling_values.yaml",
            "expected.parquet"
        ),
    ]
)
def test_merge_deepsea_pcas(
     test_data_name_dir, annotations, deepSEA_scores, annotation_columns_yaml_file, expected, tmp_path
):
    current_test_data_dir = tests_data_dir / 'merge_deepsea_pcas' / test_data_name_dir
    expected_path = current_test_data_dir / 'expected' / expected
    annotations_path = current_test_data_dir / 'input' / annotations
    deepSEA_scores_path = current_test_data_dir / 'input' / deepSEA_scores
    annotation_columns_yaml_path = current_test_data_dir / 'input' / annotation_columns_yaml_file
    output_path = tmp_path / 'out.parquet'
    cli_runner = CliRunner()
    cli_parameters = ['merge-deepsea-pcas',
                      annotations_path.as_posix(),
                      deepSEA_scores_path.as_posix(),
                      annotation_columns_yaml_path.as_posix(),
                      output_path.as_posix()
                      ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(written_results, expected_results[written_results.columns], check_exact = False)


@pytest.mark.parametrize(
    "test_data_name_dir, annotations, abSplice_score_dir, njobs, expected",
    [
        (
            "aggregate_absplice_scores_small",
            "vep_deepripe_deepsea.parquet",
            "absplice_res_dir",
            "8",
            "abSplice_score_file.parquet",
        ),
    ]
)
def test_aggregate_abscores(
     test_data_name_dir, annotations, abSplice_score_dir, njobs, expected, tmp_path
):
    current_test_data_dir = tests_data_dir / 'aggregate_absplice_scores' / test_data_name_dir
    annotations_path = current_test_data_dir / 'input' /  annotations
    abscore_path = current_test_data_dir / 'input' /abSplice_score_dir
    expected_path = current_test_data_dir / 'expected' / expected
    output_path = tmp_path / 'out.parquet'
    cli_runner = CliRunner()
    cli_parameters = [
        'aggregate-abscores',
        annotations_path.as_posix(),
        abscore_path.as_posix(),
        output_path.as_posix(),
        njobs
        ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(written_results, expected_results[written_results.columns], check_exact = False)



@pytest.mark.parametrize(
    "test_name_dir, absplice_scores, annotations, expected",
    [
        (   "merge_absplice_scores_small",
            "abSplice_score_file.parquet",
            "vep_deepripe_deepsea.parquet",
            "vep_deepripe_deepsea_absplice.parquet",
        ),
    ]
)
def test_merge_absplice_scores(
     test_name_dir, absplice_scores, annotations, expected, tmp_path
):
    current_test_data_dir = tests_data_dir / 'merge_absplice_scores' / test_name_dir
    absplice_score_path = current_test_data_dir / 'input' /  absplice_scores
    annotation_path = current_test_data_dir / 'input' / annotations
    expected_path = current_test_data_dir / 'expected' / expected
    output_path = tmp_path / 'out.parquet'
    cli_runner = CliRunner()
    cli_parameters = [
        'merge-abscores',
        annotation_path.as_posix(),
        absplice_score_path.as_posix(),
        output_path.as_posix(),
        ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(written_results, expected_results, check_exact = False)


# @pytest.mark.parametrize(
#     "test_name_dir, input_file_1, input_file_2, parameter1, expected",
#     [
#         (   "test_name_dir",
#             "input_file1.parquet",
#             "input_file2.parquet",
#             "8",
#             "expected.parquet",
#         ),
#     ]
# )
# def template(
#      test_data_name_dir, input_file_1, input_file_2, parameter1, expected, tmp_path
# ):
#     current_test_data_dir = tests_data_dir / 'test_name' / test_data_name_dir
#     input_path1 = current_test_data_dir / 'input' /  input_file_1
#     input_path2 = current_test_data_dir / 'input' /input_file_2
#     expected_path = current_test_data_dir / 'expected' / expected
#     output_path = tmp_path / 'out.parquet'
#     cli_runner = CliRunner()
#     cli_parameters = [
#         'function-name',
#         input_path1.as_posix(),
#         input_path2.as_posix(),
#         output_path.as_posix(),
#         parameter1,
#         ]
#     result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
#     assert result.exit_code == 0
#     written_results = pd.read_parquet(output_path)
#     expected_results = pd.read_parquet(expected_path)
#     assert written_results.shape == expected_results.shape
#     assert_frame_equal(written_results, expected_results[written_results.columns], check_exact = False)

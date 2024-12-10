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
    test_data_name_dir,
    deepsea_scores_1,
    deepsea_scores_2,
    out_scores,
    expected_out_scores,
    tmp_path,
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
        "8",
    ]

    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0

    written_results = pd.read_parquet(out_scores_file)

    expected_scores = pd.read_parquet(expected_out_scores_file)
    assert_frame_equal(written_results, expected_scores, check_exact=False)


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
    test_data_name_dir,
    deapseascores_file,
    variant_file,
    out_df,
    expected_out_df,
    tmp_path,
):
    cli_runner = CliRunner()

    current_test_data_dir = tests_data_dir / "add_ids_dask" / test_data_name_dir

    deepsea_score_file = current_test_data_dir / "input" / deapseascores_file
    variant_path = current_test_data_dir / "input" / variant_file
    out_scores_file = tmp_path / out_df
    expected_out_scores_file = current_test_data_dir / "expected" / expected_out_df

    cli_parameters = [
        "add-ids-dask",
        deepsea_score_file.as_posix(),
        variant_path.as_posix(),
        out_scores_file.as_posix(),
    ]

    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0

    written_results = pd.read_parquet(out_scores_file)

    expected_scores = pd.read_parquet(expected_out_scores_file)
    assert_frame_equal(written_results, expected_scores, check_exact=False)


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
    test_data_name_dir,
    deapseascores_file,
    pca_file,
    mean_sds_file,
    expected_out_df,
    tmp_path,
):
    cli_runner = CliRunner()

    current_test_data_dir = tests_data_dir / "deepsea_pca" / test_data_name_dir

    deepsea_score_file = current_test_data_dir / "input" / deapseascores_file
    pca_file = current_test_data_dir / pca_file.format(tmp_path=tmp_path)
    mean_sds_file = current_test_data_dir / mean_sds_file.format(tmp_path=tmp_path)
    expected_out_scores_file = current_test_data_dir / "expected" / expected_out_df

    cli_parameters = [
        "deepsea-pca",
        deepsea_score_file.as_posix(),
        pca_file.as_posix(),
        mean_sds_file.as_posix(),
        tmp_path.as_posix(),
    ]

    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0

    written_results = pd.read_parquet(tmp_path / expected_out_df)

    expected_scores = pd.read_parquet(expected_out_scores_file)
    assert_frame_equal(written_results, expected_scores, check_exact=False)


@pytest.mark.parametrize(
    "test_data_name_dir, expected, hg2_output, k5_output, parclip_output, variants, vcf, vep_output, vep_header_line, column_yaml_file",
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
            "annotation_colnames_filling_values.yaml",
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
            "annotation_colnames_filling_values.yaml",
        ),
        (
            "merge_annotations_allNAN_spliceAI",
            "merged_annotations_expected.parquet",
            "test_hg2_deepripe.csv.gz",
            "test_k5_deepripe.csv.gz",
            "test_parclip.csv.gz",
            "variants.parquet",
            "test.vcf",
            "test_vep.tsv",
            "0",
            "annotation_colnames_filling_values.yaml",
        ),
        (
            "merge_annotations_allbut1NAN_spliceAI",
            "merged_annotations_expected.parquet",
            "test_hg2_deepripe.csv.gz",
            "test_k5_deepripe.csv.gz",
            "test_parclip.csv.gz",
            "variants.parquet",
            "test.vcf",
            "test_vep.tsv",
            "0",
            "annotation_colnames_filling_values.yaml",
        ),
    ],
)
def test_merge_annotations(
    test_data_name_dir,
    expected,
    hg2_output,
    k5_output,
    parclip_output,
    variants,
    vcf,
    vep_output,
    vep_header_line,
    column_yaml_file,
    tmp_path,
):
    current_test_data_dir = tests_data_dir / "merge_annotations" / test_data_name_dir
    expected_path = current_test_data_dir / "expected" / expected
    hg2_deepripe_path = current_test_data_dir / "input" / hg2_output
    k5_deepripe_path = current_test_data_dir / "input" / k5_output
    parclip_deepripe_path = current_test_data_dir / "input" / parclip_output
    vcf_path = current_test_data_dir / "input" / vcf
    vep_output_path = current_test_data_dir / "input" / vep_output
    column_yaml_file_path = current_test_data_dir / "input" / column_yaml_file
    variants_path = current_test_data_dir / "input" / variants
    output_path = tmp_path / "out_merged.parquet"
    cli_runner = CliRunner()

    cli_parameters = [
        "merge-annotations",
        vep_header_line,
        vep_output_path.as_posix(),
        parclip_deepripe_path.as_posix(),
        hg2_deepripe_path.as_posix(),
        k5_deepripe_path.as_posix(),
        variants_path.as_posix(),
        vcf_path.as_posix(),
        output_path.as_posix(),
        column_yaml_file_path.as_posix(),
    ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(
        written_results[expected_results.columns], expected_results, check_exact=False
    )


@pytest.mark.parametrize(
    "test_data_name_dir, expected, merged_files",
    [
        (
            "concatenate_annotations_small",
            "expected.parquet",
            "chr3test_merged.parquet,chr4test_merged.parquet",
        ),
    ],
)
def test_concatenate_annotations(test_data_name_dir, expected, merged_files, tmp_path):
    current_test_data_dir = (
        tests_data_dir / "concatenate_annotations" / test_data_name_dir
    )
    expected_path = current_test_data_dir / "expected" / expected
    merged_files = [
        (current_test_data_dir / "input" / file).as_posix()
        for file in merged_files.split(",")
    ]
    output_path = tmp_path / "out.parquet"
    cli_runner = CliRunner()

    cli_parameters = [
        "concat-annotations",
        ",".join(merged_files),
        output_path.as_posix(),
    ]

    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(
        written_results[expected_results.columns], expected_results, check_exact=False
    )


@pytest.mark.parametrize(
    "test_data_name_dir, annotations, deepSEA_scores, annotation_columns_yaml_file, expected",
    [
        (
            "merge_deepsea_pcas_small",
            "vep_deepripe.parquet",
            "all_variants.wID.deepSea.parquet",
            "annotation_colnames_filling_values.yaml",
            "expected.parquet",
        ),
    ],
)
def test_merge_deepsea_pcas(
    test_data_name_dir,
    annotations,
    deepSEA_scores,
    annotation_columns_yaml_file,
    expected,
    tmp_path,
):
    current_test_data_dir = tests_data_dir / "merge_deepsea_pcas" / test_data_name_dir
    expected_path = current_test_data_dir / "expected" / expected
    annotations_path = current_test_data_dir / "input" / annotations
    deepSEA_scores_path = current_test_data_dir / "input" / deepSEA_scores
    annotation_columns_yaml_path = (
        current_test_data_dir / "input" / annotation_columns_yaml_file
    )
    output_path = tmp_path / "out.parquet"
    cli_runner = CliRunner()
    cli_parameters = [
        "merge-deepsea-pcas",
        annotations_path.as_posix(),
        deepSEA_scores_path.as_posix(),
        annotation_columns_yaml_path.as_posix(),
        output_path.as_posix(),
    ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(
        written_results, expected_results[written_results.columns], check_exact=False
    )


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
    ],
)
def test_aggregate_abscores(
    test_data_name_dir, annotations, abSplice_score_dir, njobs, expected, tmp_path
):
    current_test_data_dir = (
        tests_data_dir / "aggregate_absplice_scores" / test_data_name_dir
    )
    annotations_path = current_test_data_dir / "input" / annotations
    abscore_path = current_test_data_dir / "input" / abSplice_score_dir
    expected_path = current_test_data_dir / "expected" / expected
    output_path = tmp_path / "out.parquet"
    cli_runner = CliRunner()
    cli_parameters = [
        "aggregate-abscores",
        annotations_path.as_posix(),
        abscore_path.as_posix(),
        output_path.as_posix(),
        njobs,
    ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(
        written_results, expected_results[written_results.columns], check_exact=False
    )


@pytest.mark.parametrize(
    "test_name_dir, absplice_scores, annotations, expected",
    [
        (
            "merge_absplice_scores_small",
            "abSplice_score_file.parquet",
            "vep_deepripe_deepsea.parquet",
            "vep_deepripe_deepsea_absplice.parquet",
        ),
        (
            "merge_absplice_scores_exons",
            "abSplice_score_file.parquet",
            "vep_deepripe_deepsea.parquet",
            "vep_deepripe_deepsea_absplice.parquet",
        ),
    ],
)
def test_merge_absplice_scores(
    test_name_dir, absplice_scores, annotations, expected, tmp_path
):
    current_test_data_dir = tests_data_dir / "merge_absplice_scores" / test_name_dir
    absplice_score_path = current_test_data_dir / "input" / absplice_scores
    annotation_path = current_test_data_dir / "input" / annotations
    expected_path = current_test_data_dir / "expected" / expected
    output_path = tmp_path / "out.parquet"
    cli_runner = CliRunner()
    cli_parameters = [
        "merge-abscores",
        annotation_path.as_posix(),
        absplice_score_path.as_posix(),
        output_path.as_posix(),
    ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(written_results, expected_results, check_exact=False)


@pytest.mark.parametrize(
    "test_data_name_dir, genotype_file, variant_file, expected",
    [
        (
            "calculate_allele_frequency_small",
            "genotypes.h5",
            "variants.parquet",
            "af_df.parquet",
        ),
    ],
)
def test_calculate_allele_frequencies(
    test_data_name_dir, genotype_file, variant_file, expected, tmp_path
):
    current_test_data_dir = (
        tests_data_dir / "calculate_allele_frequency" / test_data_name_dir
    )
    genotype_filepath = current_test_data_dir / "input" / genotype_file
    variant_filepath = current_test_data_dir / "input" / variant_file
    expected_path = current_test_data_dir / "expected" / expected
    output_path = tmp_path / "out.parquet"
    cli_runner = CliRunner()
    cli_parameters = [
        "get-af-from-gt",
        genotype_filepath.as_posix(),
        variant_filepath.as_posix(),
        output_path.as_posix(),
    ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(written_results, expected_results, check_exact=False)


@pytest.mark.parametrize(
    "test_data_name_dir, af_df, annotaton_df, expected",
    [
        (
            "merge_af_small",
            "af_df.parquet",
            "vep_deepripe_deepsea_absplice.parquet",
            "vep_deepripe_deepsea_absplice_af.parquet",
        ),
    ],
)
def test_merge_af(test_data_name_dir, af_df, annotaton_df, expected, tmp_path):
    current_test_data_dir = tests_data_dir / "merge_af" / test_data_name_dir
    af_path = current_test_data_dir / "input" / af_df
    annotaions_path = current_test_data_dir / "input" / annotaton_df
    expected_path = current_test_data_dir / "expected" / expected
    output_path = tmp_path / "out.parquet"
    cli_runner = CliRunner()
    cli_parameters = [
        "merge-af",
        af_path.as_posix(),
        annotaions_path.as_posix(),
        output_path.as_posix(),
    ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(
        written_results, expected_results[written_results.columns], check_exact=False
    )


@pytest.mark.parametrize(
    "test_data_name_dir, annotations, expected",
    [
        (
            "calculate_MAF_small",
            "annotations.parquet",
            "expected.parquet",
        ),
    ],
)
def test_calculate_maf(test_data_name_dir, annotations, expected, tmp_path):
    current_test_data_dir = tests_data_dir / "calculate_MAF" / test_data_name_dir
    annotations_path = current_test_data_dir / "input" / annotations
    expected_path = current_test_data_dir / "expected" / expected
    output_path = tmp_path / "out.parquet"
    cli_runner = CliRunner()
    cli_parameters = [
        "calculate-maf",
        annotations_path.as_posix(),
        output_path.as_posix(),
    ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(written_results, expected_results, check_exact=False)


@pytest.mark.parametrize(
    "test_data_name_dir, gtf_file, expected",
    [
        (
            "create_gene_id_file_small",
            "gencode.v44.annotation.gtf.gz",
            "protein_coding_genes.parquet",
        ),
    ],
)
def test_create_gene_id_file(test_data_name_dir, gtf_file, expected, tmp_path):
    current_test_data_dir = tests_data_dir / "create_gene_id_file" / test_data_name_dir
    input_path1 = current_test_data_dir / "input" / gtf_file
    expected_path = current_test_data_dir / "expected" / expected
    output_path = tmp_path / "out.parquet"
    cli_runner = CliRunner()
    cli_parameters = [
        "create-gene-id-file",
        input_path1.as_posix(),
        output_path.as_posix(),
    ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(
        written_results, expected_results[written_results.columns], check_exact=False
    )


@pytest.mark.parametrize(
    "test_data_name_dir, annotations, gene_id_file , expected",
    [
        (
            "add_gene_ids_small",
            "annotations.parquet",
            "protein_coding_genes.parquet",
            "expected.parquet",
        ),
    ],
)
def test_add_gene_ids(
    test_data_name_dir, annotations, gene_id_file, expected, tmp_path
):
    current_test_data_dir = tests_data_dir / "add_gene_ids" / test_data_name_dir
    annotations_path = current_test_data_dir / "input" / annotations
    gene_id_path = current_test_data_dir / "input" / gene_id_file
    expected_path = current_test_data_dir / "expected" / expected
    output_path = tmp_path / "out.parquet"
    cli_runner = CliRunner()
    cli_parameters = [
        "add-gene-ids",
        gene_id_path.as_posix(),
        annotations_path.as_posix(),
        output_path.as_posix(),
    ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(written_results, expected_results, check_exact=False)


@pytest.mark.parametrize(
    "test_data_name_dir, gtf_file, annotations, gene_id_file, expected",
    [
        (
            "filter_by_exon_distance_small",
            "gencode.v44.annotation.gtf.gz",
            "annotations.parquet",
            "protein_coding_genes.parquet",
            "expected.parquet",
        ),
    ],
)
def test_filter_by_exon_distance(
    test_data_name_dir, gtf_file, annotations, gene_id_file, expected, tmp_path
):
    current_test_data_dir = (
        tests_data_dir / "filter_by_exon_distance" / test_data_name_dir
    )
    gtf_file_path = current_test_data_dir / "input" / gtf_file
    annotations_path = current_test_data_dir / "input" / annotations
    gene_id_path = current_test_data_dir / "input" / gene_id_file

    expected_path = current_test_data_dir / "expected" / expected
    output_path = tmp_path / "out.parquet"
    cli_runner = CliRunner()
    cli_parameters = [
        "filter-annotations-by-exon-distance",
        annotations_path.as_posix(),
        gtf_file_path.as_posix(),
        gene_id_path.as_posix(),
        output_path.as_posix(),
    ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(written_results, expected_results, check_exact=False)


@pytest.mark.parametrize(
    "test_data_name_dir, gtf_file, annotations, gene_id_file, error_msg",
    [
        (
            "filter_by_exon_distance_filteroutall",
            "gencode.v44.annotation.gtf.gz",
            "annotations.parquet",
            "protein_coding_genes.parquet",
            "Data frame is empty after filtering on exon distance, abort.",
        ),
    ],
)
def test_filter_by_exon_distance_fail(
    test_data_name_dir, gtf_file, annotations, gene_id_file, error_msg, tmp_path
):
    current_test_data_dir = (
        tests_data_dir / "filter_by_exon_distance" / test_data_name_dir
    )
    gtf_file_path = current_test_data_dir / "input" / gtf_file
    annotations_path = current_test_data_dir / "input" / annotations
    gene_id_path = current_test_data_dir / "input" / gene_id_file

    output_path = tmp_path / "out.parquet"
    cli_runner = CliRunner()
    cli_parameters = [
        "filter-annotations-by-exon-distance",
        annotations_path.as_posix(),
        gtf_file_path.as_posix(),
        gene_id_path.as_posix(),
        output_path.as_posix(),
    ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=True)
    assert result.exit_code == 1
    assert type(result.exception) == AssertionError
    assert result.exception.args[0] == error_msg


@pytest.mark.parametrize(
    "test_data_name_dir, yaml_file, annotations, expected",
    [
        (
            "select_rename_fill_columns_small",
            "annotation_colnames_filling_values.yaml",
            "annotations.parquet",
            "expected.parquet",
        ),
        (
            "select_rename_fill_columns_plof",
            "annotation_colnames_filling_values.yaml",
            "annotations.parquet",
            "expected.parquet",
        ),
        (
            "select_rename_fill_columns_plof2",
            "annotation_colnames_filling_values.yaml",
            "annotations.parquet",
            "expected.parquet",
        ),
    ],
)
def test_select_rename_fill_annotations(
    test_data_name_dir, yaml_file, annotations, expected, tmp_path
):
    current_test_data_dir = (
        tests_data_dir / "select_rename_fill_columns" / test_data_name_dir
    )
    yaml_file_path = current_test_data_dir / "input" / yaml_file
    annotations_path = current_test_data_dir / "input" / annotations
    expected_path = current_test_data_dir / "expected" / expected
    output_path = tmp_path / "out.parquet"
    cli_runner = CliRunner()
    cli_parameters = [
        "select-rename-fill-annotations",
        yaml_file_path.as_posix(),
        annotations_path.as_posix(),
        output_path.as_posix(),
    ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(
        written_results, expected_results[written_results.columns], check_exact=False
    )

@pytest.mark.parametrize(
    "test_data_name_dir, yaml_file, annotations, expected, expected_unfilled",
    [
        (
            "select_rename_fill_columns_small",
            "annotation_colnames_filling_values.yaml",
            "annotations.parquet",
            "expected.parquet",
            "expected_unfilled.parquet",
        ),
    ],
)
def test_select_rename_fill_annotations_unfilled(
    test_data_name_dir, yaml_file, annotations, expected, expected_unfilled, tmp_path
):
    current_test_data_dir = (
        tests_data_dir / "select_rename_fill_columns" / test_data_name_dir
    )
    yaml_file_path = current_test_data_dir / "input" / yaml_file
    annotations_path = current_test_data_dir / "input" / annotations
    expected_path = current_test_data_dir / "expected" / expected
    expected_unfilled_path = current_test_data_dir / "expected" / expected_unfilled
    output_path = tmp_path / "out.parquet"
    unfilled_path = tmp_path / "unfilled.parquet"
    cli_runner = CliRunner()
    cli_parameters = [
        "select-rename-fill-annotations",
        yaml_file_path.as_posix(),
        annotations_path.as_posix(),
        output_path.as_posix(),
        "--keep_unfilled",
        unfilled_path

    ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    written_unfilled = pd.read_parquet(unfilled_path)
    expected_unfilled = pd.read_parquet(expected_unfilled_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(
        written_results, expected_results[written_results.columns], check_exact=False
    )
    assert written_unfilled.shape == expected_unfilled.shape
    assert_frame_equal(written_unfilled, expected_unfilled[written_unfilled.columns],check_exact=False)


@pytest.mark.parametrize(
    "test_data_name_dir, annotations_in, expected",
    [
        (
            "compute_plof_small",
            "annotations.parquet",
            "expected.parquet",
        ),
    ],
)
def test_compute_plof(test_data_name_dir, annotations_in, expected, tmp_path):
    current_test_data_dir = tests_data_dir / "compute_plof" / test_data_name_dir
    annotations_in_path = current_test_data_dir / "input" / annotations_in
    expected_path = current_test_data_dir / "expected" / expected
    output_path = tmp_path / "out.parquet"
    cli_runner = CliRunner()
    cli_parameters = [
        "compute-plof",
        annotations_in_path.as_posix(),
        output_path.as_posix(),
    ]
    result = cli_runner.invoke(annotations_cli, cli_parameters, catch_exceptions=False)
    assert result.exit_code == 0
    written_results = pd.read_parquet(output_path)
    expected_results = pd.read_parquet(expected_path)
    assert written_results.shape == expected_results.shape
    assert_frame_equal(
        written_results, expected_results[written_results.columns], check_exact=False
    )

rule all_cv_burdens:
    input:
        expand("{phenotype}/deeprvat/burdens/merging.finished", phenotype=phenotypes),


# # ############################### Run DeepRVAT ##############################################################
# # ###########################################################################################################
# module deeprvat_associate:
#     snakefile:
#         "../training_association_testing.snakefile"
#         # f"{DEEPRVAT_DIR}/pipelines/training_association_testing.snakefile"
#         # Wit the version below the module doesn't have the local Namespace
#         # alternative is to put the 'header'/variable definitions into all snakefiles
#         # "../association_testing/association_dataset.snakefile"
#     prefix:
#         "cv_split{cv_split}/deeprvat"
#     config:
#         config


# # ############################### Computation of test set deeprvat burdens ##############################################################


rule make_deeprvat_test_config:
    input:
        data_config="cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/config.yaml",
    output:
        data_config_test="cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/config_test.yaml",
    shell:
        " && ".join(
        [
            conda_check,
        "deeprvat_cv_utils generate-test-config "
                "--fold {wildcards.cv_split} "
                f"--n-folds {cv_splits}"
                " {input.data_config} {output.data_config_test}",
            ]
        )


# generate the association data set from the test samples (as defined in the config)
# pass the sample file here
# then just use this data set nomrally for burden computation
use rule association_dataset from deeprvat_workflow as deeprvat_association_dataset with:
    input:
        data_config="cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/config_test.yaml",
    output:
        temp("cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/association_dataset.pkl"),
    threads: 4

use rule association_dataset_burdens from deeprvat_workflow as deeprvat_association_dataset_burdens with:
    input:
        data_config=f"cv_split{{cv_split}}/deeprvat/{burden_phenotype}/deeprvat/config_test.yaml",
    output:
        temp("cv_split{cv_split}/deeprvat/burdens/association_dataset.pkl"),
    threads: 4


rule combine_test_burdens:
    input:
        samples = expand('cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/xy/sample_ids.zarr',
                         cv_split=range(cv_splits),
                         phenotype=phenotypes),
        x = expand('cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/xy/x.zarr',
                   cv_split=range(cv_splits),
                   phenotype=phenotypes),
        y = expand('cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/xy/y.zarr',
                   cv_split=range(cv_splits),
                   phenotype=phenotypes),
        burdens=expand('cv_split{cv_split}/deeprvat/burdens/burdens.zarr',
                       cv_split=range(cv_splits)),
        data_config="deeprvat_config.yaml",
    output:
        "burdens/log/{phenotype}/merging.finished",
    params:
        out_dir_burdens="burdens",
        out_dir_xy=lambda wildcards: f"{wildcards.phenotype}/deeprvat/xy",
        burden_paths=" ".join(
            [
                f"--burden-dirs cv_split{fold}/deeprvat/burdens"
                for fold in range(cv_splits)
            ]
        ),
        xy_paths = lambda wildcards, input: " ".join(
            [
                f"--xy-dirs cv_split{fold}/deeprvat/{wildcards.phenotype}/deeprvat/xy"
                for fold in range(cv_splits)
            ]
        ),
        skip=lambda wildcards: (
            f"--skip-burdens"
            if wildcards.phenotype != burden_phenotype
            else ""
        ),
    resources:
        mem_mb=lambda wildcards, attempt: 32000 + attempt * 4098 * 2,
    shell:
        " && ".join(
            [
                conda_check,
                "deeprvat_cv_utils combine-test-set-burdens "
                "{params.skip} "
                "{params.burden_paths} "
                "{params.xy_paths} "
                "{params.out_dir_burdens} "
                "{params.out_dir_xy} "
                "{input.data_config}",
                "touch {output}",
            ]
        )


# use rule average_burdens from deeprvat_workflow as deeprvat_average_burdens with:
#     prefix:
#         ""


use rule combine_burdens from deeprvat_workflow as deeprvat_combine_burdens with:
    params:
        prefix="cv_split{cv_split}/deeprvat",


use rule compute_burdens from deeprvat_workflow as deeprvat_compute_burdens with:
    params:
        prefix="cv_split{cv_split}/deeprvat",


use rule compute_xy from deeprvat_workflow as deeprvat_compute_xy with:
    input:
        dataset = 'cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/association_dataset.pkl',
        data_config = 'cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/config.yaml',
    output:
        samples = directory('cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/xy/sample_ids.zarr'),
        x = directory('cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/xy/x.zarr'),
        y = directory('cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/xy/y.zarr'),


use rule reverse_models from deeprvat_workflow as deeprvat_reverse_models

rule all_cv_burdens:
    input:
        expand("{phenotype}/deeprvat/burdens/merging.finished", phenotype=phenotypes),


# # ############################### Run DeepRVAT ##############################################################
# # ###########################################################################################################
module deeprvat_associate:
    snakefile:
        "../training_association_testing.snakefile"
        # f"{DEEPRVAT_DIR}/pipelines/training_association_testing.snakefile" 
        # Wit the version below the module doesn't have the local Namespace
        # alternative is to put the 'header'/variable definitions into all snakefiles
        # "../association_testing/association_dataset.snakefile"
    prefix:
        "cv_split{cv_split}/deeprvat"
    config:
        config


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
use rule association_dataset from deeprvat_associate as deeprvat_association_dataset with:
    input:
        data_config="cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/config_test.yaml",
    output:
        "cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/association_dataset.pkl",
    threads: 4


suffix_dict = {p: "linked" if p != burden_phenotype else "finished" for p in phenotypes}


rule combine_test_burdens:
    input:
        burdens=lambda wildcards: [
            (
                f"cv_split{cv_split}/deeprvat/{wildcards.phenotype}/deeprvat/burdens/chunk{c}.{suffix_dict[wildcards.phenotype]}"
            )
            for c in range(n_burden_chunks)
            for cv_split in range(cv_splits)
        ],
        data_config="deeprvat_config.yaml",
    output:
        "{phenotype}/deeprvat/burdens/merging.finished",
    params:
        out_dir="{phenotype}/deeprvat/burdens",
        burden_paths=lambda wildcards, input: "".join(
            [
                f"--burden-dirs cv_split{fold}/deeprvat/{wildcards.phenotype}/deeprvat/burdens "
                for fold in range(cv_splits)
            ]
        ),
        link=lambda wildcards: (
            f"--link-burdens ../../../{burden_phenotype}/deeprvat/burdens/burdens.zarr"
            if wildcards.phenotype != burden_phenotype
            else " "
        ),
    resources:
        mem_mb=lambda wildcards, attempt: 32000 + attempt * 4098 * 2,
    shell:
        " && ".join(
        [
            conda_check,
        "deeprvat_cv_utils combine-test-set-burdens "
                "{params.link} "
                "{params.burden_paths} "
                "{params.out_dir} "
                "{input.data_config}",
                "touch {output}",
            ]
        )


use rule compute_burdens from deeprvat_workflow as deeprvat_compute_burdens with:
    params:
        prefix="cv_split{cv_split}/deeprvat",


use rule reverse_models from deeprvat_workflow as deeprvat_reverse_models

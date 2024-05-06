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
        config_train="cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/hpopt_config.yaml",
    output:
        config_test="cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/hpopt_config_test.yaml",
    shell:
        " && ".join(
        [
            conda_check,
        "deeprvat_cv_utils generate-test-config "
                "--fold {wildcards.cv_split} "
                f"--n-folds {cv_splits}"
                " {input.config_train} {output.config_test}",
            ]
        )


# generate the association data set from the test samples (as defined in the config)
# pass the sample file here
# then just use this data set nomrally for burden computation
use rule association_dataset from deeprvat_associate as deeprvat_association_dataset with:
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
        config="config.yaml",
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
                "{input.config}",
                "touch {output}",
            ]
        )


use rule link_burdens from deeprvat_associate as deeprvat_link_burdens with:
    input:
        checkpoints = expand(
            'cv_split{cv_split}/deeprvat' / model_path / "repeat_{repeat}/best/bag_{bag}.ckpt",
            cv_split=range(cv_splits), repeat=range(n_repeats), bag=range(n_bags)
        ),
        dataset = 'cv_split0/deeprvat/{phenotype}/deeprvat/association_dataset.pkl',
        data_config = 'cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/hpopt_config_test.yaml',
        model_config = "cv_split{cv_split}/deeprvat" / model_path / 'config.yaml',
    params:
        prefix="cv_split{cv_split}/deeprvat",


use rule compute_burdens from deeprvat_associate as deeprvat_compute_burdens with:
    input:
        reversed = "cv_split{cv_split}/deeprvat" / model_path / "reverse_finished.tmp",
        checkpoints = expand(
            'cv_split{cv_split}/deeprvat' / model_path / "repeat_{repeat}/best/bag_{bag}.ckpt",
            cv_split=range(cv_splits), repeat=range(n_repeats), bag=range(n_bags)
        ),
        dataset = 'cv_split0/deeprvat/{phenotype}/deeprvat/association_dataset.pkl',
        data_config = 'cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/hpopt_config_test.yaml',
        model_config = "cv_split{cv_split}/deeprvat" / model_path / 'config.yaml',
    params:
        prefix="cv_split{cv_split}/deeprvat",


use rule reverse_models from deeprvat_associate as deeprvat_reverse_models

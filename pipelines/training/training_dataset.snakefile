rule training_dataset:
    input:
        data_config="{phenotype}/deeprvat/config.yaml",
        training_dataset="{phenotype}/deeprvat/training_dataset.pkl",
    output:
        input_tensor=directory("{phenotype}/deeprvat/input_tensor.zarr"),
        covariates=directory("{phenotype}/deeprvat/covariates.zarr"),
        y=directory("{phenotype}/deeprvat/y.zarr"),
    threads: 8
    resources:
        mem_mb=lambda wildcards, attempt: 32000 + 12000 * attempt,
        load=16000,
    priority: 5000
    log:
        stdout="logs/training_dataset/{phenotype}.stdout", 
        stderr="logs/training_dataset/{phenotype}stderr"
    shell:
        (
            "deeprvat_train make-dataset "
            + debug
            + "--compression-level "
            + str(tensor_compression_level)
            + " "
            "--training-dataset-file {input.training_dataset} "
            "{input.data_config} "
            "{output.input_tensor} "
            "{output.covariates} "
            "{output.y} "
            + logging_redirct
        )


rule training_dataset_pickle:
    input:
        "{phenotype}/deeprvat/config.yaml",
    output:
        "{phenotype}/deeprvat/training_dataset.pkl",
    threads: 1
    resources:
        mem_mb=40000,  # lambda wildcards, attempt: 38000 + 12000 * attempt
        load=16000,
    log:
        stdout="logs/training_dataset_pickle/{phenotype}.stdout", 
        stderr="logs/training_dataset_pickle/{phenotype}stderr"
    shell:
        (
            "deeprvat_train make-dataset "
            "--pickle-only "
            "--training-dataset-file {output} "
            "{input} "
            "dummy dummy dummy "
            + logging_redirct
        )
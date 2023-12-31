
rule training_dataset:
    input:
        config = '{phenotype}/deeprvat/hpopt_config.yaml',
        training_dataset = '{phenotype}/deeprvat/training_dataset.pkl'
    output:
        input_tensor = directory('{phenotype}/deeprvat/input_tensor.zarr'),
        covariates = directory('{phenotype}/deeprvat/covariates.zarr'),
        y = directory('{phenotype}/deeprvat/y.zarr')
    threads: 8
    priority: 50
    shell:
        (
            'deeprvat_train make-dataset '
            + debug +
            '--compression-level ' + str(tensor_compression_level) + ' '
            '--training-dataset-file {input.training_dataset} '
            '{input.config} '
            '{output.input_tensor} '
            '{output.covariates} '
            '{output.y}'
        )

rule training_dataset_pickle:
    input:
        '{phenotype}/deeprvat/hpopt_config.yaml'
    output:
        '{phenotype}/deeprvat/training_dataset.pkl'
    threads: 1
    shell:
        (
            'deeprvat_train make-dataset '
            '--pickle-only '
            '--training-dataset-file {output} '
            '{input} '
            'dummy dummy dummy'
        )
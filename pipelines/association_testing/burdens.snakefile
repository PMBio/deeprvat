
rule link_burdens:
    priority: 1
    input:
        checkpoints = lambda wildcards: [
            f'{pretrained_model_path}/repeat_{repeat}/best/bag_{bag}.ckpt'
            for repeat in range(n_repeats) for bag in range(n_bags)
        ],
        dataset = '{phenotype}/deeprvat/association_dataset.pkl',
        data_config = '{phenotype}/deeprvat/hpopt_config.yaml',
        model_config = pretrained_model_path / 'config.yaml',
    output:
        '{phenotype}/deeprvat/burdens/chunk{chunk}.linked'
    threads: 8
    shell:
        ' && '.join([
            ('deeprvat_associate compute-burdens '
             + debug +
             ' --n-chunks '+ str(n_burden_chunks) + ' '
             f'--link-burdens ../../../{phenotypes[0]}/deeprvat/burdens/burdens.zarr '
             '--chunk {wildcards.chunk} '
             '--dataset-file {input.dataset} '
             '{input.data_config} '
             '{input.model_config} '
             '{input.checkpoints} '
             '{wildcards.phenotype}/deeprvat/burdens'),
            'touch {output}'
        ])

rule compute_burdens:
    priority: 10
    input:
        reversed = pretrained_model_path / "reverse_finished.tmp",
        checkpoints = lambda wildcards: [
            pretrained_model_path / f'repeat_{repeat}/best/bag_{bag}.ckpt'
            for repeat in range(n_repeats) for bag in range(n_bags)
        ],
        dataset = '{phenotype}/deeprvat/association_dataset.pkl',
        data_config = '{phenotype}/deeprvat/hpopt_config.yaml',
        model_config = pretrained_model_path / 'config.yaml',
    output:
        '{phenotype}/deeprvat/burdens/chunk{chunk}.finished'
    threads: 8
    shell:
        ' && '.join([
            ('deeprvat_associate compute-burdens '
             + debug +
             ' --n-chunks '+ str(n_burden_chunks) + ' '
             '--chunk {wildcards.chunk} '
             '--dataset-file {input.dataset} '
             '{input.data_config} '
             '{input.model_config} '
             '{input.checkpoints} '
             '{wildcards.phenotype}/deeprvat/burdens'),
            'touch {output}'
        ])

rule reverse_models:
    input:
        checkpoints = expand(pretrained_model_path / 'repeat_{repeat}/best/bag_{bag}.ckpt',
                             bag=range(n_bags), repeat=range(n_repeats)),
        model_config = pretrained_model_path / 'config.yaml',
        data_config = Path(phenotypes[0]) / "deeprvat/hpopt_config.yaml",
    output:
        temp(pretrained_model_path / "reverse_finished.tmp")
    threads: 4
    shell:
        " && ".join([
            ("deeprvat_associate reverse-models "
             "{input.model_config} "
             "{input.data_config} "
             "{input.checkpoints}"),
            "touch {output}"
        ])
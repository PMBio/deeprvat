rule average_burdens:
    input:
        burdens='{phenotype}/deeprvat/burdens/burdens.zarr',
        x='{phenotype}/deeprvat/burdens/x.zarr',
        y='{phenotype}/deeprvat/burdens/y.zarr',
        sample_ids='{phenotype}/deeprvat/burdens/sample_ids.zarr',
    output:
        '{phenotype}/deeprvat/burdens/logs/burdens_averaging_{chunk}.finished',
    params:
        burdens_in='{phenotype}/deeprvat/burdens/burdens.zarr',
        burdens_out='{phenotype}/deeprvat/burdens/burdens_average.zarr',
        repeats=lambda wildcards: ''.join([f'--repeats {r} ' for r in range(int(n_repeats))])
    threads: 1
    resources:
        mem_mb=lambda wildcards, attempt: 4098 + (attempt - 1) * 4098,
    priority: 10,
    shell:
        ' && '.join([
            ('deeprvat_associate  average-burdens '
             '--n-chunks ' + str(n_avg_chunks) + ' '
                                                 '--chunk {wildcards.chunk} '
                                                 '{params.repeats} '
                                                 '--agg-fct mean  '  #TODO remove this
                                                 '{params.burdens_in} '
                                                 '{params.burdens_out}'),
            'touch {output}'
        ])

rule compute_burdens:
    priority: 10
    input:
        reversed=model_path / "reverse_finished.tmp",
        checkpoints=lambda wildcards: [
            f'{model_path}/repeat_{repeat}/best/bag_{bag}.ckpt'
            for repeat in range(n_repeats) for bag in range(n_bags)
        ],
        dataset='{phenotype}/deeprvat/association_dataset.pkl',
        data_config='{phenotype}/deeprvat/hpopt_config.yaml',
        model_config=model_path / 'config.yaml',
    output:
        burdens='{phenotype}/deeprvat/burdens/chunks/chunk_{chunk}/burdens.zarr',
        x='{phenotype}/deeprvat/burdens/chunks/chunk_{chunk}/x.zarr',
        y='{phenotype}/deeprvat/burdens/chunks/chunk_{chunk}/y.zarr',
        sample_ids='{phenotype}/deeprvat/burdens/chunks/chunk_{chunk}/sample_ids.zarr',
    params:
        prefix='.'
    threads: 8
    resources:
        mem_mb=20000,
        gpus=1
    shell:
        ' '.join([
            'deeprvat_associate compute-burdens '
            + debug +
            ' --n-chunks ' + str(n_burden_chunks) + ' '
                                                    '--chunk {wildcards.chunk} '
                                                    '--dataset-file {input.dataset} '
                                                    '{input.data_config} '
                                                    '{input.model_config} '
                                                    '{input.checkpoints} '
                                                    '{params.prefix}/{wildcards.phenotype}/deeprvat/burdens'],
        )

rule combine_burdens:
    input:
        expand(
            '{phenotype}/deeprvat/burdens/chunks/chunk{chunk}/burdens.zarr',
            chunk=[c for c in range(n_burden_chunks)],
            phenotype=phenotypes),
        expand(
            '{phenotype}/deeprvat/burdens/chunks/chunk{chunk}/x.zarr',
            chunk=[c for c in range(n_burden_chunks)],
            phenotype=phenotypes),
        expand(
            '{phenotype}/deeprvat/burdens/chunks/chunk{chunk}/y.zarr',
            chunk=[c for c in range(n_burden_chunks)],
            phenotype=phenotypes),
        expand(
            '{phenotype}/deeprvat/burdens/chunks/chunk{chunk}/sample_ids.zarr',
            chunk=[c for c in range(n_burden_chunks)],
            phenotype=phenotypes)
    output:
        burdens='{phenotype}/deeprvat/burdens/burdens.zarr',
        x='{phenotype}/deeprvat/burdens/x.zarr',
        y='{phenotype}/deeprvat/burdens/y.zarr',
        sample_ids='{phenotype}/deeprvat/burdens/sample_ids.zarr',
    params:
        prefix='.'
    shell:
        ' '.join([
            "'{wildcards.phenotype}/deeprvat/burdens/chunks/",
            'deeprvat_associate combine-burden-chunks',
            ' --n-chunks ' + str(n_burden_chunks),
            '{params.prefix}/{wildcards.phenotype}/deeprvat/burdens',
        ])

rule reverse_models:
    input:
        checkpoints=expand(model_path / 'repeat_{repeat}/best/bag_{bag}.ckpt',
            bag=range(n_bags),repeat=range(n_repeats)),
        model_config=model_path / 'config.yaml',
        data_config=Path(phenotypes[0]) / "deeprvat/hpopt_config.yaml",
    output:
        model_path / "reverse_finished.tmp"
    threads: 4
    resources:
        mem_mb=20480,
    shell:
        " && ".join([
            ("deeprvat_associate reverse-models "
             "{input.model_config} "
             "{input.data_config} "
             "{input.checkpoints}"),
            "touch {output}"
        ])

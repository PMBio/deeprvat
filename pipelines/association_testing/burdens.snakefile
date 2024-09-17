rule average_burdens:
    input:
        'burdens/burdens.zarr'
        if not cv_exp
        else f'burdens/log/{phenotypes[0]}/merging.finished',
    output:
        'burdens/logs/burdens_averaging_{chunk}.finished',
    params:
        burdens_in = 'burdens/burdens.zarr',
        burdens_out = 'burdens/burdens_average.zarr',
        repeats = lambda wildcards: ''.join([f'--repeats {r} ' for r in range(int(n_repeats))])
    threads: 1
    resources:
        mem_mb = lambda wildcards, attempt: 4098 + (attempt - 1) * 4098,
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

 rule combine_burdens:
    input:
        expand(
            'burdens/chunks/chunk_{chunk}/burdens.zarr',
            chunk=[c for c in range(n_burden_chunks)],
          ),
        expand(
            'burdens/chunks/chunk_{chunk}/sample_ids.zarr',
            chunk=[c for c in range(n_burden_chunks)],
          )
    output:
        burdens=directory('burdens/burdens.zarr'),
        sample_ids=directory('burdens/sample_ids.zarr'),
    params:
        prefix='.'
    threads: 1
    resources:
        mem_mb = lambda wildcards, attempt: 4098 + (attempt - 1) * 4098,
    shell:
        ' '.join([
            'deeprvat_associate combine-burden-chunks',
            '{params.prefix}/burdens/chunks/',
            ' --n-chunks ' + str(n_burden_chunks),
            '{params.prefix}/burdens',
        ])
        
rule all_xy:
    input:
        samples = expand('{phenotype}/deeprvat/xy/sample_ids.zarr', phenotype=phenotypes),
        x = expand('{phenotype}/deeprvat/xy/x.zarr', phenotype=phenotypes),
        y = expand('{phenotype}/deeprvat/xy/y.zarr', phenotype=phenotypes)

rule compute_xy:
    priority: 1
    input:
        dataset = '{phenotype}/deeprvat/association_dataset.pkl',
        data_config = '{phenotype}/deeprvat/config.yaml',
    output:
        samples = directory('{phenotype}/deeprvat/xy/sample_ids.zarr'),
        x = directory('{phenotype}/deeprvat/xy/x.zarr'),
        y = directory('{phenotype}/deeprvat/xy/y.zarr'),
    threads: 8
    resources:
        mem_mb = lambda wildcards, attempt: 20480 + (attempt - 1) * 4098,
    shell:
        ' && '.join([
            ('deeprvat_associate compute-xy '
             '--dataset-file {input.dataset} '
             '{input.data_config} '
             "{output.samples} "
             "{output.x} "
             "{output.y}")
        ])


rule compute_burdens:
    priority: 10
    input:
        reversed = model_path / "reverse_finished.tmp",
        checkpoints = lambda wildcards: [
            f'{model_path}/repeat_{repeat}/best/bag_{bag}.ckpt'
            for repeat in range(n_repeats) for bag in range(n_bags)
        ],
        dataset = 'burdens/association_dataset.pkl',
        data_config = f'{phenotypes[0]}/deeprvat/config.yaml',
        model_config = model_path / 'model_config.yaml',
    output:
        burdens=directory('burdens/chunks/chunk_{chunk}/burdens.zarr'),
        sample_ids=directory('burdens/chunks/chunk_{chunk}/sample_ids.zarr'),
    params:
        prefix = '.'
    threads: 8
    resources:
        mem_mb = 32000,
        gpus = 1
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
            '{params.prefix}/burdens'],
        )


rule reverse_models:
    input:
        checkpoints = expand(model_path / 'repeat_{repeat}/best/bag_{bag}.ckpt',
                             bag=range(n_bags), repeat=range(n_repeats)),
        model_config = model_path / 'model_config.yaml',
        data_config = Path(phenotypes[0]) / "deeprvat/config.yaml",
    output:
        model_path / "reverse_finished.tmp"
    threads: 4
    resources:
        mem_mb = 20480,
    shell:
        " && ".join([
            ("deeprvat_associate reverse-models "
             "{input.model_config} "
             "{input.data_config} "
             "{input.checkpoints}"),
             "touch {output}"
        ])

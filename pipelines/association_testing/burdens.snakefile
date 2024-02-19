cv_exp = True #TODO try to remove/resolve this
n_avg_chunks = 20 # somehow it doesn't get this from the main snakefile

rule average_burdens:
    input:
        chunks = [
            (f'{p}/deeprvat/burdens/chunk{c}.' +
             ("finished" if p == phenotypes[0] else "linked"))
            for p in phenotypes
            for c in range(n_burden_chunks)
        ] if not cv_exp else '{phenotype}/deeprvat/burdens/merging.finished'
    output:
        '{phenotype}/deeprvat/burdens/logs/burdens_averaging_{chunk}.finished',
    params:
        burdens_in = '{phenotype}/deeprvat/burdens/burdens.zarr',
        burdens_out = '{phenotype}/deeprvat/burdens/burdens_average.zarr',
        repeats = lambda wildcards: ''.join([f'--repeats {r} ' for r in  range(int(n_repeats))])
    threads: 1
    resources:
        mem_mb = lambda wildcards, attempt: 4098 + (attempt - 1) * 4098,
        load = 4000,
    priority: 10,
    shell:
        ' && '.join([
            ('deeprvat_associate  average-burdens '
            '--n-chunks '+ str(n_avg_chunks) + ' '
            '--chunk {wildcards.chunk} '
            '{params.repeats} '
            '--agg-fct mean  ' #TODO remove this
            '{params.burdens_in} '
            '{params.burdens_out}'),
            'touch {output}'
        ])

rule link_burdens:
    priority: 1
    input:
        checkpoints = lambda wildcards: [
            f'{model_path}/repeat_{repeat}/best/bag_{bag}.ckpt'
            for repeat in range(n_repeats) for bag in range(n_bags)
        ],
        dataset = '{phenotype}/deeprvat/association_dataset.pkl',
        data_config = '{phenotype}/deeprvat/hpopt_config.yaml',
        model_config = model_path / 'config.yaml',
    output:
        '{phenotype}/deeprvat/burdens/chunk{chunk}.linked'
    params:
        prefix = '.'
    threads: 8
    resources:
        mem_mb = lambda wildcards, attempt: 20480 + (attempt - 1) * 4098,
        # mem_mb = 16000,
        load = lambda wildcards, attempt: 16000 + (attempt - 1) * 4000
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
             '{params.prefix}/{wildcards.phenotype}/deeprvat/burdens'),
            'touch {output}'
        ])

rule compute_burdens:
    priority: 10
    input:
        reversed = "models/reverse_finished.tmp",
        checkpoints = lambda wildcards: [
            f'{model_path}/repeat_{repeat}/best/bag_{bag}.ckpt'
            for repeat in range(n_repeats) for bag in range(n_bags)
        ],
        dataset = '{phenotype}/deeprvat/association_dataset.pkl',
        data_config = '{phenotype}/deeprvat/hpopt_config.yaml',
        model_config = model_path / 'config.yaml',
    output:
        '{phenotype}/deeprvat/burdens/chunk{chunk}.finished'
    params:
        prefix = '.'
    threads: 8
    resources:
        mem_mb = 2000000,        # Using this value will tell our modified lsf.profile not to set a memory resource
        load = 8000,
        gpus = 1
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
             '{params.prefix}/{wildcards.phenotype}/deeprvat/burdens'),
            'touch {output}'
        ])

rule reverse_models:
    input:
        checkpoints = expand(model_path / 'repeat_{repeat}/best/bag_{bag}.ckpt',
                             bag=range(n_bags), repeat=range(n_repeats)),
        model_config = model_path / 'config.yaml',
        data_config = Path(phenotypes[0]) / "deeprvat/hpopt_config.yaml",
    output:
        model_path / "reverse_finished.tmp"
    threads: 4
    resources:
        mem_mb = 20480,
        load = 20480
    shell:
        " && ".join([
            ("deeprvat_associate reverse-models "
             "{input.model_config} "
             "{input.data_config} "
             "{input.checkpoints}"),
            "touch {output}"
        ])

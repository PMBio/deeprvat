configfile: 'config.yaml'

phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes

debug_flag = config.get('debug', False)
debug = '--debug ' if debug_flag else ''

n_burden_chunks = config.get('n_burden_chunks', 1) if not debug_flag else 2
n_avg_chunks = config.get('n_avg_chunks', 40)
n_bags = config['training']['n_bags'] if not debug_flag else 3
n_repeats = config['n_repeats']
model_path = Path(config.get("pretrained_model_path", "pretrained_models"))

if not "cv_exp" in globals():
    cv_exp = config.get("cv_exp", False)

config_file_prefix = (
    "cv_split0/deeprvat/" if cv_exp else ""
)



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
    priority: 10,
    shell:
        ' && '.join([
            ('deeprvat_associate  average-burdens '
            '--n-chunks '+ str(n_avg_chunks) + ' '
            '--chunk {wildcards.chunk} '
            '{params.repeats} '
            '--agg-fct mean  ' 
            '{params.burdens_in} '
            '{params.burdens_out}'),
            'touch {output}'
        ])

rule all_xy:
    input:
        expand("{phenotype}/deeprvat/xy/chunk{chunk}.finished",
               phenotype=phenotypes,
               chunk=range(n_burden_chunks))

rule compute_xy:
    priority: 1
    input:
        dataset = '{phenotype}/deeprvat/association_dataset.pkl',
        data_config = '{phenotype}/deeprvat/config.yaml',
        model_config = model_path / 'model_config.yaml',
    output:
        samples = '{phenotype}/deeprvat/xy/sample_ids.zarr',
        x = '{phenotype}/deeprvat/xy/x.zarr',
        y = '{phenotype}/deeprvat/xy/y.zarr',
    params:
        prefix = '.'
    threads: 8
    resources:
        mem_mb = lambda wildcards, attempt: 20480 + (attempt - 1) * 4098,
    shell:
        ' && '.join([
            ('deeprvat_associate compute-xy'
             + debug +
             ' --n-chunks '+ str(n_burden_chunks) + ' '
             f'--link-burdens ../../../{phenotypes[0]}/deeprvat/burdens/burdens.zarr '
             '--chunk {wildcards.chunk} '
             '--dataset-file {input.dataset} '
             '{input.data_config} '
             "{input.samples} "
             "{input.x} "
             "{input.y}"),
            'touch {output}'
        ])

rule compute_burdens:
    priority: 10
    input:
        reversed = model_path / "reverse_finished.tmp",
        checkpoints = lambda wildcards: [
            f'{model_path}/repeat_{repeat}/best/bag_{bag}.ckpt'
            for repeat in range(n_repeats) for bag in range(n_bags)
        ],
        dataset = '{phenotype}/deeprvat/association_dataset.pkl',
        data_config = '{phenotype}/deeprvat/config.yaml',
        model_config = model_path / 'model_config.yaml',
    output:
        '{phenotype}/deeprvat/burdens/chunk{chunk}.finished'
    params:
        prefix = '.'
    threads: 8
    resources:
        mem_mb = 20000,
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

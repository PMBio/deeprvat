from pathlib import Path
from typing import List

configfile: 'config.yaml'

debug_flag = config.get('debug', False)

phenotypes = list(config['phenotypes'].keys())
n_burden_chunks = config.get('n_burden_chunks', 1) if not debug_flag else 2
n_regression_chunks = config.get('n_regression_chunks', 40) if not debug_flag else 2
n_trials = config['hyperparameter_optimization']['n_trials']
n_repeats = config['n_repeats']
debug = '--debug ' if debug_flag else ''
do_scoretest = '--do-scoretest ' if config.get('do_scoretest', False) else ''
tensor_compression_level = config['training'].get('tensor_compression_level', 1)

wildcard_constraints:
    repeat="\d+"

rule all:
    input:
        "results.parquet",
        "pvals.parquet"

rule evaluate:
    input:
        testing_associations = expand('{{phenotype}}/deeprvat/repeat_{repeat}/results/burden_associations_testing.parquet',
                                    repeat=range(n_repeats)),
        replication_associations = expand('{{phenotype}}/deeprvat/repeat_{repeat}/results/burden_associations_replication.parquet',
                                  repeat=range(n_repeats)),
        config = '{phenotype}/deeprvat/hpopt_config.yaml',
        baseline = lambda wildcards: [
            str(Path(r['base']) / wildcards.phenotype / r['type'] /
                'eval/burden_associations_testing.parquet')
            for r in config['baseline_results']
        ]
    output:
        "results.parquet",
        "pvals.parquet"
    threads: 1
    shell:
        py + 'deeprvat_evaluate '
        + debug +
        '--correction-method FDR '
        '{input.associations} '
        '{input.config} '
        '{wildcards.phenotype}/deeprvat/eval'

rule all_regression:
    input:
        expand('{phenotype}/deeprvat/repeat_{repeat}/results/burden_associations.parquet',
               phenotype=phenotypes, type=['deeprvat'], repeat=range(n_repeats)),

rule combine_regression_chunks:
    input:
        expand('{{phenotype}}/{{type}}/repeat_{{repeat}}/results/burden_associations_{chunk}.parquet', chunk=range(n_regression_chunks)),
    output:
        '{phenotype}/{type}/repeat_{repeat}/results/burden_associations.parquet',
    threads: 1
    shell:
        'deeprvat_associate combine-regression-results '
        '--model-name repeat_{wildcards.repeat} '
        '{input.testing} '
        '{output.testing}',
        ' && '.join([
            'univariate_burden_regression.py combine-regression-results '
            '--model-name repeat_{wildcards.repeat} '
            '{input} '
            '{output}'
        ])

rule regress:
    input:
        config = "{phenotype}/deeprvat/hpopt_config.yaml",
        chunks = lambda wildcards: expand(
            ('{{phenotype}}/{{type}}/burdens/chunk{chunk}.' +
             ("finished" if wildcards.phenotype == phenotypes[0] else "linked")),
            chunk=range(n_burden_chunks)
        ),
        phenotype_0_chunks =  expand(
            phenotypes[0] + '/{{type}}/burdens/chunk{chunk}.finished',
            chunk=range(n_burden_chunks)
        ),
    output:
        temp('{phenotype}/{type}/repeat_{repeat}/results/burden_associations_{chunk}.parquet'),
    threads: 2
    shell:
        'deeprvat_associate regress '
        + debug +
        '--chunk {wildcards.chunk} '
        '--n-chunks ' + str(n_regression_chunks) + ' '
        '--use-bias '
        '--repeat {wildcards.repeat} '
        + do_scoretest +
        '{input.config} '
        '{wildcards.phenotype}/{wildcards.type}/burdens ' #TODO make this w/o repeats
        '{wildcards.phenotype}/{wildcards.type}/repeat_{wildcards.repeat}/results'

rule all_burdens:
    input:
        [
            (f'{p}/deeprvat/burdens/chunk{c}.' +
             ("finished" if p == phenotypes[0] else "linked"))
            for p in phenotypes
            for c in range(n_burden_chunks)
        ]

rule link_burdens:
    priority: 1
    input:
        checkpoints = lambda wildcards: [
            f'models/repeat_{repeat}/best/bag_{bag}.ckpt'
            for repeat in range(n_repeats) for bag in range(n_bags)
        ],
        dataset = '{phenotype}/deeprvat/association_dataset.pkl',
        config = 'models/repeat_0/config.yaml' #TODO make this more generic
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
             '{input.config} '
             '{input.checkpoints} '
             '{wildcards.phenotype}/deeprvat/burdens'),
            'touch {output}'
        ])

rule compute_burdens:
    priority: 10
    input:
        checkpoints = lambda wildcards: [
            f'models/repeat_{repeat}/best/bag_{bag}.ckpt'
            for repeat in range(n_repeats) for bag in range(n_bags)
        ],
        dataset = '{phenotype}/deeprvat/association_dataset.pkl',
        config = 'models/repeat_0/config.yaml' #TODO make this more generic
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
             '{input.config} '
             '{input.checkpoints} '
             '{wildcards.phenotype}/deeprvat/burdens'),
            'touch {output}'
        ])

rule all_association_dataset:
    input:
        expand('{phenotype}/deeprvat/association_dataset.pkl',
               phenotype=phenotypes)

rule association_dataset:
    input:
        config = '{phenotype}/deeprvat/hpopt_config.yaml'
    output:
        '{phenotype}/deeprvat/association_dataset.pkl'
    threads: 4
    shell:
        'deeprvat_associate make-dataset '
        + debug +
        '{input.config} '
        '{output}'

rule reverse_models:
    input:
        checkpoints = lambda wildcards: [
            f'{training_phenotypes[wildcards.phenotype]}/deeprvat/repeat_{repeat}/models/best/bag_{bag}.ckpt'
            for repeat in range(n_repeats) for bag in range(n_bags)
        ],
        config = '{phenotype}/deeprvat/repeat_0/config.yaml',
    output:
        "{phenotype}/deeprvat/reverse_finished.tmp"
    threads: 4
    shell:
        " && ".join([
            ("deeprvat_associate reverse-models "
             "{input.config} "
             "{input.checkpoints}"),
            "touch {output}"
        ])

rule all_config:
    input:
        config = expand('{phenotype}/deeprvat/hpopt_config.yaml',
                        phenotype=phenotypes),

rule config:
    input:
        config = 'config.yaml',
    output:
        config = '{phenotype}/deeprvat/hpopt_config.yaml',
    threads: 1
    shell:
        (
            'deeprvat_config update-config '
            '--phenotype {wildcards.phenotype} '
            '{input.config} '
            '{output.config}'
        )

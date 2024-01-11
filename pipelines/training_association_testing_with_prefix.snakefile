from pathlib import Path
from typing import List

configfile: 'config.yaml'

debug_flag = config.get('debug', False)
phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes
training_phenotypes = config["training"].get("phenotypes", phenotypes)

n_burden_chunks = config.get('n_burden_chunks', 1) if not debug_flag else 2
n_regression_chunks = config.get('n_regression_chunks', 40) if not debug_flag else 2
n_trials = config['hyperparameter_optimization']['n_trials']
n_bags = config['training']['n_bags'] if not debug_flag else 3
n_repeats = config['n_repeats']
debug = '--debug ' if debug_flag else ''
do_scoretest = '--do-scoretest ' if config.get('do_scoretest', False) else ''
tensor_compression_level = config['training'].get('tensor_compression_level', 1)
n_parallel_training_jobs = config["training"].get("n_parallel_jobs", 1)


DEEPRVAT_ANALYSIS_DIR=os.environ['DEEPRVAT_ANALYSIS_DIR']
py_deeprvat_analysis= f'python {DEEPRVAT_ANALYSIS_DIR}'

wildcard_constraints:
    repeat="\d+",
    trial="\d+",

rule all:
    input:
        significant = expand("{phenotype}/deeprvat/eval/significant.parquet",
               phenotype=phenotypes),
        results = expand("{phenotype}/deeprvat/eval/all_results.parquet",
               phenotype=phenotypes),
        replication = "replication.parquet",
        plots = "dicovery_replication_plot.png"

rule plot:
    conda:
        "r-env"
    input:
        significant = expand("{phenotype}/deeprvat/eval/significant.parquet",
               phenotype=phenotypes),
        results = expand("{phenotype}/deeprvat/eval/all_results.parquet",
               phenotype=phenotypes),
        replication = "replication.parquet"
    output:
        "dicovery_replication_plot.png"
    params:
        results_dir = './',
        results_dir_pattern = '',
        code_dir = f'{DEEPRVAT_ANALYSIS_DIR}/association_testing'
    resources:
        mem_mb=20480,
        load=16000,
    script:
        f'{DEEPRVAT_ANALYSIS_DIR}/association_testing/figure_3_main.R'

#requires that comparison_results.pkl is linked to the experiment directory
rule compute_replication:
    input:
        results = expand("{phenotype}/deeprvat/eval/all_results.parquet",
            phenotype = training_phenotypes)
    output:
        'replication.parquet'
    params:
        result_files = lambda wildcards, input: ''.join([
            f'--result-files {f} '
            for f in input.results
        ]),
        n_repeats = f'{n_repeats}'
    resources:
        mem_mb = lambda wildcards, attempt: 32000 + attempt * 4098 * 2,
    shell:
        py_deeprvat_analysis + '/association_testing/compute_replication.py '
        '--out-file {output} '
        '--n-repeats {params.n_repeats} '
        '{params.result_files} '
        './ '

rule evaluate:
    input:
        associations = expand('{{phenotype}}/deeprvat/repeat_{repeat}/results/burden_associations.parquet',
                              repeat=range(n_repeats)),
        config = '{phenotype}/deeprvat/hpopt_config.yaml',
    output:
        f"{{phenotype}}/deeprvat/eval/significant_{n_repeats}repeats.parquet",
        f"{{phenotype}}/deeprvat/eval/all_results_{n_repeats}repeats.parquet"
    threads: 1
    params:
        out_path = '{phenotype}/deeprvat/eval',
        use_seed_genes = '--use-seed-genes', 
        n_repeats = f'{n_repeats}',
        repeats_to_analyze = f'{n_repeats}',
        max_repeat_combis = 1
    resources:
        mem_mb = lambda wildcards, attempt: 25000 + attempt * 4098,
    shell:
        'deeprvat_evaluate '
        + debug +
        '{params.use_seed_genes} '
        '--n-repeats {params.n_repeats} '#total number of repeats
        '--repeats-to-analyze {params.repeats_to_analyze} '#number of repeats to analyze
        '--max-repeat-combis {params.max_repeat_combis} '
        '--correction-method FDR '
        '{input.associations} '
        '{input.config} '
        '{params.out_path}'

rule all_regression:
    input:
        expand('{phenotype}/deeprvat/repeat_{repeat}/results/burden_associations.parquet',
               phenotype=phenotypes, type=['deeprvat'], repeat=range(n_repeats)),

rule combine_regression_chunks:
    input:
        expand('{{phenotype}}/deeprvat/repeat_{{repeat}}/results/burden_associations_{chunk}.parquet', chunk=range(n_regression_chunks)),
    output:
        '{phenotype}/deeprvat/repeat_{repeat}/results/burden_associations.parquet',
    threads: 1
    resources:
        mem_mb = 2048,
        load = 2000
    shell:
        'deeprvat_associate combine-regression-results '
        '--model-name repeat_{wildcards.repeat} '
        '{input} '
        '{output}'

rule regress:
    input:
        config = "{phenotype}/deeprvat/hpopt_config.yaml",
        chunks = lambda wildcards: (
            [] if wildcards.phenotype == phenotypes[0]
            else expand('{{phenotype}}/deeprvat/burdens/chunk{chunk}.linked',
                        chunk=range(n_burden_chunks))
        ),
        phenotype_0_chunks =  expand(
            phenotypes[0] + '/deeprvat/burdens/chunk{chunk}.finished',
            chunk=range(n_burden_chunks)
        ),
    params:
        prefix = '.'
    output:
        temp('{phenotype}/deeprvat/repeat_{repeat}/results/burden_associations_{chunk}.parquet'),
    threads: 2
    resources:
        mem_mb = lambda wildcards, attempt: 28676 + (attempt - 1) * 4098,
        # mem_mb = 16000,
        load = lambda wildcards, attempt: 28000 + (attempt - 1) * 4000
    shell:
        'deeprvat_associate regress '
        + debug +
        '--chunk {wildcards.chunk} '
        '--n-chunks ' + str(n_regression_chunks) + ' '
        '--use-bias '
        '--repeat {wildcards.repeat} '
        + do_scoretest +
        '{input.config} '
        '{params.prefix}/{wildcards.phenotype}/deeprvat/burdens ' #TODO make this w/o repeats
        '{params.prefix}/{wildcards.phenotype}/deeprvat/repeat_{wildcards.repeat}/results'

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
        data_config = '{phenotype}/deeprvat/hpopt_config.yaml',
        model_config = 'models/config.yaml',
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
            f'models/repeat_{repeat}/best/bag_{bag}.ckpt'
            for repeat in range(n_repeats) for bag in range(n_bags)
        ],
        dataset = '{phenotype}/deeprvat/association_dataset.pkl',
        data_config = '{phenotype}/deeprvat/hpopt_config.yaml',
        model_config = 'models/config.yaml',
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
    resources:
        mem_mb = lambda wildcards, attempt: 32000 * (attempt + 1),
        load = 64000
    priority: 30
    shell:
        'deeprvat_associate make-dataset '
        + debug +
        '{input.config} '
        '{output}'

rule reverse_models:
    input:
        checkpoints = expand('models/repeat_{repeat}/best/bag_{bag}.ckpt',
                             bag=range(n_bags), repeat=range(n_repeats)),
        model_config = 'models/config.yaml',
        data_config = Path(phenotypes[0]) / "deeprvat/hpopt_config.yaml",
    output:
        "models/reverse_finished.tmp"
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

rule all_training:
    input:
        expand('models/repeat_{repeat}/best/bag_{bag}.ckpt',
               bag=range(n_bags), repeat=range(n_repeats)),
        "models/config.yaml"

rule link_config:
    input:
        'models/repeat_0/config.yaml'
    output:
        "models/config.yaml"
    threads: 1
    shell:
        "ln -rfs {input} {output}"
        # "ln -s repeat_0/config.yaml {output}"

rule best_training_run:
    input:
        expand('models/repeat_{{repeat}}/trial{trial_number}/config.yaml',
               trial_number=range(n_trials)),
    output:
        checkpoints = expand('models/repeat_{{repeat}}/best/bag_{bag}.ckpt',
                             bag=range(n_bags)),
        config = 'models/repeat_{repeat}/config.yaml'
    params:
        prefix = '.'
    threads: 1
    resources:
        mem_mb = 2048,
        load = 2000
    shell:
        (
            'deeprvat_train best-training-run '
            + debug +
            '{params.prefix}/models/repeat_{wildcards.repeat} '
            '{params.prefix}/models/repeat_{wildcards.repeat}/best '
            '{params.prefix}/models/repeat_{wildcards.repeat}/hyperparameter_optimization.db '
            '{output.config}'
        )

rule train:
    input:
        config = expand('{phenotype}/deeprvat/hpopt_config.yaml',
                        phenotype=training_phenotypes),
        input_tensor = expand('{phenotype}/deeprvat/input_tensor.zarr',
                              phenotype=training_phenotypes),
        covariates = expand('{phenotype}/deeprvat/covariates.zarr',
                            phenotype=training_phenotypes),
        y = expand('{phenotype}/deeprvat/y.zarr',
                   phenotype=training_phenotypes),
    output:
        expand('models/repeat_{repeat}/trial{trial_number}/config.yaml',
               repeat=range(n_repeats), trial_number=range(n_trials)),
        expand('models/repeat_{repeat}/trial{trial_number}/finished.tmp',
               repeat=range(n_repeats), trial_number=range(n_trials))
    params:
        phenotypes = " ".join(
            [f"--phenotype {p} "
             f"{p}/deeprvat/input_tensor.zarr "
             f"{p}/deeprvat/covariates.zarr "
             f"{p}/deeprvat/y.zarr"
             for p in training_phenotypes]),
        prefix = '.',
    resources:
        mem_mb = 2000000,        # Using this value will tell our modified lsf.profile not to set a memory resource
        load = 8000,
        gpus = 1
    shell:
        f"parallel --jobs {n_parallel_training_jobs} --halt now,fail=1 --results train_repeat{{{{1}}}}_trial{{{{2}}}}/ "
        'deeprvat_train train '
        + debug +
        '--trial-id {{2}} '
        "{params.phenotypes} "
        'config.yaml '
        '{params.prefix}/models/repeat_{{1}}/trial{{2}} '
        "{params.prefix}/models/repeat_{{1}}/hyperparameter_optimization.db '&&' "
        "touch {params.prefix}/models/repeat_{{1}}/trial{{2}}/finished.tmp "
        "::: " + " ".join(map(str, range(n_repeats))) + " "
        "::: " + " ".join(map(str, range(n_trials)))

rule all_training_dataset:
    input:
        input_tensor = expand('{phenotype}/deeprvat/input_tensor.zarr',
                              phenotype=training_phenotypes, repeat=range(n_repeats)),
        covariates = expand('{phenotype}/deeprvat/covariates.zarr',
                            phenotype=training_phenotypes, repeat=range(n_repeats)),
        y = expand('{phenotype}/deeprvat/y.zarr',
                   phenotype=training_phenotypes, repeat=range(n_repeats))

rule training_dataset:
    input:
        config = '{phenotype}/deeprvat/hpopt_config.yaml',
        training_dataset = '{phenotype}/deeprvat/training_dataset.pkl'
    output:
        input_tensor = directory('{phenotype}/deeprvat/input_tensor.zarr'),
        covariates = directory('{phenotype}/deeprvat/covariates.zarr'),
        y = directory('{phenotype}/deeprvat/y.zarr')
    threads: 8
    resources:
        mem_mb = lambda wildcards, attempt: 32000 + 12000 * attempt,
        load = 16000
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
    resources:
        mem_mb = 40000, # lambda wildcards, attempt: 38000 + 12000 * attempt
        load = 16000
    shell:
        (
            'deeprvat_train make-dataset '
            '--pickle-only '
            '--training-dataset-file {output} '
            '{input} '
            'dummy dummy dummy'
        )

rule all_config:
    input:
        seed_genes = expand('{phenotype}/deeprvat/seed_genes.parquet',
                            phenotype=phenotypes),
        config = expand('{phenotype}/deeprvat/hpopt_config.yaml',
                        phenotype=phenotypes),
        baseline = expand('{phenotype}/deeprvat/baseline_results.parquet',
                          phenotype=phenotypes),

rule config:
    input:
        config = 'config.yaml',
        baseline = lambda wildcards: [
            str(Path(r['base']) / wildcards.phenotype / r['type'] /
                'eval/burden_associations.parquet')
            for r in config['baseline_results']
        ]
    output:
        seed_genes = '{phenotype}/deeprvat/seed_genes.parquet',
        config = '{phenotype}/deeprvat/hpopt_config.yaml',
        baseline = '{phenotype}/deeprvat/baseline_results.parquet',
    threads: 1
    resources:
        mem_mb = 1024,
        load = 1000
    params:
        baseline_results = lambda wildcards, input: ''.join([
            f'--baseline-results {b} '
            for b in input.baseline
        ])
    shell:
        (
            'deeprvat_config update-config '
            '--phenotype {wildcards.phenotype} '
            '{params.baseline_results}'
            '--baseline-results-out {output.baseline} '
            '--seed-genes-out {output.seed_genes} '
            '{input.config} '
            '{output.config}'
        )

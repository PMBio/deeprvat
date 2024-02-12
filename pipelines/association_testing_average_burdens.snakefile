from pathlib import Path

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
DEEPRVAT_DIR=os.environ['DEEPRVAT_DIR']

py_deeprvat_analysis= f'python {DEEPRVAT_ANALYSIS_DIR}'
py_deeprvat = f'python {DEEPRVAT_DIR}/deeprvat/deeprvat'

phenotypes = training_phenotypes

wildcard_constraints:
    repeat="\d+",
    trial="\d+",

n_avg_chunks = 20
repeats_to_average = config.get('n_repeats', 6)
n_combinations = 1
burden_agg_fcts = ['mean']


from itertools import combinations
import random
random.seed(10)

cv_exp = True if os.path.exists('cv_split0/') else False
config_file_prefix = 'cv_split0/deeprvat/' if cv_exp else '' #needed in case we analyse a CV experiment
print(config_file_prefix)


use_seed_opts = ['use_seed', 'wo_seed']
use_seed_opts = ['wo_seed']
use_seed_dict = {'use_seed': '--use-seed-genes ', 'wo_seed': ' '}

rule all:
    input:
        # significant = expand("{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/eval/significant.parquet",
        #        phenotype=phenotypes,
        #        n_avg_repeats = repeats_to_average,
        #        burden_agg_fct = burden_agg_fcts),
        # results = expand("{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/eval/all_results.parquet",
        #        phenotype=phenotypes,
        #        n_avg_repeats = repeats_to_average,
        #        burden_agg_fct = burden_agg_fcts),
        replication = expand("replication/{use_seed}/replication_{burden_agg_fct}_{n_avg_repeats}.parquet",
            n_avg_repeats = repeats_to_average,
               burden_agg_fct = burden_agg_fcts, use_seed = use_seed_opts),
        plots = expand(f"plots/dicovery_replication_plot_{{use_seed}}_{{burden_agg_fct}}_{n_repeats}.png",
        burden_agg_fct = burden_agg_fcts, use_seed = use_seed_opts
        )




rule plot_avg:
    conda:
        "r-env"
    input:
        significant = expand(f"{{phenotype}}/deeprvat/{{{{burden_agg_fct}}}}_agg_results/{n_repeats}_repeats/eval/{{{{use_seed}}}}/significant.parquet",
               phenotype=phenotypes),
        results = expand(f"{{phenotype}}/deeprvat/{{{{burden_agg_fct}}}}_agg_results/{n_repeats}_repeats/eval/{{{{use_seed}}}}/all_results.parquet",
               phenotype=phenotypes),
        replication = f'replication/{{use_seed}}/replication_{{burden_agg_fct}}_{n_repeats}.parquet'
    output:
        f"plots/dicovery_replication_plot_{{use_seed}}_{{burden_agg_fct}}_{n_repeats}.png"
    params:
        results_dir = './',
        results_dir_pattern = f'deeprvat/{{burden_agg_fct}}_agg_results/{n_repeats}_repeats/eval/{{use_seed}}',
        code_dir = f'{DEEPRVAT_ANALYSIS_DIR}/association_testing'
    resources:
        mem_mb=20480,
        load=16000,
    script:
        f'{DEEPRVAT_ANALYSIS_DIR}/association_testing/figure_3_main.R'

# #requires that comparison_results.pkl is linked to the experiment directory
rule compute_replication_avg:
    input:
        results = expand("{phenotype}/deeprvat/{{burden_agg_fct}}_agg_results/{{n_avg_repeats}}_repeats/eval/{{use_seed}}/all_results.parquet",
            phenotype = training_phenotypes)
    output:
        'replication/{use_seed}/replication_{burden_agg_fct}_{n_avg_repeats}.parquet'
    params:
        result_files = lambda wildcards, input: ''.join([
            f'--result-files {f} '
            for f in input.results
        ])
    resources:
        mem_mb = lambda wildcards, attempt: 8000 + (attempt - 1) * 4098
    shell:
        py_deeprvat_analysis + '/association_testing/compute_replication.py '
        '--out-file {output} '
        '--n-repeats 1 '
        '{params.result_files} '
        './ '

# # deeprvat_evaluate --n-repeats 6 --correction-method FDR LDL_direct/deeprvat/repeat_0/results/burden_associations.parquet LDL_direct/deeprvat/repeat_1/results/burden_associations.parquet  LDL_direct/deeprvat/repeat_2/results/burden_associations.parquet LDL_direct/deeprvat/repeat_3/results/burden_associations.parquet LDL_direct/deeprvat/repeat_4/results/burden_associations.parquet LDL_direct/deeprvat/repeat_5/results/burden_associations.parquet LDL_direct/deeprvat/hpopt_config.yaml ./
rule evaluate_avg:
    input:
        associations ='{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/burden_associations.parquet',
        config = f"{config_file_prefix}{{phenotype}}/deeprvat/hpopt_config.yaml"
    output:
        "{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/eval/{use_seed}/significant.parquet",
        "{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/eval/{use_seed}/all_results.parquet"
    threads: 1
    resources:
        mem_mb = 16000,
        load = 16000
    params:
        n_combis = 1,
        use_seed_genes = lambda wildcards: use_seed_dict[wildcards.use_seed]
    shell:
        'deeprvat_evaluate '
        + debug +
        '{params.use_seed_genes} '
        '--save-default '
        '--n-repeats {params.n_combis} ' #because we analyze each average combi alone, so the totatl number of combis is the total number of repeats
        '--correction-method FDR '
        '--repeats-to-analyze 1 ' #always only analyse one combination 
        '--max-repeat-combis {params.n_combis} '
        '{input.associations} '
        '{input.config} '
        '{wildcards.phenotype}/deeprvat/{wildcards.burden_agg_fct}_agg_results/{wildcards.n_avg_repeats}_repeats/eval/{wildcards.use_seed}/'



rule all_regression_avg:
    input:
        expand('{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/burden_associations.parquet',
            n_avg_repeats = repeats_to_average,
            burden_agg_fct = burden_agg_fcts,
            phenotype = phenotypes
        )

rule combine_regression_chunks_avg:
    input:
        expand('{{phenotype}}/deeprvat/{{burden_agg_fct}}_agg_results/{{n_avg_repeats}}_repeats/burden_associations_{chunk}.parquet', chunk=range(n_regression_chunks)),
    output:
        '{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/burden_associations.parquet',
    threads: 1
    resources:
        mem_mb = lambda wildcards, attempt: 12000 + (attempt - 1) * 4098,
        load = 2000
    shell:
        'deeprvat_associate combine-regression-results '
        '--model-name repeat_0 ' 
        '{input} '
        '{output}'

rule regress_avg:
    input:
        config = f"{config_file_prefix}{{phenotype}}/deeprvat/hpopt_config.yaml",
        # chunks =  '{phenotype}/deeprvat/burdens/merging.finished',
        chunks = lambda wildcards: (
            [] if wildcards.phenotype == phenotypes[0]
            else expand('{{phenotype}}/deeprvat/burdens/chunk{chunk}.linked',
                        chunk=range(n_burden_chunks))
        ) if not cv_exp  else '{phenotype}/deeprvat/burdens/merging.finished',
        phenotype_0_chunks =  expand(
            phenotypes[0] + '/deeprvat/burdens/logs/burdens_{{burden_agg_fct}}_{{n_avg_repeats}}_repeats_chunk_{chunk}.finished',
            chunk=range(n_avg_chunks)
        ),
    output:
        temp('{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/burden_associations_{chunk}.parquet'),
    threads: 2
    resources:
        mem_mb = lambda wildcards, attempt: 28676  + (attempt - 1) * 4098,
        # mem_mb = 16000,
        load = lambda wildcards, attempt: 28000 + (attempt - 1) * 4000
    params:
        burden_file = f'{phenotypes[0]}/deeprvat/burdens/burdens_{{burden_agg_fct}}_{{n_avg_repeats}}.zarr',
        burden_dir = '{phenotype}/deeprvat/burdens',
        out_dir = '{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats'
    shell:
        'deeprvat_associate regress '
        + debug +
        '--chunk {wildcards.chunk} '
        '--n-chunks ' + str(n_regression_chunks) + ' '
        '--use-bias '
        '--repeat 0 '
        '--burden-file {params.burden_file} '
        + do_scoretest +
        '{input.config} '
        '{params.burden_dir} ' #TODO make this w/o repeats
        '{params.out_dir}'


rule all_average_burdens:
    input:
        expand('{phenotype}/deeprvat/burdens/logs/burdens_{burden_agg_fct}_{n_avg_repeats}_repeats_chunk_{chunk}.finished',
            n_avg_repeats = repeats_to_average,
            chunk = range(n_avg_chunks),
            burden_agg_fct = burden_agg_fcts,
            phenotype = phenotypes[0]),


rule average_burdens:
    input:
        chunks = [
            (f'{p}/deeprvat/burdens/chunk{c}.' +
             ("finished" if p == phenotypes[0] else "linked"))
            for p in phenotypes
            for c in range(n_burden_chunks)
        ] if not cv_exp else '{phenotype}/deeprvat/burdens/merging.finished'
    output:
        '{phenotype}/deeprvat/burdens/logs/burdens_{burden_agg_fct}_{n_avg_repeats}_repeats_chunk_{chunk}.finished',
    params:
        burdens_in = '{phenotype}/deeprvat/burdens/burdens.zarr',
        burdens_out = '{phenotype}/deeprvat/burdens/burdens_{burden_agg_fct}_{n_avg_repeats}.zarr',
        repeats = ''.join([f'--repeats {r} ' for r in  range(repeats_to_average)])
    threads: 1
    resources:
        mem_mb = lambda wildcards, attempt: 4098 * 4 + (attempt - 1) * 4098,
        # mem_mb = 4098,
        load = 4000,
    priority: 10,
    shell:
        ' && '.join([
            ('deeprvat_associate  average-burdens '
            '--n-chunks '+ str(n_avg_chunks) + ' '
            '--chunk {wildcards.chunk} '
            '{params.repeats} '
            '--agg-fct {wildcards.burden_agg_fct}  '
            '{params.burdens_in} '
            '{params.burdens_out}'),
            'touch {output}'
        ])


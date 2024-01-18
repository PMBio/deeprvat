
from snakemake.utils import Paramspace
from snakemake.utils import min_version
import os
min_version("6.0")

from pathlib import Path
import pandas as pd

configfile: 'config.yaml'

debug_flag = config.get('debug', False)
# debug_flag = True #TODO change this
debug = '--debug ' if debug_flag else ''


conda_check = 'conda info | grep "active environment"'
cuda_visible_devices = 'echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES'

DEEPRVAT_DIR = os.environ['DEEPRVAT_DIR']

py_deeprvat = f'python {DEEPRVAT_DIR}/deeprvat/'

wildcard_constraints:
    repeat="\d+"

phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes
burden_phenotype = phenotypes[0]
DEEPRVAT_ANALYSIS_DIR = os.environ['DEEPRVAT_ANALYSIS_DIR']


training_phenotypes = config["training"].get("phenotypes", phenotypes)
association_testing_maf = config.get('association_testing_maf', 0.001)

n_regression_chunks = config.get('n_regression_chunks', 40) if not debug_flag else 2
n_burden_chunks = config.get('n_burden_chunks', 1) if not debug_flag else 2
n_bags = config['training']['n_bags'] if not debug_flag else 3
n_repeats = config['n_repeats']
debug = '--debug ' if debug_flag else ''

baseline_path =  [ r['base'] for r in config['baseline_results']]
assert len(set(baseline_path)) == 1
baseline_path = baseline_path[0]

repeats_to_analyze = [1, 3, 6, 10, 15, 20, 30]
max_repeat_combis = 20

phenotypes = training_phenotypes

use_seed_opts = ['use_seed', 'wo_seed']
use_seed_dict = {'use_seed': '--use-seed-genes ', 'wo_seed': ' '}

rule all:
    input:
        replication = expand('replication/replication_{use_seed}_{repeat}_repeats.parquet', 
            repeat = repeats_to_analyze,
            use_seed = use_seed_opts),
        # plots = expand(f"plots/dicovery_replication_plot_{{use_seed}}_{n_repeats}_repeats.png",
        #     use_seed = use_seed_opts)

config_file_prefix = 'cv_split0/deeprvat/' if os.path.exists('cv_split0/') else '' #needed in case we analyse a CV experiment
print(config_file_prefix)
# # ############################### Run DeepRVAT ##############################################################
# # ###########################################################################################################
module deeprvat_workflow:
    snakefile: 
        f"{DEEPRVAT_DIR}/pipelines/training_association_testing_with_prefix.snakefile"
    prefix:
        './'
        # 'cv_split{cv_split}/deeprvat'
    config:
        config

use rule * from deeprvat_workflow exclude config, best_training_run, evaluate, choose_training_genes, association_dataset, train, regress, compute_burdens, compute_burdens_test, best_bagging_run, cleanup_burden_cache, link_burdens, link_burdens_test, all_burdens  as deeprvat_*




# use rule plot from deeprvat_workflow as deeprvat_plot with:
#     input:
#         significant = expand(f"{{phenotype}}/deeprvat/eval/{{{{use_seed}}}}/significant_{n_repeats}repeats.parquet",
#             phenotype=phenotypes),
#         results = expand(f"{{phenotype}}/deeprvat/eval/{{{{use_seed}}}}/all_results_{n_repeats}repeats.parquet",
#             phenotype=phenotypes),
#         replication = f'replication/replication_{{use_seed}}_{n_repeats}_repeats.parquet'
#     params: 
#         results_dir_pattern = 'deeprvat/eval/{use_seed}/',
#         results_dir = './',
#         code_dir = f'{DEEPRVAT_ANALYSIS_DIR}/association_testing'
#     output:
#          f"plots/dicovery_replication_plot_{{use_seed}}_{n_repeats}_repeats.png"

use rule compute_replication from deeprvat_workflow as deeprvat_compute_replication with:
    input:
        results = expand("{phenotype}/deeprvat/eval/{{use_seed}}/all_results_{{repeat}}repeats.parquet",
            phenotype = training_phenotypes)
    output:
        'replication/replication_{use_seed}_{repeat}_repeats.parquet'
    params:
        result_files = lambda wildcards, input: ''.join([
            f'--result-files {f} '
            for f in input.results
        ]),
        n_repeats = '{repeat}'


rule all_evaluate:
    input:
        expand("{phenotype}/deeprvat/eval/{use_seed}/significant_{repeat}repeats.parquet",
            repeat = repeats_to_analyze, phenotype = phenotypes, use_seed = use_seed_opts),
        expand("{phenotype}/deeprvat/eval/{use_seed}/all_results_{repeat}repeats.parquet",
            repeat = repeats_to_analyze, phenotype = phenotypes, use_seed = use_seed_opts)

use rule evaluate from deeprvat_workflow as deeprvat_evaluate with:
    input:
        associations = expand('{{phenotype}}/deeprvat/repeat_{repeat}/results/burden_associations.parquet',
                              repeat=range(n_repeats)),
        config = f'{config_file_prefix}{{phenotype}}/deeprvat/hpopt_config.yaml',
    output:
        "{phenotype}/deeprvat/eval/{use_seed}/significant_{repeat}repeats.parquet",
        "{phenotype}/deeprvat/eval/{use_seed}/all_results_{repeat}repeats.parquet"
    params:
        out_path = '{phenotype}/deeprvat/eval/{use_seed}',
        use_seed_genes = lambda wildcards: use_seed_dict[wildcards.use_seed],
        n_repeats = f'{n_repeats}',
        repeats_to_analyze = '{repeat}',
        max_repeat_combis = f"{max_repeat_combis}"





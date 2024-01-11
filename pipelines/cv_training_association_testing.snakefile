
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

cv_splits = config['n_folds']
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

phenotypes = training_phenotypes
# rule all:
#     input:
#         expand('{phenotype}/deeprvat/burdens/merging.finished', 
#         phenotype = phenotypes)
rule all:
    input:
        # significant = expand("{phenotype}/deeprvat/eval/significant.parquet",
        #        phenotype=phenotypes),
        # results = expand("{phenotype}/deeprvat/eval/all_results.parquet",
        #        phenotype=phenotypes),
        # replication = "replication.parquet",
        plots = expand("dicovery_replication_plot_{use_seed}.png", use_seed = ["wo_seed", 'use_seed'])


# rule all:
#     input:
#         expand('cv_split{cv_split}/deeprvat/models/repeat_{repeat}/best/bag_{bag}.ckpt',
#                bag=range(n_bags), repeat=range(n_repeats),
#                cv_split = range(cv_splits)),
#         expand('cv_split{cv_split}/deeprvat/models/repeat_{repeat}/config.yaml',
#                repeat=range(n_repeats),
#                cv_split = range(cv_splits))

# rule all:
#     input:
#         seed_genes = expand('cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/seed_genes.parquet',
#                             phenotype=phenotypes, cv_split = range(cv_splits)),


rule spread_config:
    input:
        config = 'config.yaml'
    output:
        train = 'cv_split{cv_split}/deeprvat/config.yaml',
    params:
        out_path = 'cv_split{cv_split}/'
    threads: 1
    resources:
        mem_mb = 1024,
        load = 1000
    shell:
        ' && '.join([
            conda_check,
            py_deeprvat + 'cv_utils.py spread-config '
            '-m deeprvat '
            '--fold {wildcards.cv_split} '
            # '--fold-specific-baseline '
            f'--n-folds {cv_splits}'
            ' {input.config} {params.out_path}'
        ])



# # ############################### Run DeepRVAT ##############################################################
# # ###########################################################################################################
module deeprvat_workflow:
    snakefile: 
        f"{DEEPRVAT_DIR}/pipelines/training_association_testing_with_prefix.snakefile"
    prefix:
        'cv_split{cv_split}/deeprvat'
    config:
        config

use rule * from deeprvat_workflow exclude config, best_training_run, evaluate, choose_training_genes, association_dataset, train, regress, compute_burdens, compute_burdens_test, best_bagging_run, cleanup_burden_cache, link_burdens, link_burdens_test, all_burdens  as deeprvat_*


use_seed_dict = {'use_seed': '--use-seed-genes ', 'wo_seed': ' '}


use rule plot from deeprvat_workflow as deeprvat_plot with:
    input:
        significant = expand("{phenotype}/deeprvat/eval/{{use_seed}}/significant.parquet",
               phenotype=phenotypes),
        results = expand("{phenotype}/deeprvat/eval/{{use_seed}}/all_results.parquet",
               phenotype=phenotypes),
        replication = "replication_{use_seed}.parquet"
    params: 
        results_dir_pattern = 'deeprvat/eval/{use_seed}/',
        results_dir = './',
        code_dir = f'{DEEPRVAT_ANALYSIS_DIR}/association_testing'
    output:
        "dicovery_replication_plot_{use_seed}.png"

use rule compute_replication from deeprvat_workflow as deeprvat_compute_replication with:
    input:
        results = expand("{phenotype}/deeprvat/eval/{{use_seed}}/all_results.parquet",
            phenotype = training_phenotypes)
    output:
        'replication_{use_seed}.parquet'


use rule evaluate from deeprvat_workflow as deeprvat_evaluate with:
    input:
        associations = expand('{{phenotype}}/deeprvat/repeat_{repeat}/results/burden_associations.parquet',
                              repeat=range(n_repeats)),
        config = 'cv_split0/deeprvat/{phenotype}/deeprvat/hpopt_config.yaml',
    output:
        "{phenotype}/deeprvat/eval/{use_seed}/significant.parquet",
        "{phenotype}/deeprvat/eval/{use_seed}/all_results.parquet"
    params:
        out_path = '{phenotype}/deeprvat/eval/{use_seed}',
        use_seed_genes = lambda wildcards: use_seed_dict[wildcards.use_seed],
        n_repeats = 6,
        repeats_to_analyze = 6, 
        max_repeat_combis = 1

use rule combine_regression_chunks from deeprvat_workflow as deeprvat_combine_regression_chunks with:
    input:
        expand('{{phenotype}}/deeprvat/repeat_{{repeat}}/results/burden_associations_{chunk}.parquet', 
        chunk=range(n_regression_chunks)),
    output:
        '{phenotype}/deeprvat/repeat_{repeat}/results/burden_associations.parquet',


#TODO this rule has to use the test burdens!
use rule regress from deeprvat_workflow as deeprvat_regress with:
    input:
        config = "cv_split0/deeprvat/{phenotype}/deeprvat/hpopt_config.yaml", #just a config from any 
        chunks = '{phenotype}/deeprvat/burdens/merging.finished',
        phenotype_0_chunks = f'{burden_phenotype}/deeprvat/burdens/merging.finished',
    output:
        temp('{phenotype}/deeprvat/repeat_{repeat}/results/burden_associations_{chunk}.parquet'),
    params:
        prefix = './'


# #TODO rule that combines the burdens from different cv splits
# # into folder {wildcards.phenotype}/deeprvat/burdens



use rule link_burdens from deeprvat_workflow as deeprvat_link_burdens with:
    params:
        prefix = 'cv_split{cv_split}/deeprvat'

use rule compute_burdens from deeprvat_workflow as deeprvat_compute_burdens with:
    priority: 100
    params:
        prefix = 'cv_split{cv_split}/deeprvat'


rule all_training:
    input:
        expand('cv_split{cv_split}/deeprvat/models/repeat_{repeat}/best/bag_{bag}.ckpt',
               bag=range(n_bags), repeat=range(n_repeats),
               cv_split = range(cv_splits)),
        expand('cv_split{cv_split}/deeprvat/models/repeat_{repeat}/config.yaml',
               repeat=range(n_repeats),
               cv_split = range(cv_splits))

use rule best_training_run from deeprvat_workflow as deeprvat_best_training_run with:
    params:
        prefix = 'cv_split{cv_split}/deeprvat'

use rule train from deeprvat_workflow as deeprvat_train with:
    priority: 1000
    params:
        prefix = 'cv_split{cv_split}/deeprvat',
        phenotypes = " ".join( #TODO like need the prefix here as well
            [f"--phenotype {p} "
             f"cv_split{{cv_split}}/deeprvat/{p}/deeprvat/input_tensor.zarr "
             f"cv_split{{cv_split}}/deeprvat/{p}/deeprvat/covariates.zarr "
             f"cv_split{{cv_split}}/deeprvat/{p}/deeprvat/y.zarr"
             for p in training_phenotypes])

use rule choose_training_genes from deeprvat_workflow as deeprvat_choose_training_genes with:
    params:
        prefix = 'cv_split{cv_split}/deeprvat'

use rule config from deeprvat_workflow as deeprvat_config with:
    input:
        config = 'cv_split{cv_split}/deeprvat/config.yaml', # TODO: change this into cv specific config
        # baseline = f'{baseline_path}/cv_split{{cv_split}}/baseline/{{phenotype}}/eval/burden_associations.parquet',
        baseline = lambda wildcards: [
            str(Path(r['base']) / f'cv_split{wildcards.cv_split}'/ 'baseline' / wildcards.phenotype / r['type'] /
                'eval/burden_associations.parquet')
            for r in config['baseline_results']
        ]
    params:
        baseline_results = lambda wildcards, input: ''.join([
            f'--baseline-results {b} '
            for b in input.baseline
        ])


# # ############################### Computation of test set deeprvat burdens ##############################################################
# # ############################################################################################################################
rule make_deeprvat_test_config:
    input:
        config_train = 'cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/hpopt_config.yaml'
    output:
        config_test = 'cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/hpopt_config_test.yaml'
    shell:
        ' && '.join([
            conda_check,
            py_deeprvat + 'cv_utils.py generate-test-config '
            '--fold {wildcards.cv_split} '
            f'--n-folds {cv_splits}'
            ' {input.config_train} {output.config_test}'
        ])

#generate the association data set from the test samples (as defined in the config)
#pass the sample file here
#then just use this data set nomrally for burden computation 
use rule association_dataset from deeprvat_workflow as deeprvat_association_dataset with:
    input:
        config = 'cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/hpopt_config_test.yaml'
    output:
        'cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/association_dataset.pkl'
    threads: 4


suffix_dict = {p: 'linked' if p != burden_phenotype else 'finished' for p in phenotypes}
rule combine_test_burdens:
    input: 
        lambda wildcards: [
            (f'cv_split{cv_split}/deeprvat/{wildcards.phenotype}/deeprvat/burdens/chunk{c}.{suffix_dict[wildcards.phenotype]}')
            for c in range(n_burden_chunks) for cv_split in range(cv_splits)
            ]
    output:
        '{phenotype}/deeprvat/burdens/merging.finished'
    params:
        out_dir = '{phenotype}/deeprvat/burdens',
        burden_paths = lambda wildcards, input: ''.join([
            f'--burden-dirs cv_split{fold}/deeprvat/{wildcards.phenotype}/deeprvat/burdens '
            for fold in range(cv_splits)]),
        link = lambda wildcards: (f'--link-burdens ../../../{burden_phenotype}/deeprvat/burdens/burdens.zarr' 
            if wildcards.phenotype != burden_phenotype else ' ')
    resources:
        mem_mb = 20480,
    shell:
        ' && '.join([
            conda_check,
            py_deeprvat + 'cv_utils.py combine-test-set-burdens '
            '{params.link} '
            '{params.burden_paths} '
            '{params.out_dir}',
        'touch {output}'
        ])





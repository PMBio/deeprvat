from pathlib import Path

configfile: 'config.yaml'

conda_check = 'conda info | grep "active environment"'
DEEPRVAT_DIR = os.environ['DEEPRVAT_DIR']
py_deeprvat = f'python {DEEPRVAT_DIR}/deeprvat/'

debug_flag = config.get('debug', False)
phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes
training_phenotypes = config["training"].get("phenotypes", phenotypes)
burden_phenotype = phenotypes[0]

n_burden_chunks = config.get('n_burden_chunks', 1) if not debug_flag else 2
n_regression_chunks = config.get('n_regression_chunks', 40) if not debug_flag else 2
n_trials = config['hyperparameter_optimization']['n_trials']
n_bags = config['training']['n_bags'] if not debug_flag else 3
n_repeats = config['n_repeats']
debug = '--debug ' if debug_flag else ''
do_scoretest = '--do-scoretest ' if config.get('do_scoretest', False) else ''
tensor_compression_level = config['training'].get('tensor_compression_level', 1)
model_path = Path("models")
n_parallel_training_jobs = config["training"].get("n_parallel_jobs", 1)

wildcard_constraints:
    repeat="\d+",
    trial="\d+",


repeats_to_average = [6]
n_avg_chunks = 20
burden_agg_fcts = ['mean']

cv_splits = 5
cv_exp = True
config_file_prefix = 'cv_split0/deeprvat/' if cv_exp else '' #needed in case we analyse a CV experiment

use_seed_opts = ['use_seed', 'wo_seed']
use_seed_opts = ['wo_seed']
use_seed_dict = {'use_seed': '--use-seed-genes ', 'wo_seed': ' '}



include: "cv_training.snakefile"
include: "cv_burdens.snakefile"
include: "../association_testing/burdens.snakefile"
include: "../association_testing/regress_eval_avg.snakefile"

# phenotypes = training_phenotypes


# print(phenotypes)
# rule all:
#     input:
#         expand('{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/burden_associations.parquet',
#             n_avg_repeats = repeats_to_average,
#             burden_agg_fct = burden_agg_fcts,
#             phenotype = phenotypes
#         )


rule all_evaluate_avg:
    input:
        significant = expand("{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/eval/{use_seed}/significant.parquet",
               phenotype=phenotypes,
               n_avg_repeats = repeats_to_average,
               burden_agg_fct = burden_agg_fcts,
               use_seed = use_seed_opts),
        results = expand("{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/eval/{use_seed}/all_results.parquet",
               phenotype=phenotypes,
               n_avg_repeats = repeats_to_average,
               burden_agg_fct = burden_agg_fcts,
               use_seed =  use_seed_opts),

rule all_regression_avg:
    input:
        expand('{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/burden_associations.parquet',
            n_avg_repeats = repeats_to_average,
            burden_agg_fct = burden_agg_fcts,
            phenotype = phenotypes
        )

rule all_average_burdens:
    input:
        expand('{phenotype}/deeprvat/burdens/logs/burdens_{burden_agg_fct}_{n_avg_repeats}_repeats_chunk_{chunk}.finished',
            n_avg_repeats = repeats_to_average,
            chunk = range(n_avg_chunks),
            burden_agg_fct = burden_agg_fcts,
            phenotype = phenotypes[0]),

rule all_burdens:
    input:
        expand('{phenotype}/deeprvat/burdens/merging.finished', 
        phenotype = phenotypes)


rule all_training:
    input:
        expand('cv_split{cv_split}/deeprvat/models/repeat_{repeat}/best/bag_{bag}.ckpt',
               bag=range(n_bags), repeat=range(n_repeats),
               cv_split = range(cv_splits)),
        expand('cv_split{cv_split}/deeprvat/models/repeat_{repeat}/config.yaml',
               repeat=range(n_repeats),
               cv_split = range(cv_splits))


rule all_config:
    input:
        expand('cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/hpopt_config.yaml',
               phenotype=phenotypes,
               cv_split = range(cv_splits))




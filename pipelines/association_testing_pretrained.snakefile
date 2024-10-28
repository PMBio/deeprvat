from pathlib import Path
from deeprvat.deeprvat.config import create_main_config
import logging

create_main_config("deeprvat_input_pretrained_models_config.yaml")

for handler in logging.root.handlers[:]:
    #remove duplicate logging handlers from loaded deeprvat.config module
    logging.root.removeHandler(handler) 

configfile: 'deeprvat_config.yaml'

logging_redirct = "> {log.stdout} 2> {log.stderr}" #for Linux-based systems
debug_flag = config.get('debug', False)
phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes
training_phenotypes = []

n_burden_chunks = config.get('n_burden_chunks', 1) if not debug_flag else 2
n_regression_chunks = config.get('n_regression_chunks', 40) if not debug_flag else 2
n_avg_chunks = config.get('n_avg_chunks', 1)
n_trials = config['hyperparameter_optimization']['n_trials']
n_bags = config['training']['n_bags'] if not debug_flag else 3
n_repeats = config['n_repeats']
debug = '--debug ' if debug_flag else ''
do_scoretest = '--do-scoretest ' if config.get('do_scoretest', False) else ''
tensor_compression_level = config['training'].get('tensor_compression_level', 1)
model_path = Path(config.get("pretrained_model_path", "pretrained_models"))

wildcard_constraints:
    repeat="\d+",
    trial="\d+",

cv_exp = config.get('cv_exp', False)
config_file_prefix = (
    "cv_split0/deeprvat/" if cv_exp else ""
) 

include: "training/config.snakefile"
include: "association_testing/association_dataset.snakefile"
include: "association_testing/burdens.snakefile"
include: "association_testing/regress_eval.snakefile"

rule all:
    input:
        expand("{phenotype}/deeprvat/eval/significant.parquet",
               phenotype=phenotypes),
        expand("{phenotype}/deeprvat/eval/all_results.parquet",
               phenotype=phenotypes)

rule all_regression:  #regress_eval.snakefile
    input:
        expand(
            "{phenotype}/deeprvat/average_regression_results/burden_associations.parquet",
            phenotype=phenotypes,
        ),


rule all_average_burdens:  #burdens.snakefile
    input:
        expand(
            "{phenotype}/deeprvat/burdens/logs/burdens_averaging_{chunk}.finished",
            chunk=range(n_avg_chunks),
            phenotype=phenotypes[0],
        ),

rule all_config:  #cv_training.snakefile
    input:
        expand(
            "{phenotype}/deeprvat/config.yaml",
            phenotype=phenotypes,
        ),

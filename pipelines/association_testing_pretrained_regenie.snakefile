from pathlib import Path
from deeprvat.deeprvat.config import create_main_config
import logging

create_main_config("deeprvat_input_pretrained_models_config.yaml")

#remove duplicate logging handlers from loaded deeprvat.config module
logging.root.handlers.clear()

configfile: 'deeprvat_config.yaml'

logging_redirct = "1> {log.stdout} 2> {log.stderr}" #for Linux-based systems
debug_flag = config.get('debug', False)
phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes
training_phenotypes = []

n_burden_chunks = config.get('n_burden_chunks', 1) if not debug_flag else 2
n_regression_chunks = config.get('n_regression_chunks', 40) if not debug_flag else 2
n_avg_chunks = config.get('n_avg_chunks', 1)
n_bags = config['training']['n_bags'] if not debug_flag else 3
n_repeats = config['n_repeats']
debug = '--debug ' if debug_flag else ''
do_scoretest = '--do-scoretest ' if config.get('do_scoretest', False) else ''
model_path = Path(config.get("pretrained_model_path", "pretrained_models"))

burdens = Path(config.get("burdens", "burdens/burdens_average.zarr"))

regenie_config_step1 = config["regenie_options"]["step_1"]
regenie_config_step2 = config["regenie_options"]["step_2"]
regenie_step1_bsize = regenie_config_step1["bsize"]
regenie_step2_bsize = regenie_config_step2["bsize"]

cv_exp = False
config_file_prefix = (
    "cv_split0/deeprvat/" if cv_exp else ""
)

wildcard_constraints:
    repeat="\d+",
    trial="\d+",

include: "training/config.snakefile"
include: "association_testing/association_dataset.snakefile"
include: "association_testing/burdens.snakefile"
include: "association_testing/regress_eval_regenie.snakefile"

rule all:
    input:
        expand("{phenotype}/deeprvat/eval/significant.parquet",
               phenotype=phenotypes),
        expand("{phenotype}/deeprvat/eval/all_results.parquet",
               phenotype=phenotypes)

rule all_burdens:
    input:
        [
            (f'{p}/deeprvat/burdens/chunk{c}.' +
             ("finished" if p == phenotypes[0] else "linked"))
            for p in phenotypes
            for c in range(n_burden_chunks)
        ]

rule all_association_dataset:
    input:
        expand('{phenotype}/deeprvat/association_dataset.pkl',
               phenotype=phenotypes)

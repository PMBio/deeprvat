from pathlib import Path
from deeprvat.deeprvat.config import create_main_config
from deeprvat.deeprvat.check_input_data import check_input_data

input_config_file = "deeprvat_input_config.yaml"

if config.get("skip_sanity_check", False):
    logger.info("Skipping sanity check as skip_sanity_check was specified in config")
else:
    check_input_data(input_config_file)

create_main_config(input_config_file)

configfile: 'deeprvat_config.yaml'

debug_flag = config.get('debug', False)
phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes
training_phenotypes = []

n_burden_chunks = config.get('n_burden_chunks', 1) if not debug_flag else 2
n_avg_chunks = config.get('n_avg_chunks', 40)
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

rule all_association_dataset:
    input:
        expand('{phenotype}/deeprvat/association_dataset.pkl',
               phenotype=phenotypes),
        'association_dataset_burdens.pkl',

rule all_config:
    input:
        expand(f"{config_file_prefix}{{phenotype}}/deeprvat/config.yaml",
               phenotype=phenotypes)

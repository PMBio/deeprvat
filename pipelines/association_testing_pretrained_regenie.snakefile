from pathlib import Path
from os.path import exists

if not exists('./deeprvat_config.yaml'):
    if not config: #--configfile argument was not passed
        print("Generating deeprvat_config.yaml...")
        from deeprvat.deeprvat.config import create_main_config
        create_main_config('deeprvat_input_pretrained_models_config.yaml')
        print("     Finished.")

configfile: 'deeprvat_config.yaml'

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
regenie_step2_bsize = regenie_config_step2["bsize"] # TODO SHOULD THIS BE HERE

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

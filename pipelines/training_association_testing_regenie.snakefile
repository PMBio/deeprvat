from pathlib import Path
from os.path import exists

if not exists('./deeprvat_config.yaml'):
    if not config: #--configfile argument was not passed
        print("Generating deeprvat_config.yaml...")
        from deeprvat.deeprvat.config import create_main_config
        create_main_config('deeprvat_input_config.yaml')
        print("     Finished.")

configfile: 'deeprvat_config.yaml'

debug_flag = config.get('debug', False)
phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes
training_phenotypes = config["training"].get("phenotypes", phenotypes)

n_burden_chunks = config.get('n_burden_chunks', 1) if not debug_flag else 2
n_regression_chunks = config.get('n_regression_chunks', 40) if not debug_flag else 2
n_avg_chunks = config.get('n_avg_chunks', 1)
n_trials = config['hyperparameter_optimization']['n_trials']
n_bags = config['training']['n_bags'] if not debug_flag else 3
n_repeats = config['n_repeats']
debug = '--debug ' if debug_flag else ''
do_scoretest = '--do-scoretest ' if config.get('do_scoretest', False) else ''
tensor_compression_level = config['training'].get('tensor_compression_level', 1)
model_path = Path("models")
n_parallel_training_jobs = config["training"].get("n_parallel_jobs", 1)
cv_exp = False

wildcard_constraints:
    repeat="\d+",
    trial="\d+",

include: "training/config.snakefile"
include: "training/training_dataset.snakefile"
include: "training/train.snakefile"
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

rule all_training:
    input:
        expand(model_path / 'repeat_{repeat}/best/bag_{bag}.ckpt',
               bag=range(n_bags), repeat=range(n_repeats)),
        model_path / "model_config.yaml"

rule all_training_dataset:
    input:
        input_tensor = expand('{phenotype}/deeprvat/input_tensor.zarr',
                              phenotype=training_phenotypes, repeat=range(n_repeats)),
        covariates = expand('{phenotype}/deeprvat/covariates.zarr',
                            phenotype=training_phenotypes, repeat=range(n_repeats)),
        y = expand('{phenotype}/deeprvat/y.zarr',
                   phenotype=training_phenotypes, repeat=range(n_repeats))

rule all_config:
    input:
        seed_genes = expand('{phenotype}/deeprvat/seed_genes.parquet',
                            phenotype=phenotypes),
        data_config = expand('{phenotype}/deeprvat/config.yaml',
                        phenotype=phenotypes),
        baseline = expand('{phenotype}/deeprvat/baseline_results.parquet',
                          phenotype=phenotypes),

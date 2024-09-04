from pathlib import Path
from os.path import exists

if not exists('./deeprvat_config.yaml'):
    if not config: #--configfile argument was not passed
        print("Generating deeprvat_config.yaml...")
        from deeprvat.deeprvat.config import create_main_config
        create_main_config('deeprvat_input_training_config.yaml')
        print("     Finished.")

configfile: 'deeprvat_config.yaml'

debug_flag = config.get('debug', False)
debug = '--debug ' if debug_flag else ''
deterministic_flag = config.get('deterministic', False)
deterministic = '--deterministic ' if deterministic_flag else ''

#phenotypes = config['phenotypes'] # TODO SHOULD THIS BE HERE?
#phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes

training_phenotypes = config["training"].get("phenotypes")
training_phenotypes = list(training_phenotypes.keys()) if type(training_phenotypes) == dict else training_phenotypes
n_trials = config['hyperparameter_optimization']['n_trials']
n_bags = config['training']['n_bags'] if not debug_flag else 3
n_repeats = config['n_repeats']
tensor_compression_level = config['training'].get('tensor_compression_level', 1)
model_path = Path("models")
n_parallel_training_jobs = config["training"].get("n_parallel_jobs", 1)

wildcard_constraints:
    repeat="\d+",
    trial="\d+",

include: "training/config.snakefile"
include: "training/training_dataset.snakefile"
include: "training/train.snakefile"

rule all:
    input:
        expand( model_path / 'repeat_{repeat}/best/bag_{bag}.ckpt',
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
                            phenotype=training_phenotypes),
        data_config = expand('{phenotype}/deeprvat/config.yaml',
                        phenotype=training_phenotypes),
        baseline = expand('{phenotype}/deeprvat/baseline_results.parquet',
                          phenotype=training_phenotypes),

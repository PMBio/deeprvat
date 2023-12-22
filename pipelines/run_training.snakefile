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
        model_path / "config.yaml"

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
        config = expand('{phenotype}/deeprvat/hpopt_config.yaml',
                        phenotype=phenotypes),
        baseline = expand('{phenotype}/deeprvat/baseline_results.parquet',
                          phenotype=phenotypes),
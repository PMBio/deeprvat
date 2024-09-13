from pathlib import Path
from deeprvat.deeprvat.config import create_main_config

create_main_config("deeprvat_input_config.yaml")

configfile: "deeprvat_config.yaml"


conda_check = 'conda info | grep "active environment"'

debug_flag = config.get("debug", False)
phenotypes = config["phenotypes"]
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes
training_phenotypes = config["training"].get("phenotypes", phenotypes)
training_phenotypes = list(training_phenotypes.keys()) if type(training_phenotypes) == dict else training_phenotypes
burden_phenotype = phenotypes[0]

n_burden_chunks = config.get("n_burden_chunks", 1) if not debug_flag else 2
n_regression_chunks = config.get("n_regression_chunks", 40) if not debug_flag else 2
n_avg_chunks = config.get('n_avg_chunks', 1)
n_trials = config["hyperparameter_optimization"]["n_trials"]
n_bags = config["training"]["n_bags"] if not debug_flag else 3
n_repeats = config["n_repeats"]
debug = "--debug " if debug_flag else ""
do_scoretest = "--do-scoretest " if config.get("do_scoretest", False) else ""
tensor_compression_level = config["training"].get("tensor_compression_level", 1)
model_path = Path("pretrained_models")
n_parallel_training_jobs = config["training"].get("n_parallel_jobs", 1)


wildcard_constraints:
    repeat="\d+",
    trial="\d+",
    phenotype="[\w\d\-]+",
    cv_split="\d+",

cv_exp = config.get('cv_exp',True)
cv_splits = config.get("n_folds", 5)

include: "cv_training.snakefile"
include: "cv_burdens.snakefile"
# include: "../association_testing/burdens.snakefile"
include: "../association_testing/regress_eval.snakefile"


rule all_evaluate:  #regress_eval.snakefile
    input:
        significant=expand(
            "{phenotype}/deeprvat/eval/significant.parquet", phenotype=phenotypes
        ),
        results=expand(
            "{phenotype}/deeprvat/eval/all_results.parquet", phenotype=phenotypes
        ),


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


rule all_burdens:  #cv_burdens.snakefile
    input:
        expand("burdens/log/{phenotype}/merging.finished",
               phenotype=phenotypes),


rule all_training:  #cv_training.snakefile
    input:
        expand(
            "cv_split{cv_split}/deeprvat" / model_path / "repeat_{repeat}/best/bag_{bag}.ckpt",
            bag=range(n_bags),
            repeat=range(n_repeats),
            cv_split=range(cv_splits),
        ),
        expand(
            "cv_split{cv_split}/deeprvat/" / model_path / "repeat_{repeat}/model_config.yaml",
            repeat=range(n_repeats),
            cv_split=range(cv_splits),
        ),


rule all_config:  #cv_training.snakefile
    input:
        expand(
            "cv_split{cv_split}/deeprvat/{phenotype}/deeprvat/config.yaml",
            phenotype=phenotypes,
            cv_split=range(cv_splits),
        ),

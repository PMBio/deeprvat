import logging
import pprint
import sys
from pprint import pprint
from typing import Optional, Tuple

import click
import pandas as pd
import torch.nn.functional as F
import yaml

from deeprvat.deeprvat.evaluate import pval_correction
from pathlib import Path
import os
from copy import deepcopy

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


def create_main_config(
    config_file: str,
    output_dir: Optional[str] = ".",
):
    """
    Generates the necessary deeprvat_config.yaml file for running all pipelines.
    This function expects inputs as shown in the following config-file:
        - DEEPRVAT_DIR/example/deeprvat_input_config.yaml

    :param config_file: Path to directory of relevant config yaml file
    :type config_file: str
    :param output_dir: Path to directory where created deeprvat_config.yaml will be saved.
    :type output_dir: str
    :return: Joined configuration file saved to deeprvat_config.yaml.
    """

    with open(config_file) as f:
        input_config = yaml.safe_load(f)
    
    # Base Config
    with open(f"{input_config['deeprvat_repo_dir']}/deeprvat/deeprvat/base_configurations.yaml") as f:
        base_config = yaml.safe_load(f)
    
    full_config = base_config

    expected_input_keys = [
        "deeprvat_repo_dir",
        "phenotypes_for_association_testing",
        "phenotypes_for_training",
        "gt_filename",
        "variant_filename",
        "phenotype_filename",
        "annotation_filename",
        "gene_filename",
        "rare_variant_annotations",
        "covariates",
        "association_testing_data_thresholds",
        "training_data_thresholds",
        "seed_gene_results",
        "training",
        "n_repeats",
        "y_transformation",
        "evaluation",
        "cv_options",
        "regenie_options",
    ]

    # CV setup parameters
    if not input_config["cv_options"]["cv_exp"]:
        logger.info("Not CV setup...removing CV pipeline parameters from config")
        full_config["cv_exp"] = False
    else: #CV experiment setup specified
        if any(
            key not in input_config["cv_options"]
            for key in ["cv_exp","cv_path","n_folds"]
        ):
            raise KeyError(
                "Missing keys cv_path or n_folds under config['cv_options'] "
                "Please review DEEPRVAT_DIR/example/config/deeprvat_input_config.yaml for list of keys."
            )
        full_config["cv_path"] = input_config["cv_path"]
        full_config["n_folds"] = input_config["n_folds"]
        full_config["cv_exp"] = True

    # REGENIE setup parameters
    if not input_config["regenie_options"]["regenie_exp"]:
        logger.info("Not using REGENIE integration...removing REGENIE parameters from config")
        full_config["regenie_exp"] = False
    else: #REGENIE integration
        if any(
            key not in input_config["regenie_options"]
            for key in ["regenie_exp","step_1","step_2"]
        ):
            raise KeyError(
                "Missing keys step_1 or step_2 under config['regenie_options'] "
                "Please review DEEPRVAT_DIR/example/config/deeprvat_input_config.yaml for list of keys."
            )
        full_config["regenie_exp"] = True
        full_config["regenie_options"] = {}
        full_config["gtf_file"] = input_config["regenie_options"]["gtf_file"]
        full_config["regenie_options"]["step_1"] = input_config["regenie_options"]["step_1"]
        full_config["regenie_options"]["step_2"] = input_config["regenie_options"]["step_2"]
        
    no_pretrain = True
    if "use_pretrained_models" in input_config:
        if input_config["use_pretrained_models"]:
            no_pretrain = False
            logger.info("Pretrained Model setup specified.")
            to_remove = {"training", "phenotypes_for_training", "seed_gene_results"}
            expected_input_keys = [
                item for item in expected_input_keys if item not in to_remove
            ]
            
            pretrained_model_path = Path(input_config["pretrained_model_path"])

            expected_input_keys.extend(["use_pretrained_models", "model", "pretrained_model_path"])

            with open(f"{pretrained_model_path}/config.yaml") as f:
                pretrained_config = yaml.safe_load(f)

            for k in pretrained_config:
                input_config[k] = deepcopy(pretrained_config[k])

    if set(input_config.keys()) != set(expected_input_keys):
        if set(input_config.keys()) - set(expected_input_keys):
            raise KeyError(
                (
                    "Unspecified key(s) present in input YAML file. "
                    f"The follow extra keys are present: {set(input_config.keys()) - set(expected_input_keys)} "
                    "Please review DEEPRVAT_DIR/example/config/deeprvat_input_config.yaml for list of keys."
                )
            )
        if set(expected_input_keys) - set(input_config.keys()):
            raise KeyError(
                (
                    "Missing key(s) in input YAML file. "
                    f"The follow keys are missing: {set(expected_input_keys) - set(input_config.keys())} "
                    "Please review DEEPRVAT_DIR/example/config/deeprvat_input_config.yaml for list of keys."
                )
            )

    if no_pretrain:
        if any(
            key not in input_config["training"]
            for key in ["pl_trainer", "early_stopping"]
        ):
            raise KeyError(
                "Missing keys pl_trainer and/or early_stopping under config['training'] "
                "Please review DEEPRVAT_DIR/example/config/deeprvat_input_config.yaml for list of keys."
            )

    # Phenotypes
    full_config["phenotypes"] = {}
    for pheno in input_config["phenotypes_for_association_testing"]:
        full_config["phenotypes"][pheno] = {}
        # Can optionally specify dictionary of = {"min_seed_genes": 3, "max_seed_genes": None, "pvalue_threshold": None}
    full_config["training_data"]["dataset_config"]["y_transformation"] = input_config[
        "y_transformation"
    ]
    full_config["association_testing_data"]["dataset_config"]["y_transformation"] = (
        input_config["y_transformation"]
    )
    # genotypes.h5
    full_config["training_data"]["gt_file"] = input_config["gt_filename"]
    full_config["association_testing_data"]["gt_file"] = input_config["gt_filename"]
    # variants.parquet
    full_config["training_data"]["variant_file"] = input_config["variant_filename"]
    full_config["association_testing_data"]["variant_file"] = input_config[
        "variant_filename"
    ]
    # phenotypes.parquet
    full_config["training_data"]["dataset_config"]["phenotype_file"] = input_config[
        "phenotype_filename"
    ]
    full_config["association_testing_data"]["dataset_config"]["phenotype_file"] = (
        input_config["phenotype_filename"]
    )
    # annotations.parquet
    full_config["training_data"]["dataset_config"]["annotation_file"] = input_config[
        "annotation_filename"
    ]
    full_config["association_testing_data"]["dataset_config"]["annotation_file"] = (
        input_config["annotation_filename"]
    )
    # protein_coding_genes.parquet
    full_config["association_testing_data"]["dataset_config"]["gene_file"] = (
        input_config["gene_filename"]
    )
    full_config["association_testing_data"]["dataset_config"]["rare_embedding"][
        "config"
    ]["gene_file"] = input_config["gene_filename"]
    # rare_variant_annotations
    full_config["training_data"]["dataset_config"]["rare_embedding"]["config"][
        "annotations"
    ] = input_config["rare_variant_annotations"]
    full_config["association_testing_data"]["dataset_config"]["rare_embedding"][
        "config"
    ]["annotations"] = input_config["rare_variant_annotations"]
    # variant annotations
    anno_list = deepcopy(input_config["rare_variant_annotations"])
    for i, k in enumerate(input_config["training_data_thresholds"].keys()):
        anno_list.insert(i + 1, k)
    full_config["training_data"]["dataset_config"]["annotations"] = anno_list
    full_config["association_testing_data"]["dataset_config"]["annotations"] = anno_list
    # covariates
    full_config["training_data"]["dataset_config"]["x_phenotypes"] = input_config[
        "covariates"
    ]
    full_config["association_testing_data"]["dataset_config"]["x_phenotypes"] = (
        input_config["covariates"]
    )
    # Thresholds
    full_config["training_data"]["dataset_config"]["rare_embedding"]["config"][
        "thresholds"
    ] = {}
    full_config["association_testing_data"]["dataset_config"]["rare_embedding"][
        "config"
    ]["thresholds"] = {}
    for k, v in input_config["training_data_thresholds"].items():
        full_config["training_data"]["dataset_config"]["rare_embedding"]["config"][
            "thresholds"
        ][k] = f"{k} {v}"
    for k, v in input_config["association_testing_data_thresholds"].items():
        full_config["association_testing_data"]["dataset_config"]["rare_embedding"][
            "config"
        ]["thresholds"][k] = f"{k} {v}"
    # Results evaluation parameters; alpha parameter for significance threshold
    if "evaluation" not in full_config:
        full_config["evaluation"] = {}
    full_config["evaluation"]["correction_method"] = input_config["evaluation"][
        "correction_method"
    ]
    full_config["evaluation"]["alpha"] = input_config["evaluation"]["alpha"]
    # DeepRVAT model
    full_config["n_repeats"] = input_config["n_repeats"]

    full_config["data"] = full_config["association_testing_data"]
    del full_config["association_testing_data"]

    if no_pretrain:
        # PL trainer
        full_config["training"]["pl_trainer"] = input_config["training"]["pl_trainer"]
        # Early Stopping
        full_config["training"]["early_stopping"] = input_config["training"][
            "early_stopping"
        ]
        # Training Phenotypes
        full_config["training"]["phenotypes"] = input_config["phenotypes_for_training"]
        # Baseline results
        if "baseline_results" not in full_config:
            full_config["baseline_results"] = {}
        full_config["baseline_results"]["options"] = input_config["seed_gene_results"][
            "result_dirs"
        ]
        full_config["baseline_results"]["alpha_seed_genes"] = input_config[
            "seed_gene_results"
        ]["alpha_seed_genes"]
        full_config["baseline_results"]["correction_method"] = input_config[
            "seed_gene_results"
        ]["correction_method"]
    else:
        full_config["model"] = input_config["model"]
        full_config["pretrained_model_path"] = input_config["pretrained_model_path"]
        #need to also save deeprvat_config.yaml also to pretrained-model dir
        with open(
            f"{pretrained_model_path}/deeprvat_config.yaml", "w"
        ) as f:
            yaml.dump(full_config, f)

    with open(f"{output_dir}/deeprvat_config.yaml", "w") as f:
        yaml.dump(full_config, f)


def create_sg_discovery_config(
    config_file: str,
    output_dir: Optional[str] = ".",
):
    """
    Generates the necessary sg_discovery_config.yaml file for running the seed_gene_discovery pipelines.
    This function expects inputs as shown in the following config-file:
        - DEEPRVAT_DIR/example/seed_gene_discovery_input_config.yaml

    :param config_file: Path to directory of relevant config yaml file
    :type config_file: str
    :param output_dir: Path to directory where created sg_discovery_config.yaml will be saved.
    :type output_dir: str
    :return: Joined configuration file saved to sg_discovery_config.yaml.
    """

    with open(config_file) as f:
        input_config = yaml.safe_load(f)
    
    # Base Config
    with open(f"{input_config['deeprvat_repo_dir']}/deeprvat/seed_gene_discovery/seed_gene_base_configurations.yaml") as f:
        base_config = yaml.safe_load(f)
    
    full_config = base_config

    expected_input_keys = [
        "deeprvat_repo_dir",
        "phenotypes",
        "gt_filename",
        "variant_filename",
        "phenotype_filename",
        "annotation_filename",
        "gene_filename",
        "covariates",
        "annotations",
        "test_types",
        "variant_types",
        "rare_maf",
        "alpha_seed_genes",
        "test_config",
        "dataset_config",
    ]

    if set(input_config.keys()) != set(expected_input_keys):
        if set(input_config.keys()) - set(expected_input_keys):
            raise KeyError(
                (
                    "Unspecified key(s) present in input YAML file. "
                    f"The follow extra keys are present: {set(input_config.keys()) - set(expected_input_keys)} "
                    "Please review DEEPRVAT_DIR/example/config/seed_gene_discovery_input_config.yaml for list of keys."
                )
            )
        if set(expected_input_keys) - set(input_config.keys()):
            raise KeyError(
                (
                    "Missing key(s) in input YAML file. "
                    f"The follow keys are missing: {set(expected_input_keys) - set(input_config.keys())} "
                    "Please review DEEPRVAT_DIR/example/config/seed_gene_discovery_input_config.yaml for list of keys."
                )
            )

    # Phenotypes
    full_config["phenotypes"] = input_config["phenotypes"]
    # genotypes.h5
    full_config["data"]["gt_file"] = input_config["gt_filename"]
    # variants.parquet
    full_config["variant_file"] = input_config["variant_filename"]
    full_config["data"]["variant_file"] = input_config["variant_filename"]
    # phenotypes.parquet
    full_config["data"]["dataset_config"]["phenotype_file"] = input_config["phenotype_filename"]
    # annotations.parquet
    full_config["data"]["dataset_config"]["annotation_file"] = input_config["annotation_filename"]
    # protein_coding_genes.parquet
    full_config["data"]["dataset_config"]["gene_file"] = input_config["gene_filename"]
    full_config["data"]["dataset_config"]["rare_embedding"]["config"]["gene_file"] = input_config["gene_filename"]
    # X_phenotypes (covariates)
    full_config["data"]["dataset_config"]["x_phenotypes"] = input_config["covariates"]
    # Annotations
    full_config["data"]["dataset_config"]["annotations"] = input_config["annotations"]
    full_config["data"]["dataset_config"]["rare_embedding"]["config"]["annotations"] = input_config["annotations"]
    # Test Types
    full_config["test_types"] = input_config["test_types"]
    # Variant Types
    full_config["variant_types"] = input_config["variant_types"]
    # Minor allele frequency threshold
    full_config["rare_maf"] = input_config["rare_maf"]
    # alpha parameter
    full_config["alpha"] = input_config["alpha_seed_genes"]
    # Test Configurations
    full_config["test_config"] = input_config["test_config"]
    # Dataset Configurations
    full_config["data"]["dataset_config"]["standardize_xpheno"] = input_config["dataset_config"]["standardize_xpheno"]
    full_config["data"]["dataset_config"]["y_transformation"] = input_config["dataset_config"]["y_transformation"]
    full_config["data"]["dataset_config"]["standardize_xpheno"] = input_config["dataset_config"]["standardize_xpheno"]
    full_config["data"]["dataset_config"]["min_common_af"] = input_config["dataset_config"]["min_common_af"]
    full_config["data"]["dataset_config"]["rare_embedding"]["type"] = input_config["dataset_config"]["rare_embedding"]["type"]

    with open(f"{output_dir}/sg_discovery_config.yaml", "w") as f:
        yaml.dump(full_config, f)

@cli.command()
@click.option("--association-only", is_flag=True)
@click.option("--phenotype", type=str)
@click.option("--baseline-results", type=click.Path(exists=True), multiple=True)
@click.option("--baseline-results-out", type=click.Path())
@click.option("--seed-genes-out", type=click.Path())
@click.argument("old_config_file", type=click.Path(exists=True))
@click.argument("new_config_file", type=click.Path())
def update_config(
    association_only: bool,
    phenotype: Optional[str],
    baseline_results: Tuple[str],
    baseline_results_out: Optional[str],
    seed_genes_out: Optional[str],
    old_config_file: str,
    new_config_file: str,
):
    """
    Select seed genes based on baseline results and update the configuration file.

    :param association_only: Update config file only for association testing
    :type association_only: bool
    :param old_config_file: Path to the old configuration file.
    :type old_config_file: str
    :param phenotype: Phenotype to update in the configuration.
    :type phenotype: Optional[str]
    :param baseline_results: Paths to baseline result files.
    :type baseline_results: Tuple[str]
    :param baseline_results_out: Path to save the updated baseline results.
    :type baseline_results_out: Optional[str]
    :param seed_genes_out: Path to save the seed genes.
    :type seed_genes_out: Optional[str]
    :param new_config_file: Path to the new configuration file.
    :type new_config_file: str
    :raises ValueError: If neither --seed-gene-dir nor --baseline-results is specified.
    :return: Updated configuration file saved to new_config.yaml.
             Selected seed genes saved to seed_genes_out.parquet.
             Optionally, save baseline results to a parquet file if baseline_results_out is specified.
    """
    if not association_only and len(baseline_results) == 0:
        raise ValueError(
            "One of --baseline-results or --association-only must be specified"
        )

    with open(old_config_file) as f:
        config = yaml.safe_load(f)

    if phenotype is not None:
        logger.info(f"Updating config for phenotype {phenotype}")
        config["data"]["dataset_config"]["y_phenotypes"] = [phenotype]
        if not association_only:
            config["training_data"]["dataset_config"]["y_phenotypes"] = [phenotype]

        # For using seed genes from results of baseline methods
        if len(baseline_results) > 0:
            logger.info("Choosing seed genes based on baseline results")
            if phenotype is None or seed_genes_out is None:
                raise ValueError(
                    "--phenotype and --seed-genes-out must be "
                    "specified if --baseline-results is"
                )
            seed_config = config["phenotypes"][phenotype]
            correction_method = config["baseline_results"].get(
                "correction_method", None
            )
            min_seed_genes = seed_config.get("min_seed_genes", 3)
            max_seed_genes = seed_config.get("max_seed_genes", None)
            threshold = seed_config.get("pvalue_threshold", None)
            assert (
                min_seed_genes is None
                or max_seed_genes is None
                or min_seed_genes < max_seed_genes
            )

            baseline_columns = ["gene", "pval"]
            logger.info(f"  Reading baseline results from:")
            pprint(baseline_results)
            baseline_df = pd.concat(
                [
                    pd.read_parquet(r, columns=baseline_columns, engine="pyarrow")
                    for r in baseline_results
                ]
            )
            if "EAC" in baseline_df:
                # filter for genes with expected allele count > 50 (as done by Karcewski et al.)
                baseline_df = baseline_df.query("EAC > 50")
            else:
                logger.info("Not performing EAC filtering of baseline results")
            logger.info(f"  Correcting p-values using {correction_method} method")
            alpha = config["baseline_results"].get(
                "alpha_seed_genes", config["evaluation"].get("alpha")
            )
            baseline_df = pval_correction(
                baseline_df, alpha, correction_type=correction_method
            )
            baseline_df = baseline_df.sort_values("pval_corrected")

            if baseline_results_out is not None:
                baseline_df.to_parquet(baseline_results_out, engine="pyarrow")
            if correction_method is not None:

                logger.info(f"Using significant genes with corrected pval < {alpha}")
                if (
                    len(baseline_df.query("significant")["gene"].unique())
                    < min_seed_genes
                ):
                    logger.info(
                        f"Selecting top {min_seed_genes} genes from baseline because less than {min_seed_genes} genes are significant"
                    )
                    baseline_df = baseline_df.drop_duplicates(subset="gene").head(
                        min_seed_genes
                    )  # TODO make this flexible
                else:
                    baseline_df = baseline_df.query("significant")
            else:
                if threshold is not None:
                    baseline_temp = baseline_df.query(f"pval_corrected < @threshold")
                    logger.info(
                        f"  {len(baseline_df)} genes "
                        "from baseline passed thresholding"
                    )
                    if len(baseline_temp) >= min_seed_genes:
                        baseline_df = baseline_temp
                    else:
                        baseline_df = baseline_df.head(min_seed_genes)
                        assert len(baseline_df) == min_seed_genes
                        logger.info(
                            f"  Retaining top {min_seed_genes} "
                            "seed genes with lowest p-value"
                        )

            baseline_df = baseline_df.drop_duplicates(subset="gene")
            logger.info(f"  {len(baseline_df)} significant genes from baseline")

            genes = pd.read_parquet(
                config["data"]["dataset_config"]["gene_file"], engine="pyarrow"
            )
            seed_gene_df = pd.merge(
                baseline_df,
                genes,
                left_on="gene",
                right_on="id",
                validate="1:1",
                suffixes=("_x", ""),
            ).drop(columns="gene_x")
            logger.info(f"  {len(seed_gene_df)} seed genes matched with gene file:")
            print(seed_gene_df)

            if max_seed_genes is not None:
                seed_gene_df.sort_values("pval_corrected")
                seed_gene_df = seed_gene_df.head(max_seed_genes)
                assert len(seed_gene_df) <= max_seed_genes
                logger.info(
                    "  Restricted seed genes to "
                    f"max_seed_genes = {len(seed_gene_df)}:"
                )
                print(seed_gene_df)

            seed_gene_df = seed_gene_df.drop(
                columns=["pval", "pval_corrected"], errors="ignore"
            )
            seed_gene_df.to_parquet(seed_genes_out, engine="pyarrow")
            config["seed_gene_file"] = seed_genes_out

    with open(new_config_file, "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    cli()

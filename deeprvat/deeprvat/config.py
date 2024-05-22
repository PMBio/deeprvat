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
import deeprvat.deeprvat as deeprvat_dir
import pretrained_models as pretrained_dir
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

@cli.command()
@click.argument("config-file", type=click.Path(exists=True))
@click.argument("output-dir", type=click.Path(), default='.')
def create_main_config(
    config_file: str,
    output_dir: str,
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
    #Base Config
    with open(f'{os.path.dirname(deeprvat_dir.__file__)}/base_configurations.yaml') as f:
        base_config = yaml.safe_load(f)

    expected_input_keys = [
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
        "pl_trainer",
        "early_stopping",
        "n_repeats",
        "y_transformation",
        "cv_exp",
        "cv_path",
        "n_folds",
    ]

    full_config = base_config
            
    with open(config_file) as f:
        input_config = yaml.safe_load(f)

    #CV setup parameters
    if not input_config["cv_exp"]:
        logger.info("Not CV setup...removing CV pipeline parameters from config")
        to_remove = {"cv_path","n_folds"}
        expected_input_keys = [item for item in expected_input_keys if item not in to_remove]
        full_config["cv_exp"] = False
    else:
        full_config["cv_path"] = input_config["cv_path"]
        full_config["n_folds"] = input_config["n_folds"]
        full_config["cv_exp"] = True
    
    no_pretrain = True
    if "use_pretrained_models" in input_config:
        if input_config["use_pretrained_models"]:
            no_pretrain = False
            logger.info("Pretrained Model setup specified.")
            to_remove = {"pl_trainer","early_stopping"}
            expected_input_keys = [item for item in expected_input_keys if item not in to_remove]
            expected_input_keys.extend(["use_pretrained_models","model"])

            with open(f'{os.path.dirname(pretrained_dir.__file__)}/config.yaml') as f:
                pretrained_config = yaml.safe_load(f)
            
            for k in pretrained_config:
                input_config[k] = deepcopy(pretrained_config[k])


    if set(input_config.keys()) - set(expected_input_keys):
        extra_keys=set(input_config.keys()) - set(expected_input_keys)
        raise KeyError(("Unspecified key present in input YAML file. "
                        f"The follow extra keys are present: {extra_keys} "
                       "Please review DEEPRVAT_DIR/example/config/deeprvat_input_config.yaml for list of keys."))
 
    # Phenotypes
    full_config["phenotypes"] = input_config["phenotypes_for_association_testing"]
    full_config["training"]["phenotypes"] = input_config["phenotypes_for_training"]
    full_config["training_data"]["dataset_config"]["y_transformation"] = input_config["y_transformation"]
    full_config["assocation_testing_data"]["dataset_config"]["y_transformation"] = input_config["y_transformation"]
    # genotypes.h5    
    full_config["training_data"]["gt_file"] = input_config["gt_filename"]
    full_config["assocation_testing_data"]["gt_file"] = input_config["gt_filename"]
    # variants.parquet
    full_config["training_data"]["variant_file"] = input_config["variant_filename"]
    full_config["assocation_testing_data"]["variant_file"] = input_config["variant_filename"]
    # phenotypes.parquet
    full_config["training_data"]["dataset_config"]["phenotype_file"] = input_config["phenotype_filename"]
    full_config["assocation_testing_data"]["dataset_config"]["phenotype_file"] = input_config["phenotype_filename"]
    # annotations.parquet
    full_config["training_data"]["dataset_config"]["annotation_file"] = input_config["annotation_filename"]
    full_config["assocation_testing_data"]["dataset_config"]["annotation_file"] = input_config["annotation_filename"]
    # protein_coding_genes.parquet
    full_config["assocation_testing_data"]["dataset_config"]["gene_file"] = input_config["gene_filename"]
    full_config["assocation_testing_data"]["dataset_config"]["rare_embedding"]["config"]["gene_file"] = input_config["gene_filename"]
    # rare_variant_annotations
    full_config["training_data"]["dataset_config"]["rare_embedding"]["config"]["annotations"] = input_config["rare_variant_annotations"]
    full_config["assocation_testing_data"]["dataset_config"]["rare_embedding"]["config"]["annotations"] = input_config["rare_variant_annotations"]
    # variant annotations
    anno_list = deepcopy(input_config["rare_variant_annotations"])
    for i,k in enumerate(input_config["training_data_thresholds"].keys()):
        anno_list.insert(i+1,k)
    full_config["training_data"]["dataset_config"]["annotations"] = anno_list
    full_config["assocation_testing_data"]["dataset_config"]["annotations"] = anno_list
    # covariates
    full_config["training_data"]["dataset_config"]["x_phenotypes"] = input_config["covariates"]
    full_config["assocation_testing_data"]["dataset_config"]["x_phenotypes"] = input_config["covariates"]
    # Thresholds
    full_config["training_data"]["dataset_config"]["rare_embedding"]["config"]["thresholds"] = {}
    full_config["assocation_testing_data"]["dataset_config"]["rare_embedding"]["config"]["thresholds"] = {}
    for k,v in input_config["training_data_thresholds"].items():
        full_config["training_data"]["dataset_config"]["rare_embedding"]["config"]["thresholds"][k] = f"{k} {v}"
    for k,v in input_config["association_testing_data_thresholds"].items():
        full_config["assocation_testing_data"]["dataset_config"]["rare_embedding"]["config"]["thresholds"][k] = f"{k} {v}"
    # Baseline results
    full_config["baseline_results"]["options"] = input_config["seed_gene_results"]["options"]
    full_config["alpha"] = input_config["seed_gene_results"]["alpha"]
    #DeepRVAT model 
    full_config["n_repeats"] = input_config["n_repeats"]

    full_config["data"] = full_config["assocation_testing_data"]
    del full_config["assocation_testing_data"]
    
    if no_pretrain:
        # PL trainer
        full_config["pl_trainer"] = input_config["pl_trainer"]
        # Early Stopping
        full_config["early_stopping"] = input_config["early_stopping"]
    else:
        full_config["model"] = input_config["model"]

        with open(f"{os.path.dirname(pretrained_dir.__file__)}/deeprvat_config.yaml", "w") as f:
            yaml.dump(full_config, f)


    with open(f"{output_dir}/deeprvat_config.yaml", "w") as f:
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
            correction_method = config["baseline_results"].get("correction_method", None)
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
            alpha = config.get("alpha_seed_genes", config.get("alpha"))
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

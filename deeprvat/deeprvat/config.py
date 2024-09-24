import logging
from pprint import pprint
import sys
from typing import Optional, Tuple

import click
import pandas as pd
import yaml

from deeprvat.deeprvat.evaluate import pval_correction
from pathlib import Path
from copy import deepcopy

REPO_DIR = (Path(__file__).parent / "../..").resolve()


logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


def setup_logging(log_filename: str = "config_generate.log"):
    file_handler = logging.FileHandler(log_filename, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
    )
    logger.addHandler(file_handler)
    return file_handler


def load_yaml(file_path: str):
    with open(file_path) as f:
        return yaml.safe_load(f)


def update_defaults(base_config, input_config):
    """
    Updates base_config with values from input_config, for intersecting nested keys.

    Args:
        base_config (dict): base DeepRVAT configurations
        input_config (dict): user input DeepRVAt configurations

    Returns:
        dict: updated base_config based on any intersecting inputs from input_config
    """
    common_keys = set(base_config.keys()).intersection(input_config.keys())

    for k in common_keys:
        if isinstance(base_config[k], dict) and isinstance(input_config[k], dict):
            update_defaults(base_config[k], input_config[k])
        else:
            base_config[k] = input_config[k]

    return base_config


def handle_cv_options(input_config, full_config, expected_input_keys):
    if input_config.get("cv_options", {}).get("cv_exp", False):
        missing_keys = [
            key
            for key in ["cv_exp", "cv_path", "n_folds"]
            if key not in input_config["cv_options"]
        ]
        if missing_keys:
            raise KeyError(
                f"Missing keys {missing_keys} under config['cv_options'] \n\
                           Please review DEEPRVAT_DIR/example/config/deeprvat_input_config.yaml for list of keys."
            )
        full_config.update(
            {
                "cv_path": input_config["cv_options"]["cv_path"],
                "n_folds": input_config["cv_options"]["n_folds"],
                "cv_exp": True,
            }
        )
    else:
        logger.info("Not CV setup...removing CV pipeline parameters from config")
        full_config["cv_exp"] = False
        expected_input_keys.remove("cv_options")
        input_config.pop("cv_options", None)


def handle_regenie_options(input_config, full_config, expected_input_keys):
    if input_config.get("regenie_options", {}).get("regenie_exp", False):
        missing_keys = [
            key
            for key in ["regenie_exp", "step_1", "step_2"]
            if key not in input_config["regenie_options"]
        ]
        if missing_keys:
            raise KeyError(
                f"Missing keys {missing_keys} under config['regenie_options'] \n\
                           Please review DEEPRVAT_DIR/example/config/deeprvat_input_config.yaml for list of keys."
            )
        full_config.update(
            {
                "regenie_exp": True,
                "regenie_options": {
                    "step_1": input_config["regenie_options"]["step_1"],
                    "step_2": input_config["regenie_options"]["step_2"],
                },
                "gtf_file": input_config["regenie_options"]["gtf_file"],
            }
        )
    else:
        logger.info(
            "Not using REGENIE integration...removing REGENIE parameters from config"
        )
        full_config["regenie_exp"] = False
        expected_input_keys.remove("regenie_options")
        input_config.pop("regenie_options", None)


def handle_pretrained_models(input_config, expected_input_keys):
    if input_config.get("use_pretrained_models", False):
        logger.info("Pretrained Model setup specified.")
        to_remove = {"training", "phenotypes_for_training", "seed_gene_results"}
        for item in to_remove:
            expected_input_keys.remove(item)

        pretrained_model_path = Path(input_config["pretrained_model_path"])
        expected_input_keys.extend(
            ["use_pretrained_models", "model", "pretrained_model_path"]
        )

        pretrained_config = load_yaml(f"{pretrained_model_path}/model_config.yaml")

        required_keys = {
            "model",
            "rare_variant_annotations",
            "training_data_thresholds",
        }
        extra_keys = set(pretrained_config.keys()) - required_keys
        if extra_keys:
            raise KeyError(
                f"Unexpected key in pretrained_model_path/model_config.yaml file : {extra_keys} \n\
                            Please review DEEPRVAT_DIR/pretrained_models/model_config.yaml for expected list of keys."
            )
        logger.info("   Updating input config with keys from pretrained model config.")
        input_config.update(
            {
                "model": pretrained_config["model"],
                "rare_variant_annotations": pretrained_config[
                    "rare_variant_annotations"
                ],
                "training_data_thresholds": pretrained_config[
                    "training_data_thresholds"
                ],
            }
        )
        return True
    return False


def update_thresholds(input_config, full_config, train_only):
    if "MAF" not in input_config["training_data_thresholds"]:
        raise KeyError(
            f"Missing required MAF threshold in config['training_data_thresholds']"
        )
    if (
        not train_only
        and "MAF" not in input_config["association_testing_data_thresholds"]
    ):
        raise KeyError(
            f"Missing required MAF threshold in config['association_testing_data_thresholds']"
        )

    datasets = ["training_data", "association_testing_data"]
    if train_only:
        datasets.remove("association_testing_data")

    for data_type in datasets:
        anno_list = deepcopy(input_config["rare_variant_annotations"])
        full_config[data_type]["dataset_config"]["rare_embedding"]["config"][
            "thresholds"
        ] = {}
        threshold_key = f"{data_type}_thresholds"
        for i, (k, v) in enumerate(input_config[threshold_key].items()):
            full_config[data_type]["dataset_config"]["rare_embedding"]["config"][
                "thresholds"
            ][k] = f"{k} {v}"
            anno_list.insert(i + 1, k)
            if k == "MAF":
                full_config[data_type]["dataset_config"]["min_common_af"]["MAF"] = (
                    float(v[2:])
                )  # v is string like "< 1e-3"
        full_config[data_type]["dataset_config"]["annotations"] = anno_list


def update_full_config(input_config, full_config, train_only):
    base_mapping = {
        "gt_filename": "gt_file",  # genotypes.h5
        "variant_filename": "variant_file",
    }
    dataset_mapping = {
        "phenotype_filename": "phenotype_file",  # phenotypes.parquet
        "annotation_filename": "annotation_file",  # annotations.parquet
        "covariates": "x_phenotypes",
    }

    for key, value in base_mapping.items():
        full_config["training_data"][value] = input_config[key]
        if not train_only:
            full_config["association_testing_data"][value] = input_config[key]

    for key, value in dataset_mapping.items():
        full_config["training_data"]["dataset_config"][value] = input_config[key]
        if not train_only:
            full_config["association_testing_data"]["dataset_config"][value] = (
                input_config[key]
            )

    full_config["training_data"]["dataset_config"]["rare_embedding"]["config"][
        "annotations"
    ] = input_config["rare_variant_annotations"]
    full_config["association_testing_data"]["dataset_config"]["gene_file"] = (
        input_config["gene_filename"]
    )  # protein_coding_genes.parquet
    if not train_only:
        full_config["phenotypes"] = input_config["phenotypes_for_association_testing"]
        full_config["association_testing_data"]["dataset_config"]["rare_embedding"][
            "config"
        ]["gene_file"] = input_config["gene_filename"]
        full_config["association_testing_data"]["dataset_config"]["rare_embedding"][
            "config"
        ]["annotations"] = input_config["rare_variant_annotations"]


def validate_keys(input_config, expected_input_keys, optional_input_keys, base_config):
    input_keys_set = set(input_config.keys()) - set(optional_input_keys)
    expected_keys_set = set(expected_input_keys)
    updated_base_keys = set(base_config.keys()).intersection(input_config.keys())

    extra_keys = input_keys_set - expected_keys_set - updated_base_keys
    missing_keys = expected_keys_set - input_keys_set

    if extra_keys:
        raise KeyError(
            f"Extra key(s) present in input YAML file: {extra_keys} \n\
                        Please review DEEPRVAT_DIR/example/config/deeprvat_input_config.yaml for list of keys."
        )
    if missing_keys:
        raise KeyError(
            f"Missing key(s) in input YAML file: {missing_keys} \n\
                        Please review DEEPRVAT_DIR/example/config/deeprvat_input_config.yaml for list of keys."
        )


def create_main_config(
    config_file: str,
    output_dir: Optional[str] = ".",
    clobber: Optional[bool] = False,
):
    """
    Generates the necessary deeprvat_config.yaml file for running all pipelines.
    This function expects inputs as shown in the following config-file:
    - DEEPRVAT_DIR/example/deeprvat_input_config.yaml

    :param config_file: Path to directory of relevant config yaml file
    :type config_file: str
    :param output_dir: Path to directory where created deeprvat_config.yaml will be saved.
    :type output_dir: str
    :param clobber: Overwrite existing deeprvat_config.yaml, even if it is newer than config_file
    :type clobber: bool
    :return: Joined configuration file saved to deeprvat_config.yaml.
    """

    config_path = Path(config_file)
    output_path = Path(output_dir) / "deeprvat_config.yaml"
    if not output_path.exists():
        if not config_path.exists():
            raise ValueError(
                f"Neither input config {config_path} nor output config {output_path} exists"
            )
    else:
        if not config_path.exists():
            return
        elif config_path.stat().st_mtime > output_path.stat().st_mtime:
            logger.info("Generating deeprvat_config.yaml")
            logger.info(f"{output_path} is older than {config_path}, regenerating")
        else:
            if clobber:
                logger.info("Generating deeprvat_config.yaml")
                logger.warning(f"Overwriting newer file {output_path} as clobber=True")
            else:
                return

    file_handler = setup_logging()

    input_config = load_yaml(config_file)
    base_config = load_yaml(REPO_DIR / "deeprvat/deeprvat/base_configurations.yaml")
    full_config = deepcopy(update_defaults(base_config, input_config))

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
        "training",
        "n_repeats",
        "y_transformation",
        "evaluation",
        "cv_options",
        "regenie_options",
    ]

    optional_input_keys = [
        "deterministic",
    ]

    train_only = input_config.pop("training_only", False)
    if train_only:
        to_remove = [
            "phenotypes_for_association_testing",
            "association_testing_data_thresholds",
            "evaluation",
        ]
        for item in to_remove:
            expected_input_keys.remove(item)

    handle_cv_options(input_config, full_config, expected_input_keys)
    handle_regenie_options(input_config, full_config, expected_input_keys)
    pretrained_setup = handle_pretrained_models(input_config, expected_input_keys)

    if not pretrained_setup and "phenotypes_for_training" not in input_config:
        if train_only:
            raise KeyError("Must specify phenotypes_for_training in config file!")
        logger.info(
            "   Setting training phenotypes to be the same set as specified by phenotypes_for_association_testing."
        )
        input_config["phenotypes_for_training"] = input_config[
            "phenotypes_for_association_testing"
        ]

    if "y_transformation" in input_config:
        full_config["training_data"]["dataset_config"]["y_transformation"] = (
            input_config["y_transformation"]
        )
        if not train_only:
            full_config["association_testing_data"]["dataset_config"][
                "y_transformation"
            ] = input_config["y_transformation"]
    else:
        expected_input_keys.remove("y_transformation")

    validate_keys(input_config, expected_input_keys, optional_input_keys, base_config)
    update_thresholds(input_config, full_config, train_only)
    update_full_config(input_config, full_config, train_only)

    full_config["n_repeats"] = input_config["n_repeats"]
    full_config["deterministic"] = input_config.get("deterministic", False)

    if "sample_files" in input_config:
        for key in ["training", "association_testing"]:
            if key in input_config["sample_files"]:
                full_config[f"{key}_data"]["dataset_config"]["sample_file"] = (
                    input_config["sample_files"][key]
                )

    # Results evaluation parameters; alpha parameter for significance threshold
    if not train_only:
        full_config["evaluation"] = {
            "correction_method": input_config["evaluation"]["correction_method"],
            "alpha": input_config["evaluation"]["alpha"],
        }

    if pretrained_setup:
        full_config.update(
            {
                "model": input_config["model"],
                "pretrained_model_path": input_config["pretrained_model_path"],
            }
        )
    else:
        full_config["training"]["pl_trainer"] = input_config["training"]["pl_trainer"]
        full_config["training"]["early_stopping"] = input_config["training"][
            "early_stopping"
        ]
        full_config["training"]["phenotypes"] = {
            pheno: {} for pheno in input_config["phenotypes_for_training"]
        }
        # For each phenotype, you can optionally specify dictionary of = {"min_seed_genes": 3, "max_seed_genes": None, "pvalue_threshold": None}
        full_config["baseline_results"] = {
            "options": input_config["seed_gene_results"]["result_dirs"],
            "alpha_seed_genes": input_config["seed_gene_results"]["alpha_seed_genes"],
            "correction_method": input_config["seed_gene_results"]["correction_method"],
        }

    with open(output_path, "w") as f:
        yaml.dump(full_config, f)
        logger.info(
            f"Saving deeprvat_config.yaml to -- {output_dir}/deeprvat_config.yaml --"
        )

    logger.removeHandler(file_handler)
    file_handler.close()


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
    # Set stdout file
    file_handler = logging.FileHandler("config_generate.log", mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        )
    )
    logger.addHandler(file_handler)

    with open(config_file) as f:
        input_config = yaml.safe_load(f)

    # Base Config
    with open(
        REPO_DIR / "deeprvat/seed_gene_discovery/seed_gene_base_configurations.yaml"
    ) as f:
        base_config = yaml.safe_load(f)

    full_config = base_config

    expected_input_keys = [
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
    full_config["data"]["dataset_config"]["phenotype_file"] = input_config[
        "phenotype_filename"
    ]
    # annotations.parquet
    full_config["data"]["dataset_config"]["annotation_file"] = input_config[
        "annotation_filename"
    ]
    # protein_coding_genes.parquet
    full_config["data"]["dataset_config"]["gene_file"] = input_config["gene_filename"]
    full_config["data"]["dataset_config"]["rare_embedding"]["config"]["gene_file"] = (
        input_config["gene_filename"]
    )
    # X_phenotypes (covariates)
    full_config["data"]["dataset_config"]["x_phenotypes"] = input_config["covariates"]
    # Annotations
    full_config["data"]["dataset_config"]["annotations"] = input_config["annotations"]
    full_config["data"]["dataset_config"]["rare_embedding"]["config"]["annotations"] = (
        input_config["annotations"]
    )
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
    full_config["data"]["dataset_config"]["standardize_xpheno"] = input_config[
        "dataset_config"
    ]["standardize_xpheno"]
    if "y_transformation" in input_config["dataset_config"]:
        full_config["data"]["dataset_config"]["y_transformation"] = input_config[
            "dataset_config"
        ]["y_transformation"]
    full_config["data"]["dataset_config"]["standardize_xpheno"] = input_config[
        "dataset_config"
    ]["standardize_xpheno"]

    if "sample_file" in input_config:
        logger.info("Adding in subset sample file for seed-gene-discovery.")
        full_config["data"]["dataset_config"]["sample_file"] = input_config[
            "sample_file"
        ]

    logger.info(
        f"Saving sg_discovery_config.yaml to -- {output_dir}/sg_discovery_config.yaml --"
    )
    with open(f"{output_dir}/sg_discovery_config.yaml", "w") as f:
        yaml.dump(full_config, f)

    # close out config log file
    logger.removeHandler(file_handler)
    file_handler.close()


@cli.command()
@click.option("--association-only", is_flag=True)
@click.option("--phenotype", type=str)
@click.option("--baseline-results", type=click.Path(exists=True), multiple=True)
@click.option("--baseline-results-out", type=click.Path())
@click.option("--seed-genes-out", type=click.Path())
# @click.option("--regenie-options", type=str, multiple=True)
@click.argument("old_config_file", type=click.Path(exists=True))
@click.argument("new_config_file", type=click.Path())
def update_config(
    association_only: bool,
    phenotype: Optional[str],
    baseline_results: Tuple[str],
    baseline_results_out: Optional[str],
    # regenie_options: Optional[Tuple[str]],
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
        config["association_testing_data"]["dataset_config"]["y_phenotypes"] = [
            phenotype
        ]
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
            seed_config = config["training"]["phenotypes"][phenotype]
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
            logger.info("  Reading baseline results from:")
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
            if config["baseline_results"].get("alpha_seed_genes", False):
                alpha = config["baseline_results"]["alpha_seed_genes"]
            else:
                alpha = config["evaluation"].get("alpha")
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
                    )
                else:
                    baseline_df = baseline_df.query("significant")
            else:
                if threshold is not None:
                    baseline_temp = baseline_df.query("pval_corrected < @threshold")
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
                config["association_testing_data"]["dataset_config"]["gene_file"],
                engine="pyarrow",
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

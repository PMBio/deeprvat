from pathlib import Path
from textwrap import indent, wrap
import numpy as np
import sys
import logging
from pprint import pformat
from typing import Dict, Iterable, List, Union

import h5py
import pandas as pd
import yaml


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)


def check_needed(checked_indicator: Union[str, Path], files: List[Union[str, Path]]):
    checked_time = (
        Path(checked_indicator).stat().st_mtime
        if Path(checked_indicator).exists()
        else float("-inf")
    )
    last_file_change = max([Path(x).stat().st_mtime for x in files])
    return last_file_change < checked_time


def column_check(
    present: Iterable[str],
    required: Iterable[str],
    col_type: str,
    required_by: str,
    file: Union[str, Path],
):
    missing_cols = set(required) - set(present)
    check_pass = len(missing_cols) == 0
    if not check_pass:
        logger.error(
            f"The following {col_type} are specified in the configuration under {required_by} "
            f"but are missing from {file}:\n" + pformat(missing_cols)
        )

    return check_pass


def nan_check(annotations: pd.DataFrame, required: Iterable[str], required_by: str):
    check = annotations[required].isna().any()
    nan_cols = set(check[check])
    check_pass = len(nan_cols) == 0
    if not check_pass:
        logger.error(
            f"The following annotations are specified in the configuration under {required_by} "
            f"but have NaN values:" + pformat(nan_cols)
        )

    return check_pass


def check_annotations(
    config: Dict,
    annotation_file: Union[str, Path],
    variant_file: Union[str, Path],
    gene_file: Union[str, Path],
):
    annotations = pd.read_parquet(annotation_file)

    # Check that all annotation columns are present
    cols_present = set(annotations.columns)
    anno_cols = config["rare_variant_annotations"]
    assoc_thresh_cols = config["association_testing_data_thresholds"].keys()
    training_thresh_cols = config["training_data_thresholds"].keys()
    column_check_pass = (
        column_check(
            cols_present,
            anno_cols,
            "annotations",
            "rare_variant_annotations",
            annotation_file,
        )
        and column_check(
            cols_present,
            assoc_thresh_cols,
            "annotations",
            "association_testing_data_thresholds",
            annotation_file,
        )
        and column_check(
            cols_present,
            training_thresh_cols,
            "annotations",
            "training_data_thresholds",
            annotation_file,
        )
    )

    # Check the gene_id column
    gene_id_check_pass = "gene_id" in cols_present
    if not gene_id_check_pass:
        logger.error(f"The gene_id column is missing from {annotation_file}")
    else:
        anno_genes = annotations["gene_id"]
        if annotations["gene_id"].dtype == np.dtype("O"):
            anno_genes = anno_genes.explode()
        genes = pd.read_parquet(gene_file)

        genes_wo_variants = set(genes["id"]) - set(anno_genes)
        if not len(genes_wo_variants) > 0:
            logger.warning(
                f"Gene IDs found in {gene_file} that have no variants in {annotation_file}. These genes will be ignored"
            )

        unknown_gene_ids = set(anno_genes) - set(genes["id"])
        if len(unknown_gene_ids) > 0:
            logger.warning(
                f"Gene IDs found in {annotation_file} that are not present in {gene_file}. These variants will be ignored."
            )

    # Check that all variants are present
    variants = pd.read_parquet(variant_file)
    missing_variant_ids = set(variants["id"]) - set(annotations["id"])
    variant_check_pass = (
        annotations.query("id in @missing_variant_ids")["gene_id"].isna().all()
    )
    if not variant_check_pass:
        logger.error(
            f"Some variants from {variant_file} are missing in {annotation_file}"
        )

    # Check for NaNs in required columns
    nan_check_pass = (
        nan_check(
            annotations,
            anno_cols,
            "rare_variant_annotations",
        )
        and nan_check(
            annotations,
            assoc_thresh_cols,
            "association_testing_data_thresholds",
        )
        and nan_check(
            annotations,
            training_thresh_cols,
            "training_data_thresholds",
        )
    )

    return (
        column_check_pass
        and gene_id_check_pass
        and variant_check_pass
        and nan_check_pass
    )


def check_phenotypes(
    config: Dict,
    phenotype_file: Union[str, Path],
    gt_file: Union[str, Path],
):
    phenotypes = pd.read_parquet(phenotype_file)

    # Check that all phenotypes and covariates are present
    cols_present = phenotypes.columns
    training_phenos = config["phenotypes_for_training"]
    assoc_phenos = config["phenotypes_for_association_testing"]
    covariates = config["covariates"]
    column_check_pass = (
        column_check(
            cols_present,
            training_phenos,
            "phenotypes",
            "phenotypes_for_training",
            phenotype_file,
        )
        and column_check(
            cols_present,
            assoc_phenos,
            "phenotypes",
            "phenotypes_for_association_testing",
            phenotype_file,
        )
        and column_check(
            cols_present,
            covariates,
            "covariates",
            "covariates",
            phenotype_file,
        )
    )

    # Check for nonempty overlap between samples in genotype and phenotype files
    pheno_samples = set(phenotypes.index)
    with h5py.File(gt_file, "r") as f:
        gt_samples = set([s.decode("utf-8") for s in f["samples"][:]])

    common_samples = pheno_samples.intersection(gt_samples)
    sample_check = len(common_samples) > 0
    if not sample_check:
        logger.error(
            f"Genotype file {gt_file} and phenotype file {phenotype_file} "
            "have no samples in common."
        )

    # Do the rest of the checks only on the samples in common
    phenotypes = phenotypes.loc[list(common_samples)]

    # For each phenotype, check:
    # 1. At least two distinct non-NaN values are present (e.g., not all samples are controls)
    # 2. For every sample with a measured phenotypes, all covariates are not NaN
    pheno_check = True
    covariate_check = True
    for p in training_phenos + assoc_phenos:
        non_nan_mask = phenotypes[p].notna()
        non_nan = phenotypes.loc[non_nan_mask, p]
        if (non_nan == non_nan.iloc[0]).all():
            pheno_check = False
            logging.error(
                f"Phenotype {p} takes on only one distinct non-NaN value "
                "among samples shared between genotype file {gt_file} "
                "and phenotype file {phenotype_file}."
            )

        if phenotypes.loc[non_nan_mask, covariates].isna().any(axis=None):
            covariate_check = False
            logging.error(
                "For phenotype {p}, among samples shared between genotype file {gt_file} "
                "and phenotype file {phenotype_file}, "
                f"some covariates have missing values for samples with a measured phenotype."
            )

    return column_check_pass and sample_check and pheno_check and covariate_check


def check_input_data(config_file: Union[str, Path]):
    logger.info("Sanity checking data...")

    if not Path(config_file).exists():
        raise RuntimeError(
            indent(
                "\n".join(
                    wrap(
                        (
                            f"Input config file {config_file} not found. "
                            "Sanity checking cannot be performed. "
                            "If you're sure of what you're doing, "
                            'you can run snakemake with "--config skip_sanity_check=True" '
                            "to run the pipeline anyway."
                        ),
                        width=76,
                    )
                ),
                "!!  ",
            )
        )

    with open(config_file) as f:
        config = yaml.safe_load(f)

    # config_modification_time = Path(config_file).stat().st_mtime
    annotation_file = config["annotation_filename"]
    variant_file = config["variant_filename"]
    gene_file = config["gene_filename"]

    checked_indicator = ".files_checked"

    if check_needed(
        checked_indicator,
        [
            config_file,
            annotation_file,
            variant_file,
            gene_file,
        ],
    ):
        logger.info("Checking variants and annotations...")
        anno_check_pass = check_annotations(
            config, annotation_file, variant_file, gene_file
        )
    else:
        anno_check_pass = True

    phenotype_file = config["phenotype_filename"]
    gt_file = config["gt_filename"]

    if check_needed(
        checked_indicator,
        [
            config_file,
            phenotype_file,
            gt_file,
        ],
    ):
        logger.info("Checking genotypes and phenotypes...")
        pheno_check_pass = check_phenotypes(config, phenotype_file, gt_file)
    else:
        pheno_check_pass = True

    check_pass = anno_check_pass and pheno_check_pass

    if not check_pass:
        raise RuntimeError(
            indent(
                "\n".join(
                    wrap(
                        (
                            "Data sanity checks failed, meaning the pipeline will "
                            "very likely throw an error at some step. "
                            "Please address all the errors above and try again. "
                            "Alternatively, if you're sure of what you're doing, "
                            'you can run snakemake with "--config skip_sanity_check=True" '
                            "to run the pipeline anyway."
                        ),
                        width=76,
                    )
                ),
                "!!  ",
            )
        )
    else:
        Path(checked_indicator).touch()
        logger.info("Sanity checking complete.")

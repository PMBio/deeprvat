import logging
import pickle
import sys
import os
from pathlib import Path
from typing import Dict

import click
import pandas as pd
import plotnine as p9
import numpy as np
import scipy
import yaml
from deeprvat.utils import pval_correction

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# TODO EAC threshold as parameter
def evaluate_(associations: Dict[str, pd.DataFrame], alpha: float):
    phenotypes = associations["phenotype"].unique()
    all_evaluations = {p: {} for p in phenotypes}
    all_significant = {p: {} for p in phenotypes}
    all_metrics = {p: {} for p in phenotypes}
    all_plots = {p: {} for p in phenotypes}

    for pheno in phenotypes:
        logger.info(f"Evaluation results for {pheno}:")
        result = associations.query("phenotype == @pheno")
        metrics = all_metrics[pheno]
        plots = all_plots[pheno]

        result["-log10pval"] = -np.log10(result["pval"])

        pval_correction_types = {"FDR": "", "Bonferroni": "_bf"}
        corrected_results = []
        for correction_type, sig_col_suffix in pval_correction_types.items():
            logger.info(f"Results for {correction_type} correction")
            logger.info("Only using genes with EAC > 50")
            result = result.query("EAC > 50")
            if result["pval"].isna().sum() > 0:
                logger.info(
                    "Attention: still NA pvals after EAC filtering\
                            Removing remaining NA pvals"
                )
                result = result.dropna(subset=["pval"])
            corrected_result = pval_correction(
                result, alpha, correction_type=correction_type
            )
            corrected_result["-log10pval_corrected"] = -np.log10(
                corrected_result["pval_corrected"]
            )
            corrected_result["correction_method"] = correction_type
            corrected_results.append(corrected_result)

            sig = corrected_result.query("significant")
            n_sig = len(sig)
            logger.info(f"Significant genes: {n_sig}")
            metrics[f"significant{sig_col_suffix}"] = n_sig
            print(f"Number of gene/model pairs: {n_sig}")
            n_sig_unique = sig["gene"].unique().shape[0]
            metrics[f"significant_unique{sig_col_suffix}"] = n_sig_unique
            print(f"Number of unique significant genes: {n_sig_unique}")

        corrected_results = pd.concat(corrected_results)
        all_evaluations[pheno] = corrected_results

        all_sig = corrected_results.query("significant")
        all_significant[pheno] = all_sig

        print(all_sig)

        # Genomic inflation factor

        chisq = scipy.stats.chi2.ppf(1 - result["pval"], 1)
        lambda_ = np.median(chisq) / scipy.stats.chi2.ppf(0.5, 1)
        metrics["lambda"] = lambda_

        # Q-Q plot
        plots["qqplot"] = qqplot(corrected_results)

    return all_significant, all_evaluations, all_metrics, all_plots


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config-file", type=click.Path(exists=True))
@click.argument("train-associations", type=click.Path(exists=True), nargs=-1)
@click.argument("out-file", type=click.Path())
def evaluate(
    train_associations: str, config_file: str, out_file: str
):  # val_associations: str,
    out_dir = os.path.dirname(out_file)
    logger.info("Starting evaluation")
    with open(config_file) as f:
        config = yaml.safe_load(f)

    test_names = ["-".join(path.split("/")[-4:-2]) for path in train_associations]
    logger.info(f"Concatenating test names: {test_names}")
    all_associations = (
        pd.concat(
            [pd.read_parquet(r, engine="pyarrow") for r in train_associations],
            keys=test_names,
        )
        .reset_index(level=0)
        .rename(columns={"level_0": "method"})
    )

    associations = all_associations[["gene", "pval", "EAC", "method"]].copy()

    pheno = config["data"]["dataset_config"]["y_phenotypes"][0]
    associations["phenotype"] = pheno

    alpha = config["alpha"]

    significant, evaluations, metrics, plots = evaluate_(associations, alpha)

    logger.info("Saving results")
    out_dir = Path(out_dir)
    evaluations[pheno].to_parquet(out_file)

    with open(out_dir / "metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    all_associations.to_parquet(f"{out_dir}/all_associations.parquet")

    for p_name, plot in plots[pheno].items():
        p9.save_as_pdf_pages([plot], filename=str(out_dir / f"{p_name}_testing.pdf"))


def qqplot(df):
    df = df.copy()

    df = df.sort_values("pval")
    df["-log10pval_expected"] = -np.log10(np.arange(1, len(df) + 1) / len(df))

    if "-log10pval" not in df.columns:
        df["-log10pval"] = -np.log10(df["pval"])

    aes_kwargs = dict(x="-log10pval_expected", y="-log10pval")
    plot = (
        p9.ggplot(df, p9.aes(**aes_kwargs, color="significant"))
        + p9.labs(title="QQ plot (uncorrected p-values)")
        + p9.geom_abline(intercept=0, slope=1, color="red")
        + p9.geom_point()
        + p9.theme(legend_position="none")
    )
    return plot


if __name__ == "__main__":
    cli()

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from itertools import combinations
import random
import os

import click
import numpy as np
import pandas as pd
import yaml
from seak.cct import cct

from deeprvat.utils import pval_correction, bfcorrect_df

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

BASELINE_GROUPS = [
    f"baseline_{t}_{m}" for t in ["missense", "plof"] for m in ["burden", "skat"]
]

BURDEN_SKAT_RENAME = {"burden": "Burden", "skat": "SKAT"}
VARIANT_TYPE_RENAME = {"missense": "missense", "plof": "pLOF"}
METHOD_NAMES = {
    f"baseline_{t}_{m}": f"{BURDEN_SKAT_RENAME[m]} {VARIANT_TYPE_RENAME[t]}"
    for t in ["missense", "plof"]
    for m in ["burden", "skat"]
}
METHOD_NAMES.update({"baseline_combined": "Burden/SKAT combined"})


def count_unique(result: pd.DataFrame, query: str):
    return len(result.query(query)["gene"].unique())


def get_baseline(
    paths,
    experiment_name,
    deeprvat_genes,
    phenotype=None,
    min_eaf=50,
    alpha: float = 0.05,
    correction_method: str = "Bonferroni",
) -> pd.DataFrame:
    baseline = pd.concat([pd.read_parquet(p) for p in paths])
    if "EAC" in baseline.columns:
        print("Applying EAF filter")

        baseline = baseline.query("EAC >= @min_eaf")
    baseline["gene"] = baseline["gene"].astype(int)
    if ("phenotype" not in baseline.columns) & (phenotype is not None):
        baseline["phenotype"] = phenotype
    baseline = baseline.dropna(subset=["pval"])
    assert "phenotype" in baseline.columns

    df = pval_correction(baseline, alpha, correction_type=correction_method)
    df["experiment_group"] = experiment_name
    df["correction_method"] = correction_method
    df["experiment"] = "Baseline"

    return df


def get_baseline_results(
    config: Dict,
    pheno,
    deeprvat_genes: np.ndarray,
    alpha: float = 0.05,
    correction_method: str = "Bonferroni",
):
    min_eaf = config.get("min_eaf_baseline", 50)

    result_list = []
    baseline_paths = {
        (
            r["type"].split("/")[0],
            r["type"].split("/")[1],
        ): f"{r['base']}/{pheno}/{r['type']}/eval/burden_associations.parquet"
        for r in config["baseline_results"]
    }
    logger.info(f"reading baseline from {baseline_paths}")
    for (t, m), p in baseline_paths.items():
        if os.path.exists(p):
            result_list.append(
                get_baseline(
                    [p],
                    f"baseline_{t}_{m}",
                    deeprvat_genes,
                    phenotype=pheno,
                    min_eaf=min_eaf,
                    alpha=alpha,
                    correction_method=correction_method,
                )
            )
        else:
            logger.warning(f"Baseline path {p} doesn't exist")
    if len(result_list) > 0:
        res = pd.concat(result_list)
    else:
        logger.warning("No baseline data set existed. Returning empty data frame")
        res = pd.DataFrame()
    return res


def combine_results(
    deeprvat_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    correction_method: str = "Bonferroni",
    alpha: float = 0.05,
    combine_pval: str = "Bonferroni",
):
    baseline_original = baseline_results.copy()

    baseline_original["Discovery type"] = "Baseline"
    deeprvat_results["Discovery type"] = "DeepRVAT discovery"

    deeprvat_results = pval_correction(
        deeprvat_results, alpha, correction_type=correction_method
    )

    baseline_combined = baseline_original.copy()
    baseline_combined["experiment_group"] = "baseline_combined"

    if len(baseline_original) > 0:
        if combine_pval is not None:
            print("Aggregating baseline pvalues to one pvalue per gene")
            baseline_combined = aggregate_pvals_per_gene(
                baseline_combined, combine_pval
            )
            # should only be one pval per gene left
            assert baseline_combined.groupby("gene").size().unique() == np.array([1])
        baseline_combined = pval_correction(
            baseline_combined, alpha, correction_type=correction_method
        )
        baseline_original_corrected = pd.DataFrame()
        for method in baseline_original["experiment_group"].unique():
            this_corrected = pval_correction(
                baseline_original.copy().query("experiment_group == @method"),
                alpha,
                correction_type=correction_method,
            )
            baseline_original_corrected = pd.concat(
                [baseline_original_corrected, this_corrected]
            )
        # just for sanity check
        logger.info("Number of tests for each baseline method")
        baseline_original_corrected["n_tests"] = (
            baseline_original_corrected["pval_corrected"]
            / baseline_original_corrected["pval"]
        )
        logger.info(
            baseline_original_corrected.groupby("experiment_group")["n_tests"].unique()
        )
        baseline_original_corrected = baseline_original_corrected.drop(
            columns="n_tests"
        )
        ######
    else:
        baseline_original_corrected = baseline_original

    deeprvat_results["experiment"] = "DeepRVAT"
    deeprvat_results["experiment_group"] = "DeepRVAT"
    deeprvat_results["correction_method"] = correction_method

    combined = pd.concat(
        [deeprvat_results, baseline_original_corrected, baseline_combined]
    )

    combined["-log10pval"] = -np.log10(combined["pval"])

    combined["Discovery type"] = pd.Categorical(
        combined["Discovery type"], ["DeepRVAT discovery", "Baseline"]
    )
    return combined.astype({"significant": bool})


def get_pvals(results, method_mapping=None, phenotype_mapping={}):
    results = results.copy()

    pvals = results[
        [
            "phenotype",
            "gene",
            "experiment_group",
            "Discovery type",
            "pval",
            "-log10pval",
            "pval_corrected",
            "significant",
        ]
    ]

    pvals = pvals.rename(columns={"experiment_group": "Method",})
    if method_mapping is not None:
        pvals["Method"] = pvals["Method"].apply(
            lambda x: method_mapping[x] if x in method_mapping else x
        )

    if phenotype_mapping is not None:
        pvals["phenotype"] = pvals["phenotype"].apply(
            lambda x: (
                phenotype_mapping[x]
                if x in phenotype_mapping
                else " ".join(x.split("_"))
            )
        )

    return pvals


def min_Bonferroni_aggregate(pvals):
    pval = min(pvals * len(pvals))
    return pval


def aggregate_pvals_per_gene(df, agg_method):
    grouping_cols = [
        "phenotype",
        "gene",
        "experiment",
        "experiment_group",
        "repeat_combi",
        "correction_method",
    ]
    grouping_cols = list(set(grouping_cols).intersection(set(df.columns)))
    select_cols = grouping_cols + ["pval"]
    agg_results = df.copy()[select_cols]
    print(f"aggregating pvalues using grouping cols {grouping_cols}")
    agg_results = agg_results.groupby(grouping_cols, dropna=False)
    if agg_method == "Bonferroni":
        print("using Bonferroni")
        agg_results = agg_results.agg(min_Bonferroni_aggregate).reset_index()
    elif agg_method == "cct":
        print("using cct")
        agg_results = agg_results.agg(cct).reset_index()
    else:
        raise ValueError(f"Unknown agg_method type: {agg_method}. ")
    return agg_results


def process_results(
    results: pd.DataFrame,
    alpha: float = 0.05,
    correction_method: str = "Bonferroni",
    combine_pval: str = "Bonferroni",
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # TODO change this query!
    deeprvat_results = results.query('experiment_group == "DeepRVAT"')

    assert (deeprvat_results.groupby("gene").size() == 1).all()
    baseline_results = results.query("experiment_group in @BASELINE_GROUPS")
    if "correction_method" in baseline_results.columns:
        # if use_baseline_results is not True the correction_method column is not in results
        baseline_results = results.query("correction_method == @correction_method")

    combined_results = combine_results(
        deeprvat_results,
        baseline_results,
        correction_method=correction_method,
        alpha=alpha,
        combine_pval=combine_pval,
    )

    all_pvals = get_pvals(combined_results, method_mapping=METHOD_NAMES)

    significant = all_pvals.query("significant")
    significant = significant.sort_values(
        ["phenotype", "gene", "Method", "Discovery type", "pval"],
        ascending=[True, True, True, False, True],
    )
    significant = significant.drop_duplicates(subset=["phenotype", "gene", "Method"])

    return significant, all_pvals


def evaluate_(
    associations: pd.DataFrame,
    alpha: float,
    baseline_results: Optional[pd.DataFrame] = None,
    debug: bool = False,
    correction_method: str = "Bonferroni",
    combine_pval: str = "Bonferroni",
):

    logger.info("Evaluation results:")
    results = pd.DataFrame()
    # TODO change this!
    n_repeats = (
        1  # TODO maybe completely drop this (we don't need any filtering any more
    )
    # we just use the entire data frame)
    rep_str = f"{results} repeats"
    repeat_mask = (
        associations["model"].str.split("_").apply(lambda x: x[-1]).astype(int)
        < n_repeats
    )
    results = associations[repeat_mask].copy()

    results["experiment"] = "DeepRVAT"

    ########### change until here ##################
    results["-log10pval"] = -np.log10(results["pval"])
    results["experiment_group"] = "DeepRVAT"

    results = pd.concat([results, baseline_results])

    significant, all_pvalues = process_results(
        results,
        alpha=alpha,
        correction_method=correction_method,
        combine_pval=combine_pval,
    )
    return significant, all_pvalues


# @cli.command()
@click.command()
@click.option("--debug", is_flag=True)
@click.option("--phenotype", type=str)
@click.option("--use-baseline-results", is_flag=True)
@click.option("--correction-method", type=str, default="Bonferroni")
@click.option(
    "--combine-pval", type=str, default="Bonferroni"
)  # Bonferroni min pval per gene for multiple baseline tests
@click.argument("association-files", type=click.Path(exists=True), nargs=-1)
@click.argument("config-file", type=click.Path(exists=True))
@click.argument("out-dir", type=click.Path())
def evaluate(
    debug: bool,
    phenotype: Optional[str],
    use_baseline_results: bool,
    correction_method: str,
    association_files: Tuple[str],
    config_file: str,
    out_dir: str,
    combine_pval,
):

    with open(config_file) as f:
        config = yaml.safe_load(f)
    associations = pd.concat(
        [pd.read_parquet(f, engine="pyarrow") for f in association_files]
    )
    logger.info("Associations loaded")
    pheno = (
        phenotype
        if phenotype is not None
        else config["data"]["dataset_config"]["y_phenotypes"][0]
    )
    associations["phenotype"] = pheno

    alpha = config["alpha"]

    if use_baseline_results:
        logger.info("Reading baseline results")
        deeprvat_genes = associations["gene"].unique()
        baseline_results = get_baseline_results(
            config,
            pheno,
            deeprvat_genes,
            alpha=alpha,
            correction_method=correction_method,
        )
    else:
        baseline_results = pd.DataFrame()
    significant, all_pvals = evaluate_(
        associations,
        alpha,
        baseline_results=baseline_results,
        correction_method=correction_method,
        debug=debug,
        combine_pval=combine_pval,
    )
    logger.info("DeepRVAT discvoeries:")
    logger.info(significant.query('Method == "DeepRVAT"'))
    logger.info("Saving results")
    out_path = Path(out_dir)
    significant.to_parquet(out_path / f"significant.parquet", engine="pyarrow")
    all_pvals.to_parquet(out_path / f"all_results.parquet", engine="pyarrow")


if __name__ == "__main__":
    evaluate()

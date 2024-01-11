import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from itertools import combinations
import random

import click
import numpy as np
import pandas as pd
import yaml

from deeprvat.utils import pval_correction

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
    correction_method: str = "FDR",
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
    df["correction_method"] = "FDR"
    df["experiment"] = "Baseline"
    
    return df


def get_baseline_results(
    config: Dict,
    pheno,
    deeprvat_genes: np.ndarray,
    alpha: float = 0.05,
    correction_method: str = "FDR",
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
    logger.info(f'reading baseline from {baseline_paths}')
    for (t, m), p in baseline_paths.items():
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

    return pd.concat(result_list)


def combine_results(
    deeprvat_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    correction_method: str = "FDR",
    alpha: float = 0.05,
):
    baseline_original = baseline_results.copy()

    baseline_original["Discovery type"] = "Baseline"
    deeprvat_results["Discovery type"] = "New DeepRVAT discovery"
    baseline_results["Discovery type"] = "Seed gene"
    combined_results = pd.concat([deeprvat_results, baseline_results])

    combined = pval_correction(
        combined_results, alpha, correction_type=correction_method
    )

    baseline_combined = baseline_original.copy()
    baseline_combined["experiment_group"] = "baseline_combined"
    baseline_combined = pval_correction(
        baseline_combined, alpha, correction_type=correction_method
    )

    combined["experiment"] = "DeepRVAT"
    combined["experiment_group"] = "DeepRVAT"
    combined["correction_method"] = correction_method

    combined = pd.concat([combined, baseline_original, baseline_combined])

    combined["-log10pval"] = -np.log10(combined["pval"])

    combined["Discovery type"] = pd.Categorical(
        combined["Discovery type"], ["New DeepRVAT discovery", "Seed gene", "Baseline"]
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

    pvals = pvals.rename(
        columns={
            "experiment_group": "Method",
        }
    )
    if method_mapping is not None:
        pvals["Method"] = pvals["Method"].apply(
            lambda x: method_mapping[x] if x in method_mapping else x
        )

    if phenotype_mapping is not None:
        pvals["phenotype"] = pvals["phenotype"].apply(
            lambda x: phenotype_mapping[x]
            if x in phenotype_mapping
            else " ".join(x.split("_"))
        )

    return pvals


def process_results(
    results: pd.DataFrame,
    n_repeats: int = 6,
    alpha: float = 0.05,
    correction_method: str = "FDR",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    deeprvat_results = results.query(
        f'experiment == "DeepRVAT ({n_repeats} repeats)"'
        ' and experiment_group == "DeepRVAT"'
    )

    assert (deeprvat_results.groupby('gene').size() == n_repeats).all()
    baseline_results = results.query(
        "experiment_group in @BASELINE_GROUPS")
    if "correction_method" in baseline_results.columns:
        # if use_seed_gene is not True the correction_method column is not in results
        baseline_results = results.query("correction_method == @correction_method")

    combined_results = combine_results(
        deeprvat_results,
        baseline_results,
        correction_method=correction_method,
        alpha=alpha,
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
    seed_genes: Optional[pd.DataFrame],
    max_repeat_combis: int,
    repeats_to_analyze: int,
    repeats: Optional[int] = None,
    baseline_results: Optional[pd.DataFrame] = None,
    debug: bool = False,
    correction_method: str = "FDR",
):
    if seed_genes is not None:
        seed_gene_ids = seed_genes["id"]
        associations = associations.query("gene not in @seed_gene_ids")

    n_total_repeats = (
        repeats
        if repeats is not None
        else associations["model"]
        .str.split("_")
        .apply(lambda x: x[-1])
        .astype(int)
        .max()
        + 1
    )
    if debug:
        n_total_repeats = min(n_total_repeats, 2)

    logger.info("Evaluation results:")
    logger.info(f"Analyzing results for {repeats_to_analyze} repeats")
    
    if max_repeat_combis > 1:
        random.seed(10) #to ensoure that the same values for all_repeat_combinations
        # when running this script for each phenotype
        significant = pd.DataFrame()
        all_pvalues = pd.DataFrame()
        logger.info(f"Maximum number of repeat combinations that will be analyzed {max_repeat_combis}")
        all_repeat_combinations = list(combinations(range(n_total_repeats), repeats_to_analyze))#[:max_repeat_combis]
        n_combis_to_sample = min(max_repeat_combis, len(all_repeat_combinations))
        all_repeat_combinations = random.sample(all_repeat_combinations, n_combis_to_sample)
        for combi in all_repeat_combinations:
            logger.info(f'Repeat combination {combi}')
            rep_str = [f"repeat_{i}" for i in combi]
            repeat_mask = [i in rep_str for i in associations["model"]]

            this_result = associations[repeat_mask].copy()
            experiment_name = f"DeepRVAT ({repeats_to_analyze} repeats)"
            this_result["experiment"] = experiment_name
            this_result["repeat_combi"] =  '-'.join(str(x) for x in combi)

            this_result["-log10pval"] = -np.log10(this_result["pval"])
            this_result["experiment_group"] = "DeepRVAT"
            this_result = pd.concat([this_result, baseline_results])
            
            this_significant, this_all_pvalues = process_results(
                this_result,
                n_repeats=repeats_to_analyze,
                alpha=alpha,
                correction_method=correction_method,
            )
            combi_str = '-'.join(str(x) for x in combi)
            this_significant = this_significant.assign(repeats = repeats_to_analyze,
                repeat_combination = combi_str)
            this_all_pvalues = this_all_pvalues.assign(repeats = repeats_to_analyze,
                repeat_combination = combi_str)
            
            significant = pd.concat([significant, this_significant])
            all_pvalues = pd.concat([all_pvalues, this_all_pvalues])

    else:
        rep_str = [f"repeat_{i}" for i in range(repeats_to_analyze)]
        logger.info(f'Only evaluating for one repeat combination range({rep_str})')
        repeat_mask = [i in rep_str for i in associations["model"]]
        this_result = associations[repeat_mask].copy()
        experiment_name = f"DeepRVAT ({repeats_to_analyze} repeats)"
        this_result["experiment"] = experiment_name

        this_result["-log10pval"] = -np.log10(this_result["pval"])
        this_result["experiment_group"] = "DeepRVAT"
        this_result = pd.concat([this_result, baseline_results])
            
        significant, all_pvalues = process_results(
            this_result,
            n_repeats=repeats_to_analyze,
            alpha=alpha,
            correction_method=correction_method,
        )

    return significant, all_pvalues


# @cli.command()
@click.command()
@click.option("--debug", is_flag=True)
@click.option("--phenotype", type=str)
@click.option("--use-seed-genes", is_flag=True)
@click.option("--save-default", is_flag=True)
@click.option("--max-repeat-combis", type=int, default = 1)
@click.option("--repeats-to-analyze", type=int)
@click.option("--correction-method", type=str, default="FDR")
@click.option("--n-repeats", type=int)
@click.argument("association-files", type=click.Path(exists=True), nargs=-1)
@click.argument("config-file", type=click.Path(exists=True))
@click.argument("out-dir", type=click.Path())
def evaluate(
    debug: bool,
    phenotype: Optional[str],
    use_seed_genes: bool,
    save_default: bool,
    correction_method: str,
    n_repeats: Optional[int],
    association_files: Tuple[str],
    config_file: str,
    out_dir: str,
    max_repeat_combis: int,
    repeats_to_analyze: Optional[int]
):
    with open(config_file) as f:
        config = yaml.safe_load(f)
    repeats_to_analyze = repeats_to_analyze if repeats_to_analyze is not None else n_repeats
    associations = pd.concat(
        [pd.read_parquet(f, engine="pyarrow") for f in association_files]
    )
    logger.info('Associations loaded')
    pheno = (
        phenotype
        if phenotype is not None
        else config["data"]["dataset_config"]["y_phenotypes"][0]
    )
    associations["phenotype"] = pheno
    alpha = config["alpha"]

    repeats = n_repeats if n_repeats is not None else config["n_repeats"]

    seed_genes = (
        pd.read_parquet(config["seed_gene_file"], engine="pyarrow")
        if (use_seed_genes and "seed_gene_file" in config)
        else None
    )

    if use_seed_genes:
        logger.info("Reading seed gene discovery results")
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
        seed_genes,
        repeats=repeats,
        baseline_results=baseline_results,
        correction_method=correction_method,
        debug=debug,
        max_repeat_combis = max_repeat_combis,
        repeats_to_analyze = repeats_to_analyze
    )

    logger.info("Saving results")
    out_path = Path(out_dir)
    significant.to_parquet(out_path / f"significant_{repeats_to_analyze}repeats.parquet", engine="pyarrow")
    all_pvals.to_parquet(out_path / f"all_results_{repeats_to_analyze}repeats.parquet", engine="pyarrow")
    if (repeats_to_analyze == n_repeats) & (max_repeat_combis == 1) | save_default :
        logger.info('Also saving result without repeat suffix since its the "default"')
        #save the 'traditional' output
        significant.to_parquet(out_path / f"significant.parquet", engine="pyarrow")
        all_pvals.to_parquet(out_path / f"all_results.parquet", engine="pyarrow")



if __name__ == "__main__":
    evaluate()

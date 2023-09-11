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
@click.option("--seed-gene-dir", type=click.Path(exists=True))
@click.option("--phenotype", type=str)
@click.option("--baseline-results", type=click.Path(exists=True), multiple=True)
@click.option("--baseline-results-out", type=click.Path())
@click.option("--seed-genes-out", type=click.Path())
@click.argument("old_config_file", type=click.Path(exists=True))
@click.argument("new_config_file", type=click.Path())
def update_config(
    old_config_file: str,
    phenotype: Optional[str],
    seed_gene_dir: Optional[str],
    baseline_results: Tuple[str],
    baseline_results_out: Optional[str],
    seed_genes_out: Optional[str],
    new_config_file: str,
):
    # if seed_gene_dir is None and len(baseline_results) == 0:
    #     raise ValueError(
    #         "One of --seed-gene-dir and --baseline-results " "must be specified"
    #     )

    with open(old_config_file) as f:
        config = yaml.safe_load(f)

    if phenotype is not None:
        logger.info(f"Updating config for phenotype {phenotype}")
        config["data"]["dataset_config"]["y_phenotypes"] = [phenotype]
        config["training_data"]["dataset_config"]["y_phenotypes"] = [phenotype]

        # For using seed genes from results of baseline methods
        if len(baseline_results) > 0:
            logger.info("Choosing seed genes based on baseline results")
            if phenotype is None or seed_genes_out is None:
                raise ValueError(
                    "--phenotype and --seed-genes-out must be "
                    "specified if --baseline-results is"
                )
            seed_config = config["phenotypes"].get(phenotype, None)
            if isinstance(config["phenotypes"], dict):
                correction_method = seed_config.get("correction_method", None)
                min_seed_genes = seed_config.get("min_seed_genes", None)
                max_seed_genes = seed_config.get("max_seed_genes", None)
                threshold = seed_config.get("pvalue_threshold", None)
            else:
                logger.info('seed gene config not set, defaulting to correction method FDR')
                min_seed_genes = None
                max_seed_genes = None
                threshold =  None
                correction_method = 'FDR'
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
                baseline_df = baseline_df.query("EAC > 50")
            else:
                logger.info("Not performing EAC filtering of baseline results")
            logger.info(f"  Correcting p-values using {correction_method} method")
            baseline_df = pval_correction(
                baseline_df, config["alpha"], correction_type=correction_method
            )

            baseline_df = baseline_df.sort_values("pval_corrected")

            if baseline_results_out is not None:
                baseline_df.to_parquet(baseline_results_out, engine="pyarrow")

            if correction_method is not None:
                if len(baseline_df.query("significant")) < 5:
                    logger.info(
                        "Selecting top 5 genes from baseline because less than 5 genes are significant"
                    )
                    baseline_df = baseline_df.head(5)  # TODO make this flexible
                else:
                    baseline_df = baseline_df.query("significant")
                logger.info(f"  {len(baseline_df)} significant genes from baseline")
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

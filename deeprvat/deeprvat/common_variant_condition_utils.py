# # Implement a a pipeline that re-tests significant associations but controlling for common variants

import pandas as pd
import pyranges as pr
from pyarrow.parquet import ParquetFile
import numpy as np
import zarr
from pathlib import Path
from numcodecs import Blosc
import logging
from deeprvat.utils import pval_correction, standardize_series
import click
import sys
import yaml
import shutil
import os


compression_level = 1

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--pval-correction-method", type=str, default="Bonferroni")
@click.option("--alpha", type=str, default=0.05)
@click.option("--debug", is_flag=True)
@click.argument("config-file", type=click.Path(exists=True))
@click.argument("res-file", type=click.Path(exists=True))
@click.argument("out-parquet", type=click.Path())
@click.argument("out-gene-ids", type=click.Path())
def get_significant_genes(
    alpha,
    pval_correction_method,
    debug,
    config_file,
    res_file,
    out_parquet,
    out_gene_ids,
):

    with open(config_file) as f:
        config = yaml.safe_load(f)

    gene_file = config["association_testing_data"]["dataset_config"]["gene_file"]
    logger.info(f"reading gene file from {gene_file}")

    gene_df = pd.read_parquet(gene_file)
    gene_df = gene_df.rename(columns={"id": "gene", "gene": "ensembl_id_version"})

    logger.info(
        f"reading association testing resultsf from {res_file} and re-doing multiple testing correction using {pval_correction_method}"
    )
    res = pd.read_parquet(res_file)
    res = pval_correction(res, alpha=alpha, correction_type=pval_correction_method)

    sig_genes = res.query("significant")[["gene", "pval", "pval_corrected"]].merge(
        gene_df
    )
    logger.info(f"number of significant genes {len(sig_genes)}")
    sig_genes = sig_genes.set_index("gene")
    logger.info(sig_genes)
    if debug:
        sig_genes = sig_genes.head(2)

    sig_genes.to_parquet(out_parquet)
    genes_npy = np.array(sig_genes.index)
    np.save(out_gene_ids, genes_npy)


@cli.command()
@click.option("--gtf-file", type=click.Path(exists=True))
@click.option("--padding", type=int, default=0)
@click.option("--standardize", is_flag=True)
@click.argument("config-file", type=click.Path(exists=True))
@click.argument("sig-gene-file", type=click.Path(exists=True))
@click.argument("genotype-file", type=click.Path(exists=True))
@click.argument("sample-file")  # , type=click.Path(exists=True))
@click.argument("out-dir", type=click.Path())
def prepare_genotypes_per_gene(
    standardize: bool,
    gtf_file: str,
    padding: int,
    sample_file: str,
    sig_gene_file: str,
    genotype_file,
    config_file: str,
    out_dir: str,
):

    # Get the path to the active Conda environment
    conda_env_path = os.environ.get("CONDA_PREFIX")

    # Check if a Conda environment is activated
    if conda_env_path:
        logger.info(f"Active Conda environment: {conda_env_path}")
    else:
        logger.info("No Conda environment is currently activated.")
    fillna = True

    sig_genes = pd.read_parquet(sig_gene_file)
    logger.info(f"Number of significant genes: {len(sig_genes)}")

    logger.info(f"reading ordered samples (as in x/y/burdens.zarr) from {sample_file}")
    ordered_sample_ids = zarr.open(sample_file)[:]
    ordered_sample_ids = [int(i) for i in ordered_sample_ids]
    n_total_samples = len(ordered_sample_ids)
    logger.info(f"total number of samples: {n_total_samples}")

    logger.info(f"reading genome annotation file from {gtf_file}")
    genome_annotation = pr.read_gtf(gtf_file)
    gene_annotation = genome_annotation[genome_annotation.Feature == "gene"]

    logger.info(f"reading clumped genotypes from {genotype_file}")

    var_names = ParquetFile(genotype_file).schema.names
    split_var_names = pd.Series(var_names[6:]).str.split(":", expand=True)

    variants = pr.from_dict(
        {
            "Chromosome": split_var_names[0].astype(str),
            "Start": split_var_names[1].astype(int),
            "End": split_var_names[1].astype(int) + 1,
            "var_name": pd.Series(var_names[6:]),
        }
    )

    logger.info(f"Using padding {padding}bp around each gene")

    genes_with_no_variants = []
    for gene_id in list(sig_genes.index):
        gene_ensembl_id = sig_genes.loc[gene_id]["ensembl_id_version"].split(".")[0]
        logger.info(f"writing genotypes for gene {gene_id}, {gene_ensembl_id}")
        gene_annotation_expanded = gene_annotation.copy()
        gene_annotation_expanded.Start = gene_annotation_expanded.Start - padding
        gene_annotation_expanded.End = gene_annotation_expanded.End + padding

        included_vars = variants.intersect(
            gene_annotation_expanded[
                gene_annotation_expanded.gene_id.str.startswith(gene_ensembl_id)
            ]
        )
        included_vars = (
            included_vars.as_df()["var_name"].to_list() if included_vars else []
        )

        if len(included_vars) > 0:
            logger.info(
                f"Loading genotypes for {len(included_vars)} variants in gene region"
            )
            ref_ac_df = pd.read_parquet(genotype_file, columns=["IID"] + included_vars)

            selected_genos = pd.Series(ordered_sample_ids, name="IID").to_frame()

            selected_genos = selected_genos.merge(ref_ac_df, how="left", on="IID")
            selected_genos = selected_genos.rename(columns={"IID": "sample"}).set_index(
                "sample"
            )
            assert all(selected_genos.index == ordered_sample_ids)

            if fillna:
                logger.info("Filling nan genotypes with 0")
                selected_genos = selected_genos.fillna(0)
            logger.info(
                "taking 2 - AC to  get minor allele counts since plink returns reference allele counts"
            )
            selected_genos = 2 - selected_genos
            logger.info("summary of minor allele frquencies")

            logger.info(
                (selected_genos.sum(axis=0) / (2 * len(selected_genos))).describe()
            )

            if standardize:
                logger.info("  Standardizing input genotypes")
                for col in selected_genos:
                    selected_genos[col] = standardize_series(selected_genos[col])
            this_genos = np.array(selected_genos)
            logger.info(this_genos.shape)
        else:
            logger.info("Gene has no variants, just writing array of zeros ")
            this_genos = np.zeros((n_total_samples, 1))
            genes_with_no_variants.append(gene_ensembl_id)

        out_file = Path(out_dir) / f"genotypes_gene_{gene_id}.zarr"
        if os.path.exists(out_file):
            logger.info(f"removing existing zarr file {out_file}")
            shutil.rmtree(out_file)

        gene_x = zarr.open(
            out_file,
            mode="a",
            shape=(n_total_samples,) + this_genos.shape[1:],
            chunks=(None, None),
            dtype=np.float32,
            compressor=Blosc(clevel=compression_level),
        )
        gene_x[:] = this_genos

    logger.info(
        f"Genes with no variants: {len(genes_with_no_variants), genes_with_no_variants}"
    )
    logger.info("finished")


if __name__ == "__main__":
    cli()

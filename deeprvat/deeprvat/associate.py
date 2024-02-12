import copy
import gzip
import itertools
import logging
import math
import pickle
import os
import sys
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple

import click
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyranges as pr
import torch
import torch.nn as nn
import statsmodels.api as sm
import yaml
from bgen import BgenWriter
from numcodecs import Blosc
from seak import scoretest
from statsmodels.tools.tools import add_constant
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm, trange
import zarr
import re

import deeprvat.deeprvat.models as deeprvat_models
from deeprvat.data import DenseGTDataset

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

PLOF_COLS = [
    "Consequence_stop_gained",
    "Consequence_frameshift_variant",
    "Consequence_stop_lost",
    "Consequence_start_lost",
    "Consequence_splice_acceptor_variant",
    "Consequence_splice_donor_variant",
]


def get_burden(
    batch: Dict,
    agg_models: Dict[str, List[nn.Module]],
    device: torch.device = torch.device("cpu"),
    skip_burdens=False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute burden scores for rare variants.

    :param batch: A dictionary containing batched data from the DataLoader.
    :type batch: Dict
    :param agg_models: Loaded PyTorch model(s) for each repeat used for burden computation.
                       Each key in the dictionary corresponds to a respective repeat.
    :type agg_models: Dict[str, List[nn.Module]]
    :param device: Device to perform computations on, defaults to "cpu".
    :type device: torch.device
    :param skip_burdens: Flag to skip burden computation, defaults to False.
    :type skip_burdens: bool
    :return: Tuple containing burden scores, target y phenotype values, x phenotypes and sample ids.
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    .. note::
        Checkpoint models all corresponding to the same repeat are averaged for that repeat.
    """
    with torch.no_grad():
        X = batch["rare_variant_annotations"].to(device)
        burden = []
        if not skip_burdens:
            for key in sorted(
                list(agg_models.keys()), key=lambda x: int(x.split("_")[1])
            ):
                this_agg_models = agg_models[key]
                this_burden: torch.Tensor = sum([m(X) for m in this_agg_models]) / len(
                    this_agg_models
                )
                burden.append(this_burden.cpu().numpy())
            burden = np.concatenate(burden, axis=2)
        else:
            burden = None

    y = batch["y"]
    x = batch["x_phenotypes"]
    sample_ids = batch["sample"]

    return burden, y, x, sample_ids


def separate_parallel_results(results: List) -> Tuple[List, ...]:
    """
    Separate results from running regression on each gene.

    :param results: List of results obtained from regression analysis.
    :type results: List
    :return: Tuple of lists containing separated results of regressed_genes, betas, and pvals.
    :rtype: Tuple[List, ...]
    """
    return tuple(map(list, zip(*results)))


@click.group()
def cli():
    pass


def make_dataset_(
    config: Dict,
    debug: bool = False,
    data_key="data",
    samples: Optional[List[int]] = None,
) -> Dataset:
    """
    Create a dataset based on the configuration.

    :param config: Configuration dictionary.
    :type config: Dict
    :param debug: Flag for debugging, defaults to False.
    :type debug: bool
    :param data_key: Key for dataset configuration in the config dictionary, defaults to "data".
    :type data_key: str
    :param samples: List of sample indices to include in the dataset, defaults to None.
    :type samples: List[int]
    :return: Loaded instance of the created dataset.
    :rtype: Dataset
    """
    data_config = config[data_key]

    ds_pickled = data_config.get("pickled", None)
    if ds_pickled is not None and os.path.isfile(ds_pickled):
        logger.info("Loading pickled dataset")
        with open(ds_pickled, "rb") as f:
            ds = pickle.load(f)
    else:
        ds = DenseGTDataset(
            data_config["gt_file"],
            variant_file=data_config["variant_file"],
            split="",
            skip_y_na=False,
            **copy.deepcopy(data_config["dataset_config"]),
        )

        restrict_samples = config.get("restrict_samples", None)
        if debug:
            logger.info("Debug flag set; Using only 1000 samples")
            ds = Subset(ds, range(1_000))
        elif samples is not None:
            ds = Subset(ds, samples)
        elif restrict_samples is not None:
            ds = Subset(ds, range(restrict_samples))

    return ds


@cli.command()
@click.option("--debug", is_flag=True)
@click.option("--data-key", type=str, default="data")
@click.argument("config-file", type=click.Path(exists=True))
@click.argument("out-file", type=click.Path())
def make_dataset(debug: bool, data_key: str, config_file: str, out_file: str):
    """
    Create a dataset based on the provided configuration and save to a pickle file.

    :param debug: Flag for debugging.
    :type debug: bool
    :param data_key: Key for dataset configuration in the config dictionary, defaults to "data".
    :type data_key: str
    :param config_file: Path to the configuration file.
    :type config_file: str
    :param out_file: Path to the output file.
    :type out_file: str
    :return: Created dataset saved to out_file.pkl
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)

    ds = make_dataset_(config, debug=debug, data_key=data_key)

    with open(out_file, "wb") as f:
        pickle.dump(ds, f)


def compute_burdens_(
    debug: bool,
    config: Dict,
    ds: torch.utils.data.Dataset,
    cache_dir: str,
    agg_models: Dict[str, List[nn.Module]],
    n_chunks: Optional[int] = None,
    chunk: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    bottleneck: bool = False,
    compression_level: int = 1,
    skip_burdens: bool = False,
) -> Tuple[
    np.ndarray, zarr.core.Array, zarr.core.Array, zarr.core.Array, zarr.core.Array
]:
    """
    Compute burdens using the PyTorch model for each repeat.

    :param debug: Flag for debugging.
    :type debug: bool
    :param config: Configuration dictionary.
    :type config: Dict
    :param ds: Torch dataset.
    :type ds: torch.utils.data.Dataset
    :param cache_dir: Directory to cache zarr files of computed burdens, x phenotypes, and y phenotypes.
    :type cache_dir: str
    :param agg_models: Loaded PyTorch model(s) for each repeat used for burden computation.
        Each key in the dictionary corresponds to a respective repeat.
    :type agg_models: Dict[str, List[nn.Module]]
    :param n_chunks: Number of chunks to split data for processing, defaults to None.
    :type n_chunks: Optional[int]
    :param chunk: Index of the chunk of data, defaults to None.
    :type chunk: Optional[int]
    :param device: Device to perform computations on, defaults to "cpu".
    :type device: torch.device
    :param bottleneck: Flag to enable bottlenecking number of batches, defaults to False.
    :type bottleneck: bool
    :param compression_level: Blosc compressor compression level for zarr files, defaults to 1.
    :type compression_level: int
    :param skip_burdens: Flag to skip burden computation, defaults to False.
    :type skip_burdens: bool
    :return: Tuple containing genes, burdens, target y phenotypes, x phenotypes and sample ids.
    :rtype: Tuple[np.ndarray, zarr.core.Array, zarr.core.Array, zarr.core.Array, zarr.core.Array]

    .. note::
        Checkpoint models all corresponding to the same repeat are averaged for that repeat.
    """
    if not skip_burdens:
        logger.info("agg_models[*][*].reverse:")
        pprint(
            {
                repeat: [m.reverse for m in models]
                for repeat, models in agg_models.items()
            }
        )

    data_config = config["data"]

    ds_full = ds.dataset if isinstance(ds, Subset) else ds
    collate_fn = getattr(ds_full, "collate_fn", None)
    n_total_samples = len(ds)
    ds.rare_embedding.skip_embedding = skip_burdens

    if chunk is not None:
        if n_chunks is None:
            raise ValueError("n_chunks must be specified if chunk is not None")

        chunk_length = math.ceil(n_total_samples / n_chunks)
        chunk_start = chunk * chunk_length
        chunk_end = min(n_total_samples, chunk_start + chunk_length)
        samples = range(chunk_start, chunk_end)
        n_samples = len(samples)
        ds = Subset(ds, samples)

        logger.info(f"Processing samples in {samples} from {n_total_samples} in total")
    else:
        n_samples = n_total_samples
        chunk_start = 0
        chunk_end = n_samples

    dataloader_config = data_config["dataloader_config"]

    if torch.cuda.is_available():
        pin_memory = dataloader_config.get("pin_memory", True)

        logger.info(f"CUDA is available, setting pin_memory={pin_memory}")
        dataloader_config["pin_memory"] = pin_memory

    dl = DataLoader(ds, collate_fn=collate_fn, **dataloader_config)

    logger.info("Computing burden scores")
    batch_size = data_config["dataloader_config"]["batch_size"]
    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(dl),
            file=sys.stdout,
            total=(n_samples // batch_size + (n_samples % batch_size != 0)),
        ):
            this_burdens, this_y, this_x, this_sampleid = get_burden(
                batch, agg_models, device=device, skip_burdens=skip_burdens
            )
            if i == 0:
                if not skip_burdens:
                    chunk_burden = np.zeros(shape=(n_samples,) + this_burdens.shape[1:])
                chunk_y = np.zeros(shape=(n_samples,) + this_y.shape[1:])
                chunk_x = np.zeros(shape=(n_samples,) + this_x.shape[1:])
                chunk_sampleid = np.zeros(shape=(n_samples))

                logger.info(f"Batch size: {batch['rare_variant_annotations'].shape}")

                if not skip_burdens:
                    burdens = zarr.open(
                        Path(cache_dir) / "burdens.zarr",
                        mode="a",
                        shape=(n_total_samples,) + this_burdens.shape[1:],
                        chunks=(1000, 1000, 1),
                        dtype=np.float32,
                        compressor=Blosc(clevel=compression_level),
                    )
                    logger.info(f"burdens shape: {burdens.shape}")
                else:
                    burdens = None

                y = zarr.open(
                    Path(cache_dir) / "y.zarr",
                    mode="a",
                    shape=(n_total_samples,) + this_y.shape[1:],
                    chunks=(None, None),
                    dtype=np.float32,
                    compressor=Blosc(clevel=compression_level),
                )
                x = zarr.open(
                    Path(cache_dir) / "x.zarr",
                    mode="a",
                    shape=(n_total_samples,) + this_x.shape[1:],
                    chunks=(None, None),
                    dtype=np.float32,
                    compressor=Blosc(clevel=compression_level),
                )
                sample_ids = zarr.open(
                    Path(cache_dir) / "sample_ids.zarr",
                    mode="a",
                    shape=(n_total_samples),
                    chunks=(None),
                    dtype=np.float32,
                    compressor=Blosc(clevel=compression_level),
                )

            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, chunk_end)  # read from chunk shape

            if not skip_burdens:
                chunk_burden[start_idx:end_idx] = this_burdens

            chunk_y[start_idx:end_idx] = this_y
            chunk_x[start_idx:end_idx] = this_x
            chunk_sampleid[start_idx:end_idx] = this_sampleid

            if debug:
                logger.info(
                    "Wrote results for batch indices " f"[{start_idx}, {end_idx - 1}]"
                )

            if bottleneck and i > 20:
                break

        if not skip_burdens:
            burdens[chunk_start:chunk_end] = chunk_burden

        y[chunk_start:chunk_end] = chunk_y
        x[chunk_start:chunk_end] = chunk_x
        sample_ids[chunk_start:chunk_end] = chunk_sampleid

    if torch.cuda.is_available():
        logger.info(
            "Max GPU memory allocated: " f"{torch.cuda.max_memory_allocated(0)} bytes"
        )

    return ds_full.rare_embedding.genes, burdens, y, x, sample_ids


def make_saige_input_(
    repeat: int,
    average_repeats: bool,
    dataset_file: Path,
    burden_dir: Path,
    vcf: Path,
    group_file: Path,
    phenotype_file: Path,
):
    logger.setLevel(logging.INFO)

    if (repeat >= 0) + average_repeats != 1:
        raise ValueError(
            "Exactly one of --repeat or --average-repeats must be specified"
        )

    ## Make VCF

    # Load data
    logger.info("Loading computed burdens, covariates, phenotypes and metadata")
    sample_ids = zarr.load(burden_dir / "sample_ids.zarr")
    covariates = zarr.load(burden_dir / "x.zarr")
    y = zarr.load(burden_dir / "y.zarr")
    genes = np.load(burden_dir / "genes.npy")

    burdens_zarr = zarr.open(burden_dir / "burdens.zarr")
    if average_repeats:
        burdens = np.zeros(burdens_zarr.shape[:2])
        for repeat in range(burdens_zarr.shape[2]):
            burdens += burdens_zarr[:, :, repeat]
        burdens = burdens / burdens_zarr.shape[2]
    else:
        assert repeat < burdens_zarr.shape[2]
        burdens = burdens_zarr[:, :, repeat]

    assert burdens_zarr.shape[0] == sample_ids.shape[0]
    assert burdens_zarr.shape[0] == covariates.shape[0]
    assert burdens_zarr.shape[0] == y.shape[0]
    assert burdens_zarr.shape[1] == genes.shape[0]

    mask = np.all(~np.isnan(np.concatenate([y, covariates], axis=1)), axis=1)

    logger.info(f"Keeping {np.sum(mask)} samples with defined phenotype and covariates")

    sample_ids = sample_ids[mask]
    covariates = covariates[mask]
    y = y[mask]
    burdens = burdens[mask]

    with gzip.open(vcf, "wb", compresslevel=1) as f:
        logger.info("Creating VCF file with artificial variants and burdens as dosages")
        logger.info("  This can take a while for large datasets")

        # Make VCF header
        header = ""  # TODO: Replace this with real metadata
        header += (
            "\t".join(
                [
                    "#CHROM",
                    "POS",
                    "ID",
                    "REF",
                    "ALT",
                    "QUAL",
                    "FILTER",
                    "INFO",
                    "FORMAT",
                ]
            )
            + "\t"
        )
        header += "\t".join(sample_ids.astype(str)) + "\n"

        f.write(header.encode("utf-8"))

        # Create one fake variant per gene, with burdens as dosages
        gene_block_size = 1000
        pos = (2 * np.arange(genes.shape[0], dtype=np.int32) + 1).astype(np.bytes_)
        for start_index in trange(0, genes.shape[0], gene_block_size, file=sys.stdout):
            end_index = min(genes.shape[0], start_index + gene_block_size)
            this_burdens = burdens[:, start_index:end_index]

            this_burdens = this_burdens.transpose().astype(np.bytes_)
            this_burdens = np.char.add(np.bytes_(b"\t./.:"), this_burdens)

            metadata = np.char.add(
                np.char.add(b"1\t", pos[start_index:end_index]),
                b"\t.\tA\tC\t100\tPASS\t.\tGT:DS",
            )

            first, last = np.split(
                this_burdens,
                [
                    1,
                ],
                axis=1,
            )
            this_burdens = np.char.add(metadata.reshape(metadata.shape[0], 1), first)

            cells = np.concatenate([this_burdens, last], axis=1)
            lines = [b"".join(x) for x in cells]
            block = b"\n".join(lines) + b"\n"

            f.write(block)

        # pos = lambda i: str(2 * i + 1)
        # for i, g in tqdm(enumerate(genes), total=genes.shape[0], file=sys.stdout):
        #     if i % 1000 == 0:
        #         buffer = b""
        #         start_index = i // 1000
        #         end_index = start_index + 1000
        #         this_burdens = burdens_zarr[:, start_index:end_index, repeat][
        #             mask
        #         ].astype(str)

        #     import ipdb

        #     ipdb.set_trace()

        #     line = "\t".join(
        #         ["1", pos(i), ".", "A", "C", "100", "PASS", ".", "GT:DS", "0/1:"]
        #     )
        #     # line += "\t0/1:".join([str(float(x)) for x in burdens[:, i][mask]])
        #     line += "\t0/1:".join(this_burdens[:, i % 1000])
        #     buffer += line.encode("utf-8") + b"\n"

        #     # DEBUG
        #     if i == 21:
        #         f.write(buffer)
        #         buffer = b""

        #     # DEBUG
        #     if i > 20:
        #         break

    ## Make group file
    logger.info("Creating group file")
    with open(group_file, "w") as f:
        for i, g in enumerate(genes):
            group = f"{g} var 1:{pos[i]}:A:C\n"
            group += f"{g} anno deeprvat\n"
            group += f"{g} weight 1\n"
            f.write(group)

    ## Make phenotype file
    logger.info("Creating phenotype file")
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)

    pheno_df = pd.DataFrame({dataset.y_phenotypes[0]: y.squeeze()})
    cov_df = pd.DataFrame(covariates, columns=dataset.x_phenotypes)
    sample_df = pd.DataFrame(sample_ids, columns=["sample"])
    out_df = pd.concat([pheno_df, cov_df, sample_df], axis=1)
    out_df.to_csv(phenotype_file, sep=" ", index=False)

    logger.info("Done")


@cli.command()
@click.option("--repeat", type=int, default=-1)
@click.option("--average-repeats", is_flag=True)
@click.argument("dataset-file", type=click.Path(exists=True, path_type=Path))
@click.argument("burden-dir", type=click.Path(exists=True, path_type=Path), nargs=-1)
@click.argument("vcf", type=click.Path(path_type=Path))
@click.argument("group-file", type=click.Path(path_type=Path))
@click.argument("phenotype-file", type=click.Path(path_type=Path))
def make_saige_input(
    repeat: int,
    average_repeats: bool,
    dataset_file: Path,
    burden_dir: Tuple[Path],
    bgen: Path,
    group_file: Path,
    phenotype_file: Path,
):
    make_saige_input_(
        repeat,
        average_repeats,
        phenotype,
        # dataset_file,
        # burden_dir,
        bgen,
        group_file,
        phenotype_file,
    )


def make_regenie_input_(
    debug: bool,
    skip_samples: bool,
    skip_covariates: bool,
    skip_phenotypes: bool,
    skip_burdens: bool,
    repeat: int,
    average_repeats: bool,
    phenotype: Tuple[Tuple[str, Path, Path]],
    sample_file: Optional[Path],
    covariate_file: Optional[Path],
    phenotype_file: Optional[Path],
    bgen: Optional[Path],
    gene_file: Path,
    gtf: Path,
):
    ## Check options
    if (repeat >= 0) + average_repeats + skip_burdens != 1:
        raise ValueError(
            "Exactly one of --repeat or --average-repeats or --skip-burdens must be specified"
        )
    if not skip_samples and sample_file is None:
        raise ValueError("Either sample_file or skip_samples must be specified")
    if not skip_covariates and covariate_file is None:
        raise ValueError("Either covariate_file or skip_covariates must be specified")
    if not skip_phenotypes and phenotype_file is None:
        raise ValueError("Either phenotype_file or skip_phenotypes must be specified")
    if not skip_burdens and bgen is None:
        raise ValueError("Either bgen or skip_burdens must be specified")

    ## Make BGEN

    # Load data
    logger.info("Loading computed burdens, covariates, phenotypes and metadata")

    phenotype_names = [p[0] for p in phenotype]
    dataset_files = [p[1] for p in phenotype]
    burden_dirs = [p[2] for p in phenotype]

    sample_ids = zarr.load(burden_dirs[0] / "sample_ids.zarr")
    covariates = zarr.load(burden_dirs[0] / "x.zarr")
    ys = [zarr.load(b / "y.zarr") for b in burden_dirs]
    genes = np.load(burden_dirs[0] / "genes.npy")

    if debug:
        sample_ids = sample_ids[:1000]
        covariates = covariates[:1000]
        ys = [y[:1000] for y in ys]

    n_samples = sample_ids.shape[0]
    n_genes = genes.shape[0]
    assert covariates.shape[0] == n_samples
    assert all([y.shape[0] == n_samples for y in ys])

    # Sanity check: sample_ids, covariates, and genes should be consistent for all phenotypes
    # TODO: Check burdens as well (though this will be slow)
    if not debug:
        for i in range(1, len(phenotype)):
            assert np.array_equal(
                sample_ids, zarr.load(burden_dirs[i] / "sample_ids.zarr")
            )
            assert np.array_equal(
                covariates, zarr.load(burden_dirs[i] / "x.zarr")
            )  # TODO: Phenotype-specific covariates
            assert np.array_equal(genes, np.load(burden_dirs[i] / "genes.npy"))

    logger.warning(
        "Using burdens from first phenotype passed. "
        "Burdens from other phenotypes will be ignored."
    )
    burdens_zarr = zarr.open(burden_dirs[0] / "burdens.zarr")
    if not debug:
        assert burdens_zarr.shape[0] == n_samples
        assert burdens_zarr.shape[1] == n_genes

    sample_df = pd.DataFrame({"FID": sample_ids, "IID": sample_ids})

    if not skip_samples:
        ## Make sample file
        logger.info(f"Creating sample file {sample_file}")
        samples_out = pd.concat(
            [
                pd.DataFrame({"ID_1": 0, "ID_2": 0}, index=[0]),
                sample_df.rename(
                    columns={
                        "FID": "ID_1",
                        "IID": "ID_2",
                    }
                ),
            ]
        )
        samples_out.to_csv(sample_file, sep=" ", index=False)

    if not skip_covariates:
        ## Make covariate file
        logger.info(f"Creating covariate file {covariate_file}")
        with open(dataset_files[0], "rb") as f:
            dataset = pickle.load(f)

        covariate_names = dataset.x_phenotypes
        cov_df = pd.DataFrame(covariates, columns=covariate_names)
        cov_df = pd.concat([sample_df, cov_df], axis=1)
        cov_df.to_csv(covariate_file, sep=" ", index=False, na_rep="NA")

    if not skip_phenotypes:
        ## Make phenotype file
        logger.info(f"Creating phenotype file {phenotype_file}")
        pheno_df_list = []
        for p, y in zip(phenotype_names, ys):
            pheno_df_list.append(pd.DataFrame({p: y.squeeze()}))

        pheno_df = pd.concat([sample_df] + pheno_df_list, axis=1)
        pheno_df.to_csv(phenotype_file, sep=" ", index=False, na_rep="NA")

    if not skip_burdens:
        if average_repeats:
            logger.info("Averaging burdens across all repeats")
            burdens = np.zeros((n_samples, n_genes))
            for repeat in trange(burdens_zarr.shape[2]):
                burdens += burdens_zarr[:n_samples, :, repeat]
            burdens = burdens / burdens_zarr.shape[2]
        else:
            logger.info(f"Using burdens from repeat {repeat}")
            assert repeat < burdens_zarr.shape[2]
            burdens = burdens_zarr[:n_samples, :, repeat]

        # Read GTF file and get positions for pseudovariants (center of interval [Start, End])
        logger.info(
            f"Assigning positions to pseudovariants based on provided GTF file {gtf}"
        )
        gene_pos = pr.read_gtf(gtf)
        gene_pos = gene_pos[
            (gene_pos.Feature == "gene") & (gene_pos.gene_type == "protein_coding")
        ][["Chromosome", "Start", "End", "gene_id"]].as_df()
        gene_pos = gene_pos.set_index("gene_id")
        gene_metadata = pd.read_parquet(gene_file).set_index("id")
        this_gene_pos = gene_pos.loc[gene_metadata.loc[genes, "gene"]]
        pseudovar_pos = (this_gene_pos.End - this_gene_pos.Start).to_numpy().astype(int)
        ensgids = this_gene_pos.index.to_numpy()

        logger.info(f"Writing pseudovariants to {bgen}")
        with BgenWriter(
            bgen,
            n_samples,
            samples=list(sample_ids),
            metadata="Pseudovariants containing DeepRVAT gene impairment scores. One pseudovariant per gene.",
        ) as f:
            for i in trange(n_genes):
                varid = f"pseudovariant_gene_{ensgids[i]}"
                this_burdens = burdens[:, i]  # Rescale scores to be in range (0, 2)
                genotypes = np.stack(
                    (this_burdens, np.zeros(this_burdens.shape), 1 - this_burdens),
                    axis=1,
                )

                f.add_variant(
                    varid=varid,
                    rsid=varid,
                    chrom=this_gene_pos.iloc[i].Chromosome,
                    pos=pseudovar_pos[i],
                    alleles=[
                        "A",
                        "C",
                    ],  # TODO: This is completely arbitrary, however, we might want to match it to a reference FASTA at some point
                    genotypes=genotypes,
                    ploidy=2,
                    bit_depth=16,
                )


@cli.command()
@click.option("--debug", is_flag=True)
@click.option("--skip-samples", is_flag=True)
@click.option("--skip-covariates", is_flag=True)
@click.option("--skip-phenotypes", is_flag=True)
@click.option("--skip-burdens", is_flag=True)
@click.option("--repeat", type=int, default=-1)
@click.option("--average-repeats", is_flag=True)
@click.option(
    "--phenotype",
    type=(
        str,
        click.Path(exists=True, path_type=Path),
        click.Path(exists=True, path_type=Path),
    ),
    multiple=True,
)  # phenotype_name, dataset_file, burden_dir
@click.option("--sample-file", type=click.Path(path_type=Path))
@click.option("--bgen", type=click.Path(path_type=Path))
@click.option("--covariate-file", type=click.Path(path_type=Path))
@click.option("--phenotype-file", type=click.Path(path_type=Path))
# @click.argument("dataset-file", type=click.Path(exists=True, path_type=Path))
# @click.argument("burden-dir", type=click.Path(exists=True, path_type=Path))
@click.argument("gene-file", type=click.Path(exists=True, path_type=Path))
@click.argument("gtf", type=click.Path(exists=True, path_type=Path))
def make_regenie_input(
    debug: bool,
    skip_samples: bool,
    skip_covariates: bool,
    skip_phenotypes: bool,
    skip_burdens: bool,
    repeat: int,
    average_repeats: bool,
    phenotype: Tuple[Tuple[str, Path, Path]],
    sample_file: Optional[Path],
    covariate_file: Optional[Path],
    phenotype_file: Optional[Path],
    bgen: Optional[Path],
    gene_file: Path,
    gtf: Path,
):
    make_regenie_input_(
        debug=debug,
        skip_samples=skip_samples,
        skip_covariates=skip_covariates,
        skip_phenotypes=skip_phenotypes,
        skip_burdens=skip_burdens,
        repeat=repeat,
        average_repeats=average_repeats,
        phenotype=phenotype,
        sample_file=sample_file,
        covariate_file=covariate_file,
        phenotype_file=phenotype_file,
        bgen=bgen,
        gene_file=gene_file,
        gtf=gtf,
    )


def convert_regenie_output_(
    repeat: int, phenotype: Tuple[str, Tuple[Path, Path]], gene_file: Path
):
    genes = pd.read_parquet(gene_file)[["id", "gene"]]
    for pheno_name, regenie_results, out_file in phenotype:
        regenie_cols = ["TEST", "SE", "CHISQ"]
        regenie_col_newnames = [f"regenie_{c}" for c in regenie_cols]
        result_df = pd.read_csv(regenie_results, sep=" ")[
            ["ID", "BETA", "LOG10P"] + regenie_cols
        ]

        result_df["gene"] = result_df["ID"].str.split("_", expand=True)[2]
        old_len = len(result_df)
        result_df = pd.merge(result_df, genes, validate="1:1")
        assert len(result_df) == old_len
        result_df = result_df.drop(columns="ID")
        result_df = result_df.drop(columns="gene").rename(columns={"id": "gene"})

        result_df["phenotype"] = pheno_name
        result_df = result_df.rename(columns={"BETA": "beta"})
        result_df["pval"] = np.power(10, -result_df["LOG10P"])
        result_df = result_df.drop(columns="LOG10P")
        result_df["model"] = f"repeat_{repeat}"
        result_df = result_df.rename(
            columns=dict(zip(regenie_cols, regenie_col_newnames))
        )
        result_df = result_df[
            ["phenotype", "gene", "beta", "pval", "model"] + regenie_col_newnames
        ]
        result_df.to_parquet(out_file)


@cli.command()
@click.option("--repeat", type=int, default=0)
@click.option(
    "--phenotype",
    type=(
        str,
        click.Path(exists=True, path_type=Path),  # REGENIE output file
        click.Path(path_type=Path),  # Converted results
    ),
    multiple=True,
)
@click.argument("gene-file", type=click.Path(exists=True, path_type=Path))
def convert_regenie_output(
    repeat: int, phenotype: Tuple[str, Tuple[Path, Path]], gene_file: Path
):
    convert_regenie_output_(repeat, phenotype, gene_file)


def load_one_model(
    config: Dict,
    checkpoint: str,
    device: torch.device = torch.device("cpu"),
):
    """
    Load a single burden score computation model from a checkpoint file.

    :param config: Configuration dictionary.
    :type config: Dict
    :param checkpoint: Path to the model checkpoint file.
    :type checkpoint: str
    :param device: Device to load the model onto, defaults to "cpu".
    :type device: torch.device
    :return: Loaded PyTorch model for burden score computation.
    :rtype: nn.Module
    """
    model_class = getattr(deeprvat_models, config["model"]["type"])
    model = model_class.load_from_checkpoint(
        checkpoint,
        config=config["model"]["config"],
    )
    model = model.eval()
    model = model.to(device)
    agg_model = model.agg_model
    return agg_model


@cli.command()
@click.argument("model-config-file", type=click.Path(exists=True))
@click.argument("data-config-file", type=click.Path(exists=True))
@click.argument("checkpoint-files", type=click.Path(exists=True), nargs=-1)
def reverse_models(
    model_config_file: str, data_config_file: str, checkpoint_files: Tuple[str]
):
    """
    Determine if the burden score computation PyTorch model should reverse the output based on PLOF annotations.

    :param model_config_file: Path to the model configuration file.
    :type model_config_file: str
    :param data_config_file: Path to the data configuration file.
    :type data_config_file: str
    :param checkpoint_files: Paths to checkpoint files.
    :type checkpoint_files: Tuple[str]
    :return: checkpoint.reverse file is created if the model should reverse the burden score output.
    """
    with open(model_config_file) as f:
        model_config = yaml.safe_load(f)

    with open(data_config_file) as f:
        data_config = yaml.safe_load(f)

    annotation_file = data_config["data"]["dataset_config"]["annotation_file"]

    if torch.cuda.is_available():
        logger.info("Using GPU")
        device = torch.device("cuda")
    else:
        logger.info("Using CPU")
        device = torch.device("cpu")

    plof_df = (
        dd.read_parquet(
            annotation_file,
            columns=data_config["data"]["dataset_config"]["rare_embedding"]["config"][
                "annotations"
            ],
        )
        .query(" or ".join([f"{c} == 1" for c in PLOF_COLS]))
        .compute()
    )

    plof_zero_df = plof_df.copy()
    plof_zero_df.loc[:, PLOF_COLS] = 0.0

    plof = plof_df.to_numpy()
    plof_zero = plof_zero_df.to_numpy()

    n_variants = plof.shape[0]
    for checkpoint in checkpoint_files:
        if Path(checkpoint + ".dropped").is_file():
            # Ignore checkpoints that were chosen to be dropped
            continue

        agg_model = load_one_model(data_config, checkpoint, device=device)
        score = agg_model(
            torch.tensor(plof, dtype=torch.float, device=device).reshape(
                (n_variants, 1, -1, 1)
            )
        ).reshape(n_variants)
        score_zero = agg_model(
            torch.tensor(plof_zero, dtype=torch.float, device=device).reshape(
                (n_variants, 1, -1, 1)
            )
        ).reshape(n_variants)
        mean_difference = torch.mean(score - score_zero).item()

        if mean_difference < 0:
            logger.info(f"Reversed model at checkpoint {checkpoint}")
            Path(checkpoint + ".reverse").touch()


def load_models(
    config: Dict,
    checkpoint_files: Tuple[str],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, List[nn.Module]]:
    """
    Load models from multiple checkpoints for multiple repeats.

    :param config: Configuration dictionary.
    :type config: Dict
    :param checkpoint_files: Paths to checkpoint files.
    :type checkpoint_files: Tuple[str]
    :param device: Device to load the models onto, defaults to "cpu".
    :type device: torch.device
    :return: Dictionary of loaded PyTorch models for burden score computation for each repeat.
    :rtype: Dict[str, List[nn.Module]]

    :Examples:

    >>> config = {"model": {"type": "MyModel", "config": {"param": "value"}}}
    >>> checkpoint_files = ("checkpoint1.pth", "checkpoint2.pth")
    >>> load_models(config, checkpoint_files)
    {'repeat_0': [MyModel(), MyModel()]}
    """
    logger.info("Loading models and checkpoints")

    if all(
        re.search("repeat_\d+", file) for file in checkpoint_files
    ):  # check if this is an experiment with multiple repeats
        all_repeats = [
            re.search(r"(/)(repeat_\d+)", file).groups()[1] for file in checkpoint_files
        ]
        repeats = list(set(all_repeats))
        repeats = sorted(
            repeats, key=lambda x: int(x.split("_")[1])
        )  # sort according to the repeat number
    else:
        repeats = ["repeat_0"]

    first_repeat = repeats[0]
    logger.info(f"Number of repeats: {len(repeats)}, The repeats are: {repeats}")

    checkpoint_files = {
        repeat: [file for file in checkpoint_files if repeat in file]
        for repeat in repeats
    }

    if len(checkpoint_files[first_repeat]) > 1:
        logging.info(
            f"  Averaging results from {len(checkpoint_files[first_repeat])} models for each repeat"
        )

    agg_models = {repeat: [] for repeat in repeats}

    for repeat in repeats:
        dropped = 0
        reversed = 0
        repeat_checkpoint_files = checkpoint_files[repeat]
        for ckpt in repeat_checkpoint_files:
            if Path(ckpt + ".dropped").is_file():
                # Ignore checkpoints that were chosen to be dropped
                dropped += 1
                continue

            this_agg = load_one_model(config, ckpt, device=device)
            if Path(ckpt + ".reverse").is_file():
                this_agg.set_reverse()
                reversed += 1

            agg_models[repeat].append(this_agg)

        logger.info(
            f"Kept {len(agg_models[repeat])} models "
            f"(dropped {dropped}), reversed {reversed}, "
            f"for repeat {repeat}"
        )

    return agg_models


@cli.command()
@click.option("--debug", is_flag=True)
@click.option("--bottleneck", is_flag=True)
@click.option("--n-chunks", type=int)
@click.option("--chunk", type=int)
@click.option("--dataset-file", type=click.Path(exists=True))
@click.option("--link-burdens", type=click.Path())
@click.argument("data-config-file", type=click.Path(exists=True))
@click.argument("model-config-file", type=click.Path(exists=True))
@click.argument("checkpoint-files", type=click.Path(exists=True), nargs=-1)
@click.argument("out-dir", type=click.Path(exists=True))
def compute_burdens(
    debug: bool,
    bottleneck: bool,
    n_chunks: Optional[int],
    chunk: Optional[int],
    dataset_file: Optional[str],
    link_burdens: Optional[str],
    data_config_file: str,
    model_config_file: str,
    checkpoint_files: Tuple[str],
    out_dir: str,
):
    """
    Compute burdens based on the provided model and dataset.

    :param debug: Flag for debugging.
    :type debug: bool
    :param bottleneck: Flag to enable bottlenecking number of batches.
    :type bottleneck: bool
    :param n_chunks: Number of chunks to split data for processing, defaults to None.
    :type n_chunks: Optional[int]
    :param chunk: Index of the chunk of data, defaults to None.
    :type chunk: Optional[int]
    :param dataset_file: Path to the dataset file, i.e., association_dataset.pkl.
    :type dataset_file: Optional[str]
    :param link_burdens: Path to burden.zarr file to link.
    :type link_burdens: Optional[str]
    :param data_config_file: Path to the data configuration file.
    :type data_config_file: str
    :param model_config_file: Path to the model configuration file.
    :type model_config_file: str
    :param checkpoint_files: Paths to model checkpoint files.
    :type checkpoint_files: Tuple[str]
    :param out_dir: Path to the output directory.
    :type out_dir: str
    :return: Corresonding genes, computed burdens, y phenotypes, x phenotypes and sample ids are saved in the out_dir.
    :rtype: [np.ndarray], [zarr.core.Array], [zarr.core.Array], [zarr.core.Array], [zarr.core.Array]

    .. note::
        Checkpoint models all corresponding to the same repeat are averaged for that repeat.
    """
    if len(checkpoint_files) == 0:
        raise ValueError("At least one checkpoint file must be supplied")

    with open(data_config_file) as f:
        data_config = yaml.safe_load(f)

    with open(model_config_file) as f:
        model_config = yaml.safe_load(f)

    if dataset_file is not None:
        logger.info("Loading pickled dataset")
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = make_dataset_(config)

    if torch.cuda.is_available():
        logger.info("Using GPU")
        device = torch.device("cuda")
    else:
        logger.info("Using CPU")
        device = torch.device("cpu")

    if link_burdens is None:
        agg_models = load_models(model_config, checkpoint_files, device=device)
    else:
        agg_models = None

    genes, _, _, _, _ = compute_burdens_(
        debug,
        data_config,
        dataset,
        out_dir,
        agg_models,
        n_chunks=n_chunks,
        chunk=chunk,
        device=device,
        bottleneck=bottleneck,
        skip_burdens=(link_burdens is not None),
    )

    logger.info("Saving computed burdens, corresponding genes, and targets")
    np.save(Path(out_dir) / "genes.npy", genes)
    if link_burdens is not None:
        source_path = Path(out_dir) / "burdens.zarr"
        source_path.unlink(missing_ok=True)
        source_path.symlink_to(link_burdens)


def regress_on_gene_scoretest(
    gene: str,
    burdens: np.ndarray,
    model_score,
) -> Tuple[List[str], List[float], List[float]]:
    """
    Perform regression on a gene using the score test.

    :param gene: Gene name.
    :type gene: str
    :param burdens: Burden scores associated with the gene.
    :type burdens: np.ndarray
    :param model_score: Model for score test.
    :type model_score: Any
    :return: Tuple containing gene name, beta, and p-value.
    :rtype: Tuple[List[str], List[float], List[float]]
    """
    burdens = burdens.reshape(burdens.shape[0], -1)
    logger.info(f"Burdens shape: {burdens.shape}")

    if np.all(np.abs(burdens) < 1e-6):
        logger.warning(f"Burden for gene {gene} is 0 for all samples; skipping")
        return None

    pv = model_score.pv_alt_model(burdens)

    logger.info(f"p-value: {pv}")
    if pv < 0:
        logger.warning(
            f"Negative value encountered in p-value computation for "
            f"gene {gene}, p-value: {pv}, using saddle instead."
        )
        pv = model_score.pv_alt_model(burdens, method="saddle")
    # beta only for linear models
    try:
        beta = model_score.coef(burdens)["beta"][0, 0]
    except:
        beta = None

    genes_params_pvalues = ([], [], [])
    genes_params_pvalues[0].append(gene)
    genes_params_pvalues[1].append(beta)
    genes_params_pvalues[2].append(pv)

    return genes_params_pvalues


def regress_on_gene(
    gene: str,
    X: np.ndarray,
    y: np.ndarray,
    x_pheno: np.ndarray,
    use_bias: bool,
    use_x_pheno: bool,
) -> Tuple[List[str], List[float], List[float]]:
    """
    Perform regression on a gene using Ordinary Least Squares (OLS).

    :param gene: Gene name.
    :type gene: str
    :param X: Burden score data.
    :type X: np.ndarray
    :param y: Y phenotype data.
    :type y: np.ndarray
    :param x_pheno: X phenotype data.
    :type x_pheno: np.ndarray
    :param use_bias: Flag to include bias term.
    :type use_bias: bool
    :param use_x_pheno: Flag to include x phenotype data in regression.
    :type use_x_pheno: bool
    :return: Tuple containing gene name, beta, and p-value.
    :rtype: Tuple[List[str], List[float], List[float]]
    """
    X = X.reshape(X.shape[0], -1)
    if np.all(np.abs(X) < 1e-6):
        logger.warning(f"Burden for gene {gene} is 0 for all samples; skipping")
        return None

    # Bias shouldn't be necessary if y is centered or standardized
    if use_bias:
        try:
            X = add_constant(X, prepend=False, has_constant="raise")
        except ValueError:
            logger.warning(
                f"Burdens for gene {gene} are constant " "for all samples; skipping"
            )
            return None

    if use_x_pheno:
        if len(x_pheno.shape) == 1:
            x_pheno = np.expand_dims(x_pheno, axis=1)
        X = np.concatenate((X, x_pheno), axis=1)

    genes_params_pvalues = ([], [], [])
    for this_y in np.split(y, y.shape[1], axis=1):
        mask = ~np.isnan(this_y).reshape(-1)
        model = sm.OLS(this_y[mask], X[mask], missing="raise", hasconst=True)
        results = model.fit()
        genes_params_pvalues[0].append(gene)
        genes_params_pvalues[1].append(results.params[0])
        genes_params_pvalues[2].append(results.pvalues[0])

    return genes_params_pvalues


def regress_(
    config: Dict,
    use_bias: bool,
    burdens: np.ndarray,
    y: np.ndarray,
    gene_indices: np.ndarray,
    genes: pd.Series,
    x_pheno: np.ndarray,
    use_x_pheno: bool = True,
    do_scoretest: bool = True,
) -> pd.DataFrame:
    """
    Perform regression on multiple genes.

    :param config: Configuration dictionary.
    :type config: Dict
    :param use_bias: Flag to include bias term when performing OLS regression.
    :type use_bias: bool
    :param burdens: Burden score data.
    :type burdens: np.ndarray
    :param y: Y phenotype data.
    :type y: np.ndarray
    :param gene_indices: Indices of genes.
    :type gene_indices: np.ndarray
    :param genes: Gene names.
    :type genes: pd.Series
    :param x_pheno: X phenotype data.
    :type x_pheno: np.ndarray
    :param use_x_pheno: Flag to include x phenotype data when performing OLS regression, defaults to True.
    :type use_x_pheno: bool
    :param do_scoretest: Flag to use the scoretest from SEAK, defaults to True.
    :type do_scoretest: bool
    :return: DataFrame containing regression results on all genes.
    :rtype: pd.DataFrame
    """
    assert len(gene_indices) == len(genes)

    logger.info(f"Computing associations")
    logger.info(f"Covariates shape: {x_pheno.shape}, y shape: {y.shape}")

    regressed_genes = []
    betas = []
    pvals = []
    logger.info("Running regression on each gene")
    if do_scoretest:
        logger.info("Running regression on each gene using scoretest from SEAK")
        mask = ~np.isnan(y).reshape(-1)
        y = y[mask]
        X = np.hstack((np.ones((x_pheno.shape[0], 1)), x_pheno))[mask]
        # adding bias column
        logger.info(f"X shape: {X.shape}, Y shape: {y.shape}")

        # compute null_model for score test
        if len(np.unique(y)) == 2:
            logger.info("Fitting binary model since only found two distinct y values")
            model_score = scoretest.ScoretestLogit(y, X)
        else:
            logger.info("Fitting linear model")
            model_score = scoretest.ScoretestNoK(y, X)
        genes_betas_pvals = [
            regress_on_gene_scoretest(gene, burdens[mask, i], model_score)
            for i, gene in tqdm(
                zip(gene_indices, genes), total=genes.shape[0], file=sys.stdout
            )
        ]
    else:
        logger.info("Running regression on each gene using OLS")
        genes_betas_pvals = [
            regress_on_gene(gene, burdens[:, i], y, x_pheno, use_bias, use_x_pheno)
            for i, gene in tqdm(
                zip(gene_indices, genes), total=genes.shape[0], file=sys.stdout
            )
        ]

    genes_betas_pvals = [x for x in genes_betas_pvals if x is not None]
    regressed_genes, betas, pvals = separate_parallel_results(genes_betas_pvals)
    y_phenotypes = config["data"]["dataset_config"]["y_phenotypes"]
    regressed_phenotypes = [y_phenotypes] * len(regressed_genes)
    result = pd.DataFrame(
        {
            "phenotype": itertools.chain(*regressed_phenotypes),
            "gene": itertools.chain(*regressed_genes),
            "beta": itertools.chain(*betas),
            "pval": itertools.chain(*pvals),
        }
    )
    return result


@cli.command()
@click.option("--debug", is_flag=True)
@click.option("--chunk", type=int, default=0)
@click.option("--n-chunks", type=int, default=1)
@click.option("--use-bias", is_flag=True)
@click.option("--gene-file", type=click.Path(exists=True))
@click.option("--repeat", type=int, default=0)
@click.option("--do-scoretest", is_flag=True)
@click.option("--sample-file", type=click.Path(exists=True))
@click.argument("config-file", type=click.Path(exists=True))
@click.argument("burden-dir", type=click.Path(exists=True))
@click.argument("out-dir", type=click.Path())
def regress(
    debug: bool,
    chunk: int,
    n_chunks: int,
    use_bias: bool,
    gene_file: str,
    repeat: int,
    config_file: str,
    burden_dir: str,
    out_dir: str,
    do_scoretest: bool,
    sample_file: Optional[str],
):
    """
    Perform regression analysis.

    :param debug: Flag for debugging.
    :type debug: bool
    :param chunk: Index of the chunk of data, defaults to 0.
    :type chunk: int
    :param n_chunks: Number of chunks to split data for processing, defaults to 1.
    :type n_chunks: int
    :param use_bias: Flag to include bias term when performing OLS regression.
    :type use_bias: bool
    :param gene_file: Path to the gene file.
    :type gene_file: str
    :param repeat: Index of the repeat, defaults to 0.
    :type repeat: int
    :param config_file: Path to the configuration file.
    :type config_file: str
    :param burden_dir: Path to the directory containing burdens.zarr file.
    :type burden_dir: str
    :param out_dir: Path to the output directory.
    :type out_dir: str
    :param do_scoretest: Flag to use the scoretest from SEAK.
    :type do_scoretest: bool
    :param sample_file: Path to the sample file.
    :type sample_file: Optional[str]
    :return: Regression results saved to out_dir as "burden_associations_{chunk}.parquet"
    """
    logger.info("Loading saved burdens")
    y = zarr.open(Path(burden_dir) / "y.zarr")[:]
    burdens = zarr.open(Path(burden_dir) / "burdens.zarr")[:, :, repeat]
    x_pheno = zarr.open(Path(burden_dir) / "x.zarr")[:]
    genes = pd.Series(np.load(Path(burden_dir) / "genes.npy"))

    if sample_file is not None:
        with open(sample_file, "rb") as f:
            samples = pickle.load(f)["association_samples"]
        if debug:
            samples = [s for s in samples if s < 1000]
        burdens = burdens[samples]
        y = y[samples]
        x_pheno = x_pheno[samples]

    n_samples = burdens.shape[0]
    assert y.shape[0] == n_samples
    assert x_pheno.shape[0] == n_samples
    assert len(genes) == burdens.shape[1]

    nan_mask = ~np.isnan(y).squeeze()
    y = y[nan_mask]
    burdens = burdens[nan_mask]
    x_pheno = x_pheno[nan_mask]

    with open(config_file) as f:
        config = yaml.safe_load(f)

    if gene_file is not None:
        logger.info("Loading gene names")
        gene_df = pd.read_parquet(gene_file, engine="pyarrow")
        gene_df.set_index("id")
        genes = gene_df.loc[genes, "gene"].str.split(".").apply(lambda x: x[0])

    chunk_size = math.ceil(len(genes) / n_chunks)
    chunk_start = chunk * chunk_size
    chunk_end = min(len(genes), chunk_start + chunk_size)
    if chunk == n_chunks - 1:
        assert chunk_end == len(genes)
    gene_indices = np.arange(chunk_start, chunk_end)
    genes = genes.iloc[chunk_start:chunk_end]

    associations = regress_(
        config,
        use_bias,
        burdens,
        y,
        gene_indices,
        genes,
        x_pheno,
        do_scoretest=do_scoretest,
    )

    logger.info("Saving results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    associations.to_parquet(
        Path(out_dir) / f"burden_associations_{chunk}.parquet",
        engine="pyarrow",
    )


@cli.command()
@click.option("--model-name", type=str)
@click.argument("result-files", type=click.Path(exists=True), nargs=-1)
@click.argument("out-file", type=click.Path())
def combine_regression_results(
    result_files: Tuple[str], out_file: str, model_name: Optional[str]
):
    """
    Combine multiple regression result files.

    :param result_files: List of paths to regression result files.
    :type result_files: Tuple[str]
    :param out_file: Path to the output file.
    :type out_file: str
    :param model_name: Name of the regression model.
    :type model_name: Optional[str]
    :return: Concatenated regression results saved to a parquet file.
    """
    logger.info(f"Concatenating results")
    results = pd.concat([pd.read_parquet(f, engine="pyarrow") for f in result_files])

    if model_name is not None:
        logger.info(f'Settting model column to "{model_name}"')
        results["model"] = model_name

    logger.info(f"Writing to {out_file}")
    results.to_parquet(out_file, engine="pyarrow")


if __name__ == "__main__":
    cli()

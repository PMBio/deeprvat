import copy
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
import numpy as np
import pandas as pd
import pyranges as pr
import torch
import torch.nn as nn
import statsmodels.api as sm
import yaml
from bgen import BgenWriter
from numcodecs import Blosc, JSON
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

AGG_FCT = {"mean": np.mean, "max": np.max}


def get_burden(
    batch: Dict,
    agg_models: Dict[str, List[nn.Module]],
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        for key in sorted(list(agg_models.keys()), key=lambda x: int(x.split("_")[1])):
            this_agg_models = agg_models[key]
            this_burden: torch.Tensor = sum([m(X) for m in this_agg_models]) / len(
                this_agg_models
            )
            burden.append(this_burden.cpu().numpy())
        burden = np.concatenate(burden, axis=2)

    sample_ids = batch["sample"]

    return burden, sample_ids


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
    data_key: str = "association_testing_data",
    skip_genotypes: bool = False,
    samples: Optional[List[int]] = None,
) -> Dataset:
    """
    Create a dataset based on the configuration.

    :param config: Configuration dictionary.
    :type config: Dict
    :param debug: Flag for debugging, defaults to False.
    :type debug: bool
    :param data_key: Key for dataset configuration in the config dictionary, defaults to "association_testing_data".
    :type data_key: str
    :param skip_genotypes: Retrieve only covariates and phenotypes, not genotypes
    :type skip_genotypes: bool
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
            data_config["variant_file"],
            split="",
            skip_y_na=False,
            **copy.deepcopy(data_config["dataset_config"]),
            return_genotypes=(not skip_genotypes),
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
@click.option("--data-key", type=str, default="association_testing_data")
@click.option("--skip-genotypes", is_flag=True)
@click.argument("config-file", type=click.Path(exists=True))
@click.argument("out-file", type=click.Path())
def make_dataset(
    debug: bool, data_key: str, skip_genotypes: bool, config_file: str, out_file: str
):
    """
    Create a dataset based on the provided configuration and save to a pickle file.

    :param debug: Flag for debugging.
    :type debug: bool
    :param data_key: Key for dataset configuration in the config dictionary, defaults to "association_testing_data".
    :type data_key: str
    :param skip_genotypes: Retrieve only covariates and phenotypes, not genotypes
    :type skip_genotypes: bool
    :param config_file: Path to the configuration file.
    :type config_file: str
    :param out_file: Path to the output file.
    :type out_file: str
    :return: Created dataset saved to out_file.pkl
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)

    ds = make_dataset_(
        config, debug=debug, data_key=data_key, skip_genotypes=skip_genotypes
    )

    with open(out_file, "wb") as f:
        pickle.dump(ds, f)


def compute_xy_(
    config: Dict,
    ds: torch.utils.data.Dataset,
    data_key="association_testing_data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute burdens using the PyTorch model for each repeat.

    :param config: Configuration dictionary.
    :type config: Dict
    :param ds: Torch dataset.
    :type ds: torch.utils.data.Dataset
    :param data_key: Key for dataset configuration in the config dictionary, defaults to "association_testing_data".
    :type data_key: str
    :return: Tuple containing sample IDs, covariates x, and target phenotypes y
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]

    .. note::
        Checkpoint models all corresponding to the same repeat are averaged for that repeat.
    """
    data_config = config[data_key]

    ds_full = ds.dataset if isinstance(ds, Subset) else ds
    collate_fn = getattr(ds_full, "collate_fn", None)
    ds_full.return_genotypes = False
    n_samples = len(ds)

    dataloader_config = data_config["dataloader_config"]
    batch_size = n_samples // 100
    dataloader_config["batch_size"] = batch_size
    # dataloader_config["num_workers"] = 0

    dl = DataLoader(ds, collate_fn=collate_fn, **dataloader_config)

    logger.info("Computing covariates and target phenotypes")
    data_lists = {"sample": [], "x": [], "y": []}
    for batch in tqdm(
        dl,
        file=sys.stdout,
        total=(n_samples // batch_size + (n_samples % batch_size != 0)),
    ):
        data_lists["sample"].append(batch["sample"])
        data_lists["x"].append(batch["x_phenotypes"])
        data_lists["y"].append(batch["y"])

    data = {k: np.concatenate(v) for k, v in data_lists.items()}

    return data["sample"], data["x"], data["y"]


@cli.command()
@click.option("--dataset-file", type=click.Path(exists=True))
@click.option("--data-key", type=str, default="association_testing_data")
@click.argument("data-config-file", type=click.Path(exists=True))
@click.argument("sample-file", type=click.Path(path_type=Path))
@click.argument("x-file", type=click.Path(path_type=Path))
@click.argument("y-file", type=click.Path(path_type=Path))
def compute_xy(
    dataset_file: Optional[str],
    data_key: str,
    data_config_file: str,
    sample_file: Path,
    x_file: Path,
    y_file: Path,
):
    """
    Compute burdens based on the provided model and dataset.

    :param dataset_file: Path to the dataset file, i.e., association_dataset.pkl.
    :type dataset_file: Optional[str]
    :param data_key: Key for dataset configuration in the config dictionary, defaults to "association_testing_data".
    :type data_key: str
    :param data_config_file: Path to the data configuration file.
    :type data_config_file: str
    :param out_dir: Path to the output directory.
    :type out_dir: str
    """
    with open(data_config_file) as f:
        data_config = yaml.safe_load(f)

    if dataset_file is not None:
        logger.info("Loading pickled dataset")
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = make_dataset_(data_config, skip_genotypes=True)

    sample_ids, x, y = compute_xy_(
        data_config,
        dataset,
        data_key=data_key,
    )

    logger.info("Saving computed sample IDs, covariates, and targets")
    zarr.save_array(sample_file, sample_ids)
    zarr.save_array(x_file, x)
    zarr.save_array(y_file, y)


def make_regenie_input_(
    debug: bool,
    skip_covariates: bool,
    skip_phenotypes: bool,
    skip_burdens: bool,
    burdens_genes_samples: Optional[Tuple[Path, Path, Path]],
    repeat: int,
    phenotype: Tuple[Tuple[str, Path, Path, Path]],
    sample_file: Optional[Path],
    covariate_file: Optional[Path],
    phenotype_file: Optional[Path],
    bgen: Optional[Path],
    gene_metadata_file: Path,
    gtf: Path,
):
    logger.setLevel(logging.INFO)

    ## Check options
    if not skip_burdens and burdens_genes_samples is None:
        raise ValueError("--burdens-genes must be specified if --skip-burdens is not")
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
    xy_dirs = [p[2] for p in phenotype]

    # load only first sample_ids zarr here
    sample_ids = zarr.load(xy_dirs[0] / "sample_ids.zarr")
    covariates = zarr.load(xy_dirs[0] / "x.zarr")
    ys = [zarr.load(b / "y.zarr") for b in xy_dirs]

    if debug:
        sample_ids = sample_ids[:1000]
        covariates = covariates[:1000]
        ys = [y[:1000] for y in ys]

    n_samples = sample_ids.shape[0]
    assert covariates.shape[0] == n_samples
    # assert that ALL y.zarrs are the same lengths as the single sample_ids zarr loaded above
    assert all([y.shape[0] == n_samples for y in ys])

    # Sanity check: sample_ids and covariates should be consistent for all phenotypes
    if not debug:
        for i in range(1, len(phenotype)):
            assert np.array_equal(sample_ids, zarr.load(xy_dirs[i] / "sample_ids.zarr"))

            this_cov = zarr.load(xy_dirs[i] / "x.zarr")
            assert this_cov.shape == covariates.shape
            unequal_rows = np.array(
                [
                    i
                    for i in range(covariates.shape[0])
                    if not np.array_equal(covariates[i], this_cov[i])
                ]
            )
            for i in unequal_rows:
                assert np.all(
                    (np.abs(covariates[i] - this_cov[i]) < 1e-6)
                    | (np.isnan(covariates[i]) & np.isnan(this_cov[i]))
                )

    sample_df = pd.DataFrame({"FID": sample_ids, "IID": sample_ids})

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
        burden_file, gene_file, b_sample_file = burdens_genes_samples

        genes = np.load(gene_file)
        n_genes = genes.shape[0]

        sample_ids = zarr.load(
            b_sample_file
        )  # Might be different from those for the phenotypes
        n_samples = sample_ids.shape[0]

        ## Make sample file
        logger.info(f"Creating sample file {sample_file}")
        sample_df = pd.DataFrame({"FID": sample_ids, "IID": sample_ids})
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

        burdens_zarr = zarr.open(burden_file)
        if not debug:
            assert burdens_zarr.shape[0] == n_samples
            assert burdens_zarr.shape[1] == n_genes

        if len(burdens_zarr.shape) == 2:
            burdens = burdens_zarr[:n_samples, :]
        else:
            if not len(burdens_zarr.shape) == 3 or not burdens_zarr.shape[2] == 1:
                raise ValueError(
                    f"Expected {burden_file} to have shape (n_samples, n_genes) "
                    f"or (n_samples, n_genes, 1), got {burdens_zarr.shape}"
                )
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
        gene_metadata = pd.read_parquet(gene_metadata_file).set_index("id")
        this_gene_pos = gene_pos.loc[gene_metadata.loc[genes, "gene"]]
        pseudovar_pos = (this_gene_pos.End - this_gene_pos.Start).to_numpy().astype(int)
        ensgids = this_gene_pos.index.to_numpy()

        logger.info(f"Writing pseudovariants to {bgen}")
        with BgenWriter(
            bgen,
            n_samples,
            samples=list(sample_ids.astype(str)),
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
@click.option("--skip-covariates", is_flag=True)
@click.option("--skip-phenotypes", is_flag=True)
@click.option("--skip-burdens", is_flag=True)
@click.option(
    "--burdens-genes-samples",
    type=(
        click.Path(path_type=Path, exists=True),
        click.Path(path_type=Path, exists=True),
        click.Path(path_type=Path, exists=True),
    ),
)
@click.option("--repeat", type=int, default=-1)
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
@click.argument("gene-metadata-file", type=click.Path(exists=True, path_type=Path))
@click.argument("gtf", type=click.Path(exists=True, path_type=Path))
def make_regenie_input(
    debug: bool,
    skip_covariates: bool,
    skip_phenotypes: bool,
    skip_burdens: bool,
    burdens_genes_samples: Optional[Tuple[Path, Path, Path]],
    repeat: int,
    phenotype: Tuple[Tuple[str, Path, Path]],
    sample_file: Optional[Path],
    covariate_file: Optional[Path],
    phenotype_file: Optional[Path],
    bgen: Optional[Path],
    gene_metadata_file: Path,
    gtf: Path,
):
    make_regenie_input_(
        debug=debug,
        skip_covariates=skip_covariates,
        skip_phenotypes=skip_phenotypes,
        skip_burdens=skip_burdens,
        burdens_genes_samples=burdens_genes_samples,
        repeat=repeat,
        phenotype=phenotype,
        sample_file=sample_file,
        covariate_file=covariate_file,
        phenotype_file=phenotype_file,
        bgen=bgen,
        gene_metadata_file=gene_metadata_file,
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

    annotation_file = data_config["association_testing_data"]["dataset_config"][
        "annotation_file"
    ]

    if torch.cuda.is_available():
        logger.info("Using GPU")
        device = torch.device("cuda")
    else:
        logger.info("Using CPU")
        device = torch.device("cpu")

    # plof_df = (
    #     dd.read_parquet(
    #         annotation_file,
    #         columns=data_config["association_testing_data"]["dataset_config"]["rare_embedding"]["config"][
    #             "annotations"
    #         ],
    #     )
    #     .query(" or ".join([f"{c} == 1" for c in PLOF_COLS]))
    #     .compute()
    # )

    plof_df = pd.read_parquet(
        annotation_file,
        columns=data_config["association_testing_data"]["dataset_config"][
            "rare_embedding"
        ]["config"]["annotations"],
    )
    plof_df = plof_df[plof_df[PLOF_COLS].eq(1).any(axis=1)]

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


def compute_burdens_(
    debug: bool,
    config: Dict,
    ds: torch.utils.data.Dataset,
    cache_dir: str,
    agg_models: Dict[str, List[nn.Module]],
    data_key: str = "association_testing_data",
    n_chunks: Optional[int] = None,
    chunk: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    bottleneck: bool = False,
    compression_level: int = 1,
) -> Tuple[np.ndarray, zarr.core.Array, zarr.core.Array]:
    """
    Compute burdens using the PyTorch model for each repeat.

    :param debug: Flag for debugging.
    :type debug: bool
    :param data_key: Key for dataset configuration in the config dictionary, defaults to "association_testing_data".
    :type data_key: str
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
    :return: Tuple containing genes, burdens, target y phenotypes, x phenotypes and sample ids.
    :rtype: Tuple[np.ndarray, zarr.core.Array, zarr.core.Array, zarr.core.Array, zarr.core.Array]

    .. note::
        Checkpoint models all corresponding to the same repeat are averaged for that repeat.
    """
    logger.info("agg_models[*][*].reverse:")
    pprint(
        {repeat: [m.reverse for m in models] for repeat, models in agg_models.items()}
    )

    data_config = config[data_key]

    ds_full = ds.dataset if isinstance(ds, Subset) else ds
    collate_fn = getattr(ds_full, "collate_fn", None)
    n_total_samples = len(ds)
    ds.rare_embedding.skip_embedding = False

    if chunk is not None:
        if n_chunks is None:
            raise ValueError("n_chunks must be specified if chunk is not None")

        chunk_length = math.ceil(n_total_samples / n_chunks)
        chunk_start = chunk * chunk_length
        chunk_end = min(n_total_samples, chunk_start + chunk_length)
        samples = range(chunk_start, chunk_end)
        n_samples_chunk = len(samples)
        ds = Subset(ds, samples)

        logger.info(f"Processing samples in {samples} from {n_total_samples} in total")
    else:
        logger.info("Processing all samples as one chunk.")
        n_samples_chunk = n_total_samples
        chunk_start = 0
        chunk_end = n_samples_chunk

    dataloader_config = data_config["dataloader_config"]

    if torch.cuda.is_available():
        pin_memory = dataloader_config.get("pin_memory", True)

        logger.info(f"CUDA is available, setting pin_memory={pin_memory}")
        dataloader_config["pin_memory"] = pin_memory

    dl = DataLoader(ds, collate_fn=collate_fn, **dataloader_config)

    logger.info("Computing gene impairment scores")
    batch_size = data_config["dataloader_config"]["batch_size"]
    with torch.no_grad():
        burdens_chunk_path = Path(cache_dir) / "chunks" / f"chunk_{chunk}"
        burdens_chunk_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"Writing chunks to {burdens_chunk_path}")
        logger.info(f"Writing chunk to {burdens_chunk_path}")

        for i, batch in tqdm(
            enumerate(dl),
            file=sys.stdout,
            total=(n_samples_chunk // batch_size + (n_samples_chunk % batch_size != 0)),
        ):
            this_burdens, this_sampleid = get_burden(batch, agg_models, device=device)
            if i == 0:
                chunk_burden = np.zeros(
                    shape=(n_samples_chunk,) + this_burdens.shape[1:]
                )
                chunk_sampleid = [""] * n_samples_chunk

                logger.info(f"Batch size: {batch['rare_variant_annotations'].shape}")

                burdens = zarr.open(
                    burdens_chunk_path / "burdens.zarr",
                    mode="a",
                    shape=chunk_burden.shape,
                    chunks=(1000, 1000, 1),
                    dtype=np.float32,
                    compressor=Blosc(clevel=compression_level),
                )
                burdens.attrs["chunk"] = chunk
                logger.info(f"burdens shape: {burdens.shape}")

                sample_ids = zarr.open(
                    burdens_chunk_path / "sample_ids.zarr",
                    mode="a",
                    shape=(n_samples_chunk),
                    chunks=(None),
                    dtype="U200",
                    compressor=Blosc(clevel=compression_level),
                )
                sample_ids.attrs["n_total_samples"] = n_total_samples
                sample_ids.attrs["chunk"] = chunk

            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, chunk_end)  # read from chunk shape

            chunk_burden[start_idx:end_idx] = this_burdens

            chunk_sampleid[start_idx:end_idx] = this_sampleid

            if debug:
                logger.info(
                    "Wrote results for batch indices " f"[{start_idx}, {end_idx - 1}]"
                )

            if bottleneck and i > 20:
                break

        burdens[:] = chunk_burden[:]
        sample_ids[:] = chunk_sampleid[:]

    if torch.cuda.is_available():
        logger.info(
            "Max GPU memory allocated: " f"{torch.cuda.max_memory_allocated(0)} bytes"
        )

    return ds_full.rare_embedding.genes, burdens, sample_ids


@cli.command()
@click.option("--debug", is_flag=True)
@click.option("--bottleneck", is_flag=True)
@click.option("--data-key", type=str, default="association_testing_data")
@click.option("--n-chunks", type=int)
@click.option("--chunk", type=int)
@click.option("--dataset-file", type=click.Path(exists=True))
@click.argument("data-config-file", type=click.Path(exists=True))
@click.argument("model-config-file", type=click.Path(exists=True))
@click.argument("checkpoint-files", type=click.Path(exists=True), nargs=-1)
@click.argument("out-dir", type=click.Path(exists=True))
def compute_burdens(
    debug: bool,
    bottleneck: bool,
    data_key: str,
    n_chunks: Optional[int],
    chunk: Optional[int],
    dataset_file: Optional[str],
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
    :param data_key: Key for dataset configuration in the config dictionary, defaults to "association_testing_data".
    :type data_key: str
    :param n_chunks: Number of chunks to split data for processing, defaults to None.
    :type n_chunks: Optional[int]
    :param chunk: Index of the chunk of data, defaults to None.
    :type chunk: Optional[int]
    :param dataset_file: Path to the dataset file, i.e., association_dataset.pkl.
    :type dataset_file: Optional[str]
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
        dataset = make_dataset_(data_config)

    if torch.cuda.is_available():
        logger.info("Using GPU")
        device = torch.device("cuda")
    else:
        logger.info("Using CPU")
        device = torch.device("cpu")

    agg_models = load_models(model_config, checkpoint_files, device=device)

    genes, _, _ = compute_burdens_(
        debug,
        data_config,
        dataset,
        out_dir,
        agg_models,
        data_key=data_key,
        n_chunks=n_chunks,
        chunk=chunk,
        device=device,
        bottleneck=bottleneck,
    )

    logger.info("Saving computed burdens, corresponding genes, and targets")
    np.save(Path(out_dir) / "genes.npy", genes)


@cli.command()
@click.option("--n-chunks", type=int, required=True)
@click.option("--skip-burdens", is_flag=True, default=False)
@click.option("--overwrite", is_flag=True, default=False)
@click.argument("burdens-chunks-dir", type=click.Path(exists=True))
@click.argument("result-dir", type=click.Path(exists=True))
def combine_burden_chunks(
    n_chunks: int,
    skip_burdens: bool,
    overwrite: bool,
    burdens_chunks_dir: Path,
    result_dir: Path,
):
    combine_burden_chunks_(
        n_chunks=n_chunks,
        skip_burdens=skip_burdens,
        overwrite=overwrite,
        burdens_chunks_dir=Path(burdens_chunks_dir),
        result_dir=Path(result_dir),
    )


def combine_burden_chunks_(
    n_chunks: int,
    burdens_chunks_dir: Path,
    skip_burdens: bool,
    overwrite: bool,
    result_dir: Path,
):
    compression_level = 1
    burdens_chunks_dir = Path(burdens_chunks_dir)

    burdens, sample_ids = None, None
    start_id = None
    end_id = 0

    for i, chunk in tqdm(
        enumerate(range(0, n_chunks)), desc=f"Merging {n_chunks} chunks"
    ):
        chunk_dir = burdens_chunks_dir / f"chunk_{chunk}"

        if not skip_burdens:
            burdens_chunk = zarr.open((chunk_dir / "burdens.zarr").as_posix(), mode="r")
            assert burdens_chunk.attrs["chunk"] == chunk

        sample_ids_chunk = zarr.open(
            (chunk_dir / "sample_ids.zarr").as_posix(), mode="r"
        )

        total_samples = sample_ids_chunk.attrs["n_total_samples"]

        assert sample_ids_chunk.attrs["chunk"] == chunk

        burdens_path = result_dir / "burdens.zarr"
        sample_ids_path = result_dir / "sample_ids.zarr"

        if i == 0:
            if not skip_burdens:
                burdens_shape = (total_samples,) + burdens_chunk.shape[1:]

                if not overwrite:
                    assert not burdens_path.exists()
                else:
                    logger.debug("Overwriting existing files")

                logger.debug(f"Opening {burdens_path} in append mode")
                burdens = zarr.open(
                    burdens_path.as_posix(),
                    mode="a",
                    shape=burdens_shape,
                    chunks=(1000, 1000, 1),
                    dtype=np.float32,
                    compressor=Blosc(clevel=compression_level),
                )
                assert burdens_path.exists()

            logger.debug(f"Opening {sample_ids_path} in append mode")
            sample_ids = zarr.open(
                sample_ids_path,
                mode="a",
                shape=(total_samples),
                chunks=(None),
                dtype="U200",
                compressor=Blosc(clevel=compression_level),
            )

            assert sample_ids_path.exists()

        start_id = end_id
        end_id += len(sample_ids_chunk)

        sample_ids[start_id:end_id] = sample_ids_chunk[:]

        if not skip_burdens:
            burdens[start_id:end_id] = burdens_chunk[:]

    logger.info(f"Done merging {n_chunks} chunks.")


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
    assert np.all(burdens != 0)  # because DeepRVAT burdens are corrently all non-zero
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

    logger.info("Computing associations")
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
    y_phenotypes = config["association_testing_data"]["dataset_config"]["y_phenotypes"]
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
@click.option("--do-scoretest", is_flag=True)
@click.option("--sample-file", type=click.Path(exists=True))
@click.argument("config-file", type=click.Path(exists=True))
@click.argument("xy-dir", type=click.Path(exists=True))
@click.argument("burden-file", type=click.Path(exists=True))
@click.argument("out-dir", type=click.Path())
def regress(
    debug: bool,
    chunk: int,
    n_chunks: int,
    use_bias: bool,
    gene_file: str,
    config_file: str,
    xy_dir: str,
    burden_file: str,
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
    :param config_file: Path to the configuration file.
    :type config_file: str
    :param xy_dir: Path to the directory containing the x.zarr and y.zarr files.
    :type xy_dir: str
    :param burden_file: Path to the burdens.zarr file.
    :type burden_file: str
    :param out_dir: Path to the output directory.
    :type out_dir: str
    :param do_scoretest: Flag to use the scoretest from SEAK.
    :type do_scoretest: bool
    :param sample_file: Path to the sample file.
    :type sample_file: Optional[str]
    :return: Regression results saved to out_dir as "burden_associations_{chunk}.parquet"
    """
    burden_dir = Path(burden_file).parent
    # if burden_file is not None:
    #     logger.info(f'Loading burdens from {burden_file}')
    #     burdens = zarr.open(burden_file)[:, :, repeat]
    # else:
    #     burdens = zarr.open(Path(burden_dir) / "burdens.zarr")[:, :, repeat]
    logger.info(f"Loading covariates and targets from {xy_dir}")
    y = zarr.load(Path(xy_dir) / "y.zarr")
    x_pheno = zarr.load(Path(xy_dir) / "x.zarr")

    # Find overlap in sample IDs
    xy_sample_ids = zarr.load(Path(xy_dir) / "sample_ids.zarr")
    burden_sample_ids = zarr.load(Path(burden_dir) / "sample_ids.zarr")
    if len(set(xy_sample_ids).intersection(set(burden_sample_ids))) == 0:
        raise ValueError("No common sample IDs between xy_dir and burden_dir")
    if len(set(xy_sample_ids)) != xy_sample_ids.shape[0]:
        raise ValueError("Duplicate sample IDs in xy_dir")
    if len(set(burden_sample_ids)) != burden_sample_ids.shape[0]:
        raise ValueError("Duplicate sample IDs in burden_dir")

    if sample_file is not None:
        with open(sample_file, "rb") as f:
            samples = pickle.load(f)["association_samples"]
        if debug:
            samples = [s for s in samples if s < 1000]
        # burdens = burdens[samples]
        y = y[samples]
        x_pheno = x_pheno[samples]

    n_samples = y.shape[0]
    try:
        assert y.shape[0] == n_samples
        assert x_pheno.shape[0] == n_samples
    except:
        raise ValueError(
            "Inconsistent number of samples between covariates and targets"
        )

    x_pheno_cols = [f"xy_pheno_{i}" for i in range(x_pheno.shape[1])]
    xy_dict = {
        c: x.squeeze()
        for c, x in zip(x_pheno_cols, np.split(x_pheno, x_pheno.shape[1], axis=1))
    }
    xy_dict["sample"] = xy_sample_ids
    xy_dict["y"] = y.squeeze()
    xy_df = pd.DataFrame(xy_dict)
    burden_sample_df = pd.DataFrame({"sample": burden_sample_ids})
    xy_df = pd.merge(burden_sample_df, xy_df, how="left", sort=False)
    # Assert the rows of this DF correspond to rows of burden_dir/burdens.zarr
    # (should be guaranteed by Pandas, this is just a sanity check)
    assert np.array_equal(xy_df["sample"].to_numpy(), burden_sample_ids)

    with open(config_file) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loading saved burdens from {burden_dir}")
    genes = pd.Series(np.load(Path(burden_dir) / "genes.npy"))
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
    # gene_indices = np.arange(chunk_start, chunk_end)

    genes = genes.iloc[chunk_start:chunk_end]
    gene_indices = np.arange(len(genes))

    logger.info(f"Only extracting genes in range {chunk_start, chunk_end}")
    burdens = zarr.open(burden_file)[:, chunk_start:chunk_end, 0]

    if sample_file is not None:
        burdens = burdens[samples]
    # burdens = burdens[nan_mask]
    assert len(genes) == burdens.shape[1]

    associations = regress_(
        config,
        use_bias,
        burdens,
        xy_df["y"].to_numpy(),
        gene_indices,
        genes,
        xy_df[x_pheno_cols].to_numpy(),
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
    logger.info("Concatenating results")
    results = pd.concat([pd.read_parquet(f, engine="pyarrow") for f in result_files])

    if model_name is not None:
        logger.info(f'Settting model column to "{model_name}"')
        results["model"] = model_name

    logger.info(f"Writing to {out_file}")
    results.to_parquet(out_file, engine="pyarrow")


@cli.command()
@click.option("--n-chunks", type=int)
@click.option("--chunk", type=int)
@click.option("-r", "--repeats", multiple=True, type=int)
@click.option("--agg-fct", type=str, default="mean")
@click.argument("burden-file", type=click.Path(exists=True))
@click.argument("burden-out-file", type=click.Path())
def average_burdens(
    repeats: Tuple,
    burden_file: str,
    burden_out_file: str,
    agg_fct: Optional[str] = "mean",
    n_chunks: Optional[int] = None,
    chunk: Optional[int] = None,
):
    compression_level = 1
    logger.info(f"Analyzing repeats {repeats}")
    logger.info(f"Reading burdens to aggregate from {burden_file}")
    burdens = zarr.open(burden_file)
    n_total_samples = burdens.shape[0]
    if chunk is not None:
        if n_chunks is None:
            raise ValueError("n_chunks must be specified if chunk is not None")
        chunk_length = math.ceil(n_total_samples / n_chunks)
        chunk_start = chunk * chunk_length
        chunk_end = min(n_total_samples, chunk_start + chunk_length)
        samples = range(chunk_start, chunk_end)
        n_samples = len(samples)
        print(chunk_start, chunk_end)
    else:
        n_samples = n_total_samples
        chunk_start = 0
        chunk_end = n_samples

    logger.info(
        f"Computing result for chunk {chunk} out of {n_chunks} in range {chunk_start}, {chunk_end}"
    )

    batch_size = 100
    logger.info(f"Batch size: {batch_size}")
    n_batches = n_samples // batch_size + (n_samples % batch_size != 0)

    logger.info(f"Using aggregation function {agg_fct}")
    for i in tqdm(
        range(n_batches),
        file=sys.stdout,
        total=(n_samples // batch_size + (n_samples % batch_size != 0)),
    ):
        if i == 0:
            # if not os.path.exists(burden_out_file):
            #     logger.info('Generting new zarr file')
            burdens_new = zarr.open(
                burden_out_file,
                mode="a",
                shape=(burdens.shape[0], burdens.shape[1], 1),
                chunks=(1000, 1000),
                dtype=np.float32,
                compressor=Blosc(clevel=compression_level),
            )
            # else:
            #     logger.info('Only opening zarr file')
            #     burdens_new =  zarr.open(burden_out_file)

        start_idx = chunk_start + i * batch_size
        end_idx = min(start_idx + batch_size, chunk_end)
        print(start_idx, end_idx)
        this_burdens = np.take(burdens[start_idx:end_idx, :, :], repeats, axis=2)
        this_burdens = AGG_FCT[agg_fct](this_burdens, axis=2)

        burdens_new[start_idx:end_idx, :, 0] = this_burdens

    logger.info(
        f"Writing aggregted burdens in range {chunk_start}, {chunk_end} to {burden_out_file}"
    )


# TODO merge these functions into regress(), regress_
@cli.command()
@click.option("--debug", is_flag=True)
@click.option("--chunk", type=int, default=0)
@click.option("--n-chunks", type=int, default=1)
@click.option("--use-bias", is_flag=True)
@click.option("--gene-file", type=click.Path(exists=True))
@click.option("--repeat", type=int, default=0)
@click.option("--do-scoretest", is_flag=True)
@click.option("--sample-file", type=click.Path(exists=True))
@click.option("--burden-file", type=click.Path(exists=True))
@click.option("--genes-to-keep", type=click.Path(exists=True))
@click.option("--common-genotype-prefix", type=str)
@click.argument("config-file", type=click.Path(exists=True))
@click.argument("burden-dir", type=click.Path(exists=True))
@click.argument("out-file", type=click.Path())
def regress_common(
    debug: bool,
    chunk: int,
    n_chunks: int,
    use_bias: bool,
    gene_file: str,
    repeat: int,
    config_file: str,
    burden_dir: str,
    out_file: str,
    do_scoretest: bool,
    sample_file: Optional[str],
    burden_file: Optional[str],
    genes_to_keep: Optional[str],
    common_genotype_prefix: str,
):
    logger.info("Loading saved burdens")
    # if burden_file is not None:
    #     logger.info(f'Loading burdens from {burden_file}')
    #     burdens = zarr.open(burden_file)[:, :, repeat]
    # else:
    #     burdens = zarr.open(Path(burden_dir) / "burdens.zarr")[:, :, repeat]
    logger.info(f"Loading x, y, genes from {burden_dir}")
    y = zarr.open(Path(burden_dir) / "y.zarr")[:]
    x_pheno = zarr.open(Path(burden_dir) / "x.zarr")[:]
    genes = pd.Series(np.load(Path(burden_dir) / "genes.npy"))

    if genes_to_keep is not None:
        logger.info(f"Reading genes_to_keep file from {genes_to_keep}")
        genes_to_keep = np.load(genes_to_keep)

    if sample_file is not None:
        with open(sample_file, "rb") as f:
            samples = pickle.load(f)["association_samples"]
        if debug:
            samples = [s for s in samples if s < 1000]
        # burdens = burdens[samples]
        y = y[samples]
        x_pheno = x_pheno[samples]

    n_samples = y.shape[0]
    assert y.shape[0] == n_samples
    assert x_pheno.shape[0] == n_samples
    # assert len(genes) == burdens.shape[1]

    # TODO commented this out. is this a problem?
    # nan_mask = ~np.isnan(y).squeeze()
    # y = y[nan_mask]
    # # burdens = burdens[nan_mask]
    # x_pheno = x_pheno[nan_mask]

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
    # gene_indices = np.arange(chunk_start, chunk_end)

    # gene_indices = np.arange(chunk_start, chunk_end)
    logger.info(f"processing genes in range {chunk_start}, {chunk_end}")
    all_genes = copy.deepcopy(genes)
    genes = genes.iloc[chunk_start:chunk_end]
    if genes_to_keep is not None:
        # genes_this_chunk = set(genes).intersection(set(genes_to_keep))
        genes_this_chunk = [
            i for i in genes_to_keep if i in list(genes)
        ]  # having list is important, otherwise 'in' checks the indices, not the values in the pd.Series
        gene_indices = np.array(
            [np.where(all_genes == this_gene)[0][0] for this_gene in genes_this_chunk]
        )
        genes = pd.Series(list(genes_this_chunk))

    logger.info(f"Only extracting genes in range {chunk_start, chunk_end}")

    if burden_file is not None:
        logger.info(f"Loading burdens from {burden_file}")
    else:
        burden_file = Path(burden_dir) / "burdens.zarr"

    if genes_to_keep is not None:
        logger.info(f"Loading burdens at position {gene_indices}")
        burdens = zarr.open(burden_file)
        burdens = burdens.oindex[:, gene_indices, repeat]
        gene_indices = np.arange(len(genes))
    else:
        burdens = zarr.open(burden_file)[:, chunk_start:chunk_end, repeat]

    gene_indices = np.arange(len(genes))

    if sample_file is not None:
        burdens = burdens[samples]
    # burdens = burdens[nan_mask]
    assert len(genes) == burdens.shape[1]
    logger.info(common_genotype_prefix)
    associations = regress_common_(
        config,
        use_bias,
        burdens,
        y,
        gene_indices,
        genes,
        x_pheno,
        common_genotype_prefix,
        do_scoretest=do_scoretest,
    )

    logger.info("Saving results")
    # Path(out_dir).mkdir(parents=True, exist_ok=True)
    # associations.to_parquet(
    #     Path(out_dir) / f"burden_associations_{chunk}.parquet",
    #     engine="pyarrow",
    # )
    associations.to_parquet(Path(out_file), engine="pyarrow")


def regress_common_(
    config: Dict,
    use_bias: bool,
    burdens: np.ndarray,
    y: np.ndarray,
    gene_indices: np.ndarray,
    genes: pd.Series,
    x_pheno: np.ndarray,
    common_genotype_prefix: str,
    use_x_pheno: bool = True,
    do_scoretest: bool = True,
) -> pd.DataFrame:
    assert len(gene_indices) == len(genes)
    logger.info(common_genotype_prefix)

    logger.info("Computing associations")
    logger.info(f"Covariates shape: {x_pheno.shape}, y shape: {y.shape}")

    regressed_genes = []
    betas = []
    pvals = []
    logger.info("Running regression on each gene")
    genes_betas_pvals = []
    # for i, gene in tqdm(
    #     zip(gene_indices, genes), total=genes.shape[0], file=sys.stdout):
    mask = ~np.isnan(y).reshape(-1)
    y = y[mask]
    for i, gene in zip(gene_indices, genes):
        logger.info(f"rergressing on gene {gene}")
        if common_genotype_prefix is not None:
            logger.info(
                f"Reading commong genotypes from {common_genotype_prefix}_{gene}.zarr"
            )
            common_genotypes = zarr.open(Path(f"{common_genotype_prefix}_{gene}.zarr"))[
                :
            ]

            logger.info(f"common genotypes shape: {common_genotypes.shape}")

            assert common_genotypes.shape[0] == x_pheno.shape[0]
            X = np.hstack((x_pheno, common_genotypes))
        if do_scoretest:
            logger.info("Running regression on each gene using scoretest from SEAK")
            X = np.hstack((np.ones((X.shape[0], 1)), X))[mask]
            # adding bias column
            logger.info(f"X shape: {X.shape}, Y shape: {y.shape}")

            # compute null_model for score test
            if len(np.unique(y)) == 2:
                logger.info(
                    "Fitting binary model since only found two distinct y values"
                )
                model_score = scoretest.ScoretestLogit(y, X)
            else:
                logger.info("Fitting linear model")
                model_score = scoretest.ScoretestNoK(y, X)
            gene_stats = regress_on_gene_scoretest(gene, burdens[mask, i], model_score)
        else:
            logger.info("Running regression on each gene using OLS")
            gene_stats = regress_on_gene(
                gene, burdens[:, i], y, X, use_bias, use_x_pheno
            )

        genes_betas_pvals.append(gene_stats)
    genes_betas_pvals = [x for x in genes_betas_pvals if x is not None]
    regressed_genes, betas, pvals = separate_parallel_results(genes_betas_pvals)
    y_phenotypes = config["association_testing_data"]["dataset_config"]["y_phenotypes"]
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


if __name__ == "__main__":
    cli()

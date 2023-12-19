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
import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import statsmodels.api as sm
import yaml
from numcodecs import Blosc
from seak import scoretest
from statsmodels.tools.tools import add_constant
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import zarr
import re

import deeprvat.deeprvat.models as deeprvat_models
from deeprvat.data import DenseGTDataset

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
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
    skip_burdens=False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    :param use_common: Flag to add common variant genotype information to x_pheno, defaults to False.
    :type use_common: bool
    :return: Tuple containing burden scores, target y phenotype values, and x phenotypes.
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

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
    x = batch["x_phenotypes"]  # containes other covariates e.g. age, genetic PCs
    cvar = batch["common_variants"].numpy()

    return burden, y, x, cvar


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
            in_association=True,  # enable alternative logic if group_common is activated
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
) -> Tuple[np.ndarray, zarr.core.Array, zarr.core.Array, zarr.core.Array]:
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
    :return: Tuple containing genes, burdens, target y phenotypes, and x phenotypes.
    :rtype: Tuple[np.ndarray, zarr.core.Array, zarr.core.Array, zarr.core.Array]

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
    use_common = data_config["dataset_config"]["use_common_variants"]
    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(dl),
            file=sys.stdout,
            total=(n_samples // batch_size + (n_samples % batch_size != 0)),
        ):
            # run forward pass on all repeats to get gene burden
            this_burdens, this_y, this_x, this_cvar = get_burden(
                batch, agg_models, device=device, skip_burdens=skip_burdens
            )

            if i == 0:
                if not skip_burdens:
                    chunk_burden = np.zeros(shape=(n_samples,) + this_burdens.shape[1:])
                chunk_y = np.zeros(shape=(n_samples,) + this_y.shape[1:])
                chunk_x = np.zeros(shape=(n_samples,) + this_x.shape[1:])

                if use_common:
                    chunk_cvar = np.zeros(shape=(n_samples,) + this_cvar.shape[1:])

                logger.info(f"Batch size: {batch['rare_variant_annotations'].shape}")

                if not skip_burdens:
                    burdens = zarr.open(
                        Path(cache_dir) / "burdens.zarr",
                        mode="a",
                        shape=(n_total_samples,) + this_burdens.shape[1:],
                        chunks=(1000, 1000),
                        dtype=np.float32,
                        compressor=Blosc(clevel=compression_level),
                    )
                    logger.info(f"burdens shape: {burdens.shape}")
                else:
                    burdens = None

                if use_common:
                    cvar = zarr.open(
                        Path(cache_dir) / "common_variants.zarr",
                        mode="a",
                        shape=(n_total_samples,) + this_cvar.shape[1:],
                        chunks=(1000, 1000),
                        dtype=np.float32,
                        compressor=Blosc(clevel=compression_level),
                    )
                    logger.info(f"common genotype shape: {cvar.shape}")

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

            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, chunk_end)  # read from chunk shape

            if not skip_burdens:
                chunk_burden[start_idx:end_idx] = this_burdens

            if use_common:
                chunk_cvar[start_idx:end_idx] = this_cvar

            chunk_y[start_idx:end_idx] = this_y
            chunk_x[start_idx:end_idx] = this_x

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

        if use_common:
            cvar[chunk_start:chunk_end] = chunk_cvar

    if torch.cuda.is_available():
        logger.info(
            "Max GPU memory allocated: " f"{torch.cuda.max_memory_allocated(0)} bytes"
        )

    if use_common:
        # build dict for gene group mapping on common genotype vector
        # and store as pickle in burden dir
        cvar_group_dict = {
            gene: group_indices for gene, group_indices in zip(ds_full.group_names, ds_full.group_matrix_maps)
        }

        with open(Path(cache_dir) / "common_variants_group_map.pkl", 'wb') as file:
            pickle.dump(cvar_group_dict, file)

    return ds_full.rare_embedding.genes, burdens, y, x


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
    :return: Corresonding genes, computed burdens, y phenotypes, and x phenotypes are saved in the out_dir.
    :rtype: [np.ndarray], [zarr.core.Array], [zarr.core.Array], [zarr.core.Array]

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

    genes, _, _, _ = compute_burdens_(
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
    common_var_genotype: np.ndarray = None,
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
    :param common_var_genotype: common variant genotype vector
    : type common_var_genotype: np.ndarray

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

    if common_var_genotype is not None:
        # add common variant genotype vector to X
        X = np.concatenate((X, common_var_genotype.T), axis=1)

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
    cvar_vector_dict: Dict[int, np.array] = None,
    cvar_vector_arr: np.ndarray = None
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
    :param cvar_vector_dict: dictionary of common variant genotype data per gene.
    :type cvar_vector_dict: Dict[int, np.array]
    :param cvar_vector_arr: numpy ndarray of common variant genotype for all genes.
    :type cvar_vector_arr: np.ndarray
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
        if cvar_vector_dict is not None:
            # regression with common variant genotype per gene
            logger.info("Running regression on each gene and gene common variant genotype using OLS")
            genes_betas_pvals = [
                regress_on_gene(
                    gene,
                    burdens[:, i],
                    y,
                    x_pheno,
                    use_bias,
                    use_x_pheno,
                    common_var_genotype=cvar_vector_dict[i],
                )
                for i, gene in tqdm(
                    zip(gene_indices, genes), total=genes.shape[0], file=sys.stdout
                )
            ]
        elif cvar_vector_arr is not None:
            # regression with full common variant genotype
            logger.info("Running regression on each gene using OLS")
            genes_betas_pvals = [
                regress_on_gene(
                    gene,
                    burdens[:, i],
                    y,
                    x_pheno,
                    use_bias,
                    use_x_pheno,
                    common_var_genotype=cvar_vector_arr,
                )
                for i, gene in tqdm(
                    zip(gene_indices, genes), total=genes.shape[0], file=sys.stdout
                )
            ]
        else:
            # regression with rare only
            logger.info("Running regression on each gene and full common variant genotype using OLS")
            genes_betas_pvals = [
                regress_on_gene(
                    gene,
                    burdens[:, i],
                    y,
                    x_pheno,
                    use_bias,
                    use_x_pheno,
                )
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

    use_common = config["data"]["dataset_config"]["use_common_variants"]
    group_common = config["data"]["dataset_config"]["group_common"]

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

    cvar_vector_dict = None
    sample_indices = None

    # do everything as normal for the files needed for rare association
    # in addition to gene chunks, also introduce sample chunks

    if use_common:

        # load additional files if common variant data should be added
        cvar = zarr.open(Path(burden_dir) / "common_variants.zarr")

        # build index array for samples
        # apply nan mask to slice coordinates
        sample_indices = np.arange(0, len(cvar))[nan_mask]

        if group_common:
            # Do prev introduced code
            with open(Path(burden_dir) / "common_variants_group_map.pkl", 'rb') as file:
                cvar_group_map = pickle.load(file)

            # build dict containing needed genotype positions per gene in chunk
            cvar_vector_dict = {}


            for g_i in gene_indices:
                g_i_cvar = None
                # load common variant information for required genes only
                if g_i in cvar_group_map.keys():
                    # load common variant genotype data for gene i
                    var_indices = np.expand_dims(cvar_group_map[g_i], axis=1)
                    g_i_cvar = cvar.get_coordinate_selection(
                        (sample_indices, var_indices)
                    )

                cvar_vector_dict[g_i] = g_i_cvar

            associations = regress_(
                config,
                use_bias,
                burdens,
                y,
                gene_indices,
                genes,
                x_pheno,
                cvar_vector_dict=cvar_vector_dict,
                do_scoretest=do_scoretest,
            )
        else:
            # Full common geno regression

            var_indices = np.arange(0, cvar.shape[1])
            mask_cvar = cvar.get_coordinate_selection(
                (sample_indices, var_indices)
            )

            associations = regress_(
                config,
                use_bias,
                burdens,
                y,
                gene_indices,
                genes,
                x_pheno,
                cvar_vector_arr=mask_cvar,
                do_scoretest=False,
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

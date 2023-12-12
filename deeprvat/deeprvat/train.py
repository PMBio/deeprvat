import copy
import gc
import itertools
import logging
import pickle
import random
import shutil
import sys
from pathlib import Path
from pprint import pformat, pprint
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Tuple

import click
import math
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F
import deeprvat.deeprvat.models as deeprvat_models
import torch
import yaml
import zarr
from deeprvat.data import DenseGTDataset
from deeprvat.metrics import (
    AveragePrecisionWithLogits,
    PearsonCorr,
    PearsonCorrTorch,
    RSquared,
)
from deeprvat.utils import resolve_path_with_env, suggest_hparams
from numcodecs import Blosc
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import deeprvat.deeprvat.models as deeprvat_models
from deeprvat.data import DenseGTDataset
from deeprvat.metrics import (
    PearsonCorr,
    PearsonCorrTorch,
    RSquared,
    AveragePrecisionWithLogits,
    QuantileLoss,
    KLDIVLoss,
    BCELoss,
    LassoLossTrain,
    LassoLossVal,
)
from deeprvat.utils import suggest_hparams

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy("file_system")

METRICS = {
    "Huber": nn.SmoothL1Loss,
    "MAE": nn.L1Loss,
    "MSE": nn.MSELoss,
    "RSquared": RSquared,
    "PearsonCorr": PearsonCorr,
    "PearsonCorrTorch": PearsonCorrTorch,
    "BCEWithLogits": nn.BCEWithLogitsLoss,
    "AveragePrecisionWithLogits": AveragePrecisionWithLogits,
    "QuantileLoss": QuantileLoss,
    "KLDiv": KLDIVLoss, 
    "BCELoss": BCELoss,
    "LassoLossTrain": LassoLossTrain,
    "LassoLossVal": LassoLossVal,
}
OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sparse_adam": optim.SparseAdam,
}
ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
}

DEFAULT_OPTIMIZER = {"type": "adamw", "config": {}}


@click.group()
def cli():
    pass


def subset_samples(
    input_tensor: torch.Tensor,
    covariates: torch.Tensor,
    y: torch.Tensor,
    min_variant_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # First sum over annotations (dim 2) for each variant in each gene.
    # Then get the number of non-zero values across all variants in all
    # genes for each sample.
    n_samples_orig = input_tensor.shape[0]

    # n_variants_per_sample = np.sum(
    #     np.sum(input_tensor.numpy(), axis=2) != 0, axis=(1, 2)
    # )
    # n_variant_mask = n_variants_per_sample >= min_variant_count
    n_variant_mask = (
        np.sum(np.any(input_tensor.numpy(), axis=(1, 2)), axis=1) >= min_variant_count
    )

    # Also make sure we don't have NaN values for y
    nan_mask = ~y.squeeze().isnan()
    mask = n_variant_mask & nan_mask.numpy()

    # Subset all the tensors
    input_tensor = input_tensor[mask]
    covariates = covariates[mask]
    y = y[mask]

    logger.info(f"{input_tensor.shape[0]} / {n_samples_orig} samples kept")

    return input_tensor, covariates, y


def make_dataset_(
    config: Dict,
    debug: bool = False,
    training_dataset_file: Optional[str] = None,
    pickle_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_phenotypes = config.get("n_phenotypes", None)
    if n_phenotypes is not None:
        if "seed_genes" in config:
            pheno_codings = config["seed_genes"]["phenocodes_codings"]
            config["seed_genes"]["phenocodes_codings"] = pheno_codings[:n_phenotypes]

        for key in ("data", "training_data"):
            y_phenotypes = config[key]["dataset_config"]["y_phenotypes"]
            config[key]["dataset_config"]["y_phenotypes"] = y_phenotypes[:n_phenotypes]

        logger.info(f"Using {n_phenotypes} phenotypes:")
        pprint(config["data"]["dataset_config"]["y_phenotypes"])

    training_config = config["training"]

    use_x_pheno = training_config.get("use_x_pheno", True)
    logger.info(f"Using x_phenotypes for training and regression: {use_x_pheno}")

    logger.info(training_config)

    config["training_data"]["dataset_config"]["gene_file"] = config["seed_gene_file"]
    config["training_data"]["dataset_config"]["rare_embedding"]["config"][
        "gene_file"
    ] = config["seed_gene_file"]

    logger.info("Getting dataset")
    if (
        pickle_only
        or training_dataset_file is None
        or not Path(training_dataset_file).is_file()
    ):
        ds = DenseGTDataset(
            gt_file=config["training_data"]["gt_file"],
            variant_file=config["training_data"]["variant_file"],
            split="",
            skip_y_na=True,
            **config["training_data"]["dataset_config"],
        )
        if training_dataset_file is not None:
            logger.info("  Pickling dataset")
            with open(training_dataset_file, "wb") as f:
                pickle.dump(ds, f)
        if pickle_only:
            return None, None, None, None
    else:
        logger.info("  Loading saved dataset")
        with open(training_dataset_file, "rb") as f:
            ds = pickle.load(f)

    meta_data = ds.get_metadata()["rare_embedding_metadata"]["genes"]
    n_samples = len(ds)
    collate_fn = ds.collate_fn
    pad_value = ds.rare_embedding.pad_value
    restrict_samples = config.get("restrict_samples", None)
    if debug:
        n_samples = 1000
    elif restrict_samples is not None:
        n_samples = restrict_samples

    logger.info(f"Using {n_samples} samples for training and validation")
    ds = Subset(ds, range(n_samples))

    dl = DataLoader(
        ds, collate_fn=collate_fn, **config["training_data"]["dataloader_config"]
    )

    logger.info("  Generating dataset")
    batches = [
        batch
        for batch in tqdm(
            dl,
            file=sys.stdout,
            total=len(ds) // config["training_data"]["dataloader_config"]["batch_size"],
        )
    ]
    rare_batches = [b["rare_variant_annotations"] for b in batches]
    max_n_variants = max(r.shape[-1] for r in rare_batches)
    logger.info("Building input_tensor, covariates, and y")
    input_tensor = torch.cat(
        [
            F.pad(r, (0, max_n_variants - r.shape[-1]), value=pad_value)
            for r in rare_batches
        ]
    )
    covariates = torch.cat([b["x_phenotypes"] for b in batches])
    y = torch.cat([b["y"] for b in batches])

    return input_tensor, covariates, y, meta_data


@cli.command()
@click.option("--debug", is_flag=True)
@click.option("--pickle-only", is_flag=True)
@click.option("--compression-level", type=int, default=1)
@click.option("--training-dataset-file", type=click.Path())
@click.argument("config-file", type=click.Path(exists=True))
@click.argument("input-tensor-out-file", type=click.Path())
@click.argument("covariates-out-file", type=click.Path())
@click.argument("y-out-file", type=click.Path())
@click.argument("meta-data-out-file", type=click.Path())
def make_dataset(
    debug: bool,
    pickle_only: bool,
    compression_level: int,
    training_dataset_file: Optional[str],
    config_file: str,
    input_tensor_out_file: str,
    covariates_out_file: str,
    y_out_file: str,
    meta_data_out_file: str
):
    with open(config_file) as f:
        config = yaml.safe_load(f)

    input_tensor, covariates, y, meta_data = make_dataset_(
        config,
        debug=debug,
        training_dataset_file=training_dataset_file,
        pickle_only=pickle_only,
    )
    if not pickle_only:
        logger.info("Saving tensors")
        zarr.save_array(
            input_tensor_out_file,
            input_tensor.numpy(),
            chunks=(1000, None, None, None),
            compressor=Blosc(clevel=compression_level),
        )
        del input_tensor
        zarr.save_array(covariates_out_file, covariates.numpy())
        zarr.save_array(y_out_file, y.numpy())
        zarr.save_array(meta_data_out_file, meta_data)


class MultiphenoDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, Dict],
        min_variant_count: int,
        batch_size: int,
        split: str = "train",
        cache_tensors: bool = False,
        temp_dir: Optional[str] = None,
        chunksize: int = 1000,
        # samples: Optional[Union[slice, np.ndarray]] = None,
        # genes: Optional[Union[slice, np.ndarray]] = None
    ):
        "Initialization"
        super().__init__()

        self.data = copy.deepcopy(data)
        self.phenotypes = self.data.keys()
        logger.info(
            f"Initializing MultiphenoDataset with phenotypes:\n{pformat(list(self.phenotypes))}"
        )

        self.cache_tensors = cache_tensors
        if self.cache_tensors:
            self.zarr_root = zarr.group()
        elif temp_dir is not None:
            temp_path = Path(resolve_path_with_env(temp_dir)) / "deeprvat_training"
            temp_path.mkdir(parents=True, exist_ok=True)
            self.input_tensor_dir = TemporaryDirectory(
                prefix="training_data", dir=str(temp_path)
            )
            # Create root group here

        self.chunksize = chunksize
        if self.cache_tensors:
            logger.info("Keeping all input tensors in main memory")

        for pheno, pheno_data in self.data.items():
            if pheno_data["y"].shape == (pheno_data["input_tensor_zarr"].shape[0], 1):
                pheno_data["y"] = pheno_data["y"].squeeze()
            elif pheno_data["y"].shape != (pheno_data["input_tensor_zarr"].shape[0],):
                raise NotImplementedError(
                    "Multi-phenotype training is only implemented via multiple y files"
                )

            if self.cache_tensors:
                zarr.copy(
                    pheno_data["input_tensor_zarr"],
                    self.zarr_root,
                    name=pheno,
                    chunks=(self.chunksize, None, None, None),
                    compressor=Blosc(clevel=1),
                )
                pheno_data["input_tensor_zarr"] = self.zarr_root[pheno]
                # pheno_data["input_tensor"] = pheno_data["input_tensor_zarr"][:]
            elif temp_dir is not None:
                tensor_path = (
                    Path(self.input_tensor_dir.name) / pheno / "input_tensor.zarr"
                )
                zarr.copy(
                    pheno_data["input_tensor_zarr"],
                    zarr.DirectoryStore(tensor_path),
                    chunks=(self.chunksize, None, None, None),
                    compressor=Blosc(clevel=1),
                )
                pheno_data["input_tensor_zarr"] = zarr.open(tensor_path)

        self.min_variant_count = min_variant_count
        self.samples = {
            pheno: pheno_data["samples"][split]
            for pheno, pheno_data in self.data.items()
        }

        # self.subset_samples()

        self.total_samples = sum([s.shape[0] for s in self.samples.values()])

        self.batch_size = batch_size
        self.sample_order = pd.DataFrame(
            {
                "phenotype": itertools.chain(
                    *[[pheno] * len(self.samples[pheno]) for pheno in self.phenotypes]
                )
            }
        )
        self.sample_order = self.sample_order.astype(
            {"phenotype": pd.api.types.CategoricalDtype()}
        )
        self.sample_order = self.sample_order.sample(n=self.total_samples)  # shuffle
        self.sample_order["index"] = self.sample_order.groupby("phenotype").cumcount()

    def __len__(self):
        "Denotes the total number of batches"
        return math.ceil(len(self.sample_order) / self.batch_size)

    def __getitem__(self, index):
  
        start_idx = index * self.batch_size
        end_idx = min(self.total_samples, start_idx + self.batch_size)
        batch_samples = self.sample_order.iloc[start_idx:end_idx]
        samples_by_pheno = batch_samples.groupby("phenotype", observed=True)

        result = dict()
        for pheno, df in samples_by_pheno:
            idx = df["index"].to_numpy()
            assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1))
            slice_ = slice(idx[0], idx[-1] + 1)

            # annotations = (
            #     self.data[pheno]["input_tensor"][slice_]
            #     if self.cache_tensors
            #     else self.data[pheno]["input_tensor_zarr"][slice_, :, :, :]
            # )
            annotations = self.data[pheno]["input_tensor_zarr"][slice_, :, :, :]

            result[pheno] = {
                "indices": self.samples[pheno][slice_],
                "covariates": self.data[pheno]["covariates"][slice_],
                "rare_variant_annotations": torch.tensor(annotations),
                "y": self.data[pheno]["y"][slice_],
                "gene_id": self.data[pheno]["gene_id"][slice_],
            }

        return result

    # NOTE: This function is broken with current cache_tensors behavior
    def subset_samples(self):
        for pheno, pheno_data in self.data.items():
            # First sum over annotations (dim 2) for each variant in each gene.
            # Then get the number of non-zero values across all variants in all
            # genes for each sample.
            n_samples_orig = self.samples[pheno].shape[0]

            # TODO: Compute n_variant_mask one block of 10,000 samples at a time to reduce memory usage
            input_tensor = pheno_data["input_tensor_zarr"].oindex[self.samples[pheno]]
            n_variants_per_sample = np.sum(
                np.sum(input_tensor, axis=2) != 0, axis=(1, 2)
            )
            n_variant_mask = n_variants_per_sample >= self.min_variant_count

            # Also make sure we don't have NaN values for y
            nan_mask = ~pheno_data["y"][self.samples[pheno]].isnan()
            mask = n_variant_mask & nan_mask.numpy()

            # Set the tensor indices to use and subset all the tensors
            self.samples[pheno] = self.samples[pheno][mask]
            pheno_data["y"] = pheno_data["y"][self.samples[pheno]]
            pheno_data["covariates"] = pheno_data["covariates"][self.samples[pheno]]
            if self.cache_tensors:
                pheno_data["input_tensor"] = pheno_data["input_tensor"][
                    self.samples[pheno]
                ]
            else:
                # TODO: Again do this in blocks of 10,000 samples
                # Create a temporary directory to store the zarr array
                tensor_path = (
                    Path(self.input_tensor_dir.name) / pheno / "input_tensor.zarr"
                )
                zarr.save_array(
                    tensor_path,
                    pheno_data["input_tensor_zarr"][:][self.samples[pheno]],
                    chunks=(self.chunksize, None, None, None),
                    compressor=Blosc(clevel=1),
                )
                pheno_data["input_tensor_zarr"] = zarr.open(tensor_path)

            logger.info(
                f"{pheno}: {self.samples[pheno].shape[0]} / "
                f"{n_samples_orig} samples kept"
            )

    def index_input_tensor_zarr(self, pheno: str, indices: np.ndarray):
        # IMPORTANT!!! Never call this function after self.subset_samples()

        x = self.data[pheno]["input_tensor_zarr"]
        first_idx = indices[0]
        last_idx = indices[-1]
        slice_ = slice(first_idx, last_idx + 1)
        arange = np.arange(first_idx, last_idx + 1)
        z = x[slice_]
        slice_indices = np.nonzero(np.isin(arange, indices))
        return z[slice_indices]


class MultiphenoBaggingData(pl.LightningDataModule):
    def __init__(
        self,
        data: Dict[str, Dict],
        train_proportion: float,
        sample_with_replacement: bool = True,
        min_variant_count: int = 1,
        upsampling_factor: int = 1,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = 0,
        pin_memory: bool = False,
        cache_tensors: bool = False,
        temp_dir: Optional[str] = None,
        chunksize: int = 1000,
    ):
        logger.info("Intializing datamodule")
        super().__init__()

        if upsampling_factor < 1:
            raise ValueError("upsampling_factor must be at least 1")

        self.data = data
        self.n_genes = {
            pheno: self.data[pheno]["genes"].shape[0] for pheno in self.data.keys()
        }

        # Get the number of annotations and covariates
        # This is the same for all phenotypes, so we can look at the tensors for any one of them
        any_pheno_data = next(iter(self.data.values()))
        self.n_annotations = any_pheno_data["input_tensor_zarr"].shape[2]
        self.n_covariates = any_pheno_data["covariates"].shape[1]

        self.max_n_variants = 0
        for _, pheno_data in self.data.items():
            n_samples = pheno_data["input_tensor_zarr"].shape[0]
            assert pheno_data["covariates"].shape[0] == n_samples
            assert pheno_data["y"].shape[0] == n_samples
            num_variants = pheno_data["input_tensor_zarr"].shape[-1]
            if self.max_n_variants < num_variants: self.max_n_variants = num_variants

            self.upsampling_factor = upsampling_factor
            if self.upsampling_factor > 1:
                logger.info(
                    f"Upsampling data with original sample number: {pheno_data['y'].shape[0]}"
                )
                upsampled_indices = self.upsample(pheno_data['y'])
                samples = np.append(np.arange(n_samples),upsampled_indices)
                n_samples += upsampled_indices.shape[0]
                logger.info(f"New sample number: {n_samples}")
            else:
                samples = np.arange(n_samples)

            # Sample self.n_samples * train_proportion samples with replacement
            # for training, use all remaining samples for validation
            if train_proportion == 1.0:
                self.train_samples = self.samples
                self.val_samples = self.samples
            else:
                n_train_samples = round(n_samples * train_proportion)
                rng = np.random.default_rng()
                train_samples = np.sort(
                    rng.choice(
                        samples, size=n_train_samples, replace=sample_with_replacement
                    )
                )
                pheno_data["samples"] = {
                    "train": train_samples,
                    "val": np.setdiff1d(samples, train_samples),
                }

        self.save_hyperparameters(
            "min_variant_count",
            "train_proportion",
            "batch_size",
            "num_workers",
            "pin_memory",
            "cache_tensors",
            "temp_dir",
            "chunksize",
        )

    def upsample(self,y_data) -> np.ndarray:
        unique_values = y_data.unique()
        if unique_values.size() != torch.Size([2]):
            raise ValueError(
                "Upsampling is only supported for binary y, "
                f"but y has unique values {unique_values}")

        class_indices = [(y_data == v).nonzero(as_tuple=True)[0] for v in unique_values]
        class_sizes = [idx.shape[0] for idx in class_indices]
        minority_class = 0 if class_sizes[0] < class_sizes[1] else 1
        minority_indices = class_indices[minority_class].detach().numpy()
        rng = np.random.default_rng(seed=42)
        upsampled_indices = rng.choice(
            minority_indices,
            size=(self.upsampling_factor - 1) * class_sizes[minority_class],
        )
        logger.info(f"Minority class: {unique_values[minority_class]}")
        logger.info(f"Minority class size: {class_sizes[minority_class]}")
        logger.info(f"Increasing minority class size by {upsampled_indices.shape[0]}")

        return upsampled_indices

    def train_dataloader(self):
        logger.info(
            "Instantiating training dataloader "
            f"with batch size {self.hparams.batch_size}"
        )

        dataset = MultiphenoDataset(
            self.data,
            self.hparams.min_variant_count,
            self.hparams.batch_size,
            split="train",
            cache_tensors=self.hparams.cache_tensors,
            temp_dir=self.hparams.temp_dir,
            chunksize=self.hparams.chunksize,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        logger.info("Instantiating validation dataloader "
                    f"with batch size {self.hparams.batch_size}")
        dataset = MultiphenoDataset(
            self.data,
            self.hparams.min_variant_count,
            self.hparams.batch_size,
            split="val",
            cache_tensors=self.hparams.cache_tensors,
            temp_dir=self.hparams.temp_dir,
            chunksize=self.hparams.chunksize,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_bagging(
    config: Dict,
    data: Dict[str, Dict],
    gene_count: Dict[str, Dict],
    log_dir: str,
    checkpoint_file: Optional[str] = None,
    trial: Optional[optuna.trial.Trial] = None,
    trial_id: Optional[int] = None,
    debug: bool = False,
) -> Optional[float]:
    if trial is not None:
        if trial_id is not None:
            trial.set_user_attr("user_id", trial_id)

        config["model"]["config"] = suggest_hparams(config["model"]["config"], trial)
        logger.info("Model hyperparameters this trial:")
        pprint(config["model"]["config"])
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        config_out = Path(log_dir) / "config.yaml"
        logger.info(f"Writing config to {config_out}")
        with open(config_out, "w") as f:
            yaml.dump(config, f)

    n_bags = config["training"]["n_bags"] if not debug else 3
    train_proportion = config["training"].get("train_proportion", None)
    logger.info(f"Training {n_bags} bagged models")

    if hasattr(config['model'], "seed"):
        set_random_seed(config['model']["seed"])
        logger.info(f'Set seed from config file')

    lasso = config["model"].get("run_lasso", False)

    def run_training(k):
        logger.info(f"  Starting training for bag {k}")

        this_data = copy.deepcopy(data)
        for _, pheno_data in this_data.items():
            if pheno_data["training_genes"] is not None:
                pheno_data["genes"] = pheno_data["training_genes"][f"bag_{k}"]
                logger.info(
                    f'Using {len(pheno_data["genes"])} training genes '
                    f'(out of {pheno_data["input_tensor_zarr"].shape[1]} total) at indices:'
                )
                print(" ".join(map(str, pheno_data["genes"])))

        dm_kwargs = {
            k: v
            for k, v in config["training"].items()
            if k
            in (
                "min_variant_count",
                "upsampling_factor",
                "sample_with_replacement",
                "cache_tensors",
                "temp_dir",
                "chunksize",
            )
        }
        dm = MultiphenoBaggingData(
            this_data,
            train_proportion,
            **dm_kwargs,
            **config["training"]["dataloader_config"],
        )

        model_class = getattr(deeprvat_models, config["model"]["type"])
        model = model_class(
            config=config["model"]["config"],
            n_annotations=dm.n_annotations,
            n_covariates=dm.n_covariates,
            n_genes=dm.n_genes,
            gene_count=gene_count,
            max_n_variants=dm.max_n_variants,
            phenotypes=list(data.keys()),
            **config["model"].get("kwargs", {}),
        )

        tb_log_dir = f"{log_dir}/bag_{k}"
        logger.info(f"    Writing TensorBoard logs to {tb_log_dir}")
        tb_logger = TensorBoardLogger(log_dir, name=f"bag_{k}")

        objective = "val_" + config["model"]["config"]["metrics_val"]["objective"]
        checkpoint_callback = ModelCheckpoint(monitor=objective)
        callbacks = [checkpoint_callback]
        if "early_stopping" in config:
            callbacks.append(
                EarlyStopping(monitor=objective, **config["early_stopping"])
            )

        if debug:
            config["pl_trainer"]["min_epochs"] = 10
            config["pl_trainer"]["max_epochs"] = 20

        trainer = pl.Trainer(
            logger=tb_logger, callbacks=callbacks, **config.get("pl_trainer", {})
        )

        while True:
            try:
                trainer.fit(model, dm)
            except RuntimeError as e:
                logging.error(f"Caught RuntimeError: {e}")
                if str(e).find("CUDA out of memory") != -1:
                    if dm.hparams.batch_size > 4:
                        logging.error(
                            f"Retrying training with half the original batch size"
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                        dm.hparams.batch_size = dm.hparams.batch_size // 2
                    else:
                        logging.error("Batch size is already <= 4, giving up")
                        raise RuntimeError("Could not find small enough batch size")
                else:
                    logging.error(f"Caught unknown error: {e}")
                    raise e
            else:
                break

        logger.info(
            "Training finished, max memory used: "
            f"{torch.cuda.max_memory_allocated(0)}"
        )

        trial.set_user_attr(
            f"bag_{k}_checkpoint_path", checkpoint_callback.best_model_path
        )
        checkpoint_paths.append(checkpoint_callback.best_model_path)

        if checkpoint_file is not None:
            logger.info(
                f"Symlinking {checkpoint_callback.best_model_path}"
                f" to {checkpoint_file}"
            )
            Path(checkpoint_file).symlink_to(
                Path(checkpoint_callback.best_model_path).resolve()
            )

        results.append(model.best_objective)
        logger.info(f" Result this bag: {model.best_objective}")

        del dm
        gc.collect()
        torch.cuda.empty_cache()

    def run_lassotraining(k,current_lambda):
        nonlocal counter
        nonlocal is_dense
        logger.info(f"  Starting lasso training for lambda {current_lambda}")
        logger.info(f"  Starting lasso training for bag {k}")

        this_data = copy.deepcopy(data)
        for _, pheno_data in this_data.items():
            if pheno_data["training_genes"] is not None:
                pheno_data["genes"] = pheno_data["training_genes"][f"bag_{k}"]
                logger.info(
                    f'Using {len(pheno_data["genes"])} training genes '
                    f'(out of {pheno_data["input_tensor_zarr"].shape[1]} total) at indices:'
                )
                print(" ".join(map(str, pheno_data["genes"])))

        dm_kwargs = {
            k: v
            for k, v in config["training"].items()
            if k
            in (
                "min_variant_count",
                "upsampling_factor",
                "sample_with_replacement",
                "cache_tensors",
            )
        }
        dm = MultiphenoBaggingData(
            this_data,
            train_proportion,
            **dm_kwargs,
            **config["training"]["dataloader_config"],
        )

        model_class = getattr(deeprvat_models, config["model"]["type"])

        if counter == 0:
            model = model_class(
                config=config["model"]["config"],
                n_annotations=dm.n_annotations,
                n_covariates=dm.n_covariates,
                n_genes=dm.n_genes,
                gene_count=gene_count,
                lambda_=current_lambda, #config['model']["config"]['lambda'],
                gamma=0.0,
                gamma_skip=0.0,
                M=10.0,
                max_n_variants=dm.max_n_variants,
                phenotypes=list(data.keys()),
                **config["model"].get("kwargs", {}),
            )
        else:
            logger.info(f"Using previous checkpoint {checkpoint_paths[counter-1]} state to resume training with new Lambda {current_lambda}")
            checkpoint = checkpoint_paths[counter-1]
            model = model_class.load_from_checkpoint(
                checkpoint,
                config=config["model"]["config"],
                lambda_= current_lambda,
            )

        #####
        #mock up lambda_schedule prediction from model attribute
        # print('STARTING LAMBDA INIT FINDER NOW!!!!!!!!')
        # est_lambda = model.lambda_start(M=10.0)
        # start_l = (est_lambda 
        #             / config["model"]["config"]["optimizer"]["config"]["lr"] 
        #             / 10 # divide by 10 for initial training
        # )
        # print(f"estimated lambda: {est_lambda}")
        # print(f"Estimated Start Lambda : {start_l}")
        # import pdb; pdb.set_trace()
        ########

        tb_log_dir = f"{log_dir}/bag_{k}/lambda_{current_lambda}"
        logger.info(f"    Writing TensorBoard logs to {tb_log_dir}")
        tb_logger = TensorBoardLogger(log_dir, name=f"bag_{k}/lambda_{current_lambda}")

        objective = "val_" + config["model"]["config"]["metrics_val"]["objective"]
        checkpoint_callback = ModelCheckpoint(monitor=objective,
                                                save_on_train_epoch_end=True) 
        callbacks = [checkpoint_callback]
        if "early_stopping" in config:
            callbacks.append(
                EarlyStopping(monitor=objective, **config["early_stopping"])
            )

        if debug:
            config["pl_trainer"]["min_epochs"] = 10
            config["pl_trainer"]["max_epochs"] = 20


        trainer = pl.Trainer(
            logger=tb_logger, callbacks=callbacks, **config.get("pl_trainer", {})
        )

        while True:
            try:
                trainer.fit(model, dm)
                counter += 1
            except RuntimeError as e:
                logging.error(f"Caught RuntimeError: {e}")
                if str(e).find("CUDA out of memory") != -1:
                    if dm.hparams.batch_size > 4:
                        logging.error(
                            f"Retrying training with half the original batch size"
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                        dm.hparams.batch_size = dm.hparams.batch_size // 2
                    else:
                        logging.error("Batch size is already <= 4, giving up")
                        raise RuntimeError("Could not find small enough batch size")
                else:
                    logging.error(f"Caught unknown error: {e}")
                    raise e
            else:
                break

        logger.info(
            "Training finished, max memory used: "
            f"{torch.cuda.max_memory_allocated(0)}"
        )

        trial.set_user_attr(
            f"bag_{k}_checkpoint_path", checkpoint_callback.best_model_path
        )
        checkpoint_paths.append(checkpoint_callback.best_model_path)

        if checkpoint_file is not None:
            logger.info(
                f"Symlinking {checkpoint_callback.best_model_path}"
                f" to {checkpoint_file}"
            )
            Path(checkpoint_file).symlink_to(
                Path(checkpoint_callback.best_model_path).resolve()
            )

        results.append(model.best_objective)
        logger.info(f" Result this bag: {model.best_objective}")

        current_features = model.selected_count()

        if is_dense and current_features < dm.n_annotations: 
            is_dense = False
            if current_lambda / lambda_start < 2:
                assert (
                    f"lambda_start={lambda_start:.3f} "
                    "selected lambda might be too large.\n"
                    f"Features start to disappear at {current_lambda=:.3f}."
                )
        print(f"Lambda = {current_lambda:.2e}, "
            f"selected {current_features} features ")
        
        del dm
        gc.collect()
        torch.cuda.empty_cache()
        
        return current_features

    def build_lambda_scheduler(lambda_init, lambda_seq=None, path_multiplier=1.2, lambda_max=float("inf")):
        #Build Lambda Sequence 
        if lambda_seq is not None:
            lambda_seq = lambda_seq
        else:
            def _lambda_seq(start):
                while start <= lambda_max:
                    yield start
                    start *= path_multiplier

            if lambda_init == "auto":
                raise NotImplementedError("auto lambda initialization not yet implemented")
                logger.info("   Building an auto-generated Lambda sequence.")
                #TODO add M parameter to config and specify in DeepSetLasso init
                lambda_start_ = (
                    deeprvat_models.DeepSetLasso.lambda_start(M=10) #self.hparams['M'])
                    / config["model"]["config"]["optimizer"]["config"]["lr"]
                    / 10 # divide by 10 for initial training
                )
                lambda_seq = _lambda_seq(lambda_start_)
            else:
                lambda_seq = _lambda_seq(lambda_init)
        
        # extract first value of lambda_seq
        lambda_seq = iter(lambda_seq)
        lambda_start = next(lambda_seq)
        return lambda_start, lambda_seq

    results = []
    checkpoint_paths = []

    if lasso:
        lambda_init = config["model"].get("lambda_init", 1.64)
        lambda_start, lambda_seq = build_lambda_scheduler(lambda_init)

        is_dense = True
        counter = 0
        for current_lambda in itertools.chain([lambda_start], lambda_seq):
            for k in range(n_bags):
                current_features = run_lassotraining(k,current_lambda)

            if current_features == 0:
                print('LASSO completed. Reached Sparsity; 0 selected features.')
                break
    else:
        for k in range(n_bags):
            run_training(k)

    # Mark checkpoints with worst results to be dropped
    drop_n_bags = config["training"].get("drop_n_bags", None) if not debug else 1
    if drop_n_bags is not None:
        if config["model"]["config"]["metrics_train"].get("objective_mode", "max") == "max":
            min_result = sorted(results)[drop_n_bags]
            drop_bags = [(r < min_result) for r in results]
        else:
            max_result = sorted(results, reverse=True)[drop_n_bags]
            drop_bags = [(r > max_result) for r in results]

        results = np.array([r for r, d in zip(results, drop_bags) if not d])
        for drop, ckpt in zip(drop_bags, checkpoint_paths):
            if drop:
                Path(ckpt + ".dropped").touch()

    final_result = np.mean(results)
    n_bags_used = n_bags - drop_n_bags if drop_n_bags is not None else n_bags
    logger.info(
        f"Results (top {n_bags_used} bags): "
        f"{final_result} (mean) {np.std(results)} (std)"
    )
    return final_result

@cli.command()
@click.option("--debug", is_flag=True)
@click.option("--training-gene-file", type=click.Path(exists=True))
@click.option("--n-trials", type=int, default=1)
@click.option("--trial-id", type=int)
@click.option("--sample-file", type=click.Path(exists=True))
@click.option(
    "--phenotype",
    multiple=True,
    type=(
        str,
        click.Path(exists=True),
        click.Path(exists=True),
        click.Path(exists=True),
        click.Path(exists=True)
    ) # phenotype_name, input_tensor_file, covariates_file, y_files, meta_data_file
)
@click.argument("config-file", type=click.Path(exists=True))
@click.argument("log-dir", type=click.Path())
@click.argument("hpopt-file", type=click.Path())
def train(
    debug: bool,
    training_gene_file: Optional[str],
    n_trials: int,
    trial_id: Optional[int],
    sample_file: Optional[str],
    phenotype: Tuple[Tuple[str, str, str, str, str]],
    config_file: str,
    log_dir: str,
    hpopt_file: str,
):
    if len(phenotype) == 0:
        raise ValueError("At least one --phenotype option must be specified")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    if debug:
        config["pl_trainer"].pop("gpus", None)
        config["pl_trainer"].pop("precision", None)

    logger.info(f"Running training using config:\n{pformat(config)}")

    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory}")

    logger.info("Loading input data")
    if sample_file is not None:
        logger.info(f"Using training samples from {sample_file}")
        with open(sample_file, "rb") as f:
            samples = pickle.load(f)["training_samples"]
        if debug:
            samples = [s for s in samples if s < 1000]
    else:
        samples = slice(None)

    data = dict()
    for pheno, input_tensor_file, covariates_file, y_file, meta_data_file in phenotype:
        data[pheno] = dict()
        data[pheno]["input_tensor_zarr"] = zarr.open(
            input_tensor_file, mode="r"
        )  # TODO: subset here?
        data[pheno]["covariates"] = torch.tensor(
            zarr.open(covariates_file, mode="r")[:]
        )[
            samples
        ]  # TODO: or maybe shouldn't subset here?
        data[pheno]["y"] = torch.tensor(zarr.open(y_file, mode="r")[:])[
            samples
        ]  # TODO: or maybe shouldn't subset here?
        data[pheno]["gene_id"] = torch.tensor(
            zarr.open(meta_data_file, mode='r')[:]
        )[
            samples
        ]

        if training_gene_file is not None:
            with open(training_gene_file, "rb") as f:
                training_genes = pickle.load(f)
            if isinstance(training_genes, list):
                # In this case, training genes are the same for all bags,
                # so we set training_genes to None and they'll be ignored
                training_genes = None
            elif not isinstance(training_genes, dict):
                ValueError(
                    f"{training_gene_file} contains invalid training " "gene data"
                )
        else:
            training_genes = {
                f"bag_{k}": np.arange(data[pheno]["input_tensor_zarr"].shape[1])
                for k in range(config["training"]["n_bags"])
            }

        data[pheno]["training_genes"] = training_genes

    genes = []
    for pheno in data:
        for key in data[pheno]["gene_id"]:
            if key not in genes:
                genes.append(key.item())
    gene_count = len(genes)

    gene2id = dict(zip(genes, range(gene_count)))
    for pheno in data:
        tmp = []
        for key in data[pheno]["gene_id"]:
            tmp.append(gene2id[key.item()])
        data[pheno]["gene_id"] = tmp
    gene_count = len(genes)

    hparam_optim = config.get("hyperparameter_optimization", None)
    if hparam_optim is None:
        run_bagging(config, data, gene_count, log_dir, debug=debug)
    else:
        pruner_config = config["hyperparameter_optimization"].get("pruning", None)
        if pruner_config is not None:
            pruner: optuna.pruners.BasePruner = getattr(
                optuna.pruners, pruner_config["type"]
            )(**pruner_config["config"])
        else:
            pruner = optuna.pruners.NopPruner()

        objective_direction = config["hyperparameter_optimization"].get(
            "direction", "maximize"
        )

        sampler_config = config["hyperparameter_optimization"].get("sampler", None)
        if sampler_config is not None:
            sampler: optuna.samplers._base.BaseSampler = getattr(
                optuna.samplers, sampler_config["type"]
            )(**sampler_config["config"])
        else:
            sampler = None

        study = optuna.create_study(
            study_name=Path(hpopt_file).stem,
            direction=objective_direction,
            sampler=sampler,
            pruner=pruner,
            storage=f"sqlite:///{hpopt_file}",
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: run_bagging(
                config,
                data,
                gene_count,
                log_dir,
                trial=trial,
                trial_id=trial_id,
                debug=debug
            ),
            n_trials=n_trials,
            timeout=hparam_optim.get("timeout", None),
        )

        logger.info(f"Number of finished trials: {len(study.trials)}")

        trial = study.best_trial
        logger.info(f'Best trial: {trial.user_attrs["user_id"]}')
        logger.info(
            f'  Mean {config["model"]["config"]["metrics_val"]["objective"]}: '
            f"{trial.value}"
        )
        logger.info(f"  Params:\n{pformat(trial.params)}")


@cli.command()
@click.option("--debug", is_flag=True)
@click.argument("log-dir", type=click.Path())
@click.argument("checkpoint-dir", type=click.Path())
@click.argument("hpopt-db", type=click.Path())
@click.argument("config-file-out", type=click.Path())
def best_training_run(
    debug: bool, log_dir: str, checkpoint_dir: str, hpopt_db: str, config_file_out: str
):
    study = optuna.load_study(
        study_name=Path(hpopt_db).stem, storage=f"sqlite:///{hpopt_db}"
    )

    trials = study.trials_dataframe().query('state == "COMPLETE"')
    best_trial = trials.sort_values("value", ascending=False).iloc[0]
    if best_trial['value'] == float("inf"):
        best_trial = trials.sort_values("value", ascending=False).iloc[1]

    best_trial_id = best_trial["user_attrs_user_id"]

    logger.info(f"Best trial:\n{best_trial}")

    with open(Path(log_dir) / f"trial{best_trial_id}/config.yaml") as f:
        config = yaml.safe_load(f)

    with open(config_file_out, "w") as f:
        yaml.dump({"model": config["model"]}, f)

    n_bags = config["training"]["n_bags"] if not debug else 3
    for k in range(n_bags):
        link_path = Path(checkpoint_dir) / f"bag_{k}.ckpt"
        checkpoint = Path(best_trial[f"user_attrs_bag_{k}_checkpoint_path"])
        link_path.symlink_to(checkpoint.resolve(strict=True))

        # Keep track of models marked to be dropped
        checkpoint_dropped = Path(str(checkpoint) + ".dropped")
        if checkpoint_dropped.is_file():
            dropped_link_path = Path(checkpoint_dir) / f"bag_{k}.ckpt.dropped"
            dropped_link_path.touch()


if __name__ == "__main__":
    cli()

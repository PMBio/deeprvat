import copy
import gc
import itertools
import logging
import sys
import pickle
import shutil
from pathlib import Path
from pprint import pformat, pprint
from typing import Dict, Optional, Tuple

import torch.nn.functional as F
import numpy as np
import click
import math
import random
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
import zarr
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


def make_dataset_(
    config: Dict,
    debug: bool = False,
    training_dataset_file: str = None,
    pickle_only: bool = False,
):
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

        variant_file = config["training_data"].get(
            "variant_file",
            f'{config["training_data"]["gt_file"][:-3]}_variants.parquet',
        )
        ds = DenseGTDataset(
            gt_file=config["training_data"]["gt_file"],
            variant_file=variant_file,
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
            for r in tqdm(rare_batches, file=sys.stdout)
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
        normalization: str = "",
        balance_dataset: bool = False,
        single_pheno: bool = False):
        super().__init__()
        
        self.data = data
        self.normalization = normalization
        self.balance_dataset = balance_dataset
        self.single_pheno = single_pheno
        self.split = split
        self.batch_size = batch_size
        self.phenotypes = self.data.keys()
        self.min_variant_count = min_variant_count  * np.ones(len(self.phenotypes))
        self.gene_count = int(max([max(data[pheno]["gene_id"]) for pheno, _ in self.data.items()]) + 1)

        logger.info(
            f"Initializing MultiphenoDataset with phenotypes:\n{pformat(list(self.phenotypes))}"
        )
        self.cache_tensors = cache_tensors

        for _, pheno_data in self.data.items():
            if pheno_data["y"].shape == (pheno_data["input_tensor_zarr"].shape[0], 1):
                pheno_data["y"] = pheno_data["y"].squeeze()
            elif pheno_data["y"].shape != (
                    pheno_data["input_tensor_zarr"].shape[0], ):
                raise NotImplementedError(
                    'Multi-phenotype training is only implemented via multiple y files'
                )

            if self.cache_tensors:
                pheno_data["input_tensor"] = pheno_data["input_tensor_zarr"][:]
                
                
        self.samples = {
            pheno: pheno_data["samples"][split]
            for pheno, pheno_data in self.data.items()}
        
        if self.split == "train" and self.balance_dataset: 
            self.min_variant_count = self.compute_min_variant_scores()
            self.balance_samples()
            
            self.running_phenotypes = False
            self.index_dict = dict(zip(list(self.phenotypes), np.zeros(len(list(self.phenotypes)))))
            self.data_dict = dict()
            for phenotype in self.phenotypes:
                pheno_samples = pd.DataFrame({"phenotype": itertools.chain(*[[pheno] * len(self.samples[pheno]) for pheno in [phenotype]])})
                pheno_samples = pheno_samples.astype({"phenotype": pd.api.types.CategoricalDtype()})
                pheno_samples = pheno_samples.sample(n=len(self.samples[phenotype]))  # shuffle
                pheno_samples["index"] = pheno_samples.groupby("phenotype").cumcount()
                self.data_dict[phenotype] = pheno_samples             
        else: 
            self.subset_samples(self.min_variant_count)
            

        if self.normalization == "standardization" or self.normalization == "both":
            if self.normalization == "both":
                self.norm_params = self.get_min_max_params()
            logger.info(f"normalizing annotations with {self.normalization}")
            self.standardization_params = self.get_standardization_params()
        elif self.normalization == "min-max" or self.normalization == "neg-min-max":
            logger.info(f"normalizing annotations with {self.normalization}")
            self.norm_params = self.get_min_max_params()

        if self.split == "val" or not self.balance_dataset: 
            self.total_samples = sum([s.shape[0] for s in self.samples.values()])
            self.sample_order = pd.DataFrame({
                "phenotype":
                itertools.chain(*[[pheno] * len(self.samples[pheno])
                                for pheno in self.phenotypes])
            })
            self.sample_order = self.sample_order.astype(
                {"phenotype": pd.api.types.CategoricalDtype()})
            self.sample_order = self.sample_order.sample(
                n=self.total_samples)  # shuffle
            self.sample_order["index"] = self.sample_order.groupby(
                "phenotype").cumcount()

    def compute_subset(self, min_variant, i, pheno, pheno_data, save=False):
        # First sum over annotations (dim 2) for each variant in each gene.
        # Then get the number of non-zero values across all variants in all
        # genes for each sample.
        input_tensor = pheno_data["input_tensor_zarr"].oindex[self.samples[pheno]]
        n_variants_per_sample = np.sum(
            np.sum(input_tensor, axis=2) != 0, axis=(1, 2)
        )
        n_variant_mask = n_variants_per_sample >= min_variant[i]

        nan_mask = ~pheno_data["y"][self.samples[pheno]].isnan()
        mask = n_variant_mask & nan_mask.numpy()
        if save:
            n_samples_orig = self.samples[pheno].shape[0]
            self.samples[pheno] = self.samples[pheno][mask]
            logger.info(f"{pheno}: {self.samples[pheno].shape[0]} / "
                        f"{n_samples_orig} samples kept")
        else:
            return self.samples[pheno][mask].shape[0]

    def subset_samples(self, min_variant):
        for i, (pheno, pheno_data) in enumerate(self.data.items()):
            self.compute_subset(min_variant, i, pheno, pheno_data, save=True)
    
    def invert_subset(self, pheno_samples, min_variant, last_min_variant):
        for i, (pheno, pheno_data) in enumerate(self.data.items()):
            if min_variant[i] != last_min_variant[i]:
                pheno_samples[i] = self.compute_subset(min_variant, i, pheno, pheno_data)
        return pheno_samples

    def compute_min_variant_scores(self, threshold = 0.5):
        logger.info(f"Compute min variant each phenotype") 
        min_variant = self.min_variant_count.copy()
        last_min_variant = np.zeros(len(self.phenotypes))
        pheno_samples = np.zeros(len(self.phenotypes))
        min_sample = False
        while True:
            check = 0
            pheno_samples = self.invert_subset(pheno_samples, min_variant, last_min_variant)
            if not min_sample: min_sample = np.min(pheno_samples)
            last_min_variant = min_variant.copy()
            for i, sample in enumerate(pheno_samples):
                if (sample - min_sample) > (1 + threshold) * min_sample: 
                    min_variant[i] += 1
                    check += 1
                elif sample < min_sample:
                    min_variant[i] -= 1
            if check == 0: 
                pheno_samples = self.subset_samples(min_variant)
                break
        return min_variant

    def balance_samples(self):
        low = min([self.samples[pheno].shape[0] for pheno, _ in self.data.items()])
        logger.info(f"Balanced all phenotype classes to have {low} samples")  
        for pheno, _ in self.data.items():
            self.samples[pheno] = self.samples[pheno][:low]
        

    def __len__(self):   
        if self.split == "train" and self.single_pheno: 
            return np.sum([math.ceil((len(self.samples[pheno])) / self.batch_size) for pheno in self.phenotypes])
        else: return math.ceil(len(self.sample_order) / self.batch_size)

    def __getitem__(self, index):
        if self.split == "train" and self.single_pheno: 
            if not self.running_phenotypes: 
                self.running_phenotypes = list(self.phenotypes)
                self.current_pheno = 0

            phenotype = self.running_phenotypes[self.current_pheno % len(self.running_phenotypes)]
            samples4pheno = len(self.samples[phenotype])
            start_idx = int(self.index_dict[phenotype])
            end_idx = int(min(samples4pheno, start_idx + self.batch_size))
            if end_idx == samples4pheno: 
                self.index_dict[phenotype] = 0
                self.data_dict[phenotype] = self.data_dict[phenotype].sample(frac=1).reset_index(drop=True)
                self.running_phenotypes.remove(phenotype)
            else: self.index_dict[phenotype] += self.batch_size
            samples_by_pheno = self.data_dict[phenotype][start_idx:end_idx]
        else:
            start_idx = index * self.batch_size
            end_idx = min(self.total_samples, start_idx + self.batch_size)
            samples_by_pheno = self.sample_order.iloc[start_idx:end_idx]
        samples_by_pheno = samples_by_pheno.groupby("phenotype")

        result = dict()
        for pheno, df in samples_by_pheno:
            idx = df["index"].to_numpy()
            annotations = (
                self.data[pheno]["input_tensor"][idx]
                if self.cache_tensors else
                self.data[pheno]["input_tensor_zarr"].oindex[idx, :, :, :])
            
            if self.normalization == "standardization":
                annotations = self.standardization_annotations(annotations)
            elif self.normalization == "min-max":
                annotations = self.min_max_annotations(annotations)
            elif self.normalization == "neg-min-max":
                annotations = self.neg_min_max_annotations(annotations)
            elif self.normalization == "both":
                annotations = self.min_max_annotations(annotations)
                annotations = self.standardization_annotations(annotations)

            result[pheno] = {
                "indices": self.samples[pheno][idx],
                "covariates": self.data[pheno]["covariates"][idx],
                "rare_variant_annotations": annotations,
                "y": self.data[pheno]["y"][idx],
                "gene_id": self.data[pheno]["gene_id"]
            }

            if self.split == "train" and self.single_pheno: 
                # +1 works if we are in a setting with balanced classes. 
                # +random is better, to not always process the "bigger" data classes at the end of training
                # self.current_pheno += 1 
                self.current_pheno += random.randrange(len(self.phenotypes))
        return result
    
    def get_standardization_params(self):
        tmp = False
        counter = 0
        for pheno in self.data:   
            annotations = self.data[pheno]["input_tensor" if self.cache_tensors else "input_tensor_zarr"]
            if not tmp: tmp = dict(zip(range(annotations.shape[-2]), [0 for _ in range(annotations.shape[-2])]))
            for i in range(annotations.shape[-2]):
                annotation = annotations[:, :, i, :]
                if self.normalization == "both": 
                    min = self.norm_params[i]["min"]
                    max = self.norm_params[i]["max"]
                    annotation = (annotation - min) / (max - min)     
                tmp[i] += annotation.sum() #sum across all elements
            counter += 1

        standarization_params = dict()
        for key in tmp:
            mean = tmp[key] / counter   
            std = np.sqrt( ((tmp[key] - mean)**2) / counter )
            standarization_params.update({key: {"mean": mean, "std": std}})
        del tmp
        return standarization_params
    
    def standardization_annotations(self, annotations):
        for key in self.standardization_params:
            mean = self.standardization_params[key]["mean"]
            std = self.standardization_params[key]["std"]
            annotations[:, :, key, :] = (annotations[:, :, key, :] - mean) / std   
        return annotations
    
    def get_min_max_params(self):
        min_max_params = False
        for pheno in self.data:   
            annotations = self.data[pheno]["input_tensor" if self.cache_tensors else "input_tensor_zarr"]
            if not min_max_params: min_max_params = dict(zip(range(annotations.shape[-2]), 
                                                             [{"min": np.inf, "max": -np.inf} for _ in range(annotations.shape[-2])]))
            for i in range(annotations.shape[-2]):
                i_max = annotations[:, :, i, :].max() 
                i_min = annotations[:, :, i, :].min() 
                if min_max_params[i]["min"] > i_min: min_max_params[i]["min"] = i_min
                if min_max_params[i]["max"] < i_max: min_max_params[i]["max"] = i_max
        return min_max_params  

    # def min_max_annotations(self, annotations):
    #     for key in self.norm_params:
    #         min = self.norm_params[key]["min"]
    #         max = self.norm_params[key]["max"]
    #         annotations[:, :, key, :] = (annotations[:, :, key, :] - min) / (max - min)    
    #     return annotations
    def min_max_annotations(self, annotations):
        for key in self.norm_params:
            min_val = self.norm_params[key]["min"]
            max_val = self.norm_params[key]["max"]
            if min_val == max_val:
                annotations[:, :, key, :] = max_val
            else: 
                annotations[:, :, key, :] = (annotations[:, :, key, :] - min_val) / (max_val - min_val) 
        return annotations
        
    # def neg_min_max_annotations(self, annotations):
    #     for key in self.norm_params:
    #         min = self.norm_params[key]["min"]
    #         max = self.norm_params[key]["max"]
    #         annotations[:, :, key, :] = (2 * ((annotations[:, :, key, :] - min) / (max - min)) ) - 1    
    #     return annotations
    def neg_min_max_annotations(self, annotations):
        for key in self.norm_params:
            min_val = self.norm_params[key]["min"]
            max_val = self.norm_params[key]["max"]
            if min_val == max_val:
                annotations[:, :, key, :] = max_val
            else:
                annotations[:, :, key, :] = (2 * ((annotations[:, :, key, :] - min_val) / (max_val - min_val)) ) - 1  
        return annotations


class MultiphenoBaggingData(pl.LightningDataModule):
    def __init__(
            self,
            # input_tensor: torch.Tensor,
            # covariates: torch.Tensor,
            # y: torch.Tensor,
            data: Dict[str, Dict],
            train_proportion: float,
            sample_with_replacement: bool = True,
            min_variant_count: int = 1,
            upsampling_factor: int = 1,  # NOTE: Changed
            batch_size: Optional[int] = None,
            num_workers: Optional[int] = 0,
            cache_tensors: bool = False,
            normalization: str = "",
            balance_dataset: bool = False,
            single_pheno: bool = False):
        logger.info("Intializing datamodule")
        super().__init__()

        if upsampling_factor < 1:
            raise ValueError("upsampling_factor must be at least 1")

        # self.input_tensor = input_tensor
        # self.covariates = covariates
        # self.y = y
        self.normalization = normalization
        self.data = data
        self.n_genes = {
            pheno: self.data[pheno]["genes"].shape[0]
            for pheno in self.data.keys()
        }

        # Get the number of annotations and covariates
        # This is the same for all phenotypes, so we can look at the tensors for any one of them
        any_pheno_data = next(iter(self.data.values()))
        self.n_annotations = any_pheno_data["input_tensor_zarr"].shape[2]
        self.n_covariates = any_pheno_data["covariates"].shape[1]

        self.max_n_variants = 0
        for pheno, pheno_data in self.data.items():
            n_samples = pheno_data["input_tensor_zarr"].shape[0]
            assert pheno_data["covariates"].shape[0] == n_samples
            assert pheno_data["y"].shape[0] == n_samples
            # self.n_genes = pheno_data["input_tensor_zarr"].shape[1]
            num_variants = pheno_data["input_tensor_zarr"].shape[-1]
            if self.max_n_variants < num_variants: self.max_n_variants = num_variants

            # NOTE: Do upsampling here
            # TODO: Rewrite this for multiphenotype data
            self.upsampling_factor = upsampling_factor
            if self.upsampling_factor > 1:
                logger.info(
                    f"Upsampling data with original sample number: {self.y.shape[0]}"
                )
                samples = self.upsample()
                n_samples = self.samples.shape[0]
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
                    rng.choice(samples,
                               size=n_train_samples,
                               replace=sample_with_replacement))
                pheno_data["samples"] = {
                    "train": train_samples,
                    "val": np.setdiff1d(samples, train_samples)
                }
                
        self.balance_dataset = balance_dataset
        self.single_pheno = single_pheno
        self.save_hyperparameters("min_variant_count", "train_proportion",
                                  "batch_size", "num_workers", "cache_tensors")

        # batch_size = self.hparams.batch_size
        # if batch_size is None:
        #     if torch.cuda.is_available():
        #         # n_samples = len(self.train_samples)
        #         # batch_size = min(suggest_batch_size(self.input_tensor.shape[1:]),
        #         #                  n_samples)
        #         batch_size = suggest_batch_size(self.train_tensor.shape[1:])
        #     else:
        #         batch_size = 50_000
        # logger.info(f"Using batch size: {batch_size}")
        # self.batch_size = batch_size

    def upsample(self) -> np.ndarray:
        unique_values = self.y.unique()
        if unique_values.size() != torch.Size([2]):
            raise ValueError("Upsampling is only supported for binary y, "
                             f"but y has unique values {unique_values}")

        class_indices = [(self.y == v).nonzero(as_tuple=True)[0]
                         for v in unique_values]
        class_sizes = [idx.shape[0] for idx in class_indices]
        minority_class = 0 if class_sizes[0] < class_sizes[1] else 1
        minority_indices = class_indices[minority_class].detach().numpy()
        rng = np.random.default_rng()
        upsampled_indices = rng.choice(minority_indices,
                                       size=(self.upsampling_factor - 1) *
                                       class_sizes[minority_class])
        logger.info(f"Minority class: {unique_values[minority_class]}")
        logger.info(f"Minority class size: {class_sizes[minority_class]}")
        logger.info(
            f"Increasing minority class size by {upsampled_indices.shape[0]}")

        self.samples = upsampled_indices

    def train_dataloader(self):
        logger.info("Instantiating training dataloader "
                    f"with batch size {self.hparams.batch_size}")
        dataset = MultiphenoDataset(
            # self.train_tensor,
            # self.covariates[self.train_samples, :],
            # self.y[self.train_samples, :],
            self.data,
            self.hparams.min_variant_count,
            self.hparams.batch_size,
            split="train",
            cache_tensors=self.hparams.cache_tensors,
            normalization=self.normalization,
            balance_dataset=self.balance_dataset,
            single_pheno=self.single_pheno)
        return DataLoader(dataset,
                          shuffle=True,
                          batch_size=None,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        logger.info("Instantiating validation dataloader "
                    f"with batch size {self.hparams.batch_size}")
        dataset = MultiphenoDataset(
            # self.val_tensor,
            # self.covariates[self.val_samples, :],
            # self.y[self.val_samples, :],
            self.data,
            self.hparams.min_variant_count,
            self.hparams.batch_size,
            split="val",
            cache_tensors=self.hparams.cache_tensors,
            normalization=self.normalization,
            balance_dataset=self.balance_dataset,
            single_pheno=self.single_pheno)
        return DataLoader(dataset,
                          batch_size=None,
                          num_workers=self.hparams.num_workers)

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

    results = []
    checkpoint_paths = []
    for k in range(n_bags):
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

        objective = "val_" + config["model"]["config"]["metrics"]["objective"]
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

    # Mark checkpoints with worst results to be dropped
    drop_n_bags = config["training"].get("drop_n_bags", None) if not debug else 1
    if drop_n_bags is not None:
        if config["model"]["config"]["metrics"].get("objective_mode", "max") == "max":
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
        data[pheno]["input_tensor_zarr"] = zarr.open(input_tensor_file, mode="r")
        data[pheno]["covariates"] = torch.tensor(
            zarr.open(covariates_file, mode="r")[:]
        )[samples]
        data[pheno]["y"] = torch.tensor(zarr.open(y_file, mode="r")[:])[samples]
        data[pheno]["gene_id"] = torch.tensor(zarr.open(meta_data_file, 
                                                        mode='r')[:])[samples]

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
            f'  Mean {config["model"]["config"]["metrics"]["objective"]}: '
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

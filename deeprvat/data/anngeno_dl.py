import logging
import sys
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Set, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from anngeno.anngeno import (
    AnnGeno,
)  # TODO: Fix anngeno __init__.py so we can "from anngeno import AnnGeno"
from torch.utils.data import DataLoader

PathLike = Union[str, Path]

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # TODO: Change to INFO (maybe add a verbose flag)


def standardize(x: torch.Tensor, dim: int) -> torch.Tensor:
    means = x.mean(dim=dim, keepdim=True)
    stds = x.std(dim=dim, keepdim=True)
    return (x - means) / stds


def quantile_transform(x, dim, seed=1) -> torch.Tensor:
    """
    Gaussian quantile transform for values in a torch.Tensor.

    :param x: Input tensor.
    :type x: torch.Tensor
    :param dim: Dimension over which to transform.
    :type dim: int
    :param seed: Random seed.
    :type seed: int
    :return: Transformed tensor.
    :rtype: torch.Tensor

    .. note::
        "nan" values are kept
    """
    np.random.seed(seed)
    x_transform = x.copy()
    if isinstance(x_transform, pd.Series):
        x_transform = x_transform.to_numpy()

    is_nan = np.isnan(x_transform)
    n_quantiles = np.sum(~is_nan)

    x_transform[~is_nan] = quantile_transform(
        x_transform[~is_nan].reshape([-1, 1]),
        n_quantiles=n_quantiles,
        subsample=n_quantiles,
        output_distribution="normal",
        copy=True,
    )[:, 0]

    return x_transform


# TODO:
# Option to shuffle samples (or at least batches of samples) during training
#
# Option to score only certain regions, even if not in training mode
#
##### Possible optimizations:
#
# Cache masks for all samples (as packed,possibly conpressed binary matrices?)
#
# Cache genotypes (as packed, in memory zarr array)
#
# If sample_set is given, automatically update variant_set to remove unobserved variants
# (requires reading much of genotype matrix, so maybe not worth it?)
class AnnGenoDataset:
    def __init__(
        self,
        filename: PathLike,
        sample_batch_size: Optional[int] = None,
        mask_type: Literal["max", "sum"] = "max",
        # gene_batch_size: int = 1,
        training_mode: bool = False,
        training_regions: Optional[Dict[int, np.ndarray]] = None,
        covariates: Optional[List[str]] = None,
        standardize_covariates: bool = True,
        # phenotypes: Optional[List[str]] = None,
        quantile_transform_phenotypes: bool = True,  # TODO: This is different from current default
        annotation_columns: Optional[List[str]] = None,
        variant_set: Optional[Set[int]] = None,
        sample_set: Optional[Set[str]] = None,
        dtype=torch.float32,
    ):
        self.training_mode = training_mode

        self.anngeno = AnnGeno(filename=filename, filemode="r")
        self.anngeno.compute_regions(
            annotation_columns=annotation_columns, variant_set=variant_set
        )

        self.cache_genotypes = False

        self.samples = self.anngeno.samples
        self.n_samples = len(self.samples)

        self.sample_batch_size = (
            min(sample_batch_size, self.n_samples)
            if sample_batch_size is not None
            else self.n_samples
        )
        # self.gene_batch_size = gene_batch_size

        self.dtype = dtype
        self.mask_type = mask_type
        self.standardize_covariates = standardize_covariates
        # TODO: Implement this
        self.quantile_transform_phenotypes = quantile_transform_phenotypes

        if self.training_mode:
            if training_regions is None or covariates is None:  # or phenotypes is None:
                raise ValueError(
                    "training_regions and covariates "
                    "must be provided if training_mode=True"
                )

            # Store regions
            self.training_regions = training_regions
            # TODO: What if there are overlapping genes?
            self.regions = np.concatenate(list(self.training_regions.values()))

            # region_boundaries = np.cumsum([r.shape[0] for r in self.regions])
            region_sizes = [self.anngeno.masked_region_sizes[k] for k in self.regions]
            region_boundaries = np.concatenate([[0], np.cumsum(region_sizes)])
            region_indices = zip(region_boundaries[:-1], region_boundaries[1:])
            self.gene_indices = dict(zip(self.training_regions.keys(), region_indices))

            n_variants = region_boundaries[-1]
            n_genes = self.regions.shape[0]
            self.variant_gene_mask = torch.zeros(
                (n_variants, n_genes), dtype=self.dtype
            )
            for i, (_, (start, stop)) in enumerate(self.gene_indices.items()):
                self.variant_gene_mask[start:stop, i] = 1

            self.covariate_cols = covariates
            self.phenotype_cols = list(self.training_regions.keys())

            # Build gene-to-phenotype mask for MaskedLinear layer
            n_phenos = len(self.training_regions)
            self.gene_phenotype_mask = torch.zeros(
                (n_phenos, n_genes), dtype=torch.bool
            )
            for i, (start, stop) in enumerate(self.gene_indices.values()):
                self.gene_phenotype_mask[i, start:stop] = True
            self.gene_covariatephenotype_mask = torch.cat(
                (
                    torch.ones(
                        (len(self.phenotype_cols), len(self.covariate_cols)),
                        dtype=torch.bool,
                    ),
                    self.gene_phenotype_mask,
                ),
                dim=1,
            )
        else:
            # Use all regions
            self.regions = self.anngeno.region_ids

        self.n_regions = self.regions.shape[0]
        self.set_samples(sample_set)

    def set_samples(self, sample_set: Optional[Set[str]]):
        self.anngeno.subset_samples(sample_set)
        self.samples = self.anngeno.samples
        self.n_samples = len(self.samples)
        self.sample_batch_size = min(self.sample_batch_size, self.n_samples)

        if self.training_mode:
            self.phenotype_df = self.anngeno.get_phenotypes(
                columns=self.covariate_cols + self.phenotype_cols
            )

            # TODO: Sanity check, can be removed or moved to test
            assert np.array_equal(self.phenotype_df["sample"].to_numpy(), self.samples)

            self.covariates = torch.tensor(
                self.phenotype_df[self.covariate_cols].to_numpy(), dtype=self.dtype
            )
            if self.standardize_covariates:
                self.covariates = standardize(self.covariates, dim=0)

            self.phenotypes = torch.tensor(
                self.phenotype_df[self.phenotype_cols].to_numpy(), dtype=self.dtype
            )

    def __len__(self):
        if self.training_mode:
            return math.ceil(self.n_samples / self.sample_batch_size)
        else:
            return math.ceil(self.n_regions) * math.ceil(  # / self.gene_batch_size
                self.n_samples / self.sample_batch_size
            )

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # start = time.process_time()

        result = {}

        if self.training_mode:
            regions = self.regions
            sample_idx = idx
        else:
            region_idx = idx % self.n_regions
            regions = [self.regions[region_idx]]
            sample_idx = idx // self.n_regions
            result["region_idx"] = region_idx

        sample_slice = slice(
            sample_idx * self.sample_batch_size,
            (sample_idx + 1) * self.sample_batch_size,
        )

        if self.cache_genotypes:
            slice_cache = self.anngeno.get_cached_regions(sample_slice=sample_slice)
            genotypes = torch.tensor(slice_cache["genotypes"], dtype=self.dtype)
            annotations = torch.tensor(
                slice_cache["annotations"], dtype=self.dtype
            )  # TODO: these actually only need to be fetched once
        else:
            by_gene = [
                self.anngeno.get_region(r, sample_slice=sample_slice) for r in regions
            ]
            genotypes = torch.concatenate(
                [torch.tensor(x["genotypes"], dtype=self.dtype) for x in by_gene],
                axis=1,
            )
            annotations = torch.concatenate(
                [
                    torch.tensor(x["annotations"].to_numpy(), dtype=self.dtype)
                    for x in by_gene
                ],
                axis=0,
            )
            del by_gene

        # # apply gene_mask to genotypes
        # # TODO: Do this in model (on GPU)
        # genotypes = torch.einsum(
        #     "ij,jk->ijk", genotypes, self.gene_mask
        # )  # samples x variants x genes

        result["genotypes"] = genotypes
        result["annotations"] = annotations

        # # TODO: Could also do this within model class
        # if self.mask_type == "max":
        #     # build mask for max aggregation
        #     max_mask = torch.where(genotypes, 0, float("-inf")).type(self.dtype)
        #     result["max_mask"] = max_mask

        result["sample_slice"] = sample_slice
        result["samples"] = self.samples[sample_slice]
        result["variant_gene_mask"] = None

        if self.training_mode:
            result["covariates"] = self.covariates[sample_slice]
            result["phenotypes"] = self.phenotypes[sample_slice]
            result["variant_gene_mask"] = self.variant_gene_mask

        # print(f"getitem total: {time.process_time() - start}")

        return result

    def cache_regions(self, compress: bool = False):
        self.anngeno.cache_regions(self.regions, compress=compress)
        self.cache_genotypes = True


class AnnGenoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        anngeno_filename: PathLike,
        num_workers: int = 0,
        batch_size: Optional[int] = None,
        training_regions: Optional[Dict[int, np.ndarray]] = None,
        variant_set: Optional[Set[int]] = None,
        mask_type: Literal["max", "sum"] = "max",
        covariates: Optional[List[str]] = None,
        standardize_covariates: bool = True,
        phenotypes: Optional[List[str]] = None,
        quantile_transform_phenotypes: bool = True,  # TODO: This is different from current default
        annotation_columns: Optional[List[str]] = None,
        sample_set: Optional[Set[str]] = None,
        train_proportion: Optional[float] = None,
        shuffle: bool = True,
        cache_genotypes: bool = True,  # TODO: Change to False?
        compress_cache: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.setup_done = dict()

    def setup(self, stage: Literal["fit", "associate", "score"]):
        """
        :param stage: "trainval" sets up LightningModule train_dataloader and val_dataloader. "associate" sets up test_dataloader. "score" sets up predict_dataloader.
        :type stage: Literal["trainval", "associate", "score"]
        """
        if self.setup_done.get(stage, False):
            return

        if stage == "fit":
            dataset_args = dict(
                filename=self.hparams.anngeno_filename,
                sample_batch_size=self.hparams.batch_size,
                variant_set=self.hparams.variant_set,
                mask_type=self.hparams.mask_type,
                covariates=self.hparams.covariates,
                standardize_covariates=self.hparams.standardize_covariates,
                phenotypes=self.hparams.phenotypes,
                quantile_transform_phenotypes=self.hparams.quantile_transform_phenotypes,
                annotation_columns=self.hparams.annotation_columns,
                training_mode=True,
                training_regions=self.hparams.training_regions,
            )

            logger.info("Instantiating training dataset")
            self.train_dataset = AnnGenoDataset(
                **dataset_args,
                sample_set=self.hparams.sample_set,
            )

            # Choose training samples at random
            all_samples = self.train_dataset.samples
            n_train_samples = round(
                self.hparams.train_proportion * self.train_dataset.n_samples
            )
            rng = np.random.default_rng()
            train_samples = set(
                rng.choice(
                    all_samples, size=n_train_samples, replace=False, shuffle=False
                )
            )
            self.train_dataset.set_samples(train_samples)

            self.gene_covariatephenotype_mask = (
                self.train_dataset.gene_covariatephenotype_mask
            )

            # Pass sample_set's and options along to AnnGenoDataset
            logger.info("Instantiating validation dataset")
            self.val_dataset = AnnGenoDataset(
                **dataset_args, sample_set=set(all_samples) - train_samples
            )

            if self.hparams.cache_genotypes:
                self.train_dataset.cache_regions(compress=self.hparams.compress_cache)
                self.val_dataset.cache_regions(compress=self.hparams.compress_cache)

            self.setup_done["fit"] = True
        else:
            raise NotImplementedError("Coming soon...")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
            batch_size=None,  # No automatic batching
            batch_sampler=None,  # No automatic batching
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
            batch_size=None,  # No automatic batching
            batch_sampler=None,  # No automatic batching
        )

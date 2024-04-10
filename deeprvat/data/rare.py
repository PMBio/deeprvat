import itertools
import logging
import random
import sys
from pathlib import Path
from pprint import pformat
from scipy.sparse import coo_matrix, vstack
from typing import Dict, List, Optional, Union, Set
import numpy as np
import pandas as pd
import copy
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset

from deeprvat.utils import calculate_mean_std, standardize_series_with_params

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# this class is used DeepRVAT
class PaddedAnnotations:
    def __init__(
        self,
        base_dataset,
        annotations: List[str],
        thresholds: Dict[str, str] = None,
        gene_file: Optional[str] = None,
        genes_to_keep: Optional[Set[str]] = None,
        pad_value: Union[float, int, str] = 0.0,
        verbose: bool = False,
        low_memory: bool = False,
        skip_embedding: bool = False,
    ):
        if verbose:
            logger.setLevel(logging.DEBUG)

        self.base_dataset = base_dataset
        self.annotations = annotations
        self.grouping_column = base_dataset.grouping_column
        self.pad_value = float(pad_value)
        self.low_memory = low_memory
        self.stand_params = None
        self.skip_embedding = skip_embedding

        if self.base_dataset.train_dataset is not None:
            logger.debug("Setting up based on training dataset")
            train_embedding = self.base_dataset.train_dataset.rare_embedding
            self.annotation_df = train_embedding.annotation_df
            self.exploded_annotations = train_embedding.exploded_annotations
            self.exploded_annotations_np = self.exploded_annotations_np
            self.gene_map = train_embedding.gene_map
            self.genes = train_embedding.genes
            self.genes_np = train_embedding.genes_np
        else:
            logger.debug("Setting up annotations")
            assert base_dataset.variants.index.name == "id"
            if "id" in base_dataset.variants.columns:
                assert (
                    base_dataset.variants.index == base_dataset.variants["id"]
                ).all()
            rare_variant_ids = base_dataset.variants.index[
                base_dataset.variants["rare_variant_mask"]
            ]
            self.setup_annotations(
                rare_variant_ids, thresholds, gene_file, genes_to_keep
            )

            logger.debug(f"Applying thresholds:\n{pformat(thresholds)}")
            self.apply_thresholds(thresholds)

            logger.debug("Remapping group IDs")
            self.remap_group_ids()

            logger.debug("Setting up metadata")
            self.setup_metadata()

        if self.low_memory:
            logger.info(f"  Cleaning up to save memory")
            self.annotation_df = None
            self.exploded_annotations = None
            self.base_dataset.annotation_df = None

    def embed(
        self, idx: int, variant_ids: np.ndarray, genotype: np.ndarray
    ) -> List[List[torch.Tensor]]:
        """Returns: List[List[torch.Tensor]]

        One outer list element for each gene; inner list elements are annotations
        for variants, one element for each variant in a gene for this sample
        """
        if self.skip_embedding:
            return torch.tensor([])

        variants_mapped = self.variant_map[variant_ids]
        mask = variants_mapped >= 0
        variant_ids = variant_ids[mask]
        genotype = genotype[mask]
        rows = []
        for v, g in zip(variant_ids, genotype):
            ids = self.exp_anno_id_indices[v]  # np.ndarray
            # homozygous variants are considered twice
            rows += [ids] * g  # List[np.ndarray]

        result = [[] for _ in range(len(self.genes))]
        if len(rows) > 0:
            rows = np.concatenate(rows)
            # logger.info(f"rows {rows}")
            for i in rows:
                gene = self.gene_map[self.genes_np[i]]  # NOTE: Changed
                result[gene].append(self.exploded_annotations_np[i, :])

        return result

    def collate_fn(
        self,
        batch: List[List[List[np.ndarray]]],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Returns: torch.Tensor

        Dimensions of tensor: samples x genes x annotations x variants. Last
        dimension is padded to fit all variants.
        """
        if self.skip_embedding:
            return torch.tensor([])

        n_samples = len(batch)
        max_n_variants = max(len(gene) for sample in batch for gene in sample)
        n_annotations = len(self.annotations)
        result = np.zeros(
            (n_samples, self.n_genes, n_annotations, max_n_variants), dtype=np.float32
        )
        for i, sample in enumerate(batch):
            for j, gene in enumerate(sample):
                for k, variant in enumerate(gene):
                    result[i, j, :, k] = variant

        return torch.tensor(result, dtype=torch.float, device=device)

    def setup_annotations(
        self,
        rare_variant_ids: pd.Series,
        thresholds: Optional[Dict[str, str]],
        gene_file: Optional[str],
        genes_to_keep: Optional[Set[str]] = None,
    ):
        self.variant_map = -(2**24) * np.ones(
            rare_variant_ids.max() + 1, dtype=np.int32
        )

        logger.debug("  Filtering by rare variant IDs and by gene")
        annotation_df = self.base_dataset.annotation_df

        threshold_cols = list(
            set()
            if thresholds is None
            else set(thresholds.keys()) & set(annotation_df.columns)
        )
        mask = annotation_df.index.isin(rare_variant_ids)
        mask &= annotation_df[self.grouping_column].apply(lambda x: len(x) > 0)
        annotation_df = annotation_df.loc[
            mask, set(self.annotations + [self.grouping_column] + threshold_cols)
        ].copy()
        # standardize here
        if (
            self.base_dataset.standardize_rare_anno
            or self.base_dataset.standardize_rare_anno_columns
        ):
            logger.debug("  Standardizing annotations")

            if self.base_dataset.standardize_rare_anno_columns:
                cols = self.base_dataset.standardize_rare_anno_columns
            else:
                # all columns will be standardized
                cols = self.annotations

            self.stand_params = {}
            for col in cols:
                if self.base_dataset.standardize_rare_anno_params:
                    logger.info("Using pre-defined std and mean for standardization")
                    std, mean = self.base_dataset.standardize_rare_anno_params[col]

                else:
                    logger.info(
                        f"Calculating mean and standard deviation for col {col}"
                    )
                    std, mean = calculate_mean_std(annotation_df[col])

                logger.info(
                    f"Standardising annotation {col} with mean {mean} and std {std}"
                )
                annotation_df[col] = standardize_series_with_params(
                    annotation_df[col], std, mean
                )
                self.stand_params[col] = (std, mean)

                # return standardization params

        logger.debug("  Exploding annotations by groups")
        annotation_df[self.grouping_column] = annotation_df[self.grouping_column].apply(
            lambda x: list(set(list(x)))
        )

        exploded_annotations = annotation_df[
            set([self.grouping_column] + self.annotations + threshold_cols)
        ].explode(self.grouping_column)

        if gene_file is not None:
            logger.debug("  Filtering by provided genes")
            genes_df = pd.read_parquet(gene_file, columns=["id", "gene"])
            genes = set(genes_df["id"])
            mask = exploded_annotations["gene_ids"].isin(genes)
            if genes_to_keep is not None:
                genes_to_keep_ids = set(
                    genes_df[genes_df["gene"].isin(genes_to_keep)]["id"]
                )
                mask &= exploded_annotations["gene_ids"].isin(genes_to_keep_ids)
            exploded_annotations = exploded_annotations[mask]
            annotation_df = annotation_df[
                annotation_df.index.isin(exploded_annotations.index)
            ]

        self.annotation_df = annotation_df[set(self.annotations + threshold_cols)]
        self.exploded_annotations = exploded_annotations[
            set([self.grouping_column] + self.annotations + threshold_cols)
        ].astype({self.grouping_column: np.int32})

        if len(self.exploded_annotations) == 0:
            raise RuntimeError(f"No rare variants found in provided genes")

    def apply_thresholds(self, thresholds: Optional[Dict[str, str]]):
        if thresholds is not None:
            self.annotation_df["mask"] = True
            self.exploded_annotations["mask"] = True
            for op in thresholds.values():
                self.annotation_df["mask"] &= self.annotation_df.eval(op)
                self.exploded_annotations["mask"] &= self.exploded_annotations.eval(op)
            self.annotation_df = self.annotation_df[self.annotation_df["mask"]]
            self.exploded_annotations = self.exploded_annotations[
                self.exploded_annotations["mask"]
            ]

        self.annotation_df = self.annotation_df[self.annotations]
        self.exploded_annotations = self.exploded_annotations[
            [self.grouping_column] + self.annotations
        ]
        self.kept_variants = np.sort(self.annotation_df.index.to_numpy())
        assert np.all(self.kept_variants == np.unique(self.kept_variants))
        self.variant_map[self.kept_variants] = np.arange(len(self.annotation_df))

        if len(self.annotation_df) == 0:
            raise RuntimeError(f"  No variants passed thresholding")

        logger.info(f" {len(self.annotation_df)} variants passed thresholding")

        self.exploded_annotations_np = self.exploded_annotations[
            self.annotations
        ].to_numpy()
        self.genes_np = copy.deepcopy(
            self.exploded_annotations[self.grouping_column].to_numpy()
        )

    def remap_group_ids(self):
        self.gene_map = -(2**24) * np.ones(
            self.exploded_annotations[self.grouping_column].max() + 1, dtype=np.int32
        )
        self.genes = np.sort(self.exploded_annotations[self.grouping_column].unique())
        self.n_genes = len(self.genes)
        logger.info(
            f"Found {self.n_genes} genes with rare variants " "that pass thresholds"
        )

        self.gene_map[self.genes] = np.arange(self.genes.shape[0])
        self.exploded_annotations[self.grouping_column] = self.gene_map[
            self.exploded_annotations[self.grouping_column].to_numpy()
        ]

    def setup_metadata(self):
        logger.debug("  Precomputing integer indices for exploded dataframe")
        self.exp_anno_id_indices = [
            np.array([], dtype=np.int32)
            for _ in range(self.annotation_df.index.max() + 1)
        ]
        for i in range(len(self.exploded_annotations)):
            j = self.exploded_annotations.index[i]
            self.exp_anno_id_indices[j] = np.append(self.exp_anno_id_indices[j], i)

    def get_metadata(self) -> Dict[str, np.ndarray]:
        return {
            "genes": self.genes,  # Gene IDs corresponding to dim 1 as returned from collate_fn
            "gene_map": self.gene_map,
            "variants": self.kept_variants,
            "variant_map": self.variant_map,
        }


# #this class is used for the seed gene discovery
class SparseGenotype:
    def __init__(
        self,
        base_dataset,
        annotations: List[str],
        thresholds: Dict[str, str] = None,
        gene_file: Optional[str] = None,
        genes_to_keep: Optional[Set[str]] = None,
        verbose: bool = False,
        low_memory: bool = False,
    ):
        if verbose:
            logger.setLevel(logging.DEBUG)

        self.base_dataset = base_dataset
        self.annotations = annotations
        self.grouping_column = base_dataset.grouping_column
        self.stand_params = None
        self.low_memory = low_memory

        self.max_variant_id = base_dataset.variants.index[
            base_dataset.variants["rare_variant_mask"]
        ].max()

        if self.base_dataset.train_dataset is not None:
            logger.debug("Setting up based on training dataset")
            train_embedding = self.base_dataset.train_dataset.rare_embedding
            self.annotation_df = train_embedding.annotation_df
            self.exploded_annotations = train_embedding.exploded_annotations
            self.exploded_annotations_np = self.exploded_annotations_np
            self.gene_map = train_embedding.gene_map
            self.genes = train_embedding.genes
        else:
            logger.debug("Setting up annotations")
            rare_variant_ids = base_dataset.variants.index[
                base_dataset.variants["rare_variant_mask"]
            ]
            self.setup_annotations(
                rare_variant_ids, thresholds, gene_file, genes_to_keep
            )

            logger.debug(f"Applying thresholds:\n{pformat(thresholds)}")
            self.apply_thresholds(thresholds)

            logger.debug("Remapping group IDs")
            self.remap_group_ids()

            logger.debug("Setting up metadata")
            self.setup_metadata()

        if self.low_memory:
            logger.info(f"  Cleaning up to save memory")
            self.annotation_df = None
            self.exploded_annotations = None
            self.base_dataset.annotation_df = None

    def embed(
        self, idx: int, variant_ids: np.ndarray, genotype: np.ndarray
    ) -> coo_matrix:
        """Returns: List[List[torch.Tensor]]

        One outer list element for each gene; inner list elements are annotations
        for variants, one element for each variant in a gene for this sample
        """
        variants_mapped = self.variant_map[variant_ids]
        mask = variants_mapped >= 0
        variant_ids = variant_ids[mask]
        genotype = genotype[mask]

        result = coo_matrix(
            (genotype, (np.zeros(len(variant_ids)), variant_ids)),
            shape=(1, self.max_variant_id + 1),
        )

        return result

    def collate_fn(self, batch: List[coo_matrix]) -> coo_matrix:
        return vstack(batch)

    def setup_annotations(
        self,
        rare_variant_ids: pd.Series,
        thresholds: Optional[Dict[str, str]],
        gene_file: Optional[str],
        genes_to_keep: Optional[Set[str]] = None,
    ):
        self.variant_map = -(2**24) * np.ones(self.max_variant_id + 1, dtype=np.int32)

        logger.debug("  Filtering by rare variant IDs and by gene")
        annotation_df = self.base_dataset.annotation_df
        threshold_cols = list(
            set()
            if thresholds is None
            else set(thresholds.keys()) & set(annotation_df.columns)
        )
        mask = annotation_df.index.isin(rare_variant_ids)
        mask &= annotation_df[self.grouping_column].apply(lambda x: len(x) > 0)
        annotation_df = annotation_df.loc[
            mask, set(self.annotations + [self.grouping_column] + threshold_cols)
        ].copy()
        # standardize here
        if (
            self.base_dataset.standardize_rare_anno
            or self.base_dataset.standardize_rare_anno_columns
        ):
            logger.debug("  Standardizing annotations")

            if self.base_dataset.standardize_rare_anno_columns:
                cols = self.base_dataset.standardize_rare_anno_columns
            else:
                # all columns will be standardized
                cols = self.annotations

            self.stand_params = {}
            for col in cols:
                if self.base_dataset.standardize_rare_anno_params:
                    logger.info("Using pre-defined std and mean for standardization")
                    std, mean = self.base_dataset.standardize_rare_anno_params[col]

                else:
                    logger.info(
                        f"Calculating mean and standard deviation for col {col}"
                    )
                    std, mean = calculate_mean_std(annotation_df[col])

                logger.info(
                    f"Standardising annotation {col} with mean {mean} and std {std}"
                )
                annotation_df[col] = standardize_series_with_params(
                    annotation_df[col], std, mean
                )
                self.stand_params[col] = (std, mean)

                # return standardization params

        logger.debug("  Exploding annotations by groups")
        annotation_df[self.grouping_column] = annotation_df[self.grouping_column].apply(
            lambda x: list(set(list(x)))
        )
        exploded_annotations = annotation_df[
            set([self.grouping_column] + self.annotations + threshold_cols)
        ].explode(self.grouping_column)
        if gene_file is not None:
            logger.debug("  Filtering by provided genes")
            genes_df = pd.read_parquet(gene_file, columns=["id", "gene"])
            genes = set(genes_df["id"])
            mask = exploded_annotations["gene_ids"].isin(genes)

            if genes_to_keep is not None:
                genes_to_keep_ids = set(
                    genes_df[genes_df["gene"].isin(genes_to_keep)]["id"]
                )
                mask &= exploded_annotations["gene_ids"].isin(genes_to_keep_ids)

            exploded_annotations = exploded_annotations[mask]
            annotation_df = annotation_df[
                annotation_df.index.isin(exploded_annotations.index)
            ]

        self.annotation_df = annotation_df[set(self.annotations + threshold_cols)]
        self.exploded_annotations = exploded_annotations[
            set([self.grouping_column] + self.annotations + threshold_cols)
        ].astype({self.grouping_column: np.int32})

    def apply_thresholds(self, thresholds: Optional[Dict[str, str]]):
        if thresholds is not None:
            self.annotation_df["mask"] = True
            self.exploded_annotations["mask"] = True
            for op in thresholds.values():
                self.annotation_df["mask"] &= self.annotation_df.eval(op)
                self.exploded_annotations["mask"] &= self.exploded_annotations.eval(op)
            self.annotation_df = self.annotation_df[self.annotation_df["mask"]]
            self.exploded_annotations = self.exploded_annotations[
                self.exploded_annotations["mask"]
            ]

        self.annotation_df = self.annotation_df[self.annotations]
        self.exploded_annotations = self.exploded_annotations[
            [self.grouping_column] + self.annotations
        ]
        self.exploded_annotations_np = self.exploded_annotations[
            self.annotations
        ].to_numpy()

        self.kept_variants = np.sort(self.annotation_df.index.to_numpy())
        assert np.all(self.kept_variants == np.unique(self.kept_variants))
        self.variant_map[self.kept_variants] = np.arange(len(self.annotation_df))

    def remap_group_ids(self):
        self.gene_map = -(2**24) * np.ones(
            self.exploded_annotations[self.grouping_column].max() + 1, dtype=np.int32
        )
        self.genes = np.sort(self.exploded_annotations[self.grouping_column].unique())
        self.n_genes = len(self.genes)
        logger.info(
            f"Found {self.n_genes} genes with rare variants " "that pass thresholds"
        )

        self.gene_map[self.genes] = np.arange(self.genes.shape[0])
        self.exploded_annotations[self.grouping_column] = self.gene_map[
            self.exploded_annotations[self.grouping_column].to_numpy()
        ]

    def setup_metadata(self):
        logger.debug("  Precomputing integer indices for exploded dataframe")
        self.exp_anno_id_indices = [
            np.array([], dtype=np.int32)
            for _ in range(self.annotation_df.index.max() + 1)
        ]
        for i in range(len(self.exploded_annotations)):
            j = self.exploded_annotations.index[i]
            self.exp_anno_id_indices[j] = np.append(self.exp_anno_id_indices[j], i)

    def get_metadata(self) -> Dict[str, np.ndarray]:
        return {
            "genes": self.genes,
            "gene_map": self.gene_map,
            "variants": self.kept_variants,
            "variant_map": self.variant_map,
        }

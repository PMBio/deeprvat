import copy
import logging
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import dask.dataframe as dd
import h5py
import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

import deeprvat.data.rare as rare_embedders
from deeprvat.utils import (
    safe_merge,
    standardize_series,
    my_quantile_transform,
)

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

DEFAULT_CHROMOSOMES = [f"chr{x}" for x in range(1, 23)]

AGGREGATIONS = {"max": np.max, "sum": np.sum}


class DenseGTDataset(Dataset):
    def __init__(
        self,
        gt_file: str = None,
        variant_file: str = None,
        split: str = "",
        train_dataset: Optional[Dataset] = None,
        chromosomes: List[str] = None,
        phenotype_file: str = None,
        standardize_xpheno: bool = True,
        standardize_anno: bool = False,
        standardize_rare_anno: bool = False,
        standardize_rare_anno_columns: Optional[List] = None,
        standardize_rare_anno_params: Optional[Dict] = None,
        permute_y: bool = False,
        y_transformation: Optional[str] = None,
        x_phenotypes: List[str] = [],
        grouping_level: Optional[str] = "gene",
        group_common: bool = False,
        return_sparse: bool = False,
        annotations: List[str] = [],
        annotation_file: Optional[str] = None,
        precomputed_annotations: Optional[
            Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]
        ] = None,
        annotation_aggregation: Union[str, dict] = "max",
        y_phenotypes: List[str] = [],
        skip_y_na: bool = True,
        skip_x_na: bool = False,
        sim_phenotype_file: Optional[str] = None,
        min_common_variant_count: Optional[int] = None,
        min_common_af: Optional[Dict[str, float]] = None,
        max_rare_af: Optional[Dict[str, float]] = None,
        use_common_variants: bool = True,
        use_rare_variants: bool = False,
        rare_embedding: Optional[Dict] = None,
        rare_ignore_unknown_gene: bool = True,
        exons_to_keep: Optional[Set[int]] = None,
        genes_to_keep: Optional[Set[str]] = None,
        gene_file: Optional[str] = None,
        gene_types_to_keep: Optional[List[str]] = None,
        ignore_by_annotation: Optional[List[Tuple[str, Any]]] = None,
        max_pval: Optional[Dict[str, float]] = None,
        variants: Optional[pd.DataFrame] = None,
        variants_to_keep: Optional[Union[List[str], str]] = None,
        zarr_dir: Optional[str] = None,
        cache_matrices: bool = False,
        verbose: bool = False,
    ):
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        self.split = split
        self.train_dataset = train_dataset
        self.chromosomes = (
            chromosomes if chromosomes is not None else DEFAULT_CHROMOSOMES
        )
        self.standardize_xpheno = standardize_xpheno
        self.standardize_anno = standardize_anno
        self.permute_y = permute_y
        self.y_transformation = y_transformation

        self.standardize_rare_anno = standardize_rare_anno
        self.standardize_rare_anno_columns = standardize_rare_anno_columns
        self.standardize_rare_anno = standardize_rare_anno_params
        self.skip_y_na = skip_y_na

        self.x_phenotypes = x_phenotypes
        self.y_phenotypes = y_phenotypes
        logger.debug(
            f"Using phenotypes: x: {self.x_phenotypes}, " f"y: {self.y_phenotypes}"
        )

        if gt_file is None:
            raise ValueError("gt_file must be specified")
        self.gt_filename = gt_file
        if variant_file is None:
            raise ValueError("variant_file must be specified")
        if phenotype_file is None:
            raise ValueError("phenotype_file must be specified")
        self.variant_filename = variant_file
        self.variant_matrix = None
        self.genotype_matrix = None
        self.cache_matrices = cache_matrices
        if zarr_dir is not None:
            self.setup_zarr(zarr_dir)
        else:
            if self.cache_matrices:
                logger.debug("Caching variant and genotype matrices")
                with h5py.File(self.gt_filename, "r") as f:
                    self.variant_matrix = f["variant_matrix"][:]
                    self.genotype_matrix = f["genotype_matrix"][:]

        logger.info(
            f"Using phenotype file {phenotype_file} and genotype file {self.gt_filename}"
        )
        self.setup_phenotypes(
            phenotype_file,
            sim_phenotype_file,
            skip_y_na,
            skip_x_na,
        )

        self.max_rare_af = max_rare_af
        self.use_common_variants = use_common_variants
        self.use_rare_variants = use_rare_variants
        self.rare_ignore_unknown_gene = rare_ignore_unknown_gene
        self.exons_to_keep = exons_to_keep
        self.genes_to_keep = genes_to_keep
        self.gene_types_to_keep = (
            set(gene_types_to_keep)
            if gene_types_to_keep is not None
            else gene_types_to_keep
        )
        self.gene_file = gene_file

        if grouping_level is not None:
            if grouping_level == "gene":
                self.grouping_column = "gene_ids"
            elif grouping_level == "exon":
                self.grouping_column = "exon_ids"
            else:
                raise ValueError(f"Unknown aggregation level {grouping_level}")
        else:
            if group_common:
                raise ValueError(
                    "grouping_level must be specified "
                    "if grouping/aggregation of common variants enabled"
                )

        self.group_common = group_common
        self.return_sparse = return_sparse

        self.annotations = annotations
        self.ignore_by_annotation = ignore_by_annotation
        self.max_pval = max_pval

        if isinstance(variants_to_keep, str):
            with open(variants_to_keep, "rb") as f:
                self.variants_to_keep = pickle.load(f)
                self.variants_to_keep = [
                    int(var_id) for var_id in self.variants_to_keep
                ]
        else:
            self.variants_to_keep = variants_to_keep

        self.setup_annotations(
            annotation_file, annotation_aggregation, precomputed_annotations
        )

        self.transform_data()
        self.setup_variants(min_common_variant_count, min_common_af, variants)

        self.get_variant_metadata(grouping_level)

        if rare_embedding is not None:
            self.rare_embedding = getattr(rare_embedders, rare_embedding["type"])(
                self, **rare_embedding["config"]
            )

        else:
            self.rare_embedding = None

    def __getitem__(self, idx: int) -> torch.tensor:
        if self.variant_matrix is None:
            gt_file = h5py.File(self.gt_filename, "r")
            self.variant_matrix = gt_file["variant_matrix"]
            self.genotype_matrix = gt_file["genotype_matrix"]

            if self.cache_matrices:
                self.variant_matrix = self.variant_matrix[:]
                self.genotype_matrix = self.genotype_matrix[:]

        idx = self.index_map[idx]

        sparse_variants = self.variant_matrix[idx, :]
        sparse_genotype = self.genotype_matrix[idx, :]
        (
            common_variants,
            all_sparse_variants,
            sparse_genotype,
        ) = self.get_common_variants(sparse_variants, sparse_genotype)

        rare_variant_annotations = self.get_rare_variants(
            idx, all_sparse_variants, sparse_genotype
        )

        phenotypes = self.phenotype_df.iloc[idx, :]

        x_phenotype_tensor = torch.tensor(
            phenotypes[self.x_phenotypes].to_numpy(dtype=np.float32), dtype=torch.float
        )

        y = torch.tensor(
            phenotypes[self.y_phenotypes].to_numpy(dtype=np.float32), dtype=torch.float
        )

        return {
            "sample": self.samples[idx],
            "x_phenotypes": x_phenotype_tensor,
            "common_variants": common_variants,
            "rare_variant_annotations": rare_variant_annotations,
            "y": y,
        }

    def __len__(self) -> int:
        return self.n_samples

    def get_stand_params(self):
        if hasattr(self.rare_embedding, "stand_params"):
            stand_params = self.rare_embedding.stand_params
        else:
            stand_params = None

        return stand_params

    def setup_phenotypes(
        self,
        phenotype_file: str,
        sim_phenotype_file: Optional[str],
        skip_y_na: bool,
        skip_x_na: bool,
    ):
        logger.debug("Reading phenotype dataframe")
        self.phenotype_df = pd.read_parquet(phenotype_file, engine="pyarrow")
        if sim_phenotype_file is not None:
            logger.info(
                f"Using phenotypes and covariates from simulated phenotype file {sim_phenotype_file}"
            )
            sim_phenotype = pd.read_parquet(sim_phenotype_file, engine="pyarrow")
            self.phenotype_df = self.phenotype_df.join(
                sim_phenotype
            )  # TODO on = , validate = "1:1"

        binary_cols = [
            c for c in self.y_phenotypes if self.phenotype_df[c].dtype == bool
        ]

        mask_cols = copy.deepcopy(self.x_phenotypes)
        if skip_y_na:
            mask_cols += self.y_phenotypes
        if skip_x_na:
            mask_cols += self.x_phenotypes
        mask = (self.phenotype_df[mask_cols].notna()).all(axis=1)
        self.n_samples = mask.sum()
        logger.info(
            f"Number of samples with phenotype and covariates: {self.n_samples}"
        )
        self.samples = self.phenotype_df.index.to_numpy()

        self.index_map = np.arange(len(self.phenotype_df))[mask]

    def get_variant_ids(self, matrix_indices: np.ndarray) -> np.ndarray:
        return self.variant_id_map.loc[matrix_indices, "id"].to_numpy()

    def dense_to_sparse(
        self,
        dense_genotype: Union[torch.Tensor, Dict[str, torch.Tensor]],
        keep_groups: bool = False,
    ) -> pd.DataFrame:
        if type(dense_genotype) == torch.Tensor:
            assert torch.all(dense_genotype >= 0)
            dense_genotype = dense_genotype.detach().numpy()
            ids = self.variant_id_map.loc[dense_genotype > 0, "id"]
            gt = dense_genotype[dense_genotype > 0]
            sparse_genotype = pd.DataFrame({"id": ids, "gt": gt}).astype(
                {"gt": np.int8}
            )
        else:
            common_variant_df = self.common_variant_groups.explode(self.grouping_column)
            ids_by_group = {
                name: group["id"]
                for name, group in common_variant_df.groupby(self.grouping_column)
            }
            sparse_genotype = {}
            for name, gt in dense_genotype.items():
                assert torch.all(gt >= 0)
                mask = gt.detach().numpy() > 0
                sparse_genotype[name] = pd.DataFrame(
                    {"id": ids_by_group[name][mask], "gt": gt[mask].detach().numpy()}
                ).astype({"gt": np.int8})

            if not keep_groups:
                sparse_genotype = pd.concat(sparse_genotype.values()).drop_duplicates()

        return sparse_genotype

    def get_annotations(
        self,
        variant_ids: Union[np.ndarray, pd.Series],
        group: bool = False,
        aggregate_groups: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        columns = copy.copy(self.annotations)
        if group or aggregate_groups:
            columns.append(self.grouping_column)
            row_indices = np.concatenate(
                [self.exp_anno_id_indices[v] for v in variant_ids]
            )
            column_indices = [
                i
                for i, c in enumerate(self.exploded_annotations.columns)
                if c in columns
            ]
            annotations = self.exploded_annotations.iloc[row_indices, column_indices]
        else:
            annotations = self.annotation_df.loc[variant_ids, columns]

        if group:
            annotations = {
                name: group.drop(columns=self.grouping_column)
                for name, group in annotations.groupby(self.grouping_column)
            }
            if aggregate_groups:
                annotations = {
                    name: group.agg(self.annotation_aggregation)
                    for name, group in annotations.items()
                }
            annotations = {
                name: torch.tensor(group.to_numpy(), dtype=torch.float)
                for name, group in annotations.items()
            }
        else:
            if aggregate_groups:
                annotations_df = annotations.groupby(self.grouping_column)
                annotations_df = annotations_df.agg(self.annotation_aggregation)
                annotations = np.zeros((self.n_groups, len(self.annotations)))
                annotations[annotations_df.index.to_numpy()] = annotations_df[
                    self.annotations
                ].to_numpy()
                annotations = torch.tensor(annotations, dtype=torch.float).view(-1)
            else:
                annotations = torch.tensor(annotations.to_numpy(), dtype=torch.float)

        return annotations

    def setup_zarr(self, zarr_dir: str):
        logger.debug("Setting up Zarr arrays")
        zarr_varmat = os.path.join(zarr_dir, "variant_matrix.zarr")
        zarr_gtmat = os.path.join(zarr_dir, "genotype_matrix.zarr")
        if os.path.getmtime(zarr_varmat) < os.path.getmtime(
            self.gt_filename
        ) or os.path.getmtime(zarr_gtmat) < os.path.getmtime(self.gt_filename):
            logger.warning(
                "GT file is newer than Zarr arrays -  "
                "Perhaps Zarr arrays should be exported again?"
            )
        self.variant_matrix = zarr.open(zarr_varmat)
        self.genotype_matrix = zarr.open(zarr_gtmat)

        if self.cache_matrices:
            self.variant_matrix = self.variant_matrix[:]
            self.genotype_matrix = self.genotype_matrix[:]

    def transform_data(self):
        logger.debug("Standardizing phenotypes and annotations")
        if self.standardize_xpheno:
            logger.debug("  Standardizing input phenotypes")
            for col in self.x_phenotypes:
                self.phenotype_df[col] = standardize_series(self.phenotype_df[col])

        if self.standardize_anno:
            logger.debug("  Standardizing annotations")
            for col in self.annotations:
                self.annotation_df[col] = standardize_series(self.annotation_df[col])

        # standardization of annotations for the rare embedding is done by rare.py

        if self.permute_y:
            logger.info("  Permuting target phenotypes")
            for col in self.y_phenotypes:
                rng = np.random.default_rng()
                self.phenotype_df[col] = rng.permutation(
                    self.phenotype_df[col].to_numpy()
                )
        if len(self.y_phenotypes) > 0:
            unique_y_val = self.phenotype_df[self.y_phenotypes[0]].unique()
            n_unique_y_val = np.count_nonzero(~np.isnan(unique_y_val))
            logger.info(f"unique y values {unique_y_val}")
            logger.info(n_unique_y_val)
        else:
            n_unique_y_val = 0
        if n_unique_y_val == 2:
            logger.warning(
                "Not applying y transformation because y only has two values and seems to be binary"
            )
            self.y_transformation = None
        if self.y_transformation is not None:
            if self.y_transformation == "standardize":
                logger.debug("  Standardizing target phenotype")
                for col in self.y_phenotypes:
                    self.phenotype_df[col] = standardize_series(self.phenotype_df[col])
            elif self.y_transformation == "quantile_transform":
                logger.debug(
                    f"  Quantile transforming target phenotype {self.y_phenotypes}"
                )
                for col in self.y_phenotypes:
                    self.phenotype_df[col] = my_quantile_transform(
                        self.phenotype_df[col]
                    )
            else:
                raise ValueError(f"Unknown y_transformation: {self.y_transformation}")
        else:
            logger.warning("Not transforming phenotype")

    def setup_annotations(
        self,
        annotation_file: Optional[str],
        annotation_aggregation: Union[str, dict],
        precomputed_annotations: Optional[
            Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]
        ] = None,
    ):
        logger.debug("Setting up annotations")
        if precomputed_annotations is not None:
            (
                self.annotation_df,
                self.exploded_annotations,
                self.exp_anno_id_indices,
            ) = precomputed_annotations
            return

        self.annotation_df = None
        if (
            len(self.annotations) > 0
            or self.ignore_by_annotation is not None
            or self.use_rare_variants
        ):
            columns = ["id"] + self.annotations
            if self.ignore_by_annotation is not None:
                columns += [col for col, _ in self.ignore_by_annotation]
            if self.max_rare_af is not None:
                columns.append(list(self.max_rare_af.keys())[0])
            if self.use_rare_variants:
                columns.append(self.grouping_column)

            logger.debug("    Reading annotation dataframe")
            self.annotation_df = dd.read_parquet(
                annotation_file, columns=list(set(columns)), engine="pyarrow"
            ).compute()
            self.annotation_df = self.annotation_df.set_index("id")

            if type(annotation_aggregation) == str:
                self.annotation_aggregation = AGGREGATIONS.get(
                    annotation_aggregation, annotation_aggregation
                )
            else:
                self.annotation_aggregation = {
                    k: AGGREGATIONS.get(v, v) for k, v in annotation_aggregation
                }
        else:
            logger.debug("    No annotations requested")

    def setup_variants(
        self,
        min_common_variant_count: Optional[int],
        min_common_af: Optional[Dict[str, float]],
        train_variants: Optional[pd.DataFrame],
    ):
        logger.debug("Setting up variants")
        if min_common_variant_count is None and min_common_af is None:
            raise ValueError(
                "At least one of min_common_variant_count"
                " or min_common_af must be specified"
            )

        logger.debug("    Reading variant dataframe")
        variants = dd.read_parquet(self.variant_filename, engine="pyarrow").compute()
        variants = variants.set_index("id", drop=False)
        variants = variants.drop(columns="matrix_index", errors="ignore")

        if self.variants_to_keep is not None:
            logger.info("Selecting subset of variants as defined by variants_to_keep")
            variants = variants.loc[self.variants_to_keep]
        logger.debug("    Filtering variants")
        if min_common_variant_count is not None:
            mask = (variants["count"] >= min_common_variant_count) & (
                variants["count"] <= self.n_samples - min_common_variant_count
            )
            mask = mask.to_numpy()
            logger.debug(f'    {mask.sum()} variants "common" by count filter')
        elif min_common_af is not None:
            af_col, af_threshold = list(min_common_af.items())[0]
            variants_with_af = safe_merge(
                variants[["id"]].reset_index(drop=True),
                self.annotation_df[[af_col]].reset_index(),
            )
            assert np.all(
                variants_with_af["id"].to_numpy() == variants["id"].to_numpy()
            )
            mask = (variants_with_af[af_col] >= af_threshold) & (
                variants_with_af[af_col] <= 1 - af_threshold
            )
            mask = mask.to_numpy()
            del variants_with_af
            logger.debug(f'    {mask.sum()} variants "common" by AF filter')
        else:
            raise ValueError(
                "Either min_common_variant_count or " "min_common_af must be specified"
            )

        if train_variants is not None:
            # Sanity check
            matrix_ids = train_variants.loc[
                train_variants["matrix_index"] >= 0, "matrix_index"
            ]
            assert np.all(matrix_ids == np.sort(matrix_ids))

        rare_variant_mask = ~mask
        chromosome_mask = variants["chrom"].isin(self.chromosomes).to_numpy()
        additional_mask = chromosome_mask
        if self.exons_to_keep is not None:
            raise NotImplementedError("The variant dataframes have outdated exon_ids")
            additional_mask &= (
                variants["exon_ids"]
                .apply(lambda x: len(set(x) & self.exons_to_keep) != 0)
                .to_numpy()
            )
        if self.genes_to_keep is not None:
            raise NotImplementedError("The variant dataframes have outdated gene_ids")
            additional_mask &= (
                variants["gene_ids"]
                .apply(lambda x: len(set(x) & set(self.genes_to_keep)) != 0)
                .to_numpy()
            )
        if self.gene_file is not None:
            genes = set(pd.read_parquet(self.gene_file, columns=["id"])["id"])
            logger.debug(f"    Retaining {len(genes)} genes from {self.gene_file}")
            variants_with_gene_ids = safe_merge(
                variants[["id"]].reset_index(drop=True),
                self.annotation_df[["gene_ids"]].reset_index(),
            )
            assert np.all(
                variants_with_gene_ids["id"].to_numpy() == variants["id"].to_numpy()
            )
            additional_mask &= (
                variants_with_gene_ids["gene_ids"]
                .apply(lambda x: len(set(x) & genes) != 0)
                .to_numpy()
            )
            del variants_with_gene_ids
        if self.gene_types_to_keep is not None:
            additional_mask &= (
                variants["gene_types"]
                .apply(lambda x: len(set(x) & self.gene_types_to_keep) != 0)
                .to_numpy()
            )
        if self.ignore_by_annotation is not None:
            for col, val in self.ignore_by_annotation:
                if self.annotation_df[col].dtype == np.dtype("object"):
                    additional_mask &= (
                        self.annotation_df.loc[variants["id"], col].apply(set)
                        == set(val).to_numpy()
                    )
                else:
                    additional_mask &= (
                        self.annotation_df.loc[variants["id"], col] == val
                    ).to_numpy()
        if self.max_pval is not None:
            col, bound = list(self.max_pval.items())[0]
            additional_mask &= (variants[col] < bound).to_numpy()
        rare_variant_mask &= additional_mask
        if self.rare_ignore_unknown_gene and (
            self.genes_to_keep is None
            and self.gene_file is None
            and self.gene_types_to_keep is None
        ):
            rare_variant_mask &= (
                variants["gene_ids"].apply(lambda x: len(x) > 0).to_numpy()
            )

        variants["rare_variant_mask"] = rare_variant_mask

        if train_variants is None:
            common_variant_mask = mask
            if self.max_rare_af is not None:
                common_variant_mask &= ~af_mask
            common_variant_mask &= additional_mask
            if self.group_common:
                common_variant_mask &= (
                    variants["gene_ids"].apply(lambda x: len(x) > 0).to_numpy()
                )

            variants["matrix_index"] = -1
            matrix_index_mask = common_variant_mask
            variants.loc[matrix_index_mask, "matrix_index"] = np.arange(
                sum(matrix_index_mask)
            )
            variants["common_variant_mask"] = common_variant_mask
        else:
            variants = pd.concat(
                [
                    variants,
                    train_variants.loc[
                        variants["id"], ["common_variant_mask", "matrix_index"]
                    ],
                ],
                axis=1,
            )

        n_common_variants = variants["common_variant_mask"].sum()
        n_rare_variants = variants["rare_variant_mask"].sum()
        logger.debug(f"    Retained {n_common_variants} common variants")
        logger.debug(f"    Retained {n_rare_variants} rare variants")

        variants = variants[
            ["id", "matrix_index", "common_variant_mask", "rare_variant_mask"]
        ]
        self.variants = variants

    def get_variant_metadata(self, grouping_level: Optional[str]):
        logger.debug("Computing metadata for variants")
        self.n_variants = self.variants["common_variant_mask"].sum()

        logger.debug("  Computing variant id to matrix index map")
        self.variant_id_map = self.variants.loc[
            self.variants["common_variant_mask"], ["matrix_index", "id"]
        ]
        self.variant_id_map = self.variant_id_map.sort_values("matrix_index")
        assert (self.variant_id_map["matrix_index"] == np.arange(self.n_variants)).all()
        self.variant_id_map = self.variant_id_map.set_index("matrix_index", drop=False)

        self.matrix_index = -np.ones(self.variants["id"].max() + 2, dtype=np.int32)
        self.matrix_index[self.variants["id"].to_numpy()] = self.variants[
            "matrix_index"
        ]

        if self.group_common:
            self.setup_common_groups()

    def setup_common_groups(self):
        logger.debug("Setting up groups for common variants")
        logger.debug("    Computing grouping")
        common_variant_groups = self.variants.loc[
            self.variants["common_variant_mask"],
            ["id", "matrix_index", self.grouping_column],
        ].set_index("matrix_index", drop=False)

        common_variant_groups = common_variant_groups.explode(self.grouping_column)
        common_variant_groups = common_variant_groups[
            common_variant_groups["gene_ids"].notna()
        ]

        if self.return_sparse:
            logger.debug("    Computing group IDs")
            if not hasattr(self, "group_names"):
                self.group_names = list(
                    common_variant_groups[self.grouping_column].unique()
                )
            group_ids = pd.DataFrame(
                {
                    self.grouping_column: self.group_names,
                    "group_ids": np.arange(len(self.group_names)),
                }
            )
            common_variant_groups = safe_merge(
                common_variant_groups,
                group_ids,
                on=self.grouping_column,
                validate="m:1",
            )

            common_variant_groups = common_variant_groups[["matrix_index", "group_ids"]]
            common_variant_groups = common_variant_groups.groupby("matrix_index")
            common_variant_groups = common_variant_groups.agg(lambda x: x.to_list())

            common_variant_groups = common_variant_groups.sort_index()
            self.common_variant_groups = common_variant_groups["group_ids"]

            assert (
                len(self.common_variant_groups)
                == self.variants["matrix_index"].max() + 1
            )
            assert np.all(
                self.common_variant_groups.index.to_numpy()
                == np.arange(len(self.common_variant_groups))
            )
        else:
            logger.debug("    Computing within-group matrix indices")
            common_variant_groups = common_variant_groups.groupby(self.grouping_column)

            self.group_names = []
            self.group_matrix_maps = []
            for name, group in common_variant_groups:
                self.group_names += [name]
                self.group_matrix_maps.append(group["matrix_index"].to_numpy())

    def get_variant_groups(self):
        logger.debug("Setting up groups for common variants")
        logger.debug("    Computing grouping")
        common_variant_groups = self.variants.loc[
            self.variants["common_variant_mask"],
            ["id", "matrix_index", self.grouping_column],
        ].set_index("matrix_index", drop=False)
        common_variant_groups = common_variant_groups.explode(self.grouping_column)
        logger.info("    Computing within-group matrix indices")
        gene_df = pd.DataFrame(
            {
                self.grouping_column: list(
                    common_variant_groups[self.grouping_column].dropna().unique()
                ),
                "gene_index": np.arange(
                    common_variant_groups[self.grouping_column].nunique()
                ),
            }
        )
        common_variant_groups = common_variant_groups.merge(
            gene_df, how="inner", on=self.grouping_column
        )
        return common_variant_groups

    def get_common_variants(
        self, sparse_variants: np.ndarray, sparse_genotype: np.ndarray
    ):
        padding_mask = sparse_variants >= 0
        if self.variants_to_keep is not None:
            padding_mask &= np.isin(sparse_variants, self.variants_to_keep)

        masked_sparse_variants = sparse_variants[padding_mask]
        sparse_common_variants = self.matrix_index[masked_sparse_variants]
        masked_sparse_genotype = sparse_genotype[padding_mask]

        common_variant_mask = sparse_common_variants >= 0
        sparse_common_variants = sparse_common_variants[common_variant_mask]
        sparse_genotype = masked_sparse_genotype[common_variant_mask]

        if not self.use_common_variants:
            common_variants = torch.tensor([], dtype=torch.float)
        else:
            if self.return_sparse:
                if self.group_common:
                    variants = [[] for _ in self.group_names]
                    genotypes = [[] for _ in self.group_names]
                    for var, gt in zip(sparse_common_variants, sparse_genotype):
                        for group_id in self.common_variant_groups.iloc[var]:
                            variants[group_id].append(var)
                            genotypes[group_id].append(gt)

                    common_variants = (
                        [torch.tensor(x, dtype=torch.long) for x in variants],
                        [torch.tensor(x, dtype=torch.float) for x in genotypes],
                    )
                else:
                    common_variants = (
                        torch.tensor(sparse_common_variants, dtype=torch.long),
                        torch.tensor(sparse_genotype, dtype=torch.float),
                    )
            else:
                common_variants = np.zeros(self.n_variants)
                common_variants[sparse_common_variants] = sparse_genotype
                common_variants = torch.tensor(common_variants, dtype=torch.float)

                if self.group_common:
                    common_variants = [
                        common_variants[vmap] for vmap in self.group_matrix_maps
                    ]

        return common_variants, masked_sparse_variants, masked_sparse_genotype

    def get_rare_variants(self, idx, all_sparse_variants, sparse_genotype):
        if not self.use_rare_variants:
            return torch.tensor([], dtype=torch.float)

        mask = self.variants.loc[all_sparse_variants, "rare_variant_mask"]
        rare_variants = all_sparse_variants[mask]
        rare_genotype = sparse_genotype[mask]
        rare_variant_annotations = self.rare_embedding.embed(
            idx, rare_variants, rare_genotype
        )

        return rare_variant_annotations

    def collate_fn(
        self, batch: Dict[str, List[Union[int, torch.Tensor]]]
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        key_lists = {k: [] for k in batch[0].keys()}
        for sample in batch:
            for k, v in sample.items():
                key_lists[k].append(v)

        result = {
            k: torch.stack(v)
            for k, v in key_lists.items()
            if k != "sample" and k != "rare_variant_annotations"
        }
        result["sample"] = key_lists["sample"]
        if self.use_rare_variants and hasattr(self.rare_embedding, "collate_fn"):
            result["rare_variant_annotations"] = self.rare_embedding.collate_fn(
                key_lists["rare_variant_annotations"]
            )
        else:
            result["rare_variant_annotations"] = torch.stack(
                key_lists["rare_variant_annotations"]
            )
        return result

    def get_metadata(self) -> Dict[str, Any]:
        result = {
            "variant_metadata": self.variants[
                ["id", "common_variant_mask", "rare_variant_mask", "matrix_index"]
            ]
        }
        if self.use_rare_variants:
            if hasattr(self.rare_embedding, "get_metadata"):
                result.update(
                    {"rare_embedding_metadata": self.rare_embedding.get_metadata()}
                )
        return result

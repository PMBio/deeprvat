import os
import sys
import logging
from deeprvat.data import DenseGTDataset
import yaml
from typing import Dict, Tuple
import pandas as pd
import itertools
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import numpy as np
from pathlib import Path
import pytest
import torch
import zarr

from deeprvat.deeprvat.train import make_dataset_, MultiphenoDataset

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# TODO:
# 1. Test cache_tensors
# 2. Test edge cases for data
# 3. Maybe fix expected data?
# 4. Test subset in data[p]["samples"]
# 5. Test entire script (loading from CLI)
# 6. Test "indices" element of dataset batches
# 7. Test val dataset
# 8. Test inputting temporary directory explicitly
# 9. Different min_variant_counts

script_dir = Path(__file__).resolve().parent
repo_base_dir = script_dir.parent.parent
tests_data_dir = script_dir / "test_data" / "training"
example_data_dir = script_dir.parent / "example"
test_config_file = tests_data_dir / "config.yaml"


with open(tests_data_dir / "phenotypes.txt", "r") as f:
    phenotypes = f.read().strip().split("\n")

arrays = ("input_tensor", "covariates", "y")


def make_multipheno_data():
    data = {
        p: {
            a: torch.tensor(zarr.open(tests_data_dir / p / "deeprvat" / f"{a}.zarr")[:])
            for a in arrays[1:]
        }
        for p in phenotypes
    }
    for p in phenotypes:
        data[p]["input_tensor_zarr"] = zarr.open(
            tests_data_dir / p / "deeprvat/input_tensor.zarr"
        )
        data[p]["input_tensor"] = data[p]["input_tensor_zarr"][:]
        data[p]["samples"] = {"train": np.arange(data[p]["y"].shape[0])}

    return data


def subset_samples(
    data: Dict[str, torch.Tensor], min_variant_count: int = 0
) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
    data = copy.deepcopy(data)

    n_variant_mask = np.sum(np.any(data["input_tensor"], axis=(1, 2)), axis=1) >= 1
    nan_mask = ~np.isnan(data["y"].squeeze())
    mask = n_variant_mask & nan_mask

    for a in arrays:
        data[a] = data[a][mask]

    return mask, data


def multiphenodataset_reference(data):
    data = copy.deepcopy(data)
    data = {p: {a: data[p][a].squeeze() for a in arrays} for p in phenotypes}
    return data


def reconstruct_from_batches(dl: DataLoader):
    array_lists = {p: {a: [] for a in arrays} for p in phenotypes}
    for batch in tqdm(dl):
        for p, data in batch.items():
            array_lists[p]["input_tensor"].append(
                data["rare_variant_annotations"].numpy()
            )
            array_lists[p]["covariates"].append(data["covariates"].numpy())
            array_lists[p]["y"].append(data["y"].numpy())

    return {
        p: {a: np.concatenate(array_lists[p][a]) for a in arrays} for p in phenotypes
    }


@pytest.fixture
def multipheno_data():
    data = make_multipheno_data()
    reference = multiphenodataset_reference(data)
    return data, reference


@pytest.mark.parametrize(
    "cache_tensors, batch_size",
    list(itertools.product([False, True], [1, 13, 1024])),
)
def test_multiphenodataset(multipheno_data, cache_tensors: bool, batch_size: int):
    data, reference = multipheno_data
    dataset = MultiphenoDataset(data, batch_size, cache_tensors=cache_tensors)
    dl = DataLoader(dataset, batch_size=None, num_workers=0)
    reconstructed = reconstruct_from_batches(dl)

    for p in phenotypes:
        for a in arrays:
            assert np.allclose(reference[p][a], reconstructed[p][a])


@pytest.mark.parametrize(
    "phenotype, min_variant_count",
    list(zip(phenotypes, [0, 1, 2])),
)
def test_make_dataset(phenotype: str, min_variant_count: int, tmp_path: Path):
    os.chdir(repo_base_dir)

    with open(test_config_file, "r") as f:
        config = yaml.safe_load(f)

    # Set phenotype and seed gene files in config
    config["training_data"]["dataset_config"]["y_phenotypes"] = [phenotype]
    seed_gene_file = str(tests_data_dir / phenotype / "deeprvat" / "seed_genes.parquet")
    config["seed_gene_file"] = seed_gene_file
    config["training_data"]["dataset_config"]["gene_file"] = seed_gene_file
    config["training_data"]["dataset_config"]["rare_embedding"]["config"][
        "gene_file"
    ] = seed_gene_file
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    # This is the function we want to test
    input_tensor_out_file = str(tmp_path / "input_tensor.zarr")
    covariates_out_file = str(tmp_path / "covariates.zarr")
    y_out_file = str(tmp_path / "y.zarr")
    logger.info("Constructing test dataset")
    test_ds = make_dataset_(
        False,
        False,
        1,
        None,
        config_file,
        input_tensor_out_file,
        covariates_out_file,
        y_out_file,
    )

    # Load the data it output
    test_data = {}
    test_data["input_tensor"] = zarr.load(input_tensor_out_file)
    test_data["covariates"] = zarr.load(covariates_out_file)
    test_data["y"] = zarr.load(y_out_file)

    # Assert data shapes agree
    assert test_data["input_tensor"].shape[0] == test_data["covariates"].shape[0]
    assert test_data["input_tensor"].shape[0] == test_data["y"].shape[0]

    # Assert all of kept (up to index min_variant_count - 1) has some nonzero values
    if min_variant_count > 0:
        assert np.all(
            np.any(
                test_data["input_tensor"][:, :, :min_variant_count, :] != 0.0,
                axis=(1, 3),
            )
        )

    # Load data in single batch as reference to check against make_dataset_
    logger.info("Constructing reference dataset")
    reference_ds = DenseGTDataset(
        gt_file=config["training_data"]["gt_file"],
        variant_file=config["training_data"]["variant_file"],
        split="",
        skip_y_na=True,
        **config["training_data"]["dataset_config"],
    )
    reference_dl = DataLoader(
        reference_ds, collate_fn=reference_ds.collate_fn, batch_size=len(reference_ds)
    )
    reference_data = next(iter(reference_dl))
    reference_data = {
        "input_tensor": reference_data["rare_variant_annotations"].numpy(),
        "covariates": reference_data["x_phenotypes"].numpy(),
        "y": reference_data["y"].numpy(),
    }

    # Subset reference data
    mask, reference_subset = subset_samples(reference_data, min_variant_count)

    # Assert all of dropped (beyond index min_variant_count - 1) is 0.
    assert np.all(
        reference_subset["input_tensor"][~mask, :, min_variant_count:, :] == 0.0
    )

    for a in arrays:
        # Compare make_dataset_ output to reference_subset
        assert np.array_equal(test_data[a], reference_subset[a])

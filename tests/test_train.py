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

from deeprvat.deeprvat.train import MultiphenoDataset

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
tests_data_dir = script_dir / "test_data/training"

with open(tests_data_dir / "phenotypes.txt", "r") as f:
    phenotypes = f.read().strip().split("\n")

arrays = ("input_tensor", "covariates", "y")


def make_data():
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


def multiphenodataset_reference(data):
    data = copy.deepcopy(data)
    data = {p: {a: data[p][a].squeeze() for a in arrays} for p in phenotypes}
    for _, d in data.items():
        n_variant_mask = np.sum(np.any(d["input_tensor"], axis=(1, 2)), axis=1) >= 1
        nan_mask = ~np.isnan(d["y"].numpy().squeeze())
        mask = n_variant_mask & nan_mask
        for a in arrays:
            d[a] = d[a][mask]

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
def simple_data():
    data = make_data()
    reference = multiphenodataset_reference(data)
    return data, reference


@pytest.mark.parametrize(
    "cache_tensors, batch_size", itertools.product([False, True], [1, 13, 1024])
)
def test_multiphenodataset(simple_data, cache_tensors: bool, batch_size: int):
    data, reference = simple_data
    dataset = MultiphenoDataset(data, 1, batch_size, cache_tensors=cache_tensors)
    dl = DataLoader(dataset, batch_size=None, num_workers=0)
    reconstructed = reconstruct_from_batches(dl)

    for p in phenotypes:
        for a in arrays:
            assert np.allclose(reference[p][a], reconstructed[p][a])

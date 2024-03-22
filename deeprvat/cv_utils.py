import pandas as pd
import yaml
import os
import sys
from typing import Optional
import re

# import pickle
import logging
import click
import copy
import zarr
import numpy as np
from numcodecs import Blosc
from pathlib import Path
from deeprvat.utils import (
    standardize_series,
    my_quantile_transform,
)

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
DATA_SLOT_DICT = {
    "deeprvat": ["data", "training_data"],
    "seed_genes": ["data"],
}

module_folder_dict = {
    "seed_genes": "baseline",
    "deeprvat": "deeprvat",
    "alternative_burdens": "alternative_burdens",
}


@click.group()
def cli():
    pass


@cli.command()
@click.option("--module", "-m", multiple=True)
@click.option("--fold", type=int)
@click.option("--fold-specific-baseline", is_flag=True)
@click.option("--n-folds", type=int, default=5)
@click.argument("input_config", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path(), default="./")
def spread_config(
    input_config, out_path, module, fold_specific_baseline, fold, n_folds
):
    data_modules = module

    with open(input_config) as f:
        config_template = yaml.safe_load(f)
    split = "train"
    cv_path = f"{config_template['cv_path']}/{n_folds}_fold"
    for module in data_modules:
        config = copy.deepcopy(config_template)
        sample_file = f"{cv_path}/samples_{split}{fold}.pkl"
        logger.info(f"setting sample file {sample_file}")
        config["sample_file"] = sample_file

        if (module == "deeprvat") | (module == "deeprvat_pretrained"):
            logger.info("Writing baseline directories")
            old_baseline = copy.deepcopy(config["baseline_results"])
            if fold_specific_baseline:
                config["baseline_results"] = [
                    {"base": f'{r["base"]}/cv_split{fold}/baseline', "type": r["type"]}
                    for r in old_baseline
                ]
            logger.info(config["baseline_results"])
        logger.info(f"Writing config for module {module}")
        with open(f"{out_path}/{module_folder_dict[module]}/config.yaml", "w") as f:
            yaml.dump(config, f)


@cli.command()
@click.option("--fold", type=int)
@click.option("--n-folds", type=int, default=5)
@click.argument("input_config", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
def generate_test_config(input_config, out_file, fold, n_folds):
    with open(input_config) as f:
        config = yaml.safe_load(f)
    cv_path = f"{config['cv_path']}/{n_folds}_fold"
    split = "test"
    sample_file = f"{cv_path}/samples_{split}{fold}.pkl"
    logger.info(f"setting sample file {sample_file}")
    config["sample_file"] = sample_file
    with open(out_file, "w") as f:
        yaml.dump(config, f)


@cli.command()
@click.option("--link-burdens", type=click.Path())
@click.option("--burden-dirs", "-b", multiple=True)
@click.argument("out_dir", type=click.Path(), default="./")
@click.argument("config_file", type=click.Path(exists=True))
def combine_test_set_burdens(
    out_dir,
    link_burdens,
    burden_dirs,
    config_file,
):
    with open(config_file) as f:
        config = yaml.safe_load(f)
    compression_level = 1
    skip_burdens = link_burdens is not None
    n_total_samples = []
    for burden_dir in burden_dirs:
        print(burden_dir)
        this_y = zarr.open(f"{burden_dir}/y.zarr")
        this_x = zarr.open(f"{burden_dir}/x.zarr")
        # this_burdens = zarr.open(f'{burden_dir}/burdens.zarr')

        assert this_y.shape[0] == this_x.shape[0]  # == this_burdens.shape[0]
        n_total_samples.append(this_y.shape[0])

    n_total_samples = np.sum(n_total_samples)
    print(f"Total number of samples {n_total_samples}")
    if not skip_burdens:
        this_burdens = zarr.open(
            f"{burden_dir}/burdens.zarr"
        )  # any burden tensor (here from the last file to get dims 1 -n)
        burdens = zarr.open(
            Path(out_dir) / "burdens.zarr",
            mode="a",
            shape=(n_total_samples,) + this_burdens.shape[1:],
            chunks=(1000, 1000),
            dtype=np.float32,
            compressor=Blosc(clevel=compression_level),
        )
        print(f"burdens shape: {burdens.shape}")
    else:
        burdens = None

    y = zarr.open(
        Path(out_dir) / "y.zarr",
        mode="a",
        shape=(n_total_samples,) + this_y.shape[1:],
        chunks=(None, None),
        dtype=np.float32,
        compressor=Blosc(clevel=compression_level),
    )
    x = zarr.open(
        Path(out_dir) / "x.zarr",
        mode="a",
        shape=(n_total_samples,) + this_x.shape[1:],
        chunks=(None, None),
        dtype=np.float32,
        compressor=Blosc(clevel=compression_level),
    )

    start_idx = 0

    for burden_dir in burden_dirs:
        this_y = zarr.open(f"{burden_dir}/y.zarr")[:]
        end_idx = start_idx + this_y.shape[0]
        this_x = zarr.open(f"{burden_dir}/x.zarr")[:]
        if not skip_burdens:
            logger.info("writing burdens")
            this_burdens = zarr.open(f"{burden_dir}/burdens.zarr")[:]
            burdens[start_idx:end_idx] = this_burdens
        print((start_idx, end_idx))
        y[start_idx:end_idx] = this_y
        x[start_idx:end_idx] = this_x
        start_idx = end_idx

    y_transformation = config["data"]["dataset_config"].get("y_transformation", None)
    standardize_xpheno = config["data"]["dataset_config"].get(
        "standardize_xpheno", True
    )

    ## Analogously to what is done in densegt
    if standardize_xpheno:
        this_x = x[:]
        logger.info("  Standardizing combined covariates")
        for col in range(this_x.shape[1]):
            this_x[:, col] = standardize_series(this_x[:, col])
        x[:] = this_x
    if y_transformation is not None:
        this_y = y[:]
        n_unique_y_val = np.count_nonzero(~np.isnan(np.unique(this_y)))
        if n_unique_y_val == 2:
            logger.warning(
                "Not applying y transformation because y only has two values and seems to be binary"
            )
            y_transformation = None
        if y_transformation is not None:
            if y_transformation == "standardize":
                logger.info("  Standardizing combined target phenotype (y)")
                for col in range(this_y.shape[1]):
                    this_y[:, col] = standardize_series(this_y[:, col])
            elif y_transformation == "quantile_transform":
                logger.info(f"  Quantile transforming combined target phenotype (y)")
                for col in range(this_y.shape[1]):
                    this_y[:, col] = my_quantile_transform(this_y[:, col])
            y[:] = this_y
    print("done")
    if link_burdens is not None:
        source_path = Path(out_dir) / "burdens.zarr"
        source_path.unlink(missing_ok=True)
        source_path.symlink_to(link_burdens)
    genes = np.load(f"{burden_dirs[0]}/genes.npy")
    np.save(Path(out_dir) / "genes.npy", genes)


if __name__ == "__main__":
    cli()

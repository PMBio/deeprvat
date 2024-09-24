import logging
import zarr
import sys
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pandas as pd
import yaml
from tqdm import trange

from deeprvat.deeprvat.associate import load_one_model

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--tolerance", type=float, default=1e-2)
@click.option("--rtol", type=float, default=1e-5)
@click.option("--atol", type=float, default=1e-8)
@click.option("--n-repeats", type=int, default=1)
@click.argument("results-dir", type=click.Path(exists=True, path_type=Path))
@click.argument("reference-dir", type=click.Path(exists=True, path_type=Path))
@click.argument("phenotype", type=str, nargs=-1)
def compare_training(
    results_dir: Path,
    reference_dir: Path,
    phenotype: Tuple[str],
    tolerance: float,
    rtol: float,
    atol: float,
    n_repeats: int,
):
    # Compare training data

    for p in phenotype:
        # Compare seed genes
        seed_genes = list(
            pd.read_parquet(results_dir / p / "deeprvat/seed_genes.parquet")["gene"]
        )
        reference_seed_genes = list(
            pd.read_parquet(reference_dir / p / "deeprvat/seed_genes.parquet")["gene"]
        )

        if not seed_genes == reference_seed_genes:
            raise RuntimeError(f"FAIL! Seed genes differ from reference.")
        # else:
        #     logger.info(
        #         f"PASS! Seed genes agree with reference"
        #     )

        # Compare input data arrays
        for a in ("covariates", "input_tensor", "y"):
            array = zarr.load(results_dir / p / f"deeprvat/{a}.zarr")
            reference_array = zarr.load(reference_dir / p / f"deeprvat/{a}.zarr")
            if not np.allclose(
                array,
                reference_array,
                rtol=rtol,
                atol=atol,
            ):
                raise RuntimeError(
                    f"FAIL! Max difference between reference and results larger than tolerance"
                    f"for array {f'{p}/{a}.zarr'}"
                )

    logger.info("PASS! Training data agrees within tolerance.")

    # Compare models

    model_dir = results_dir / "models"
    model_reference_dir = reference_dir / "models"

    with open(model_dir / "model_config.yaml") as f:
        config = yaml.safe_load(f)

    with open(model_reference_dir / "model_config.yaml") as f:
        reference_config = yaml.safe_load(f)

    for r in range(n_repeats):
        logger.info(f"Checking repeat {r}")

        model = load_one_model(config, str(model_dir / f"repeat_{r}/best/bag_0.ckpt"))
        reference_model = load_one_model(
            reference_config, str(model_reference_dir / f"repeat_{r}/best/bag_0.ckpt")
        )

        max_difference = max(
            [
                (p2 - p1).abs().max().item()
                for p1, p2 in zip(model.parameters(), reference_model.parameters())
            ]
        )

        if max_difference > tolerance:
            raise RuntimeError(
                f"FAIL! Max difference between model and reference parameters (repeat {r}) "
                f"differs by {max_difference} > {tolerance=}"
            )
        else:
            logger.info(
                f"PASS! Max difference between model and reference parameters (repeat {r}) "
                f"is {max_difference} <= {tolerance=}"
            )

    logger.info("PASS! Model parameters agree.")


@cli.command()
@click.option("--rtol-burdens", type=float, default=1e-3)
@click.option("--atol-burdens", type=float, default=2e-2)
@click.option("--rtol-xy", type=float, default=1e-2)
@click.option("--atol-xy", type=float, default=1e-2)
@click.option("--rtol-assoc", type=float, default=1e-2)
@click.option("--atol-assoc", type=float, default=1e-2)
@click.argument("results-dir", type=click.Path(exists=True, path_type=Path))
@click.argument("reference-dir", type=click.Path(exists=True, path_type=Path))
@click.argument("phenotype", type=str, nargs=-1)
def compare_burdens(
    results_dir: Path,
    reference_dir: Path,
    phenotype: Tuple[str],
    rtol_burdens: float,
    atol_burdens: float,
    rtol_xy: float,
    atol_xy: float,
    rtol_assoc: float,
    atol_assoc: float,
):
    for a in ("sample_ids.zarr", "burdens.zarr", "genes.npy"):
        load_fn = np.load if a == "genes.npy" else zarr.load
        array = load_fn(results_dir / "burdens" / a)
        reference_array = load_fn(reference_dir / "burdens" / a)

        if a == "sample_ids.zarr":
            all_close = np.array_equal(array, reference_array)
        else:
            all_close = np.allclose(
                array,
                reference_array,
                rtol=rtol_burdens,
                atol=atol_burdens,
            )

        if not all_close:
            raise RuntimeError(
                f"FAIL! Max difference between results and reference results "
                f"(array {a}) larger than tolerance.\n"
                f"{reference_array}\n"
                f"{array}"
                #f"{np.max(np.abs(reference_array - array))=}"
            )

    for p in phenotype:
        for a in ("sample_ids.zarr", "x.zarr", "y.zarr"):
            array = zarr.load(results_dir / p / "deeprvat/xy" / a)
            reference_array = zarr.load(reference_dir / p / "deeprvat/xy" / a)

            if a == "sample_ids.zarr":
                all_close = np.array_equal(array, reference_array)
            else:
                all_close = np.allclose(
                    array,
                    reference_array,
                    equal_nan=True,
                    rtol=rtol_xy,
                    atol=atol_xy,
                )

            if not all_close:
                raise RuntimeError(
                    f"FAIL! Max difference between results and reference results "
                    f"(phenotype {p}, array {a}) larger than tolerance"
                )

        # else:
        #     logger.info(
        #         f"PASS! Max difference between results and reference results (phenotype {p}) "
        #         f"within tolerance"
        #     )

    logger.info("PASS! All tests successful")


@cli.command()
@click.option("--rtol-assoc", type=float, default=1e-2)
@click.option("--atol-assoc", type=float, default=1e-2)
@click.argument("results-dir", type=click.Path(exists=True, path_type=Path))
@click.argument("reference-dir", type=click.Path(exists=True, path_type=Path))
@click.argument("phenotype", type=str, nargs=-1)
def compare_association(
    results_dir: Path,
    reference_dir: Path,
    phenotype: Tuple[str],
    rtol_assoc: float,
    atol_assoc: float,
):
    for p in phenotype:
        logger.info(f"Checking phenotype {p}")

        results = pd.read_parquet(results_dir / p / "deeprvat/eval/all_results.parquet")
        reference_results = pd.read_parquet(
            reference_dir / p / "deeprvat/eval/all_results.parquet"
        )
        string_cols = ["phenotype", "Method", "Discovery type"]
        for c in string_cols:
            assert (
                (results[c].isna() & results[c].isna())
                | (results[c] == reference_results[c])
            ).all()

        numerical_cols = ["gene", "beta", "pval", "-log10pval", "pval_corrected"]
        all_close = np.allclose(
            results[numerical_cols].to_numpy(),
            reference_results[numerical_cols].to_numpy(),
            equal_nan=True,
            rtol=rtol_assoc,
            atol=atol_assoc,
        )

        if not all_close:
            raise RuntimeError(
                f"FAIL! Max difference between results and reference results (phenotype {p}) "
                f"larger than tolerance"
            )
        # else:
        #     logger.info(
        #         f"PASS! Max difference between results and reference results (phenotype {p}) "
        #         f"within tolerance"
        #     )

    logger.info("PASS! All tests successful")


if __name__ == "__main__":
    cli()
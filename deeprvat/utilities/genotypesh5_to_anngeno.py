import logging
import sys
from pathlib import Path
from typing import Optional, Union

import click
import h5py
import numpy as np
import pandas as pd
from anngeno import AnnGeno
from tqdm import trange

PathLike = Union[str, Path]

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _convert_genotypes_h5(
    variant_file: PathLike,
    phenotype_file: PathLike,
    genotype_file: PathLike,
    out_file: PathLike,
    batch_size: int = 100,
    max_samples: Optional[int] = None,
):
    logger.info("Reading sample IDs")
    samples = pd.read_parquet(phenotype_file).index.to_numpy().astype(str)
    if max_samples is not None:
        samples = samples[:max_samples]
    n_samples = len(samples)
    logger.info("Reading variant metadata")
    variant_metadata = pd.read_parquet(
        variant_file, columns=["id", "chrom", "pos", "ref", "alt"]
    )
    n_variants = len(variant_metadata)
    with h5py.File(genotype_file, "r") as g:
        gt_matrix = g["genotype_matrix"]
        variant_matrix = g["variant_matrix"]
        assert (
            n_samples == gt_matrix.shape[0]
            or max_samples is not None
            and max_samples < gt_matrix.shape[0]
        )

        # TODO: Rewrite below here to reflect changes in AnnGeno
        logger.info("Initializing AnnGeno object")
        ag = AnnGeno(
            out_file,
            filemode="w",
            samples=samples,
            variant_metadata=variant_metadata.drop(columns="id"),
        )

        logger.info("Transforming genotype file")
        variant_ids = ag.make_variant_ids(variant_metadata.sort_values("id"))
        reindexer = ag.variant_id_reindexer(variant_ids)
        for start_idx in trange(0, n_samples, batch_size, desc="Chunks"):
            end_idx = min(start_idx + batch_size, n_samples)
            sample_slice = slice(start_idx, end_idx)
            this_genotypes = []
            for i in range(start_idx, end_idx):
                ids = variant_matrix[i, variant_matrix[i] != -1]
                gts = gt_matrix[i, gt_matrix[i] != -1]

                genotypes_dense = np.zeros(n_variants, dtype=np.uint8)
                genotypes_dense[ids] = gts

                this_genotypes.append(genotypes_dense)

            ag.set_samples(sample_slice, np.stack(this_genotypes, axis=0)[:, reindexer])


@click.command()
@click.option("--batch-size", type=int, default=100)
@click.option("--max-samples", type=int)
@click.argument("variant-file", type=click.Path(exists=True, path_type=Path))
@click.argument("phenotype-file", type=click.Path(exists=True, path_type=Path))
@click.argument("genotype-file", type=click.Path(exists=True, path_type=Path))
@click.argument("out-file", type=click.Path(path_type=Path))
def convert_genotypes_h5(
    variant_file: PathLike,
    phenotype_file: PathLike,
    genotype_file: PathLike,
    out_file: PathLike,
    batch_size: int = 100,
    max_samples: Optional[int] = None,
):
    _convert_genotypes_h5(
        variant_file=variant_file,
        phenotype_file=phenotype_file,
        genotype_file=genotype_file,
        out_file=out_file,
        batch_size=batch_size,
        max_samples=max_samples,
    )


if __name__ == "__main__":
    convert_genotypes_h5()

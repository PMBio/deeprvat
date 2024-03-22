import gc
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import click
import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm, trange

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def drop_rows(df: pd.DataFrame, df_to_drop: pd.DataFrame) -> pd.DataFrame:
    df_to_drop = pd.merge(df, df_to_drop)
    df = pd.concat([df, df_to_drop])
    return df.drop_duplicates(keep=False)


def ragged_to_matrix(rows: List[np.ndarray], pad_value: int = -1) -> np.ndarray:
    max_len = max([r.shape[0] for r in rows])
    matrix = np.stack(
        [
            np.pad(
                r,
                (0, max_len - r.shape[0]),
                "constant",
                constant_values=(pad_value, pad_value),
            )
            for r in tqdm(rows)
        ]
    )
    return matrix


def process_sparse_gt_file(
    file: str,
    variants: pd.DataFrame,
    samples: List[str],
    calls_to_exclude: pd.DataFrame = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    sparse_gt = pd.read_table(
        file,
        names=["chrom", "pos", "ref", "alt", "sample", "gt"],
        engine="pyarrow",
        index_col=None,
    )
    sparse_gt = sparse_gt[sparse_gt["sample"].isin(samples)]

    sparse_gt = drop_rows(sparse_gt, calls_to_exclude)

    try:
        sparse_gt = pd.merge(sparse_gt, variants, validate="m:1")
    except:
        logging.error(f"Error while validating merge of {file} with variants")
        sparse_gt.to_csv(f"{file}_sgt_error.tsv.gz")
        variants.to_csv(f"{file}_variants_error.tsv.gz")
        raise

    sparse_gt = sparse_gt[["sample", "gt", "id"]]

    sparse_gt = sparse_gt.groupby("sample")
    variant_ids = []
    genotype = []
    for s in samples:
        if s in sparse_gt.groups.keys():
            group = sparse_gt.get_group(s)
            variant_ids.append(group["id"].values)
            genotype.append(group["gt"].values)
        else:
            variant_ids.append(np.array([], dtype=np.int32))
            genotype.append(np.array([], dtype=np.int32))

    return variant_ids, genotype


def postprocess_sparse_gt(
    processed: List[Tuple[List[np.ndarray], List[np.ndarray]]],
    result_index: int,
    n_rows: int,
) -> np.ndarray:
    result = [p[result_index] for p in processed]
    return [np.concatenate([r[i] for r in result]) for i in range(n_rows)]


def write_genotype_file(
    f: h5py.File,
    samples: np.ndarray,
    variant_matrix: np.ndarray,
    genotype_matrix: np.ndarray,
):
    f.create_dataset("samples", data=samples, dtype=h5py.string_dtype())
    f.create_dataset("variant_matrix", data=variant_matrix, dtype=np.int32)
    f.create_dataset("genotype_matrix", data=genotype_matrix, dtype=np.int8)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("variant_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path(writable=True))
@click.argument("duplicates_file", type=click.Path(writable=True))
def add_variant_ids(variant_file: str, out_file: str, duplicates_file: str):
    variants = pd.read_csv(
        variant_file, sep="\t", names=["chrom", "pos", "ref", "alt"], index_col=False
    )

    duplicates = variants[variants.duplicated()]

    if Path(duplicates_file).suffix == ".parquet":
        logging.info("Writing duplicates in parquet format")
        duplicates.to_parquet(duplicates_file, index=False)
    else:
        logging.info("Writing duplicates in tsv format")
        duplicates.to_csv(duplicates_file, sep="\t", header=False, index=False)

    logging.info(f"Wrote {len(duplicates)} duplicates to {duplicates_file}")

    total_variants = len(variants)
    variants = variants.drop_duplicates()
    variants["id"] = range(len(variants))

    if Path(out_file).suffix == ".parquet":
        logging.info("Writing duplicates in parquet format")
        variants.to_parquet(out_file, index=False)
    else:
        logging.info("Writing duplicates in tsv format")
        variants.to_csv(out_file, sep="\t", index=False)

    logging.info(
        f"Wrote {len(variants)} variants to {out_file} "
        f"(dropped {total_variants - len(variants)} duplicates)"
    )


def get_file_chromosome(file, col_names, chrom_field="chrom"):
    chrom_df = pd.read_csv(
        file, names=col_names, sep="\t", index_col=None, nrows=1, usecols=[chrom_field]
    )

    chrom = None
    if not chrom_df.empty:
        chrom = chrom_df[chrom_field][0]

    return chrom


@cli.command()
@click.option("--exclude-variants", type=click.Path(exists=True), multiple=True)
@click.option("--exclude-samples", type=click.Path(exists=True))
@click.option("--exclude-calls", type=click.Path(exists=True))
@click.option("--chromosomes", type=str)
@click.option("--threads", type=int, default=1)
@click.option("--skip-sanity-checks", is_flag=True)
@click.argument("variant-file", type=click.Path(exists=True))
@click.argument("samples", type=click.Path(exists=True))
@click.argument("sparse-gt", type=click.Path(exists=True))
@click.argument("out-file", type=click.Path())
def process_sparse_gt(
    exclude_variants: List[str],
    exclude_samples: Optional[str],
    exclude_calls: Optional[str],
    chromosomes: Optional[str],
    threads: int,
    skip_sanity_checks: bool,
    variant_file: str,
    samples: str,
    sparse_gt: str,
    out_file: str,
):
    logging.info("Reading variants...")
    start_time = time.time()

    variants = pd.read_table(variant_file, engine="pyarrow")

    # Filter all variants based on chromosome
    if chromosomes is not None:
        chromosomes = [f"chr{chrom}" for chrom in chromosomes.split(",")]
        variants = variants[variants["chrom"].isin(chromosomes)]
    total_variants = len(variants)

    if len(exclude_variants) > 0:
        variant_exclusion_files = [
            v for directory in exclude_variants for v in Path(directory).rglob("*.tsv*")
        ]

        exclusion_file_cols = ["chrom", "pos", "ref", "alt"]
        variants_to_exclude = pd.concat(
            [
                pd.read_csv(v, sep="\t", names=exclusion_file_cols)
                for v in tqdm(variant_exclusion_files)
            ],
            ignore_index=True,
        )
        if chromosomes is not None:
            variants_to_exclude = variants_to_exclude[
                variants_to_exclude["chrom"].isin(chromosomes)
            ]
        variants_to_exclude = variants_to_exclude.drop_duplicates(ignore_index=True)
        variant_ids_to_exclude = pd.merge(
            variants_to_exclude, variants, validate="1:1"
        )["id"]
        variants = variants[~variants["id"].isin(variant_ids_to_exclude)]
        if not skip_sanity_checks:
            assert total_variants - len(variants) == len(variants_to_exclude)

    logging.info(f"Dropped {total_variants - len(variants)} variants")
    logging.info(f"...done ({time.time() - start_time} s)")

    logging.info("Processing samples")
    samples = set(pd.read_csv(samples, header=None).loc[:, 0])
    if exclude_samples is not None:
        total_samples = len(samples)

        if sample_exclusion_files := list(Path(exclude_samples).rglob("*.csv")):
            samples_to_exclude = set(
                pd.concat(
                    [
                        pd.read_csv(s, header=None).loc[:, 0]
                        for s in sample_exclusion_files
                    ],
                    ignore_index=True,
                )
            )
            samples -= samples_to_exclude
            logging.info(f"Dropped {total_samples - len(samples)} samples")
        else:
            logging.info(f"Found no samples to exclude in {exclude_samples}")

    samples = list(samples)

    logging.info("Processing sparse GT files by chromosome")
    total_calls_dropped = 0
    variant_groups = variants.groupby("chrom")
    logger.info(f"variant groups are {variant_groups.groups.keys()}")
    for chrom in tqdm(variant_groups.groups.keys()):
        logging.info(f"Processing chromosome {chrom}")
        logging.info("Reading in filtered calls")

        exclude_calls_file_cols = ["chrom", "pos", "ref", "alt", "sample"]

        calls_to_exclude = pd.DataFrame(columns=exclude_calls_file_cols)

        if exclude_calls is not None:
            exclude_calls_chrom = Path(exclude_calls).rglob("*.tsv*")
            exclude_calls_chrom_files = []
            for exclude_call_file in exclude_calls_chrom:
                file_chrom = get_file_chromosome(
                    exclude_call_file, col_names=exclude_calls_file_cols
                )

                if file_chrom == chrom:
                    exclude_calls_chrom_files.append(exclude_call_file)

            if exclude_calls_chrom_files:
                calls_to_exclude = pd.concat(
                    [
                        pd.read_csv(
                            c,
                            names=exclude_calls_file_cols,
                            sep="\t",
                            index_col=None,
                        )
                        for c in tqdm(exclude_calls_chrom_files)
                    ],
                    ignore_index=True,
                )
                calls_dropped = len(calls_to_exclude)
                total_calls_dropped += calls_dropped
                logging.info(f"Dropped {calls_dropped} calls")

        logging.info("Processing sparse GT files")

        variants_chrom = variant_groups.get_group(chrom)

        sparse_file_cols = ["chrom", "pos", "ref", "alt", "sample", "gt"]

        sparse_gt_chrom = [
            f
            for f in Path(sparse_gt).rglob("*.tsv*")
            if get_file_chromosome(f, col_names=sparse_file_cols) == chrom
        ]

        logging.info(f"sparse gt chrom(es) are: {sparse_gt_chrom}")

        processed = Parallel(n_jobs=threads, verbose=50)(
            delayed(process_sparse_gt_file)(
                f.as_posix(), variants_chrom, samples, calls_to_exclude
            )
            for f in sparse_gt_chrom
        )

        postprocessed_gt = postprocess_sparse_gt(processed, 1, len(samples))
        postprocessed_variants = postprocess_sparse_gt(processed, 0, len(samples))

        logging.info("Ordering gt and variant matrices")
        order = [np.argsort(i) for i in postprocessed_variants]

        postprocessed_variants = [
            postprocessed_variants[i][order[i]]
            for i in range(len(postprocessed_variants))
        ]
        postprocessed_gt = [
            postprocessed_gt[i][order[i]] for i in range(len(postprocessed_gt))
        ]

        logging.info("Preparing GT arrays for storage")

        logger.info("padding ragged matrix")
        variant_matrix = ragged_to_matrix(postprocessed_variants)
        gt_matrix = ragged_to_matrix(postprocessed_gt)

        out_file_chrom = f"{out_file}_{chrom}.h5"
        Path(out_file_chrom).parents[0].mkdir(exist_ok=True, parents=True)
        logging.info(f"Writing to {out_file_chrom}")
        with h5py.File(out_file_chrom, "w") as f:
            sample_array = np.array(samples, dtype="S")
            write_genotype_file(f, sample_array, variant_matrix, gt_matrix)

    logging.info(f"Dropped {total_calls_dropped} calls in total")
    logging.info("Finished!")


@cli.command()
@click.argument("genotype-files", nargs=-1, type=click.Path(exists=True))
@click.argument("out-file", type=click.Path())
def combine_genotypes(genotype_files: List[str], out_file: str):
    with h5py.File(genotype_files[0]) as f:
        samples = f["samples"][:]
        n_samples = samples.shape[0]
        variant_matrix_list: List[List] = [[] for _ in range(n_samples)]
        genotype_matrix_list: List[List] = [[] for _ in range(n_samples)]

    mat_idx_offset = 0
    for file in tqdm(genotype_files, desc="Files"):
        with h5py.File(file) as f:
            var_matrix = f["variant_matrix"][:]
            gt_matrix = f["genotype_matrix"][:]
            for i in trange(n_samples, desc="Samples"):
                this_var = var_matrix[i, var_matrix[i, :] != -1]
                this_gt = gt_matrix[i, gt_matrix[i, :] != -1]
                assert this_var.shape == this_gt.shape

                variant_matrix_list[i].append(this_var + mat_idx_offset)
                genotype_matrix_list[i].append(this_gt)

    ragged_variant_matrix = [np.concatenate(vm) for vm in tqdm(variant_matrix_list)]
    del variant_matrix_list
    gc.collect()
    ragged_genotype_matrix = [np.concatenate(gt) for gt in tqdm(genotype_matrix_list)]
    del genotype_matrix_list
    gc.collect()
    variant_matrix = ragged_to_matrix(ragged_variant_matrix)
    del ragged_variant_matrix
    gc.collect()
    genotype_matrix = ragged_to_matrix(ragged_genotype_matrix)
    del ragged_genotype_matrix
    gc.collect()
    with h5py.File(out_file, "a") as f:
        write_genotype_file(f, samples, variant_matrix, genotype_matrix)


if __name__ == "__main__":
    cli()

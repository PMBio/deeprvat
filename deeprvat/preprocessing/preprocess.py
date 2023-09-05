import gc
import logging
import os
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


def ragged_to_matrix(rows: List[np.ndarray],
                     pad_value: int = -1) -> np.ndarray:
    max_len = max([r.shape[0] for r in rows])
    matrix = np.stack([
        np.pad(
            r,
            (0, max_len - r.shape[0]),
            "constant",
            constant_values=(pad_value, pad_value),
        ) for r in tqdm(rows)
    ])
    return matrix


def process_sparse_gt_file(
    file: str,
    variants: pd.DataFrame,
    samples: List[str],
    calls_to_exclude: pd.DataFrame = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    sparse_gt = pd.read_csv(
        file,
        names=["chrom", "pos", "ref", "alt", "sample", "gt"],
        sep="\t",
        index_col=None,
    )
    sparse_gt = sparse_gt[sparse_gt["sample"].isin(samples)]

    if calls_to_exclude is not None:
        sparse_gt = drop_rows(sparse_gt, calls_to_exclude)

    try:
        sparse_gt = pd.merge(sparse_gt, variants, validate="m:1")
    except:
        logging.error(f"Error while validating merge of {file} with variants")
        # sparse_gt.to_csv(f"{file}_sgt_error.tsv.gz")
        # variants.to_csv(f"{file}_variants_error.tsv.gz")
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


def write_genotype_file(f: h5py.File,
                        samples: np.ndarray,
                        variant_matrix: np.ndarray,
                        genotype_matrix: np.ndarray,
                        count_variants: Optional[np.ndarray] = None):
    f.create_dataset("samples", data=samples, dtype=h5py.string_dtype())
    f.create_dataset("variant_matrix", data=variant_matrix, dtype=np.int32)
    f.create_dataset("genotype_matrix", data=genotype_matrix, dtype=np.int8)
    if count_variants is not None:
        f.create_dataset("count_variants", data=count_variants, dtype=np.int32)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("variant_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path(writable=True))
@click.argument("duplicates_file", type=click.Path(writable=True))
def add_variant_ids(variant_file: str, out_file: str, duplicates_file: str):
    variants = pd.read_csv(variant_file,
                           sep="\t",
                           names=["chrom", "pos", "ref", "alt"],
                           index_col=False)

    duplicates = variants[variants.duplicated()]

    if Path(duplicates_file).suffix == ".parquet":
        logging.info(f"Writing duplicates in parquet format")
        duplicates.to_parquet(duplicates_file, index=False)
    else:
        logging.info(f"Writing duplicates in tsv format")
        duplicates.to_csv(duplicates_file, sep="\t", header=False, index=False)

    logging.info(f"Wrote {len(duplicates)} duplicates to {duplicates_file}")

    total_variants = len(variants)
    variants = variants.drop_duplicates()
    variants["id"] = range(len(variants))

    if Path(out_file).suffix == ".parquet":
        logging.info(f"Writing duplicates in parquet format")
        variants.to_parquet(out_file, index=False)
    else:
        logging.info(f"Writing duplicates in tsv format")
        variants.to_csv(out_file, sep="\t", index=False)

    logging.info(f"Wrote {len(variants)} variants to {out_file} "
                 f"(dropped {total_variants - len(variants)} duplicates)")


@cli.command()
@click.option("--exclude-variants",
              type=click.Path(exists=True),
              multiple=True)
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
    logging.info("Reading and processing variants...")
    start_time = time.time()
    variants = pd.read_parquet(variant_file, engine="pyarrow")
    if chromosomes is not None:
        chromosomes = [f"chr{chrom}" for chrom in chromosomes.split(",")]
        variants = variants[variants["chrom"].isin(chromosomes)]
    total_variants = len(variants)
    if len(exclude_variants) > 0:
        variant_exclusion_files = [
            Path(directory) / v for directory in exclude_variants
            for v in Path(directory).glob("*.tsv*")
        ]

        variants_to_exclude = pd.concat(
            [
                pd.read_csv(v, sep="\t", names=["chrom", "pos", "ref", "alt"])
                for v in tqdm(variant_exclusion_files, )
            ],
            ignore_index=True,
        )
        if chromosomes is not None:
            variants_to_exclude = variants_to_exclude[
                variants_to_exclude["chrom"].isin(chromosomes)]
        variants_to_exclude = variants_to_exclude.drop_duplicates(
            ignore_index=True)
        variant_ids_to_exclude = pd.merge(variants_to_exclude,
                                          variants,
                                          validate="1:1")["id"]
        variants = variants[~variants["id"].isin(variant_ids_to_exclude)]
        if not skip_sanity_checks:
            try:
                assert total_variants - len(variants) == len(
                    variants_to_exclude)
            except:
                import ipdb
                ipdb.set_trace()

    logging.info(f"Dropped {total_variants - len(variants)} variants")
    logging.info(f"...done ({time.time() - start_time} s)")

    logging.info("Processing samples")
    samples = set(pd.read_csv(samples, header=None).loc[:, 0])
    if exclude_samples is not None:
        total_samples = len(samples)

        if sample_exclusion_files := list(Path(exclude_samples).glob("*.csv")):
            samples_to_exclude = set(
                pd.concat(
                    [
                        pd.read_csv(s, header=None).loc[:, 0]
                        for s in sample_exclusion_files
                    ],
                    ignore_index=True,
                ))
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
        if exclude_calls is not None:
            chrom_dir = os.path.join(exclude_calls, chrom)
            exclude_calls_chrom = list(Path(chrom_dir).glob("*.tsv*"))
        else:
            calls_to_exclude = pd.DataFrame(
                columns=["chrom", "pos", "ref", "alt", "sample"])

        logging.info("Processing sparse GT files")

        chrom_dir = os.path.join(sparse_gt, chrom)
        logging.info(f"chrom dir is {chrom_dir}")

        variants_chrom = variant_groups.get_group(chrom)

        sparse_gt_chrom = sorted(list(Path(chrom_dir).glob("*.tsv*")))
        logging.info(
            f"{len(sparse_gt_chrom)} sparse GT files: {[str(s) for s in sparse_gt_chrom]}"
        )

        variant_ids = [np.array([], dtype=np.int32) for _ in samples]
        genotypes = [np.array([], dtype=np.int8) for _ in samples]
        for sparse_gt_file in sparse_gt_chrom:
            # load corresponding calls_to_exclude
            file_stem = sparse_gt_file
            while str(file_stem).find(".") != -1:
                file_stem = Path(file_stem.stem)

            calls_to_exclude = None
            if exclude_calls is not None:
                exclude_call_files = [f for f in exclude_calls_chrom if f.name.startswith(str(file_stem))]
                if len(exclude_call_files) > 0:
                    calls_to_exclude = pd.concat(
                        [
                            pd.read_csv(
                                c,
                                names=["chrom", "pos", "ref", "alt", "sample"],
                                sep="\t",
                                index_col=None,
                            ) for c in tqdm(exclude_call_files, desc="Filtered calls")
                        ],
                        ignore_index=True,
                    )

                    calls_dropped = len(calls_to_exclude)
                    total_calls_dropped += calls_dropped
                    logging.info(f"Dropped {calls_dropped} calls")

            # process_sparse_gt_file
            this_variant_ids, this_genotypes = process_sparse_gt_file(sparse_gt_file.as_posix(), variants_chrom,
                                            samples, calls_to_exclude)
            assert len(this_variant_ids) == len(samples)
            assert len(this_genotypes) == len(samples)

            # concatenate to existing results
            for i, (v, g) in enumerate(zip(this_variant_ids, this_genotypes)):
                variant_ids[i] = np.append(variant_ids[i], v)
                genotypes[i] = np.append(genotypes[i], g)

        del calls_to_exclude
        gc.collect()

        logging.info("Ordering gt and variant matrices")
        order = [np.argsort(i) for i in variant_ids]

        variant_ids = [
            variant_ids[i][order[i]]
            for i in range(len(variant_ids))
        ]
        genotypes = [
            genotypes[i][order[i]] for i in range(len(genotypes))
        ]

        gc.collect()

        logging.info("Preparing GT arrays for storage")

        logger.info("  Padding ragged matrix")
        variant_matrix = ragged_to_matrix(variant_ids)
        gt_matrix = ragged_to_matrix(genotypes)

        del variant_ids
        del genotypes
        gc.collect()

        count_variants = (gt_matrix >= 0).sum(axis=1)

        out_file_chrom = f"{out_file}_{chrom}.h5"
        Path(out_file_chrom).parents[0].mkdir(exist_ok=True, parents=True)
        logging.info(f"Writing to {out_file_chrom}")
        with h5py.File(out_file_chrom, "w") as f:
            sample_array = np.array(samples, dtype="S")
            write_genotype_file(f,
                                sample_array,
                                variant_matrix,
                                gt_matrix,
                                count_variants=count_variants)

        del variant_matrix
        del gt_matrix
        gc.collect()

    logging.info(f"Dropped {total_calls_dropped} calls in total")
    logging.info("Finished!")


@cli.command()
@click.option("--chunksize", type=int)
@click.argument("genotype-files", nargs=-1, type=click.Path(exists=True))
@click.argument("out-file", type=click.Path())
def combine_genotypes(chunksize: Optional[int], genotype_files: List[str],
                      out_file: str):
    with h5py.File(genotype_files[0]) as f:
        samples = f["samples"][:]

    n_samples: int = samples.shape[0]
    # variant_matrix_list: List[List] = [[] for _ in range(n_samples)]
    # genotype_matrix_list: List[List] = [[] for _ in range(n_samples)]

    if chunksize is None:
        chunksize = n_samples

    count_variants = np.zeros(n_samples, dtype=np.int32)
    for file in genotype_files:
        with h5py.File(file) as f:
            count_variants += f["count_variants"][:]

    max_n_variants = int(np.max(count_variants))

    with h5py.File(out_file, "w") as f:
        f.create_dataset("samples", data=samples, dtype=h5py.string_dtype())
        f.create_dataset("variant_matrix", (n_samples, max_n_variants),
                         dtype=np.int32)
        f.create_dataset("genotype_matrix", (n_samples, max_n_variants),
                         dtype=np.int8)

    running_count = np.zeros(n_samples, dtype=np.int32)
    for start_sample in trange(0, n_samples, chunksize, desc="Chunks"):
        end_sample = min(start_sample + chunksize, n_samples)
        chunk_var_matrix = -1 * np.ones(
            (end_sample - start_sample, max_n_variants), dtype=np.int32)
        chunk_gt_matrix = -1 * np.ones(
            (end_sample - start_sample, max_n_variants), dtype=np.int8)

        for file in tqdm(genotype_files, desc="Files"):
            with h5py.File(file) as f:
                var_matrix = f["variant_matrix"][start_sample:end_sample, :]
                gt_matrix = f["genotype_matrix"][start_sample:end_sample, :]
                for i in trange(start_sample, end_sample, desc="Samples"):
                    i_rel = i - start_sample
                    this_var = var_matrix[i_rel, var_matrix[i_rel, :] != -1]
                    this_gt = gt_matrix[i_rel, gt_matrix[i_rel, :] != -1]
                    assert this_var.shape == this_gt.shape

                    n_variants = this_var.shape[0]
                    start_index = running_count[i]
                    end_index = start_index + n_variants
                    assert start_index == 0 or chunk_var_matrix[i_rel,
                                                                start_index -
                                                                1] >= 0
                    if n_variants > 0:
                        assert chunk_var_matrix[i_rel, start_index] == -1
                        chunk_var_matrix[i_rel,
                                         start_index:end_index] = this_var
                        chunk_gt_matrix[i_rel, start_index:end_index] = this_gt
                        running_count[i] += n_variants

        assert np.all(
            np.sum(chunk_var_matrix >= 0, axis=1) ==
            count_variants[start_sample:end_sample])

        # ragged_variant_matrix = [
        #     np.concatenate(vm) for vm in tqdm(variant_matrix_list)
        # ]
        # del variant_matrix_list
        # gc.collect()
        # ragged_genotype_matrix = [
        #     np.concatenate(gt) for gt in tqdm(genotype_matrix_list)
        # ]
        # del genotype_matrix_list
        # gc.collect()
        # variant_matrix = ragged_to_matrix(ragged_variant_matrix)
        # del ragged_variant_matrix
        # gc.collect()
        # genotype_matrix = ragged_to_matrix(ragged_genotype_matrix)
        # del ragged_genotype_matrix
        # gc.collect()
        with h5py.File(out_file, "r+") as f:
            f["variant_matrix"][start_sample:end_sample, :] = chunk_var_matrix
            f["genotype_matrix"][start_sample:end_sample, :] = chunk_gt_matrix


if __name__ == "__main__":
    cli()

import gc
import logging
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import List, Optional, Tuple

import click
import h5py
import numpy as np
import pandas as pd
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


def ragged_to_matrix(
    rows: List[np.ndarray], pad_value: int = -1, max_len: Optional[int] = None
) -> np.ndarray:
    if max_len is None:
        max_len = max([r.shape[0] for r in rows])
    matrix = np.stack(
        [
            np.pad(
                r,
                (0, max_len - r.shape[0]),
                "constant",
                constant_values=(pad_value, pad_value),
            )
            for r in tqdm(rows, desc="Ragged to matrix", file=sys.stdout)
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

    if calls_to_exclude is not None:
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
    count_variants: Optional[np.ndarray] = None,
):
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
@click.option("--chromosomes", type=str)
def add_variant_ids(variant_file: str, out_file: str, duplicates_file: str, chromosomes: Optional[str]=None,
):
    variants = pd.read_csv(
        variant_file, sep="\t", names=["chrom", "pos", "ref", "alt"], index_col=False
    )

    if chromosomes is not None:
        chromosomes = [f"chr{chrom}" for chrom in chromosomes.split(",")]
        variants = variants[variants["chrom"].isin(chromosomes)]

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


def parse_file_path_list(file_path_list_path: Path):
    with open(file_path_list_path) as file:
        vcf_files = [Path(line.rstrip()) for line in file]
        vcf_stems = [vf.stem.replace(".vcf", "") for vf in vcf_files]

        assert len(vcf_stems) == len(vcf_files)

        vcf_look_up = {stem: file for stem, file in zip(vcf_stems, vcf_files)}

        return vcf_stems, vcf_files, vcf_look_up


@cli.command()
@click.option("--threshold", type=float, default=0.1)
@click.argument("file-paths-list", type=click.Path(exists=True))
@click.argument("imiss-dir", type=click.Path(exists=True))
@click.argument("out-file", type=click.Path())
def process_individual_missingness(
    threshold: float, file_paths_list: Path, imiss_dir: str, out_file: str
):
    vcf_stems, _, _ = parse_file_path_list(file_paths_list)

    imiss_dir = Path(imiss_dir)

    imiss_blocks = []
    total_variants = 0
    for vcf_stem in tqdm(vcf_stems, desc="VCFs"):
        missing_counts = pd.read_csv(
            imiss_dir / "samples" / f"{vcf_stem}.tsv",
            sep="\t",
            header=None,
            usecols=[1, 11],
        )
        missing_counts.columns = ["sample", "n_missing"]
        imiss_blocks.append(missing_counts)
        total_variants += pd.read_csv(
            imiss_dir / "sites" / f"{vcf_stem}.tsv",
            header=None,
            sep="\t",
        ).iloc[0, 1]

    imiss = pd.concat(imiss_blocks, ignore_index=True)
    sample_groups = imiss.groupby("sample")
    sample_counts = sample_groups.agg(np.sum).reset_index()
    sample_counts["missingness"] = sample_counts["n_missing"] / total_variants
    sample_counts = sample_counts.loc[
        sample_counts["missingness"] >= threshold, ["sample", "missingness"]
    ]
    sample_counts[["sample"]].to_csv(out_file, index=False, header=None)


@cli.command()
@click.option("--chunksize", type=int, default=1000)
@click.option("--exclude-variants", type=click.Path(exists=True), multiple=True)
@click.option("--exclude-samples", type=click.Path(exists=True))
@click.option("--exclude-calls", type=click.Path(exists=True))
@click.option("--chromosomes", type=str)
@click.option("--skip-sanity-checks", is_flag=True)
@click.argument("variant-file", type=click.Path(exists=True))
@click.argument("samples-path", type=click.Path(exists=True))
@click.argument("sparse-gt", type=click.Path(exists=True))
@click.argument("out-file", type=click.Path())
def process_sparse_gt(
    chunksize: int,
    exclude_variants: List[str],
    exclude_samples: Optional[str],
    exclude_calls: Optional[str],
    chromosomes: Optional[str],
    skip_sanity_checks: bool,
    variant_file: str,
    samples_path: str,
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
            v for directory in exclude_variants for v in Path(directory).rglob("*.tsv*")
        ]
        variants_to_exclude = pd.concat(
            [
                pd.read_csv(v, sep="\t", names=["chrom", "pos", "ref", "alt"])
                for v in tqdm(variant_exclusion_files, file=sys.stdout)
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
    samples = set(pd.read_csv(samples_path, header=None).loc[:, 0])
    if exclude_samples is not None:
        total_samples = len(samples)

        if sample_exclusion_files := list(Path(exclude_samples).rglob("*.csv")):

            sample_exclusion_files = [
                s for s in sample_exclusion_files if s.stat().st_size > 0
            ]
            if sample_exclusion_files:
                logging.info(
                    f"Found {len(sample_exclusion_files)} sample exclusion files"
                )
                samples_to_exclude = set(
                    pd.concat(
                        [
                            pd.read_csv(s, header=None).loc[:, 0]
                            for s in sample_exclusion_files
                        ],
                        ignore_index=True,
                    )
                )
            else:
                samples_to_exclude = set()
            samples -= samples_to_exclude
            logging.info(f"Dropped {total_samples - len(samples)} samples")
        else:
            logging.info(f"Found no samples to exclude in {exclude_samples}")

    samples = sorted(list(samples))

    logging.info("Processing sparse GT files by chromosome")
    total_calls_dropped = 0
    variant_groups = variants.groupby("chrom")
    logger.info(f"variant groups are {variant_groups.groups.keys()}")
    for chrom in tqdm(
        variant_groups.groups.keys(), desc="Chromosomes", file=sys.stdout
    ):
        logging.info(f"Processing chromosome {chrom}")
        logging.info("Reading in filtered calls")

        if exclude_calls is not None:
            exclude_calls_chrom = Path(exclude_calls).rglob("*.tsv*")

        logging.info("Processing sparse GT files")

        variants_chrom = variant_groups.get_group(chrom)

        sparse_file_cols = ["chrom", "pos", "ref", "alt", "sample", "gt"]

        sparse_gt_chrom = [
            f
            for f in Path(sparse_gt).rglob("*.tsv*")
            if get_file_chromosome(f, col_names=sparse_file_cols) == chrom
        ]

        logging.info(
            f"{len(sparse_gt_chrom)} sparse GT files: {[str(s) for s in sparse_gt_chrom]}"
        )

        variant_ids = [np.array([], dtype=np.int32) for _ in samples]
        genotypes = [np.array([], dtype=np.int8) for _ in samples]
        for sparse_gt_file in tqdm(sparse_gt_chrom, desc="Files", file=sys.stdout):
            # Load calls to exclude that correspond to current file
            file_stem = sparse_gt_file
            while str(file_stem).find(".") != -1:
                file_stem = Path(file_stem.stem)

            calls_to_exclude = None
            if exclude_calls is not None:
                exclude_call_files = [
                    f for f in exclude_calls_chrom if f.name.startswith(str(file_stem))
                ]
                if len(exclude_call_files) > 0:
                    calls_to_exclude = pd.concat(
                        [
                            pd.read_csv(
                                c,
                                names=["chrom", "pos", "ref", "alt", "sample"],
                                sep="\t",
                                index_col=None,
                            )
                            for c in tqdm(
                                exclude_call_files,
                                desc="Filtered calls",
                                file=sys.stdout,
                            )
                        ],
                        ignore_index=True,
                    )

                    calls_dropped = len(calls_to_exclude)
                    total_calls_dropped += calls_dropped
                    logging.info(f"Dropped {calls_dropped} calls")

            # Load sparse_gt_file as data frame, add variant IDs, remove excluded calls
            this_variant_ids, this_genotypes = process_sparse_gt_file(
                sparse_gt_file.as_posix(), variants_chrom, samples, calls_to_exclude
            )
            assert len(this_variant_ids) == len(samples)
            assert len(this_genotypes) == len(samples)

            # Concatenate to existing results
            for i, (v, g) in enumerate(zip(this_variant_ids, this_genotypes)):
                variant_ids[i] = np.append(variant_ids[i], v)
                genotypes[i] = np.append(genotypes[i], g)

            gc.collect()

        gc.collect()

        out_file_chrom = f"{out_file}_{chrom}.h5"
        Path(out_file_chrom).parents[0].mkdir(exist_ok=True, parents=True)
        logging.info(f"Writing to {out_file_chrom}")
        count_variants = np.array([g.shape[0] for g in genotypes])
        max_len = np.max(count_variants)
        with h5py.File(out_file_chrom, "w") as f:
            sample_array = np.array(samples, dtype="S")
            f.create_dataset("samples", data=sample_array, dtype=h5py.string_dtype())
            f.create_dataset("variant_matrix", (len(samples), max_len), dtype=np.int32)
            f.create_dataset("genotype_matrix", (len(samples), max_len), dtype=np.int8)
            f.create_dataset("count_variants", data=count_variants, dtype=np.int32)

            # Operate in chunks to reduce memory usage
            for start_sample in trange(
                0, len(samples), chunksize, desc="Chunks", file=sys.stdout
            ):
                end_sample = min(start_sample + chunksize, len(samples))

                order = [np.argsort(v) for v in variant_ids[start_sample:end_sample]]

                this_variant_ids = [
                    variant_ids[i][order[i - start_sample]]
                    for i in range(start_sample, end_sample)
                ]
                this_genotypes = [
                    genotypes[i][order[i - start_sample]]
                    for i in range(start_sample, end_sample)
                ]

                gc.collect()

                variant_matrix = ragged_to_matrix(this_variant_ids, max_len=max_len)
                gt_matrix = ragged_to_matrix(this_genotypes, max_len=max_len)

                f["variant_matrix"][start_sample:end_sample] = variant_matrix
                f["genotype_matrix"][start_sample:end_sample] = gt_matrix

                del variant_matrix
                del gt_matrix
                gc.collect()

    logging.info(f"Dropped {total_calls_dropped} calls in total")
    logging.info("Finished!")


@cli.command()
@click.option("--chunksize", type=int)
@click.argument("genotype-files", nargs=-1, type=click.Path(exists=True))
@click.argument("out-file", type=click.Path())
def combine_genotypes(
    chunksize: Optional[int], genotype_files: List[str], out_file: str
):
    with h5py.File(genotype_files[0]) as f:
        samples = f["samples"][:]

    for gt_file in genotype_files:
        with h5py.File(gt_file) as f:
            if not np.array_equal(samples, f["samples"][:]):
                raise ValueError(
                    f"Error when processing {gt_file}: "
                    "All genotype files must contain the same samples in the same order"
                )

    n_samples: int = samples.shape[0]

    if chunksize is None:
        chunksize = n_samples

    count_variants = np.zeros(n_samples, dtype=np.int32)
    for file in genotype_files:
        with h5py.File(file) as f:
            count_variants += f["count_variants"][:]

    max_n_variants = int(np.max(count_variants))

    with h5py.File(out_file, "w") as f:
        f.create_dataset("samples", data=samples, dtype=h5py.string_dtype())
        f.create_dataset("variant_matrix", (n_samples, max_n_variants), dtype=np.int32)
        f.create_dataset("genotype_matrix", (n_samples, max_n_variants), dtype=np.int8)

    with h5py.File(out_file, "r+") as g:
        for start_sample in trange(
            0, n_samples, chunksize, desc="Chunks", file=sys.stdout
        ):
            end_sample = min(start_sample + chunksize, n_samples)
            this_chunksize = end_sample - start_sample

            # Use ExitStack so we don't have to reopen every HDF5 file for each sample
            variant_matrix_list = [[] for _ in range(this_chunksize)]
            genotype_matrix_list = [[] for _ in range(this_chunksize)]
            with ExitStack() as stack:
                genotype_file_objs = [
                    stack.enter_context(h5py.File(g)) for g in genotype_files
                ]
                for f in tqdm(genotype_file_objs, desc="Files", file=sys.stdout):
                    var_matrix = f["variant_matrix"][start_sample:end_sample, :]
                    gt_matrix = f["genotype_matrix"][start_sample:end_sample, :]
                    for i in trange(this_chunksize, desc="Samples", file=sys.stdout):
                        this_var = var_matrix[i, var_matrix[i, :] != -1]
                        this_gt = gt_matrix[i, gt_matrix[i, :] != -1]
                        assert this_var.shape == this_gt.shape

                        variant_matrix_list[i].append(this_var)
                        genotype_matrix_list[i].append(this_gt)

            ragged_variant_matrix = [np.concatenate(vm) for vm in variant_matrix_list]
            del variant_matrix_list
            gc.collect()
            ragged_genotype_matrix = [np.concatenate(gt) for gt in genotype_matrix_list]
            del genotype_matrix_list
            gc.collect()
            variant_matrix = ragged_to_matrix(
                ragged_variant_matrix, max_len=max_n_variants
            )
            del ragged_variant_matrix
            gc.collect()
            genotype_matrix = ragged_to_matrix(
                ragged_genotype_matrix, max_len=max_n_variants
            )
            del ragged_genotype_matrix
            gc.collect()

            g["variant_matrix"][start_sample:end_sample, :] = variant_matrix
            g["genotype_matrix"][start_sample:end_sample, :] = genotype_matrix


if __name__ == "__main__":
    cli()

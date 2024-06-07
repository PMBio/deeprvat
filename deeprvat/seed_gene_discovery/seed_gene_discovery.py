import logging
import pickle
import sys
import os
import re
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import copy


import click
import numpy as np
import pandas as pd
import yaml
from scipy.stats import beta
from scipy.sparse import coo_matrix, spmatrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from deeprvat.data import DenseGTDataset
from seak import scoretest

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

variant_weight_th_dict = {"sift_polyphen": 0.8}


class GotNone(Exception):
    pass


def replace_in_array(arr, old_val, new_val):
    return np.where(arr == old_val, new_val, arr)


def get_caf(G):
    # get the cumulative allele frequency
    ac = G.sum(axis=0)  # allele count of each variant
    af = ac / (G.shape[0] * 2)  # allele frequency of each variant
    caf = af.sum()
    return caf


#     return mask
def save_burdens(GW_list, GW_full_list, split, chunk, out_dir):
    burdens_path = Path(f"{out_dir}/burdens")
    burdens_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing burdens for split {split} to {burdens_path}")

    with open(
        f"{burdens_path}/burdens_agg_{split}_chunk{chunk}.pickle", "wb"
    ) as handle:
        pickle.dump(GW_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(
        f"{burdens_path}/burdens_all_{split}_chunk{chunk}.pickle", "wb"
    ) as handle:
        pickle.dump(GW_full_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def subset_matrix(M: Any, train_proportion: int):
    n_discovery_samples = round(M.shape[0] * train_proportion)
    if isinstance(M, spmatrix):
        M = M.tocsr()
    M_discovery = M[:n_discovery_samples, :]
    M_val = M[n_discovery_samples:, :]
    return M_discovery, M_val


# copied from seak package
def center(X, inplace=False):
    """Mean centers genotype values, excluding missing values from computation.

    :param numpy.ndarray X: 2D array with dimensions :math:`n*m` with :math:`n:=` number of individuals and :math:`m:=` number of SNVs.
    :return: mean-centered array
    :rtype: numpy.ndarray
    """
    mean = np.nanmean(X, axis=0)  # computes mean for each of the variants

    if inplace:
        X -= mean
        return X
    else:
        return X - mean


def collapse_burden(X, method="max"):
    """Collapses burdens with specified method.

    :param numpy.ndarray X: 2D array with dimensions :math:`n*m` with :math:`n:=` number of individuals and :math:`m:=` number of SNVs with non-zero weight.
    :param string method: collapsing method
    :return: collapsed array :math:`n*1`
    :rtype: numpy.ndarray
    """
    if method == "max":
        X_collapse = np.max(X, axis=1)
    elif method == "binary":
        X_collapse = np.sum(X, axis=1)
        X_collapse[X_collapse > 0] = 1
    elif method == "sum":
        X_collapse = np.sum(X, axis=1)
    else:
        raise NotImplementedError(
            "Collapsing mehthod has to be on of max, binary or sum!"
        )
    X_collapse = X_collapse[:, np.newaxis]

    return X_collapse


# start: kernel definition functions
def get_weights(variant_ids, annotation_df, weight_cols, var_weight_function, maf_col):
    anno = annotation_df.loc[variant_ids, :]
    logger.info(f"total number of variants: {len(anno)}")
    logger.info(f"Using {var_weight_function} for variant weighting")
    if var_weight_function == "sift_polyphen":
        weight = calculate_sift_polyphen_weights(anno)
    elif var_weight_function == "beta_maf":
        weight = calculate_beta_maf_weights(anno, maf_col)
    else:
        raise NotImplementedError(
            f"var_weight_function {var_weight_function} not implemented"
        )

    if "is_plof" not in anno.columns:
        is_plof = pd.Series([False] * len(anno), index=anno.index)
        for c in (
            "splice_acceptor_variant",
            "splice_donor_variant",
            "frameshift_variant",
            "stop_gained",
            "stop_lost",
            "start_lost",
        ):
            is_plof |= anno[f"Consequence_{c}"] > 0
    else:
        is_plof = anno["is_plof"]
    is_plof_mask = np.array(is_plof, dtype=bool)

    logger.info(f"Number of plof variants: {np.sum(is_plof_mask)}")

    # SIFT_Polyphen Scheme consistent with Monti et al. implementation
    logger.info("Setting pLOF variant weights to 1")
    weight[is_plof_mask] = 1

    return weight, is_plof_mask


def calculate_beta_maf_weights(anno, maf_col, beta_weights=(1, 25)):
    weight = beta.pdf(anno[maf_col], beta_weights[0], beta_weights[1])

    return weight


def calculate_sift_polyphen_weights(anno):
    weight = (1 - anno["sift_score"] + anno["polyphen_score"]) / 2

    return weight.to_numpy()


# end: kernel parameters / functions


# set up the annotation and weight matrices
def get_anno(
    G: np.ndarray,
    variant_ids: np.ndarray,
    annotation_df: pd.DataFrame,
    weight_cols: List[str],
    var_weight_function: str,
    maf_col: str,
):
    assert np.all(np.isfinite(G))

    weights, plof_mask = get_weights(
        variant_ids, annotation_df, weight_cols, var_weight_function, maf_col
    )

    return weights, plof_mask


def call_score(GV, null_model_score, pval_dict, test_type):
    # score test
    # p-value for the score-test
    start_time = time.time()
    pv = null_model_score.pv_alt_model(GV)
    end_time = time.time()
    time_diff = end_time - start_time
    pval_dict["time"] = time_diff
    logger.info(f"p-value: {pv}")
    if pv < 0.0:
        logger.warning(
            f"Negative value encountered in p-value computation "
            f"p-value: {pv}, using saddle instead."
        )
        # get test time
        start_time = time.time()
        pv = null_model_score.pv_alt_model(GV, method="saddle")
        end_time = time.time()
        time_diff = end_time - start_time
        pval_dict["time"] = time_diff
    pval_dict["pval"] = pv
    if pv < 1e-3 and test_type == "burden":
        logger.info("Computing regression coefficient")
        # if gene is quite significant get the regression coefficient + SE
        # only works for quantitative traits
        try:
            beta = null_model_score.coef(GV)
            logger.info(f"Regression coefficient: {beta}")
            pval_dict["beta"] = beta["beta"][0, 0]
            pval_dict["betaSd"] = np.sqrt(beta["var_beta"][0, 0])
        except:
            pval_dict["beta"] = None
            pval_dict["betaSd"] = None
    return pval_dict


# set up the test-function for a single gene
def test_gene(
    G_full: spmatrix,
    gene: int,
    grouped_annotations: pd.DataFrame,
    Y,
    weight_cols: List[str],
    null_model_score: scoretest.ScoretestNoK,
    test_config: Dict,
    var_type,
    test_type,
    maf_col,
    min_mac,
) -> Dict[str, Any]:
    # Find variants present in gene
    # Convert sparse genotype to CSC
    # Slice genotype and annotation matrices by variants present in gene
    annotation_df = grouped_annotations.get_group(gene)
    variant_ids = annotation_df.index.unique().to_numpy()
    logger.info(f"variant_ids len: {len(variant_ids)}")

    ## sparse Compressed Sparse Column
    G = G_full.tocsc()[:, variant_ids].todense()
    # Important
    # Important: cast G into numpy array. Otherwise it will be a matrix and
    # the * operator does matrix mutiplication (.dot()) instead of scalar multiplication (.multiply())
    G = np.asarray(G)
    logger.info(f"G shape and sum {G.shape, G.sum()}")
    # GET expected allele count (EAC) as in Karczewski et al. 2022/Genebass
    vars_per_sample = np.sum(G, axis=1)
    samples_with_variant = vars_per_sample[vars_per_sample > 0].shape[0]
    if len(np.unique(Y)) == 2:
        n_cases = (Y > 0).sum()
    else:
        n_cases = Y.shape[0]
    EAC = get_caf(G) * n_cases

    pval_dict = {}

    pval_dict["EAC"] = EAC
    pval_dict["n_cases"] = n_cases
    pval_dict["gene"] = gene
    pval_dict["pval"] = np.nan
    pval_dict["EAC_filtered"] = np.nan
    pval_dict["n_QV"] = np.nan
    pval_dict["n_cluster"] = np.nan
    pval_dict["max_var_count"] = np.nan
    pval_dict["max_var_count_wo_homoz"] = np.nan
    pval_dict["time"] = np.nan

    var_weight_function = test_config.get("var_weight_function", "sift_polyphen")
    max_n_markers = test_config.get("max_n_markers", 5000)
    # skips genes with more than max_n_markers qualifying variants

    logger.info(f"Using function {var_weight_function} for variant weighting")

    (
        weights,
        _,
    ) = get_anno(
        G, variant_ids, annotation_df, weight_cols, var_weight_function, maf_col
    )
    variant_weight_th = (
        variant_weight_th_dict[var_weight_function]
        if var_weight_function in list(variant_weight_th_dict.keys())
        else 0
    )
    logger.info(f"Setting variant weight threshold to {variant_weight_th}")
    logger.info(
        f"Number of variants before thresholding: {len(np.where(weights >= 0)[0])}"
    )
    pos = np.where(weights >= variant_weight_th)[0]
    logger.info(
        f"Number of variants after thresholding using threshold {variant_weight_th}: {len(pos)}"
    )
    pval_dict["n_QV"] = len(pos)
    pval_dict["markers_after_mac_collapsing"] = len(pos)
    if (len(pos) > 0) & (len(pos) < max_n_markers):
        G_f = G[:, pos]
        EAC_filtered = EAC = get_caf(G_f) * n_cases
        pval_dict["EAC_filtered"] = EAC_filtered
        MAC = G_f.sum(axis=0)
        count = G_f[G_f == 2].shape[0]

        # confirm that variants we include are rare variants
        count_with_homoz = G_f.sum(axis=0).max()
        count_without_homoz = replace_in_array(G_f, 2, 1).sum(axis=0).max()
        pval_dict["max_var_count"] = count_with_homoz
        pval_dict["max_var_count_wo_homoz"] = count_without_homoz
        if test_config["neglect_homozygous"] & count > 0:
            logger.info("Replacing homozygous counts with 1")
            np.place(G_f, G_f == 2, [1])

        if test_config["center_genotype"]:
            # mean center genotype
            G_f = center(G_f)
        weights_f = np.sqrt(
            weights[pos]
        )  # Monti et al also always use the square root of the weight

        logger.info("Multiplying genotypes with weights")
        GW = G_f.dot(np.diag(weights_f, k=0))

        GW_full = np.copy(GW)  # only required to record the burdens
        pval_dict["n_cluster"] = GW.shape[1]

        ### COLLAPSE kernel if doing burden test
        collapse_ultra_rare = True
        if test_type == "skat":
            logger.info("Running Skat test")
            if collapse_ultra_rare:
                logger.info(f"Max Collapsing variants with MAC <= {min_mac}")
                MAC_mask = MAC <= min_mac
                if MAC_mask.sum() > 0:
                    logger.info(f"Number of collapsed positions: {MAC_mask.sum()}")
                    GW_collapse = copy.deepcopy(GW)
                    GW_collapse = GW_collapse[:, MAC_mask].max(axis=1).reshape(-1, 1)
                    GW = GW[:, ~MAC_mask]
                    GW = np.hstack((GW_collapse, GW))
                    logger.info(f"GW shape {GW.shape}")
                else:
                    logger.info(
                        f"No ultra rare variants to collapse ({MAC_mask.sum()})"
                    )
                    GW = GW
            else:
                GW = GW

        pval_dict["markers_after_mac_collapsing"] = GW.shape[1]
        if test_type == "burden":
            collapse_method = test_config.get("collapse_method", "binary")
            logger.info(f"Running burden test with collapsing method {collapse_method}")
            GW = collapse_burden(GW, method=collapse_method)

        call_score(GW, null_model_score, pval_dict, test_type)

    else:
        GW = GW_full = np.zeros(G.shape[0])

    return pval_dict, GW, GW_full


def run_association_(
    Y: np.ndarray,
    X: np.ndarray,
    gene_ids,
    G_full,
    grouped_annotations: pd.DataFrame,
    dataset: DenseGTDataset,
    config: Dict[str, Any],
    var_type: str,
    test_type: str,
    persist_burdens: bool,
) -> pd.DataFrame:
    # initialize the null models
    # ScoretestNoK automatically adds a bias column if not present
    if len(np.unique(Y)) == 2:
        print("Fitting binary model since only found two distinct y values")
        null_model_score = scoretest.ScoretestLogit(Y, X)
    else:
        null_model_score = scoretest.ScoretestNoK(Y, X)
    stats = []
    GW_list = {}
    GW_full_list = {}
    time_list_inner = {}
    weight_cols = config.get("weight_cols", [])
    logger.info(f"Testing with this config: {config['test_config']}")
    min_mac = config["test_config"].get("min_mac", 0)
    # Get column with minor allele frequency
    annotations = config["data"]["dataset_config"]["annotations"]
    maf_col = [
        annotation
        for annotation in annotations
        if re.search(r"_AF|_MAF|^MAF", annotation)
    ]
    assert len(maf_col) == 1
    maf_col = maf_col[0]
    logger.info(f"Variant allele frequency column used for weighting: {maf_col}")
    for gene in tqdm(gene_ids, file=sys.stdout):
        try:
            logger.info(f"Testing for Gene{gene}.")
            gene_stats, GW, GW_full = test_gene(
                G_full,
                gene,
                grouped_annotations,
                Y,
                weight_cols,
                null_model_score,
                config["test_config"],
                var_type,
                test_type,
                maf_col,
                min_mac,
            )
            if persist_burdens:
                GW_list[gene] = GW
                GW_full_list[gene] = GW_full
        except GotNone:
            continue
        time_list_inner[gene] = gene_stats["time"]
        del gene_stats["time"]
        stats.append(gene_stats)

    # generate results table:
    stats = pd.DataFrame.from_dict(stats).set_index("gene")

    stats = stats.reset_index().rename(columns={"index": "gene"})
    logger.info(f"Total number of genes: {len(stats)}")
    logger.info(
        f'Number of genes with computed p-value: {len(stats[stats["pval"].notna()])}'
    )

    logger.info(stats.head())

    return stats, GW_list, GW_full_list, time_list_inner


@click.group()
def cli():
    pass


def _add_annotation_cols(annotations, config):
    logger.info(f"Adding annotations {annotations} to config")
    for anno in annotations:
        config["data"]["dataset_config"]["annotations"].append(anno)
        config["data"]["dataset_config"]["rare_embedding"]["config"][
            "annotations"
        ].append(anno)

    return config


@cli.command()
@click.option("--phenotype", type=str)
@click.option("--variant-type", type=str)
@click.option("--rare-maf", type=float)
@click.option("--maf-column", type=str, default="MAF")
@click.option("--simulated-phenotype-file", type=str)
@click.argument("old_config_file", type=click.Path(exists=True))
@click.argument("new_config_file", type=click.Path())
def update_config(
    old_config_file: str,
    phenotype: Optional[str],
    simulated_phenotype_file: str,
    variant_type: Optional[str],
    rare_maf: Optional[float],
    maf_column: str,
    new_config_file: str,
):
    with open(old_config_file) as f:
        config = yaml.safe_load(f)
    if simulated_phenotype_file is not None:
        logger.info("Using simulated phenotype file")
        config["data"]["dataset_config"][
            "sim_phenotype_file"
        ] = simulated_phenotype_file
    logger.info(f"Reading MAF column from column {maf_column}")

    if phenotype is not None:
        config["data"]["dataset_config"]["y_phenotypes"] = [phenotype]
    try:
        rare_threshold_config = config["data"]["dataset_config"]["rare_embedding"][
            "config"
        ]["thresholds"]
    except:
        config["data"]["dataset_config"]["rare_embedding"]["config"]["thresholds"] = {}
        rare_threshold_config = config["data"]["dataset_config"]["rare_embedding"][
            "config"
        ]["thresholds"]

    if variant_type is not None:
        logger.info(f"Variant type is {variant_type}")
        if variant_type == "missense":
            rare_threshold_config["Consequence_missense_variant"] = (
                "Consequence_missense_variant == 1"
            )
        elif variant_type == "plof":
            rare_threshold_config["is_plof"] = "is_plof == 1"
        elif variant_type == "all":
            logger.info("not filtering variant types")
        else:
            raise NotImplementedError(
                f"Variant type is {variant_type} but can either be missense or plof"
            )
        logger.info(config["data"]["dataset_config"]["annotations"])
        logger.info(
            config["data"]["dataset_config"]["rare_embedding"]["config"]["annotations"]
        )
    else:
        logger.info("No variant type specified. Setting type to plof")
        rare_threshold_config["is_plof"] = "is_plof == 1"

    if rare_maf is not None:
        logger.info(f"setting association testing maf to {rare_maf}")
        config["data"]["dataset_config"]["min_common_af"][maf_column] = rare_maf
        rare_threshold_config[maf_column] = (
            f"{maf_column} < {rare_maf} and {maf_column} > 0"
        )

    logger.info(f"Rare variant thresholds: {rare_threshold_config}")
    with open(new_config_file, "w") as f:
        yaml.dump(config, f)


def make_dataset_(
    config: Dict, pickled_dataset_file: str = None, debug: bool = False, data_key="data"
) -> Dataset:
    data_config = config[data_key]

    if pickled_dataset_file is not None and os.path.isfile(pickled_dataset_file):
        logger.info("Loading pickled dataset")
        with open(pickled_dataset_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        logger.info("Instantiating dataset")

        dataset = DenseGTDataset(
            gt_file=data_config["gt_file"],
            skip_y_na=True,
            skip_x_na=True,
            variant_file=data_config["variant_file"],
            **data_config["dataset_config"],
        )
        logger.info("Writing pickled data set")
        with open(pickled_dataset_file, "wb") as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if debug:
        logger.info("Debug mode: Using only 1000 samples")
        batch_size = 1000
    else:
        logger.info(f"Setting batch size to length of dataset")
        batch_size = len(dataset)

    logger.info(f"Read dataset, batch size {data_config['dataloader_config']}")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        **data_config["dataloader_config"],
    )

    logger.info("Loading data")
    data_full = next(iter(dataloader))

    logger.info("Data succesfully loaded. Data set generation completed.")

    return dataset, data_full


@cli.command()
@click.option("--debug", is_flag=True)
@click.option("--data-key", type=str, default="data")
@click.option("--pickled-dataset-file", type=str, default=None)
@click.argument("config-file", type=click.Path(exists=True))
@click.argument("out-file", type=click.Path())
def make_dataset(
    debug: bool,
    data_key: str,
    config_file: str,
    pickled_dataset_file: str,
    out_file: str,
):
    with open(config_file) as f:
        config = yaml.safe_load(f)

    _, ds = make_dataset_(
        config,
        debug=debug,
        data_key=data_key,
        pickled_dataset_file=pickled_dataset_file,
    )

    with open(out_file, "wb") as f:
        pickle.dump(ds, f)


@cli.command()
@click.option("--debug", is_flag=True)
@click.option("--n-chunks", type=int)
@click.option("--chunk", type=int)
@click.option("--dataset-file", type=click.Path(exists=True))
@click.option("--data-file", type=click.Path(exists=True))  # dataset_full
@click.option("--persist-burdens", is_flag=True)
@click.argument("config-file", type=click.Path(exists=True))
@click.argument("var-type", type=str)
@click.argument("test-type", type=str)
@click.argument("out-path", type=click.Path())
def run_association(
    debug: bool,
    dataset_file: Optional[DenseGTDataset],
    data_file: Optional[DenseGTDataset],
    config_file: str,
    var_type: str,
    test_type: str,
    out_path: str,
    persist_burdens: bool,
    n_chunks: Optional[int] = None,
    chunk: Optional[int] = None,
):
    logger.info(f"Saving burdens: {persist_burdens}")

    if test_type not in ["skat", "burden"]:
        raise NotImplementedError(f"Test type {test_type} is invalid/not implemented")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    if data_file is not None:
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
        with open(data_file, "rb") as f:
            data_full = pickle.load(f)
    else:
        dataset, data_full = make_dataset_(config, debug=debug)

    this_data_idx = [i for i in range(len(data_full["sample"]))]

    G_full = data_full["rare_variant_annotations"]
    all_variants = np.unique(G_full.col)  # SparseGenotype
    # all_variants = np.unique(G_full)       #PaddedAnnotations
    if isinstance(G_full, spmatrix):
        G_full = G_full.tocsr()
    G_full = G_full[this_data_idx]
    logger.info(f"all_variants shape: {all_variants.shape}")

    X = data_full["x_phenotypes"].numpy()[this_data_idx]
    logger.info(f"X shape: {X.shape}")
    # Don't add bias columns since
    # ScoretestNoK automatically adds a bias column if not present

    Y = data_full["y"].numpy()[this_data_idx]

    logger.info(f"X shape: {X.shape}")
    logger.info(f"Y shape: {Y.shape}")

    logger.info("Grouping variants by gene")
    exploded_annotations = (
        dataset.annotation_df.query("id in @all_variants")
        .explode("gene_id")
        .reset_index()
        .drop_duplicates()
        .set_index("id")
    )
    grouped_annotations = exploded_annotations.groupby("gene_id")
    gene_ids = pd.read_parquet(dataset.gene_file, columns=["id"])["id"].to_list()
    gene_ids = list(
        set(gene_ids).intersection(set(exploded_annotations["gene_id"].unique()))
    )

    logger.info(f"Number of genes to test: {len(gene_ids)}")
    # Grouped annotations also contains non-coding gene ids

    if debug:
        logger.info("Debug mode: Using only 100 genes")
        gene_ids = gene_ids[:100]

    n_total_genes = len(gene_ids)
    logger.info("Reading variant file")
    logger.info(f"Genotype matrix shape: {G_full.shape}")

    ### TODO: read this from config
    logger.info("Training split: Running tests for each gene")

    if chunk is not None:
        if n_chunks is None:
            raise ValueError("n_chunks must be specified if chunk is not None")

        chunk_length = math.floor(n_total_genes / n_chunks)
        chunk_start = chunk * chunk_length
        chunk_end = min(n_total_genes, chunk_start + chunk_length)
        if chunk == n_chunks - 1:
            chunk_end = n_total_genes
    else:
        n_genes = n_total_genes
        chunk_start = 0
        chunk_end = n_genes

    genes = range(chunk_start, chunk_end)
    n_genes = len(genes)
    if n_genes == 0:
        logger.info(
            f"Number of chunks is too large. The pipeline will throw an error beacause there are no genes to test"
        )
    logger.info(f"Processing genes in {genes} from {n_total_genes} in total")
    this_gene_ids = [gene_ids[i] for i in genes]

    discovery_stats, GW_list, GW_full_list, time_list = run_association_(
        Y,
        X,
        this_gene_ids,
        G_full,
        grouped_annotations,
        dataset,
        config,
        var_type,
        test_type,
        persist_burdens,
    )

    if persist_burdens:
        out_dir = os.path.dirname(out_path)
        save_burdens(GW_list, GW_full_list, "testing", chunk, out_dir)
    logger.info("Writing timings")
    out_dir = os.path.dirname(out_path)
    with open(f"{out_dir}/timing_chunk_{chunk}.pickle", "wb") as f:
        pickle.dump(time_list, f)
    logger.info(f"Saving results to {out_path}")
    discovery_stats.to_parquet(out_path, engine="pyarrow")


@cli.command()
@click.argument("result-files", type=click.Path(exists=True), nargs=-1)
@click.argument("out-file", type=click.Path())
def combine_results(result_files: Tuple[str], out_file: str):
    logger.info(f"Concatenating results to {out_file}")
    if "/results/" in out_file:
        out_file_eval = Path(out_file.replace("/results/", "/eval/"))
    else:
        out_file_eval = out_file

    res_df = pd.concat([pd.read_parquet(f, engine="pyarrow") for f in result_files])
    logger.info(f"Writing combined results to {out_file}")
    res_df.to_parquet(out_file, engine="pyarrow")

    logger.info("Doing simple postprocessing of results")

    cols_to_keep = ["gene", "EAC", "pval"]

    if "EAC" in res_df.columns:
        logger.info("Filtering for genes with EAC > 50")
        res_df = res_df.query("EAC > 50")

    res_df = res_df.dropna(subset=["pval"])[cols_to_keep]
    logger.info(f"Writing filtered results to {out_file_eval}")
    out_file_eval.parent.mkdir(exist_ok=True, parents=True)
    res_df.to_parquet(out_file_eval)


if __name__ == "__main__":
    cli()

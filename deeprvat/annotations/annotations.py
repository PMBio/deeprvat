import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Optional
import dask.dataframe as dd
import numpy as np
import click
import keras.backend as K
import pandas as pd
import pybedtools
import tensorflow as tf
from joblib import Parallel, delayed
from keras.models import load_model
from sklearn.decomposition import PCA
from tqdm import tqdm, trange
from fastparquet import ParquetFile
import yaml


def precision(y_true, y_pred):
    """
    Calculate precision, a metric for the accuracy of the positive predictions.

    Precision is defined as the the fraction of relevant instances among the retrieved instances.

    Parameters:
    - y_true (Tensor): The true labels (ground truth).
    - y_pred (Tensor): The predicted labels.

    Returns:
    float: Precision value.

    Notes:
    - This function uses the Keras backend functions to perform calculations.
    - Precision is calculated as `true_positives / (predicted_positives + epsilon)`, where epsilon is a small constant to avoid division by zero.


    References:
    - https://en.wikipedia.org/wiki/Precision_and_recall

    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """
    Calculate recall, a metric for the ability to capture true positive instances.

    Recall is defined as the fraction of relevant instances that were retrieved.

    Parameters:
    - y_true (Tensor): The true labels (ground truth).
    - y_pred (Tensor): The predicted labels.

    Returns:
    - float: Recall value.

    Notes:
    - This function uses the Keras backend functions to perform calculations.
    - Recall is calculated as `true_positives / (possible_positives + epsilon)`, where epsilon is a small constant to avoid division by zero.


    References:
    - https://en.wikipedia.org/wiki/Precision_and_recall
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def deepripe_get_model_info(saved_models_dict, saved_deepripe_models_path):
    """
    Retrieve information about the paths and names of saved deepRiPe models.

    Parameters:
    - saved_models_dict (dict): A dictionary containing keys for different types of models. Keys include "parclip" for PAR-CLIP models, "eclip_hg2" for eCLIP models in HepG2, and "eclip_k5" for eCLIP models in K562. Values are model identifiers.
    - saved_deepripe_models_path (str): The path to the directory where the deepRiPe models are saved.

    Returns:
    tuple: A tuple containing two dictionaries.
    The first dictionary contains paths for each type of model, with keys
    "parclip", "eclip_hg2", and "eclip_k5" and values as lists of paths corresponding to high,
    medium, and low sequence models.
    The second dictionary contains lists of RBP names for each type of model, with keys
    "parclip", "eclip_hg2", and "eclip_k5" and values as lists of RBP names for high, medium, and
    low sequence models.

    Notes:
    - The function constructs file paths based on the provided model identifiers.
    - The resulting dictionary structure allows easy access to model paths for different types.


    """
    shared_path = Path(saved_deepripe_models_path)

    # parclip
    path_to_m_high = shared_path / f'{saved_models_dict["parclip"]}_high_seq.h5'
    path_to_m_med = shared_path / f'{saved_models_dict["parclip"]}_med_seq.h5'
    path_to_m_low = shared_path / f'{saved_models_dict["parclip"]}_low_seq.h5'

    # eclip HepG2
    path_to_hg_high1 = shared_path / f'{saved_models_dict["eclip_hg2"]}_high1_seq.h5'
    path_to_hg_high2 = shared_path / f'{saved_models_dict["eclip_hg2"]}_high2_seq.h5'
    path_to_hg_mid1 = shared_path / f'{saved_models_dict["eclip_hg2"]}_mid1_seq.h5'
    path_to_hg_mid2 = shared_path / f'{saved_models_dict["eclip_hg2"]}_mid2_seq.h5'
    path_to_hg_low = shared_path / f'{saved_models_dict["eclip_hg2"]}_low_seq.h5'

    # eclip K562
    path_to_k5_high1 = shared_path / f'{saved_models_dict["eclip_k5"]}_high1_seq.h5'
    path_to_k5_high2 = shared_path / f'{saved_models_dict["eclip_k5"]}_high2_seq.h5'
    path_to_k5_mid1 = shared_path / f'{saved_models_dict["eclip_k5"]}_mid1_seq.h5'
    path_to_k5_mid2 = shared_path / f'{saved_models_dict["eclip_k5"]}_mid2_seq.h5'
    path_to_k5_low = shared_path / f'{saved_models_dict["eclip_k5"]}_low_seq.h5'

    saved_paths = {
        "parclip": [path_to_m_high, path_to_m_med, path_to_m_low],
        "eclip_hg2": [
            path_to_hg_high1,
            path_to_hg_high2,
            path_to_hg_mid1,
            path_to_hg_mid2,
            path_to_hg_low,
        ],
        "eclip_k5": [
            path_to_k5_high1,
            path_to_k5_high2,
            path_to_k5_mid1,
            path_to_k5_mid2,
            path_to_k5_low,
        ],
    }

    ### parclip
    pc_RBPnames_low = np.array(
        [
            "MBNL1",
            "P53_NONO",
            "PUM2",
            "QKI",
            "AGO3",
            "FUS",
            "TAF15",
            "ZFP36",
            "DICER1",
            "EIF3A",
            "EIF3D",
            "EIF3G",
            "SSB",
            "PAPD5",
            "CPSF4",
            "CPSF3",
            "RTCB",
            "FXR1",
            "NOP58",
            "NOP56",
            "FBL",
            "LIN28A",
            "LIN28B",
            "UPF1",
            "G35",
            "G45",
            "XPO5",
        ]
    )  # 27

    pc_RBPnames_med = np.array(
        [
            "TARDBP",
            "ELAVL2",
            "ELAVL3",
            "ELAVL4",
            "RBM20",
            "IGF2BP1",
            "IGF2BP2",
            "IGF2BP3",
            "EWSR1",
            "HNRNPD",
            "RBPMS",
            "SRRM4",
            "AGO2",
            "NUDT21",
            "FIP1L1",
            "CAPRIN1",
            "FMR1iso7",
            "FXR2",
            "AGO1",
            "L1RE1",
            "ORF1",
        ]
    )

    pc_RBPnames_high = np.array(
        [
            "DND1",
            "CPSF7",
            "CPSF6",
            "CPSF1",
            "CSTF2",
            "CSTF2T",
            "ZC3H7B",
            "FMR1iso1",
            "RBM10",
            "MOV10",
            "ELAVL1",
        ]
    )

    ### eclip HepG2
    hg2_RBPnames_high1 = np.array(
        [
            "DDX3X",
            "PCBP2",
            "FAM120A",
            "HNRNPL",
            "RBFOX2",
            "PTBP1",
            "MATR3",
            "EFTUD2",
            "PRPF4",
            "UPF1",
        ]
    )

    hg2_RBPnames_high2 = np.array(
        [
            "GRWD1",
            "PRPF8",
            "PPIG",
            "CSTF2T",
            "QKI",
            "U2AF2",
            "SUGP2",
            "HNRNPM",
            "AQR",
            "BCLAF1",
        ]
    )

    hg2_RBPnames_mid1 = np.array(
        [
            "LSM11",
            "NKRF",
            "SUB1",
            "NCBP2",
            "UCHL5",
            "LIN28B",
            "IGF2BP3",
            "SF3A3",
            "AGGF1",
            "DROSHA",
            "DDX59",
            "CSTF2",
            "DKC1",
            "EIF3H",
            "FUBP3",
            "SFPQ",
            "HNRNPC",
            "ILF3",
            "TIAL1",
            "HLTF",
            "ZNF800",
            "PABPN1",
            "YBX3",
            "FXR2",
        ]
    )

    hg2_RBPnames_mid2 = np.array(
        [
            "GTF2F1",
            "IGF2BP1",
            "HNRNPK",
            "XPO5",
            "RPS3",
            "SF3B4",
            "LARP4",
            "BUD13",
            "SND1",
            "G3BP1",
            "AKAP1",
            "KHSRP",
        ]
    )

    hg2_RBPnames_low = np.array(
        [
            "RBM22",
            "GRSF1",
            "CDC40",
            "NOLC1",
            "FKBP4",
            "DGCR8",
            "ZC3H11A",
            "XRN2",
            "SLTM",
            "DDX55",
            "TIA1",
            "SRSF1",
            "U2AF1",
            "RBM15",
        ]
    )

    ### eclip K562
    k562_RBPnames_high1 = np.array(["BUD13", "PTBP1", "DDX24", "EWSR1", "RBM15"])

    k562_RBPnames_high2 = np.array(
        [
            "SF3B4",
            "YBX3",
            "UCHL5",
            "KHSRP",
            "ZNF622",
            "NONO",
            "EXOSC5",
            "PRPF8",
            "CSTF2T",
            "AQR",
            "UPF1",
        ]
    )

    k562_RBPnames_mid1 = np.array(
        [
            "U2AF2",
            "AKAP8L",
            "METAP2",
            "SMNDC1",
            "GEMIN5",
            "HNRNPK",
            "SLTM",
            "SRSF1",
            "FMR1",
            "SAFB2",
            "DROSHA",
            "RPS3",
            "IGF2BP2",
            "ILF3",
            "RBFOX2",
            "QKI",
            "PCBP1",
            "ZNF800",
            "PUM1",
        ]
    )

    k562_RBPnames_mid2 = np.array(
        [
            "EFTUD2",
            "LIN28B",
            "AGGF1",
            "HNRNPL",
            "SND1",
            "GTF2F1",
            "EIF4G2",
            "TIA1",
            "TARDBP",
            "FXR2",
            "HNRNPM",
            "IGF2BP1",
            "PUM2",
            "FAM120A",
            "DDX3X",
            "MATR3",
            "FUS",
            "GRWD1",
            "PABPC4",
        ]
    )

    k562_RBPnames_low = np.array(
        [
            "MTPAP",
            "RBM22",
            "DHX30",
            "DDX6",
            "DDX55",
            "TRA2A",
            "XRN2",
            "U2AF1",
            "LSM11",
            "ZC3H11A",
            "NOLC1",
            "KHDRBS1",
            "GPKOW",
            "DGCR8",
            "AKAP1",
            "FXR1",
            "DDX52",
            "AATF",
        ]
    )

    saved_RBP_names = {
        "parclip": [pc_RBPnames_high, pc_RBPnames_med, pc_RBPnames_low],
        "eclip_hg2": [
            hg2_RBPnames_high1,
            hg2_RBPnames_high2,
            hg2_RBPnames_mid1,
            hg2_RBPnames_mid2,
            hg2_RBPnames_low,
        ],
        "eclip_k5": [
            k562_RBPnames_high1,
            k562_RBPnames_high2,
            k562_RBPnames_mid1,
            k562_RBPnames_mid2,
            k562_RBPnames_low,
        ],
    }

    return saved_paths, saved_RBP_names


def seq_to_1hot(seq, randomsel=True):
    """
    Convert a nucleotide sequence to one-hot encoding.

    Parameters:
    - seq (str): The input nucleotide sequence.
    - randomsel (bool): If True, treat ambiguous base as random base.
    If False, return only zero rows for ambiguous case.

    Returns:
    numpy.ndarray: A 2D array representing the one-hot encoding of the input sequence.
    Rows correspond to nucleotides 'A', 'C', 'G', 'T' in that order.
    Columns correspond to positions in the input sequence.

    Notes:
    - Ambiguous bases are handled based on the 'randomsel' parameter.



    References:
    - one-hot encoding: https://en.wikipedia.org/wiki/One-hot
    """

    seq_len = len(seq)
    seq = seq.upper()
    seq_code = np.zeros((4, seq_len), dtype="int")
    for i in range(seq_len):
        nt = seq[i]
        if nt == "A":
            seq_code[0, i] = 1
        elif nt == "C":
            seq_code[1, i] = 1
        elif nt == "G":
            seq_code[2, i] = 1
        elif nt == "T":
            seq_code[3, i] = 1
        elif randomsel:
            rn = random.randint(0, 3)
            seq_code[rn, i] = 1
    return seq_code


def convert2bed(variants_file, output_dir):
    """
    Convert a variants file to BED format.

    Parameters:
    - variants_file (str): The path to the input variants file.
    - output_dir (str): The directory where the BED file will be saved.

    Returns:
    None

    Notes:
    - The input variants file should be in tab-separated format with columns: "#CHROM", "POS", "ID", "REF", "ALT".
    - The generated BED file will have columns: "CHR", "Start", "End", "ID", "VAR", "Strand".
    - The "Start" and "End" columns are set to the "POS" values, and "Strand" is set to '.' for all entries.
    """
    file_name = variants_file.split("/")[-1]
    print(f"Generating BED file: {output_dir}/{file_name[:-3]}bed")

    df_variants = pd.read_csv(
        variants_file, sep="\t", names=["#CHROM", "POS", "ID", "REF", "ALT"]
    )

    print(df_variants.head())

    df_bed = pd.DataFrame()
    df_bed["CHR"] = df_variants["#CHROM"].astype(str)
    df_bed["Start"] = df_variants["POS"].astype(str)
    df_bed["End"] = df_variants["POS"].astype(str)
    df_bed["ID"] = df_variants["ID"].astype(str)
    df_bed["VAR"] = df_variants.apply(lambda x: f'{x["REF"]}/{x["ALT"]}', axis=1)
    df_bed["Strand"] = ["." for _ in range(len(df_variants))]

    df_bed.to_csv(
        f"{output_dir}/{file_name[:-3]}bed", sep="\t", index=False, header=None
    )


def deepripe_encode_variant_bedline(bedline, genomefasta, flank_size=75):
    """
    Encode a variant bedline into one-hot encoded sequences.

    Parameters:
    - bedline (list): A list representing a variant bedline, containing elements for chromosome, start position, end position, reference allele, alternate allele, and strand.
    - genomefasta (str): The path to the genome FASTA file for sequence retrieval.
    - flank_size (int): The size of flanking regions to include in the sequence around the variant position.

    Returns:
    numpy.ndarray: A 3D array representing one-hot encoded sequences. The dimensions are (num_sequences, sequence_length, nucleotide_channels).

    Notes:
    - The input bedline should follow the format: [chromosome, start position, end position, reference allele, alternate allele, strand].
    - The function retrieves the wild-type and mutant sequences flanked by the specified size.
    - The wild-type sequence is extracted from the genome FASTA file and mutated at the variant position.
    - The resulting sequences are one-hot encoded and returned as a numpy array.

    References:
    - pybedtools.BedTool: https://daler.github.io/pybedtools/main.html
    - FATSA format: https://en.wikipedia.org/wiki/FASTA_format
    """
    mut_a = bedline[4].split("/")[1]
    strand = bedline[5]
    if len(mut_a) == 1:
        wild = pybedtools.BedTool(
            bedline[0]
            + "\t"
            + str(int(bedline[1]) - flank_size)
            + "\t"
            + str(int(bedline[2]) + flank_size)
            + "\t"
            + bedline[3]
            + "\t"
            + str(mut_a)
            + "\t"
            + bedline[5],
            from_string=True,
        )
        if strand == "-":
            mut_pos = flank_size
        else:
            mut_pos = flank_size - 1

        wild = wild.sequence(fi=genomefasta, tab=True, s=True)
        fastalist = open(wild.seqfn).read().split("\n")
        del fastalist[-1]
        seqs = [fasta.split("\t")[1] for fasta in fastalist]
        mut = seqs[0]
        mut = list(mut)
        mut[mut_pos] = mut_a
        mut = "".join(mut)
        seqs.append(mut)
        encoded_seqs = np.array([seq_to_1hot(seq) for seq in seqs])
        encoded_seqs = np.transpose(encoded_seqs, axes=(0, 2, 1))

        return encoded_seqs


def readYamlColumns(annotation_columns_yaml_file):
    with open(annotation_columns_yaml_file, "r") as fd:
        config = yaml.safe_load(fd)
    columns = config["annotation_column_names"]
    prior_names = list(columns.keys())
    post_names = [list(columns[k].keys())[0] for k in columns]
    fill_vals = [list(columns[k].values())[0] for k in columns]
    column_name_mapping = dict(zip(prior_names, post_names))
    fill_value_mapping = dict(zip(post_names, fill_vals))
    return prior_names, post_names, fill_vals, column_name_mapping, fill_value_mapping


def get_parquet_columns(parquet_file):
    pfile = ParquetFile(parquet_file)
    pcols = pfile.columns
    return pcols


@click.group()
def cli():
    pass


@cli.command()
@click.argument("anno_path", type=click.Path(exists=True))
@click.argument("gtf_path", type=click.Path(exists=True))
@click.argument("genes_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(exists=False))
@click.option("--max_dist", type=int, default=300)
def filter_annotations_by_exon_distance(
    anno_path: str, gtf_path: str, genes_path: str, output_path: str, max_dist: int
) -> None:
    """
    Filters annotation based on distance to the nearest exon of gene it is associated with.

    Args:
        anno_path (str): Annotation parquet file containing variant annotations to filter.
        gtf_path (str): GTF file containing start and end positions of all relevant exons of all relevant genes. DataFrame is filtered for protein coding exons.
        genes_path (str): List of protein coding genes and their IDs in the annotation DataFrame.
        output_path (str): Where to write the resulting parquet file.
        max_dist (int): Base pairs used to filter.

    Returns:
        None

    Writes:
        Parquet file containing filtered annotations.
    """
    import pyranges as pr

    logger.info("read gtf file as pandas df")
    gtf = pr.read_gtf(gtf_path)
    gtf = gtf.as_df()

    logger.info("filter gtf for protein coding exons from the HAVANA DB")
    gtf = gtf.query(
        "Source == 'HAVANA' and Feature == 'exon' and gene_type == 'protein_coding' and transcript_type == 'protein_coding'"
    )

    logger.info("split gene ID column on '.'")
    gtf[["gene_base", "feature"]] = gtf["gene_id"].str.split(".", expand=True)

    logger.info(" read protein_coding_genes")
    pcg = pd.read_parquet(genes_path, columns=["gene", "id"])

    logger.info(" only select necessary columns, rename to fit gtf file")
    gtf = gtf[["gene_id", "Start", "End"]].rename(columns={"gene_id": "gene"})

    logger.info(" add gene ids to gtf file")

    gtf = gtf.merge(pcg, on="gene")

    logger.info(" only select necessary columns, rename to fit gtf file")
    gtf = gtf[["Start", "End", "id"]].rename(columns={"id": "gene_id"})

    logger.info("reading annotations to filter ")
    anno_df = pd.read_parquet(anno_path)
    anno_df = anno_df[["id", "pos", "gene_id"]]

    logger.info("adding exons to annotations (1:M merge)")

    merged = anno_df.merge(gtf, how="left", on="gene_id")
    del anno_df

    logger.info(
        "adding positons of start and end of each exon relative to variant position to df"
    )
    merged["start_diff"] = merged["Start"] - merged["pos"]
    merged["end_diff"] = merged["End"] - merged["pos"]

    logger.info(
        f"filtering all rows that are further than {max_dist}bp away from each exon "
    )
    len_bf_filtering = len(merged)
    filtered_merge = merged.query(
        "(start_diff <= 0 & end_diff >= 0) | abs(start_diff) <= @max_dist | abs(end_diff) <= @max_dist"
    )
    del merged
    len_after_filtering = len(filtered_merge)
    logger.info(
        f"filtered rows by exon distance ({max_dist}bp), dropped({len_bf_filtering - len_after_filtering} rows / {np.round(100*(len_bf_filtering - len_after_filtering)/len_bf_filtering)}%)"
    )

    logger.info("select necessary columns, drop duplicates")
    filtered_merge = filtered_merge[["id", "gene_id"]]
    filtered_merge = filtered_merge.drop_duplicates()
    logger.info(
        f"dropped dublicates in data frame (dropped {len_after_filtering - len(filtered_merge)}rows/ {np.round(100*(len_after_filtering - len(filtered_merge))/len_after_filtering)}%)."
    )

    logger.info("Reading in annotations for filtering")
    anno_df = pd.read_parquet(anno_path)
    len_anno = len(anno_df)
    filtered = anno_df.merge(filtered_merge, on=["id", "gene_id"], how="left")

    logger.info(
        f"filtered annotations based on filterd id, gene_id (dropped {len(anno_df) - len(filtered)} / {np.round(100*(len(anno_df)-len(filtered))/len(anno_df))}% of rows)."
    )
    logger.info("performing sanity check")
    assert len(filtered == len_anno)
    logger.info(f"writing result to {output_path}")
    filtered.to_parquet(output_path)


@cli.command()
@click.argument("deepsea-file", type=click.Path(exists=True))
@click.argument("pca-object", type=click.Path())
@click.argument("means_sd_df", type=click.Path())
@click.argument("out-dir", type=click.Path(exists=True))
@click.option("--n-components", type=int, default=100)
def deepsea_pca(
    deepsea_file: str,
    pca_object: str,
    means_sd_df: str,
    out_dir: str,
    n_components: int,
):
    """
    Perform Principal Component Analysis (PCA) on DeepSEA data and save the results.

    Parameters:
    - n_components (int): Number of principal components to retain, default is 100.
    - deepsea_file (str): Path to the DeepSEA data in parquet format.
    - pca_object (str): Path to save or load the PCA object (components) in npy or pickle format.
    - means_sd_df (str): Path to a DataFrame containing pre-calculated means and SDs for standardization. If path does not exist, standardization will be done using the calculated mean and SD, result will then be saved under this path
    - out_dir (str): Path to the output directory where the PCA results will be saved.

    Returns:
    None

    Raises:
    AssertionError: If there are NaN values in the PCA results DataFrame.

    Notes:
    - If 'means_sd_df' is provided, the data will be standardized using the existing mean and SD. Otherwise, the data will be standardized using the mean and SD calculated from the data.
    - If 'pca_object' exists, it will be loaded as a PCA object. If it doesn't exist, a new PCA object will be created, and its components will be saved to 'pca_object'.

    Example:
    $ python annotations.py deepsea_pca --n-components 50 deepsea_data.parquet pca_components.npy means_sd.parquet results/
    """
    logger.info("Loading deepSea data")
    df = pd.read_parquet(deepsea_file)
    logger.info("filling NAs")
    df = df.fillna(0)
    logger.info("Extracting matrix for PCA")
    key_df = df[["#CHROM", "POS", "REF", "ALT"]].reset_index(drop=True)
    logger.info("transforming values to numpy")
    deepSEAcols = [c for c in df.columns if c.startswith("DeepSEA")]
    X = df[deepSEAcols].to_numpy()
    del df
    logger.info(
        "checking wether input contains data frame with pre-calculated means and SDs"
    )
    if os.path.exists(means_sd_df):
        logger.info("standardizing values using existing mean and SD")
        means_sd_data = pd.read_parquet(means_sd_df)

        means = means_sd_data["means"].to_numpy()
        sds = means_sd_data["SDs"].to_numpy()
        del means_sd_data
        X_std = (X - means) / sds
        del means
        del sds

    else:
        logger.info("standardizing values")
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        means_sd_data = pd.DataFrame(
            {
                "names": deepSEAcols,
                "means": np.mean(X, axis=0),
                "SDs": np.std(X, axis=0),
            }
        )
        means_sd_data.to_parquet(means_sd_df)
    del X

    out_path = Path(out_dir)

    if os.path.exists(pca_object):
        if ".pkl" in pca_object:
            with open(pca_object, "rb") as pickle_file:
                logger.info("loading pca objectas pickle file")
                pca = pickle.load(pickle_file)
                X_pca = pca.transform(X_std)
        else:
            if ".npy" not in pca_object:
                logger.error("did not recognize file format, assuming npy")
            logger.info("loading pca components as npy object")
            components = np.load(pca_object)
            logger.info(f"Projecting data to {components.shape[0]} PCs")
            X_pca = np.matmul(X_std, components.transpose())
    else:
        logger.info(f"creating pca object and saving it to {pca_object}")
        logger.info(f"Projecting rows to {n_components} PCs")
        pca = PCA(n_components=n_components)
        pca.fit(X_std)
        np.save(pca_object, pca.components_)

        X_pca = np.matmul(X_std, pca.components_.transpose())

    del X_std

    logger.info(f"Writing values to data frame")
    pca_df = pd.DataFrame(
        X_pca, columns=[f"DeepSEA_PC_{i}" for i in range(1, n_components + 1)]
    )
    del X_pca
    logger.info(f"adding key values to data frame")
    pca_df = pd.concat([key_df, pca_df], axis=1)

    logger.info("Sanity check of results")
    assert pca_df.isna().sum().sum() == 0

    logger.info(f"Writing results to path { out_path / 'deepsea_pca.parquet' }")
    pca_df.to_parquet(out_path / "deepsea_pca.parquet", engine="pyarrow")

    logger.info("Done")


@cli.command()
@click.argument("variants-file", type=click.Path(exists=True))
@click.argument("output-dir", type=click.Path(exists=True))
@click.argument("genomefasta", type=click.Path(exists=True))
@click.argument("pybedtools_tmp_dir", type=click.Path(exists=True))
@click.argument("saved_deepripe_models_path", type=click.Path(exists=True))
@click.argument("n_jobs", type=int)
@click.argument("saved-model-type", type=str)
def scorevariants_deepripe(
    variants_file: str,
    output_dir: str,
    genomefasta: str,
    pybedtools_tmp_dir: str,
    saved_deepripe_models_path: str,
    n_jobs: int,
    saved_model_type: str = "parclip",
):
    """
    Score variants using deep learning models trained on PAR-CLIP and eCLIP data.

    Parameters:
    - variants_file (str): Path to the file containing variant information to be annotated.
    - output_dir (str): Path to the output directory where the results will be saved.
    - genomefasta (str): Path to the reference genome in FASTA format.
    - pybedtools_tmp_dir (str): Path to the temporary directory for pybedtools.
    - saved_deepripe_models_path (str): Path to the directory containing saved deepRiPe models.
    - n_jobs (int): Number of parallel jobs for scoring variants.
    - saved_model_type (str, optional): Type of the saved deepRiPe model to use (parclip, eclip_hg2, eclip_k5). Default is "parclip".

    Returns:
    None

    Raises:
    AssertionError: If there are NaN values in the generated DataFrame.

    Notes:
    - This function scores variants using deepRiPe models trained on different CLIP-seq datasets.
    - The results are saved as a CSV file in the specified output directory.

    Example:
    $ python annotations.py scorevariants_deepripe variants.csv results/ reference.fasta tmp_dir/ saved_models/ 8 eclip_k5
    """
    file_name = variants_file.split("/")[-1]
    bed_file = f"{output_dir}/{file_name[:-3]}bed"

    ### setting pybedtools tmp dir
    parent_dir = pybedtools_tmp_dir
    file_stripped = file_name.split(".")[0]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    tmp_path = os.path.join(parent_dir, f"tmp_{file_stripped}_{timestr}")
    os.mkdir(tmp_path)
    pybedtools.set_tempdir(tmp_path)

    ### reading variants to score
    df_variants = pd.read_csv(
        variants_file,
        sep="\t",
        header=None,
        names=["chr", "pos", "Uploaded_variant", "ref", "alt"],
    )

    if not os.path.exists(bed_file):
        convert2bed(variants_file, output_dir)

    variant_bed = pybedtools.BedTool(bed_file)
    print(f"Scoring variants for: {bed_file}")

    ### paths for experiments
    saved_models_dict = {
        "parclip": "parclip_model",
        "eclip_hg2": "eclip_model_encodeHepG2",
        "eclip_k5": "eclip_model_encodeK562",
    }

    saved_paths, saved_RBP_names = deepripe_get_model_info(
        saved_models_dict, saved_deepripe_models_path
    )

    ## Experiment. parclip
    parclip_RBPs = saved_RBP_names["parclip"]
    parclip_models = [
        load_model(path_i, custom_objects={"precision": precision, "recall": recall})
        for path_i in saved_paths["parclip"]
    ]

    parclip_model_group = {
        keyw: (parclip_models[i], parclip_RBPs[i])
        for i, keyw in enumerate(["high", "med", "low"])
    }

    ## Experiment. eclip HepG2 cell line
    model_opts = ["high1", "high2", "mid1", "mid2", "low"]

    eclip_HG_RBPs = saved_RBP_names["eclip_hg2"]
    eclip_HG_models = [
        load_model(path_i, custom_objects={"precision": precision, "recall": recall})
        for path_i in saved_paths["eclip_hg2"]
    ]

    eclip_HG_model_group = {
        keyw: (eclip_HG_models[i], eclip_HG_RBPs[i])
        for i, keyw in enumerate(model_opts)
    }

    ## Experiment. eclip K562 cell line
    eclip_K5_RBPs = saved_RBP_names["eclip_k5"]
    eclip_K5_models = [
        load_model(path_i, custom_objects={"precision": precision, "recall": recall})
        for path_i in saved_paths["eclip_k5"]
    ]

    eclip_K5_model_group = {
        keyw: (eclip_K5_models[i], eclip_K5_RBPs[i])
        for i, keyw in enumerate(model_opts)
    }

    model_seq_len = {"parclip": 150, "eclip_hg2": 200, "eclip_k5": 200}
    list_of_model_groups = {
        "parclip": parclip_model_group,
        "eclip_hg2": eclip_HG_model_group,
        "eclip_k5": eclip_K5_model_group,
    }

    ## using only sequence
    current_model_type = list_of_model_groups[saved_model_type]
    predictions = deepripe_score_variant_onlyseq_all(
        current_model_type,
        variant_bed,
        genomefasta,
        seq_len=model_seq_len[saved_model_type],
        n_jobs=n_jobs,
    )

    for choice in current_model_type.keys():
        print(choice)
        _, RBPnames = current_model_type[choice]
        score_list = predictions[choice]
        score_list = np.asarray(score_list)
        print(f"Output size: {score_list.shape}")

        ### write predictions to df
        for ix, RBP_name in enumerate(RBPnames):
            df_variants[RBP_name] = score_list[:, ix]
    print(
        f"saving file to: {output_dir}/{file_name[:-3]}{saved_model_type}_deepripe.csv.gz"
    )
    df_variants.to_csv(
        f"{output_dir}/{file_name[:-3]}{saved_model_type}_deepripe.csv.gz", index=False
    )


def process_chunk(
    chrom_file,
    abs_splice_res_dir,
    tissues_to_exclude,
    tissue_agg_function,
    ca_shortened,
):
    """
    Process a chunk of data from absplice site results and merge it with the remaining annotation data.

    Parameters:
    - chrom_file (str): The filename for the chunk of absplice site results.
    - abs_splice_res_dir (Path): The directory containing the absplice site results.
    - tissues_to_exclude (list): List of tissues to exclude from the absplice site results.
    - tissue_agg_function (str): The aggregation function to use for tissue-specific AbSplice scores.
    - ca_shortened (DataFrame): The remaining annotation data to merge with the absplice site results.

    Returns:
    DataFrame: Merged DataFrame containing aggregated tissue-specific AbSplice scores and remaining annotation data.

    Notes:
    - The function reads the absplice site results for a specific chromosome, excludes specified tissues, and aggregates AbSplice scores using the specified tissue aggregation function.
    - The resulting DataFrame is merged with the remaining annotation data based on the chromosome, position, reference allele, alternative allele, and gene ID.

    Example:
    merged_data = process_chunk("chr1_results.csv", Path("abs_splice_results/"), ["Brain", "Heart"], "max", ca_shortened_df)
    """
    logger.info(f"Reading file {chrom_file}")

    ab_splice_res = pd.read_csv(
        abs_splice_res_dir / chrom_file, engine="pyarrow"
    ).reset_index()

    ab_splice_res = ab_splice_res.query("tissue not in @tissues_to_exclude")
    logger.info(
        f"AbSplice tissues excluded: {tissues_to_exclude}, Aggregating AbSplice scores using {tissue_agg_function}"
    )
    logger.info(f"Number of unique variants {len(ab_splice_res['variant'].unique())}")

    #### aggregate tissue specific ab splice scores
    ab_splice_res = (
        ab_splice_res.groupby(["variant", "gene_id"])
        .agg({"AbSplice_DNA": tissue_agg_function})
        .reset_index()
    )

    ab_splice_res[["chrom", "pos", "var"]] = ab_splice_res["variant"].str.split(
        ":", expand=True
    )

    ab_splice_res[["ref", "alt"]] = ab_splice_res["var"].str.split(">", expand=True)

    ab_splice_res["pos"] = ab_splice_res["pos"].astype(int)
    logger.info(f"Number of rows of ab_splice df {len(ab_splice_res)}")
    merged = ab_splice_res.merge(
        ca_shortened, how="left", on=["chrom", "pos", "ref", "alt", "gene_id"]
    )
    logger.info(f"Number of unique variants(id) in merged {len(merged['id'].unique())}")
    logger.info(
        f"Number of unique variants(variant) in merged {len(merged['variant'].unique())}"
    )

    return merged

    del merged
    del ab_splice_res


@cli.command()
@click.argument("current_annotation_file", type=click.Path(exists=True))
@click.argument("abs_splice_res_dir", type=click.Path(exists=True))
@click.argument("absplice_score_file", type=click.Path())
@click.argument("njobs", type=int)
def aggregate_abscores(
    current_annotation_file: str,
    abs_splice_res_dir: str,
    absplice_score_file: str,
    njobs: int,
):
    """
    Aggregate AbSplice scores from AbSplice  results and save the results.

    Parameters:
    - current_annotation_file (str): Path to the current annotation file in parquet format.
    - abs_splice_res_dir (str): Path to the directory containing AbSplice results.
    - absplice_score_file (str): Path to save the aggregated AbSplice scores in parquet format.
    - njobs (int): Number of parallel jobs for processing AbSplice results.

    Returns:
    None

    Notes:
    - The function reads the current annotation file and extracts necessary information for merging.
    - It then processes AbSplice results in parallel chunks, aggregating AbSplice scores.
    - The aggregated scores are saved to the specified file.

    Example:
    $ python annotations.py aggregate_abscores annotations.parquet abs_splice_results/ absplice_scores.parquet 4
    """
    current_annotation_file = Path(current_annotation_file)
    logger.info("reading current annotations file")
    current_annotations = pd.read_parquet(current_annotation_file)

    if "AbSplice_DNA" in current_annotations.columns:
        if "AbSplice_DNA_old" in current_annotations.columns:
            current_annotations.drop("AbSplice_DNA_old", inplace=True)
        current_annotations = current_annotations.rename(
            columns={"AbSplice_DNA": "AbSplice_DNA_old"}
        )
    ca_shortened = current_annotations[["id", "Gene", "chrom", "pos", "ref", "alt"]]
    ca_shortened = ca_shortened.rename(columns={"Gene": "gene_id"})

    logger.info(ca_shortened.columns)

    abs_splice_res_dir = Path(abs_splice_res_dir)

    tissue_agg_function = "max"
    tissues_to_exclude = ["Testis"]
    tissues_to_exclude = []
    ab_splice_agg_score_file = absplice_score_file

    logger.info("creating abSplice score file.. ")
    all_absplice_scores = []
    parallel = Parallel(n_jobs=njobs, return_as="generator", verbose=50)
    output_generator = parallel(
        delayed(process_chunk)(
            i, abs_splice_res_dir, tissues_to_exclude, tissue_agg_function, ca_shortened
        )
        for i in tqdm(os.listdir(abs_splice_res_dir))
    )
    all_absplice_scores = list(output_generator)

    logger.info("concatenating files")
    all_absplice_scores = pd.concat(all_absplice_scores)
    logger.info(f"saving score file to {ab_splice_agg_score_file}")
    all_absplice_scores.to_parquet(ab_splice_agg_score_file)


logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def deepripe_score_variant_onlyseq_all(
    model_group, variant_bed, genomefasta, seq_len=200, batch_size=1024, n_jobs=32
):
    """
    Compute variant scores using a deep learning model for each specified variant.

    Parameters:
        - model_group (dict): A dictionary containing deep learning models for different choices. Each entry should be a key-value pair, where the key is the choice name and the value is a tuple containing the model and additional information.
        - variant_bed (list): A list of variant bedlines, where each bedline represents a variant.
        - genomefasta (str): Path to the reference genome in FASTA format.
        - seq_len (int, optional): The length of the sequence to use around each variant. Default is 200.
        - batch_size (int, optional): Batch size for parallelization. Default is 1024.
        - n_jobs (int, optional): Number of parallel jobs for processing variant bedlines. Default is 32.

    Returns:
        dict: A dictionary containing variant scores for each choice in the model_group.
              Each entry has the choice name as the key and the corresponding scores as the value.
    """
    predictions = {}

    # Parallelize the encoding of variant bedlines
    encoded_seqs_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(deepripe_encode_variant_bedline)(
            bedline, genomefasta, flank_size=(seq_len // 2) + 2
        )
        for bedline in variant_bed
    )

    # Handle cases where encoding is None
    encoded_seqs_list = [
        (x if x is not None else np.ones((2, seq_len + 4, 4)) * float("nan"))
        for x in encoded_seqs_list
    ]

    # Concatenate encoded sequences
    encoded_seqs = tf.concat(encoded_seqs_list, 0)

    logger.info("Computing predictions")

    # Compute predictions for each choice in the model group
    for choice in tqdm(model_group.keys(), desc="Model group"):
        avg_score = 0.0
        for i in range(4):
            cropped_seqs = encoded_seqs[:, i : i + seq_len, :]
            model, _ = model_group[choice]
            pred = model.predict_on_batch(cropped_seqs)
            wild_indices = tf.range(pred.shape[0], delta=2)
            mut_indices = tf.range(1, pred.shape[0], delta=2)
            pred_wild = pred[wild_indices, :]
            pred_mut = pred[mut_indices, :]
            score = pred_mut - pred_wild
            avg_score += score
        predictions[choice] = avg_score / 4

    return predictions


def calculate_scores_max(scores):
    if scores is None:
        return None
    else:
        # Split the string and extract values from index 1 to 5
        values = [float(score) for score in scores.split("|")[1:5] if score != "nan"]
        # Calculate the sum
        if len(values) > 0:
            return np.max(values)
        else:
            return np.NaN


@cli.command()
@click.argument("current_annotation_file", type=click.Path(exists=True))
@click.argument("absplice_score_file", type=click.Path())
@click.argument("out_file", type=click.Path())
def merge_abscores(
    current_annotation_file: str,
    absplice_score_file: str,
    out_file: str,
):
    """
    Merge AbSplice scores with the current annotation file and save the result.

    Parameters:
    - current_annotation_file (str): Path to the current annotation file in parquet format.
    - absplice_score_file (str): Path to the AbSplice scores file in parquet format.
    - out_file (str): Path to save the merged annotation file with AbSplice scores.

    Returns:
    None

    Notes:
    - The function reads AbSplice scores and the current annotation file.
    - It merges the AbSplice scores with the current annotation file based on chromosome, position, reference allele, alternative allele, and gene ID.
    - The merged file is saved with AbSplice scores.

    Example:
    $ python annotations.py merge_abscores current_annotation.parquet absplice_scores.parquet merged_annotations.parquet
    """
    all_absplice_scores = pd.read_parquet(absplice_score_file)

    all_absplice_scores = all_absplice_scores[
        ["chrom", "pos", "ref", "alt", "gene_id", "AbSplice_DNA"]
    ]

    annotations = pd.read_parquet(current_annotation_file, engine="pyarrow").drop(
        columns=["AbSplice_DNA"], errors="ignore"
    )
    annotations = annotations.rename(columns={"Gene": "gene_id"})
    annotations.drop_duplicates(inplace=True, subset=["gene_id", "id"])
    original_len = len(annotations)

    logger.info("Merging")
    merged = pd.merge(
        annotations,
        all_absplice_scores,
        validate="1:1",
        how="left",
        on=["chrom", "pos", "ref", "alt", "gene_id"],
    )

    logger.info("Sanity checking merge")
    assert len(merged) == original_len
    logger.info(
        f"len of merged after dropping duplicates: {len(merged.drop_duplicates(subset=['id', 'gene_id']))}"
    )
    logger.info(f"len of merged without dropping duplicates: {len(merged)}")

    assert len(merged.drop_duplicates(subset=["id", "gene_id"])) == len(merged)

    logger.info(
        f'Filling {merged["AbSplice_DNA"].isna().sum()} '
        "missing AbSplice values with 0"
    )
    merged["AbSplice_DNA"] = merged["AbSplice_DNA"].fillna(0)

    annotation_out_file = out_file

    logger.info(f"Writing to {annotation_out_file}")
    merged.to_parquet(annotation_out_file, engine="pyarrow")


pd.options.mode.chained_assignment = None


@cli.command()
@click.argument("annotation_file", type=click.Path(exists=True))
@click.argument("deepripe_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
@click.argument("column_prefix", type=str)
def merge_deepripe(
    annotation_file: str, deepripe_file: str, out_file: str, column_prefix: str
):
    """
    Merge deepRiPe scores with an annotation file and save the result.

    Parameters:
    - annotation_file (str): Path to the annotation file in parquet format.
    - deepripe_file (str): Path to the deepRiPe scores file in CSV format.
    - out_file (str): Path to save the merged file with deepRiPe scores.
    - column_prefix (str): Prefix to add to the deepRiPe score columns in the merged file.

    Returns:
    None

    Notes:
    - The function reads the annotation file and deepRiPe scores file.
    - It renames the columns in the deepRiPe scores file with the specified prefix.
    - The two dataframes are merged based on chromosome, position, reference allele, alternative allele, and variant ID.
    - The merged file is saved with deepRiPe scores.

    Example:
    $ python annotations.py merge_deepripe annotations.parquet deepripe_scores.csv merged_deepripe.parquet deepripe
    """
    annotations = pd.read_parquet(annotation_file)
    deepripe_df = pd.read_csv(deepripe_file)
    orig_len = len(annotations)
    deepripe_df = deepripe_df.rename(columns={"chr": "chrom"})
    deepripe_df = deepripe_df.drop(
        columns=["Uploaded_variant", "Unnamed: 0"], errors="ignore"
    )
    key_cols = ["chrom", "pos", "ref", "alt", "id"]
    prefix_cols = [x for x in deepripe_df.columns if x not in key_cols]
    new_names = [(i, i + f"_{column_prefix}") for i in prefix_cols]

    deepripe_df = deepripe_df.rename(columns=dict(new_names))
    logger.info(deepripe_df.columns)
    merged = annotations.merge(
        deepripe_df, how="left", on=["chrom", "pos", "ref", "alt", "id"]
    )
    assert len(merged) == orig_len
    merged.to_parquet(out_file)


@cli.command()
@click.argument("annotation_file", type=click.Path(exists=True))
@click.argument("deepripe_pca_file", type=click.Path(exists=True))
@click.argument("column_yaml_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
def merge_deepsea_pcas(
    annotation_file: str, deepripe_pca_file: str, column_yaml_file: str, out_file: str
):
    """
    Merge deepRiPe PCA scores with an annotation file and save the result.

    Parameters:
    - annotation_file (str): Path to the annotation file in parquet format.
    - deepripe_pca_file (str): Path to the deepRiPe PCA scores file in parquet format.
    - column_yaml_file(str): Path to the yaml file containing all needed columns for the model, including their filling values.
    - out_file (str): Path to save the merged file with deepRiPe PCA scores.

    Returns:
    None

    Notes:
    - The function reads the annotation file and deepRiPe PCA scores file.
    - It drops duplicates in both files based on chromosome, position, reference allele, alternative allele, variant ID, and gene ID.
    - The two dataframes are merged based on chromosome, position, reference allele, alternative allele, and variant ID.
    - The merged file is saved with deepRiPe PCA scores.

    Example:
    $ python annotations.py merge_deepsea_pcas annotations.parquet deepripe_pca_scores.parquet merged_deepsea_pcas.parquet
    """

    pcols = get_parquet_columns(deepripe_pca_file)
    anno_cols = get_parquet_columns(annotation_file)
    logger.info("reading current annotations")
    prior_names, *_ = readYamlColumns(column_yaml_file)

    DScommonCols = list(set(prior_names).intersection(set(pcols)))
    AnnoCommonCols = list(set(prior_names).intersection(set(anno_cols)))
    annotations = pd.read_parquet(
        annotation_file,
        columns=AnnoCommonCols + ["chrom", "pos", "ref", "alt", "id", "Gene"],
    )
    logger.info("reading PCAs")
    deepripe_pcas = pd.read_parquet(
        deepripe_pca_file, columns=DScommonCols + ["chrom", "pos", "ref", "alt", "id"]
    )
    deepripe_pcas = deepripe_pcas.drop_duplicates(
        subset=["chrom", "pos", "ref", "alt", "id"]
    )
    orig_len = len(annotations)
    logger.info(f"length of annotation file before merge: {orig_len}")
    annotations = annotations.drop_duplicates(
        subset=["chrom", "pos", "ref", "alt", "id", "Gene"]
    )
    noduplicates_len = len(annotations)
    logger.info(
        f"length of annotation file after dropping duplicates: {noduplicates_len}"
    )
    logger.info("merging")
    merged = annotations.merge(
        deepripe_pcas, how="left", on=["chrom", "pos", "ref", "alt", "id"]
    )

    logger.info(f"length of annotation file after merge: {len(merged)}")
    logger.info("checking lengths")
    assert len(merged) == noduplicates_len
    logger.info(f"writing file to {out_file}")
    merged.to_parquet(out_file)


@cli.command()
@click.argument("in_variants", type=click.Path(exists=True))
@click.argument("out_variants", type=click.Path())
def process_annotations(in_variants: str, out_variants: str):
    """
    Process variant annotations, filter for canonical variants, and aggregate consequences.

    Parameters:
    - in_variants (str): Path to the input variant annotation file in parquet format.
    - out_variants (str): Path to save the processed variant annotation file in parquet format.

    Returns:
    None

    Notes:
    - The function reads the input variant annotation file.
    - It filters for canonical variants where the 'CANONICAL' column is equal to 'YES'.
    - The 'Gene' column is renamed to 'gene_id'.
    - Consequences for different alleles are aggregated by combining the variant ID with the gene ID.
    - The processed variant annotations are saved to the specified output file.

    Example:
    $ python annotations.py process_annotations input_variants.parquet output_variants.parquet
    """
    variant_path = Path(in_variants)
    variants = pd.read_parquet(variant_path)

    logger.info("filtering for canonical variants")

    variants = variants.loc[variants.CANONICAL == "YES"]
    variants.rename(columns={"Gene": "gene_id"}, inplace=True)

    logger.info("aggregating consequences for different alleles")

    # combining variant id with gene id
    variants["censequence_id"] = variants["id"].astype(str) + variants["gene_id"]
    variants.to_parquet(out_variants, compression="zstd")


def process_chunk_addids(chunk: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    """
    Process a chunk of data by adding identifiers from a variants dataframe.

    Parameters:
    - chunk (pd.DataFrame): Chunk of data containing variant information.
    - variants (pd.DataFrame): Dataframe containing variant identifiers.

    Returns:
    pd.DataFrame: Processed chunk with added variant identifiers.

    Raises:
    AssertionError: If the shape of the processed chunk does not match expectations.

    Notes:
    - The function renames columns for compatibility.
    - Drops duplicates in the chunk based on the key columns.
    - Merges the chunk with the variants dataframe based on the key columns.
    - Performs assertions to ensure the shape of the processed chunk meets expectations.

    Example:
    ```python
    chunk = pd.read_csv("chunk_data.csv")
    variants = pd.read_csv("variants_data.csv")
    processed_chunk = process_chunk_addids(chunk, variants)
    ```
    """
    chunk = chunk.rename(
        columns={
            "#CHROM": "chrom",
            "POS": "pos",
            "ID": "variant_name",
            "REF": "ref",
            "ALT": "alt",
            "chr": "chrom",
        }
    )
    key_cols = ["chrom", "pos", "ref", "alt"]

    chunk.drop_duplicates(subset=key_cols, inplace=True)
    chunk_shape = chunk.shape

    chunk = pd.merge(chunk, variants, on=key_cols, how="left", validate="1:1")

    try:
        assert chunk_shape[0] == chunk.shape[0]
    except AssertionError:
        logger.error(
            f"df.shape[0] was {chunk.shape[0]} but chunk_shape[0] was {chunk_shape[0]}"
        )
        raise AssertionError
    try:
        assert chunk.shape[1] == chunk_shape[1] + 1
    except AssertionError:
        logger.error(
            f"chunk.shape[1] was {chunk.shape[1]} but chunk_shape[1] + 1 was {chunk_shape[1] + 1}"
        )
        raise AssertionError
    return chunk


@cli.command()
@click.argument("annotation_file", type=click.Path(exists=True))
@click.argument("variant_file", type=click.Path(exists=True))
@click.argument("njobs", type=int)
@click.argument("out_file", type=click.Path())
def add_ids(annotation_file: str, variant_file: str, njobs: int, out_file: str):
    """
    Add identifiers from a variant file to an annotation file and save the result.

    Parameters:
    - annotation_file (str): Path to the input annotation file in CSV format.
    - variant_file (str): Path to the input variant file in TSV format.
    - njobs (int): Number of parallel jobs to process the data.
    - out_file (str): Path to save the processed data in Parquet format.

    Returns:
    None

    Notes:
    - The function reads the annotation file in chunks and the entire variant file.
    - It uses parallel processing to apply the 'process_chunk_addids' function to each chunk.
    - The result is saved in Parquet format.

    Example:
    $ python annotations.py add_ids annotation_data.csv variant_data.tsv 4 processed_data.parquet
    """

    data = pd.read_csv(annotation_file, chunksize=100_000)

    all_variants = pd.read_csv(variant_file, sep="\t")
    parallel = Parallel(n_jobs=njobs, return_as="generator", verbose=50)

    output_generator = parallel(
        delayed(process_chunk_addids)(chunk, all_variants) for chunk in data
    )
    first = True
    for batch in tqdm(output_generator):
        if first:
            batch.to_parquet(out_file, engine="fastparquet")
        else:
            batch.to_parquet(out_file, engine="fastparquet", append=True)
        first = False


@cli.command()
@click.argument("annotation_file", type=click.Path(exists=True))
@click.argument("variant_file", type=click.Path(exists=True))
@click.argument("njobs", type=int)
@click.argument("out_file", type=click.Path())
def add_ids_dask(annotation_file: str, variant_file: str, njobs: int, out_file: str):
    """
    Add identifiers from a variant file to an annotation file using Dask and save the result.

    Parameters:
    - annotation_file (str): Path to the input annotation file in Parquet format.
    - variant_file (str): Path to the input variant file in Parquet format.
    - njobs (int): Number of parallel jobs to process the data.
    - out_file (str): Path to save the processed data in Parquet format.

    Returns:
    None

    Notes:
    - The function uses Dask to read annotation and variant files with large block sizes.
    - It renames columns for compatibility and drops duplicates based on key columns.
    - Merges the Dask dataframes using the 'merge' function.
    - The result is saved in Parquet format with compression.

    Example:
    $ python annotations.py add_ids_dask annotation_data.parquet variant_data.parquet 4 processed_data.parquet
    """
    data = dd.read_parquet(annotation_file, blocksize=25e9)
    all_variants = pd.read_table(variant_file)
    data = data.rename(
        columns={
            "#CHROM": "chrom",
            "POS": "pos",
            "ID": "variant_name",
            "REF": "ref",
            "ALT": "alt",
            "chr": "chrom",
        }
    )
    key_cols = ["chrom", "pos", "ref", "alt"]
    data.drop_duplicates(subset=key_cols, inplace=True)
    data = dd.merge(data, all_variants, on=key_cols, how="left")
    data.to_parquet(out_file, engine="fastparquet", compression="zstd")


def chunks(lst, n):
    """
    Split a list into chunks of size 'n'.

    Parameters:
    - lst (list): The input list to be split into chunks.
    - n (int): The size of each chunk.

    Yields:
    list: A chunk of the input list.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def read_deepripe_file(f: str):
    """
    Read a DeepRipe file from the specified path.

    Parameters:
    - f (str): Path to the DeepRipe file.

    Returns:
    pd.DataFrame: DataFrame containing the data from the DeepRipe file.

    Example:
    ```python
    file_path = "path/to/deepripe_file.txt"
    deepripe_data = read_deepripe_file(file_path)
    ```
    """
    f = pd.read_table(f, engine="c")
    return f


@cli.command()
@click.argument("deepsea_files", type=str)
@click.argument("out_file", type=click.Path())
@click.argument("njobs", type=int)
def concatenate_deepsea(
    deepsea_files: str,
    out_file: str,
    njobs: int,
):
    """
    Concatenate DeepSEA files based on the provided patterns and chromosome blocks.

    Parameters:
    - deepSEA_name_pattern (str): comma-separated list of deepsea files to concatenate
    - out_file (str): Path to save the concatenated output file in Parquet format.
    - njobs (int): Number of parallel jobs for processing.

    Returns:
    None

    Example:
    $ python annotations.py concatenate_deepSEA chr1_block0.CLI.deepseapredict.diff.tsv,chr1_block1.CLI.deepseapredict.diff.tsv,chr1_block2.CLI.deepseapredict.diff.tsv concatenated_output.parquet 4
    """

    file_paths = deepsea_files.split(",")
    logger.info("check if out_file already exists")
    if os.path.exists(out_file):
        logger.info("file exists, removing existing file")
        os.remove(out_file)
    else:
        logger.info("out_file does not yet exist")

    logger.info("reading in f")

    parallel = Parallel(n_jobs=njobs, backend="loky", return_as="generator")
    chunked_files = list(chunks(file_paths, njobs))
    logger.info(f"processing {len(chunked_files)} files")
    for chunk in tqdm(chunked_files):
        logger.info(f"Chunk consist of {len(chunk)} files")
        this_generator = parallel((delayed(read_deepripe_file)(f) for f in chunk))
        current_file = pd.concat(list(this_generator))
        if chunk == chunked_files[0]:
            logger.info("creating new file")
            current_file.to_parquet(out_file, engine="fastparquet")
        else:
            try:
                current_file.to_parquet(out_file, engine="fastparquet", append=True)
            except ValueError:
                out_df_columns = pd.read_parquet(out_file, engine="fastparquet").columns

                logger.error(
                    f"columns are not equal in saved/appending file: {[i for i in out_df_columns if i not in current_file.columns]} and {[i for i in current_file.columns if i not in out_df_columns]} "
                )

                raise ValueError


@cli.command()
@click.argument("vep_header_line", type=int)
@click.argument("vep_file", type=click.Path(exists=True))
@click.argument("deepripe_parclip_file", type=click.Path(exists=True))
@click.argument("deepripe_hg2_file", type=click.Path(exists=True))
@click.argument("deepripe_k5_file", type=click.Path(exists=True))
@click.argument("variant_file", type=click.Path(exists=True))
@click.argument("vcf_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
@click.option("--vepcols_to_retain", type=str)
def merge_annotations(
    vep_header_line: int,
    vep_file: str,
    deepripe_parclip_file: str,
    deepripe_hg2_file: str,
    deepripe_k5_file: str,
    variant_file: str,
    vcf_file: str,
    out_file: str,
    vepcols_to_retain: Optional[str],
):
    """
    Merge VEP, DeepRipe (parclip, hg2, k5), and variant files into one dataFrame and save result as parquet file

    Parameters:
    - vep_header_line (int): Line number of the header line in the VEP output file.
    - vep_file (str): Path to the VEP file.
    - deepripe_parclip_file (str): Path to the DeepRipe parclip file.
    - deepripe_hg2_file (str): Path to the DeepRipe hg2 file.
    - deepripe_k5_file (str): Path to the DeepRipe k5 file.
    - variant_file (str): Path to the variant file.
    - vcf_file (str): vcf file containing chrom, pos, ref and alt information
    - out_file (str): Path to save the merged output file in Parquet format.
    - vepcols_to_retain (Optional[str]): Comma-separated list of additional VEP columns to retain.

    Returns:
    None

    Example:
    $ python annotations.py merge_annotations 1 vep_file.tsv deepripe_parclip.csv deepripe_hg2.csv deepripe_k5.csv variant_file.tsv merged_output.parquet --vepcols_to_retain="AlphaMissense,PolyPhen"
    """
    # load vep file
    vep_df = pd.read_csv(vep_file, header=vep_header_line, sep="\t", na_values="-")
    if vepcols_to_retain is not None:
        vepcols_to_retain = [c for c in vepcols_to_retain.split(",")]
    vep_df = process_vep(
        vep_file=vep_df, vcf_file=vcf_file, vepcols_to_retain=vepcols_to_retain
    )
    logger.info(f"vep_df shape is {vep_df.shape}")
    logger.info("load deepripe_parclip")

    deepripe_parclip_df = pd.read_csv(deepripe_parclip_file)
    deepripe_parclip_df = process_deepripe(deepripe_parclip_df, "parclip")
    logger.info("load deepripe_k5")

    deepripe_k5_df = pd.read_csv(deepripe_k5_file)
    deepripe_k5_df = process_deepripe(deepripe_k5_df, "k5")
    logger.info("load deepripe_hg2")

    deepripe_hg2_df = pd.read_csv(deepripe_hg2_file)
    deepripe_hg2_df = process_deepripe(deepripe_hg2_df, "hg2")
    logger.info("load variant_file")

    logger.info(f"reading in {variant_file}")
    variants = pd.read_csv(variant_file, sep="\t")

    logger.info("merge vep to variants M:1")
    ca = vep_df.merge(
        variants, how="left", on=["chrom", "pos", "ref", "alt"], validate="m:1"
    )
    del vep_df
    logger.info("merge deepripe files to variants 1:1")
    ca = ca.merge(
        deepripe_parclip_df,
        how="left",
        on=["chrom", "pos", "ref", "alt"],
        validate="m:1",
    )
    ca = ca.merge(
        deepripe_k5_df, how="left", on=["chrom", "pos", "ref", "alt"], validate="m:1"
    )
    ca = ca.merge(
        deepripe_hg2_df, how="left", on=["chrom", "pos", "ref", "alt"], validate="m:1"
    )

    ca.to_parquet(out_file, compression="zstd")


def process_deepripe(deepripe_df: pd.DataFrame, column_prefix: str) -> pd.DataFrame:
    """
    Process the DeepRipe DataFrame, rename columns and drop duplicates.

    Parameters:
    - deepripe_df (pd.DataFrame): DataFrame containing DeepRipe data.
    - column_prefix (str): Prefix to be added to column names.

    Returns:
    pd.DataFrame: Processed DeepRipe DataFrame.

    Example:
    deepripe_df = process_deepripe(deepripe_df, "parclip")
    """
    logger.info("renaming deepripe columns")
    deepripe_df = deepripe_df.rename(columns={"chr": "chrom"})

    deepripe_df = deepripe_df.drop(
        columns=["Uploaded_variant", "Unnamed: 0"], errors="ignore"
    )
    key_cols = ["chrom", "pos", "ref", "alt", "id"]
    prefix_cols = [x for x in deepripe_df.columns if x not in key_cols]
    new_names = [(i, i + f"_{column_prefix}") for i in prefix_cols]
    deepripe_df = deepripe_df.rename(columns=dict(new_names))
    deepripe_df.drop_duplicates(subset=["chrom", "pos", "ref", "alt"], inplace=True)
    return deepripe_df


def process_vep(
    vep_file: pd.DataFrame, vcf_file: str, vepcols_to_retain: list = []
) -> pd.DataFrame:
    """
    Process the VEP DataFrame, extracting relevant columns and handling data types.

    Parameters:
    - vep_file (pd.DataFrame): DataFrame containing VEP data.
    - vepcols_to_retain (list, optional): List of additional columns to retain. Defaults to an empty list.

    Returns:
    pd.DataFrame: Processed VEP DataFrame.

    Example:
    vep_file = process_vep(vep_file, vepcols_to_retain=["additional_col1", "additional_col2"])
    """
    vcf_df = pd.read_table(
        vcf_file, names=["chrom", "pos", "#Uploaded_variation", "ref", "alt"]
    )
    if "#Uploaded_variation" in vep_file.columns:
        vep_file = vep_file.merge(vcf_df, on="#Uploaded_variation")

    if "pos" in vep_file.columns:
        vep_file["pos"] = vep_file["pos"].astype(int)

    vep_file["chrom"] = vep_file["chrom"].apply(
        lambda x: "{}{}".format("chr", x.split("chr")[-1])
    )

    str_cols = [
        "STRAND",
        "TSL",
        "GENE_PHENO",
        "CADD_PHRED",
        "CADD_RAW",
        "SpliceAI_pred",
        "BIOTYPE",
        "Gene",
    ]
    str_cols_present = [i for i in str_cols if i in vep_file.columns]
    vep_file[str_cols_present] = vep_file[str_cols_present].astype(str)

    float_vals = [
        "DISTANCE",
        "gnomADg_FIN_AF",
        "AF",
        "AFR_AF",
        "AMR_AF",
        "EAS_AF",
        "EUR_AF",
        "SAS_AF",
        "MAX_AF",
        "MOTIF_POS",
        "MOTIF_SCORE_CHANGE",
        "CADD_PHRED",
        "CADD_RAW",
        "PrimateAI",
        "TSL",
        "Condel",
    ]
    float_vals_present = [i for i in float_vals if i in vep_file.columns]
    vep_file[float_vals_present] = (
        vep_file[float_vals_present].replace("-", "NaN").astype(float)
    )

    necessary_columns = (
        [
            "chrom",
            "pos",
            "ref",
            "alt",
            "Gene",
            "gnomADe_NFE_AF",
            "CADD_PHRED",
            "CADD_RAW",
            "Consequence",
            "PrimateAI",
            "Alpha_Missense",
            "am_pathogenicity",
            "AbSplice_DNA",
            "PolyPhen",
            "SIFT",
            "SIFT_score",
            "PolyPhen_score",
            "UKB_AF",
            "combined_UKB_NFE_AF",
            "combined_UKB_NFE_AF_MB",
            "gene_id",
            "Condel",
        ]
        + str_cols
        + float_vals
        + (vepcols_to_retain or [])
    )
    necessary_columns_present = [i for i in necessary_columns if i in vep_file.columns]
    

    vep_file = vep_file[list(set(necessary_columns_present))]

    if "SpliceAI_pred" in vep_file.columns:
        vep_file["SpliceAI_delta_score"] = vep_file["SpliceAI_pred"].apply(
            calculate_scores_max
        )

    if "Consequence" in vep_file.columns:
        dummies = (
            vep_file["Consequence"].str.get_dummies(",").add_prefix("Consequence_")
        )
    else:
        raise ValueError("'Consequence' column expected to be in VEP output")
    all_consequences = [
        "Consequence_splice_acceptor_variant",
        "Consequence_5_prime_UTR_variant",
        "Consequence_TFBS_ablation",
        "Consequence_start_lost",
        "Consequence_incomplete_terminal_codon_variant",
        "Consequence_intron_variant",
        "Consequence_stop_gained",
        "Consequence_splice_donor_5th_base_variant",
        "Consequence_downstream_gene_variant",
        "Consequence_intergenic_variant",
        "Consequence_splice_donor_variant",
        "Consequence_NMD_transcript_variant",
        "Consequence_protein_altering_variant",
        "Consequence_splice_polypyrimidine_tract_variant",
        "Consequence_inframe_insertion",
        "Consequence_mature_miRNA_variant",
        "Consequence_synonymous_variant",
        "Consequence_regulatory_region_variant",
        "Consequence_non_coding_transcript_exon_variant",
        "Consequence_stop_lost",
        "Consequence_TF_binding_site_variant",
        "Consequence_splice_donor_region_variant",
        "Consequence_stop_retained_variant",
        "Consequence_splice_region_variant",
        "Consequence_coding_sequence_variant",
        "Consequence_upstream_gene_variant",
        "Consequence_frameshift_variant",
        "Consequence_start_retained_variant",
        "Consequence_3_prime_UTR_variant",
        "Consequence_inframe_deletion",
        "Consequence_missense_variant",
        "Consequence_non_coding_transcript_variant",
    ]
    all_consequences = list(set(all_consequences))
    mask = pd.DataFrame(
        data=np.zeros(shape=(len(vep_file), len(all_consequences))),
        columns=all_consequences,
        dtype=float,
    )
    mask[list(dummies.columns)] = dummies
    vep_file[mask.columns] = mask

    return vep_file


@cli.command()
@click.argument("filenames", type=str)
@click.argument("out_file", type=click.Path())
def concat_annotations(
    filenames: str,
    out_file: str,
):
    """
    Concatenate multiple annotation files based on the specified pattern and create a single output file.

    Parameters:
    - filenames (str): File paths for annotation files to concatenate
    - out_file (str): Output file path.

    Returns:
    None

    Example:
    concat_annotations "annotations/chr1_block0_merged.parquet,annotations/chr1_block1_merged.parquet,annotations/chr1_block2_merged.parquet " "output.parquet")
    """
    file_paths = filenames.split(",")
    for f in tqdm(file_paths):
        logger.info(f"processing file {f}")
        file = pd.read_parquet(f)
        logger.info(file.shape)
        logger.info(file.columns)

        if f == file_paths[0]:
            logger.info("creating new file")
            file.to_parquet(out_file, engine="fastparquet")
        else:
            try:
                file.to_parquet(out_file, engine="fastparquet", append=True)
            except ValueError:
                out_df_columns = pd.read_parquet(out_file, engine="fastparquet").columns

                logger.error(
                    f"columns are not equal in saved/appending file: {[i for i in out_df_columns if i not in file.columns]} and {[i for i in file.columns if i not in out_df_columns]} "
                )

                raise ValueError


@cli.command()
@click.argument("genotype_file", type=click.Path(exists=True))
@click.argument("variants_filepath", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
def get_af_from_gt(genotype_file: str, variants_filepath: str, out_file: str):
    """
    Compute allele frequencies from genotype data.

    Parameters:
    - genotype_file (str): Path to the genotype file.
    - variants_filepath (str): Path to the variants file.
    - out_file (str): Output file path for storing allele frequencies.
    """
    import h5py

    variants = pd.read_table(variants_filepath)
    max_variant_id = variants["id"].max()

    logger.info("Computing allele frequencies")
    variant_counts = np.zeros(max_variant_id + 1)
    with h5py.File(genotype_file, "r") as f:
        variant_matrix = f["variant_matrix"]
        genotype_matrix = f["genotype_matrix"]
        n_samples = variant_matrix.shape[0]
        for i in trange(n_samples):
            variants = variant_matrix[i]
            mask = variants > 0
            variants = variants[mask]
            genotype = genotype_matrix[i]
            genotype = genotype[mask]
            variant_counts[variants] += genotype

    af = variant_counts / (2 * n_samples)
    af_df = pd.DataFrame({"id": np.arange(max_variant_id + 1), "af": af})
    af_df.to_parquet(out_file)


@cli.command()
@click.argument("annotations_path", type=click.Path(exists=True))
@click.argument("af_df_path", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
def merge_af(annotations_path: str, af_df_path: str, out_file: str):
    """
    Merge allele frequency data into annotations and save to a file.

    Parameters:
    - annotations_path (str): Path to the annotations file.
    - af_df_path (str): Path to the allele frequency DataFrame file.
    - out_file (str): Path to the output file to save merged data.
    """
    annotations_df = pd.read_parquet(annotations_path)
    af_df = pd.read_parquet(af_df_path)
    merged_df = annotations_df.merge(af_df, how="left", on="id")
    merged_df.to_parquet(out_file)


@cli.command()
@click.argument("annotations_path", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
def calculate_maf(annotations_path: str, out_file: str):
    """
    Calculate minor allele frequency (MAF) from allele frequency data in annotations.

    Parameters:
    - annotations_path (str): Path to the annotations file containing allele frequency data.
    - out_file (str): Path to the output file to save the calculated MAF data.
    """
    annotation_file = pd.read_parquet(annotations_path)
    af = annotation_file["af"]
    annotation_file = annotation_file.drop(
        columns=["UKB_AF_MB", "UKB_MAF"], errors="ignore"
    )
    annotation_file["maf"] = af.apply(lambda x: min(x, 1 - x))
    annotation_file["maf_mb"] = (af * (1 - af) + 1e-8) ** (-0.5)
    annotation_file.to_parquet(out_file)


@cli.command()
@click.argument("protein_id_file", type=click.Path(exists=True))
@click.argument("annotations_path", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
def add_protein_ids(protein_id_file: str, annotations_path: str, out_file: str):
    """
    Add protein IDs to the annotations based on protein ID mapping file.

    Parameters:
    - protein_id_file (str): Path to the protein ID mapping file.
    - annotations_path (str): Path to the annotations file.
    - out_file (str): Path to the output file to save the annotations with protein IDs.
    """
    genes = pd.read_parquet(protein_id_file)
    genes[["gene_base", "feature"]] = genes["gene"].str.split(".", expand=True)
    genes.drop(columns=["feature", "gene", "gene_name", "gene_type"], inplace=True)
    genes.rename(columns={"id": "gene_id"}, inplace=True)
    annotations = pd.read_parquet(annotations_path)
    len_anno = len(annotations)
    annotations.rename(columns={"gene_id": "gene_base"}, inplace=True)
    merged = annotations.merge(genes, on=["gene_base"], how="left")
    assert len(merged) == len_anno
    merged.to_parquet(out_file)


@cli.command()
@click.argument("gtf_filepath", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
def create_protein_id_file(gtf_filepath: str, out_file: str):
    """
    Create a protein ID mapping file from the GTF file.

    Parameters:
    - gtf_filepath (str): Path to the GTF file.
    - out_file (str): Path to save the output protein ID mapping file.
    """
    import pyranges as pr

    gtf = pr.read_gtf(gtf_filepath)
    gtf = gtf.as_df()
    gtf = gtf.query(
        "Feature =='gene' and gene_type=='protein_coding' and Source=='HAVANA'"
    )
    gtf = (
        gtf[["gene_id", "gene_type", "gene_name"]]
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"gene_id": "gene", "index": "id"})
    )
    gtf.to_parquet(out_file)


@cli.command()
@click.argument("annotation_columns_yaml_file", type=click.Path(exists=True))
@click.argument("annotations_path", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
def select_rename_fill_annotations(
    annotation_columns_yaml_file: str, annotations_path: str, out_file: str
):
    """
    Select, rename, and fill missing values in annotation columns based on a YAML configuration file.

    Parameters:
    - annotation_columns_yaml_file (str): Path to the YAML file containing name and fill value mappings.
    - annotations_path (str): Path to the annotations file.
    - out_file (str): Path to save the modified annotations file.
    """

    logger.info(
        f"reading  in yaml file containing name and fill value mappings from {annotation_columns_yaml_file}"
    )
    prior_names, _, _, column_name_mapping, fill_value_mapping = readYamlColumns(
        annotation_columns_yaml_file
    )
    key_cols = ["id", "gene_id"]
    anno_df = pd.read_parquet(
        annotations_path, columns=list(set(prior_names + key_cols))
    )
    anno_df.rename(columns=column_name_mapping, inplace=True)
    anno_df.fillna(fill_value_mapping, inplace=True)
    anno_df.to_parquet(out_file)


if __name__ == "__main__":
    cli()

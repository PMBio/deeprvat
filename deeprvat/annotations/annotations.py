import itertools
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

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


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def deepripe_get_model_info(saved_models_dict, saved_deepripe_models_path):
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
    "converts the sequence to one-hot encoding"

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
    file_name = variants_file.split("/")[-1]
    print(f"Generating BED file: {output_dir}/{file_name[:-3]}bed")

    df_variants = pd.read_csv(
        variants_file, sep="\t", names=["#CHROM", "POS", "ID", "REF", "ALT"]
    )  # hg38

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


def deepripe_score_variant_onlyseq_all(
        model_group, variant_bed, genomefasta, seq_len=200, batch_size=1024, n_jobs=32
):
    predictions = {}
    # counter = 0
    # encoded_seqs_list_old = []
    # logger.info("Encoding sequences")
    # for bedline in tqdm(variant_bed):
    #     counter += 1
    #     encoded_seqs = deepripe_encode_variant_bedline(
    #         bedline, genomefasta, flank_size=(seq_len // 2) + 2
    #     )

    #     if encoded_seqs is None:
    #         encoded_seqs = np.ones(encoded_seqs_list_old[0].shape) * float("nan")

    #     encoded_seqs_list_old.append(encoded_seqs)

    #     if counter % 100000 == 0:
    #         pybedtools.cleanup(remove_all=True)

    encoded_seqs_list = Parallel(n_jobs=n_jobs, verbose=10)(delayed(deepripe_encode_variant_bedline)(
            bedline, genomefasta, flank_size=(seq_len // 2) + 2
        ) for bedline in variant_bed)
    encoded_seqs_list = [(x if x is not None
                          else np.ones((2, seq_len + 4, 4)) * float("nan"))
                         for x in encoded_seqs_list]
    encoded_seqs = tf.concat(encoded_seqs_list, 0)

    logger.info("Computing predictions")
    ## shifting around (seq_len+4) 4 bases
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


@click.group()
def cli():
    pass


@cli.command()
@click.option("--n-components", type=int, default=100)
@click.argument("deepsea-file", type=click.Path(exists=True))
@click.argument("pca-object", type=click.Path())
@click.argument("out-dir", type=click.Path(exists=True))
def deepsea_pca(n_components: int, deepsea_file: str, pca_onject: str, out_dir: str):
    logger.info("Loading deepSea data")
    df = pd.read_csv(deepsea_file)
    logger.info("filling NAs")
    df = df.fillna(0)
    logger.info("Extracting matrix for PCA")
    key_df = df[["chrom", "pos", "ref", "alt", "id"]].reset_index(drop=True)
    logger.info("transforming values to numpy")
    X = df[[c for c in df.columns if c.startswith("DeepSEA")]].to_numpy()
    del df
    logger.info("standardizing values")
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    del X

    

    out_path = Path(out_dir)

    if os.path.exists(pca_object):
        if '.pkl' in pca_object:
            with open(pca_object, 'rb') as pickle_file:
                logger.info("loading pca objectas pickle file")
                pca = pickle.load(pickle_file)
                X_pca = pca.transform(X_std)
        else: 
            if '.npy' not in pca_object:
                logger.error('did not recognize file format, assuming npy')
            logger.info('loading pca components as npy object')
            components = np.load(pca_object)
            logger.info(f"Projecting data to {pca.components_.shape[0]} PCs")
            X_pca = np.matmul(X_std, pca.components_.transpose())
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
        n_jobs=n_jobs
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


def process_chunk(chrom_file, abs_splice_res_dir, tissues_to_exclude,tissue_agg_function, ca_shortened ):
    logger.info(f"Reading file {chrom_file}")
    ab_splice_res = pd.read_csv(abs_splice_res_dir / chrom_file).reset_index()
    ab_splice_res = ab_splice_res.query("tissue not in @tissues_to_exclude")
    logger.info(
        f"AbSplice tissues excluded: {tissues_to_exclude}, Aggregating AbSplice scores using {tissue_agg_function}"
    )
    logger.info(
        f"Number of unique variants {len(ab_splice_res['variant'].unique())}"
    )

    #### aggregate tissue specific ab splice scores
    ab_splice_res = (
        ab_splice_res.groupby(["variant", "gene_id"])
        .agg({"AbSplice_DNA": tissue_agg_function})
        .reset_index()
    )

    ab_splice_res[["chrom", "pos", "var"]] = ab_splice_res["variant"].str.split(
        ":", expand=True
    )

    ab_splice_res[["ref", "alt"]] = ab_splice_res["var"].str.split(
        ">", expand=True
    )

    ab_splice_res["pos"] = ab_splice_res["pos"].astype(int)
    logger.info(f"Number of rows of ab_splice df {len(ab_splice_res)}")
    merged = ab_splice_res.merge(
        ca_shortened, how="left", on=["chrom", "pos", "ref", "alt", "gene_id"]
    )
    logger.info(
        f"Number of unique variants(id) in merged {len(merged['id'].unique())}"
    )
    logger.info(
        f"Number of unique variants(variant) in merged {len(merged['variant'].unique())}"
    )

    return merged

    del merged
    del ab_splice_res


@cli.command()
@click.argument("current_annotation_file", type=click.Path(exists=True))
@click.argument("abs_splice_res_dir", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
@click.argument("absplice_score_file", type=click.Path())
@click.argument("njobs", type=int)
def get_abscores(
    current_annotation_file: str,
    abs_splice_res_dir: str,
    out_file: str,
    absplice_score_file: str,
    njobs:int
):
    current_annotation_file = Path(current_annotation_file)
    logger.info("reading current annotations file")
    current_annotations = pd.read_parquet(current_annotation_file)

    if "AbSplice_DNA" in current_annotations.columns:
        if "AbSplice_DNA_old" in current_annotations.columns:
            current_annotations.drop("AbSplice_DNA_old", inplace=True)
        current_annotations = current_annotations.rename(
            columns={"AbSplice_DNA": "AbSplice_DNA_old"}
        )
    ca_shortened = current_annotations[["id", "gene_id", "chrom", "pos", "ref", "alt"]]

    logger.info(ca_shortened.columns)

    abs_splice_res_dir = Path(abs_splice_res_dir)

    tissue_agg_function = "max"
    tissues_to_exclude = ["Testis"]
    tissues_to_exclude = []
    ab_splice_agg_score_file = absplice_score_file

    if not Path(ab_splice_agg_score_file).exists():
        logger.info("creating abSplice score file.. ")
        all_absplice_scores = []
        parallel = Parallel(n_jobs=njobs, return_as="generator")
        output_generator = parallel(delayed(process_chunk)(i , abs_splice_res_dir, tissues_to_exclude, tissue_agg_function, ca_shortened) for i in tqdm(os.listdir(abs_splice_res_dir)))
        all_absplice_scores = list(output_generator)
        

        logger.info("concatenating files")
        all_absplice_scores = pd.concat(all_absplice_scores)
        logger.info(f"saving score file to {ab_splice_agg_score_file}")
        all_absplice_scores.to_parquet(ab_splice_agg_score_file)

    else:
        logger.info("reading existing abSplice Score file")
        all_absplice_scores = pd.read_parquet(ab_splice_agg_score_file)

    all_absplice_scores = all_absplice_scores[
        ["chrom", "pos", "ref", "alt", "gene_id", "AbSplice_DNA"]
    ]

    annotations = pd.read_parquet(current_annotation_file, engine="pyarrow").drop(
        columns=["AbSplice_DNA"], errors="ignore"
    )
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
    assert merged["censequence_id"].unique().shape[0] == len(merged)

    logger.info(
        f'Filling {merged["AbSplice_DNA"].isna().sum()} '
        "missing AbSplice values with 0"
    )
    merged["AbSplice_DNA"] = merged["AbSplice_DNA"].fillna(0)

    annotation_out_file = out_file

    logger.info(f"Writing to {annotation_out_file}")
    merged.to_parquet(annotation_out_file, engine="pyarrow")



pd.options.mode.chained_assignment = None


logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@cli.command()
@click.option("--n-components", type=int, default=59)
@click.argument("deepripe-file", type=click.Path(exists=True))
@click.argument("out-dir", type=click.Path(exists=True))
def deepripe_pca(n_components: int, deepripe_file: str, out_dir: str):
    logger.info("Reading deepripe file")
    df = pd.read_csv(deepripe_file)
    df = df.drop(["Uploaded_variant"], axis=1)
    print(df.columns)
    df = df.dropna()
    key_df = df[["chrom", "pos", "ref", "alt", "id"]].reset_index(drop=True)

    logger.info("Extracting matrix for PCA")
    X = df[[c for c in df.columns if c not in key_df.columns]].to_numpy()
    del df
    logger.info("transforming columns to z scores")
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    del X

    logger.info("Running PCA")
    pca = PCA(n_components=n_components)
    pca.fit(X_std)
    out_path = Path(out_dir)
    with open(out_path / "pca.pkl", "wb") as f:
        pickle.dump(pca, f)

    logger.info(f"Projecting rows to {n_components} PCs")
    X_pca = pca.transform(X_std)
    del X_std
    pca_df = pd.DataFrame(
        X_pca, columns=[f"DeepRipe_PC_{i}" for i in range(1, n_components + 1)]
    )
    del X_pca
    pca_df = pd.concat([key_df, pca_df], axis=1)
    pca_df.to_parquet(out_path / "deepripe_pca.parquet", engine="pyarrow")

    logger.info("Done")


@cli.command()
@click.argument("annotation_file", type=click.Path(exists=True))
@click.argument("deepripe_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
@click.argument("column_prefix", type=str)
def merge_deepripe(
    annotation_file: str, deepripe_file: str, out_file: str, column_prefix: str
):
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
@click.argument("out_file", type=click.Path())
def merge_deepsea_pcas(annotation_file: str, deepripe_pca_file: str, out_file: str):
    annotations = pd.read_parquet(annotation_file)
    deepripe_pcas = pd.read_parquet(deepripe_pca_file)
    orig_len = len(annotations)
    merged = annotations.merge(
        deepripe_pcas, how="left", on=["chrom", "pos", "ref", "alt"]
    )
    assert len(merged) == orig_len
    merged.to_parquet(out_file)


@cli.command()
@click.argument("in_variants", type=click.Path(exists=True))
@click.argument("out_variants", type=click.Path())
def process_annotations(in_variants: str, out_variants: str):
    variant_path = Path(in_variants)
    variants = pd.read_parquet(variant_path)

    logger.info("filtering for canonical variants")

    variants = variants.loc[variants.CANONICAL == "YES"]
    variants.rename(columns={"Gene": "gene_id"}, inplace=True)

    logger.info("aggregating consequences for different alleles")

    # combining variant id with gene id
    variants["censequence_id"] = variants["id"].astype(str) + variants["gene_id"]
    variants.to_parquet(out_variants)


def process_chunk_addids(chunk, variants):
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
    chunk_shape= chunk.shape
    chunk.drop_duplicates(subset=key_cols, inplace = True)
    chunk = pd.merge(chunk, variants,  on=key_cols, how="left", validate="1:1")
    
    try: 
        assert chunk_shape[0] == chunk.shape[0]
    except AssertionError: 
        logger.error(f"df.shape[0] was {chunk.shape[0]} but chunk_shape[0] was {chunk_shape[0]}")
        raise AssertionError
    try:
        assert chunk.shape[1] == chunk_shape[1] + 1
    except AssertionError:
        logger.error(f"chunk.shape[1] was {chunk.shape[1]} but chunk_shape[1] + 1 was {chunk_shape[1] + 1}")
        raise AssertionError       
    return chunk

@cli.command()
@click.argument("annotation_file", type=click.Path(exists=True))
@click.argument("variant_file", type=click.Path(exists=True))
@click.argument("njobs", type=int)
@click.argument("out_file", type=click.Path())
def add_ids(annotation_file: str, variant_file: str, njobs:int, out_file: str):
    data = pd.read_csv(annotation_file, chunksize=10_000)
    
    

    all_variants = pd.read_csv(variant_file, sep="\t")
    parallel = Parallel(n_jobs=njobs, return_as="generator")

    output_generator = parallel(delayed(process_chunk_addids)(chunk, all_variants) for chunk in data)
    pd.concat([batch for batch in output_generator]).to_csv(out_file, index=False)
    


@cli.command()
@click.option("--included-chromosomes", type=str)
@click.option("--comment-lines", is_flag=True)
@click.option("--sep", type=str, default=",")
@click.argument("annotation_dir", type=click.Path(exists=True))
@click.argument("deepripe_name_pattern", type=str)
@click.argument("pvcf-blocks_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
def concatenate_deepripe(
    included_chromosomes: Optional[str],
    sep: str,
    comment_lines: bool,
    annotation_dir: str,
    deepripe_name_pattern: str,
    pvcf_blocks_file: str,
    out_file: str,
):
    annotation_dir = Path(annotation_dir)

    logger.info("Reading variant file")

    logger.info("reading pvcf block file")
    pvcf_blocks_df = pd.read_csv(
        pvcf_blocks_file,
        sep="\t",
        header=None,
        names=["Index", "Chromosome", "Block", "First position", "Last position"],
        dtype={"Chromosome": str},
    ).set_index("Index")
    if included_chromosomes is not None:
        included_chromosomes = [int(c) for c in included_chromosomes.split(",")]
        pvcf_blocks_df = pvcf_blocks_df[
            pvcf_blocks_df["Chromosome"].isin([str(c) for c in included_chromosomes])
        ]
    pvcf_blocks = zip(pvcf_blocks_df["Chromosome"], pvcf_blocks_df["Block"])
    file_paths = [
        annotation_dir / deepripe_name_pattern.format(chr=p[0], block=p[1])
        for p in pvcf_blocks
    ]
    logger.info("check if out_file already exists")
    if os.path.exists(out_file):
        logger.info("file exists, removing existing file")
        os.remove(out_file)
    else:
        logger.info("out_file does not yet exist")

    logger.info("reading in f")
    for f in tqdm(file_paths):
        if comment_lines:
            current_file = pd.read_csv(f, comment="#", sep=sep, low_memory=False)
        else:
            current_file = pd.read_csv(f, sep=sep, low_memory=False)
        if f == file_paths[0]:
            logger.info("creating new file")
            current_file.to_csv(out_file, mode="a", index=False)
        else:
            current_file.to_csv(out_file, mode="a", index=False, header=False)



@cli.command()
@click.argument("vep_header_line", type=int)
@click.argument("vep_file", type=click.Path(exists=True))
@click.argument("deepripe_parclip_file", type=click.Path(exists=True))
@click.argument("deepripe_hg2_file", type=click.Path(exists=True))
@click.argument("deepripe_k5_file", type=click.Path(exists=True))
@click.argument("variant_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())

def merge_annotations(vep_header_line:int,
                    vep_file:str,
                    deepripe_parclip_file:str,
                    deepripe_hg2_file:str,
                    deepripe_k5_file:str,
                    variant_file:str,
                    out_file:str,
                  ):
    #load vep file
    vep_df = pd.read_csv(
                vep_file,
                header=vep_header_line,
                sep="\t",
                na_values = "-"
            )
    vep_df = process_vep(vep_file=vep_df)
    logger.info(f"vep_df shape is {vep_df.shape}")
    #load deepripe_parclip
    deepripe_parclip_df = pd.read_csv(deepripe_parclip_file)
    deepripe_parclip_df = process_deepripe(deepripe_parclip_df, "parclip")
    #load deepripe_k5
    deepripe_k5_df = pd.read_csv(deepripe_k5_file)
    deepripe_k5_df = process_deepripe(deepripe_k5_df, "k5")
    #load deepripe_hg2
    deepripe_hg2_df = pd.read_csv(deepripe_hg2_file)
    deepripe_hg2_df = process_deepripe(deepripe_hg2_df, "hg2")
    #load variant_file
    logger.info(f"reading in {variant_file}")
    variants = pd.read_csv(variant_file, sep="\t")

    #merge vep to variants M:1
    ca = vep_df.merge(variants, how = "left",  on=["chrom", "pos", "ref", "alt"], validate= "m:1")
    del vep_df
    #merge deepripe files to variants 1:1
    logger.info(ca.columns)
    logger.info(deepripe_parclip_df.columns)
    ca = ca.merge(deepripe_parclip_df, how = "left", on=["chrom", "pos", "ref", "alt"], validate="m:1")
    ca = ca.merge(deepripe_k5_df, how = "left", on=["chrom", "pos", "ref", "alt"], validate="m:1")
    ca = ca.merge(deepripe_hg2_df, how = "left", on=["chrom", "pos", "ref", "alt"], validate="m:1")
    
    ca.to_parquet(out_file)


def process_deepripe(deepripe_df:object, column_prefix:str)->object:
    
    logger.info("renaming deepripe columns")
    deepripe_df= deepripe_df.rename(columns={"chr": "chrom"})
    
    deepripe_df = deepripe_df.drop(
        columns=["Uploaded_variant", "Unnamed: 0"], errors="ignore"
    )
    key_cols = ["chrom", "pos", "ref", "alt", "id"]
    prefix_cols = [x for x in deepripe_df.columns if x not in key_cols]
    new_names = [(i, i + f"_{column_prefix}") for i in prefix_cols]
    deepripe_df = deepripe_df.rename(columns=dict(new_names))
    deepripe_df.drop_duplicates(subset=["chrom", "pos", "ref", "alt"], inplace=True)
    return deepripe_df

def process_vep(vep_file: object) -> object:
    vep_file[["chrom", "pos", "ref", "alt"]] = (
        vep_file["#Uploaded_variation"]
        .str.replace("_", ":")
        .str.replace("/", ":")
        .str.split(":", expand=True)
    )
    
    vep_file["pos"] = vep_file["pos"].astype(int)   
    logger.info(vep_file.columns)
    vep_file[["STRAND","TSL", "GENE_PHENO", "CADD_PHRED","CADD_RAW"]] =vep_file[["STRAND","TSL", "GENE_PHENO", "CADD_PHRED","CADD_RAW"]].astype(str)
    float_vals = ['DISTANCE', 'gnomADg_FIN_AF', 'AF', 'AFR_AF', 'AMR_AF','EAS_AF', 'EUR_AF', 'SAS_AF', 'MAX_AF','MOTIF_POS', 'MOTIF_SCORE_CHANGE',  'CADD_PHRED', 'CADD_RAW', 'PrimateAI', 'TSL', 'Condel']    
    vep_file[float_vals] = vep_file[float_vals].replace('-', 'NaN').astype(float)
    dummies = vep_file["Consequence"].str.get_dummies(",").add_prefix("Consequence_")
    hopefully_all_consequences=  ['Consequence_splice_acceptor_variant','Consequence_5_prime_UTR_variant','Consequence_TFBS_ablation','Consequence_start_lost','Consequence_incomplete_terminal_codon_variant','Consequence_intron_variant', 'Consequence_stop_gained', 'Consequence_splice_donor_5th_base_variant', 'Consequence_downstream_gene_variant', 'Consequence_intergenic_variant', 'Consequence_splice_donor_variant','Consequence_NMD_transcript_variant', 'Consequence_protein_altering_variant', 'Consequence_splice_polypyrimidine_tract_variant', 'Consequence_inframe_insertion', 'Consequence_mature_miRNA_variant', 'Consequence_synonymous_variant', 'Consequence_regulatory_region_variant', 'Consequence_non_coding_transcript_exon_variant', 'Consequence_stop_lost', 'Consequence_TF_binding_site_variant', 'Consequence_splice_donor_region_variant', 'Consequence_stop_retained_variant', 'Consequence_splice_region_variant', 'Consequence_coding_sequence_variant', 'Consequence_upstream_gene_variant', 'Consequence_frameshift_variant', 'Consequence_start_retained_variant', 'Consequence_3_prime_UTR_variant', 'Consequence_inframe_deletion', 'Consequence_missense_variant', 'Consequence_non_coding_transcript_variant']
    hopefully_all_consequences = list(set(hopefully_all_consequences))
    mask = pd.DataFrame(data = np.zeros(shape= ( len(vep_file), len(hopefully_all_consequences))), columns=hopefully_all_consequences ,  dtype=float)
    mask[list(dummies.columns)]=dummies
    vep_file[mask.columns]=mask
    return vep_file

@cli.command()
@click.argument("pvcf_blocks_file", type=click.Path(exists=True))
@click.argument("annotation_dir", type=click.Path(exists=True))
@click.argument("filename_pattern", type=str)
@click.argument("out_file", type=click.Path())
@click.option("--included-chromosomes", type=str)
def concat_annotations(pvcf_blocks_file:str, annotation_dir:str, filename_pattern:str, out_file:str, included_chromosomes:Optional[str]):
    logger.info("reading pvcf block file")
    pvcf_blocks_df = pd.read_csv(
        pvcf_blocks_file,
        sep="\t",
        header=None,
        names=["Index", "Chromosome", "Block", "First position", "Last position"],
        dtype={"Chromosome": str},
    ).set_index("Index")
    if included_chromosomes is not None:
        included_chromosomes = [int(c) for c in included_chromosomes.split(",")]
        pvcf_blocks_df = pvcf_blocks_df[
            pvcf_blocks_df["Chromosome"].isin([str(c) for c in included_chromosomes])
        ]
    pvcf_blocks = zip(pvcf_blocks_df["Chromosome"], pvcf_blocks_df["Block"])
    annotation_dir = Path(annotation_dir)
    file_paths = [
        annotation_dir / filename_pattern.format(chr=p[0], block=p[1])
        for p in pvcf_blocks
    ]
    for f in tqdm(file_paths):
        logger.info(f"processing file {f}")
        file = pd.read_parquet(f)
        logger.info(file.shape)
        logger.info(file.columns)

        if f == file_paths[0]:
            logger.info("creating new file")
            file.to_parquet(out_file, engine= "fastparquet")
        else:
            try:
                file.to_parquet(out_file, engine= "fastparquet", append=True)   
            except ValueError: 
                
                out_df_columns = pd.read_parquet(out_file, engine= "fastparquet").columns
                
                logger.error(f"columns are not equal in saved/appending file: {[i for i in out_df_columns if i not in file.columns]} and {[i for i in file.columns if i not in out_df_columns]} ")
                
                raise ValueError
if __name__ == "__main__":
    cli()

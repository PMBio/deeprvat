from pathlib import Path
import click
import numpy as np
import pandas as pd
import logging
import sys
import pickle
from typing import List, Optional
from sklearn.decomposition import PCA
import dask.dataframe as dd
import os
import tqdm
import pybedtools
import time
import random 
from keras.models import load_model
import keras.backend as K




def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #TPs=K.sum(K.round(K.clip(y_true * y_pred , 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #TPs=K.sum(K.round(K.clip(y_ture * y_pred , 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def deepripe_get_model_info(saved_models_dict, saved_deepripe_models_path):
    #path_to_model = 'Results/PARCLIP_models/'
    shared_path = Path(saved_deepripe_models_path)
    
    # parclip
    path_to_m_high= shared_path / f'{saved_models_dict["parclip"]}_high_seq.h5' 
    path_to_m_med=  shared_path / f'{saved_models_dict["parclip"]}_med_seq.h5'   
    path_to_m_low=  shared_path / f'{saved_models_dict["parclip"]}_low_seq.h5'   
    
    #eclip HepG2
    path_to_hg_high1= shared_path / f'{saved_models_dict["eclip_hg2"]}_high1_seq.h5' 
    path_to_hg_high2= shared_path / f'{saved_models_dict["eclip_hg2"]}_high2_seq.h5'   
    path_to_hg_mid1=  shared_path / f'{saved_models_dict["eclip_hg2"]}_mid1_seq.h5' 
    path_to_hg_mid2=  shared_path / f'{saved_models_dict["eclip_hg2"]}_mid2_seq.h5' 
    path_to_hg_low=   shared_path / f'{saved_models_dict["eclip_hg2"]}_low_seq.h5'   
    
    #eclip K562
    path_to_k5_high1= shared_path / f'{saved_models_dict["eclip_k5"]}_high1_seq.h5' 
    path_to_k5_high2= shared_path / f'{saved_models_dict["eclip_k5"]}_high2_seq.h5'   
    path_to_k5_mid1=  shared_path / f'{saved_models_dict["eclip_k5"]}_mid1_seq.h5' 
    path_to_k5_mid2=  shared_path / f'{saved_models_dict["eclip_k5"]}_mid2_seq.h5' 
    path_to_k5_low=   shared_path / f'{saved_models_dict["eclip_k5"]}_low_seq.h5'  
    
    
    saved_paths = { 'parclip': [path_to_m_high, path_to_m_med, path_to_m_low],
                    'eclip_hg2': [path_to_hg_high1, path_to_hg_high2, path_to_hg_mid1,
                                  path_to_hg_mid2, path_to_hg_low],
                    'eclip_k5':  [path_to_k5_high1, path_to_k5_high2, path_to_k5_mid1,
                                  path_to_k5_mid2, path_to_k5_low]
                  }
    
    ### parclip
    pc_RBPnames_low=np.array(['MBNL1', 'P53_NONO', 'PUM2', 'QKI', 'AGO3', 'FUS', 
                              'TAF15', 'ZFP36', 'DICER1','EIF3A', 'EIF3D', 
                              'EIF3G', 'SSB', 'PAPD5', 'CPSF4', 'CPSF3', 'RTCB', 
                              'FXR1', 'NOP58', 'NOP56', 'FBL', 'LIN28A', 'LIN28B', 
                              'UPF1', 'G35', 'G45', 'XPO5']) #27
    
    pc_RBPnames_med=np.array(['TARDBP', 'ELAVL2', 'ELAVL3', 'ELAVL4', 'RBM20', 
                              'IGF2BP1', 'IGF2BP2', 'IGF2BP3','EWSR1', 'HNRNPD', 
                              'RBPMS', 'SRRM4', 'AGO2', 'NUDT21', 'FIP1L1', 'CAPRIN1', 
                              'FMR1iso7', 'FXR2', 'AGO1', 'L1RE1', 'ORF1']) #21
    
    pc_RBPnames_high=np.array(['DND1', 'CPSF7', 'CPSF6', 'CPSF1', 'CSTF2', 
                               'CSTF2T', 'ZC3H7B', 'FMR1iso1', 
                               'RBM10', 'MOV10', 'ELAVL1']) #11
    
    
    ### eclip HepG2
    hg2_RBPnames_high1 = np.array(['DDX3X', 'PCBP2', 'FAM120A', 'HNRNPL', 'RBFOX2', 
                          'PTBP1', 'MATR3', 'EFTUD2', 'PRPF4', 'UPF1'])
    
    hg2_RBPnames_high2 = np.array([ 'GRWD1', 'PRPF8', 'PPIG', 'CSTF2T', 'QKI', 
                          'U2AF2', 'SUGP2', 'HNRNPM', 'AQR', 'BCLAF1'])
    
    hg2_RBPnames_mid1 = np.array([ 'LSM11', 'NKRF', 'SUB1', 'NCBP2', 'UCHL5', 'LIN28B', 
                         'IGF2BP3', 'SF3A3', 'AGGF1', 'DROSHA', 'DDX59','CSTF2', 
                         'DKC1', 'EIF3H', 'FUBP3','SFPQ', 'HNRNPC', 'ILF3', 
                         'TIAL1', 'HLTF', 'ZNF800', 'PABPN1', 'YBX3', 'FXR2', ])
    
    hg2_RBPnames_mid2 = np.array([ 'GTF2F1', 'IGF2BP1', 'HNRNPK', 'XPO5', 'RPS3',
                         'SF3B4', 'LARP4', 'BUD13', 'SND1', 'G3BP1', 'AKAP1', 'KHSRP'])
    
    hg2_RBPnames_low = np.array([ 'RBM22', 'GRSF1', 'CDC40', 'NOLC1', 'FKBP4', 'DGCR8', 
                        'ZC3H11A', 'XRN2', 'SLTM', 'DDX55', 'TIA1', 'SRSF1', 'U2AF1', 'RBM15'])
    
    
    ### eclip K562
    k562_RBPnames_high1 = np.array(['BUD13', 'PTBP1', 'DDX24', 'EWSR1', 'RBM15'])
    
    k562_RBPnames_high2 = np.array(['SF3B4', 'YBX3', 'UCHL5', 'KHSRP', 'ZNF622', 
                           'NONO', 'EXOSC5', 'PRPF8', 'CSTF2T', 'AQR', 'UPF1'])
    
    k562_RBPnames_mid1 = np.array([ 'U2AF2', 'AKAP8L', 'METAP2', 'SMNDC1', 'GEMIN5', 
                          'HNRNPK', 'SLTM', 'SRSF1', 'FMR1', 'SAFB2', 'DROSHA','RPS3', 
                          'IGF2BP2', 'ILF3','RBFOX2', 'QKI', 'PCBP1', 'ZNF800', 'PUM1'])
    
    k562_RBPnames_mid2 = np.array(['EFTUD2', 'LIN28B', 'AGGF1', 'HNRNPL', 'SND1', 'GTF2F1', 
                          'EIF4G2', 'TIA1', 'TARDBP', 'FXR2', 'HNRNPM','IGF2BP1', 
                          'PUM2', 'FAM120A', 'DDX3X', 'MATR3', 'FUS', 'GRWD1', 'PABPC4'])
    
    k562_RBPnames_low = np.array([ 'MTPAP', 'RBM22', 'DHX30', 'DDX6', 'DDX55', 'TRA2A', 
                          'XRN2', 'U2AF1', 'LSM11', 'ZC3H11A', 'NOLC1', 'KHDRBS1', 
                          'GPKOW', 'DGCR8', 'AKAP1', 'FXR1', 'DDX52', 'AATF']   )   
    
    
    saved_RBP_names = { 'parclip': [pc_RBPnames_high, pc_RBPnames_med, pc_RBPnames_low],
                        'eclip_hg2': [hg2_RBPnames_high1, hg2_RBPnames_high2, hg2_RBPnames_mid1,
                                     hg2_RBPnames_mid2, hg2_RBPnames_low],
                        'eclip_k5':  [k562_RBPnames_high1, k562_RBPnames_high2, k562_RBPnames_mid1,
                                     k562_RBPnames_mid2, k562_RBPnames_low]
                      }
    
    
    return saved_paths, saved_RBP_names
    

def seq_to_1hot(seq,randomsel=True):
    'converts the sequence to one-hot encoding'
	
    seq_len = len(seq)
    seq= seq.upper()
    seq_code = np.zeros((4,seq_len), dtype='int')
    for i in range(seq_len):
        nt = seq[i]
        if nt == 'A':
            seq_code[0,i] = 1
        elif nt == 'C':
            seq_code[1,i] = 1
        elif nt == 'G':
            seq_code[2,i] = 1
        elif nt == 'T':
            seq_code[3,i] = 1
        elif randomsel:
            rn = random.randint(0,3)
            seq_code[rn,i] = 1
    return seq_code

def convert2bed(variants_file, output_dir):
    file_name = variants_file.split('/')[-1]
    print(f'Generating BED file: {output_dir}/{file_name[:-3]}bed')
    #df_variants = pd.read_csv(variants_file, skiprows=97, sep='\t') #hg19_lifted
    df_variants = pd.read_csv(variants_file, sep='\t', names = ['#CHROM', 'POS', 'ID', 'REF', 'ALT']) #hg38

    print(df_variants.head())
    
    
    df_bed = pd.DataFrame()
    df_bed['CHR'] = df_variants['#CHROM'].astype(str)
    df_bed['Start'] = df_variants['POS'].astype(str)
    df_bed['End'] = df_variants['POS'].astype(str)
    df_bed['ID'] = df_variants['ID'].astype(str)
    df_bed['VAR'] = df_variants.apply(lambda x: f'{x["REF"]}/{x["ALT"]}', axis=1)
    df_bed['Strand'] = ['.' for _ in range(len(df_variants))]
                
    df_bed.to_csv(f'{output_dir}/{file_name[:-3]}bed', sep='\t', index=False, header=None)

def deepripe_encode_variant_bedline(bedline,genomefasta,flank_size=75):
    mut_a = bedline[4].split("/")[1]
    strand = bedline[5]
    if len(mut_a)==1:
        wild = pybedtools.BedTool(bedline[0] + "\t" + str(int(bedline[1])-flank_size) + "\t"  + str(int(bedline[2])+flank_size) + "\t" + 
                                  bedline[3] + "\t" + str(mut_a) + "\t" + bedline[5], from_string=True )
        if strand == "-" :
            mut_pos= flank_size
        else:
            mut_pos= flank_size-1
                            
        #wild = pybedtools.BedTool(bedline[0] + "\t" + bedline[1] + "\t" + bedline[2] + "\t" + bedline[3] + "\t"+ bedline[4] + "\t" + bedline[5], from_string=True)
        wild = wild.sequence(fi=genomefasta, tab=True, s=True)
        fastalist = open(wild.seqfn).read().split("\n")
        del fastalist[-1]
        seqs=[fasta.split("\t")[1] for fasta in fastalist]
        mut=seqs[0]
        mut = list(mut)
        mut[mut_pos] = mut_a
        mut = "".join(mut)
        seqs.append(mut)
        encoded_seqs =np.array([seq_to_1hot(seq) for seq in seqs])
        encoded_seqs = np.transpose(encoded_seqs,axes=(0,2,1))
        
        return(encoded_seqs)

def deepripe_score_variant_onlyseq_all(model_group, variant_bed, genomefasta, seq_len=200):
    predictions = { k: [] for k in model_group.keys() }
    counter = 0
    for bedline in variant_bed:
        counter +=1
        encoded_seqs = deepripe_encode_variant_bedline(bedline, 
                                              genomefasta, 
                                              flank_size=(seq_len//2)+2)
        
        if encoded_seqs is not None:
         ## shifting around (seq_len+4) 4 bases
            for choice in model_group.keys():
                avg_score = 0.
                for i in range(4):
                    cropped_seqs = encoded_seqs[:, i:i+seq_len, :]
                    model, _  = model_group[choice]
                    pred = model.predict(cropped_seqs)
                    score = (pred[1]-pred[0])
                    avg_score += score
                predictions[choice].append((avg_score/4))
        else: 
            ## this is for indel groups that cannot be scored e.g. C/CAA
            for choice in model_group.keys():
                _, RBPnames  = model_group[choice]
                predictions[choice].append([np.nan for _ in range(len(RBPnames))])
                
        if counter % 100000 == 0:
                pybedtools.cleanup(remove_all=True)

    return predictions

@click.group()
def cli():
    pass

@cli.command()
@click.option('--debug', is_flag=True)
@click.option('--n-components', type=int, default=100)
@click.argument('deepsea-files', type=click.Path(exists=True), nargs=-1)
@click.argument('annotation-file', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=True))
def deepsea_pca(debug: bool, n_components: int, deepsea_files: List[str],
                annotation_file, out_dir: str):
    logger.info('Reading deepsea file(s)')
    if len(deepsea_files) == 1:
        if debug:
            columns = dd.read_parquet(deepsea_files[0],
                                      engine='pyarrow').columns
            df = dd.read_parquet(deepsea_files[0],
                                 engine='pyarrow',
                                 columns=columns[:20]).compute()
            n_components = 10
        else:
            df = dd.read_parquet(deepsea_files[0], engine='pyarrow').compute()
    else:
        df = pd.concat([
            pd.read_csv(f, index_col=0)
            for f in tqdm(deepsea_files, file=sys.stdout)
        ])

    df = df.rename(columns={
        '#CHROM': 'chrom',
        'POS': 'pos',
        'ID': 'id',
        'REF': 'ref',
        'ALT': 'alt'
    })
    n_deepsea_variants = len(df)
    key_cols = ['chrom', 'pos', 'ref', 'alt']

    logger.info('Sanity check of DeepSEA file IDs')
    all_variants = dd.read_parquet(annotation_file,
                                   engine='pyarrow',
                                   columns=['id'] + key_cols).compute()
    df = pd.merge(all_variants,
                  df,
                  on='id',
                  how='left',
                  validate='1:1',
                  suffixes=('', '_deepsea'))
    try:
        assert len(df) == len(all_variants)
        assert (df['DeepSEA/predict/8988T_DNase_None/diff'].notna().sum() ==
                n_deepsea_variants)
        assert df[['id'] + key_cols].isna().sum().sum() == 0
    except Exception as e:
        print(e)
        import ipdb
        ipdb.set_trace()
    # assert all([
    #     ((df[c] == df[c + '_deepsea'] | df[c + '_deepsea'].isna()).all())
    #     for c in key_cols
    # ])
    df = df.drop(columns=[c + '_deepsea' for c in key_cols])

    logger.info('Adding default of 0 for missing variants')
    df = df.fillna(0)

    logger.info('Extracting matrix for PCA')
    key_df = df[['id'] + key_cols].reset_index(drop=True)
    X = df[[c for c in df.columns if c.startswith('DeepSEA')]].to_numpy()
    del df

    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    del X

    logger.info('Running PCA')
    pca = PCA(n_components=n_components)
    pca.fit(X_std)
    out_path = Path(out_dir)
    with open(out_path / 'pca.pkl', 'wb') as f:
        pickle.dump(pca, f)

    logger.info(f'Projecting rows to {n_components} PCs')
    X_pca = pca.transform(X_std)
    del X_std
    pca_df = pd.DataFrame(
        X_pca, columns=[f'DeepSEA_PC_{i}' for i in range(1, n_components + 1)])
    del X_pca
    pca_df = pd.concat([key_df, pca_df], axis=1)  # , ignore_index=True

    logger.info('Sanity check of results')
    assert pca_df.isna().sum().sum() == 0

    dd.from_pandas(pca_df, chunksize=len(pca_df) // 24).to_parquet(
        out_path / 'deepsea_pca.parquet', engine='pyarrow')

    logger.info('Done')



@cli.command()
@click.option('--n-components', type=int, default=59)
@click.argument('deepripe-file', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=True))
def deepripe_pca(n_components: int, deepripe_file: str, out_dir: str):
    logger.info('Reading deepripe file')
    df = pd.read_csv(deepripe_file)
    df = df.rename(columns={
        '#CHROM': 'chrom',
        'POS': 'pos',
        'ID': 'id',
        'REF': 'ref',
        'ALT': 'alt'
    })
    df = df.drop(columns=['QUAL', 'FILTER', 'INFO'])
    key_df = df[['chrom', 'pos', 'ref', 'alt', 'id']].reset_index(drop=True)

    logger.info('Extracting matrix for PCA')
    X = df[[c for c in df.columns if c not in key_df.columns]].to_numpy()
    del df

    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    del X

    logger.info('Running PCA')
    pca = PCA(n_components=n_components)
    pca.fit(X_std)
    out_path = Path(out_dir)
    with open(out_path / 'pca.pkl', 'wb') as f:
        pickle.dump(pca, f)

    logger.info(f'Projecting rows to {n_components} PCs')
    X_pca = pca.transform(X_std)
    del X_std
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f'DeepRipe_PC_{i}' for i in range(1, n_components + 1)])
    del X_pca
    pca_df = pd.concat([key_df, pca_df], axis=1)
    dd.from_pandas(pca_df, chunksize=len(pca_df) // 24).to_parquet(
        out_path / 'deepripe_pca.parquet', engine='pyarrow')

    logger.info('Done')


@cli.command()
@click.argument('variants-file', type=click.Path(exists=True))
@click.argument('output-dir', type=click.Path(exists=True))
@click.argument('genomefasta',type=click.Path(exists=True))      
@click.argument('pybedtools_tmp_dir', type = click.Path(exists= True))
@click.argument('saved_deepripe_models_path', type = click.Path(exists= True))
@click.argument('saved-model-type',type=str)   
def scorevariants_deepripe(variants_file:str, 
                  output_dir:str,
                  genomefasta:str,
                  pybedtools_tmp_dir:str,
                  saved_deepripe_models_path:str,
                  saved_model_type:str = 'parclip'):
    file_name = variants_file.split('/')[-1]
    bed_file = f'{output_dir}/{file_name[:-3]}bed'
    
    ### setting pybedtools tmp dir
    parent_dir = pybedtools_tmp_dir
    file_stripped = file_name.split('.')[0]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    tmp_path = os.path.join(parent_dir, f'tmp_{file_stripped}_{timestr}')
    os.mkdir(tmp_path)
    pybedtools.set_tempdir(tmp_path)
    
    ### reading variants to score
    df_variants = pd.read_csv(variants_file, sep='\t', header= None, names=["chr", "pos", "Uploaded_variant","ref", "alt"])

    if not os.path.exists(bed_file):
        convert2bed(variants_file, output_dir)  
    
    variant_bed = pybedtools.BedTool(bed_file)
    print(f'Scoring variants for: {bed_file}')
    
    
    ### paths for experiments
    saved_models_dict = { 'parclip':  'parclip_model', 
                          'eclip_hg2': 'eclip_model_encodeHepG2', 
                          'eclip_k5': 'eclip_model_encodeK562', 
                         }
    
    saved_paths, saved_RBP_names = deepripe_get_model_info(saved_models_dict, saved_deepripe_models_path)
    
    #print("Num GPUs Available: ", len(K._get_available_gpus()))

    ## Experiment. parclip
    parclip_RBPs = saved_RBP_names['parclip']
    parclip_models = [ load_model(path_i, 
                       custom_objects={'precision': precision,'recall': recall })  
                       for path_i in saved_paths['parclip']]
    
    parclip_model_group = { keyw: (parclip_models[i], parclip_RBPs[i])
                          for i, keyw in enumerate(['high', 'med', 'low'])}

    ## Experiment. eclip HepG2 cell line
    model_opts = ['high1', 'high2', 'mid1', 'mid2', 'low']
    
    
    eclip_HG_RBPs = saved_RBP_names['eclip_hg2']
    eclip_HG_models = [ load_model(path_i, 
                         custom_objects={'precision': precision,'recall': recall })  
                       for path_i in saved_paths['eclip_hg2']]
    
    eclip_HG_model_group = { keyw: (eclip_HG_models[i], eclip_HG_RBPs[i])
                          for i, keyw in enumerate(model_opts)}
    

    ## Experiment. eclip K562 cell line
    eclip_K5_RBPs = saved_RBP_names['eclip_k5']
    eclip_K5_models = [ load_model(path_i, 
                         custom_objects={'precision': precision,'recall': recall })  
                       for path_i in saved_paths['eclip_k5']]
    
    eclip_K5_model_group = { keyw: (eclip_K5_models[i], eclip_K5_RBPs[i])
                          for i, keyw in enumerate(model_opts)}
    
    
    model_seq_len = { 'parclip':   150, 'eclip_hg2': 200, 'eclip_k5':  200}
    list_of_model_groups = { 'parclip':   parclip_model_group, 
                             'eclip_hg2': eclip_HG_model_group,
                             'eclip_k5':  eclip_K5_model_group
                         }

    ## using only sequence
    current_model_type = list_of_model_groups[saved_model_type]
    predictions = deepripe_score_variant_onlyseq_all(current_model_type, 
                                            variant_bed, 
                                            genomefasta,
                                            seq_len = model_seq_len[saved_model_type])
    
    for choice in current_model_type.keys():
        print(choice)
        _, RBPnames  = current_model_type[choice]
        score_list = predictions[choice]
        score_list = np.asarray(score_list)
        print(f'Output size: {score_list.shape}')
        
        ### write predictions to df
        for ix, RBP_name in enumerate(RBPnames):
            df_variants[RBP_name] = score_list[:, ix]
    print(f'saving file to: {output_dir}/{file_name[:-3]}{saved_model_type}_deepripe.csv')
    df_variants.to_csv(f'{output_dir}/{file_name[:-3]}{saved_model_type}_deepripe.csv', 
                       index=False)



@cli.command()
@click.argument("current_annotation_file", type=click.Path(exists=True))
@click.argument("abs_splice_res_dir", type=click.Path(exists=True))
@click.argument("out_file", type = click.Path())
def get_abscores(current_annotation_file : str, 
                 abs_splice_res_dir : str,
                 out_file : str, 
                 abSplice_score_file : str):
    
    
    current_annotation_file = Path(current_annotation_file)
    logger.info('reading current annotations file')
    current_annotations = pd.read_parquet(current_annotation_file)
    


    if "AbSplice_DNA" in current_annotations.columns:
        if "AbSplice_DNA_old" in current_annotations.columns:
            current_annotations.drop("AbSplice_DNA_old", inplace = True)
        current_annotations = current_annotations.rename(columns = {'AbSplice_DNA': 'AbSplice_DNA_old'})
    ca_shortened = current_annotations[['id', 'gene_id', 'chrom', 'pos', 'ref', 'alt']]
    
    logger.info(ca_shortened.columns)

    abs_splice_res_dir = Path(abs_splice_res_dir)
    
    tissue_agg_function = 'max'
    tissues_to_exclude = ['Testis']
    tissues_to_exclude = []
    ab_splice_agg_score_file = abSplice_score_file

    if not Path(ab_splice_agg_score_file).exists():
        logger.info("creating abSplice score file.. ")
        all_absplice_scores = []
        for chrom_file in os.listdir(abs_splice_res_dir):
            logger.info(f'Reading file {chrom_file}')
            ab_splice_res = pd.read_parquet(abs_splice_res_dir/ chrom_file).reset_index()
            ab_splice_res = ab_splice_res.query('tissue not in @tissues_to_exclude')
            logger.info(f"AbSplice tissues excluded: {tissues_to_exclude}, Aggregating AbSplice scores using {tissue_agg_function}")
            logger.info(f"Number of unique variants {len(ab_splice_res['variant'].unique())}")

            #### aggregate tissue specific ab splice scores
            ab_splice_res = ab_splice_res.groupby(['variant', 'gene_id']).agg({'AbSplice_DNA':tissue_agg_function}).reset_index()

            ab_splice_res[['chrom', 'pos', 'var']] = ab_splice_res['variant'].str.split(":", expand = True)
            
            ab_splice_res[['ref', 'alt']] = ab_splice_res['var'].str.split(">", expand = True)
            
            ab_splice_res['pos'] = ab_splice_res['pos'].astype(int)
            logger.info(f"Number of rows of ab_splice df {len(ab_splice_res)}")
            merged = ab_splice_res.merge(ca_shortened, how = 'left', on = ["chrom", "pos", "ref", "alt", "gene_id"])
            logger.info(f"Number of unique variants(id) in merged {len(merged['id'].unique())}")
            logger.info(f"Number of unique variants(variant) in merged {len(merged['variant'].unique())}")

            all_absplice_scores.append(merged)

            del merged
            del ab_splice_res

            
        all_absplice_scores = pd.concat(all_absplice_scores)
        all_absplice_scores.to_parquet(ab_splice_agg_score_file)


    else:
        logger.info("reading existing abSplice Score file")
        all_absplice_scores = pd.read_parquet(ab_splice_agg_score_file)

    all_absplice_scores = all_absplice_scores[["chrom", "pos", "ref", "alt", "gene_id", "AbSplice_DNA" ]]


    annotations = pd.read_parquet(
        current_annotation_file, engine='pyarrow').drop(
            columns=["AbSplice_DNA"], errors="ignore")
    original_len = len(annotations)

    logger.info('Merging')
    merged = pd.merge(annotations, all_absplice_scores, validate='1:1', how='left', on =["chrom", "pos", "ref", "alt", "gene_id"])

    logger.info('Sanity checking merge')
    assert len(merged) == original_len
    assert merged['censequence_id'].unique().shape[0] == len(merged)

    logger.info(f'Filling {merged["AbSplice_DNA"].isna().sum()} '
                'missing AbSplice values with 0')
    merged['AbSplice_DNA'] = merged['AbSplice_DNA'].fillna(0)

    # logger.info(f'Filling {merged["SpliceAI_delta_score"].isna().sum()} '
    #             'missing SpliceAI values with 0')
    # merged['SpliceAI_delta_score'] = merged['SpliceAI_delta_score'].fillna(0)
    annotation_out_file = out_file
    

    logger.info(f'Writing to {annotation_out_file}')
    dd.from_pandas(merged,
                    chunksize=len(merged) // 24).to_parquet(annotation_out_file,
                                                            engine='pyarrow')

pd.options.mode.chained_assignment = None


logging.basicConfig(format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
                    level='INFO',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)




@cli.command()
@click.option('--n-components', type=int, default=59)
@click.argument('deepripe-file', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=True))
def deepripe_pca(n_components: int, deepripe_file: str, out_dir: str):
    logger.info('Reading deepripe file')
    df = pd.read_csv(deepripe_file)
    df = df.drop(['Unnamed: 0', 'Uploaded_variant'], axis=1)
    print(df.columns)
    df = df.dropna()
    key_df = df[['chr', 'pos', 'ref', 'alt']].reset_index(drop=True)

    logger.info('Extracting matrix for PCA')
    X = df[[c for c in df.columns if c not in key_df.columns]].to_numpy()
    del df
    logger.info('transforming columns to z scores')
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    del X

    logger.info('Running PCA')
    pca = PCA(n_components=n_components)
    pca.fit(X_std)
    out_path = Path(out_dir)
    with open(out_path / 'pca.pkl', 'wb') as f:
        pickle.dump(pca, f)

    logger.info(f'Projecting rows to {n_components} PCs')
    X_pca = pca.transform(X_std)
    del X_std
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f'DeepRipe_PC_{i}' for i in range(1, n_components + 1)])
    del X_pca
    pca_df = pd.concat([key_df, pca_df], axis=1)
    dd.from_pandas(pca_df, chunksize=len(pca_df) // 24).to_parquet(
        out_path / 'deepripe_pca.parquet', engine='pyarrow')

    logger.info('Done')

@cli.command()
@click.argument("annotation_file", type = click.Path(exists = True))
@click.argument("deepripe_pca_file", type = click.Path(exists = True))
@click.argument("out_file", type = click.Path())
def merge_deepripe_pcas(annotation_file:str, deepripe_pca_file:str, out_file:str):
    annotations = pd.read_parquet(annotation_file)
    deepripe_pcas = pd.read_parquet(deepripe_pca_file)
    orig_len= len(annotations)
    deepripe_pcas = deepripe_pcas.rename(columns={"chr":"chrom"})
    merged = annotations.merge(deepripe_pcas, how = "left", on = ["chrom", "pos", "ref", "alt"])
    assert len(merged)== orig_len
    merged.to_parquet(out_file)

@cli.command()
@click.argument("annotation_file", type = click.Path(exists = True))
@click.argument("deepripe_pca_file", type = click.Path(exists = True))
@click.argument("out_file", type = click.Path())
def merge_deepsea_pcas(annotation_file:str, deepripe_pca_file:str, out_file:str):
    annotations = pd.read_parquet(annotation_file)
    deepripe_pcas = pd.read_parquet(deepripe_pca_file)
    orig_len= len(annotations)
    merged = annotations.merge(deepripe_pcas, how = "left", on = ["chrom", "pos", "ref", "alt"])
    assert len(merged)== orig_len
    merged.to_parquet(out_file)


@cli.command()
@click.argument("in_variants", type = click.Path(exists = True))
@click.argument("out_variants", type = click.Path())
def process_annotations(in_variants:str, out_variants:str):
    variant_path = Path(in_variants)
    variants = pd.read_parquet(variant_path)
    
    logger.info("filtering for canonical variants")

    variants = variants.loc[variants.CANONICAL == "YES"]
    variants.rename(columns= {"Gene":"gene_id"}, inplace=True)

    logger.info("aggregating consequences for different alleles")
    
    #combining variant id with gene id
    variants["censequence_id"] = variants['id'].astype(str) + variants['gene_id']
    variants.to_parquet(out_variants)


@cli.command()
@click.option("--included-chromosomes", type=str)
@click.option("--comment-lines", is_flag = True, default = True)
@click.option("--sep", type=str, default = ",")
@click.argument("annotation_dir", type=click.Path(exists=True))
@click.argument("deepripe_name_pattern", type=str)
@click.argument("pvcf-blocks_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
def concatenate_deepripe(
    included_chromosomes: Optional[str],
    sep:str,
    comment_lines:bool,
    annotation_dir: str, 
    deepripe_name_pattern: str, 
    pvcf_blocks_file: str, 
    out_file: str):

    annotation_dir=Path(annotation_dir)

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
    file_paths = [annotation_dir / deepripe_name_pattern.format(chr = p[0], block = p[1] ) for p in pvcf_blocks]
    logger.info("reading in f")
    if comment_lines:
        concatted_file = pd.concat([pd.read_csv(v, comment = "#", sep = sep, ) for v in file_paths])
    else:
        concatted_file = pd.concat([pd.read_csv(v, sep = sep, ) for v in file_paths])
    concatted_file.to_csv(out_file)

@cli.command()
@click.option("--included-chromosomes", type=str)
@click.argument("annotation_dir", type=click.Path(exists=True))
@click.argument("vep_name_pattern", type=str)
@click.argument("variant_file", type = click.Path(exists = True))
#@click.argument("cadd_name_pattern", type=str)
@click.argument("pvcf-blocks_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
def concatenate_annotations(
    included_chromosomes: Optional[str],
    annotation_dir: str, 
    vep_name_pattern: str, 
    variant_file:str,
    pvcf_blocks_file: str, 
    out_file: str):

    annotation_dir=Path(annotation_dir)

    logger.info("Reading variant file")
    variants = pd.read_csv(variant_file, sep = '\t')

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

    logger.info("reading vep files")
    vep_file_paths = [annotation_dir / vep_name_pattern.format(chr = p[0], block = p[1] ) for p in pvcf_blocks]
    vep_colnames = ["Uploaded_variation", "Location", "Allele", "Gene", "Feature", 
                    "Feature_type", "Consequence", "cDNA_position", "CDS_position", 
                    "Protein_position", "Amino_acids", "Codons", "Existing_variation", 
                    "IMPACT", "DISTANCE", "STRAND", "FLAGS", "VARIANT_CLASS", "SYMBOL",
                    "SYMBOL_SOURCE", "HGNC_ID", "BIOTYPE", "CANONICAL", "MANE_SELECT",
                    "MANE_PLUS_CLINICAL", "TSL", "APPRIS", "CCDS", "ENSP", "SWISSPROT", 
                    "TREMBL", "UNIPARC", "UNIPROT_ISOFORM", "RefSeq", "GENE_PHENO", "SIFT", 
                    "PolyPhen", "EXON", "INTRON", "DOMAINS", "miRNA", "HGVSc", "HGVSp", 
                    "HGVS_OFFSET", "AF", "AFR_AF", "AMR_AF", "EAS_AF", "EUR_AF", "SAS_AF", 
                    "gnomADe_AF", "gnomADe_AFR_AF", "gnomADe_AMR_AF", "gnomADe_ASJ_AF", 
                    "gnomADe_EAS_AF", "gnomADe_FIN_AF", "gnomADe_NFE_AF", "gnomADe_OTH_AF", 
                    "gnomADe_SAS_AF", "gnomADg_AF", "gnomADg_AFR_AF", "gnomADg_AMI_AF",
                    "gnomADg_AMR_AF", "gnomADg_ASJ_AF", "gnomADg_EAS_AF", "gnomADg_FIN_AF",
                    "gnomADg_MID_AF", "gnomADg_NFE_AF", "gnomADg_OTH_AF", "gnomADg_SAS_AF", 
                    "MAX_AF", "MAX_AF_POPS", "CLIN_SIG", "SOMATIC", "PHENO", "PUBMED", "MOTIF_NAME", 
                    "MOTIF_POS", "HIGH_INF_POS", "MOTIF_SCORE_CHANGE", "TRANSCRIPTION_FACTORS", 
                    "CADD_PHRED", "CADD_RAW", "SpliceAI_pred", "PrimateAI"]
    
  # cadd_file_paths = [cadd_name_pattern.format(chr = p[0], block = p[1] ) for p in pvcf_blocks]
    
    vep_file = pd.concat([pd.read_csv(v, comment = "#", sep = "\t", names = vep_colnames, dtype = {"STRAND":str, "TSL":str, "GENE_PHENO": str, "CADD_PHRED":str, "CADD_RAW": str}) for v in vep_file_paths])
    # cadd_files = [pd.read_csv(c, sep = "\t") for c in cadd_file_paths]
     

    
    logger.info(f"VEP file shape after concatenation of single files is {vep_file.shape}")
    
   
    
    logger.info("splitting Consequence column")
    vep_file= pd.get_dummies(vep_file, columns = ["Consequence"], prefix_sep="_", prefix = "Consequence")
    logger.info(f"VEP file shape after splitting consequence column is {vep_file.shape}")
    
    

    logger.info("splitting variant name")
    vep_file[["chrom", "pos", "ref", "alt"]] = vep_file["Uploaded_variation"].str.split("_", expand = True)
    vep_file["pos"] = vep_file["pos"].astype(int)
    
    
    logger.info(f"VEP file shape after splitting variant name column is  {vep_file.shape}")
    
    logger.info(f"Shape of variant file is {variants.shape}")
    
    logger.info("merging variants and annotations")
    result = pd.merge(vep_file, variants,  on = ["chrom", "pos", "ref", "alt"])
    logger.info(f"Shape of annotations after merge is {result.shape}")
    
    #cadd_file = pd.concat(cadd_files)
    #combined_annotations = pd.merge(vep_file, cadd_file, on=["CHROM", "POS", "REF", "ALT"])
    combined_annotations = result
    combined_annotations.to_parquet(out_file)
    logger.info("Finished")

if __name__ == '__main__':
    cli()
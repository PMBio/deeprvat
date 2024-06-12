# DeepRVAT configuration and input data


## Configuration file

Configuration for all pipelines is specified in the file `config.yaml`.

In the following, we describe the parameters (both optional and required) that can be specified in these files by way of an [example file](https://github.com/PMBio/deeprvat/blob/main/example/config/deeprvat_input_config.yaml), which we explain block by block.

```
deeprvat_repo_dir: ../..
```

_Required._ This specifies the path to your copy of the DeepRVAT repository.

```
use_pretrained_models: True
pretrained_model_path : ../../pretrained_models
```

These parameters are relevant when using pretrained models. `use_pretrained_models` defaults to `False` if not specified.

```
phenotypes_for_association_testing:
  - Apolipoprotein_A
  - Apolipoprotein_B
  - Calcium
  - Cholesterol
  - Red_blood_cell_erythrocyte_count
  - HDL_cholesterol
  - IGF_1
  - LDL_direct
  - Lymphocyte_percentage
  - Mean_platelet_thrombocyte_volume
  - Mean_corpuscular_volume
  - Mean_reticulocyte_volume
  - Neutrophill_count
  - Platelet_count
  - Platelet_crit
  - Platelet_distribution_width
  - SHBG
  - Standing_height
  - Total_bilirubin
  - Triglycerides
  - Urate
```

_Required._ This specifies the list of phenotypes on which to perform association testing. These may be the same or different from those used for training. All phenotypes must be present in the file specified by `phenotype_filename`.

```
phenotypes_for_training:
  - Apolipoprotein_A
  - Apolipoprotein_B
  - Calcium
  - Cholesterol
  - Red_blood_cell_erythrocyte_count
```

_Optional._ This specifies the list of phenotypes on which to perform training. These may be the same or different from those used for association testing. All phenotypes must be present in the file specified by `phenotype_filename`.

If omitted, this defaults to be equal to `phenotypes_for_association_testing`. 

```
#File paths of necessary input files to DeepRVAT
gt_filename: genotypes.h5
variant_filename: variants.parquet
phenotype_filename: phenotypes.parquet
annotation_filename: annotations.parquet
gene_filename: protein_coding_genes.parquet
```

_Required._ These specify the paths for required input data files. See [Input data formats](#input-data-formats) for more details.

```
rare_variant_annotations:
  - MAF_MB
  - CADD_raw
  - sift_score
  - polyphen_score
  - Consequence_splice_acceptor_variant
  - Consequence_splice_donor_variant
  - Consequence_stop_gained
  - Consequence_frameshift_variant
  - Consequence_stop_lost
  - Consequence_start_lost
  - Consequence_inframe_insertion
  - Consequence_inframe_deletion
  - Consequence_missense_variant
  - Consequence_protein_altering_variant
  - Consequence_splice_region_variant
  - condel_score
  - DeepSEA_PC_1
  - DeepSEA_PC_2
  - DeepSEA_PC_3
  - DeepSEA_PC_4
  - DeepSEA_PC_5
  - DeepSEA_PC_6
  - PrimateAI_score
  - AbSplice_DNA
  - DeepRipe_plus_QKI_lip_hg2
  - DeepRipe_plus_QKI_clip_k5
  - DeepRipe_plus_KHDRBS1_clip_k5
  - DeepRipe_plus_ELAVL1_parclip
  - DeepRipe_plus_TARDBP_parclip
  - DeepRipe_plus_HNRNPD_parclip
  - DeepRipe_plus_MBNL1_parclip
  - DeepRipe_plus_QKI_parclip
  - SpliceAI_delta_score
  - alphamissense
```

_Required when executing training._ Specifies rare variant annotations to use in the DeepRVAT model. Each annotation must be present as a column in the file given by `annotation_filename`. 

```
covariates: #x_phenotypes
  - age
  - age2
  - age_sex
  - genetic_sex
  - genetic_PC_1
  - genetic_PC_2
  - genetic_PC_3
  - genetic_PC_4
  - genetic_PC_5
  - genetic_PC_6
  - genetic_PC_7
  - genetic_PC_8
  - genetic_PC_9
  - genetic_PC_10
  - genetic_PC_11
  - genetic_PC_12
  - genetic_PC_13
  - genetic_PC_14
  - genetic_PC_15
  - genetic_PC_16
  - genetic_PC_17
  - genetic_PC_18
  - genetic_PC_19
  - genetic_PC_20
```

_Required._ This specifies the list of covariates to use, and are the same for association testing and (if executed) training. Each must correspond to a column in the file given by `phenotype_filename`.

```
association_testing_data_thresholds:
  MAF: "< 1e-3"
  CADD_PHRED: "> 5"

training_data_thresholds:
  MAF: "< 1e-2"
  CADD_PHRED: "> 5"
```

_Optional._ Parameters specified here are used to select variants used for association and training, respectively. They are passed into a `pandas` query and joined by `and`. E.g., if `df` contains the full set of rare variants, then those used for association testing will be selected by `df.query("MAF < 1e-3 and CADD_PHRED > 5")`. All keys used in these blocks must be present in the file given in `annotation_filename`.

```
#Seed Gene Baseline data settings
seed_gene_results: #baseline_results
  result_dirs:
      -
          base: baseline_results
          type: plof/burden
      -
          base: baseline_results
          type: missense/burden
      -
          base: baseline_results
          type: plof/skat
      -
          base: baseline_results
          type: missense/skat

  correction_method: Bonferroni
  alpha_seed_genes: 0.05
```

_Required when running training._ This set of parameters configures the selection of seed genes and is only required when training is executed. 

`result_dirs` specifies the locations of results from the seed gene discovery pipeline. Each list element specifies the `base` (top-level directory of the results from seed gene discovery) and `type` (in the form of `variant_file/test_method`).

`correction method` specifies the multiple testing correction method to use on seed gene _p_-values, and `alpha_seed_genes` is the maximum threshold for corrected _p_-values to select seed genes.


If you custom seed genes should be used, they can be stored like `baseline_results/{phenotype}/combined_baseline/eval/burden_associations.parquet` and setting
```
seed_gene_results: #baseline_results
  result_dirs:
    - base: baseline_results
      type: combined_baseline
```
The `burden_associations.parquet` must be be provided for each `{phenotype}` in `phenotypes_for_training`, containing the columns `gene` (gene id as assigned in `gene_filename`) and `pval` (see `deeprvat/example/baseline_results`). To include all genes in `burden_associations.parquet` as seed genes, set `alpha_seed_genes` arbitraily high (e.g., `1000`). 

```
#DeepRVAT training settings
training:
  pl_trainer: #PyTorch Lightening trainer settings
      gpus: 1
      precision: 16
      min_epochs: 50
      max_epochs: 1000
      log_every_n_steps: 1
      check_val_every_n_epoch: 1
  early_stopping: #PyTorch Lightening Early Stopping Criteria
      mode: min
      patience: 3
      min_delta: 0.00001
      verbose: True
```

_Required when running training._ Parameters here configure training. The items under `pl_trainer` are passed as keyword arguments to PyTorch Lightning's `Trainer.fit()`. Items under `early_stopping` are passed as keyword arguments to the initialization of a PyTorch Lightning `EarlyStopping` callback.

```  
#DeepRVAT model settings
n_repeats: 30
y_transformation: quantile_transform
```

_Required._ `n_repeats` specifies the number of DeepRVAT models used in association testing and/or trained as part of the model ensemble.

_Optional._ `y_transformation` optionally specifies a transformation applied to phenotypes during association testing and (when executed) training. Possible values are `standardize` and `quantile_transform`.

```
# Results evaluation settings
evaluation:
  correction_method: Bonferroni
  alpha: 0.05
```

_Required._ Specifies the multiple-testing correction method and maximum corrected _p_-value for significance in evaluation of association testing results.

```
# Subsetting samples for training or association testing
#sample_files:
#  training: training_samples.pkl
#  association_testing: association_testing_samples.pkl
```
_Optional._ 

Subsetting of samples used during training and association testing. If no sample files are provided, all samples in the index of `phenotype_filename` are used for training and association testing. 
Only `.pkl` files containing a list of sample IDs (string) are supported at the moment.

```
#Additional settings if using the CV pipeline
cv_options:
  cv_exp: False
  #cv_path: sample_files
  #n_folds: 5
```

_Optional._

`cv_exp` is a flag determining whether the cross-validation pipeline is run for training and association tresting. It defaults to `False`.

`cv_path` must be specified if `cv_exp` is `True`. It specifies the path to sample files; see [Input data formats](#input-data-formats) for more.

`n_folds` is the number of folds to use for CV. Defaults to 5.

```
#Additional settings if using the REGENIE integration
regenie_options:
  regenie_exp: True
  gtf_file: gencode.v38.basic.annotation.gtf.gz
  step_1:
      bgen: imputation.bgen
      snplist: imputation.snplist
      bsize: 1000
      options:
          - "--sample imputation.sample"
          - "--qt"
  step_2:
      bsize: 400
      options:
          - "--qt"
```

_Optional._

`regenie_exp`: When `True`, REGENIE is used for association testing, and the parameters `gtf_file`, `step_1`, and `step_2` are required. Otherwise, SEAK is used for association testing, and those parameters are optional.

`gtf_file` specifies a GTF file, which must contain all genes present in the file given by `gene_filename`. We recommend using a GTF file from GENCODE.

In `step_1` and `step_2`, the parameters `bgen`, `snplist` and `bsize` control the corresponding options in REGENIE. Additional options for each step of REGENIE may (but need not be) specified as a list under `options`.


## Input data formats


### Genotype file

Contained in the file given by `gt_filename`. Output of the preprocessing pipeline. HDF5 format. The file has three datasets. `samples` is a byte-encoded array of sample ID strings. `variant_matrix` and `genotype_matrix` encode genotypes in a custom sparse format. Both matrices are the same size, with samples along the rows and variants along the columns. `variant_matrix` gives an integer variant ID (assigned by the preprocessing pipeline), and the corresponding cell of `genotype_matrix` is given by 1 or 2. To account for samples having different total number of variants, the rows of each matrix are padded with -1.


### Variant file

Contained in the file given by `variant_filename`. Output of the preprocessing pipeline. Parquet format. The fiel contains metadata about variants, namely `id` (an integer variant ID assigned by the preprocessing pipeline and corresponding to those from the genotype file), `chrom`, `pos`, `ref` and `alt`. 


### Phenotype file

Contained in the file given by `phenotype_filename`. Must be created by the user. Parquet format. The index (named `samples`) should contain string sample IDs corresponding to those in the genotype file. Additional columns contain covariates and target phenotypes, with column names corresponding to those in the lists `covariates`, `phenotypes_for_association_testing`, and `phenotypes_for_training` in the configuration file. All covariate and phenotype columns should have numerical data types.


### Annotation file

Contained in the file given by `annotation_filename`. Output of the annotation pipeline. Parquet format. Required columns:
* `id`: An integer ID corresponding to those in the genotype and variant files
* `MAF`: Minor allele frequency in a numerical data type

Additional columns contain rare variant annotations used for filtering variants and/or gene impairment module training and evaluation. These must have numerical datatypes.


### Gene file

Contained in the file given by `gene_filename`. Must be created by the user. Parquet format. Required columns:
* `id`: An integer ID which may be chosen by the user
* `gene`: A string gene descriptor which may be chosen by the user

An arbitrary number of additional columns may also be contained in the file.


# Training and association testing with DeepRVAT

We have developed multiple modes of running DeepRVAT to suit your needs. Below are listed various running setups that entail just training DeepRVAT, using pretrained DeepRVAT models for association testing, using precomputed burdens for association testing, including REGENIE in training and association testing and also combinations of these scenarios. The general procedure is to have the relevant input data for a given setup appropriately prepared, which may include having already completed the [preprocessing pipeline](https://deeprvat.readthedocs.io/en/latest/preprocessing.html) and [annotation pipeline](https://deeprvat.readthedocs.io/en/latest/annotations.html).


(common requirements for input data)=
## Input data: Common requirements for all pipelines

An example overview of what your `experiment` directory should contain can be seen here: 
`[path_to_deeprvat]/example/`

Replace `[path_to_deeprvat]` with the path to your clone of the repository.
Note that the example data contained within the example directory is randomly generated, and is only suited for testing.

- `genotypes.h5`
contains the genotypes for all samples in a custom sparse format. The sample ids in the `samples` dataset are the same as in the VCF files the `genotypes.h5` has been read from. 
This is output by the preprocessing pipeline. Instructions [here](https://deeprvat.readthedocs.io/en/latest/preprocessing.html).

- `variants.parquet`
contains variant characteristics (`chrom`, `pos`, `ref`, `alt`) and the assigned variant `id` for all unique variants in `genotypes.h5`. This 
is output from the input VCF files using the preprocessing pipeline. Instructions [here](https://deeprvat.readthedocs.io/en/latest/preprocessing.html).

- `annotations.parquet`
contains the variant annotations for all variants in `variants.parquet`, which is an output from the annotation pipeline. Each variant is identified by its `id`. Instructions [here](https://deeprvat.readthedocs.io/en/latest/annotations.html).

- `protein_coding_genes.parquet`
Maps the integer `gene_id` used in `annotations.parquet` to standard gene IDs (EnsemblID and HGNC gene name). This is an output from the annotation pipeline. Instructions [here](https://deeprvat.readthedocs.io/en/latest/annotations.html).

- `config.yaml`
contains the configuration parameters for setting phenotypes, training data, model, training, and association data variables.

- `phenotypes.parquet` contains the measured phenotypes for all samples (see `[path_to_deeprvat]/example/`). The row index must be the sample id as strings (same ids as used in the VCF file) and the column names the phenotype name. Phenotypes can be quantitative or binary (0,1). Use `NA` for missing values. 
Samples missing in `phenotypes.parquet` won't be used in DeepRVAT training/testing. The user must generate this file as it's not output by the preprocessing/annotation pipeline.
This file must also contain all covariates that should be used during training/association testing (e.g., genetic sex, age, genetic principal components).

- `baseline_results`
directory containing the results of the seed gene discovery pipline. Insturctions [here](seed_gene_discovery.md)


(common configuration parameters)=
## Configuration file: Common parameters

The `config.yaml` file located in your `experiment` directory contains the configuration parameters of key sections: phenotypes, baseline_results, training_data, and data. It also allows to set many other configurations detailed below. 

`config['training_data']` contains the relevant specifications for the training dataset creation.

`config['data']` contains the relevant specifications for the association dataset creation.

### Baseline results
`config['baseline_results']` specifies paths to results from the seed gene discovery pipeline (Burden/SKAT test with pLoF and missense variants). When using the seed gene discovery pipeline provided with this package, simply link the directory as 'baseline_results' in the experiment directory without any further changes.

If you want to provide custom baseline results (already combined across tests), store them like `baseline_results/{phenotype}/combined_baseline/eval/burden_associations.parquet` and set the `baseline_results` in the config to 
```
- base: baseline_results
  type: combined_baseline
```
Baseline files have to be provided for each `{phenotype}` in `config['training']['phenotypes']`. The `burden_associations.parquet` must have the columns `gene` (gene id as assigned in `protein_coding_genes.parquet`) and `pval` (see `[path_to_deeprvat]/example/baseline_results`). 

<!--- *TODO* add that seed gene config can be set via the `config['phenotypes']` --->


### Phenotypes
`config['phenotypes]` should consist of a complete list of phenotypes. To change phenotypes used during training, use `config['training']['phenotypes']`. The phenotypes that are not listed under `config['training']['phenotypes']`, but are listed under 
`config['phenotypes]` will subsequently be used only for association testing.
All phenotypes listed either in `config['phenotypes']` or `config['training']['phenotypes']` have to be in the column names of `phenotypes.parquet`.


### Customizing the input data via the config file

#### Data transformation 

The pipeline supports z-score standardization (`standardize`) and quantile transformation (`quantile_transform`) as transformation to of the target phenotypes. It has to be set in `config[key]['dataset_config']['y_transformation]`, where `key` is `training_data` or `data` to transform the training data and association testing data, respectively. 

For the annotations and the covariates, we allow standardization via `config[key]['dataset_config']['standardize_xpheno'] = True` (default = True) and `config[key]['dataset_config']['standardize_anno'] = True` (default = False). 

If custom transformations are whished, we recommend to replace the respective columns in `phenotypes.parquet` or `annotations.parquet` with the transformed values. 

#### Variant annotations
All variant anntations that should be included in DeepRVAT's variant annotation vectors have to be listed in `config[key]['dataset_config']['annotations']` and `config[key]['dataset_config']['rare_embedding']['config']['annotations']` (this will be simplified in future). Any annotation that is used for variant filtering `config[key]['dataset_config']['rare_embedding']['config']['thresholds']` also has to be included in `config[key]['dataset_config']['annotations']`.

#### Variant minor allele frequency filter

To filter for variants with a MAF below a certain value (e.g., UKB_MAF < 0.1%), use:
`config[key]['dataset_config']['rare_embedding']['config']['thresholds']['UKB_MAF'] = "UKB_MAF < 1e-3"`. In this example,  `UKB_MAF` represents the MAF column from `annotations.parquet` here denoting MAF in the UK Biobank. 

#### Additional variant filters
Additional variant filters can be added via `config[key]['dataset_config']['rare_embedding']['config']['thresholds'][{anno}] = "{anno} > X"`. For example, `config['data]['dataset_config']['rare_embedding']['config']['thresholds']['CADD_PHRED'] = "CADD_PHRED > 5"` will only include variants with a CADD score > 5 during association testing. Mind that all annotations used in the `threshold` section also have to be listed in `config[key]['dataset_config']['annotations']`.

#### Subsetting samples
To specify a sample file for training or association testing, use: `config[key]['dataset_config']['sample_file']`. 
Only `.pkl` files containing a list of sample IDs (string) are supported at the moment.
For example, if DeepRVAT training and association testing should be done on two separate data sets, you can provide two sample files `training_samples.pkl` and `test_samples.pkl` via `config['training_data']['dataset_config']['sample_file] = training_samples.pkl` and `config['data']['dataset_config']['sample_file] = test_samples.pkl`. 

## Association testing using precomputed burdens

_Coming soon_

<!--- *TODO:* With and without REGENIE --->

(Association_testing)=
## Association testing using pretrained models

If you already have a pretrained DeepRVAT model, we have setup pipelines for runing only the association testing stage. This includes creating the association dataset files, computing burdens, regression, and evaluation. 


### Input data
The following files should be contained within your `experiment` directory: 
- `config.yaml`
- `genotypes.h5`
- `variants.parquet`
- `annotations.parquet`
- `phenotypes.parquet`
- `protein_coding_genes.parquet`

### Configuration file
The annotations in `config['data']['dataset_config']['rare_embedding']['config']['annotations']` must be the same (and in the same order) as in `config['data']['dataset_config']['rare_embedding']['config']['annotations']` from the pre-trained model. 
If you use the pre-trained DeepRVAT model provided with this package, use `config['data']['dataset_config']['rare_embedding']['config']['annotations']` from the `[path_to_deeprvat]/example/config.yaml` to ensure the ordering of annotations is correct. 

### Running the association testing pipeline with REGENIE

_Coming soon_

<!---

#### Input data
For running with REGENIE, in addition to the default input data, the following REGENIE specific files should also be included in your `experiment` directory:


To run REGENIE Step 1
- `.sample` Inclusion file that lists individuals to retain in the analysis
- `.sniplist` Inclusion file that lists IDs of variants to keep
- `.bgen` input genetic data file
- `.bgen.bgi` index bgi file corresponding to input BGEN file

For these REGENIE specific files, please refer to the [REGENIE documentation](https://rgcgithub.github.io/regenie/).

For running REGENIE Step 2:
- `gtf file` gencode gene annotation gtf file 
- `keep_samples.txt` (optional file of samples to include)
- `protein_coding_genes.parquet`

#### Config file

Use the `[path_to_deeprvat]/example/config_regenie.yaml` as `config.yaml` which includes REGENIE specific parameters. 
You can set any parameter explained in the [REGENIE documentation](https://rgcgithub.github.io/regenie/) via this config.
Most importantly, for association testing of binary traits use:
```
step_2:
        options:
            - "--bt"
            - "--firth --approx --pThresh 0.01"

```
and for quantitative traits:
```
step_2:
        options:
            - "--qt"
```

#### Run REGENIE


```
cd experiment
ln -s [path_to_deeprvat]/pretrained_models
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/association_testing_pretrained.snakefile
```


#### Testing multiple sub-chohorts
For testing multiple sub-cohorts, remember that REGENIE Step 1 (compute intense) only needs to be executed once per sample and phenotype. We suggest running REGENIE Step 1 on all samples and phenotypes initially and then linking the output as `regenie_output/step1/` in each experiment directory for testing a sub-cohort.

Samples to be considered when testing sub-cohorts can be provided via `keep_samples.txt` which look like 

``` 
12345 12345
56789 56789
````
for keeping two samples with ids `12345` and `56789`

### Running the association testing pipeline with SEAK

```shell
cd experiment
ln -s [path_to_deeprvat]/pretrained_models
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/association_testing_pretrained.snakefile
```

--->

## Training
To run only the training stage of DeepRVAT, comprised of training data creation and running the DeepRVAT model, we have setup a training pipeline.

### Input data
The following files should be contained within your `experiment` directory: 
- `config.yaml`
- `genotypes.h5`
- `variants.parquet`
- `annotations.parquet`
- `phenotypes.parquet`
- `protein_coding_genes.parquet`
- `baseline_results` directory where `[path_to_deeprvat]/pipelines/seed_gene_discovery.snakefile` has been run 

### Configuration file
Changes to the model architecture and training parameters can be made via `config['training']`, `config['pl_trainer']`, `config['early_stopping']`, `config['model']`. 
Per default, DeepRVAT scores are ensembled from 6 models. This can be changed via `config['n_repeats']`. 


### Running the training pipeline
```shell
cd experiment
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/run_training.snakefile
```


## Training and association testing using cross-validation

DeepRVAT offers a CV scheme, where it's trained on all samples except those in the held-out fold. Then, it computes gene impairment scores for the held-out samples using models that excluded them. This is repeated for all folds, yielding DeepRVAT scores for all samples. 

### Input data and configuration file
The following files should be contained within your `experiment` directory: 
- `config.yaml`
- `genotypes.h5`
- `variants.parquet`
- `annotations.parquet`
- `phenotypes.parquet`
- `protein_coding_genes.parquet`
- `baseline_results` directory
- `sample_files` provides training and test samples for each cross-validation fold as pickle files. 

### Config and sample files
For running 5-fold cross-validation include the following configuration in the config: 
``` 
cv_path: sample_files
n_folds: 5
```
Provide sample files structured as `sample_files/5_fold/samples_{split}{fold}.pkl`, where `{split}` represents train/test and `{fold}` is a number from `0 to 4`.

### Run the pipeline
```shell
cd experiment
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/cv_training/cv_training_association_testing.snakefile
```

<!---
## Training and association testing without CV

To run the full pipeline from training through association testing without CV, use the below procedure. This includes training and association testing dataset generation, DeepRVAT model training, computation of burdens, regression and evaluation. 

### Input data and configuration file
The following files should be contained within your `experiment` directory: 
- `config.yaml`
- `genotypes.h5`
- `variants.parquet`
- `annotations.parquet`
- `phenotypes.parquet`
- `protein_coding_genes.parquet`
- `baseline_results` directory

### Running the training and association testing pipelinewith SEAK

```shell
cd experiment
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/training_association_testing.snakefile
```

#### Running with REGENIE

```shell
cd experiment
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/training_association_testing_regenie.snakefile
```

-->

## Running only a portion of any pipeline
The snakemake pipelines outlined above are compromised of integrated common workflows. These smaller snakefiles which breakdown specific pipelines sections are in the following directories:
- `[path_to_deeprvat]/pipeline/association_testing` contains snakefiles breaking down stages of the association testing.
- `[path_to_deeprvat]/pipeline/cv_training` contains snakefiles used to run training in a cross-validation setup.
- `[path_to_deeprvat]/pipeline/training` contains snakefiles used in setting up deepRVAT training.

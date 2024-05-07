# Training and association testing with DeepRVAT
We have developed multiple flavors of running DeepRVAT to suite your needs. Below lists various running setups that entail just training DeepRVAT, using pretrained DeepRVAT models for association testing, using precomputed burdens for association testing, including REGENIE in training and association testing and also combinations of these scenarios. The general procedure is to have the relevant input data for a given setup appropriately prepared, which may include having already completed the [preprocessing pipeline](https://deeprvat.readthedocs.io/en/latest/preprocessing.html) and [annotation pipeline](https://deeprvat.readthedocs.io/en/latest/annotations.html).

*TODO* also add CV training ?

## Installation
First the deepRVAT repository must be cloned in your `experiment` directory and the corresponding environment activated. Instructions are [here](installation.md) to setup the deepRVAT repository.

## Input data: Common requirements for all pipelines
An example overview of what your `experiment`` directory should contain can be seen here: 
`[path_to_deeprvat]/example/`

Replace `[path_to_deeprvat]` with the path to your clone of the repository.
Note that the example data contained within the example directory is randomly generated, and is only suited for testing.

- `genotypes.h5`
contains the  *TODO*  Which is an output from the preprocessing pipeline. Instructions [here](https://deeprvat.readthedocs.io/en/latest/preprocessing.html).

- `variants.parquet`
contains the list of variants from the input vcf files, which is an output from the preprocessing pipeline. Instructions [here](https://deeprvat.readthedocs.io/en/latest/preprocessing.html).

- `annotations.parquet`
contains all the variant annotations, which is an output from the annotation pipeline. Instructions [here](https://deeprvat.readthedocs.io/en/latest/annotations.html).

- `protein_coding_genes.parquet`
contains the IDs to all the protein coding genes, which is an output from the annotation pipeline. Instructions [here](https://deeprvat.readthedocs.io/en/latest/annotations.html).

- `config.yaml`
contains the configuration parameters for setting phenotypes, training data, model, training, and association data variables.

- `phenotypes.parquet`
contains the *TODO*

- `[path_to_deeprvat]/example/baseline_results`
directory containing the results of the seed gene discovery pipline. Insturctions [here](seed_gene_discovery.md)

## Configuration file: Common parameters
*TODO:* Describe common parameters, give example

The `config.yaml` file located in your `experiment` directory contains the configuration parameters of key sections: phenotypes, baseline_results, training_data, and data.

`config['phenotypes]` should consist of a complete list of phenotypes. To adjust only those phenotypes that should be used in training, add the phenotype names as a list under `config['training']['phenotypes']`. The phenotypes that are not listed under `config['training']['phenotypes']`, but are listed under 
`config['phenotypes]` will subsequently be used only for association testing.

*TODO* baseline results

`config['training_data']` contains the relevant specifications for the training dataset creation.

`config['data']` contains the relevant specifications for the association dataset creation.

## Training
To run only the training stage of DeepRVAT, comprised of training data creation and running the deepRVAT model, we have setup a training pipeline.

### Input data
The following files should be contained within your `experiment` directory: 
- `config.yaml`
- `genotypes.h5`
- `variants.parquet`
- `annotations.parquet`
- `phenotypes.parquet`
- `protein_coding_genes.parquet`
- `baseline_results` directory

### Configuration file

### Running the training pipeline
```shell
cd experiment
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/run_training.snakefile
```

## Association testing
If you already have a pretrained DeepRVAT model, we have setup pipelines for runing only the association testing stage. This includes creating the association dataset files, computing burdens, regression, and evaluation. 

### Input data
The following files should be contained within your `experiment` directory: 
- `config.yaml`
- `genotypes.h5`
- `variants.parquet`
- `annotations.parquet`
- `phenotypes.parquet`
- `protein_coding_genes.parquet`
- `baseline_results` directory

### Configuration file

### Running the association testing pipeline with REGENIE
*TODO*
For running with REGENIE, in addition the input data, the following REGENIE specific files should also be included in your `experiment` directory:
- `.sample` file containing the sample ID, genetic sex
- `.sniplist` file containing *TODO*
- `.bgen`
- `.bgen.bgi`

For the REGENIE specific files, please refer to the [REGENIE documentation](https://rgcgithub.github.io/regenie/).

```shell
cd experiment
ln -s [path_to_deeprvat]/pretrained_models
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/association_testing_pretrained.snakefile
```

### Running the association testing pipeline with SEAK
```shell
cd experiment
ln -s [path_to_deeprvat]/pretrained_models
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/association_testing_pretrained.snakefile
```

### Association testing using precomputed burdens
*TODO:* With and without REGENIE

## Training and association testing with a combined pipeline
To run the full pipeline from training through association testing, use the below procedure. This includes training and association testing dataset generation, deepRVAT model training, computation of burdens, regression and evaluation. 

### Input data and configuration file
The following files should be contained within your `experiment` directory: 
- `config.yaml`
- `genotypes.h5`
- `variants.parquet`
- `annotations.parquet`
- `phenotypes.parquet`
- `protein_coding_genes.parquet`
- `baseline_results` directory

### Running the training and association testing pipeline
The process is as follows:
```shell
cd experiment
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/training_association_testing.snakefile
```

#### Running with REGENIE
*TODO:* 
```shell
cd experiment
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/training_association_testing_regenie.snakefile
```

## Running only a portion of any pipeline
The snakemake pipelines outlined above are compromised of integrated common workflows. These smaller snakefiles which breakdown specific pipelines sections are in the following directories:
- `[path_to_deeprvat]/pipeline/association_testing` contains snakefiles breakingdown stages of the association testing.
- `[path_to_deeprvat]/pipeline/cv_training` contains snakefiles used to run training in a cross-validation setup.
- `[path_to_deeprvat]/pipeline/training` contains snakefiles used in setting up deepRVAT training.

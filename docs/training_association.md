# Training and association testing

For using the pretrained DeepRVAT model provided as part of the package, or a custom pretrained model, we have setup pipelines for running only the association testing stage. This includes creating the association dataset files, computing gene impairment scores, regression, and evaluation. 


## Configuration and input files

Configuration parameters must be specified in `deeprvat_input_config.yaml`. For details on the meanings of the parameters and the format of input files, see [here](input_data).

You must specify 
```
use_pretrained_models: True
```
in your configuration file.

The following parameters specify the locations of required input files:
 
```
pretrained_model_paths
gt_filename
variant_filename
phenotype_filename
annotation_filename
gene_filename
seed_gene_results
```

These parameters specify options for running DeepRVAT. Those marked `(optional)` have default values; see [here](input_data) for details.

```
phenotypes_for_association_testing
phenotypes_for_training
rare_variant_annotations
covariates
training
n_repeats
evaluation
y_transformation (optional)
association_testing_data_thresholds (optional)
training_data_thresholds (optional)
cv_options (required only when running cross validation)
```

Note that the file specified by `annotation_filename` must contain a column corresponding to each annotation in the list `rare_variant_annotations` in `deeprvat_input_config.yaml`. 


## Executing the pipeline

```
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/association_testing_pretrained.snakefile
```

Replace `[path_to_deeprvat]` with the path to your copy of the DeepRVAT repository.


## Using cross validation

_Coming soon_


## Running the association testing pipeline with REGENIE

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

# Association testing using pretrained DeepRVAT models

For using the pretrained DeepRVAT model provided as part of the package, or a custom pretrained model, we have setup pipelines for running only the association testing stage. This includes creating the association dataset files, computing gene impairment scores, regression, and evaluation. 

_**Important note:**_ DeepRVAT currently supports association testing using REGENIE or SEAK. Because REGENIE is actively maintained and gives better control for ancestry effects, as well as other aspects of genetic background, association testing with SEAK is _deprecated_ and will be removed in a future version. Note especially that for binary phenotypes, REGENIE _must_ be used.


## Association testing with REGENIE

For specifics on input files and options for REGENIE steps 1 and 2, please refer to the [REGENIE documentation](https://rgcgithub.github.io/regenie/).

### Configuration and input files

Configuration parameters must be specified in `deeprvat_input_pretrained_models_config.yaml` (rename the example file). For details on the meanings of the parameters and the format of input files, see [here](input_data).

For an example, refer to [this configuration file](https://github.com/PMBio/deeprvat/blob/main/example/config/deeprvat_input_config_regenie.yaml) (rename or link it to `deeprvat_input_pretrained_models_config.yaml` in your working directory), which includes REGENIE-specific parameters for steps 1 and 2. Importantly, specify
```
regenie_options:
  regenie_exp: True
  gtf_file: path/to/annotation_file.gtf.gz
```

To use pretrained models, you must specify `use_pretrained_models: True` in your `deeprvat_input_pretrained_models_config.yaml` configuration file. Additionally, provide the path to pretrained models (an output of the training pipeline) in the parameter `pretrained_model_path`. Within the `pretrained_model_path` directory, there must be a `config.yaml` file in that directory with the following set of specified keys that were used for training the pretrained models; `rare_variant_annotations`, `training_data_thresholds`, and `model` . See [example file](https://github.com/PMBio/deeprvat/blob/main/pretrained_models/model_config.yaml).

Below outlines the configuration parameters specified in `deeprvat_input_pretrained_models_config.yaml`.

The following parameters specify the locations of required input files:

```
gt_filename
variant_filename
phenotype_filename
annotation_filename
gene_filename
```

These parameters specify options for running DeepRVAT. Those marked `(optional)` have default values; see [here](input_data) for details.

```
phenotypes_for_association_testing
covariates
n_repeats
evaluation
y_transformation (optional)
association_testing_data_thresholds (optional)
cv_options (optional)
```

Note that the file specified by `annotation_filename` must contain a column corresponding to each annotation in the list `rare_variant_annotations` from `deeprvat/pretrained_models/model_config.yaml`. 

You can set any parameter explained in the [REGENIE documentation](https://rgcgithub.github.io/regenie/) via the config.
Most importantly, for association testing of binary traits use `--bt` in Step 2. For imbalanced traits, we also recommend Firth logistic regression:
```
step_2:
        options:
            - "--bt"
            - "--firth --approx --pThresh 0.01"

```
For quantitative traits:
```
step_2:
        options:
            - "--qt"
```

### Run REGENIE


```
cd experiment
ln -s [path_to_deeprvat]/pretrained_models
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/association_testing_pretrained.snakefile
```


<!-- ### Testing multiple sub-cohorts -->

<!-- For testing multiple sub-cohorts, remember that REGENIE Step 1 (compute intense) only needs to be executed once per sample and phenotype. We suggest running REGENIE Step 1 on all samples and phenotypes initially and then linking the output as `regenie_output/step1/` in each experiment directory for testing a sub-cohort. -->

<!-- Samples to be considered when testing sub-cohorts can be provided via `keep_samples.txt` which look like  -->

<!-- ```  -->
<!-- 12345 12345 -->
<!-- 56789 56789 -->
<!-- ```` -->
<!-- for keeping two samples with ids `12345` and `56789` -->

## Association testing with SEAK

### Configuration file

Follow the instructions as above, but specify
```
regenie_options:
  regenie_exp: False
```

For an example, see [this configuration file](https://github.com/PMBio/deeprvat/blob/main/example/config/deeprvat_input_pretrained_models_config.yaml)

### Executing the pipeline

```
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/association_testing_pretrained.snakefile
```

Replace `[path_to_deeprvat]` with the path to your copy of the DeepRVAT repository.




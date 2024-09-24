# Quick start


## Customize pipelines

Before running any of the snakefiles, you may want to adjust the number of threads used by different steps in the pipeline. To do this, modify the `threads:` property of a given rule.

If you are running on a computing cluster, you will need a [profile](https://github.com/snakemake-profiles) and may need to add `resources:` directives to the snakefiles.


## Run the preprocessing pipeline on your VCF files

Instructions [here](preprocessing.md)


## Annotate variants

Instructions [here](annotations.md)


## Example DeepRVAT runs

In each case, replace `[path_to_deeprvat]` with the path to your clone of the repository.

Note that the example data used here is randomly generated, and so is only suited for testing whether the `deeprvat` package has been correctly installed.


### Run the association testing pipeline with pretrained models

```shell
DEEPRVAT_REPO_PATH="[path_to_deeprvat]"
mkdir deeprvat_associate
cd deeprvat_associate
ln -s "$DEEPRVAT_REPO_PATH"/example/* .
ln -s "$DEEPRVAT_REPO_PATH"/pretrained_models
ln -s config/deeprvat_input_pretrained_models_config.yaml . # Get the corresponding config.
snakemake -j 1 --snakefile "$DEEPRVAT_REPO_PATH"/pipelines/association_testing_pretrained.snakefile
```


### Run association testing using REGENIE on precomputed burdens

```shell
mkdir deeprvat_associate_regenie
cd deeprvat_associate_regenie
ln -s [path_to_deeprvat]/example/* .
ln -s precomputed_burdens/burdens.zarr .
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/association_testing_pretrained_regenie.snakefile
```


### Run the training pipeline on some example data

```shell
DEEPRVAT_REPO_PATH="[path_to_deeprvat]"
mkdir deeprvat_train
cd deeprvat_train
ln -s "$DEEPRVAT_REPO_PATH"/example/* .
ln -s config/deeprvat_input_training_config.yaml . #get the corresponding config.
snakemake -j 1 --snakefile "$DEEPRVAT_REPO_PATH"/pipelines/run_training.snakefile
```


### Run the full training and association testing pipeline on some example data

```shell
DEEPRVAT_REPO_PATH="[path_to_deeprvat]"
mkdir deeprvat_train_associate
cd deeprvat_train_associate
ln -s "$DEEPRVAT_REPO_PATH"/example/* .
ln -s config/deeprvat_input_config.yaml .
snakemake -j 1 --snakefile "$DEEPRVAT_REPO_PATH"/pipelines/training_association_testing.snakefile
```

## Basic usage

### Customize pipelines

Before running any of the snakefiles, you may want to adjust the number of threads used by different steps in the pipeline. To do this, modify the `threads:` property of a given rule.

If you are running on a computing cluster, you will need a [profile](https://github.com/snakemake-profiles) and may need to add `resources:` directives to the snakefiles.


### Run the preprocessing pipeline on your VCF files

Instructions [here](preprocessing.md)


### Annotate variants

Instructions [here](annotations.md)



### Run the full training and association testing pipeline on some example data

```shell
mkdir example
cd example
ln -s [path_to_deeprvat]/example/* .
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/training_association_testing.snakefile
```

Replace `[path_to_deeprvat]` with the path to your clone of the repository.

Note that the example data is randomly generated, and so is only suited for testing whether the `deeprvat` package has been correctly installed.


### Run the training pipeline on some example data

```shell
mkdir example
cd example
ln -s [path_to_deeprvat]/example/* .
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/run_training.snakefile
```

Replace `[path_to_deeprvat]` with the path to your clone of the repository.

Note that the example data is randomly generated, and so is only suited for testing whether the `deeprvat` package has been correctly installed.


### Run the association testing pipeline with pretrained models

```shell
mkdir example
cd example
ln -s [path_to_deeprvat]/example/* .
ln -s [path_to_deeprvat]/pretrained_models
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/association_testing_pretrained.snakefile
```

Replace `[path_to_deeprvat]` with the path to your clone of the repository.

Again, note that the example data is randomly generated, and so is only suited for testing whether the `deeprvat` package has been correctly installed.

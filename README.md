# DeepRVAT

Rare variant association testing using deep learning and data-driven burden scores

[![Documentation Status](https://readthedocs.org/projects/deeprvat/badge/?version=latest)](https://deeprvat.readthedocs.io/en/latest/?badge=latest)

## Installation

1. Clone this repository:
```
git clone git@github.com:PMBio/deeprvat.git
```
1. Change directory to the repository: `cd deeprvat`
1. Install the conda environment. We recommend using [mamba](https://mamba.readthedocs.io/en/latest/index.html), though you may also replace `mamba` with `conda` 
 
   *note: [the current deeprvat env does not support cuda when installed with conda](https://github.com/PMBio/deeprvat/issues/16), install using mamba for cuda support.*
```
mamba env create -n deeprvat -f deeprvat_env.yaml 
```
1. Activate the environment: `mamba activate deeprvat`
1. Install the `deeprvat` package: `pip install -e .`

If you don't want to install the gpu related requirements use the `deeprvat_env_no_gpu.yml` environment instead.
```
mamba env create -n deeprvat -f deeprvat_env_no_gpu.yaml 
```


## Basic usage

### Customize pipelines

Before running any of the snakefiles, you may want to adjust the number of threads used by different steps in the pipeline. To do this, modify the `threads:` property of a given rule.

If you are running on an computing cluster, you will need a [profile](https://github.com/snakemake-profiles) and may need to add `resources:` directives to the snakefiles.


### Run the preprocessing pipeline on VCF files

Instructions [here](https://github.com/PMBio/deeprvat/blob/main/deeprvat/docs/preprocessing.md)


### Annotate variants

Instructions [here](https://github.com/PMBio/deeprvat/blob/main/deeprvat/docs/annotations.md)



### Try the full training and association testing pipeline on some example data

```
mkdir example
cd example
ln -s [path_to_deeprvat]/example/* .
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/training_association_testing.snakefile
```

Replace `[path_to_deeprvat]` with the path to your clone of the repository.

Note that the example data is randomly generated, and so is only suited for testing whether the `deeprvat` package has been correctly installed.


### Run the association testing pipeline with pretrained models

```
mkdir example
cd example
ln -s [path_to_deeprvat]/example/* .
ln -s [path_to_deeprvat]/pretrained_models
snakemake -j 1 --snakefile [path_to_deeprvat]/pipelines/association_testing_pretrained.snakefile
```

Replace `[path_to_deeprvat]` with the path to your clone of the repository.

Again, note that the example data is randomly generated, and so is only suited for testing whether the `deeprvat` package has been correctly installed.


## Credits

Portions of code for the seed gene discovery methods have been adapted from [SEAK](https://seak.readthedocs.io/)

This package was created with Cookiecutter and the `audreyr/cookiecutter-pypackage` project template.

Cookiecutter: https://github.com/audreyr/cookiecutter
`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

# Installation

1. Clone this repository:
```shell
git clone git@github.com:PMBio/deeprvat.git
```
1. Change directory to the repository: `cd deeprvat`
1. Install the conda environment. We recommend using [mamba](https://mamba.readthedocs.io/en/latest/index.html), though you may also replace `mamba` with `conda` 
 
```shell
mamba env create -n deeprvat -f deeprvat_env.yaml 
```
1. Activate the environment: `mamba activate deeprvat`
1. Install the `deeprvat` package: `pip install -e .`

If you don't want to install the GPU-related requirements, use the `deeprvat_env_no_gpu.yml` environment instead.
```shell
mamba env create -n deeprvat -f deeprvat_env_no_gpu.yaml 
```


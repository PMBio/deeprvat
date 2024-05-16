# Cluster execution

## Pipeline resource requirements

For cluster exectution, resource requirements are expected under `resources:` in all rules. All pipelines have some suggested resource requirements, but they may need to be adjusted for your data or cluster.


## Cluster execution

If you are running on a computing cluster, you will need a [profile](https://github.com/snakemake-profiles). We have tested execution on LSF. If you run into issues running on other clusters, please [let us know](https://github.com/PMBio/deeprvat/issues).


## Execution on GPU vs. CPU

Two steps in the pipelines use GPU by default: Training (rule `train` from [train.snakefile](https://github.com/PMBio/deeprvat/blob/main/pipelines/training/train.snakefile)) and burden computation (rule `compute_burdens` from [burdens.snakefile](https://github.com/PMBio/deeprvat/blob/main/pipelines/association_testing/burdens.snakefile)). To run on CPU on a computing cluster, you may need to remove the line `gpus = 1` from the `resources:` of those rules.

Bear in mind that this will make burden computation substantially slower, but still feasible for most datasets. Training without GPU is not practical on large datasets such as UK Biobank.

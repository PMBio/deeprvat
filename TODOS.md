# DeepRVAT

Rare variant association testing using deep learning and data-driven burden scores


## Execution flow

* [x] `preprocess.snakefile` (`preprocess.snakemake`) -> preprocess.py
  * `index_fasta`
  * `extract_samples`
  * `variants`
  * `concatinate_variants`
  * `add_variant_ids`
  * `create_parquet_variant_ids`
  * `normalize`
  * `sparsify`
  * `qc_allelic_imbalance`
  * `qc_read_depth`
  * `qc_hwe`
  * `qc_varmiss`
  * `qc_indmiss`
  * `preprocess`
  * `combine_genotype`
* [x] `preprocess.py`
  * `add_variant_ids`
  * `process_sparse_gt`
  * `combine_gemotype`

* [ ] Annotation pipeline - **Marcel**
  * annotation.snakefile
* [x] `baseline.smk` - **Eva**
  * [x] `baseline_scoretest_smk.py` - **Eva**
    * `update-config`
    * `make-dataset`
    * `test-evaluate`
    * `combine-results`
  * [x] `evaluate_baseline.py` - **Eva**
* [x] `train_multipheno.snakefile` - **Brian**
  * [x] `train_associate.py`
    * [x] rename to `config.py`
    * `update-config`
    * `make-dataset`
      * [x] Move to `train.py`
  * [x] `train.py` - **Brian**
    * `train-bagging`
    * [x] `pl-models.py` - **Felix**
* [x] `associate_multipheno.snakefile` - **Brian**
  * `train.py`
    * `best-bagging-run`
    * `pl_models.py`
    * [x] `data/__init__.py`
    * [x] `data/dense_gt.py` - **Brian**
    * [x] `data/rare.py` - **Brian**
    * [x] `metrics.py`
    * [x] `utils.py`
  * [x] `univariate_burden_regression.py`
    * `make-dataset`
    * `compute-burdens`
    * `regress`
    * `combine-regression-results`
    * [x] Remove all logic for splitting training and replication - **Brian**
  * [x] `evaluate_new.py` - **Brian**
    * [x] Rename
    * [x] Remove testing/replication split
  * [x] `compare_discoveries.py` - **Brian**
    * [x] Merge into `evaluate_new.py`


## Cleanup

For each file:
* Remove orphaned functions/rules
  1. Remove rules not needed in pipelines
  1. Remove click commands not used in any pipeline
  1. Remove functions not used in any Click command
* Remove unused imports, global variables, command/function arguments
* Remove all comments that are not explanations of code
* Remove `resources` from snakefiles
* Convert `genopheno` imports to corresponding `deeprvat` imports
* Reformat files with black/snakefmt

## TODO

* Usage with pretrained models:
  * [ ] Create an association testing pipeline
  * [x] Upload pretrained models
* Documentation
  * [ ] Installation
  * [ ] Basic usage
  * [ ] More detailed docs
* [ ] Acknowledge code used from SEAK
* [x] Merge training/association snakefiles - **Brian**
  * [x] Create entry points
* [x] Create a minimal conda env with all required packages
* [x] Create example files showing the input data format - **Brian**
  * `annotations.parquet`
  * `protein_coding_genes.parquet`
  * `genotypes.h5`
  * `variants.parquet`
  * `config.yaml`
    * add `do_scoretest: True`
* [ ] Run an example pipeline to make sure everything still works
* [x] Replace `py = ...` in `train_associate.snakefile` with placeholder (user should enter path to their clone of `deeprvat` repo) - **Brian**
  * Maybe even add command-line entry points?

Lower priority:
* Really clean up files to remove features that aren't used, esp.:
  * `univariate_burden_regression.py`
    * Remove testing/replication split
  * `data/dense_gt.py`
* Improve `update_config` function to help simplify config files (i.e., give the minimal config and generate everything else that's needed for `hpopt_config.yaml`)
* Refactor certain files, e.g.
  * `data/rare.py`
* Resolve any TODOs commented in the files

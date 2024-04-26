rule all_cv_training:
    input:
        expand('cv_split{cv_split}/deeprvat/models/repeat_{repeat}/best/bag_{bag}.ckpt',
               bag=range(n_bags), repeat=range(n_repeats),
               cv_split = range(cv_splits)),
        expand('cv_split{cv_split}/deeprvat/models/repeat_{repeat}/config.yaml',
               repeat=range(n_repeats),
               cv_split = range(cv_splits))

# make a config for each cv_split (specifying the samples for the current fold)
rule spread_config:
    input:
        config = 'config.yaml'
    output:
        train = 'cv_split{cv_split}/deeprvat/config.yaml',
    params:
        out_path = 'cv_split{cv_split}/'
    threads: 1
    resources:
        mem_mb = 1024,
    shell:
        ' && '.join([
            conda_check,
            'deeprvat_cv_utils spread-config '
            '-m deeprvat '
            '--fold {wildcards.cv_split} '
            # '--fold-specific-baseline '
            f'--n-folds {cv_splits}'
            ' {input.config} {params.out_path}'
        ])


# # ############################### Run DeepRVAT ##############################################################
# # ###########################################################################################################
module deeprvat_workflow:
    snakefile: 
        "../training_association_testing.snakefile"
        # f"{DEEPRVAT_DIR}/pipelines/training_association_testing_with_prefix.snakefile"
    prefix:
        'cv_split{cv_split}/deeprvat'
    config:
        config

# use rule * from deeprvat_workflow exclude config, evaluate, association_dataset, train, regress, best_training_run, compute_burdens, link_burdens as deeprvat_*

use rule link_config from deeprvat_workflow as deeprvat_link_config

use rule best_training_run from deeprvat_workflow as deeprvat_best_training_run with:
    params:
        prefix = 'cv_split{cv_split}/deeprvat'


use rule train from deeprvat_workflow as deeprvat_train with:
    priority: 1000
    params:
        prefix = 'cv_split{cv_split}/deeprvat',
        phenotypes = " ".join( #TODO like need the prefix here as well
            [f"--phenotype {p} "
             f"cv_split{{cv_split}}/deeprvat/{p}/deeprvat/input_tensor.zarr "
             f"cv_split{{cv_split}}/deeprvat/{p}/deeprvat/covariates.zarr "
             f"cv_split{{cv_split}}/deeprvat/{p}/deeprvat/y.zarr"
             for p in training_phenotypes])

use rule training_dataset from deeprvat_workflow as deeprvat_training_dataset

use rule training_dataset_pickle from deeprvat_workflow as deeprvat_training_dataset_pickle

use rule config from deeprvat_workflow as deeprvat_config with:
    input:
        config = 'cv_split{cv_split}/deeprvat/config.yaml', # TODO: change this into cv specific config
        baseline = lambda wildcards: [
            str(Path(r['base']) /wildcards.phenotype / r['type'] /
                'eval/burden_associations.parquet')
            for r in config['baseline_results'] 
        ] if wildcards.phenotype in training_phenotypes else []
    params:
        baseline_results = lambda wildcards, input: ''.join([
            f'--baseline-results {b} '
            for b in input.baseline
        ])  if wildcards.phenotype in training_phenotypes else ' ',
        baseline_out = lambda wildcards: f'--baseline-results-out cv_split{wildcards.cv_split}/deeprvat/{wildcards.phenotype}/deeprvat/baseline_results.parquet' if wildcards.phenotype in training_phenotypes else ' ',
        seed_genes_out = lambda wildcards: f'--seed-genes-out cv_split{wildcards.cv_split}/deeprvat/{wildcards.phenotype}/deeprvat/seed_genes.parquet' if wildcards.phenotype in training_phenotypes else ' '









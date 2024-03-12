configfile: 'config.yaml'

phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes

debug_flag = config.get('debug', False)
debug = '--debug ' if debug_flag else ''

if not "cv_exp" in globals():
    cv_exp = config.get("cv_exp", False)

config_file_prefix = (
    "cv_split0/deeprvat/" if cv_exp else ""
)


rule all_evaluate:  #plot.snakefile
    input:
        significant=expand(
            "{phenotype}/deeprvat/eval/significant.parquet", phenotype=phenotypes
        ),
        results=expand(
            "{phenotype}/deeprvat/eval/all_results.parquet", phenotype=phenotypes
        ),


rule evaluate:
    input:
        associations ='{phenotype}/deeprvat/average_regression_results/burden_associations.parquet',
        config = f"{config_file_prefix}{{phenotype}}/deeprvat/hpopt_config.yaml"
    output:
        "{phenotype}/deeprvat/eval/significant.parquet",
        "{phenotype}/deeprvat/eval/all_results.parquet"
    threads: 1
    resources:
        mem_mb = 16000,
        load = 16000
    params:
        n_combis = 1,
        use_baseline_results = '--use-baseline-results'
    shell:
        'deeprvat_evaluate '
        + debug +
        '{params.use_baseline_results} '
        '--correction-method Bonferroni '
        '--phenotype {wildcards.phenotype} '
        '{input.associations} '
        '{input.config} '
        '{wildcards.phenotype}/deeprvat/eval'

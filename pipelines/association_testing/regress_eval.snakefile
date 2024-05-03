config_file_prefix = (
    "cv_split0/deeprvat/" if cv_exp else ""
)  
########### Average regression 
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


rule combine_regression_chunks:
    input:
        expand('{{phenotype}}/deeprvat/average_regression_results/burden_associations_{chunk}.parquet', chunk=range(n_regression_chunks)),
    output:
        '{phenotype}/deeprvat/average_regression_results/burden_associations.parquet',
    threads: 1
    resources:
        mem_mb = lambda wildcards, attempt: 12000 + (attempt - 1) * 4098,
    shell:
        'deeprvat_associate combine-regression-results '
        '--model-name repeat_0 ' 
        '{input} '
        '{output}'


rule regress:
    input:
        config = f"{config_file_prefix}{{phenotype}}/deeprvat/hpopt_config.yaml",
        chunks = '{phenotype}/deeprvat/burdens/burdens.zarr' if not cv_exp  else '{phenotype}/deeprvat/burdens/merging.finished',
        phenotype_0_chunks =  expand(
            phenotypes[0] + '/deeprvat/burdens/logs/burdens_averaging_{chunk}.finished',
            chunk=range(n_avg_chunks)
        ),
    output:
        temp('{phenotype}/deeprvat/average_regression_results/burden_associations_{chunk}.parquet'),
    threads: 2
    resources:
        mem_mb = lambda wildcards, attempt: 28676  + (attempt - 1) * 4098,
    params:
        burden_file = f'{phenotypes[0]}/deeprvat/burdens/burdens_average.zarr',
        burden_dir = '{phenotype}/deeprvat/burdens',
        out_dir = '{phenotype}/deeprvat/average_regression_results'
    shell:
        'deeprvat_associate regress '
        + debug +
        '--chunk {wildcards.chunk} '
        '--n-chunks ' + str(n_regression_chunks) + ' '
        '--use-bias '
        '--repeat 0 '
        '--burden-file {params.burden_file} '
        + do_scoretest +
        '{input.config} '
        '{params.burden_dir} ' #TODO make this w/o repeats
        '{params.out_dir}'


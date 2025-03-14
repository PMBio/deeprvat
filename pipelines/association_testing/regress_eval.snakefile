config_file_prefix = (
    "cv_split0/deeprvat/" if cv_exp else ""
)
########### Average regression 
# rule all_evaluate:
#     input:
#         expand("{phenotype}/deeprvat/eval/significant.parquet",
#                phenotype=phenotypes),
#         expand("{phenotype}/deeprvat/eval/all_results.parquet",
#                phenotype=phenotypes),

rule evaluate:
    input:
        associations ='{phenotype}/deeprvat/average_regression_results/burden_associations.parquet',
        data_config = f"{config_file_prefix}{{phenotype}}/deeprvat/config.yaml"
    output:
        "{phenotype}/deeprvat/eval/significant.parquet",
        "{phenotype}/deeprvat/eval/all_results.parquet"
    threads: 1
    resources:
        mem_mb = 16000,
    params:
        n_combis = 1,
        use_baseline_results = '--use-baseline-results' if 'baseline_results' in config else ''
    log:
        stdout="logs/evaluate/{phenotype}.stdout", 
        stderr="logs/evaluate/{phenotype}.stderr"
    shell:
        'deeprvat_evaluate '
        + debug +
        '{params.use_baseline_results} '
        '--phenotype {wildcards.phenotype} '
        '{input.associations} '
        '{input.data_config} '
        '{wildcards.phenotype}/deeprvat/eval '
        + logging_redirct


rule combine_regression_chunks:
    input:
        expand('{{phenotype}}/deeprvat/average_regression_results/burden_associations_{chunk}.parquet', chunk=range(n_regression_chunks)),
    output:
        '{phenotype}/deeprvat/average_regression_results/burden_associations.parquet',
    threads: 1
    resources:
        mem_mb = lambda wildcards, attempt: 12000 + (attempt - 1) * 4098,
    log:
        stdout="logs/combine_regression_chunks/{phenotype}.stdout", 
        stderr="logs/combine_regression_chunks/{phenotype}.stderr"
    shell:
        'deeprvat_associate combine-regression-results '
        '--model-name repeat_0 ' 
        '{input} '
        '{output} '
        + logging_redirct


rule regress:
    input:
        data_config = f"{config_file_prefix}{{phenotype}}/deeprvat/config.yaml",
        chunks_xy =  expand(
            'burdens/log/burdens_averaging_{chunk}.finished',
            chunk=range(n_avg_chunks)
        ) + ['{phenotype}/deeprvat/xy/x.zarr', '{phenotype}/deeprvat/xy/y.zarr']
        if not cv_exp
        else expand('burdens/log/{phenotype}/merging.finished',
                    phenotype=phenotypes),
        chunks_burden_average = expand('burdens/log/burdens_averaging_{chunk}.finished',
                                       chunk=range(n_avg_chunks)),
        # x = '{phenotype}/deeprvat/xy/x.zarr',
        # y = '{phenotype}/deeprvat/xy/y.zarr',
    output:
        temp('{phenotype}/deeprvat/average_regression_results/burden_associations_{chunk}.parquet'),
    threads: 2
    resources:
        mem_mb = lambda wildcards, attempt: 28676  + (attempt - 1) * 4098,
    params:
        burden_file = 'burdens/burdens_average.zarr',
        xy_dir = "{phenotype}/deeprvat/xy",
        # burden_dir = 'burdens',
        out_dir = '{phenotype}/deeprvat/average_regression_results'
    log:
        stdout="logs/regress/{phenotype}_regress_{chunk}.stdout", 
        stderr="logs/regress/{phenotype}_regress_{chunk}.stderr"
    shell:
        'deeprvat_associate regress '
        + debug +
        '--chunk {wildcards.chunk} '
        '--n-chunks ' + str(n_regression_chunks) + ' '
        '--use-bias '
        # '--repeat 0 '
        + do_scoretest +
        '{input.data_config} '
        "{params.xy_dir} "
        "{params.burden_file} "
        '{params.out_dir} '
        + logging_redirct


rule average_burdens:
    input:
        'burdens/burdens.zarr'
        if not cv_exp
        else f'burdens/log/{phenotypes[0]}/merging.finished',
    output:
        'burdens/log/burdens_averaging_{chunk}.finished',
    params:
        burdens_in = 'burdens/burdens.zarr',
        burdens_out = 'burdens/burdens_average.zarr',
        repeats = lambda wildcards: ''.join([f'--repeats {r} ' for r in range(int(n_repeats))])
    threads: 1
    resources:
        mem_mb = lambda wildcards, attempt: 4098 + (attempt - 1) * 4098,
    priority: 10,
    log:
        stdout="logs/average_burdens/average_burdens_{chunk}.stdout", 
        stderr="logs/average_burdens/average_burdens_{chunk}.stderr"
    shell:
        ' && '.join([
            ('deeprvat_associate average-burdens '
             '--n-chunks ' + str(n_avg_chunks) + ' '
             '--chunk {wildcards.chunk} '
             '{params.repeats} '
             '--agg-fct mean  '  #TODO remove this
             '{params.burdens_in} '
             '{params.burdens_out} '
             + logging_redirct),
            'touch {output}'
        ])

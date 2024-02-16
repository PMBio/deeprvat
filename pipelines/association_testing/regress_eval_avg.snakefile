
########### Average regression 
# TODO try to remove this/get rid
cv_exp = True
config_file_prefix = 'cv_split0/deeprvat/' if cv_exp else '' #needed in case we analyse a CV experiment

rule evaluate_avg:
    input:
        associations ='{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/burden_associations.parquet',
        config = f"{config_file_prefix}{{phenotype}}/deeprvat/hpopt_config.yaml"
    output:
        "{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/eval/{use_seed}/significant.parquet",
        "{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/eval/{use_seed}/all_results.parquet"
    threads: 1
    resources:
        mem_mb = 16000,
        load = 16000
    params:
        n_combis = 1,
        use_seed_genes = lambda wildcards: use_seed_dict[wildcards.use_seed]
    shell:
        'deeprvat_evaluate '
        + debug +
        '{params.use_seed_genes} '
        '--save-default '
        '--n-repeats {params.n_combis} ' #because we analyze each average combi alone, so the totatl number of combis is the total number of repeats
        '--correction-method FDR '
        '--repeats-to-analyze 1 ' #always only analyse one combination 
        '--max-repeat-combis {params.n_combis} '
        '{input.associations} '
        '{input.config} '
        '{wildcards.phenotype}/deeprvat/{wildcards.burden_agg_fct}_agg_results/{wildcards.n_avg_repeats}_repeats/eval/{wildcards.use_seed}/'


rule combine_regression_chunks_avg:
    input:
        expand('{{phenotype}}/deeprvat/{{burden_agg_fct}}_agg_results/{{n_avg_repeats}}_repeats/burden_associations_{chunk}.parquet', chunk=range(n_regression_chunks)),
    output:
        '{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/burden_associations.parquet',
    threads: 1
    resources:
        mem_mb = lambda wildcards, attempt: 12000 + (attempt - 1) * 4098,
        load = 2000
    shell:
        'deeprvat_associate combine-regression-results '
        '--model-name repeat_0 ' 
        '{input} '
        '{output}'


rule regress_avg:
    input:
        config = f"{config_file_prefix}{{phenotype}}/deeprvat/hpopt_config.yaml",
        chunks = lambda wildcards: (
            [] if wildcards.phenotype == phenotypes[0]
            else expand('{{phenotype}}/deeprvat/burdens/chunk{chunk}.linked',
                        chunk=range(n_burden_chunks))
        ) if not cv_exp  else '{phenotype}/deeprvat/burdens/merging.finished',
        phenotype_0_chunks =  expand(
            phenotypes[0] + '/deeprvat/burdens/logs/burdens_{{burden_agg_fct}}_{{n_avg_repeats}}_repeats_chunk_{chunk}.finished',
            chunk=range(n_avg_chunks)
        ),
    output:
        temp('{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/burden_associations_{chunk}.parquet'),
    threads: 2
    resources:
        mem_mb = lambda wildcards, attempt: 28676  + (attempt - 1) * 4098,
        # mem_mb = 16000,
        load = lambda wildcards, attempt: 28000 + (attempt - 1) * 4000
    params:
        burden_file = f'{phenotypes[0]}/deeprvat/burdens/burdens_{{burden_agg_fct}}_{{n_avg_repeats}}.zarr',
        burden_dir = '{phenotype}/deeprvat/burdens',
        out_dir = '{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats'
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


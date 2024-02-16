
rule evaluate: #TODO needs to be simplified!
    input:
        associations = expand('{{phenotype}}/deeprvat/repeat_{repeat}/results/burden_associations.parquet',
                              repeat=range(n_repeats)),
        config = '{phenotype}/deeprvat/hpopt_config.yaml',
    output:
        "{phenotype}/deeprvat/eval/significant.parquet",
        "{phenotype}/deeprvat/eval/all_results.parquet"
    threads: 1
    params:
        out_path = '{phenotype}/deeprvat/eval',
        use_seed_genes = '--use-seed-genes', 
        n_repeats = f'{n_repeats}',
        repeats_to_analyze = f'{n_repeats}',
        max_repeat_combis = 1,
        combine_pval = ''
    resources:
        mem_mb = lambda wildcards, attempt: 25000 + attempt * 4098,
    shell:
        'deeprvat_evaluate '
        + debug +
        '{params.use_seed_genes} '
        '--n-repeats {params.n_repeats} '#total number of repeats
        '--repeats-to-analyze {params.repeats_to_analyze} '#number of repeats to analyze
        '--max-repeat-combis {params.max_repeat_combis} '
        '--correction-method FDR '
        '{params.combine_pval} '
        '{input.associations} '
        '{input.config} '
        '{params.out_path}'

# rule evaluate:
#     input:
#         associations = expand('{{phenotype}}/deeprvat/repeat_{repeat}/results/burden_associations.parquet',
#                               repeat=range(n_repeats)),
#         config = '{phenotype}/deeprvat/hpopt_config.yaml',
#     output:
#         f"{{phenotype}}/deeprvat/eval/significant_{n_repeats}repeats.parquet",
#         f"{{phenotype}}/deeprvat/eval/all_results_{n_repeats}repeats.parquet"
#     threads: 1
#     params:
#         out_path = '{phenotype}/deeprvat/eval',
#         use_seed_genes = '--use-seed-genes', 
#         n_repeats = f'{n_repeats}',
#         repeats_to_analyze = f'{n_repeats}',
#         max_repeat_combis = 1,
#         combine_pval = ''
#     resources:
#         mem_mb = lambda wildcards, attempt: 25000 + attempt * 4098,
#     shell:
#         'deeprvat_evaluate '
#         + debug +
#         '{params.use_seed_genes} '
#         '--n-repeats {params.n_repeats} '#total number of repeats
#         '--repeats-to-analyze {params.repeats_to_analyze} '#number of repeats to analyze
#         '--max-repeat-combis {params.max_repeat_combis} '
#         '--correction-method FDR '
#         '{params.combine_pval} '
#         '{input.associations} '
#         '{input.config} '
#         '{params.out_path}'

rule all_regression:
    input:
        expand('{phenotype}/deeprvat/repeat_{repeat}/results/burden_associations.parquet',
               phenotype=phenotypes, type=['deeprvat'], repeat=range(n_repeats)),

rule combine_regression_chunks:
    input:
        expand('{{phenotype}}/deeprvat/repeat_{{repeat}}/results/burden_associations_{chunk}.parquet', chunk=range(n_regression_chunks)),
    output:
        '{phenotype}/deeprvat/repeat_{repeat}/results/burden_associations.parquet',
    threads: 1
    resources:
        mem_mb = 2048,
        load = 2000
    shell:
        'deeprvat_associate combine-regression-results '
        '--model-name repeat_{wildcards.repeat} '
        '{input} '
        '{output}'
    
rule regress:
    input:
        config = "{phenotype}/deeprvat/hpopt_config.yaml",
        chunks = lambda wildcards: expand(
            ('{{phenotype}}/deeprvat/burdens/chunk{chunk}.' +
             ("finished" if wildcards.phenotype == phenotypes[0] else "linked")),
            chunk=range(n_burden_chunks)
        ),
        phenotype_0_chunks =  expand(
            phenotypes[0] + '/deeprvat/burdens/chunk{chunk}.finished',
            chunk=range(n_burden_chunks)
        ),
    params:
        prefix = '.'
    output:
        temp('{phenotype}/deeprvat/repeat_{repeat}/results/burden_associations_{chunk}.parquet'),
    threads: 2
    resources:
        mem_mb = lambda wildcards, attempt: 28676 + (attempt - 1) * 4098,
        # mem_mb = 16000,
        load = lambda wildcards, attempt: 28000 + (attempt - 1) * 4000
    shell:
        'deeprvat_associate regress '
        + debug +
        '--chunk {wildcards.chunk} '
        '--n-chunks ' + str(n_regression_chunks) + ' '
        '--use-bias '
        '--repeat {wildcards.repeat} '
        + do_scoretest +
        '{input.config} '
        '{params.prefix}/{wildcards.phenotype}/deeprvat/burdens ' #TODO make this w/o repeats
        '{params.prefix}/{wildcards.phenotype}/deeprvat/repeat_{wildcards.repeat}/results'

########### Average regression 
# TODO try to remove this/get rid
cv_exp = True
config_file_prefix = 'cv_split0/deeprvat/' if cv_exp else '' #needed in case we analyse a CV experiment
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
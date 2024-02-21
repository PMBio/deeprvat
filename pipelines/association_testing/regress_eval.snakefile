configfile: "config.yaml"

debug_flag = config.get('debug', False)
debug = '--debug ' if debug_flag else ''

n_repeats = config['n_repeats']

phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes

n_burden_chunks = config.get('n_burden_chunks', 1) if not debug_flag else 2

rule evaluate:
    input:
        associations = expand('{{phenotype}}/deeprvat/repeat_{repeat}/results/burden_associations.parquet',
                              repeat=range(n_repeats)),
        config = '{phenotype}/deeprvat/hpopt_config.yaml',
    output:
        "{phenotype}/deeprvat/eval/significant.parquet",
        "{phenotype}/deeprvat/eval/all_results.parquet"
    threads: 1
    shell:
        'deeprvat_evaluate '
        + debug +
        '--use-seed-genes '
        '--n-repeats {n_repeats} '
        '--correction-method FDR '
        '{input.associations} '
        '{input.config} '
        '{wildcards.phenotype}/deeprvat/eval'

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
    output:
        temp('{phenotype}/deeprvat/repeat_{repeat}/results/burden_associations_{chunk}.parquet'),
    threads: 2
    shell:
        'deeprvat_associate regress '
        + debug +
        '--chunk {wildcards.chunk} '
        '--n-chunks ' + str(n_regression_chunks) + ' '
        '--use-bias '
        '--repeat {wildcards.repeat} '
        + do_scoretest +
        '{input.config} '
        '{wildcards.phenotype}/deeprvat/burdens ' #TODO make this w/o repeats
        '{wildcards.phenotype}/deeprvat/repeat_{wildcards.repeat}/results'

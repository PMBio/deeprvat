
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

rule make_regenie_input:
    input:
        burdens = lambda wildcards: expand(
            ('{{phenotype}}/deeprvat/burdens/chunk{chunk}.' +
             ("finished" if wildcards.phenotype == phenotypes[0] else "linked")),
            chunk=range(n_burden_chunks)
        ),
        datasets = expand("{phenotype}/deeprvat/association_dataset.pkl",
                          phenotype=phenotypes),
    params:
        phenotypes = " ".join([f"--phenotype {p} {p}/deeprvat/association_dataset.pkl {p}/deeprvat/burdens"
                               for ]) + " "
    output:
        # bgen = "{phenotype}/deeprvat/regenie_input/pseudo_variants.bgen",
        covariant_file = "regenie_input/covariates.txt",
        phenotype_file = "regenie_input/phenotypes.txt",
    threads: 1
    shell:
        "deeprvat_associate make-saige-input "
        "--average-repeats "
        "{params.phenotypes}"
        # "{input.dataset} "
        # "{wildcards.phenotype}/deeprvat/burdens "
        # "{output.vcf} "
        "{output.covariate_file} "
        "{output.phenotype_file}"


rule association_dataset:
    input:
        config = '{phenotype}/deeprvat/hpopt_config.yaml'
    output:
        '{phenotype}/deeprvat/association_dataset.pkl'
    threads: 4
    shell:
        'deeprvat_associate make-dataset '
        + debug +
        '{input.config} '
        '{output}'

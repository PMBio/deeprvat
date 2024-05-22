configfile: "config.yaml"

debug_flag = config.get('debug', False)
debug = '--debug ' if debug_flag else ''

phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes


rule association_dataset:
    input:
        config = '{phenotype}/deeprvat/hpopt_config.yaml'
    output:
        temp('{phenotype}/deeprvat/association_dataset.pkl')
    threads: 4
    resources:
        mem_mb = lambda wildcards, attempt: 32000 * (attempt + 1)
    priority: 30
    shell:
        'deeprvat_associate make-dataset '
        + debug +
        "--skip-genotypes "
        '{input.config} '
        '{output}'


rule association_dataset_burdens:
    input:
        config = f'{phenotypes[0]}/deeprvat/hpopt_config.yaml'
    output:
        temp('burdens/association_dataset.pkl')
    threads: 4
    resources:
        mem_mb = lambda wildcards, attempt: 32000 * (attempt + 1)
    priority: 30
    shell:
        'deeprvat_associate make-dataset '
        + debug +
        '{input.config} '
        '{output}'

configfile: "config.yaml"

debug_flag = config.get('debug', False)
debug = '--debug ' if debug_flag else ''

phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes


rule all_association_dataset:
    input:
        expand('{phenotype}/deeprvat/association_dataset.pkl',
               phenotype=phenotypes)

rule association_dataset:
    input:
        config = '{phenotype}/deeprvat/hpopt_config.yaml'
    output:
        '{phenotype}/deeprvat/association_dataset.pkl'
    threads: 4
    resources:
        mem_mb = lambda wildcards, attempt: 32000 * (attempt + 1),
        load = 64000
    priority: 30
    shell:
        'deeprvat_associate make-dataset '
        + debug +
        "--skip-burdens "
        '{input.config} '
        '{output}'

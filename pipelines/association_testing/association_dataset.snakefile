configfile: "deeprvat_config.yaml"

debug_flag = config.get('debug', False)
debug = '--debug ' if debug_flag else ''

phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes


rule association_dataset:
    input:
        data_config = '{phenotype}/deeprvat/config.yaml'
    output:
        '{phenotype}/deeprvat/association_dataset.pkl'
    threads: 4
    resources:
        mem_mb = lambda wildcards, attempt: 32000 * (attempt + 1),
    priority: 30
    log:
        stdout="logs/association_dataset/{phenotype}.stdout", 
        stderr="logs/association_dataset/{phenotype}.stderr"
    shell:
        'deeprvat_associate make-dataset '
        + debug +
        "--skip-genotypes "
        '{input.data_config} '
        '{output} '
        + logging_redirct


rule association_dataset_burdens:
    input:
        data_config = f'{phenotypes[0]}/deeprvat/config.yaml'
    output:
        'burdens/association_dataset.pkl'
    threads: 4
    resources:
        mem_mb = lambda wildcards, attempt: 32000 * (attempt + 1)
    priority: 30
    log:
        stdout=f"logs/association_dataset_burdens/{phenotypes[0]}.stdout", 
        stderr=f"logs/association_dataset_burdens/{phenotypes[0]}.stderr"
    shell:
        'deeprvat_associate make-dataset '
        + debug +
        '{input.data_config} '
        '{output} '
        + logging_redirct

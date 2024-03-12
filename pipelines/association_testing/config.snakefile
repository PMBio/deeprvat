configfile: "config.yaml"

phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes


rule all_config:
    input:
        config = expand('{phenotype}/deeprvat/hpopt_config.yaml',
                        phenotype=phenotypes),

rule config:
    input:
        config = 'config.yaml',
    output:
        config = '{phenotype}/deeprvat/hpopt_config.yaml',
    threads: 1
    shell:
        (
            'deeprvat_config update-config '
            "--association-only "
            '--phenotype {wildcards.phenotype} '
            '{input.config} '
            '{output.config}'
        )

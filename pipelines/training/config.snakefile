
rule config:
    input:
        config = 'config.yaml',
        baseline = lambda wildcards: [
            str(Path(r['base']) / wildcards.phenotype / r['type'] /
                'eval/burden_associations.parquet')
            for r in config['baseline_results']
        ]
    output:
        seed_genes = '{phenotype}/deeprvat/seed_genes.parquet',
        config = '{phenotype}/deeprvat/hpopt_config.yaml',
        baseline = '{phenotype}/deeprvat/baseline_results.parquet',
    threads: 1
    params:
        baseline_results = lambda wildcards, input: ''.join([
            f'--baseline-results {b} '
            for b in input.baseline
        ])
    shell:
        (
            'deeprvat_config update-config '
            '--phenotype {wildcards.phenotype} '
            '{params.baseline_results}'
            '--baseline-results-out {output.baseline} '
            '--seed-genes-out {output.seed_genes} '
            '{input.config} '
            '{output.config}'
        )
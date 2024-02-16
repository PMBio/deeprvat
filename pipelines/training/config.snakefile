
rule config:
    input:
        config = 'config.yaml',
        baseline = lambda wildcards: [
            str(Path(r['base']) / wildcards.phenotype / r['type'] /
                'eval/burden_associations.parquet')
            for r in config['baseline_results']
        ]
    output:
        # seed_genes = '{phenotype}/deeprvat/seed_genes.parquet',
        config = '{phenotype}/deeprvat/hpopt_config.yaml',
        # baseline = '{phenotype}/deeprvat/baseline_results.parquet',
    threads: 1
    resources:
        mem_mb = 1024,
        load = 1000
    params:
        baseline_results = lambda wildcards, input: ''.join([
            f'--baseline-results {b} '
            for b in input.baseline
        ]),
        baseline_out = '--baseline-results-out {phenotype}/deeprvat/baseline_results.parquet',
        seed_genes_out = '--seed-genes-out {phenotype}/deeprvat/seed_genes.parquet'
    shell:
        (
            'deeprvat_config update-config '
            '--phenotype {wildcards.phenotype} '
            '{params.baseline_results}'
            '{params.baseline_out} '
            '{params.seed_genes_out} '
            # '--baseline-results-out {params.baseline} '
            # '--seed-genes-out {output.seed_genes} '
            '{input.config} '
            '{output.config}'
        )
#requires that comparison_results.pkl is linked to the experiment directory
#requires deeprvat-analyis to be installed
DEEPRVAT_ANALYSIS_DIR = os.environ['DEEPRVAT_ANALYSIS_DIR']
py_deeprvat_analysis = f'python {DEEPRVAT_ANALYSIS_DIR}'

rule plot:
    conda:
        "r-env"
    input:
        significant = expand("{phenotype}/deeprvat/eval/significant.parquet",
               phenotype=phenotypes),
        results = expand("{phenotype}/deeprvat/eval/all_results.parquet",
               phenotype=phenotypes),
        replication = "replication.parquet"
    output:
        "dicovery_replication_plot.png"
    params:
        results_dir = './',
        results_dir_pattern = '',
        code_dir = f'{DEEPRVAT_ANALYSIS_DIR}/association_testing'
    resources:
        mem_mb=20480,
        load=16000,
    script:
        f'{DEEPRVAT_ANALYSIS_DIR}/association_testing/figure_3_main.R'

rule compute_replication:
    input:
        results = expand("{phenotype}/deeprvat/eval/all_results.parquet",
            phenotype = training_phenotypes)
    output:
        'replication.parquet'
    params:
        result_files = lambda wildcards, input: ''.join([
            f'--result-files {f} '
            for f in input.results
        ]),
        n_repeats = f'{n_repeats}'
    resources:
        mem_mb = lambda wildcards, attempt: 32000 + attempt * 4098 * 2,
    shell:
        py_deeprvat_analysis + '/association_testing/compute_replication.py '
        '--out-file {output} '
        '--n-repeats {params.n_repeats} '
        '{params.result_files} '
        './ '
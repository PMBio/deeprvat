config_file_prefix = (
    "cv_split0/deeprvat/" if cv_exp else ""
)


rule evaluate:
    input:
        associations ='{phenotype}/deeprvat/average_regression_results/burden_associations.parquet',
        data_config = f"{config_file_prefix}{{phenotype}}/deeprvat/config.yaml"
    output:
        "{phenotype}/deeprvat/eval/significant.parquet",
        "{phenotype}/deeprvat/eval/all_results.parquet"
    threads: 1
    resources:
        mem_mb = 16000,
    params:
        use_baseline_results = '--use-baseline-results' if 'baseline_results' in config else ''
    log:
        stdout="logs/evaluate/{phenotype}.stdout", 
        stderr="logs/evaluate/{phenotype}.stderr"
    shell:
        'deeprvat_evaluate '
        + debug +
        '{params.use_baseline_results} '
        '--phenotype {wildcards.phenotype} '
        '{input.associations} '
        '{input.data_config} '
        '{wildcards.phenotype}/deeprvat/eval '
        + logging_redirct

rule all_regenie:
    input:
        expand('{phenotype}/deeprvat/average_regression_results/burden_associations.parquet',
               phenotype=phenotypes),

rule convert_regenie_output:
    input:
        expand("regenie_output/step2/deeprvat_{phenotype}.regenie",
               phenotype=phenotypes)
    output:
        expand('{phenotype}/deeprvat/average_regression_results/burden_associations.parquet',
               phenotype=phenotypes)
    params:
        pheno_options = " ".join([
            f"--phenotype {phenotype} regenie_output/step2/deeprvat_{phenotype}.regenie "
            f"{phenotype}/deeprvat/average_regression_results/burden_associations.parquet"
        for phenotype in phenotypes]),
        gene_file = config["association_testing_data"]["dataset_config"]["rare_embedding"]["config"]["gene_file"]
    threads: 1
    resources:
        mem_mb = 2048
    log:
        stdout="logs/convert_regenie_output/convert_regenie_output.stdout",
        stderr="logs/convert_regenie_output/convert_regenie_output.stderr"
    shell:
        "deeprvat_associate convert-regenie-output "
        "{params.pheno_options} "
        "{params.gene_file} "
        + logging_redirct

rule regenie_step2:
    input:
        sample_file = "regenie_input/deeprvat_pseudovariants.sample",
        bgen = "regenie_input/deeprvat_pseudovariants.bgen",
        covariate_file = "regenie_input/covariates.txt",
        phenotype_file = "regenie_input/phenotypes.txt",
        # step1_loco = expand("regenie_output/step1/deeprvat_{pheno_num}.loco",
        #                     pheno_num=range(1, len(phenotypes) + 1)),
        step1_predlist = "regenie_output/step1/deeprvat_pred.list"
        # step1_loco = expand("regenie_output/step1/deeprvat_l1_{pheno_number}.loco",
        #                     pheno_number=range(len(phenotypes))),
        # step1_predlist = "regenie_output/step1/deeprvat_l1_pred.list",
    output:
        expand("regenie_output/step2/deeprvat_{phenotype}.regenie",
               phenotype=phenotypes)
    threads: 16
    resources:
        mem_mb = 16384
    log:
        stdout="logs/regenie_step2/regenie_step2.stdout",
        stderr="logs/regenie_step2/regenie_step2.stderr",
    shell:
        "regenie "
        "--step 2 "
        "--bgen {input.bgen} "
        "--ref-first "
        "--sample {input.sample_file} "
        "--phenoFile {input.phenotype_file} "
        "--covarFile {input.covariate_file} "
        "--pred {input.step1_predlist} "
        f"--bsize {regenie_step2_bsize} "
        "--threads 16 "
        + " ".join(regenie_config_step2.get("options", [])) + " " +
        "--out regenie_output/step2/deeprvat "
        + logging_redirct

rule regenie_step1:
    input:
        bgen = regenie_config_step1['bgen'],
        sample_file = "regenie_input/deeprvat_pseudovariants.sample",
        snplist = regenie_config_step1["snplist"],
        covariate_file = "regenie_input/covariates.txt",
        phenotype_file = "regenie_input/phenotypes.txt",
    output:
        # expand("regenie_output/step1/deeprvat_{pheno_num}.loco",
        #        pheno_num=range(1, len(phenotypes) + 1)),
        "regenie_output/step1/deeprvat_pred.list"
    threads: 24
    resources:
        mem_mb = 16000
    log:
        stdout="logs/regenie_step1/regenie_step1.stdout", 
        stderr="logs/regenie_step1/regenie_step1.stderr"
    shell:
        "mkdir -p regenie_step1_tmp && "
        "regenie "
        "--step 1 "
        "--bgen {input.bgen} "
        "--extract {input.snplist} "
        "--keep {input.sample_file} "
        "--phenoFile {input.phenotype_file} "
        "--covarFile {input.covariate_file} "
        f"--bsize {regenie_step1_bsize} "
        "--threads 24 "
        "--lowmem "
        "--lowmem-prefix regenie_step1_tmp/deeprvat "
        + " ".join(regenie_config_step1.get("options", [])) + " " +
        "--out regenie_output/step1/deeprvat "
        + logging_redirct + " ; "
        "rm -rf regenie_step1_tmp "
        


# rule regenie_step1_runl1:
#     input:
#         expand("regenie_output/step1/deeprvat_parallel_job{job}_l0_Y{pheno_number}",
#                job=regenie_joblist, pheno_number=range(1, len(phenotypes) + 1)),
#         bgen = regenie_config_step1['bgen'],
#         sample_file = "regenie_input/deeprvat_pseudovariants.sample",
#         snplist = regenie_config_step1["snplist"],
#         covariate_file = "regenie_input/covariates.txt",
#         phenotype_file = "regenie_input/phenotypes.txt",
#     output:
#         expand("regenie_output/step1/deeprvat_l1_{pheno_number}.loco",
#                pheno_number=range(len(phenotypes))),
#         "regenie_output/step1/deeprvat_l1_pred.list"
#     threads: 16
#     resources:
#         mem_mb = 16000
#     shell:
#         "regenie "
#         "--step 1 "
#         "--bgen {input.bgen} "
#         "--extract {input.snplist} "
#         "--keep {input.sample_file} "
#         "--phenoFile {input.phenotype_file} "
#         "--covarFile {input.covariate_file} "
#         f"--bsize {regenie_step1_bsize} "
#         "--lowmem "
#         "--lowmem-prefix regenie_step1_tmp "
#         "--threads 16 "
#         + " ".join(regenie_config_step1.get("options", [])) + " " +
#         "--out regenie_output/step1/deeprvat_l1 "
#         f"--run-l1 regenie_output/step1/deeprvat_parallel.master"

# rule regenie_step1_runl0:
#     input:
#         master = "regenie_output/step1/deeprvat_parallel.master",
#         snplists = expand("regenie_output/step1/deeprvat_parallel_job{job}.snplist",
#                           job=regenie_joblist),
#         bgen = regenie_config_step1['bgen'],
#         sample_file = "regenie_input/deeprvat_pseudovariants.sample",
#         covariate_file = "regenie_input/covariates.txt",
#         phenotype_file = "regenie_input/phenotypes.txt",
#     output:
#         expand("regenie_output/step1/deeprvat_parallel_job{{job}}_l0_Y{pheno_number}",
#                pheno_number=range(1, len(phenotypes) + 1))
#     threads: 8
#     resources:
#         mem_mb = 16000
#     shell:
#         " mkdir -p regenie_step1_tmp_job{wildcards.job} && "
#         "regenie "
#         "--step 1 "
#         "--bgen {input.bgen} "
#         "--keep {input.sample_file} "
#         "--phenoFile regenie_input/phenotypes.txt "
#         "--covarFile regenie_input/covariates.txt "
#         f"--bsize {regenie_step1_bsize} "
#         "--lowmem "
#         "--lowmem-prefix regenie_step1_tmp_job{wildcards.job} "
#         "--threads 8 "
#         + " ".join(regenie_config_step1.get("options", [])) + " " +
#         "--out regenie_output/step1/deeprvat "
#         "--run-l0 regenie_output/step1/deeprvat_parallel.master,{wildcards.job} && "
#         "rm -rf regenie_step1_tmp_job{wildcards.job}"

# rule regenie_step1_splitl0:
#     input:
#         bgen = regenie_config_step1['bgen'],
#         sample_file = "regenie_input/deeprvat_pseudovariants.sample",
#         snplist = regenie_config_step1["snplist"],
#         covariate_file = "regenie_input/covariates.txt",
#         phenotype_file = "regenie_input/phenotypes.txt",
#     output:
#         "regenie_output/step1/deeprvat_parallel.master",
#         expand("regenie_output/step1/deeprvat_parallel_job{job}.snplist",
#                job=regenie_joblist)
#     threads: 8
#     resources:
#         mem_mb = 16000
#     shell:
#         "regenie "
#         "--step 1 "
#         "--bgen {input.bgen} "
#         "--extract {input.snplist} "
#         "--keep {input.sample_file} "
#         "--phenoFile {input.phenotype_file} "
#         "--covarFile {input.covariate_file} "
#         f"--bsize {regenie_step1_bsize} "
#         "--threads 8 "
#         + " ".join(regenie_config_step1.get("options", [])) + " " +
#         "--out regenie_output/step1/deeprvat "
#         f"--split-l0 regenie_output/step1/deeprvat_parallel,{regenie_njobs}"

rule make_regenie_burdens:
    input:
        gene_file = config["association_testing_data"]["dataset_config"]["rare_embedding"]["config"]["gene_file"],
        gtf_file = config["gtf_file"],
        datasets = expand("{phenotype}/deeprvat/association_dataset.pkl",
                          phenotype=phenotypes),
        chunks =  expand(
            'burdens/log/burdens_averaging_{chunk}.finished',
            chunk=range(n_avg_chunks)
        ),
    params:
        phenotypes = " ".join([f"--phenotype {p} {p}/deeprvat/association_dataset.pkl {p}/deeprvat/xy"
                               for p in phenotypes]) + " ",
        burdens = burdens,
        genes = burdens.parent / "genes.npy",
        samples = burdens.parent / "sample_ids.zarr",
    output:
        sample_file = "regenie_input/deeprvat_pseudovariants.sample",
        bgen = "regenie_input/deeprvat_pseudovariants.bgen",
    threads: 8
    resources:
        mem_mb = 64000
    log:
        stdout="logs/make_regenie_burdens/make_regenie_burdens.stdout", 
        stderr="logs/make_regenie_burdens/make_regenie_burdens.stderr"
    shell:
        "deeprvat_associate make-regenie-input "
        + debug +
        "--skip-covariates "
        "--skip-phenotypes "
        "--average-repeats "
        "{params.phenotypes} "
        # "{input.dataset} "
        # "{wildcards.phenotype}/deeprvat/burdens "
        "--sample-file {output.sample_file} "
        "--bgen {output.bgen} "
        "--burdens-genes-samples {params.burdens} {params.genes} {params.samples} "
        "{input.gene_file} "
        "{input.gtf_file} "
        + logging_redirct

rule make_regenie_metadata:
    input:
        gene_file = config["association_testing_data"]["dataset_config"]["rare_embedding"]["config"]["gene_file"],
        gtf_file = config["gtf_file"],
        samples = expand('{phenotype}/deeprvat/xy/sample_ids.zarr', phenotype=phenotypes),
        x = expand('{phenotype}/deeprvat/xy/x.zarr', phenotype=phenotypes),
        y = expand('{phenotype}/deeprvat/xy/y.zarr', phenotype=phenotypes),
        datasets = expand("{phenotype}/deeprvat/association_dataset.pkl",
                          phenotype=phenotypes),
    params:
        phenotypes = " ".join([f"--phenotype {p} {p}/deeprvat/association_dataset.pkl "
                               f"{p}/deeprvat/xy"
                               for p in phenotypes]) + " "
    output:
        covariate_file = "regenie_input/covariates.txt",
        phenotype_file = "regenie_input/phenotypes.txt",
    threads: 1
    resources:
        mem_mb = 16000
    log:
        stdout="logs/make_regenie_metadata/make_regenie_metadata.stdout",
        stderr="logs/make_regenie_metadata/make_regenie_metadata.stderr",
    shell:
        "deeprvat_associate make-regenie-input "
        + debug +
        "--skip-burdens "
        "{params.phenotypes}"
        # "{input.dataset} "
        # "{wildcards.phenotype}/deeprvat/burdens "
        "--covariate-file {output.covariate_file} "
        "--phenotype-file {output.phenotype_file} "
        "{input.gene_file} "
        "{input.gtf_file} "
        + logging_redirct


rule average_burdens:
    input:
        'burdens/burdens.zarr'
        if not cv_exp
        else f'burdens/log/{phenotypes[0]}/merging.finished',
    output:
        'burdens/log/burdens_averaging_{chunk}.finished',
    params:
        burdens_in = 'burdens/burdens.zarr',
        burdens_out = 'burdens/burdens_average.zarr',
        repeats = lambda wildcards: ''.join([f'--repeats {r} ' for r in range(int(n_repeats))])
    threads: 1
    resources:
        mem_mb = lambda wildcards, attempt: 4098 + (attempt - 1) * 4098,
    priority: 10,
    log:
        stdout="logs/average_burdens/average_burdens_{chunk}.stdout", 
        stderr="logs/average_burdens/average_burdens_{chunk}.stderr"
    shell:
        ' && '.join([
            ('deeprvat_associate  average-burdens '
             '--n-chunks ' + str(n_avg_chunks) + ' '
             '--chunk {wildcards.chunk} '
             '{params.repeats} '
             '--agg-fct mean  '  #TODO remove this
             '{params.burdens_in} '
             '{params.burdens_out} '
             + logging_redirct),
            'touch {output}'
        ])

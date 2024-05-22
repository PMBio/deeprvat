configfile: "deeprvat_config.yaml"

debug_flag = config.get('debug', False)
debug = '--debug ' if debug_flag else ''

# n_repeats = config['n_repeats']

phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes

n_burden_chunks = config.get('n_burden_chunks', 1) if not debug_flag else 2

regenie_config_step1 = config["regenie"]["step_1"]
regenie_config_step2 = config["regenie"]["step_2"]
regenie_step1_bsize = regenie_config_step1["bsize"]
regenie_step2_bsize = regenie_config_step2["bsize"]
regenie_njobs = regenie_config_step1.get("njobs", 1)
regenie_joblist = range(1, regenie_njobs)


wildcard_constraints:
    job="\d+"


# rule evaluate:
#     input:
#         associations = expand('{{phenotype}}/deeprvat/mean_agg_results/burden_associations.parquet',
#                               repeat=range(n_repeats)),
#         config = '{phenotype}/deeprvat/hpopt_config.yaml',
#     output:
#         "{phenotype}/deeprvat/eval/significant.parquet",
#         "{phenotype}/deeprvat/eval/all_results.parquet"
#     threads: 1
#     shell:
#         'deeprvat_evaluate '
#         + debug +
#         '--use-seed-genes '
#         '--n-repeats {n_repeats} '
#         '--correction-method FDR '
#         '{input.associations} '
#         '{input.config} '
#         '{wildcards.phenotype}/deeprvat/eval'

rule all_regenie:
    input:
        expand('{phenotype}/deeprvat/mean_agg_results/burden_associations.parquet',
               phenotype=phenotypes),

rule convert_regenie_output:
    input:
        expand("regenie_output/step2/deeprvat_{phenotype}.regenie",
               phenotype=phenotypes)
    output:
        expand('{phenotype}/deeprvat/mean_agg_results/burden_associations.parquet',
               phenotype=phenotypes)
    params:
        pheno_options = " ".join([
            f"--phenotype {phenotype} regenie_output/step2/deeprvat_{phenotype}.regenie "
            f"{phenotype}/deeprvat/mean_agg_results/burden_associations.parquet"
        for phenotype in phenotypes]),
        gene_file = config["data"]["dataset_config"]["rare_embedding"]["config"]["gene_file"]
    threads: 1
    resources:
        mem_mb = 2048
    shell:
        "deeprvat_associate convert-regenie-output "
        "{params.pheno_options} "
        "{params.gene_file}"

rule regenie_step2:
    input:
        sample_file = "regenie_input/deeprvat_pseudovariants.sample",
        bgen = "regenie_input/deeprvat_pseudovariants.bgen",
        covariate_file = "regenie_input/covariates.txt",
        phenotype_file = "regenie_input/phenotypes.txt",
        step1_loco = expand("regenie_output/step1/deeprvat_{pheno_num}.loco",
                            pheno_num=range(1, len(phenotypes) + 1)),
        step1_predlist = "regenie_output/step1/deeprvat_pred.list"
        # step1_loco = expand("regenie_output/step1/deeprvat_l1_{pheno_number}.loco",
        #                     pheno_number=range(len(phenotypes))),
        # step1_predlist = "regenie_output/step1/deeprvat_l1_pred.list",
    output:
        expand("regenie_output/step2/deeprvat_{phenotype}.regenie",
               phenotype=phenotypes)
    threads: 16
    resources:
        mem_mb = 4096
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
        "--out regenie_output/step2/deeprvat"

rule regenie_step1:
    input:
        bgen = regenie_config_step1['bgen'],
        sample_file = "regenie_input/deeprvat_pseudovariants.sample",
        snplist = regenie_config_step1["snplist"],
        covariate_file = "regenie_input/covariates.txt",
        phenotype_file = "regenie_input/phenotypes.txt",
    output:
        expand("regenie_output/step1/deeprvat_{pheno_num}.loco",
               pheno_num=range(1, len(phenotypes) + 1)),
        "regenie_output/step1/deeprvat_pred.list"
    threads: 24
    resources:
        mem_mb = 16000
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
        "--out regenie_output/step1/deeprvat ; "
        "rm -rf regenie_step1_tmp"


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
        gene_file = config["data"]["dataset_config"]["rare_embedding"]["config"]["gene_file"],
        gtf_file = config["gtf_file"],
        burdens = [f'{phenotype}/deeprvat/burdens/chunk{chunk}.' +
                   ("finished" if phenotype == phenotypes[0] else "linked")
                   for phenotype in phenotypes
                   for chunk in range(n_burden_chunks)],
        datasets = expand("{phenotype}/deeprvat/association_dataset.pkl",
                          phenotype=phenotypes),
    params:
        phenotypes = " ".join([f"--phenotype {p} {p}/deeprvat/association_dataset.pkl {p}/deeprvat/burdens"
                               for p in phenotypes]) + " "
    output:
        bgen = "regenie_input/deeprvat_pseudovariants.bgen",
    threads: 8
    resources:
        mem_mb = 64000
    shell:
        "deeprvat_associate make-regenie-input "
        + debug +
        "--skip-samples "
        "--skip-covariates "
        "--skip-phenotypes "
        "--average-repeats "
        "{params.phenotypes}"
        # "{input.dataset} "
        # "{wildcards.phenotype}/deeprvat/burdens "
        "--bgen {output.bgen} "
        "{input.gene_file} "
        "{input.gtf_file} "

rule make_regenie_metadata:
    input:
        gene_file = config["data"]["dataset_config"]["rare_embedding"]["config"]["gene_file"],
        gtf_file = config["gtf_file"],
        burdens = [f'{phenotype}/deeprvat/burdens/chunk{chunk}.' +
                   ("finished" if phenotype == phenotypes[0] else "linked")
                   for phenotype in phenotypes
                   for chunk in range(n_burden_chunks)],
        datasets = expand("{phenotype}/deeprvat/association_dataset.pkl",
                          phenotype=phenotypes),
    params:
        phenotypes = " ".join([f"--phenotype {p} {p}/deeprvat/association_dataset.pkl {p}/deeprvat/burdens"
                               for p in phenotypes]) + " "
    output:
        sample_file = "regenie_input/deeprvat_pseudovariants.sample",
        covariate_file = "regenie_input/covariates.txt",
        phenotype_file = "regenie_input/phenotypes.txt",
    threads: 1
    resources:
        mem_mb = 16000
    shell:
        "deeprvat_associate make-regenie-input "
        + debug +
        "--skip-burdens "
        "{params.phenotypes}"
        # "{input.dataset} "
        # "{wildcards.phenotype}/deeprvat/burdens "
        "--sample-file {output.sample_file} "
        "--covariate-file {output.covariate_file} "
        "--phenotype-file {output.phenotype_file} "
        "{input.gene_file} "
        "{input.gtf_file} "

#configfile: "config.yaml" # TODO SHOULD THIS BE HERE?

debug_flag = config.get('debug', False)
debug = '--debug ' if debug_flag else ''

# n_repeats = config['n_repeats']

phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes

n_burden_chunks = config.get('n_burden_chunks', 1) if not debug_flag else 2

burdens = Path(config["burden_file"])

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
#         associations = expand('{{phenotype}}/deeprvat/average_regression_results/burden_associations.parquet',
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
        expand('{phenotype}/deeprvat/average_regression_results/burden_associations.parquet',
               phenotype=phenotypes),

rule convert_regenie_output:
    input:
        "regenie_output/step2/deeprvat_{phenotype}.regenie",
    output:
        '{phenotype}/deeprvat/average_regression_results/burden_associations.parquet',
    params:
        gene_file = config["data"]["dataset_config"]["rare_embedding"]["config"]["gene_file"]
    threads: 1
    resources:
        mem_mb = 2048
    log:
        stdout="logs/convert_regenie_output/convert_regenie_output.stdout",
        stderr="logs/convert_regenie_output/convert_regenie_output.stderr"
    shell:
        "deeprvat_associate convert-regenie-output "
        "--phenotype {wildcards.phenotype} {input} {output} "
        "{params.gene_file}"

rule regenie_step2:
    input:
        sample_file = "regenie_input/deeprvat_pseudovariants.sample",
        bgen = "regenie_input/deeprvat_pseudovariants.bgen",
        covariate_file = "{phenotype}/deeprvat/regenie_covariates.txt",
        phenotype_file = "{phenotype}/deeprvat/regenie_phenotypes.txt",
        # step1_loco = expand("regenie_output/step1/deeprvat_{pheno_num}.loco",
        #                     pheno_num=range(1, len(phenotypes) + 1)),
        step1_predlist = "regenie_output/step1/deeprvat_pred.list"
        # step1_loco = expand("regenie_output/step1/deeprvat_l1_{pheno_number}.loco",
        #                     pheno_number=range(len(phenotypes))),
        # step1_predlist = "regenie_output/step1/deeprvat_l1_pred.list",
    output:
        "regenie_output/step2/deeprvat_{phenotype}.regenie"
    threads: 16
    resources:
        mem_mb = lambda wildcards, attempt: 32768 * attempt
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
        "--condition-list {wildcards.phenotype}/deeprvat/condition_list_rsid.txt "
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
        chunks =  expand(
            'burdens/log/burdens_averaging_{chunk}.finished',
            chunk=range(n_avg_chunks)
        ),
        genes = burdens.parent / "genes.npy",
        samples = burdens.parent / "sample_ids.zarr",
        datasets = expand("{phenotype}/deeprvat/association_dataset.pkl",
                          phenotype=phenotypes),
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

rule make_regenie_step2_metadata:
    input:
        gene_file = config["data"]["dataset_config"]["rare_embedding"]["config"]["gene_file"],
        gtf_file = config["gtf_file"],
        samples = '{phenotype}/deeprvat/xy/sample_ids.zarr',
        x = '{phenotype}/deeprvat/xy/x.zarr',
        y = '{phenotype}/deeprvat/xy/y.zarr',
        datasets = "{phenotype}/deeprvat/association_dataset.pkl",
    output:
        covariate_file = "{phenotype}/deeprvat/regenie_covariates.txt",
        phenotype_file = "{phenotype}/deeprvat/regenie_phenotypes.txt",
    threads: 1
    resources:
        mem_mb = 16000
    log:
        stdout="logs/make_regenie_step2_metadata/make_regenie_step2_metadata.stdout",
        stderr="logs/make_regenie_step2_metadata/make_regenie_step2_metadata.stderr",
    shell:
        "deeprvat_associate make-regenie-input "
        + debug +
        "--skip-burdens "
        "--phenotype {wildcards.phenotype} {wildcards.phenotype}/deeprvat/association_dataset.pkl {wildcards.phenotype}/deeprvat/xy "
        # "{input.dataset} "
        # "{wildcards.phenotype}/deeprvat/burdens "
        "--covariate-file {output.covariate_file} "
        "--phenotype-file {output.phenotype_file} "
        "{input.gene_file} "
        "{input.gtf_file} "

rule make_regenie_step1_metadata:
    input:
        gene_file = config["data"]["dataset_config"]["rare_embedding"]["config"]["gene_file"],
        gtf_file = config["gtf_file"],
        chunks =  expand(
            'burdens/log/burdens_averaging_{chunk}.finished',
            chunk=range(n_avg_chunks)
        ),
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
    log:
        stdout="logs/make_regenie_step1_metadata/make_regenie_step1_metadata.stdout",
        stderr="logs/make_regenie_step1_metadata/make_regenie_step1_metadata.stderr",
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
             '{params.burdens_out}'),
            'touch {output}'
        ])

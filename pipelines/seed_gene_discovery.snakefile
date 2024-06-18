from os.path import exists

if not exists('./sg_discovery_config.yaml'):
    if not config: #--configfile argument was not passed
        print("Generating sg_discovery_config.yaml...")
        from deeprvat.deeprvat.config import create_sg_discovery_config
        create_sg_discovery_config('seed_gene_discovery_input_config.yaml') #Name your input config here
        print("     Finished.")

configfile: "sg_discovery_config.yaml"

debug_flag = config.get("debug", False)
phenotypes = config["phenotypes"]
vtypes = config.get("variant_types", ["plof"])
ttypes = config.get("test_types", ["burden"])
rare_maf = config.get("rare_maf", 0.001)
n_chunks_missense = 15
n_chunks_plof = 4
debug = "--debug " if debug_flag else ""
persist_burdens = "--persist-burdens" if config.get("persist_burdens", False) else ""

conda_check = 'conda info | grep "active environment"'

wildcard_constraints:
    repeat="\d+",


rule all:
    input:
        expand("{phenotype}/eval/burden_associations.parquet", phenotype=phenotypes),


rule evaluate:
    input:
        associations=expand(
            "{{phenotype}}/{vtype}/{ttype}/results/burden_associations.parquet",
            vtype=vtypes,
            ttype=ttypes,
        ),
        config="{phenotype}/plof/config.yaml",  #simply any config
    output:
        out_file="{phenotype}/eval/burden_associations.parquet",
    threads: 1
    resources:
        mem_mb=16000,
        load=16000,
    shell:
        " && ".join(
        [
            conda_check,
        "seed_gene_evaluate evaluate "
                "{input.config} "
                "{input.associations} "
                "{output.out_file}",
            ]
        )


rule all_regression:
    input:
        expand(
            "{phenotype}/{vtype}/{ttype}/results/burden_associations.parquet",
            phenotype=phenotypes,
            vtype=vtypes,
            ttype=ttypes,
        ),


rule combine_regression_chunks_plof:
    input:
        train=expand(
            "{{phenotype}}/plof/{{ttype}}/results/burden_associations_chunk{chunk}.parquet",
            chunk=range(n_chunks_plof),
        ),
    output:
        train="{phenotype}/plof/{ttype}/results/burden_associations.parquet",
    threads: 1
    resources:
        mem_mb=2048,
        load=2000,
    shell:
        " && ".join(
            [
                conda_check,
                "seed_gene_pipeline combine-results " "{input.train} " "{output.train}",
            ]
        )

rule combine_regression_chunks_missense:
    input:
        train=expand(
            "{{phenotype}}/missense/{{ttype}}/results/burden_associations_chunk{chunk}.parquet",
            chunk=range(n_chunks_missense),
        ),
    output:
        train="{phenotype}/missense/{ttype}/results/burden_associations.parquet",
    threads: 1
    resources:
        mem_mb=2048,
        load=2000,
    shell:
        " && ".join(
            [
                conda_check,
                "seed_gene_pipeline combine-results " "{input.train} " "{output.train}",
            ]
        )


rule all_regression_results_plof:
    input:
        expand(
            "{phenotype}/plof/{ttype}/results/burden_associations_chunk{chunk}.parquet",
            phenotype=phenotypes,
            vtype=vtypes,
            ttype=ttypes,
            chunk=range(n_chunks_plof),
        ),

rule all_regression_results_missense:
    input:
        expand(
            "{phenotype}/missense/{ttype}/results/burden_associations_chunk{chunk}.parquet",
            phenotype=phenotypes,
            vtype=vtypes,
            ttype=ttypes,
            chunk=range(n_chunks_missense),
        ),

rule regress_plof:
    input:
        data="{phenotype}/plof/association_dataset_full.pkl",
        dataset="{phenotype}/plof/association_dataset_pickled.pkl",
        config="{phenotype}/plof/config.yaml",
    output:
        out_path=temp(
            "{phenotype}/plof/{ttype}/results/burden_associations_chunk{chunk}.parquet"
        ),
    threads: 1
    priority: 30
    resources:
        mem_mb=lambda wildcards, attempt: 20000 + 2000 * attempt,
        load=8000,
        # gpus = 1
    shell:
        " && ".join(
        [
            conda_check,
            (
        "seed_gene_pipeline run-association "
                    + debug
                    + " --n-chunks "
                    + str(n_chunks_plof)
                    + " "
                    "--chunk {wildcards.chunk} "
                    "--dataset-file {input.dataset} "
                    "--data-file {input.data} " + persist_burdens + " "
                    " {input.config} "
                    "plof "
                    "{wildcards.ttype} "
                    "{output.out_path}"
                ),
            ]
        )

rule regress_missense:
    input:
        data="{phenotype}/missense/association_dataset_full.pkl",
        dataset="{phenotype}/missense/association_dataset_pickled.pkl",
        config="{phenotype}/missense/config.yaml",
    output:
        out_path=temp(
            "{phenotype}/missense/{ttype}/results/burden_associations_chunk{chunk}.parquet"
        ),
    threads: 1
    priority: 30
    resources:
        mem_mb=lambda wildcards, attempt: 30000 + 6000 * attempt,
        load=8000,
        # gpus = 1
    shell:
        " && ".join(
        [
            conda_check,
            (
        "seed_gene_pipeline run-association "
                    + debug
                    + " --n-chunks "
                    + str(n_chunks_missense)
                    + " "
                    "--chunk {wildcards.chunk} "
                    "--dataset-file {input.dataset} "
                    "--data-file {input.data} " + persist_burdens + " "
                    " {input.config} "
                    "missense "
                    "{wildcards.ttype} "
                    "{output.out_path}"
                ),
            ]
        )


rule all_association_dataset:
    priority: 40
    input:
        expand(
            "{phenotype}/{vtype}/association_dataset_full.pkl",
            phenotype=phenotypes,
            vtype=vtypes,
        ),
        expand(
            "{phenotype}/{vtype}/association_dataset_pickled.pkl",
            phenotype=phenotypes,
            vtype=vtypes,
        ),


rule association_dataset:
    input:
        "{phenotype}/{vtype}/config.yaml",
    output:
        full="{phenotype}/{vtype}/association_dataset_full.pkl",
        pickled="{phenotype}/{vtype}/association_dataset_pickled.pkl",
    threads: 1
    priority: 40
    resources:
        mem_mb=40000,
        load=16000,
    shell:
        " && ".join(
        [
            conda_check,
            (
        "seed_gene_pipeline make-dataset "
                    + debug
                    + "--pickled-dataset-file {output.pickled} "
                    "{input} "
                    "{output.full}"
                ),
            ]
        )


rule config:
    input:
        config="sg_discovery_config.yaml",
    output:
        "{phenotype}/{vtype}/config.yaml",
    params:
        rare_maf=str(rare_maf),
        maf_column="MAF",
    threads: 1
    resources:
        mem_mb=1024,
        load=1000,
    shell:
        " && ".join(
        [
            conda_check,
            (
        "seed_gene_pipeline update-config "
                    + "--phenotype {wildcards.phenotype} "
                    + "--variant-type {wildcards.vtype} "
                    + "--maf-column {params.maf_column} "
                    + "--rare-maf "
                    + "{params.rare_maf}"
                    + " {input.config} "
                    + "{output}"
                ),
            ]
        )

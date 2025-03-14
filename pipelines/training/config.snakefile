rule config:
    input:
        data_config="deeprvat_config.yaml",
        baseline=lambda wildcards: [
            str(
                Path(r["base"])
                / wildcards.phenotype
                / r["type"]
                / "eval/burden_associations.parquet"
            )
            for r in config["baseline_results"]["options"]
        ]
        if wildcards.phenotype in training_phenotypes
        else [],
    output:
        # seed_genes = '{phenotype}/deeprvat/seed_genes.parquet',
        data_config="{phenotype}/deeprvat/config.yaml",
        # baseline = '{phenotype}/deeprvat/baseline_results.parquet',
    threads: 1
    resources:
        mem_mb=1024,
        load=1000,
    params:
        baseline_results=lambda wildcards, input: "".join(
            [f"--baseline-results {b} " for b in input.baseline]
        )
        if wildcards.phenotype in training_phenotypes
        else " ",
        seed_genes_out=lambda wildcards: f"--seed-genes-out {wildcards.phenotype}/deeprvat/seed_genes.parquet"
        if wildcards.phenotype in training_phenotypes
        else " ",
        baseline_out=lambda wildcards: f"--baseline-results-out  {wildcards.phenotype}/deeprvat/baseline_results.parquet"
        if wildcards.phenotype in training_phenotypes
        else " ",
        association_only=lambda wildcards: f"--association-only"
        if wildcards.phenotype not in training_phenotypes
        else " ",
    log: 
        stdout="logs/config/config_{phenotype}.stdout", 
        stderr="logs/config/config_{phenotype}.stderr"
    shell:
        (
            "deeprvat_config update-config "
            "--phenotype {wildcards.phenotype} "
            "{params.association_only} "
            "{params.baseline_results} "
            "{params.baseline_out} "
            "{params.seed_genes_out} "
            "{input.data_config} "
            "{output.data_config} "
            + logging_redirct
        )

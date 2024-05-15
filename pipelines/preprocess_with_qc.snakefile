include: "preprocessing/preprocess.snakefile"
include: "preprocessing/qc.snakefile"


rule all:
    input:
        combined_genotypes=rules.combine_genotypes.output,
        variants_tsv=rules.add_variant_ids.output.variants,
        variants_parquet=rules.create_parquet_variant_ids.output.variants,


rule preprocess:
    input:
        variants=rules.add_variant_ids.output.variants,
        variants_parquet=rules.create_parquet_variant_ids.output.variants,
        samples=rules.extract_samples.output,
        sparse_tg=expand(rules.sparsify.output.tsv, vcf_stem=vcf_stems),
        qc_varmiss=expand(rules.qc_varmiss.output, vcf_stem=vcf_stems),
        qc_hwe=expand(rules.qc_hwe.output, vcf_stem=vcf_stems),
        qc_read_depth=expand(rules.qc_read_depth.output, vcf_stem=vcf_stems),
        qc_allelic_imbalance=expand(
            rules.qc_allelic_imbalance.output, vcf_stem=vcf_stems
        ),
        qc_indmiss_samples=rules.process_individual_missingness.output,
    output:
        expand(preprocessed_dir / "genotypes_chr{chr}.h5", chr=chromosomes),
    shell:
        " ".join(
            [
                f"{preprocessing_cmd}",
                "process-sparse-gt",
                f"--exclude-variants {qc_allelic_imbalance_dir}",
                f"--exclude-variants {qc_hwe_dir}",
                f"--exclude-variants {qc_varmiss_dir}",
                f"--exclude-variants {qc_duplicate_vars_dir}",
                f"--exclude-calls {qc_read_depth_dir}",
                f"--exclude-samples {qc_filtered_samples_dir}",
                "--chromosomes ",
                ",".join(str(chr) for chr in set(chromosomes)),
                "{input.variants_parquet}",
                "{input.samples}",
                f"{sparse_dir}",
                f"{preprocessed_dir/ 'genotypes'}",
            ]
        )

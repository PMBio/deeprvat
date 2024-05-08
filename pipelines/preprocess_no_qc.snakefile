include: "preprocessing/preprocess.snakefile"


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
    output:
        expand(preprocessed_dir / "genotypes_chr{chr}.h5", chr=chromosomes),
    shell:
        " ".join(
            [
                f"{preprocessing_cmd}",
                "process-sparse-gt",
                f"--exclude-variants {qc_duplicate_vars_dir}",
                "--chromosomes ",
                ",".join(str(chr) for chr in set(chromosomes)),
                "{input.variants_parquet}",
                "{input.samples}",
                f"{sparse_dir}",
                f"{preprocessed_dir/ 'genotypes'}",
            ]
        )

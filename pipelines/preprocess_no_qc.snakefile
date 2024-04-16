include: "preprocessing/preprocess.snakefile"


rule all:
    input:
        preprocessed_dir / "genotypes.h5",
        norm_variants_dir / "variants.tsv.gz",
        variants=norm_variants_dir / "variants.parquet",


rule preprocess_no_qc:
    input:
        variants=norm_variants_dir / "variants.tsv.gz",
        variants_parquet=norm_variants_dir / "variants.parquet",
        samples=norm_dir / "samples_chr.csv",
        sparse_tg=expand(sparse_dir / "{vcf_stem}.tsv.gz", vcf_stem=vcf_stems),
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
                f"--threads {preprocess_threads}",
                "{input.variants}",
                "{input.samples}",
                f"{sparse_dir}",
                f"{preprocessed_dir / 'genotypes'}",
            ]
        )


include: "preprocessing/preprocess.snakefile"
include: "preprocessing/qc.snakefile"


rule all:
    input:
        preprocessed_dir / "genotypes.h5",
        norm_variants_dir / "variants.tsv.gz",
        variants=norm_variants_dir / "variants.parquet",


rule preprocess_with_qc:
    input:
        variants=norm_variants_dir / "variants.tsv.gz",
        variants_parquet=norm_variants_dir / "variants.parquet",
        samples=norm_dir / "samples_chr.csv",
        sparse_tg=expand(sparse_dir / "{vcf_stem}.tsv.gz", vcf_stem=vcf_stems),
        qc_varmiss=expand(qc_varmiss_dir / "{vcf_stem}.tsv.gz", vcf_stem=vcf_stems),
        qc_hwe=expand(qc_hwe_dir / "{vcf_stem}.tsv.gz", vcf_stem=vcf_stems),
        qc_read_depth=expand(
            qc_read_depth_dir / "{vcf_stem}.tsv.gz", vcf_stem=vcf_stems
        ),
        qc_allelic_imbalance=expand(
            qc_allelic_imbalance_dir / "{vcf_stem}.tsv.gz", vcf_stem=vcf_stems
        ),
        qc_filtered_samples=qc_filtered_samples_dir,
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
                f"{preprocessed_dir / 'genotypes'}",
            ]
        )

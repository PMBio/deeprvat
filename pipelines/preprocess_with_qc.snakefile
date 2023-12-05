
include: "preprocessing/preprocess.snakefile"
include: "preprocessing/qc.snakefile"


rule all:
    input:
        preprocessed_dir / "genotypes.h5",
        norm_variants_dir / "variants.tsv.gz",
        variants=norm_variants_dir / "variants.parquet",

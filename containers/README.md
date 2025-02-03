# Build Container
apptainer build deeprvat_preprocessing.sif apptainer_deeprvat_preprocessing.def
apptainer build deeprvat.sif apptainer_deeprvat.def

NOTE: deeprvat image ~= 5 GB, deeprvat_preprocessing image ~= 1 GB

# Verify Working Image
apptainer exec deeprvat_preprocessing.sif pip list | grep "deeprvat"

# Run Preprocessing
cd ./my_exp_dir
    .
    ├── data
    │   └── vcf
    │       ├── test_vcf_data_c21_b1.vcf.gz
    │       └── test_vcf_data_c22_b1.vcf.gz
    ├── deeprvat_preprocess_config.yaml
    ├── deeprvat_preprocessing.sif
    └── vcf_files_list.txt


[~/my_exp_dir]$ apptainer run deeprvat_preprocessing.sif snakemake -j 1 --snakefile /opt/deeprvat/pipelines/preprocess_no_qc.snakefile --configfile deeprvat_preprocess_config.yaml --ri -n


# Run Training + Association Testing
cd ./my_exp_dir
    .
    ├── annotations.parquet
    ├── baseline_results
    │   ├── Apolipoprotein_A
    │   │   ├── missense
    │   │   │   ├── burden
    │   │   │   │   └── eval
    │   │   │   │       └── burden_associations.parquet
    │   │   │   └── skat
    │   │   │       └── eval
    │   │   │           └── burden_associations.parquet
    │   │   └── plof
    │   │       ├── burden
    │   │       │   └── eval
    │   │       │       └── burden_associations.parquet
    │   │       └── skat
    │   │           └── eval
    │   │               └── burden_associations.parquet
    │   ├── Calcium
    │   │   ├── missense
    │   │   │   ├── burden
    │   │   │   │   └── eval
    │   │   │   │       └── burden_associations.parquet
    │   │   │   └── skat
    │   │   │       └── eval
    │   │   │           └── burden_associations.parquet
    │   │   └── plof
    │   │       ├── burden
    │   │       │   └── eval
    │   │       │       └── burden_associations.parquet
    │   │       └── skat
    │   │           └── eval
    │   │               └── burden_associations.parquet
    │   ├── Cholesterol
    │   │   ├── missense
    │   │   │   ├── burden
    │   │   │   │   └── eval
    │   │   │   │       └── burden_associations.parquet
    │   │   │   └── skat
    │   │   │       └── eval
    │   │   │           └── burden_associations.parquet
    │   │   └── plof
    │   │       ├── burden
    │   │       │   └── eval
    │   │       │       └── burden_associations.parquet
    │   │       └── skat
    │   │           └── eval
    │   │               └── burden_associations.parquet
    │   ├── Platelet_count
    │   │   ├── missense
    │   │   │   ├── burden
    │   │   │   │   └── eval
    │   │   │   │       └── burden_associations.parquet
    │   │   │   └── skat
    │   │   │       └── eval
    │   │   │           └── burden_associations.parquet
    │   │   └── plof
    │   │       ├── burden
    │   │       │   └── eval
    │   │       │       └── burden_associations.parquet
    │   │       └── skat
    │   │           └── eval
    │   │               └── burden_associations.parquet
    ├── deeprvat_config.yaml
    ├── deeprvat.sif
    ├── gencode.v38.basic.annotation.gtf.gz
    ├── genotypes.h5
    ├── phenotypes.parquet
    ├── protein_coding_genes.parquet
    └── variants.parquet

[~/my_exp_dir]$ apptainer run deeprvat.sif snakemake -j 1 --snakefile /opt/deeprvat/pipelines/training_association_testing.snakefile --configfile deeprvat_config.yaml --ri -n

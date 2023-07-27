# Seed gene discovery

This pipeline discovers *seed genes* for DeepRVAT training. The pipeline runs SKAT and burden tests for missense and pLOF variants, weighting variants with Beta(MAF,1,25). To run the tests, we use the `Scoretest` from the [SEAK](https://github.com/HealthML/seak) package (has to be installed from github).

To run the pipeline, an experiment directory with the `config.yaml` has to be created. An `lsf.yaml` file specifiying the compute resources for each rule in `seed_gene_discovery.snakefile` might also be needed depending on your system (see as an example the `lsf.yaml` file in this directory). 

## Input data

The experiment directory in addition requires to have the same input data as specified for [DeepRVAT](https://github.com/PMBio/deeprvat/tree/main/README.md), including
- `annotations.parquet`
- `protein_coding_genes.parquet`
- `genotypes.h5`
- `variants.parquet`
- `phenotypes.parquet`

The `annotations.parquet` data frame should have the following columns:

- id (variant id, **should be the index column**)
- gene_ids (list) gene(s) the variant is assigned to
- is_plof (binary, indicating if the variant is loss of function)
- Consequence_missense_variant: 
- MAF:  Maximum of the MAF in the UK Biobank cohort and in gnomAD release 3.0 (non-Finnish European population) can also be changed by using the --maf-column {maf_col_name} flag for the rule config and replacing MAF in the config.yaml with the {maf_col_name} but it must contain the string '_AF', '_MAF' OR '^MAF'

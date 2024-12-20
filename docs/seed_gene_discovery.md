# Seed gene discovery

This pipeline discovers *seed genes* for DeepRVAT training. The pipeline runs SKAT and burden tests for missense and pLOF variants, weighting variants with Beta(MAF,1,25). To run the tests, we use the `Scoretest` from the [SEAK](https://github.com/HealthML/seak) package (has to be installed from github).

To run the pipeline, an experiment directory with the `seed_gene_discovery_input_config.yaml` has to be created. See [example file](https://github.com/PMBio/deeprvat/blob/main/example/config/seed_gene_discovery_input_config.yaml). When the seed gene discovery pipeline is executed, a comprehensive `sg_discovery_config.yaml` file is automatically generated based on the `seed_gene_discovery_input_config.yaml` input.

(input-data)=
## Input data

The experiment directory in addition requires to have the same input data as specified for [DeepRVAT](#input-data-formats), including
- `annotations.parquet`
- `protein_coding_genes.parquet`
- `genotypes.h5`
- `variants.parquet`
- `phenotypes.parquet`
- `seed_gene_discovery_input_config.yaml` (use [this](https://github.com/PMBio/deeprvat/blob/main/example/config/seed_gene_discovery_input_config.yaml) as a template)

The `annotations.parquet` dataframe, generated by the annotation pipeline, can be utilized. To indicate if a variant is a loss of function (pLoF) variant, a column `is_plof` has to be added with values 0 or 1. We recommend to set this to `1` if the variant has been classified as any of these VEP consequences: `["splice_acceptor_variant", "splice_donor_variant", "frameshift_variant", "stop_gained", "stop_lost", "start_lost"]`.

(configuration-file)=
## Configuration file

You can restrict to only missense variants (identified by the `Consequence_missense_variant` column in `annotations.parquet` ) or pLoF variants (`is_plof` column) via 
```
variant_types:
    - missense
    - plof
```
and specify the test types that will be run via 
```
test_types:
   - skat
   - burden
```

The minor allele frequency threshold is set via 

```
rare_maf: 0.001
```

You can specify further test details in the test config using the following parameters:

- `center_genotype` center the genotype matrix (True or False)
- `neglect_homozygous` Should the genotype value for homozyoogus variants be 1 (True) or 2 (False)
- `collapse_method` Burden test collapsing method. Supported are `sum` and `max`
- `var_weight` Variant weighting function. Supported are `beta_maf` (Beta(MAF, 1, 25)) or `sift_polpyen` (mean of 1-SIFT and Polyphen2 score)
- `min_mac` minimum expected allele count for genes to be included. This is the cumulative allele frequency of variants in the burden mask (e.g., pLoF variants) for a given gene (e.g. pLoF variants) multiplied by the cohort size or number of cases for quantitative and binary traits, respectively. 

```
test_config:
    center_genotype: True
    neglect_homozygous: False
    collapse_method: sum #collapsing method for burden, 
    var_weight_function: beta_maf 
    min_mac: 50 # minimum expected allele count

```

## Running the seed gene discovery pipeline

In a directory with all the [input data](#input-data) required and your [configuration file](#configuration-file) set up, run: 

```
[path_to_deeprvat]/pipelines/seed_gene_discovery.snakefile
```

Replace `[path_to_deeprvat]` with the path to your clone of the repository.


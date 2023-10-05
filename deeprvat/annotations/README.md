# DeepRVAT Annotation pipeline

This pipeline is based on [snakemake](https://snakemake.readthedocs.io/en/stable/). It uses [bcftools + samstools](https://www.htslib.org/), as well as [perl](https://www.perl.org/), [deepRiPe](https://ohlerlab.mdc-berlin.de/software/DeepRiPe_140/) and [deepSEA](http://deepsea.princeton.edu/) as well as [VEP](http://www.ensembl.org/info/docs/tools/vep/index.html), including plugins for [primateAI](https://github.com/Illumina/PrimateAI) and  [spliceAI](https://github.com/Illumina/SpliceAI). DeepRiPe annotations were acquired using [faatpipe repository by HealthML](https://github.com/HealthML/faatpipe)[[1]](#1) and DeepSea annotations were calculated using [kipoi-veff2](https://github.com/kipoi/kipoi-veff2)[[2]](#2), abSplice scores were computet using [abSplice](https://github.com/gagneurlab/absplice/)[[3]](#3)

![dag](https://github.com/PMBio/deeprvat/assets/23211603/d483831e-3558-4e21-9845-4b62ad4eecc3)
*Figure 1: Example DAG of annoation pipeline using only two bcf files as input.*

## Input

The pipeline uses left-normalized bcf files containing variant information, a reference fasta file as well as a text file that maps data blocks to chromosomes as input. It is expected that the bcf files contain the columns "CHROM" "POS" "ID" "REF" and "ALT". Any other columns, including genotype information are stripped from the data before annotation tools are used on the data. The variants may be split into several vcf files for each chromosome and each "block" of data. The filenames should then contain the corresponding chromosome and block number. The pattern of the file names, as well as file structure may be specified in the corresponding [config file](config/deeprvat_annotation_config.yaml).

## Requirements 
BCFtools as well as HTSlib should be installed on the machine, 
- [CADD](https://github.com/kircherlab/CADD-scripts/tree/master/src/scripts) as well as 
- [VEP](http://www.ensembl.org/info/docs/tools/vep/script/vep_download.html),  
- [absplice](https://github.com/gagneurlab/absplice/tree/master), 
- [kipoi-veff2](https://github.com/kipoi/kipoi-veff2)
- [faatpipe](https://github.com/HealthML/faatpipe), and the
- [vep-plugins repository](https://github.com/Ensembl/VEP_plugins/)

will be installed by the pipeline together with the [plugins](https://www.ensembl.org/info/docs/tools/vep/script/vep_plugins.html) for primateAI and spliceAI. Annotation data for CADD, spliceAI and primateAI should be downloaded. The path to the data may be specified in the corresponding [config file](config/deeprvat_annotation_config.yaml). 
Download path:
- [CADD](http://cadd.gs.washington.edu/download): "All possible SNVs of GRCh38/hg38" and "gnomad.genomes.r3.0.indel.tsv.gz" incl. their  Tabix Indices
- [SpliceAI](https://basespace.illumina.com/s/otSPW8hnhaZR): "genome_scores_v1.3"/"spliceai_scores.raw.snv.hg38.vcf.gz" and "spliceai_scores.raw.indel.hg38.vcf.gz" 
- [PrimateAI](https://basespace.illumina.com/s/yYGFdGih1rXL) PrimateAI supplementary data/"PrimateAI_scores_v0.2_GRCh38_sorted.tsv.bgz"


## Output

The pipeline outputs one annotation file for VEP, CADD, DeepRiPe, DeepSea and Absplice for each input vcf-file. The tool further creates concatenated files for each tool and one merged file containing Scores from AbSplice, VEP incl. CADD, primateAI and spliceAI as well as principal components from DeepSea and DeepRiPe.

## Configure the annotation pipeline
The snakemake annotation pipeline is configured using a yaml file with the format akin to the [example file](config/deeprvat_annotation_config.yaml).

The config above would use the following directory structure:
```shell

|-- reference
|   |-- fasta file


|-- metadata
|   |-- pvcf_blocks.txt

|-- preprocessing_workdir
|   |--reference
|   |   |-- fasta file
|   |-- norm
|   |   |-- bcf
|   |   |   |-- bcf_input_files
|   |   |   |-- ...
|   |   |-- variants
|   |   |   |-- variants.tsv.gz

|-- output_dir
|   |-- annotations
|   |   |-- tmp

|-- repo_dir
|   |-- ensembl-vep
|   |   |-- cache
|   |   |-- plugins
|   |-- abSplice
|   |-- faatpipe
|   |-- kipoi-veff2

|-- annotation_data
|   |-- cadd
|   |-- spliceAI
|   |-- primateAI



```

Bcf files created by the [preprocessing pipeline](https://github.com/PMBio/deeprvat/blob/Annotations/deeprvat/preprocessing/README.md) are used as input data. 
The pipeline also uses the variant.tsv file as well as the reference file from the preprocesing pipeline. 
The pipeline beginns by installing the repositories needed for the annotations, it will automatically install all repositories in the `repo_dir` folder that can be specified in the config file relative to the annotation working directory.
The text file mapping blocks to chromosomes is stored in `metadata` folder. The output is stored in the `output_dir/annotations` folder and any temporary files in the `tmp` subfolder. All repositories used including VEP with its corresponding cache as well as plugins are stored in `repo_dir/ensempl-vep`.
Data for VEP plugins and the CADD cache are stored in `annotation data`. 

## Running the annotation pipeline
### Preconfiguration
- Inside the annotation directory create a directory `repo_dir` and run the [annotation setup script](setup_annotation_workflow.sh) 
  ```shell
    setup_annotation_workflow.sh repo_dir/ensembl-vep/cache repo_dir/ensembl-vep/Plugins repo_dir
  ```
  or manually clone the repositories mentioned in the [requirements](#requirements) into `repo_dir` and install the needed conda environments with  
    ```shell
    mamba env create -f repo_dir/absplice/environment.yaml
    mamba env create -f repo_dir/kipoi-veff2/environment.minimal.linux.yml
    ```
  If you already have some of the needed repositories on your machine you can edit the paths in the [config](../../pipelines/config/deeprvat_annotation_config.yaml).
  

- Inside the annotation directory create a directory `annotation_dir` and download/link the prescored files for CADD, SpliceAI, and PrimateAI (see [requirements](#requirements))


### Running the pipeline
After configuration and activating the environment run the pipeline using snakemake:

```shell
  snakemake -j <nr_cores> -s annotations.snakemake --configfile config/deeprvat_annotation.config 
```
## Running the annotation pipeline without the preprocessing pipeline

It is possible to run the annotation pipeline without having run the preprocessing prior to that. 
However, the annotation pipeline requires some files from this pipeline that then have to be created manually.
- Left normalized bcf files from the input. These files do not have to contain any genotype information. "chrom, "pos", "ref" and "alt" columns will suffice.
- a reference fasta file will have to be provided
- A tab separated file containing all input variants "chrom, "pos", "ref" and "alt" entries each with a unique id.


## References
<a id="1">[1]</a> Monti, R., Rautenstrauch, P., Ghanbari, M. et al. Identifying interpretable gene-biomarker associations with functionally informed kernel-based tests in 190,000 exomes. Nat Commun 13, 5332 (2022). https://doi.org/10.1038/s41467-022-32864-2

<a id="2">[2]</a> Žiga Avsec et al., “Kipoi: accelerating the community exchange and reuse of predictive models for genomics,” bioRxiv, p. 375345, Jan. 2018, doi: 10.1101/375345.

<a id="3">[3]</a>N. Wagner et al., “Aberrant splicing prediction across human tissues,” Nature Genetics, vol. 55, no. 5, pp. 861–870, May 2023, doi: 10.1038/s41588-023-01373-3.

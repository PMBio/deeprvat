# DeepRVAT annotation pipeline

This pipeline is based on [snakemake](https://snakemake.readthedocs.io/en/stable/). It uses [bcftools + samstools](https://www.htslib.org/), as well as [perl](https://www.perl.org/), [deepRiPe](https://ohlerlab.mdc-berlin.de/software/DeepRiPe_140/) and [deepSEA](http://deepsea.princeton.edu/) as well as [VEP](http://www.ensembl.org/info/docs/tools/vep/index.html), including plugins for [primateAI](https://github.com/Illumina/PrimateAI) and  [spliceAI](https://github.com/Illumina/SpliceAI). DeepRiPe annotations were acquired using [faatpipe repository by HealthML](https://github.com/HealthML/faatpipe)[[1]](#reference-1-target) and DeepSea annotations were calculated using [kipoi-veff2](https://github.com/kipoi/kipoi-veff2)[[2]](#reference-2-target), abSplice scores were computed using [abSplice](https://github.com/gagneurlab/absplice/)[[3]](#reference-3-target). The pipeline was tested on linux and macOS.

![dag](_static/annotations_rulegraph.svg)

*Figure 1: Rule graph of the annotation pipeline.*

## Output 
This pipeline outputs a parquet file including all annotations as well as a file containing IDs to all protein coding genes needed to run DeepRVAT. 
Apart from this the pipeline outputs a PCA transformation matrix for deepSEA as well as means and standard deviations used to standardize deepSEA scores before PCA analysis. This is helpful to recreate results using a different dataset. 
Furthermore, the pipeline outputs one annotation file for VEP, CADD, DeepRiPe, DeepSea and Absplice for each input vcf-file. The tool then creates concatenates the files, performs PCA on the deepSEA scores and merges the result into a single file. 

## Input

The pipeline uses left-normalized bcf files containing variant information (e.g. the bcf files created by the [preprocessing pipeline](https://deeprvat.readthedocs.io/en/latest/preprocessing.html)) , a reference fasta file, and a gtf file for gene information. It is expected that the bcf files contain the columns "CHROM" "POS" "ID" "REF" and "ALT". 
Any other columns, including genotype information, are stripped from the data before annotation tools are used on the data. The variants may be split into several vcf files for each chromosome and each "block" of data. 
The filenames should then contain the corresponding chromosome and block number. The pattern of the file names, as well as file structure may be specified in the corresponding [config file](https://github.com/PMBio/deeprvat/blob/main/example/config/deeprvat_annotation_config.yaml). The pipeline also requires input data and repositories descried in [requirements](#requirements).

(requirements)=
## Requirements

BCFtools as well as HTSlib should be installed on the machine, [VEP](http://www.ensembl.org/info/docs/tools/vep/script/vep_download.html) should be installed for running the pipeline. The [faatpipe](https://github.com/HealthML/faatpipe) repo, [kipoi-veff2](https://github.com/kipoi/kipoi-veff2) repo and  [vep-plugins repository](https://github.com/Ensembl/VEP_plugins/) should be cloned. Annotation data for CADD, spliceAI and primateAI should be downloaded. The path to the data may be specified in the corresponding [config file](https://github.com/PMBio/deeprvat/blob/main/example/config/deeprvat_annotation_config.yaml). 
Download paths:
- [CADD](https://cadd.gs.washington.edu/download): "All possible SNVs of GRCh38/hg38" and "gnomad.genomes.r3.0.indel.tsv.gz" incl. their Tabix Indices
- [SpliceAI](https://basespace.illumina.com/s/otSPW8hnhaZR): "genome_scores_v1.3"/"spliceai_scores.raw.snv.hg38.vcf.gz" and "spliceai_scores.raw.indel.hg38.vcf.gz" 
- [PrimateAI](https://basespace.illumina.com/s/yYGFdGih1rXL) PrimateAI supplementary data/"PrimateAI_scores_v0.2_GRCh38_sorted.tsv.bgz"
- [AlphaMissense](https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz) 

Also a reference GTF file containing transcript annotations is required, this can be downloaded from [here](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz)


## Configure the annotation pipeline
The snakemake annotation pipeline is configured using a yaml file with the format akin to the [example config file](https://github.com/PMBio/deeprvat/blob/main/example/config/deeprvat_annotation_config.yaml).

The config above would use the following directory structure:
```shell
|--reference
|   |-- fasta file
|   |-- GTF file 

|-- preprocessing_workdir
|   |-- norm
|   |   |-- bcf
|   |   |   |-- bcf_input_files
|   |   |   |-- ...
|   |   |-- variants
|   |   |   |-- variants.tsv.gz
|   |-- preprocessed
|   |   |-- genotypes.h5


|-- output_dir
|   |-- annotations
|   |   |-- tmp
|   |   |   |-- deepSEA_PCA
|   |   |   |-- absplice

|-- repo_dir
|   |-- ensembl-vep
|   |   |-- cache
|   |   |-- plugins
|   |-- faatpipe
|   |-- kipoi-veff2

|-- annotation_data
|   |-- cadd
|   |-- spliceAI
|   |-- primateAI




```


Bcf files created by the [preprocessing pipeline](https://deeprvat.readthedocs.io/en/latest/preprocessing.html) are used as input data. The input data directory should only contain the files needed. 
The pipeline also uses the variant.tsv file, the reference file and the genotypes file from the preprocessing pipeline. 
A GTF file as described in [requirements](#requirements) and the FASTA file used for preprocessing is also necessary.
The output is stored in the `output_dir/annotations` folder and any temporary files in the `tmp` subfolder. All repositories used including VEP with its corresponding cache as well as plugins are stored in `repo_dir`.
Data for VEP plugins and the CADD cache are stored in `annotation_data`. 

## Running the annotation pipeline on example data


1. If you haven't done so already, run the [preprocessing pipeline](https://deeprvat.readthedocs.io/en/latest/preprocessing.html)
1. change into the `example/annotations` directory

1. Install the annotation environment
    ```shell
    mamba env create -f path/to/deeprvat/deeprvat_annotations.yml
    mamba activate deeprvat_annotations
    pip install -e path/to/deeprvat
    ```
1. Clone/Link the repositories mentioned in [requirements](#requirements) into `repo_dir` and install the required kipoi-veff2 conda environment with  
    ```shell
    mamba env create -f repo_dir/kipoi-veff2/environment.minimal.linux.yml
    ```
    If you already have some of the needed repositories on your machine you can edit the paths in the [config](https://github.com/PMBio/deeprvat/blob/main/example/config/deeprvat_annotation_config.yaml).
  

1. Inside the annotation directory create a directory `annotation_data` and download/link the prescored files for CADD, SpliceAI, and PrimateAI (see [requirements](#requirements))

1.  Run the annotations pipeline inside the deeprvat_annotations environment:
    ```shell
      mamba activate deeprvat_annotations
      snakemake -j <nr_cores> -s ../../pipelines/annotations.snakefile --configfile ../config/deeprvat_annotation_config.yaml --use-conda
    ```
    

## Running the pipeline on your own data 
Modify the path in the [config file](https://github.com/PMBio/deeprvat/blob/main/example/config/deeprvat_annotation_config.yaml) s.t. they point to the output directory of the preprocessing pipeline run on your data. 
You can add/remove VEP plugins in the `additional_vep_plugin_cmds` part of the config by adding /removing plugin commands to be added to the vep run command. You can omit absplice/deepSea by setting `include_absplice`/ `include_deepSEA` to `False`in the config. When you add/remove annotations you have to alter the values in `example/config/annotation_colnames_filling_values.yaml`. This file consist of  the names of the columns of the tool used, the name to be used in the output data frame, the default value replacing all `NA` values as well as the data type, for example:
```shell
  'CADD_RAW' : 
    - 'CADD_raw'
    - 0
    - float
```
Here `CADD_RAW` is the name of the column of the VEP output when the plugin is used, it is then renamed in the final annotation dataframe to `CADD_raw`, all `NA` values are set to `0` and the values are of type `float`. 

You can also change the way the allele frequencies are calculated by adding `af_mode` key to the [config file](https://github.com/PMBio/deeprvat/blob/main/example/config/deeprvat_annotation_config.yaml). By default, the allele frequencies are calculated from the data the annotation pipeline is run with. To use gnomade or gnomadg allele frequncies (from VEP ) instead, add 
```shell
af_mode : 'af_gnomade'
```
or 
```shell
af_mode : 'af_gnomadg'
```
to the config file.

## References

(reference-1-target)=
<a id="1">[1]</a> Monti, R., Rautenstrauch, P., Ghanbari, M. et al. Identifying interpretable gene-biomarker associations with functionally informed kernel-based tests in 190,000 exomes. Nat Commun 13, 5332 (2022). https://doi.org/10.1038/s41467-022-32864-2

(reference-2-target)=
<a id="2">[2]</a> Žiga Avsec et al., “Kipoi: accelerating the community exchange and reuse of predictive models for genomics,” bioRxiv, p. 375345, Jan. 2018, doi: 10.1101/375345.

(reference-3-target)=
<a id="3">[3]</a>N. Wagner et al., “Aberrant splicing prediction across human tissues,” Nature Genetics, vol. 55, no. 5, pp. 861–870, May 2023, doi: 10.1038/s41588-023-01373-3.



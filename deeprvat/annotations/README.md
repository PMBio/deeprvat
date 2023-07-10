# DeepRVAT Annotation pipeline

This pipeline is based on [snakemake](https://snakemake.readthedocs.io/en/stable/). It uses [bcftools+samstools](https://www.htslib.org/), as well as [perl](https://www.perl.org/), [CADD](https://cadd.gs.washington.edu/) and [VEP](http://www.ensembl.org/info/docs/tools/vep/index.html), including plugins for [primateAI](https://github.com/Illumina/PrimateAI) and [spliceAI](https://github.com/Illumina/SpliceAI),  Future releases will include further annotation tools like [abSplice](https://github.com/gagneurlab/absplice), [deepSEA](http://deepsea.princeton.edu/job/analysis/create/) and [deepRiPe](https://ohlerlab.mdc-berlin.de/software/DeepRiPe_140/).

## Input

The pipeline uses compressed vcf files containing variant information, a reference fasta file as well as a text file that maps data blocks to chromosomes as input. It is expected that the vcf files contain the columns "CHROM" "POS" "ID" "REF" and "ALT". Any other columns, including genotype information are stripped from the data before annotation tools are used on the data. The variants may be split into several vcf files for each chromosome and each "block" of data. The filenames should then contain the corresponding chromosome and block number. The pattern of the file names, as well as file structure may be specified in the corresponding [config file](config/deeprvat_annotation_config.yaml).

## Requirements 

[CADD](https://github.com/kircherlab/CADD-scripts/tree/master/src/scripts) as well as [VEP](http://www.ensembl.org/info/docs/tools/vep/script/vep_download.html#docker) should be installed together with the [plugins](https://www.ensembl.org/info/docs/tools/vep/script/vep_plugins.html) for primateAI and spliceAI. Annotation data for VEP, CADD, spliceAI and primateAI should be downloaded. The path to the data may be specified in the corresponding [config file](config/deeprvat_annotation_config.yaml). 

## Output

The pipeline outputs one annotation file for VEP and one annotation file for CADD for each input vcf-file. Further releases will concatenate and merge the output data into one file. 

## Configure the annotation pipeline
The snakemake annotation pipeline is configured using a yaml file with the format akin to the [example file](config/deeprvat_annotation_config.yaml).

The config above would use the following directory structure:
```shell
parent_directory

|-- reference
|   |-- GRCh38.primary_assembly.genome.fa.gz
|-- vcf
|   |-- metadata
|   |   |-- pvcf_blocks.txt
|   |-- raw
|-- annotations
|   |-- tmp
|-- annotation_data
|   |-- cadd
|   |-- spliceAI
|   |-- primateAI
|-- software
|   |-- ensembl-vep
|   |   |-- cache
|   |   |-- plugins
|   |-- CADD-scripts
```
The variant input files are then stored in the `vcf/raw` directory, the reference fasta file is stored in the `reference` folder. The text file mapping blocks to chromosomes is stored in `vcf/metadata` folder. The output is stored in the `annotations` folder and any temporary files in the `tmp` subfolder. VEPwith its corresponding cache as well as scripts for CADD are stored in `software`.
Data for VEP plugins and the CADD cache are stored in `annotation data`. 

## Running the annotation pipeline

After configuration and activating the environment run the pipeline using snakemake:

```shell
  snakemake -j <nr_cores> -s annotations.snakemake --configfile config/deeprvat_annotation.config 
```

## Next Releases
Further releases will include further annotation tools like [abSplice](https://github.com/gagneurlab/absplice), [deepSEA](http://deepsea.princeton.edu/job/analysis/create/) and [deepRiPe](https://ohlerlab.mdc-berlin.de/software/DeepRiPe_140/). 
Furthermore, annotations will be concatenated and merged into a single file containing annotations from every tool used on every input variant. Support for gene specificity of variants will be also be included in coming releases (e.g. some variants may have several annotations for each gene they are mapped to).

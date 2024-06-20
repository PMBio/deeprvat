# DeepRVAT Nextflow pipeline

## Build docker image

if on m* macs:

`docker build --platform linux/amd64 --tag deeprvat_preprocessing .`

else

`docker build --tag deeprvat_preprocessing .`

## Get fasta file

```shell
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh38.primary_assembly.genome.fa.gz -P example/workdir/reference
gzip -d example/workdir/reference/GRCh38.primary_assembly.genome.fa.gz
```

## Run
`nextflow run preprocessing.nf`


## Create dag preview
`nextflow run preprocessing.nf  -preview -with-dag`


rule all:
    input:
        "workdir/filtered_output.vcf.gz",


rule fiter_gtf:
    input:
        "gtf/gencode.v44.annotation.gtf",
    output:
        "workdir/filtered_genes.gtf",
    shell:
        'get_features.pl --in "{input}" --out "{output}" --include "gene_biotype=protein_coding" --feature "gene"'
#        'zgrep -w "gene" "{input}" | grep "protein_coding" > "{output}"'

# https://github.com/tjparnell/biotoolbox
# https://metacpan.org/dist/Bio-ToolBox/view/scripts/get_features.pl
# https://www.biostars.org/p/56280/
# https://sciberg.com/resources/bioinformatics-scripts/converting-gtf-files-into-bed-files
# https://gffutils.readthedocs.io/en/latest/gtf2bed.html
# https://github.com/fls-bioinformatics-core/GFFUtils
# mamba install -c bioconda bedops
# pip install git+https://github.com/fls-bioinformatics-core/genomics@1.13.0


rule create_bed:
    input:
        "workdir/filtered_genes.gtf",
    output:
        "workdir/filtered_genes.bed",
    shell:
        'convert2bed --input=gtf --output=bed  < "{input}" > "{output}"'
#        'awk -F"\t" \'{{print $1"\t"$4"\t"$5"\t"$9}}\' "{input}" > "{output}"'


rule index_fasta:
    input:
        "gtf/GRCh38_full_analysis_set_plus_decoy_hla.fa",
    output:
        "gtf/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai",
    shell:
        "samtools faidx {input}"


# https://www.biostars.org/p/70795/
# https://www.biostars.org/p/206140/


# 1.fai file can also be directly used with -g. As per bedtools: ".fai files may be used as genome (-g) files."
rule expand_regions:
    input:
        bed="workdir/filtered_genes.bed",
        faidx="gtf/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai",
    params:
        region_expand=3000,
    output:
        "workdir/expanded_regions.bed",
    shell:
        'bedtools slop -i "{input.bed}" -g "{input.faidx}" -b {params.region_expand}  > "{output}"'


rule filter_vcf_file:
    input:
        bed="workdir/expanded_regions.bed",
        vcf="gtf/ALL.chr1.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz",
    output:
        "workdir/filtered_output.vcf.gz",
    shell:
        'bcftools view -R "{input.bed}" "{input.vcf}" | pv | gzip > "{output}"'

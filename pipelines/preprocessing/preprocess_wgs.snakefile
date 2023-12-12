
rule all:
    input:
        "workdir/filtered_output.vcf.gz",


rule fiter_gtf:
    input:
        "gtf/gencode.v44.annotation.gtf.gz",
    output:
        "workdir/filtered_genes.gtf",
    shell:
        'zgrep -w "gene" "{input}" | grep "protein_coding" > "{output}"'


rule create_bed:
    input:
        "workdir/filtered_genes.gtf",
    output:
        "workdir/filtered_genes.bed",
    shell:
        'awk -F"\t" \'{{print $1"\t"$4"\t"$5"\t"$9}}\' "{input}" > "{output}"'


rule index_fasta:
    input:
        "gtf/GRCh38_full_analysis_set_plus_decoy_hla.fa",
    output:
        "gtf/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai",
    shell:
        "samtools faidx {input}"


rule create_genome_file:
    input:
        "gtf/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai",
    output:
        "workdir/genome_file.genome",
    shell:
        'cut -f 1,2 "{input}" > "{output}"'


rule expand_regions:
    input:
        bed="workdir/filtered_genes.bed",
        genome="workdir/genome_file.genome",
    params:
        region_expand=3000,
    output:
        "workdir/expanded_regions.bed",
    shell:
        'bedtools slop -i "{input.bed}" -g "{input.genome}" -b {params.region_expand}  > "{output}"'


rule filter_vcf_file:
    input:
        bed="workdir/expanded_regions.bed",
        vcf="gtf/ALL.chr1.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz",
    output:
        "workdir/filtered_output.vcf.gz",
    shell:
        'bcftools view -R "{input.bed}" "{input.vcf}" | pv | gzip > "{output}"'

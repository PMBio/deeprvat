from pathlib import Path


configfile: "config/deeprvat_preprocess_config.yaml"


load_samtools = config.get("samtools_load_cmd") or ""
load_bcftools = config.get("bcftools_load_cmd") or ""
zcat_cmd = config.get("zcat_cmd") or "zcat"

preprocessing_cmd = "deeprvat_preprocess"

working_dir = Path(config["working_dir"])
preprocessed_dir = working_dir / config["preprocessed_dir_name"]
reference_dir = working_dir / config["reference_dir_name"]

preprocess_threads = config["preprocess_threads"]

fasta_file = reference_dir / config["reference_fasta_file"]
fasta_index_file = reference_dir / f"{config['reference_fasta_file']}.fai"

norm_dir = working_dir / config["norm_dir_name"]
sparse_dir = norm_dir / config["sparse_dir_name"]
bcf_dir = norm_dir / "bcf"
norm_variants_dir = norm_dir / "variants"

qc_dir = working_dir / "qc"
qc_indmiss_stats_dir = qc_dir / "indmiss/stats"
qc_indmiss_samples_dir = qc_dir / "indmiss/samples"
qc_indmiss_sites_dir = qc_dir / "indmiss/sites"
qc_varmiss_dir = qc_dir / "varmiss"
qc_hwe_dir = qc_dir / "hwe"
qc_read_depth_dir = qc_dir / "read_depth"
qc_allelic_imbalance_dir = qc_dir / "allelic_imbalance"
qc_duplicate_vars_dir = qc_dir / "duplicate_vars"
qc_filtered_samples_dir = qc_dir / "filtered_samples"

gtf_workdir = working_dir / "gtf"

gtf_file = reference_dir / config["gtf_file"]
gtf_filtered_file = gtf_workdir / f"{gtf_file.stem}_filtered_genes.gtf"
bed_file = gtf_workdir / f"{gtf_file.stem}_filtered_genes.bed"
expanded_bed = gtf_workdir / f"{gtf_file.stem}_filtered_expanded_regions.bed"


with open(config["vcf_files_list"]) as file:
    vcf_files = [Path(line.rstrip()) for line in file]
    vcf_stems = [vf.stem.replace(".vcf", "") for vf in vcf_files]

    assert len(vcf_stems) == len(vcf_files)

    vcf_look_up = {stem: file for stem, file in zip(vcf_stems, vcf_files)}

chromosomes = config["included_chromosomes"]


rule combine_genotypes:
    input:
        expand(
            preprocessed_dir / "genotypes_chr{chr}.h5",
            chr=chromosomes,
        ),
    output:
        preprocessed_dir / "genotypes.h5",
    shell:
        f"{preprocessing_cmd} combine-genotypes {{input}} {{output}}"


rule normalize:
    input:
        samplefile=norm_dir / "samples_chr.csv",
        fasta=fasta_file,
        fastaindex=fasta_index_file,
        expanded_bed=expanded_bed,
    params:
        vcf_file=lambda wildcards: vcf_look_up[wildcards.vcf_stem],
    output:
        bcf_file=bcf_dir / "{vcf_stem}.bcf",
    shell:
        f"""{load_bcftools} bcftools view -R "{{input.expanded_bed}}" "{{params.vcf_file}}" --output-type u \
        | bcftools view --samples-file {{input.samplefile}} --output-type u  \
        | bcftools view --include 'COUNT(GT="alt") > 0' --output-type u \
        | bcftools norm -m-both -f {{input.fasta}} --output-type b --output {{output.bcf_file}}"""


rule fiter_gtf:
    input:
        gtf_file,
    output:
        gtf_filtered_file,
    shell:
        'get_features.pl --in "{input}" --out "{output}" --include "gene_type=protein_coding" --feature "gene" --gtf'


rule create_bed:
    input:
        gtf_filtered_file,
    output:
        bed_file
    params:
        maxmem=config["convert2bed_max_mem"]
    shell:
        'convert2bed --max-mem={params.maxmem} --input=gtf --output=bed  < "{input}" > "{output}"'


rule expand_regions:
    input:
        bed=bed_file,
        faidx=fasta_index_file,
    params:
        region_expand=config["region_expand"],
    output:
        expanded_bed
    shell:
        'bedtools slop -i "{input.bed}" -g "{input.faidx}" -b {params.region_expand}  > "{output}"'


rule index_fasta:
    input:
        fasta=fasta_file,
    output:
        fasta_index_file,
    shell:
        f"{load_samtools} samtools faidx {{input.fasta}}"


rule sparsify:
    input:
        bcf=bcf_dir / "{vcf_stem}.bcf",
    output:
        tsv=sparse_dir / "{vcf_stem}.tsv.gz",
    shell:
        f"""{load_bcftools} bcftools query --format '[%CHROM\t%POS\t%REF\t%ALT\t%SAMPLE\t%GT\n]' --include 'GT!="RR" & GT!="mis"' {{input.bcf}} \
        | sed 's/0[/,|]1/1/; s/1[/,|]0/1/; s/1[/,|]1/2/; s/0[/,|]0/0/' | gzip > {{output.tsv}}"""


rule variants:
    input:
        bcf=bcf_dir / "{vcf_stem}.bcf",
    output:
        norm_variants_dir / "{vcf_stem}.tsv.gz",
    shell:
        f"{load_bcftools} bcftools query --format '%CHROM\t%POS\t%REF\t%ALT\n' {{input}} | gzip > {{output}}"


rule concatenate_variants:
    input:
        expand(norm_variants_dir / "{vcf_stem}.tsv.gz", vcf_stem=vcf_stems),
    output:
        norm_variants_dir / "variants_no_id.tsv.gz",
    shell:
        "{zcat_cmd} {input} | gzip > {output}"


rule add_variant_ids:
    input:
        norm_variants_dir / "variants_no_id.tsv.gz",
    output:
        variants=norm_variants_dir / "variants.tsv.gz",
        duplicates=qc_duplicate_vars_dir / "duplicates.tsv",
    shell:
        f"{preprocessing_cmd} add-variant-ids {{input}} {{output.variants}} {{output.duplicates}}"


rule create_parquet_variant_ids:
    input:
        norm_variants_dir / "variants_no_id.tsv.gz",
    output:
        variants=norm_variants_dir / "variants.parquet",
        duplicates=qc_duplicate_vars_dir / "duplicates.parquet",
    shell:
        f"{preprocessing_cmd} add-variant-ids {{input}} {{output.variants}} {{output.duplicates}}"


rule extract_samples:
    input:
        vcf_files,
    output:
        norm_dir / "samples_chr.csv",
    shell:
        f"{load_bcftools} bcftools query --list-samples {{input}} > {{output}}"

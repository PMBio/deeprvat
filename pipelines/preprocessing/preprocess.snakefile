from pathlib import Path

import deeprvat.preprocessing.preprocess as deeprvat_preprocess


configfile: "config/deeprvat_preprocess_config.yaml"

load_samtools = config.get("samtools_load_cmd") or ""
load_bcftools = config.get("bcftools_load_cmd") or ""
zcat_cmd = config.get("zcat_cmd") or "zcat"

preprocessing_cmd = "deeprvat_preprocess"

working_dir = Path(config["working_dir"])
preprocessed_dir = working_dir / config["preprocessed_dir_name"]
reference_dir = working_dir / config["reference_dir_name"]

fasta_file = reference_dir / config["reference_fasta_file"]
fasta_index_file = reference_dir / f"{config['reference_fasta_file']}.fai"

norm_dir = working_dir / config["norm_dir_name"]
sparse_dir = norm_dir / config["sparse_dir_name"]
bcf_dir = norm_dir / "bcf"
norm_variants_dir = norm_dir / "variants"

qc_dir = working_dir / "qc"
qc_indmiss_dir = qc_dir / "indmiss"
qc_indmiss_stats_dir = qc_indmiss_dir / "stats"
qc_indmiss_samples_dir = qc_indmiss_dir / "samples"
qc_indmiss_sites_dir = qc_indmiss_dir / "sites"
qc_varmiss_dir = qc_dir / "varmiss"
qc_hwe_dir = qc_dir / "hwe"
qc_read_depth_dir = qc_dir / "read_depth"
qc_allelic_imbalance_dir = qc_dir / "allelic_imbalance"
qc_duplicate_vars_dir = qc_dir / "duplicate_vars"
qc_filtered_samples_dir = qc_dir / "filtered_samples"

vcf_stems, vcf_files, vcf_look_up = deeprvat_preprocess.parse_file_path_list(config["vcf_files_list"])
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

rule extract_samples:
    input:
        vcf_files,
    output:
        norm_dir / "samples_chr.csv",
    shell:
        f"{load_bcftools} bcftools query --list-samples {{input}} > {{output}}"

rule normalize:
    input:
        samplefile=rules.extract_samples.output,
        fasta=fasta_file,
        fastaindex=fasta_index_file,
    params:
        vcf_file=lambda wildcards: vcf_look_up[wildcards.vcf_stem],
    output:
        bcf_file=bcf_dir / "{vcf_stem}.bcf",
    resources:
        mem_mb=lambda wildcards, attempt: 16384 * (attempt + 1),
    shell:
        f"""{load_bcftools} bcftools view --samples-file {{input.samplefile}} --output-type u {{params.vcf_file}} | bcftools view --include 'COUNT(GT="alt") > 0' --output-type u | bcftools norm -m-both -f {{input.fasta}} --output-type b --output {{output.bcf_file}}"""


rule index_fasta:
    input:
        fasta=fasta_file,
    output:
        fasta_index_file,
    shell:
        f"{load_samtools} samtools faidx {{input.fasta}}"


rule sparsify:
    input:
        bcf=rules.normalize.output.bcf_file
    output:
        tsv=sparse_dir / "{vcf_stem}.tsv.gz",
    resources:
        mem_mb=512,
    shell:
        f"""{load_bcftools} bcftools query --format '[%CHROM\t%POS\t%REF\t%ALT\t%SAMPLE\t%GT\n]' --include 'GT!="RR" & GT!="mis"' {{input.bcf}} \
        | sed 's/0[/,|]1/1/; s/1[/,|]0/1/; s/1[/,|]1/2/; s/0[/,|]0/0/' | gzip > {{output.tsv}}"""


rule variants:
    input:
        bcf=rules.normalize.output.bcf_file,
    output:
        norm_variants_dir / "{vcf_stem}.tsv.gz",
    resources:
        mem_mb=512,
    shell:
        f"{load_bcftools} bcftools query --format '%CHROM\t%POS\t%REF\t%ALT\n' {{input}} | gzip > {{output}}"


rule concatenate_variants:
    input:
        expand(rules.variants.output,vcf_stem=vcf_stems),
    output:
        norm_variants_dir / "variants_no_id.tsv.gz",
    resources:
        mem_mb=256,
    shell:
        "{zcat_cmd} {input} | gzip > {output}"


rule add_variant_ids:
    input:
        rules.concatenate_variants.output
    output:
        variants=norm_variants_dir / "variants.tsv.gz",
        duplicates=qc_duplicate_vars_dir / "duplicates.tsv",
    resources:
        mem_mb=2048,
    shell:
        f"{preprocessing_cmd} add-variant-ids {{input}} {{output.variants}} {{output.duplicates}}"


rule create_parquet_variant_ids:
    input:
        rules.concatenate_variants.output
    output:
        variants=norm_variants_dir / "variants.parquet",
        duplicates=qc_duplicate_vars_dir / "duplicates.parquet",
    resources:
        mem_mb=2048,
    shell:
        f"{preprocessing_cmd} add-variant-ids {{input}} {{output.variants}} {{output.duplicates}}"

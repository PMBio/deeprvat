from pathlib import Path


configfile: "config/deeprvat_preprocess_config.yaml"

load_samtools = config.get("samtools_load_cmd") or ""
load_bcftools = config.get("bcftools_load_cmd") or ""
zcat_cmd = config.get("zcat_cmd") or "zcat"

preprocessing_cmd = "deeprvat_preprocess"

working_dir = Path(config["working_dir"])
data_dir = Path(config["data_dir"])
preprocessed_dir = working_dir / config["preprocessed_dir_name"]
vcf_dir = data_dir / config["input_vcf_dir_name"]
metadata_dir = data_dir / config["metadata_dir_name"]
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


with open(config["vcf_files_list"]) as file:

    vcf_files = [Path(line.rstrip()) for line in file]
    vcf_stems = [vf.stem.split('.')[0] for vf in vcf_files]

    vcf_look_up = {stem: file for stem, file in zip(vcf_stems, vcf_files)}


chromosomes = config["included_chromosomes"]



rule all:
    input:
        expand(bcf_dir / "{vcf_stem}.bcf",vcf_stem=vcf_stems),


rule normalize:
    input:
        samplefile=norm_dir / "samples_chr.csv",
        fasta=fasta_file,
        fastaindex=fasta_index_file,
    params:
        vcf_file= lambda wildcards: vcf_look_up[wildcards.vcf_stem],
    output:
        bcf_file=bcf_dir / "{vcf_stem}.bcf",
    shell:
        f"""{load_bcftools} bcftools view --samples-file {{input.samplefile}} --output-type u {{params.vcf_file}} | bcftools view --include 'COUNT(GT="alt") > 0' --output-type u | bcftools norm -m-both -f {{input.fasta}} --output-type b --output {{output.bcf_file}}"""

rule extract_samples:
    input:
        vcf_files,
    output:
        norm_dir / "samples_chr.csv",
    shell:
        f"{load_bcftools} bcftools query --list-samples {{input}} > {{output}}"


rule index_fasta:
    input:
        fasta=fasta_file,
    output:
        fasta_index_file,
    shell:
        f"{load_samtools} samtools faidx {{input.fasta}}"


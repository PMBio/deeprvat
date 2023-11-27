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
        expand(sparse_dir / "{vcf_stem}.tsv.gz",vcf_stem=vcf_stems),
        expand(norm_variants_dir / "{vcf_stem}.tsv.gz",vcf_stem=vcf_stems),
        norm_variants_dir / "variants_no_id.tsv.gz",
        norm_variants_dir / "variants.tsv.gz",
        qc_duplicate_vars_dir / "duplicates.tsv",
        norm_variants_dir / "variants.parquet",
        qc_duplicate_vars_dir / "duplicates.parquet",

        expand(qc_varmiss_dir / "{vcf_stem}.tsv.gz",vcf_stem=vcf_stems),
        expand(qc_hwe_dir / "{vcf_stem}.tsv.gz",vcf_stem=vcf_stems),
        expand(qc_read_depth_dir / "{vcf_stem}.tsv.gz",vcf_stem=vcf_stems),
        expand(qc_allelic_imbalance_dir / "{vcf_stem}.tsv.gz",vcf_stem=vcf_stems),

        expand(preprocessed_dir / "genotypes_chr{chr}.h5",chr=chromosomes)



rule preprocess:
    input:
        variants=norm_variants_dir / "variants.tsv.gz",
        variants_parquet=norm_variants_dir / "variants.parquet",
        samples=norm_dir / "samples_chr.csv",
        sparse_tg=expand(sparse_dir / "{vcf_stem}.tsv.gz",vcf_stem=vcf_stems),
        qc_varmiss=expand(qc_varmiss_dir / "{vcf_stem}.tsv.gz",vcf_stem=vcf_stems),
        qc_hwe=expand(qc_hwe_dir / "{vcf_stem}.tsv.gz", vcf_stem=vcf_stems),
        qc_read_depth=expand(qc_read_depth_dir / "{vcf_stem}.tsv.gz",vcf_stem=vcf_stems),
        qc_allelic_imbalance=expand(qc_allelic_imbalance_dir / "{vcf_stem}.tsv.gz",
        vcf_stem=vcf_stems),
        qc_filtered_samples=qc_filtered_samples_dir,
    output:
        expand(preprocessed_dir / "genotypes_chr{chr}.h5",chr=set(chromosomes)),
    shell:
        " ".join(
            [
                f"{preprocessing_cmd}",
                "process-sparse-gt",
                f"--exclude-variants {qc_allelic_imbalance_dir}",
                f"--exclude-variants {qc_hwe_dir}",
                f"--exclude-variants {qc_varmiss_dir}",
                f"--exclude-variants {qc_duplicate_vars_dir}",
                f"--exclude-calls {qc_read_depth_dir}",
                f"--exclude-samples {qc_filtered_samples_dir}",
                "--chromosomes ",
                ",".join(str(chr) for chr in set(chromosomes)),
                f"--threads {preprocess_threads}",
                "{input.variants}",
                "{input.samples}",
                f"{sparse_dir}",
                f"{preprocessed_dir / 'genotypes'}",
            ]
        )



rule create_excluded_samples_dir:
    output:
        directory(qc_filtered_samples_dir),
    shell:
        "mkdir -p {output}"



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
        expand( norm_variants_dir / "{vcf_stem}.tsv.gz",vcf_stem=vcf_stems),
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





rule qc_allelic_imbalance:
    input:
        bcf_dir / "{vcf_stem}.bcf",
    output:
        qc_allelic_imbalance_dir / "{vcf_stem}.tsv.gz",
    shell:
        f"""{load_bcftools} bcftools query --format '%CHROM\t%POS\t%REF\t%ALT\n' --exclude 'COUNT(GT="het")=0 || (GT="het" & ((TYPE="snp" & (FORMAT/AD[*:1] / FORMAT/AD[*:0]) > 0.15) | (TYPE="indel" & (FORMAT/AD[*:1] / FORMAT/AD[*:0]) > 0.20)))' {{input}} | gzip > {{output}}"""




rule qc_varmiss:
    input:
        bcf_dir / "{vcf_stem}.bcf",
    output:
        qc_varmiss_dir / "{vcf_stem}.tsv.gz",
    shell:
        f'{load_bcftools} bcftools query --format "%CHROM\t%POS\t%REF\t%ALT\n" --include "F_MISSING >= 0.1" {{input}} | gzip > {{output}}'


rule qc_hwe:
    input:
        bcf_dir / "{vcf_stem}.bcf",
    output:
        qc_hwe_dir / "{vcf_stem}.tsv.gz",
    shell:
        f'{load_bcftools} bcftools +fill-tags --output-type u {{input}} -- --tags HWE | bcftools query --format "%CHROM\t%POS\t%REF\t%ALT\n" --include "INFO/HWE <= 1e-15" | gzip > {{output}}'


rule qc_read_depth:
    input:
        bcf_dir / "{vcf_stem}.bcf",
    output:
        qc_read_depth_dir / "{vcf_stem}.tsv.gz",
    shell:
        f"""{load_bcftools} bcftools query --format '[%CHROM\\t%POS\\t%REF\\t%ALT\\t%SAMPLE\\n]' --include '(GT!="RR" & GT!="mis" & TYPE="snp" & FORMAT/DP < 7) | (GT!="RR" & GT!="mis" & TYPE="indel" & FORMAT/DP < 10)' {{input}} | gzip > {{output}}"""




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


rule create_excluded_samples_dir:
    output:
        directory(qc_filtered_samples_dir),
    shell:
        "mkdir -p {output}"

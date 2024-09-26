rule deepRiPe_parclip:
    input:
        variants=rules.extract_variants.output,
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / ("{file_stem}_variants.parclip_deepripe.csv.gz"),
    threads: n_jobs_deepripe
    resources:
        mem_mb=lambda wildcards, attempt: 5_000 * (attempt + 1),
    shell:
        f"mkdir -p {pybedtools_tmp_path/ 'parclip'} && deeprvat_annotations scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path/ 'parclip'} {saved_deepripe_models_path} {{threads}} 'parclip'"


rule deepRiPe_eclip_hg2:
    input:
        variants=rules.extract_variants.output,
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / ("{file_stem}_variants.eclip_hg2_deepripe.csv.gz"),
    threads: lambda wildcards, attempt: n_jobs_deepripe * attempt
    resources:
        mem_mb=lambda wildcards, attempt: 5_000 * (attempt + 1),
    shell:
        f"mkdir -p {pybedtools_tmp_path/ 'hg2'} && deeprvat_annotations scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path/ 'hg2'} {saved_deepripe_models_path} {{threads}} 'eclip_hg2'"


rule deepRiPe_eclip_k5:
    input:
        variants=rules.extract_variants.output,
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / ("{file_stem}_variants.eclip_k5_deepripe.csv.gz"),
    threads: lambda wildcards, attempt: n_jobs_deepripe * attempt
    resources:
        mem_mb=lambda wildcards, attempt: 5_000 * (attempt + 1),
    shell:
        f"mkdir -p {pybedtools_tmp_path/ 'k5'} && deeprvat_annotations scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path/ 'k5'} {saved_deepripe_models_path} {{threads}} 'eclip_k5'"


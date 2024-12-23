
rule encode_150:
    input:
        variants=rules.extract_with_header.output,
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_tmp_dir / ("{file_stem}_encoded_150.pkl"),
    threads: 1
    resources:
        mem_mb=lambda wildcards, attempt: 5_000 + 500 * (attempt),
    shell:
        f"cp {{input.fasta}} $TMPDIR && cp {{input.fasta}}.fai $TMPDIR && python /home/m991k/deeprvat_vep_plugin/deeprvat/deeprvat/annotations/annotations.py flank-encode-vcf {{input.variants}}  $TMPDIR/{fasta_file_name} {{output}} 150"


rule encode_200:
    input:
        variants=rules.extract_with_header.output,
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_tmp_dir / ("{file_stem}_encoded_200.pkl"),
    threads: 1
    resources:
        mem_mb=lambda wildcards, attempt: 5_000 + 500 * (attempt),
    shell:
        f"cp {{input.fasta}} $TMPDIR && cp {{input.fasta}}.fai $TMPDIR && python /home/m991k/deeprvat_vep_plugin/deeprvat/deeprvat/annotations/annotations.py flank-encode-vcf {{input.variants}} $TMPDIR/{fasta_file_name} {{output}} 200"


rule deepRiPe_parclip:
    input:
        variants=rules.extract_variants.output,
        encoded_seqs= rules.encode_150.output,
    output:
        anno_dir / ("{file_stem}_variants.parclip_deepripe.csv.gz"),
    threads: 8
    resources:
        mem_mb=lambda wildcards, attempt: 100_000 + 25_000 * (attempt),
    shell:
        f"cp -R {saved_deepripe_models_path} $TMPDIR && python /home/m991k/deeprvat_vep_plugin/deeprvat/deeprvat/annotations/annotations.py scorevariants-deepripe {{input.variants}} {anno_dir} $TMPDIR/deepripe_models {{input.encoded_seqs}} 512 'parclip'"


rule deepRiPe_eclip_hg2:
    input:
        variants=rules.extract_variants.output,
        fasta=fasta_dir / fasta_file_name,
        encoded_seqs= rules.encode_200.output,
    output:
        anno_dir / ("{file_stem}_variants.eclip_hg2_deepripe.csv.gz"),
    threads: 8
    resources:
        mem_mb=lambda wildcards, attempt: 100_000 + 25_000 * (attempt),
    shell:
        f"cp -R {saved_deepripe_models_path} $TMPDIR && python /home/m991k/deeprvat_vep_plugin/deeprvat/deeprvat/annotations/annotations.py scorevariants-deepripe {{input.variants}} {anno_dir} $TMPDIR/deepripe_models {{input.encoded_seqs}} 512  'eclip_hg2'"


rule deepRiPe_eclip_k5:
    input:
        variants=rules.extract_variants.output,
        fasta=fasta_dir / fasta_file_name,
        encoded_seqs= rules.encode_200.output,
    output:
        anno_dir / ("{file_stem}_variants.eclip_k5_deepripe.csv.gz"),
    threads: 8
    resources:
        mem_mb=lambda wildcards, attempt: 100_000 + 25_000 * (attempt),
    shell:
        f"cp -R {saved_deepripe_models_path} $TMPDIR && python /home/m991k/deeprvat_vep_plugin/deeprvat/deeprvat/annotations/annotations.py scorevariants-deepripe {{input.variants}} {anno_dir} $TMPDIR/deepripe_models {{input.encoded_seqs}} 512  'eclip_k5'"

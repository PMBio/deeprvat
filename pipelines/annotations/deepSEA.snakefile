
rule deepSea:
    input:
        variants=rules.extract_with_header.output,
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / "{file_stem}.CLI.deepseapredict.diff.tsv",
    threads: n_jobs_deepripe
    resources:
        mem_mb=lambda wildcards, attempt: 5_000 * (attempt + 1),
    conda:
        "kipoi-veff2"
    shell:
        "kipoi_veff2_predict {input.variants} {input.fasta} {output} -l 1000 -m 'DeepSEA/predict' -s 'diff'"


rule concat_deepSea:
    input:
        deepSEAscoreFiles=expand(rules.deepSea.output, file_stem=file_stems),
    params:
        joined=lambda w, input: ",".join(input.deepSEAscoreFiles),
    threads: 8
    resources:
        mem_mb=lambda wildcards, attempt: 50_000 * (attempt + 1),
    output:
        anno_dir / "all_variants.deepSea.parquet",
    shell:
        " ".join(
            [
                "deeprvat_annotations",
                "concatenate-deepsea",
                "{params.joined}",
                "{output}",
                "{threads}",
            ]
        )


rule deepSea_PCA:
    input:
        deepsea_anno=str(rules.concat_deepSea.output),
    output:
        deepSEA_tmp_dir / "deepsea_pca.parquet",
    resources:
        mem_mb=lambda wildcards, attempt: 50_000 * (attempt + 1),
    shell:
        " ".join(
            [
                "mkdir -p",
                str(deepSEA_tmp_dir),
                "&&",
                "deeprvat_annotations",
                "deepsea-pca",
                "{input.deepsea_anno}",
                f"{str(deepSEA_pca_obj)}",
                f"{str(deepSEA_means_and_sds)}",
                f"{deepSEA_tmp_dir}",
                f"--n-components {n_pca_components}",
            ]
        )


rule add_ids_deepSea:
    input:
        variant_file=variant_pq,
        annotation_file=rules.deepSea_PCA.output,
    output:
        directory(anno_dir / "all_variants.wID.deepSea.parquet"),
    threads: ncores_addis
    resources:
        mem_mb=lambda wildcards, attempt: 50_000 * (attempt + 1),
    shell:
        " ".join(
            [
                "deeprvat_annotations",
                "add-ids-dask",
                "{input.annotation_file}",
                "{input.variant_file}",
                "{output}",
            ]
        )


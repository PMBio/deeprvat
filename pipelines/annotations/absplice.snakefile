
include: Path("../resources") / "absplice_download.snakefile"


include: Path("../resources") / "absplice_splicing_pred_DNA.snakefile"


rule aggregate_absplice_scores:
    input:
        abscore_files=expand(
            rules.absplice_dna.output.absplice_dna, genome=genome, file_stem=file_stems
        ),
        current_annotation_file=rules.merge_deepsea_pcas.output,
    output:
        score_file=anno_tmp_dir / "abSplice_score_file.parquet",
    threads: ncores_agg_absplice
    resources:
        mem_mb=lambda wildcards, attempt: 15_000 * (attempt + 1),
    shell:
        " ".join(
            [
                "deeprvat_annotations",
                "aggregate-abscores {input.current_annotation_file}",
                str(absplice_output_dir / absplice_main_conf["genome"] / "dna"),
                "{output.score_file} {threads}",
            ]
        )


rule merge_absplice_scores:
    input:
        absplice_scores=rules.aggregate_absplice_scores.output.score_file,
        current_annotation_file=rules.merge_deepsea_pcas.output,
    output:
        anno_dir / "vep_deepripe_deepsea_absplice.parquet",
    threads: ncores_merge_absplice
    resources:
        mem_mb=lambda wildcards, attempt: 19_000 * (attempt + 1),
    shell:
        " ".join(
            [
                "deeprvat_annotations",
                "merge-abscores",
                "{input.current_annotation_file}",
                "{input.absplice_scores}",
                "{output}",
            ]
        )

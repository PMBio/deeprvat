
include: Path("../resources") / "absplice_download.snakefile"


include: Path("../resources") / "absplice_splicing_pred_DNA.snakefile"


rule aggregate_absplice_scores:
    input:
        abscore_files=expand(
            rules.absplice_dna.output.absplice_dna, genome=genome, file_stem=file_stems
        ),
        current_annotation_file=anno_dir / "annotations.parquet",
    output:
        score_file=anno_tmp_dir / "abSplice_score_file.parquet",
    threads: ncores_agg_absplice
    resources:
        mem_mb=lambda wildcards, attempt: 15_000 * (attempt + 1),
    conda:
        "absplice"
    shell:
        " ".join(
            [
                "deeprvat_annotations",
                "aggregate-abscores {input.current_annotation_file}",
                str(absplice_output_dir / absplice_main_conf["genome"] / "dna"),
                "{output.score_file} {threads}",
            ]
        )



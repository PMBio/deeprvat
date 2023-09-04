import pandas as pd
import os
from pathlib import Path


configfile: "config/deeprvat_annotation_config.yaml"


# init general

species = config.get("species") or "homo_sapiens"
genome_assembly = config.get("genome_assembly") or "GRCh38"
fasta_dir = Path(config["fasta_dir"])
fasta_file_name = config["fasta_file_name"]
deeprvat_parent_path = Path(config["deeprvat_repo_dir"])
annotation_python_file = (
    deeprvat_parent_path / "deeprvat" / "annotations" / "annotations.py"
)
setup_shell_path = (
    deeprvat_parent_path / "deeprvat" / "annotations" / "setup_annotation_workflow.sh"
)
included_chromosomes = config["included_chromosomes"]
variant_file = config["variant_file_path"]
pybedtools_tmp_path = Path(config["pybedtools_tmp_path"])
repo_dir = Path(config["repo_dir"])
saved_deepripe_models_path = repo_dir / "faatpipe" / "data" / "deepripe_models"

# init modules
load_bfc = " ".join([config["bcftools_load_cmd"], "&&"])
load_hts = " ".join([config["htslib_load_cmd"], "&&"])
load_perl = " ".join([config["perl_load_cmd"], "&&"])
load_vep = " ".join([config["vep_load_cmd"], "&&"])

# init data path
vcf_pattern = config["vcf_file_pattern"]
bcf_dir = Path(config["bcf_dir"])
anno_tmp_dir = Path(config["anno_tmp_dir"])
anno_dir = Path(config["anno_dir"])
metadata_dir = Path(config["metadata_dir"])

# init cadd
cadd_shell_path = repo_dir / "CADD-scripts" / "CADD.sh"
cadd_snv_file = config["cadd_snv_file"]
cadd_indel_file = config["cadd_indel_file"]


# init vep
vep_source_dir = repo_dir / "ensembl-vep"
vep_cache_dir = Path(config["vep_cache_dir"])
vep_plugin_dir = Path(config.get("vep_plugin_dir")) or ""
vep_input_format = config.get("vep_input_format") or "vcf"
vep_nfork = int(config.get("vep_nfork") or 5)


# init plugIns
spliceAI_snv_file = config["spliceAI_snv_file"]
spliceAI_indel_file = config["spliceAI_indel_file"]
primateAIfile = config["primateAI_file"]

pvcf_blocks_df = pd.read_csv(
    metadata_dir / config["pvcf_blocks_file"],
    sep="\t",
    header=None,
    names=["Index", "Chromosome", "Block", "First position", "Last position"],
    dtype={"Chromosome": str},
).set_index("Index")
# init absplice
n_cores_absplice = int(config.get("n_cores_absplice") or 4)
# Filter out which chromosomes to work with
pvcf_blocks_df = pvcf_blocks_df[
    pvcf_blocks_df["Chromosome"].isin([str(c) for c in included_chromosomes])
]

chr_mapping = pd.Series(
    [str(x) for x in range(1, 23)] + ["X", "Y"], index=[str(x) for x in range(1, 25)]
)
inv_chr_mapping = pd.Series(
    [str(x) for x in range(1, 25)], index=[str(x) for x in range(1, 23)] + ["X", "Y"]
)

pvcf_blocks_df["chr_name"] = chr_mapping.loc[pvcf_blocks_df["Chromosome"].values].values

chromosomes = pvcf_blocks_df["chr_name"]
block = pvcf_blocks_df["Block"]


rule all:
    input:
        anno_dir
        / "current_annotations_absplice_deepsea_deepripe_parclip_hg2_k5.parquet",


rule aggregate_and_merge_absplice:
    input:
        abscore_files=expand(
            [anno_tmp_dir / "absplice" / (vcf_pattern + "_AbSplice_DNA.csv")],
            zip,
            chr=chromosomes,
            block=block,
        ),
        current_annotation_file=anno_dir
        / "current_annotations_deepsea_deepripe_parclip_hg2_k5.parquet",
    output:
        annotations=anno_dir
        / "current_annotations_absplice_deepsea_deepripe_parclip_hg2_k5.parquet",
        scores=anno_tmp_dir / "abSplice_score_file.parquet",

    shell:
        " ".join(
            [
                "python",
                f"{annotation_python_file}",
                "get-abscores",
                "{input.current_annotation_file}",
                str(anno_tmp_dir / "absplice"),
                "{output.annotations}",
                "{output.scores}",
            ]
        )


rule merge_deepripe_k5:
    input:
        anno_dir / "current_annotations_deepripe_parclip_hg2.parquet",
        deepripe_file=anno_dir / "all_variants.wID.k5.deepripe.csv",
    output:
        anno_dir / "current_annotations_deepripe_parclip_hg2_k5.parquet",

    shell:
        " ".join(
            [
                "python",
                f"{annotation_python_file}",
                "merge-deepripe",
                "{input.annotations}",
                "{input.deepripe_file}",
                "{output}",
                "k5",
            ]
        )


rule merge_deepripe_hg2:
    input:
        anno_dir / "current_annotations_deepripe_parclip.parquet",
        deepripe_file=anno_dir / "all_variants.wID.hg2.deepripe.csv",
    output:
        anno_dir / "current_annotations_deepripe_parclip_hg2.parquet",
    shell:
        " ".join(
            [
                "python",
                f"{annotation_python_file}",
                "merge-deepripe",
                "{input.annotations}",
                "{input.deepripe_file}",
                "{output}",
                "hg2",
            ]
        )


rule merge_deepripe_parclip:
    input:
        annotations=anno_dir / "current_annotations_processed.parquet",
        deepripe_file=anno_dir / "all_variants.wID.parclip.deepripe.csv",
    output:
        anno_dir / "current_annotations_deepripe_parclip.parquet",

    shell:
        " ".join(
            [
                "python",
                f"{annotation_python_file}",
                "merge-deepripe",
                "{input.annotations}",
                "{input.deepripe_file}",
                "{output}",
                "parclip",
            ]
        )


rule merge_deepsea_pcas:
    input:
        annotations=anno_dir / "current_annotations_deepripe_parclip_hg2_k5.parquet",
        deepsea_pcas=anno_dir / "deepSea_pca" / "deepsea_pca.parquet",
    output:
        anno_dir / "current_annotations_deepsea_deepripe_parclip_hg2_k5.parquet",

    shell:
        " ".join(
            [
                "python",
                f"{annotation_python_file}",
                "merge-deepsea-pcas",
                "{input.annotations}",
                "{input.deepsea_pcas}",
                "{output}",
            ]
        )


rule mv_absplice_files:
    input:
        str(
            repo_dir
            / "absplice"
            / "example"
            / "data"
            / "results"
            / "hg38"
            / (vcf_pattern + "_AbSplice_DNA.csv")
        ),
    output:
        anno_tmp_dir / "absplice" / (vcf_pattern + "_AbSplice_DNA.csv"),
    shell:
        " ".join(
            [
                "mkdir",
                "-p",
                str(anno_tmp_dir / "absplice"),
                "&&",
                "cp",
                "{input}",
                "{output}",
            ]
        )


rule absplice:
    conda:
        "absplice"
    input:
        vcf=expand(
            [
                repo_dir
                / "absplice/example/data/resources/analysis_files/vcf_files"
                / (vcf_pattern),
            ],
            zip,
            chr=chromosomes,
            block=block,
        ),
        config=repo_dir / "absplice" / "example" / "workflow" / "mv_config.done",
    output:
        expand(
            [
                str(
                    repo_dir
                    / "absplice"
                    / "example"
                    / "data"
                    / "results"
                    / "hg38"
                    / (vcf_pattern + "_AbSplice_DNA.csv")
                ),
            ],
            zip,
            chr=chromosomes,
            block=block,
        ),

       threads: n_cores_absplice
    
    shell:
        f"""python -m snakemake -s {str(repo_dir/"absplice"/"example"/"workflow"/"Snakefile")} -j 1 --use-conda --rerun-incomplete --directory {str(repo_dir/"absplice"/"example"/"workflow")} -c{n_cores_absplice} """


rule mod_config_absplice:
    output:
        repo_dir / "absplice" / "example" / "workflow" / "mv_config.done",
    shell:
        f""" rm {repo_dir}/absplice/example/workflow/config.yaml && mv {deeprvat_parent_path}/pipelines/resources/absplice_config.yaml {repo_dir}/absplice/example/workflow/config.yaml && touch {repo_dir}/absplice/example/workflow/mv_config.done"""


rule link_files_absplice:
    input:
        anno_tmp_dir / (vcf_pattern + "_variants_header.vcf.gz"),
    output:
        repo_dir
        / "absplice/example/data/resources/analysis_files/vcf_files"
        / (vcf_pattern),
    shell:
        " ".join(["ln", "-s", "-r", "{input}", "{output}"])


# f"""rm -r {repo_dir}/absplice/example/data/resources/analysis_files/vcf_files && mkdir {repo_dir}/absplice/example/data/resources/analysis_files/vcf_files && ln -s -r {vcf_dir}/* {repo_dir}/absplice/example/data/resources/analysis_files/vcf_files/"""


rule all_deepSea:
    input:
        expand(
            [
                anno_dir / (vcf_pattern + ".CLI.deepseapredict.diff.tsv"),
            ],
            zip,
            chr=chromosomes,
            block=block,
        ),
        anno_dir / "deepSea_pca" / "deepsea_pca.parquet",


rule deepSea_PCA:
    input:
        anno_dir / "all_variants.wID.deepSea.csv",
    output:
        anno_dir / "deepSea_pca" / "deepsea_pca.parquet",
    shell:
        " ".join(
            [
                "mkdir -p",
                str(anno_dir / "deepSea_pca"),
                "&&",
                "python",
                f"{annotation_python_file}",
                "deepsea-pca",
                "--n-components 100",
                "{input}",
                str(anno_dir / "deepSea_pca"),
            ]
        )


rule add_ids_deepSea:
    input:
        variant_file=variant_file,
        annotation_file=anno_dir / "all_variants.deepSea.csv",
    output:
        anno_dir / "all_variants.wID.deepSea.csv",
    shell:
        " ".join(
            [
                "python",
                f"{annotation_python_file}",
                "add-ids",
                "{input.annotation_file}",
                "{input.variant_file}",
                "{output}",
            ]
        )


rule concat_deepSea:
    input:
        expand(
            [
                anno_dir / (vcf_pattern + ".CLI.deepseapredict.diff.tsv"),
            ],
            zip,
            chr=chromosomes,
            block=block,
        ),
    output:
        anno_dir / "all_variants.deepSea.csv",

    shell:
        " ".join(
        [
            "python",
            f"{annotation_python_file}",
            "concatenate-deepripe",
            "--included-chromosomes",
            ",".join(included_chromosomes),
            "--sep '\t'",
            f"{anno_dir}",
            str(vcf_pattern + ".CLI.deepseapredict.diff.tsv").format(
        chr="{{chr}}", block="{{block}}"
                ),
                str(metadata_dir / config["pvcf_blocks_file"]),
                str(
                    anno_dir / "all_variants.deepSea.csv",
                ),
            ]
        )


rule deepSea:
    input:
        variants=anno_tmp_dir / (vcf_pattern + "_variants_header.vcf.gz"),
        fasta=fasta_dir / fasta_file_name,
        setup=repo_dir / "annotation-workflow-setup.done",
    output:
        anno_dir / (vcf_pattern + ".CLI.deepseapredict.diff.tsv"),
    conda:
        "kipoi-veff2"
    shell:
        "kipoi_veff2_predict {input.variants} {input.fasta} {output} -l 1000 -m 'DeepSEA/predict' -s 'diff'"


rule all_deepripe:
    input:
        expand(
            [
                anno_dir / (vcf_pattern + "_variants.parclip_deepripe.csv"),
            ],
            zip,
            chr=chromosomes,
            block=block,
        ),
        anno_dir / "all_variants.deepripe.csv",
        anno_dir / "deepripe_pca" / "deepripe_pca.parquet",


rule add_ids_deepripe_parclip:
    input:
        variant_file=variant_file,
        annotation_file=anno_dir / "all_variants.deepripe.parclip.csv",
    output:
        anno_dir / "all_variants.wID.parclip.deepripe.csv",
    shell:
        " ".join(
            [
                "python",
                f"{annotation_python_file}",
                "add-ids",
                "{input.annotation_file}",
                "{input.variant_file}",
                "{output}",
            ]
        )


rule add_ids_deepripe_hg2:
    input:
        variant_file=variant_file,
        annotation_file=anno_dir / "all_variants.deepripe.hg2.csv",
    output:
        anno_dir / "all_variants.wID.hg2.deepripe.csv",
    shell:
        " ".join(
            [
                "python",
                f"{annotation_python_file}",
                "add-ids",
                "{input.annotation_file}",
                "{input.variant_file}",
                "{output}",
            ]
        )


rule add_ids_deepripe_k5:
    input:
        variant_file=variant_file,
        annotation_file=anno_dir / "all_variants.deepripe.k5.csv",
    output:
        anno_dir / "all_variants.wID.k5.deepripe.csv",
    shell:
        " ".join(
            [
                "python",
                f"{annotation_python_file}",
                "add-ids",
                "{input.annotation_file}",
                "{input.variant_file}",
                "{output}",
            ]
        )


rule concat_deepRiPe_parclip:
    input:
        files=expand(
            [
                anno_dir / (vcf_pattern + "_variants.parclip_deepripe.csv"),
            ],
            zip,
            chr=chromosomes,
            block=block,
        ),
    output:
        anno_dir / "all_variants.deepripe.parclip.csv",

    shell:
        " ".join(
        [
            "python",
            f"{annotation_python_file}",
            "concatenate-deepripe",
            "--included-chromosomes",
            ",".join(included_chromosomes),
            f"{anno_dir}",
            str(vcf_pattern + "_variants.parclip_deepripe.csv").format(
        chr="{{chr}}", block="{{block}}"
                ),
                str(metadata_dir / config["pvcf_blocks_file"]),
                "{output}"
            ]
        )


rule concat_deepRiPe_eclip_hg2:
    input:
        files=expand(
            [
                anno_dir / (vcf_pattern + "_variants.eclip_hg2_deepripe.csv"),
            ],
            zip,
            chr=chromosomes,
            block=block,
        ),
    output:
        anno_dir / "all_variants.deepripe.hg2.csv",

    shell:
        " ".join(
        [
            "python",
            f"{annotation_python_file}",
            "concatenate-deepripe",
            "--included-chromosomes",
            ",".join(included_chromosomes),
            f"{anno_dir}",
            str(vcf_pattern + "_variants.eclip_hg2_deepripe.csv").format(
        chr="{{chr}}", block="{{block}}"
                ),
                str(metadata_dir / config["pvcf_blocks_file"]),
                "{output}"
            ]
        )


rule concat_deepRiPe_eclip_k5:
    input:
        files=expand(
            [
                anno_dir / (vcf_pattern + "_variants.eclip_k5_deepripe.csv"),
            ],
            zip,
            chr=chromosomes,
            block=block,
        ),
    output:
        anno_dir / "all_variants.deepripe.k5.csv",

    shell:
        " ".join(
        [
            "python",
            f"{annotation_python_file}",
            "concatenate-deepripe",
            "--included-chromosomes",
            ",".join(included_chromosomes),
            f"{anno_dir}",
            str(vcf_pattern + "_variants.eclip_k5_deepripe.csv").format(
        chr="{{chr}}", block="{{block}}"
                ),
            str(metadata_dir / config["pvcf_blocks_file"]),
            "{output}"
            ]
        )


rule deepRiPe_parclip:
    input:
        variants=anno_tmp_dir / (vcf_pattern + "_variants.vcf"),
        fasta=fasta_dir / fasta_file_name,
        setup=repo_dir / "annotation-workflow-setup.done",
    output:
        anno_dir / (vcf_pattern + "_variants.parclip_deepripe.csv"),

    shell:
        f"mkdir -p {pybedtools_tmp_path/'parclip'} && python {annotation_python_file} scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path/'parclip'} {saved_deepripe_models_path} 'parclip'"
         


rule deepRiPe_eclip_hg2:
    input:
        variants=anno_tmp_dir / (vcf_pattern + "_variants.vcf"),
        fasta=fasta_dir / fasta_file_name,
        setup=repo_dir / "annotation-workflow-setup.done",
    output:
        anno_dir / (vcf_pattern + "_variants.eclip_hg2_deepripe.csv"),

    shell:
        f"mkdir -p {pybedtools_tmp_path/'hg2'} && python {annotation_python_file} scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path/'hg2'} {saved_deepripe_models_path} 'eclip_hg2'"


rule deepRiPe_eclip_k5:
    input:
        variants=anno_tmp_dir / (vcf_pattern + "_variants.vcf"),
        fasta=fasta_dir / fasta_file_name,
        setup=repo_dir / "annotation-workflow-setup.done",
    output:
        anno_dir / (vcf_pattern + "_variants.eclip_k5_deepripe.csv"),

    shell:
        f"mkdir -p {pybedtools_tmp_path/'k5'} && python {annotation_python_file} scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path/'k5'} {saved_deepripe_models_path} 'eclip_k5'"


rule all_vep:
    input:
        expand(
            [
                anno_dir / (vcf_pattern + "_vep_anno.tsv"),
            ],
            zip,
            chr=chromosomes,
            block=block,
        ),
        anno_dir / "current_annotations.parquet",
        anno_dir / "current_annotations_processed.parquet",


rule process_merged_annotations:
    input:
        anno_dir / "current_annotations.parquet",
    output:
        anno_dir / "current_annotations_processed.parquet",
    shell:
        " ".join(
            [
                "python",
                f"{annotation_python_file}",
                "process-annotations",
                "{input}",
                "{output}",
            ]
        )


rule combine_annotations:
    input:
        expand(
            [
                anno_dir / (vcf_pattern + "_vep_anno.tsv"),
            ],
            zip,
            chr=chromosomes,
            block=block,
        ),
        variant_file=variant_file,
    output:
        anno_dir / "current_annotations.parquet",
    shell:
        " ".join(
        [
            "python",
            f"{annotation_python_file}",
            "concatenate-annotations",
            f"{anno_dir}",
            "--included-chromosomes",
            ",".join(included_chromosomes),
            str(vcf_pattern + "_vep_anno.tsv").format(
        chr="{{chr}}", block="{{block}}"
                ),
                "{input.variant_file}",
                # vcf_pattern + "_cadd_anno.tsv.gz",
                str(metadata_dir / config["pvcf_blocks_file"]),
                "{output}",
            ]
        )


rule vep:
    input:
        vcf=anno_tmp_dir / (vcf_pattern + "_stripped.vcf.gz"),
        fasta=fasta_dir / fasta_file_name,
        setup=repo_dir / "annotation-workflow-setup.done",
    output:
        anno_dir / (vcf_pattern + "_vep_anno.tsv"),
    threads: vep_nfork
    shell:
        " ".join(
            [
                load_perl,
                load_hts,
                load_bfc,
                load_vep,
                "vep",
                #str(vep_source_dir / "vep"),
                "--input_file",
                "{input.vcf}",
                "--output_file",
                "{output}",
                "--species",
                str(species),
                "--assembly",
                str(genome_assembly),
                "--format",
                str(vep_input_format),
                "--offline",
                "--cache",
                "--dir_cache",
                str(vep_cache_dir),
                "--dir_plugins",
                str(vep_plugin_dir),
                "--force_overwrite",
                "--fork",
                str(vep_nfork),
                "--fasta",
                "{input.fasta}",
                "--everything",
                "--tab",
                "--total_length",
                "--no_escape",
                "--xref_refseq",
                "--force_overwrite",
                "--no_stats",
                f"--plugin CADD,{cadd_snv_file},{cadd_indel_file}",
                f"--plugin SpliceAI,snv={spliceAI_snv_file},indel={spliceAI_indel_file}",
                f"--plugin PrimateAI,{primateAIfile}",
            ]
        )


rule extract_with_header:
    input:
        bcf_dir / (vcf_pattern + ".bcf"),
    output:
        anno_tmp_dir / (vcf_pattern + "_variants_header.vcf.gz"),
    shell:
        (
            load_bfc
            + load_hts
            + """ bcftools view  -s '' --force-samples {input} |bgzip  > {output}"""
        )


# load_bfc + """  bcftools query -f '%CHROM\\t%POS\\t%ID\\t%REF\\t%ALT\\t%QUAL\\t%FILTER\\n'    --print-header {input} > {output}"""


rule strip_chr_name:
    input:
        anno_tmp_dir / (vcf_pattern + "_variants.vcf"),
    output:
        anno_tmp_dir / (vcf_pattern + "_stripped.vcf.gz"),
    shell:
        f"{load_hts} cut -c 4- {{input}} |bgzip > {{output}}"


rule extract_variants:
    input:
        bcf_dir / (vcf_pattern + ".bcf"),
    output:
        anno_tmp_dir / (vcf_pattern + "_variants.vcf"),
    shell:
        " ".join(
            [
                load_bfc,
                "bcftools query -f",
                "'%CHROM\t%POS\t%ID\t%REF\t%ALT\n'",
                "{input} > {output}",
            ]
        )


rule setup:
    output:
        repo_dir / "annotation-workflow-setup.done",
    shell:
        f""" {load_vep} bash {setup_shell_path}  {vep_cache_dir} {vep_plugin_dir} {repo_dir}"""

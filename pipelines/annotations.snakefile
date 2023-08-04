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
annotation_python_file = deeprvat_parent_path / "deeprvat" / "annotations" / "annotations.py"
setup_shell_path = deeprvat_parent_path / "deeprvat" / "annotations" / "setup_annotation_workflow.sh"
included_chromosomes = config["included_chromosomes"]
variant_file = config["variant_file_path"]
pybedtools_tmp_path = config["pybedtools_tmp_path"]
repo_dir = Path(config["repo_dir"])
saved_deepripe_models_path = repo_dir / "faatpipe" / "data" / "deepripe_models"

# init modules
load_bfc = " ".join([config["bcftools_load_cmd"], "&&"])
load_hts = " ".join([config["htslib_load_cmd"], "&&"])
load_perl = " ".join([config["perl_load_cmd"], "&&"])
load_vep = " ".join([config["vep_load_cmd"], "&&"])

# init data path
vcf_pattern = config["vcf_file_pattern"]
vcf_dir = Path(config["vcf_dir"])
anno_tmp_dir = Path(config["anno_tmp_dir"])
anno_dir = Path(config["anno_dir"])
metadata_dir = Path(config["metadata_dir"])

# init cadd
cadd_shell_path = repo_dir/ "CADD-scripts" / "CADD.sh"
cadd_snv_file = config["cadd_snv_file"]
cadd_indel_file = config["cadd_indel_file"]

# init vep
vep_source_dir = repo_dir / "ensembl_vep"
vep_cache_dir = Path(config["vep_cache_dir"])
vep_plugin_dir = Path(config.get("vep_plugin_dir")) or ""
vep_input_format = config.get("vep_input_format") or "vcf"
vep_nfork = config.get("vep_nfork") or 5
cadd_ncores = config.get("cadd_ncores") or 5

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
        anno_dir / "current_annotations.parquet",


rule merge_deepripe_pcas:
    input:
        annotations=anno_dir / "current_annotations_processed.parquet",
        deepripe_pcas=anno_dir / "deepripe_pca" / "deepripe_pca.parquet",
    output:
        anno_dir / "current_annotations_deepripe.parquet",
    resources:
        mem_mb=lambda wildcards, attempt: 12500 * (attempt + 1),
    shell:
        " ".join(
            [
                "python",
                f"{annotation_python_file}",
                "merge-deepripe-pcas",
                "{input.annotations}",
                "{input.deepripe_pcas}",
                "{output}",
            ]
        )


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
        anno_dir / "all_variants.deepSea.csv",
    output:
        directory(anno_dir / "deepSea_pca" / "deepsea_pca.parquet"),
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
                str(anno_dir / "current_annotations.parquet"),
                str(anno_dir / "deepSea_pca"),
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
    resources:
        mem_mb=lambda wildcards, attempt: 30000 * (attempt + 1),
    shell:
        " ".join(
        [
            "python",
            f"{annotation_python_file}",
            "concatenate-deepripe",
            "--included-chromosomes",
            ",".join(included_chromosomes),
            "--sep '\\t'",
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
        variants = anno_tmp_dir / (vcf_pattern + "_variants_header.vcf.gz"),
        fasta = fasta_dir / fasta_file_name,
        setup = repo_dir / "annotation-workflow-setup.done"
    output:
        anno_dir / (vcf_pattern + ".CLI.deepseapredict.diff.tsv"),
    resources:
        mem_mb=lambda wildcards, attempt: 25000 * (attempt + 1),
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


rule deepripe_PCA:
    input:
        anno_dir / "all_variants.deepripe.csv",
    output:
        directory(anno_dir / "deepripe_pca" / "deepripe_pca.parquet"),
    resources:
        mem_mb=lambda wildcards, attempt: 25000 * (attempt + 1),
    shell:
        " ".join(
            [
                "mkdir -p",
                str(anno_dir / "deepripe_pca"),
                "&&",
                "python",
                f"{annotation_python_file}",
                "deepripe-pca",
                "--n-components 59",
                "{input}",
                "deepripe_pca",
            ]
        )


rule concat_deepRiPe:
    input:
        files = expand(
            [
                anno_dir / (vcf_pattern + "_variants.parclip_deepripe.csv"),
            ],
            zip,
            chr=chromosomes,
            block=block,
        ),
        setup = repo_dir / "annotation-workflow-setup.done"
    output:
        anno_dir / "all_variants.deepripe.csv",
    resources:
        mem_mb=lambda wildcards, attempt: 15000 * (attempt + 1),
    shell:
        " ".join(
        [
            "python",
            f"{annotation_python_file}",
            "concatenate-deepripe",
            "--included-chromosomes",
            ",".join(included_chromosomes),
            "--comment-lines" f"{anno_dir}",
            str(vcf_pattern + "_variants.parclip_deepripe.csv").format(
        chr="{{chr}}", block="{{block}}"
                ),
                str(metadata_dir / config["pvcf_blocks_file"]),
                str(
                    anno_dir / "all_variants.deepripe.csv",
                ),
            ]
        )


rule deepRiPe:
    input:
        variants=anno_tmp_dir / (vcf_pattern + "_variants.vcf"),
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / (vcf_pattern + "_variants.parclip_deepripe.csv"),
    resources:
        mem_mb=lambda wildcards, attempt: 25000 * (attempt + 1),
    shell:
        f"python {annotation_python_file} scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path} {saved_deepripe_models_path} 'parclip'"


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
    resources:
        mem_mb=lambda wildcards, attempt: 15000 * (attempt + 1),
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
    output:
        anno_dir / "current_annotations.parquet",
    resources:
        mem_mb=lambda wildcards, attempt: 15000 * (attempt + 1),
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
                str(variant_file),
                # vcf_pattern + "_cadd_anno.tsv.gz",
                str(metadata_dir / config["pvcf_blocks_file"]),
                str(
                    anno_dir / "current_annotations.parquet",
                ),
            ]
        )


rule vep:
    input:
        vcf=anno_tmp_dir / (vcf_pattern + "_stripped.vcf.gz"),
        fasta=fasta_dir / fasta_file_name,
        setup = repo_dir / "annotation-workflow-setup.done",
    output:
        anno_dir / (vcf_pattern + "_vep_anno.tsv"),
    threads: vep_nfork
    resources:
        mem_mb=lambda wildcards, attempt: 5000 * (attempt + 1),
    shell:
        " ".join(
            [
                str(vep_source_dir / "vep"),
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
                "--canonical",
                "--per_gene",
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


rule all_cadd:
    input:
        expand(
            [
                anno_dir / (vcf_pattern + "_cadd_anno.tsv.gz"),
            ],
            zip,
            chr=chromosomes,
            block=block,
        ),


rule annototate_cadd:
    input:
        variants =anno_tmp_dir / (vcf_pattern + "_stripped.vcf.gz"),
        setup = repo_dir / "annotation-workflow-setup.done",
    output:
        anno_dir / (vcf_pattern + "_cadd_anno.tsv.gz"),
    threads: cadd_ncores
    resources:
        mem_mb=20000,
    shell:
        " ".join(
            [
                str(cadd_shell_path),
                "-a",
                "-g",
                str(genome_assembly),
                "-c",
                str(cadd_ncores),
                "-o",
                "{output}",
                "{input.variants}",
            ]
        )


rule extract_with_header:
    input:
        vcf_dir / (vcf_pattern + ".vcf.gz"),
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
        vcf_dir / (vcf_pattern + ".vcf.gz"),
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
        repo_dir / "annotation-workflow-setup.done"
    shell:
        f""" bash {setup_shell_path}  {vep_cache_dir} {vep_plugin_dir} {repo_dir}""" 
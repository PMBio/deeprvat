import pandas as pd
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
saved_deepripe_models_path = (
    Path(config["faatpipe_repo_dir"]) / "data" / "deepripe_models"
)
merge_nthreads = int(config.get("merge_nthreads") or 64)

# If modules are used we load them here
load_bfc = " ".join([config["bcftools_load_cmd"], "&&" if config["bcftools_load_cmd"] else ""])
load_hts = " ".join([config["htslib_load_cmd"], "&&" if config["htslib_load_cmd"] else ""])
load_perl = " ".join([config["perl_load_cmd"], "&&" if config["perl_load_cmd"] else ""])
load_vep = " ".join([config["vep_load_cmd"], "&&" if config["vep_load_cmd"] else ""])


# init data path
vcf_pattern = config["vcf_file_pattern"]
bcf_dir = Path(config["bcf_dir"])
anno_tmp_dir = Path(config["anno_tmp_dir"])
anno_dir = Path(config["anno_dir"])
metadata_dir = Path(config["metadata_dir"])
vep_plugin_repo = Path(config["vep_plugin_repo"])
condel_config_path = vep_plugin_repo / "config" / "Condel" / "config"


# init cadd PLugin
cadd_snv_file = config["cadd_snv_file"]
cadd_indel_file = config["cadd_indel_file"]


# init vep
vep_source_dir = Path(config["vep_repo_dir"])
vep_cache_dir = Path(config.get("vep_cache_dir")) or vep_source_dir / "cache"
vep_plugin_dir = Path(config.get("vep_plugin_dir")) or vep_source_dir / "Plugin"
vep_input_format = config.get("vep_input_format") or "vcf"
vep_nfork = int(config.get("vep_nfork") or 5)
af_mode= config.get("af_mode") or "af"

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
absplice_repo_dir = Path(config["absplice_repo_dir"])
n_cores_absplice = int(config.get("n_cores_absplice") or 4)
ncores_merge_absplice = int(config.get("n_cores_merge_absplice") or 64)
#init deepripe
n_jobs_deepripe = int(config.get("n_jobs_deepripe") or 8)
# init kipoi-veff2
kipoi_repo_dir = Path(config["kipoiveff_repo_dir"])
ncores_addis  = int(config.get("n_jobs_addids") or 32)
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
        anno_dir / "current_annotations_absplice.parquet",


rule aggregate_and_merge_absplice:
    input:
        abscore_files=expand(
            [anno_tmp_dir / "absplice" / (vcf_pattern + "_AbSplice_DNA.csv")],
            zip,
            chr=chromosomes,
            block=block,
        ),
        current_annotation_file=anno_dir / "vep_deepripe_deepsea.parquet"
    output:
        annotations=anno_dir / "current_annotations_absplice.parquet",
        scores=anno_tmp_dir / "abSplice_score_file.parquet"

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
                f"{ncores_merge_absplice}"
            ])




rule merge_deepsea_pcas:
    input:
        annotations=anno_dir / "vep_deepripe.parquet",
        deepsea_pcas=anno_dir / "deepSea_pca" / "deepsea_pca.parquet",
    output:
        anno_dir / "vep_deepripe_deepsea.parquet"
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

rule concat_annotations:
    input:
        pvcf = metadata_dir / config['pvcf_blocks_file'],
        anno_dir = anno_dir,
        vcf_files=
            expand([anno_dir / f"{vcf_pattern}_merged.parquet"],
            zip,
            chr=chromosomes,
            block=block)
    output: anno_dir / "vep_deepripe.parquet"
    shell:
        " ".join([
            "python",
            str(annotation_python_file),
            "concat-annotations",
            "{input.pvcf}",
            "{input.anno_dir}",
            f"{str(vcf_pattern+'_merged.parquet').format(chr='{{chr}}', block='{{block}}')}",
            "{output}",
            f" --included-chromosomes {','.join(included_chromosomes)}"
            ])

rule merge_annotations:
    input:
        vep =  anno_dir / (vcf_pattern + "_vep_anno.tsv"),
        deepripe_parclip =  anno_dir / (vcf_pattern + "_variants.parclip_deepripe.csv.gz"),
        deepripe_k5 = anno_dir / (vcf_pattern + "_variants.eclip_k5_deepripe.csv.gz"),
        deepripe_hg2 = anno_dir / (vcf_pattern + "_variants.eclip_hg2_deepripe.csv.gz"),
        variant_file = variant_file


    output:
        anno_dir / f"{vcf_pattern}_merged.parquet",
    shell: "HEADER=$(grep  -n  '#Uploaded_variation' "+"{input.vep}" +"| head | cut -f 1 -d ':') && python "+f"{annotation_python_file} "+"merge-annotations $(($HEADER-1)) {input.vep} {input.deepripe_parclip} {input.deepripe_hg2} {input.deepripe_k5} {input.variant_file} {output}"

rule mv_absplice_files:
    input:
        str(
            absplice_repo_dir
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
                absplice_repo_dir
                / "example/data/resources/analysis_files/vcf_files"
                / (vcf_pattern),
            ],
            zip,
            chr=chromosomes,
            block=block,
        ),
        config=absplice_repo_dir / "example" / "workflow" / "mv_config.done",

    output:
        expand(
            [
                str(
                    absplice_repo_dir
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
        f"""python -m snakemake -s {str(absplice_repo_dir/"example"/"workflow"/"Snakefile")} -j 1 --use-conda --rerun-incomplete --directory {str(absplice_repo_dir /"example"/"workflow")} -c"""+"{threads}"


rule mod_config_absplice:
    output:
        absplice_repo_dir / "example" / "workflow" / "mv_config.done",
    shell:
        f""" rm {absplice_repo_dir}/example/workflow/config.yaml && cp {deeprvat_parent_path}/pipelines/resources/absplice_config.yaml {absplice_repo_dir}/example/workflow/config.yaml && touch {absplice_repo_dir}/example/workflow/mv_config.done"""

rule link_files_absplice:
    input:
        anno_tmp_dir / (vcf_pattern + "_variants_header.vcf.gz"),
    output:
        absplice_repo_dir
        / "example/data/resources/analysis_files/vcf_files"
        / (vcf_pattern),
    shell:
        " ".join(["ln", "-s", "-r", "{input}", "{output}"])




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
                str(anno_dir / "deepSea_pca"/ "pca.pkl"),
                str(anno_dir / "deepSea_pca"),
            ]
        )


rule add_ids_deepSea:
    input:
        variant_file=variant_file,
        annotation_file=anno_dir / "all_variants.deepSea.csv",
    output:
        anno_dir / "all_variants.wID.deepSea.csv"
    threads: ncores_addis

    shell:
        " ".join(
            [
                "python",
                f"{annotation_python_file}",
                "add-ids",
                "{input.annotation_file}",
                "{input.variant_file}",
                "{threads}",
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
    output:
        anno_dir / (vcf_pattern + ".CLI.deepseapredict.diff.tsv"),
    conda:
        "kipoi-veff2"
    shell:
        "kipoi_veff2_predict {input.variants} {input.fasta} {output} -l 1000 -m 'DeepSEA/predict' -s 'diff'"




rule deepRiPe_parclip:
    input:
        variants=anno_tmp_dir / (vcf_pattern + "_variants.vcf"),
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / (vcf_pattern + "_variants.parclip_deepripe.csv.gz"),

    shell:
        f"mkdir -p {pybedtools_tmp_path/'parclip'} && python {annotation_python_file} scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path/'parclip'} {saved_deepripe_models_path} {{threads}} 'parclip'"


rule deepRiPe_eclip_hg2:
    input:
        variants=anno_tmp_dir / (vcf_pattern + "_variants.vcf"),
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / (vcf_pattern + "_variants.eclip_hg2_deepripe.csv.gz"),
    threads: lambda wildcards, attempt: n_jobs_deepripe * attempt
    shell:
        f"mkdir -p {pybedtools_tmp_path/'hg2'} && python {annotation_python_file} scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path/'hg2'} {saved_deepripe_models_path} {{threads}} 'eclip_hg2'"


rule deepRiPe_eclip_k5:
    input:
        variants=anno_tmp_dir / (vcf_pattern + "_variants.vcf"),
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / (vcf_pattern + "_variants.eclip_k5_deepripe.csv.gz"),

    threads: lambda wildcards, attempt: n_jobs_deepripe * attempt
    shell:
        f"mkdir -p {pybedtools_tmp_path/'k5'} && python {annotation_python_file} scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path/'k5'} {saved_deepripe_models_path} {{threads}} 'eclip_k5'"



rule vep:
    input:
        vcf=anno_tmp_dir / (vcf_pattern + "_stripped.vcf.gz"),
        fasta=fasta_dir / fasta_file_name,
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
                # str(vep_source_dir / "vep"),
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
                "--{af_mode}",
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
                "--per_gene",
                "--pick_order biotype,mane_select,mane_plus_clinical,canonical,appris,tsl,ccds,rank,length,ensembl,refseq",
                f"--plugin CADD,{cadd_snv_file},{cadd_indel_file}",
                f"--plugin SpliceAI,snv={spliceAI_snv_file},indel={spliceAI_indel_file}",
                f"--plugin PrimateAI,{primateAIfile}",
                f"--plugin Condel,{condel_config_path},s,2"
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

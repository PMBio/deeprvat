from pathlib import Path
from glob import glob
from itertools import chain
import re
import yaml


configfile: "deeprvat_annotation_config.yaml"


## helper functions
def tryint(s):
    """
    Return an int if possible, or `s` unchanged.
    """
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """
    Turn a string into a list of string and number chunks.

    >>> alphanum_key("z23a")
    ["z", 23, "a"]

    """
    return [tryint(c) for c in re.split("([0-9]+)", s)]


def human_sort(l):
    """
    Sort a list by regarding int values inside strings as numbers
    """
    l.sort(key=alphanum_key)


# init general

species = config.get("species") or "homo_sapiens"
genome_assembly = config.get("genome_assembly") or "GRCh38"
fasta_dir = Path(config["fasta_dir"])
fasta_file_name = config["fasta_file_name"]
gtf_file = fasta_dir / config["gtf_file_name"]
gene_id_file = config.get("gene_id_parquet")

deeprvat_parent_path = Path(config["deeprvat_repo_dir"])
annotation_python_file = (
    deeprvat_parent_path / "deeprvat" / "annotations" / "annotations.py"
)
annotation_columns_yaml_file = (
    config.get("annotation_columns_yaml_file") # "deeprvat_parent_path/example/config/annotation_colnames_filling_values.yaml"
)
included_chromosomes = config.get(
    "included_chromosomes", [f"{c}" for c in range(1, 23)] + ["X", "Y"]
)

preprocess_dir = Path(config.get("preprocessing_workdir", ""))
variant_pq = (
    config.get("variant_parquetfile_path")
    or preprocess_dir / "norm" / "variants" / "variants.parquet"
)
genotype_file = (
    config.get("genotype_file_path") or preprocess_dir / "preprocessed" / "genotypes.h5"
)
saved_deepripe_models_path = (
    Path(config["faatpipe_repo_dir"]) / "data" / "deepripe_models"
)
merge_nthreads = int(config.get("merge_nthreads") or 8)

# If modules are used we load them here
load_bfc = (
    f'{config["bcftools_load_cmd"]} &&' if config.get("bcftools_load_cmd") else ""
)
load_hts = f'{config["htslib_load_cmd"]} &&' if config.get("htslib_load_cmd") else ""
load_perl = f'{config["perl_load_cmd"]} &&' if config.get("perl_load_cmd") else ""
load_vep = f'{config["vep_load_cmd"]} &&' if config.get("vep_load_cmd") else ""

# init data path
source_variant_file_pattern = config["source_variant_file_pattern"]
source_variant_file_type = config["source_variant_file_type"]
source_variant_dir = Path(config["source_variant_dir"])
anno_tmp_dir = Path(config["anno_tmp_dir"])
anno_dir = Path(config["anno_dir"])
pybedtools_tmp_path = Path(
    config.get("pybedtools_tmp_path", anno_tmp_dir / "pybedtools")
)

# init vep
vep_source_dir = Path(config["vep_repo_dir"])
vep_cache_dir = Path(config.get("vep_cache_dir")) or vep_source_dir / "cache"
vep_plugin_dir = Path(config.get("vep_plugin_dir")) or vep_source_dir / "Plugin"
vep_input_format = config.get("vep_input_format") or "vcf"
vep_nfork = int(config.get("vep_nfork") or 5)
af_mode = config.get("af_mode")
condel_config_path = vep_plugin_dir / "config" / "Condel" / "config"
vep_online = config.get("vep_online", False ) 
vep_no_cache = config.get("vep_no_cache", False)
if config.get("additional_vep_plugin_cmds"):
    VEP_plugin_cmds = config["additional_vep_plugin_cmds"].values()
else:
    VEP_plugin_cmds = []

# init deepSEA
deepSEA_tmp_dir = config.get("deepSEA_tmp_dir") or anno_tmp_dir / "deepSEA_PCA"
deepSEA_pca_obj = (
    config.get("deepSEA_pca_object") or anno_tmp_dir / "deepSEA_PCA" / "pca.npy"
)
deepSEA_means_and_sds = (
    config.get("deepSEA_means_and_sds")
    or anno_tmp_dir / "deepSEA_PCA" / "deepSEA_means_SDs.parquet"
)
n_pca_components = config.get("deepsea_pca_n_components", 100)

# init deepripe
n_jobs_deepripe = int(config.get("n_jobs_deepripe") or 8)

# init kipoi-veff2
kipoi_repo_dir = Path(config["kipoiveff_repo_dir"])
ncores_addis = int(config.get("n_jobs_addids") or 32)

# init absplice
n_cores_absplice = int(config.get("n_cores_absplice") or 4)
ncores_merge_absplice = int(config.get("n_cores_merge_absplice") or 8)
ncores_agg_absplice = int(config.get("ncores_agg_absplice") or 4)

source_variant_file_pattern_complete = (
    source_variant_file_pattern + "." + source_variant_file_type
)

file_paths = [
    glob(
        str(
            source_variant_dir
            / source_variant_file_pattern_complete.format(chr=c, block="*")
        )
    )
    for c in included_chromosomes
]

file_paths = list(chain.from_iterable(file_paths))
human_sort(file_paths)
file_stems = [
    re.compile(source_variant_file_pattern.format(chr="(\d+|X|Y)", block="\d+"))
    .search(i)
    .group()
    for i in file_paths
]

absplice_download_dir = (
    config.get("absplice_download_dir")
    or anno_tmp_dir / "absplice" 
)
absplice_output_dir = config.get("absplice_output_dir", anno_tmp_dir / "absplice")
vcf_id = anno_tmp_dir / "{file_stem}"
vcf_dir = anno_tmp_dir

config_download_path = (
    deeprvat_parent_path / "pipelines" / "resources" / "absplice_config_download.yaml"
)
with open(config_download_path, "r") as fd:
    config_download = yaml.safe_load(fd)

config_pred_path = (
    deeprvat_parent_path / "pipelines" / "resources" / "absplice_config_pred.yaml"
)
with open(config_pred_path, "r") as fd:
    config_pred = yaml.safe_load(fd)

config_cat_path = (
    deeprvat_parent_path / "pipelines" / "resources" / "absplice_config_cat.yaml"
)
with open(config_cat_path, "r") as fd:
    config_cat = yaml.safe_load(fd)

absplice_main_conf_path = (
    deeprvat_parent_path / "pipelines" / "resources" / "config_absplice.yaml"
)
with open(absplice_main_conf_path, "r") as fd:
    absplice_main_conf = yaml.safe_load(fd)

include_absplice = config.get('include_absplice', True)
include_deepSEA = config.get('include_deepSEA', True)



rule all:
    input:
        chckpt = anno_dir / 'chckpts' / 'select_rename_fill_columns.chckpt',
        annotations = anno_dir / 'annotations.parquet'

if not gene_id_file:
    gene_id_file = fasta_dir / "protein_coding_genes.parquet"

    rule create_gene_id_file:
        input:
            gtf_file,
        output:
            gene_id_file,
        resources:
            mem_mb=lambda wildcards, attempt: 15_000 * (attempt + 1),
        shell:
            " ".join(
                [f"deeprvat_annotations", "create-gene-id-file", "{input}", "{output}"]
            )

rule extract_with_header:
    input:
        source_variant_dir / f"{{file_stem}}.{source_variant_file_type}",
    output:
        anno_tmp_dir / "{file_stem}_variants_header.vcf.gz",
    shell:
        (
            load_bfc
            + load_hts
            + """ bcftools view  -G {input} |bgzip  > {output}"""
        )


rule extract_variants:
    input:
        source_variant_dir / f"{{file_stem}}.{source_variant_file_type}",
    output:
        anno_tmp_dir / "{file_stem}_variants.vcf",
    shell:
        " ".join(
            [
                load_bfc,
                "bcftools query -f",
                "'%CHROM\t%POS\t%ID\t%REF\t%ALT\n'",
                "{input} > {output}",
            ]
        )

rule strip_chr_name:
    input:
        rules.extract_variants.output,
    output:
        anno_tmp_dir / "{file_stem}_stripped.vcf.gz",
    shell:
        load_hts + """ cut -c 4- {input} | awk -F'\\t' 'BEGIN {{OFS = FS}} {{$3 = "chr"$1"_"$2"_"$4"_"$5; print}}' |bgzip > {output} """


rule vep:
    input:
        vcf=rules.strip_chr_name.output,
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / "{file_stem}_vep_anno.tsv",
    threads: vep_nfork
    params: 
        af =lambda w:  f'--{af_mode}' if af_mode else '',
        offline = lambda w:  '--offline' if not vep_online else '',
        cache = lambda w:  '--cache' if not vep_no_cache else '--database',
        dir_cache = lambda w: f'--dir_cache {str(vep_cache_dir)}' if not vep_no_cache else ''
    resources:
        mem_mb=lambda wildcards, attempt: 5_000 * (attempt + 1),
    shell:
        " ".join(
            [
                load_perl,
                load_hts,
                load_bfc,
                load_vep,
                "vep",
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
                "{params.af}",
                "{params.offline}",
                "{params.cache}",
                "{params.dir_cache}",
                "--dir_plugins",
                str(vep_plugin_dir),
                "--fork",
                str(vep_nfork),
                "--fasta",
                "{input.fasta}",
                "--tab",
                "--total_length",
                "--no_escape",
                "--polyphen s",
                "--sift s",
                "--canonical",
                "--protein",
                "--biotype",
                "--force_overwrite",
                "--no_stats",
                "--per_gene",
                "--pick_order biotype,mane_select,mane_plus_clinical,canonical,appris,tsl,ccds,rank,length,ensembl,refseq",
            ]
            + ["--plugin " + i for i in VEP_plugin_cmds]
        )



include: "annotations/deepripe.snakefile"

rule merge_annotations:
    input:
        vep=rules.vep.output,
        deepripe_parclip=rules.deepRiPe_parclip.output,
        deepripe_k5=rules.deepRiPe_eclip_k5.output,
        deepripe_hg2=rules.deepRiPe_eclip_hg2.output,
        variant_file=variant_pq,
        vcf_file=rules.extract_variants.output,
        col_yaml_file=annotation_columns_yaml_file,
    output:
        anno_dir / "{file_stem}_merged.parquet",
    resources:
        mem_mb=lambda wildcards, attempt: 5_000 * (attempt + 1),
    shell:
        (
            "HEADER=$(grep  -n  '#Uploaded_variation' "
            + "{input.vep}"
            + "| head | cut -f 1 -d ':') && deeprvat_annotations "
            + "merge-annotations $(($HEADER-1)) {input.vep} {input.deepripe_parclip} {input.deepripe_hg2} {input.deepripe_k5} {input.variant_file} {input.vcf_file} {output} {input.col_yaml_file}"
        )



rule concat_annotations:
    input:
        vcf_files=expand(
            rules.merge_annotations.output,
            file_stem=file_stems,
        ),
    output:
        annotations = anno_dir / "annotations.parquet",
        chckpt = anno_dir / 'chckpts' / "concat_annotations.chckpt"
    params:
        joined=lambda w, input: ",".join(input.vcf_files),
    resources:
        mem_mb=lambda wildcards, attempt: 15_000 * (attempt + 1),
    shell:
        " ".join(
            [
                "deeprvat_annotations",
                "concat-annotations",
                "{params.joined}",
                "{output.annotations}",
            ]
        )+" && touch {output.chckpt}"


if(include_deepSEA):
    include: "annotations/deepSEA.snakefile"

    rule merge_deepsea_pcas:
        input:
            chckpt = rules.concat_annotations.output.chckpt,
            deepsea_pcas=rules.add_ids_deepSea.output,
            col_yaml_file=annotation_columns_yaml_file,
        output:
            chckpt = anno_dir / 'chckpts' / 'merge_deepsea_pcas.chckpt'
        resources:
            mem_mb=lambda wildcards, attempt: 30_000 * (attempt + 1),
        params: 
            annotations_in = anno_dir / "annotations.parquet",
            annotations_out = anno_dir / "annotations.parquet",
        shell:
            " ".join(
                [
                    "deeprvat_annotations",
                    "merge-deepsea-pcas",
                    "{params.annotations_in}",
                    "{input.deepsea_pcas}",
                    "{input.col_yaml_file}",
                    "{params.annotations_out}",
                ]
            )+" && touch {output.chckpt}"
else: 
    rule omit_deepSEA:
        input: 
            chckpt = rules.concat_annotations.output.chckpt,
        output: 
            chckpt = anno_dir / 'chckpts' / 'merge_deepsea_pcas.chckpt' 
        shell:
            'touch {output.chckpt}'

if (include_absplice):
    include: "annotations/absplice.snakefile"
    rule merge_absplice_scores:
        input:
            absplice_scores=rules.aggregate_absplice_scores.output.score_file,
            chckpt = anno_dir / 'chckpts' / 'merge_deepsea_pcas.chckpt'
        output:
            chckpt = anno_dir / 'chckpts' / 'merge_absplice_scores.chckpt'
        threads: ncores_merge_absplice
        params: 
            annotations_in = anno_dir / "annotations.parquet",
            annotations_out = anno_dir / "annotations.parquet",
        resources:
            mem_mb=lambda wildcards, attempt: 19_000 * (attempt + 1),
        shell:
            " ".join(
                [
                    "deeprvat_annotations",
                    "merge-abscores",
                    "{params.annotations_in}",
                    "{input.absplice_scores}",
                    "{params.annotations_out}",
                ]
            )+" && touch {output.chckpt}"
            
else: 
    rule omit_absplice:
        input: 
            chckpt = anno_dir / 'chckpts' / 'merge_deepsea_pcas.chckpt'
        output:
            chckpt = anno_dir / 'chckpts' / 'merge_absplice_scores.chckpt'
        shell: 
            'touch {output.chckpt}'


if af_mode is None:
    rule calculate_allele_frequency:
        input:
            genotype_file=genotype_file,
            variants=variant_pq,
        output:
            allele_frequencies = anno_tmp_dir / "af_df.parquet",
        resources:
            mem_mb=lambda wildcards, attempt: 15_000 * (attempt + 1),
        shell:
            " ".join(
                [
                    f"deeprvat_annotations",
                    "get-af-from-gt",
                    "{input.genotype_file}",
                    "{input.variants}",
                    "{output.allele_frequencies}",
                ]
            )


    rule merge_allele_frequency:
        input:
            allele_frequencies=rules.calculate_allele_frequency.output.allele_frequencies,
            chckpt_absplice = anno_dir / 'chckpts' / 'merge_absplice_scores.chckpt',
            chckpt_deepsea = anno_dir / 'chckpts' / 'merge_deepsea_pcas.chckpt',
            ckckpt_concat_annotations = rules.concat_annotations.output.chckpt,
            
        output:
            #annotations = anno_dir / "annotations.parquet",
            chckpt = anno_dir / 'chckpts' / 'merge_allele_frequency.chckpt'
        params:
            annotations_in = anno_dir / "annotations.parquet",
            annotations_out = anno_dir / "annotations.parquet",
        resources:
            mem_mb=lambda wildcards, attempt: 15_000 * (attempt + 1),
        shell:
            " ".join(
                [
                    f"deeprvat_annotations",
                    "merge-af",
                    "{params.annotations_in}",
                    "{input.allele_frequencies}",
                    "{params.annotations_out}",
                ]
            )+" && touch {output.chckpt}"

    rule calculate_MAF:
        input:
            chckpt = rules.merge_allele_frequency.output.chckpt,
        output:
            chckpt = anno_dir / 'chckpts' / 'calculate_MAF.chckpt'
        params: 
            annotations_in = rules.merge_allele_frequency.params.annotations_out,
            annotations_out = anno_dir / "annotations.parquet",
        resources:
            mem_mb=lambda wildcards, attempt: 15_000 * (attempt + 1),
        shell:
            " ".join([f"deeprvat_annotations", "calculate-maf", "{params.annotations_in}", "{params.annotations_out}"])+ " && touch {output.chckpt}"

elif(af_mode == 'af_gnomade'):
    rule calculate_MAF:
        input:
            chckpt_absplice = anno_dir / 'chckpts' / 'merge_absplice_scores.chckpt',
            chckpt_deepsea = anno_dir / 'chckpts' / 'merge_deepsea_pcas.chckpt',
            ckckpt_concat_annotations = rules.concat_annotations.output.chckpt,
        output:
            chckpt = anno_dir / 'chckpts' / 'calculate_MAF.chckpt'
        params: 
            annotations_in = anno_dir / "annotations.parquet",
            annotations_out = anno_dir / "annotations.parquet",
        resources:
            mem_mb=lambda wildcards, attempt: 15_000 * (attempt + 1),
        shell:
            " ".join([f"deeprvat_annotations", "calculate-maf","--af-column-name gnomADe_AF" , "{params.annotations_in}", "{params.annotations_out}"])+ " && touch {output.chckpt}"

elif(af_mode == 'af_gnomadg'):
    rule calculate_MAF:
        input:
            chckpt_absplice = anno_dir / 'chckpts' / 'merge_absplice_scores.chckpt',
            chckpt_deepsea = anno_dir / 'chckpts' / 'merge_deepsea_pcas.chckpt',
            ckckpt_concat_annotations = rules.concat_annotations.output.chckpt,
        output:
            chckpt = anno_dir / 'chckpts' / 'calculate_MAF.chckpt'
        params: 
            annotations_in = anno_dir / "annotations.parquet",
            annotations_out = anno_dir / "annotations.parquet",
        resources:
            mem_mb=lambda wildcards, attempt: 15_000 * (attempt + 1),
        shell:
            " ".join([f"deeprvat_annotations", "calculate-maf", "--af-column-name gnomADg_AF", "{params.annotations_in}", "{params.annotations_out}" ])+ " && touch {output.chckpt}"

else: print('af_mode unknown')


rule add_gene_ids:
    input:
        gene_id_file=gene_id_file,
        chckpt = rules.calculate_MAF.output.chckpt
    output: 
        chckpt = anno_dir / 'chckpts' / 'add_gene_ids.chckpt'
    params: 
        annotations_in = rules.calculate_MAF.params.annotations_out,
        annotations_out = anno_dir / "annotations.parquet",
    resources:
        mem_mb=lambda wildcards, attempt: 19_000 * (attempt + 1),
    shell:
        " ".join(
            [
                f"deeprvat_annotations",
                "add-gene-ids",
                "{input.gene_id_file}",
                "{params.annotations_in}",
                "{params.annotations_out}",
            ]
        )+" && touch {output.chckpt}"


rule filter_by_exon_distance:
    input:
        gtf_file=gtf_file,
        protein_coding_genes=gene_id_file,
        chckpt = rules.add_gene_ids.output.chckpt
    output:
        chckpt = anno_dir / 'chckpts' / 'filter_by_exon_distance.chckpt'
    params: 
        annotations_in = rules.add_gene_ids.params.annotations_out,
        annotations_out = anno_dir / "annotations.parquet",
    resources:
        mem_mb=lambda wildcards, attempt: 25_000 * (attempt + 1),
    shell:
        " ".join(
            [
                f"deeprvat_annotations",
                "filter-annotations-by-exon-distance",
                "{params.annotations_in}",
                "{input.gtf_file}",
                "{input.protein_coding_genes}",
                "{params.annotations_out}",
            ]
        ) +" && touch {output.chckpt}"

rule compute_plof_column:
    input: 
        chckpt = rules.filter_by_exon_distance.output.chckpt
    output: 
        chckpt = anno_dir / 'chckpts' / 'compute_plof_column.chckpt'
    params:
        annotations_in = rules.filter_by_exon_distance.params.annotations_out,
        annotations_out = anno_dir / "annotations.parquet",
    resources: mem_mb=lambda wildcards, attempt: 15_000 * (attempt + 1),
    shell: 'deeprvat_annotations compute-plof {params.annotations_in} {params.annotations_out} && touch {output.chckpt}'




rule select_rename_fill_columns:
    input:
        yaml_file=annotation_columns_yaml_file,
        chckpt = rules.compute_plof_column.output.chckpt,
    output:
        chckpt = anno_dir / 'chckpts' / 'select_rename_fill_columns.chckpt'
    params: 
        annotations_in=rules.compute_plof_column.params.annotations_out,
        annotations_out = anno_dir / "annotations.parquet",
        unfilled = lambda w: f"--keep_unfilled {anno_dir / 'unfilled_annotations.parquet'}" if (config.get('keep_unfilled')) else ""
    resources:
        mem_mb=lambda wildcards, attempt: 15_000 * (attempt + 1),
    shell:
        " ".join(
            [
                f"deeprvat_annotations",
                "select-rename-fill-annotations",
                "{input.yaml_file}",
                "{params.annotations_in}",
                "{params.annotations_out}",
                "{params.unfilled}"
            ]
        ) +" && touch {output.chckpt}"




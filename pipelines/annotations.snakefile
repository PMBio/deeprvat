import pandas as pd
from pathlib import Path
from glob import glob
from itertools import chain
import re
import os
import yaml

configfile: "config/deeprvat_annotation_config.yaml"
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
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

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
gtf_file = fasta_dir / config['gtf_file_name']
gene_id_file = config.get('gene_id_parquet')

deeprvat_parent_path = Path(config["deeprvat_repo_dir"])
annotation_python_file = (
    deeprvat_parent_path / "deeprvat" / "annotations" / "annotations.py"
)
annotation_columns_yaml_file = config.get('annotation_columns_yaml_file') or deeprvat_parent_path/'pipelines'/'config'/'annotation_colnames_filling_values.yaml'
included_chromosomes = config.get(
    "included_chromosomes", [f"{c}" for c in range(1, 23)] + ["X", "Y"]
)

preprocess_dir = Path(config.get("preprocessing_workdir", ""))
variant_file = config.get("variant_file_path") or preprocess_dir / 'norm' / 'variants' / 'variants.tsv.gz'
genotype_file = config.get("genotype_file_path") or preprocess_dir / 'preprocessed' / 'genotypes.h5'
saved_deepripe_models_path = (
    Path(config["faatpipe_repo_dir"]) / "data" / "deepripe_models"
)
merge_nthreads = int(config.get("merge_nthreads") or 8)

# If modules are used we load them here
load_bfc = f'{config["bcftools_load_cmd"]} &&' if config.get("bcftools_load_cmd") else ""
load_hts = f'{config["htslib_load_cmd"]} &&' if config.get("htslib_load_cmd") else ""
load_perl = f'{config["perl_load_cmd"]} &&' if config.get("perl_load_cmd") else ""
load_vep = f'{config["vep_load_cmd"]} &&' if config.get("vep_load_cmd") else ""


# init data path
source_variant_file_pattern = config["source_variant_file_pattern"]
source_variant_file_type = config["source_variant_file_type"]
source_variant_dir = Path(config["source_variant_dir"])
anno_tmp_dir = Path(config["anno_tmp_dir"])
anno_dir = Path(config["anno_dir"])
pybedtools_tmp_path = Path(config.get("pybedtools_tmp_path" ,  anno_tmp_dir / 'pybedtools'))



# init vep
vep_source_dir = Path(config["vep_repo_dir"])
vep_cache_dir = Path(config.get("vep_cache_dir")) or vep_source_dir / "cache"
vep_plugin_dir = Path(config.get("vep_plugin_dir")) or vep_source_dir / "Plugin"
vep_input_format = config.get("vep_input_format") or "vcf"
vep_nfork = int(config.get("vep_nfork") or 5)
af_mode = config.get("af_mode") or "af"
condel_config_path = vep_plugin_dir / "config" / "Condel" / "config"
if config.get('additional_vep_plugin_cmds'):
    VEP_plugin_cmds = config['additional_vep_plugin_cmds'].values()
else:
    VEP_plugin_cmds = []

#init deepSEA
deepSEA_tmp_dir = config.get('deepSEA_tmp_dir')or anno_tmp_dir / 'deepSEA_PCA'
deepSEA_pca_obj = config.get('deepSEA_pca_object')or anno_tmp_dir / 'deepSEA_PCA' / 'pca.npy'
deepSEA_means_and_sds = config.get('deepSEA_means_and_sds')or anno_tmp_dir / 'deepSEA_PCA' / 'deepSEA_means_SDs.parquet'
n_pca_components = config.get('deepsea_pca_n_components', 100)

# init deepripe
n_jobs_deepripe = int(config.get("n_jobs_deepripe") or 8)

# init kipoi-veff2
kipoi_repo_dir = Path(config["kipoiveff_repo_dir"])
ncores_addis = int(config.get("n_jobs_addids") or 32)


# init absplice
absplice_repo_dir = Path(config["absplice_repo_dir"])
n_cores_absplice = int(config.get("n_cores_absplice") or 4)
ncores_merge_absplice = int(config.get("n_cores_merge_absplice") or 8)
ncores_agg_absplice = int(config.get("ncores_agg_absplice") or 4)

source_variant_file_pattern_complete = source_variant_file_pattern+'.'+source_variant_file_type
print(included_chromosomes)
file_paths = [
    glob(str(source_variant_dir / source_variant_file_pattern_complete.format(chr=c, block='*')))
    for c in included_chromosomes
]

file_paths = list(chain.from_iterable(file_paths))
human_sort(file_paths)
file_stems = [Path(p).stem.split('.')[0] for p in file_paths]

absplice_download_dir = config.get('absplice_download_dir') or  absplice_repo_dir /'example'/'data'/'resources'/'downloaded_files'
absplice_output_dir = config.get('absplice_output_dir', anno_tmp_dir /'absplice')
vcf_id = anno_tmp_dir / '{vcf_id}'
vcf_dir = anno_tmp_dir

config_download_path = deeprvat_parent_path/'pipelines'/'resources'/"absplice_config_download.yaml"
with open(config_download_path, "r") as fd:
    config_download = yaml.safe_load(fd)

config_pred_path = deeprvat_parent_path / 'pipelines'/'resources'/"absplice_config_pred.yaml"
with open(config_pred_path, "r") as fd:
    config_pred = yaml.safe_load(fd)

config_cat_path = deeprvat_parent_path / 'pipelines'/'resources'/"absplice_config_cat.yaml"
with open(config_cat_path, "r") as fd:
    config_cat = yaml.safe_load(fd)

absplice_main_conf_path = deeprvat_parent_path / 'pipelines'/'resources'/"config_absplice.yaml"
with open(absplice_main_conf_path, "r") as fd:
    absplice_main_conf = yaml.safe_load(fd)


include: Path('resources')/"absplice_download.snakefile"
include: Path('resources')/"absplice_splicing_pred_DNA.snakefile"
if absplice_main_conf['AbSplice_RNA'] == True:
    include: deeprvat_parent_path / 'deeprvat' / 'pipelines'/'resources'/"absplice_splicing_pred_RNA.snakefile"

all_absplice_output_files = list()
all_absplice_output_files.append(rules.all_download.input)
all_absplice_output_files.append(rules.all_predict_dna.input)

if absplice_main_conf['AbSplice_RNA'] == True:
    all_absplice_output_files.append(rules.all_predict_rna.input)


rule all:
    input:
        anno_dir / "vep_deepripe_deepsea_absplice_maf_pIDs_filtered_filled.parquet",

rule select_rename_fill_columns:
    input:
        yaml_file = annotation_columns_yaml_file,
        annotations_path = anno_dir / "vep_deepripe_deepsea_absplice_maf_pIDs_filtered.parquet",
    output:
        anno_dir / "vep_deepripe_deepsea_absplice_maf_pIDs_filtered_filled.parquet",
    shell:
        " ".join([
            f"deeprvat_annotations", 
            "select-rename-fill-annotations",
            "{input.yaml_file}",
            "{input.annotations_path}",
            "{output}"

        
        ])        

if not gene_id_file:
    gene_id_file = anno_tmp_dir / 'protein_coding_genes.parquet'
    rule create_gene_id_file:
        input: gtf_file
        output: gene_id_file
        shell:
            " ".join([
                f"deeprvat_annotations", 
                "create-protein-id-file",
                "{input}",
                "{output}"
            ])
            
rule filter_by_exon_distance:
    input:
        annotations_path = anno_dir / "vep_deepripe_deepsea_absplice_maf_pIDs.parquet",
        gtf_file = gtf_file,
        protein_coding_genes = gene_id_file
    output:
        anno_dir / "vep_deepripe_deepsea_absplice_maf_pIDs_filtered.parquet",
    shell:
        " ".join([
            f"deeprvat_annotations", 
            "filter-annotations-by-exon-distance",
            "{input.annotations_path}",
            "{input.gtf_file}",
            "{input.protein_coding_genes}",
            "{output}"
        ])

rule add_gene_ids:
    input: 
        gene_id_file = gene_id_file,
        annotations_path = anno_dir / "vep_deepripe_deepsea_absplice_maf.parquet",
    output: anno_dir / "vep_deepripe_deepsea_absplice_maf_pIDs.parquet",  
    shell:
        " ".join([
            f"deeprvat_annotations", 
            "add-protein-ids",
            "{input.gene_id_file}",
            "{input.annotations_path}",
            "{output}"
            ])


rule calculate_MAF:
    input:
        anno_dir / "vep_deepripe_deepsea_absplice_af.parquet"
    output:
        anno_dir / "vep_deepripe_deepsea_absplice_maf.parquet"
    shell:
        " ".join([
            f"deeprvat_annotations", 
            "calculate-maf",
            "{input}",
            "{output}"

        
        ])



rule merge_allele_frequency: 
    input:
        allele_frequencies =  anno_tmp_dir / "af_df.parquet",
        annotation_file = anno_dir / "vep_deepripe_deepsea_absplice.parquet"
    output:
        anno_dir / "vep_deepripe_deepsea_absplice_af.parquet"
    shell:
        " ".join([
            f"deeprvat_annotations", 
            "merge-af",
            "{input.annotation_file}",
            "{input.allele_frequencies}",
            "{output}"

        
        ])




rule calculate_allele_frequency:
    input: 
        genotype_file = genotype_file,
        variants = variant_file
    output:
        allele_frequencies = anno_tmp_dir / "af_df.parquet"
    shell:
        " ".join([
            f"deeprvat_annotations", 
            "get-af-from-gt",
            "{input.genotype_file}",
            "{input.variants}",
            "{output.allele_frequencies}"

        
        ])
        
        


rule merge_absplice_scores:
    input: 
        absplice_scores = anno_tmp_dir / "abSplice_score_file.parquet",
        current_annotation_file= anno_dir / "vep_deepripe_deepsea.parquet",
    output: 
        anno_dir / "vep_deepripe_deepsea_absplice.parquet"
    threads: ncores_merge_absplice
    shell: 
        " ".join(
            [
                "deeprvat_annotations",
                "merge-abscores",
                "{input.current_annotation_file}",
                "{input.absplice_scores}",
                "{output}",
            ])

rule aggregate_absplice_scores:
    input:
        abscore_files= expand([absplice_output_dir / absplice_main_conf['genome'] / 'dna' / '{file_stem}_variants_header.vcf.gz_AbSplice_DNA.csv'],file_stem = file_stems),
        current_annotation_file= anno_dir / "vep_deepripe_deepsea.parquet",
    output:
        score_file = anno_tmp_dir / "abSplice_score_file.parquet",
    threads: ncores_agg_absplice
    shell:
        " ".join(
            [
                "deeprvat_annotations",
                "aggregate-abscores {input.current_annotation_file}",
                str(absplice_output_dir / absplice_main_conf['genome'] / 'dna' ),
                "{output.score_file} {threads}"
            ])


rule merge_deepsea_pcas:
    input:
        annotations = anno_dir / "vep_deepripe.parquet",
        deepsea_pcas = anno_dir / "all_variants.wID.deepSea.parquet",
        col_yaml_file = annotation_columns_yaml_file
    output:
        anno_dir / "vep_deepripe_deepsea.parquet",
    shell:
        " ".join(
            [
                "deeprvat_annotations",
                "merge-deepsea-pcas",
                "{input.annotations}",
                "{input.deepsea_pcas}",
                "{input.col_yaml_file}",
                "{output}",
            ]
        )


rule concat_annotations:
    input:
        vcf_files=expand(
            [anno_dir / "{file_stem}_merged.parquet"],
            file_stem = file_stems,
        ),
    output:
        anno_dir / "vep_deepripe.parquet",
    params: joined=lambda w, input: ",".join(input.vcf_files)
    shell:
        " ".join(
            [
                "deeprvat_annotations",
                "concat-annotations",
                '{params.joined}',
                "{output}"
            ]
        )


rule merge_annotations:
    input:
        vep=anno_dir / (source_variant_file_pattern + "_vep_anno.tsv"),
        deepripe_parclip=anno_dir
        / (source_variant_file_pattern + "_variants.parclip_deepripe.csv.gz"),
        deepripe_k5=anno_dir
        / (source_variant_file_pattern + "_variants.eclip_k5_deepripe.csv.gz"),
        deepripe_hg2=anno_dir
        / (source_variant_file_pattern + "_variants.eclip_hg2_deepripe.csv.gz"),
        variant_file=variant_file,
        vcf_file= anno_tmp_dir / (source_variant_file_pattern + "_variants.vcf"),
    output:
        anno_dir / f"{source_variant_file_pattern}_merged.parquet",
    shell:
        (
            "HEADER=$(grep  -n  '#Uploaded_variation' "
            + "{input.vep}"
            + "| head | cut -f 1 -d ':') && deeprvat_annotations "
            + "merge-annotations $(($HEADER-1)) {input.vep} {input.deepripe_parclip} {input.deepripe_hg2} {input.deepripe_k5} {input.variant_file} {input.vcf_file} {output}"
        )

rule deepSea_PCA:
    input:
        deepsea_anno = str(anno_dir / "all_variants.deepSea.parquet")
    output:
        deepSEA_tmp_dir / "deepsea_pca.parquet",
    shell:
        " ".join(
            ["mkdir -p",
              str(deepSEA_tmp_dir),
              "&&",
              "deeprvat_annotations",
              "deepsea-pca",
              "{input.deepsea_anno}",
              f"{str(deepSEA_pca_obj)}",
              f"{str(deepSEA_means_and_sds)}",
              f"{deepSEA_tmp_dir}",
              f"--n-components {n_pca_components}"
            ]
        )


rule add_ids_deepSea:
    input:
        variant_file=variant_file,
        annotation_file=deepSEA_tmp_dir / "deepsea_pca.parquet",
    output:
        directory(anno_dir / "all_variants.wID.deepSea.parquet"),
    threads: ncores_addis
    shell:
        " ".join(
            [
                "deeprvat_annotations",
                "add-ids-dask",
                "{input.annotation_file}",
                "{input.variant_file}",
                "{threads}",
                "{output}",
            ]
        )


rule concat_deepSea:
    input:
        deepSEAscoreFiles = expand(
            [
                anno_dir
                / ("{file_stem}" + ".CLI.deepseapredict.diff.tsv"),
            ],
            file_stem = file_stems
        ),
    params: joined=lambda w, input: ",".join(input.deepSEAscoreFiles)
    threads:8
    output:
        anno_dir / "all_variants.deepSea.parquet",
    shell:
        " ".join(
        [
            "deeprvat_annotations",
            "concatenate-deepsea",
            "{params.joined}",
            "{output}",
            "{threads}"
            ]
        )


rule deepSea:
    input:
        variants=anno_tmp_dir
        / (source_variant_file_pattern + "_variants_header.vcf.gz"),
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / (source_variant_file_pattern + ".CLI.deepseapredict.diff.tsv"),
    threads: n_jobs_deepripe
    conda:
        "kipoi-veff2"
    shell:
        "kipoi_veff2_predict {input.variants} {input.fasta} {output} -l 1000 -m 'DeepSEA/predict' -s 'diff'"


rule deepRiPe_parclip:
    input:
        variants=anno_tmp_dir / (source_variant_file_pattern + "_variants.vcf"),
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / (source_variant_file_pattern + "_variants.parclip_deepripe.csv.gz"),
    threads: n_jobs_deepripe
    shell:
        f"mkdir -p {pybedtools_tmp_path / 'parclip'} && deeprvat_annotations scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path / 'parclip'} {saved_deepripe_models_path} {{threads}} 'parclip'"


rule deepRiPe_eclip_hg2:
    input:
        variants=anno_tmp_dir / (source_variant_file_pattern + "_variants.vcf"),
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / (source_variant_file_pattern + "_variants.eclip_hg2_deepripe.csv.gz"),
    threads: lambda wildcards, attempt: n_jobs_deepripe * attempt
    shell:
        f"mkdir -p {pybedtools_tmp_path / 'hg2'} && deeprvat_annotations scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path / 'hg2'} {saved_deepripe_models_path} {{threads}} 'eclip_hg2'"


rule deepRiPe_eclip_k5:
    input:
        variants=anno_tmp_dir / (source_variant_file_pattern + "_variants.vcf"),
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / (source_variant_file_pattern + "_variants.eclip_k5_deepripe.csv.gz"),
    threads: lambda wildcards, attempt: n_jobs_deepripe * attempt
    shell:
        f"mkdir -p {pybedtools_tmp_path / 'k5'} && deeprvat_annotations scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path / 'k5'} {saved_deepripe_models_path} {{threads}} 'eclip_k5'"


rule vep:
    input:
        vcf=anno_tmp_dir / (source_variant_file_pattern + "_stripped.vcf.gz"),
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / (source_variant_file_pattern + "_vep_anno.tsv"),
    threads: vep_nfork
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
                "--tab",
                "--total_length",
                "--no_escape",
                "--polyphen s",
                "--sift s",
                "--canonical",
                "--protein", 
                "--biotype", 
                "--af", 
                "--force_overwrite",
                "--no_stats",
                "--per_gene",
                "--pick_order biotype,mane_select,mane_plus_clinical,canonical,appris,tsl,ccds,rank,length,ensembl,refseq"
            ]+['--plugin '+i for i in VEP_plugin_cmds]
        )


rule extract_with_header:
    input:
        source_variant_dir
        / (source_variant_file_pattern + f".{source_variant_file_type}"),
    output:
        anno_tmp_dir / (source_variant_file_pattern + "_variants_header.vcf.gz"),
    shell:
        (
            load_bfc
            + load_hts
            + """ bcftools view  -s '' --force-samples {input} |bgzip  > {output}"""
        )



rule strip_chr_name:
    input:
        anno_tmp_dir / (source_variant_file_pattern + "_variants.vcf"),
    output:
        anno_tmp_dir / (source_variant_file_pattern + "_stripped.vcf.gz"),
    shell:
        f"{load_hts} cut -c 4- {{input}} |bgzip > {{output}}"


rule extract_variants:
    input:
        source_variant_dir
        / (source_variant_file_pattern + f".{source_variant_file_type}"),
    output:
        anno_tmp_dir / (source_variant_file_pattern + "_variants.vcf"),
    shell:
        " ".join(
            [
                load_bfc,
                "bcftools query -f",
                "'%CHROM\t%POS\t%ID\t%REF\t%ALT\n'",
                "{input} > {output}",
            ]
        )
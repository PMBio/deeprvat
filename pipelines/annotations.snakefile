import pandas as pd
from pathlib import Path
import os
import yaml

configfile: "config/deeprvat_annotation_config.yaml"

# init general

species = config.get("species") or "homo_sapiens"
genome_assembly = config.get("genome_assembly") or "GRCh38"
fasta_dir = Path(config["fasta_dir"])
fasta_file_name = config["fasta_file_name"]
gtf_file = fasta_dir / config['gtf_file_name']

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
genotype_file = config.get("genotype_file_path") or preprocess_dir / 'preprocesed' / 'genotypes.h5'
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
source_variant_dir = Path(config["source_variant_dir"])
anno_tmp_dir = Path(config["anno_tmp_dir"])
anno_dir = Path(config["anno_dir"])
metadata_dir = Path(config["metadata_dir"])
pybedtools_tmp_path = Path(config.get("pybedtools_tmp_path" ,  anno_tmp_dir / 'pybedtools'))



# init vep
vep_source_dir = Path(config["vep_repo_dir"])
vep_cache_dir = Path(config.get("vep_cache_dir")) or vep_source_dir / "cache"
vep_plugin_dir = Path(config.get("vep_plugin_dir")) or vep_source_dir / "Plugin"
vep_input_format = config.get("vep_input_format") or "vcf"
vep_nfork = int(config.get("vep_nfork") or 5)
af_mode = config.get("af_mode") or "af"
condel_config_path = vep_plugin_dir / "config" / "Condel" / "config"



pvcf_blocks_file = config["pvcf_blocks_file"]
pvcf_blocks_df = pd.read_csv(
    metadata_dir / pvcf_blocks_file,
    sep="\t",
    header=None,
    names=["Index", "Chromosome", "Block", "First position", "Last position"],
    dtype={"Chromosome": str},
).set_index("Index")

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

# init absplice
absplice_repo_dir = Path(config["absplice_repo_dir"])
n_cores_absplice = int(config.get("n_cores_absplice") or 4)
ncores_merge_absplice = int(config.get("n_cores_merge_absplice") or 8)
ncores_agg_absplice = int(config.get("ncores_agg_absplice") or 4)



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
            f"python {annotation_python_file}", 
            "select-rename-fill-annotations",
            "{input.yaml_file}",
            "{input.annotations_path}",
            "{output}"

        
        ])        
rule filter_by_exon_distance:
    input:
        annotations_path = anno_dir / "vep_deepripe_deepsea_absplice_maf_pIDs.parquet",
        gtf_file = gtf_file,
        protein_coding_genes = anno_tmp_dir / 'protein_coding_genes.parquet'
    output:
        anno_dir / "vep_deepripe_deepsea_absplice_maf_pIDs_filtered.parquet",
    shell:
        " ".join([
            f"python {annotation_python_file}", 
            "filter-annotations-by-exon-distance",
            "{input.annotations_path}",
            "{input.gtf_file}",
            "{input.protein_coding_genes}",
            "{output}"

        
        ])


rule add_protein_ids:
    input: 
        protein_id_file = anno_tmp_dir / 'protein_coding_genes.parquet',
        annotations_path = anno_dir / "vep_deepripe_deepsea_absplice_maf.parquet",
    output: anno_dir / "vep_deepripe_deepsea_absplice_maf_pIDs.parquet",
    shell:
        " ".join([
            f"python {annotation_python_file}", 
            "add-protein-ids",
            "{input.protein_id_file}",
            "{input.annotations_path}",
            "{output}"

        
        ])


rule create_protein_id_file:
    input: gtf_file
    output: anno_tmp_dir / 'protein_coding_genes.parquet'
    shell:
        " ".join([
            f"python {annotation_python_file}", 
            "create-protein-id-file",
            "{input}",
            "{output}"

        
        ])


rule calculate_MAF:
    input:
        anno_dir / "vep_deepripe_deepsea_absplice_af.parquet"
    output:
        anno_dir / "vep_deepripe_deepsea_absplice_maf.parquet"
    shell:
        " ".join([
            f"python {annotation_python_file}", 
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
            f"python {annotation_python_file}", 
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
            f"python {annotation_python_file}", 
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
                "python",
                f"{annotation_python_file}",
                "merge-abscores",
                "{input.current_annotation_file}",
                "{input.absplice_scores}",
                "{output}",
            ])

rule aggregate_absplice_scores:
    input:
        abscore_files= expand([absplice_output_dir / absplice_main_conf['genome'] / 'dna' / f'{source_variant_file_pattern}_variants_header.vcf.gz_AbSplice_DNA.csv'], zip, chr=chromosomes, block=block),
        current_annotation_file= anno_dir / "vep_deepripe_deepsea.parquet",
    output:
        score_file = anno_tmp_dir / "abSplice_score_file.parquet",
    threads: ncores_agg_absplice
    shell:
        " ".join(
            [
                "python",
                f"{annotation_python_file}",
                "aggregate-abscores {input.current_annotation_file}",
                str(absplice_output_dir / absplice_main_conf['genome'] / 'dna' ),
                "{output.score_file} {threads}"
            ])


rule merge_deepsea_pcas:
    input:
        annotations=anno_dir / "vep_deepripe.parquet",
        deepsea_pcas=deepSEA_tmp_dir / "deepsea_pca.parquet",
    output:
        anno_dir / "vep_deepripe_deepsea.parquet",
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
        pvcf=metadata_dir / pvcf_blocks_file,
        vcf_files=expand(
            [anno_dir / f"{source_variant_file_pattern}_merged.parquet"],
            zip,
            chr=chromosomes,
            block=block,
        ),
    output:
        anno_dir / "vep_deepripe.parquet",
    shell:
        " ".join(
            [
                "python",
                str(annotation_python_file),
                "concat-annotations",
                "{input.pvcf}",
                str(anno_dir),
                f"{str(source_variant_file_pattern + '_merged.parquet').format(chr='{{chr}}',block='{{block}}')}",
                "{output}",
                f" --included-chromosomes {','.join(included_chromosomes)}",
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
    output:
        anno_dir / f"{source_variant_file_pattern}_merged.parquet",
    shell:
        (
            "HEADER=$(grep  -n  '#Uploaded_variation' "
            + "{input.vep}"
            + "| head | cut -f 1 -d ':') && python "
            + f"{annotation_python_file} "
            + "merge-annotations $(($HEADER-1)) {input.vep} {input.deepripe_parclip} {input.deepripe_hg2} {input.deepripe_k5} {input.variant_file} {output}"
        )


rule deepSea_PCA:
    input:
        deepsea_anno = str(anno_dir / "all_variants.wID.deepSea.parquet")
    output:
        deepSEA_tmp_dir / "deepsea_pca.parquet",
    shell:
        " ".join(
            ["mkdir -p",
              str(deepSEA_tmp_dir),
              "&&",
              "python",
              f"{annotation_python_file}",
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
        annotation_file=anno_dir / "all_variants.deepSea.parquet",
    output:
        directory(anno_dir / "all_variants.wID.deepSea.parquet"),
    threads: ncores_addis
    shell:
        " ".join(
            [
                "python",
                f"{annotation_python_file}",
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
                / (source_variant_file_pattern + ".CLI.deepseapredict.diff.tsv"),
            ],
            zip,
            chr=chromosomes,
            block=block,
        ),
        pvcf_blocks_file = str(metadata_dir / pvcf_blocks_file)
    output:
        anno_dir / "all_variants.deepSea.parquet",
    shell:
        " ".join(
        [
            "python",
            f"{annotation_python_file}",
            "concatenate-deepsea",
            "--included-chromosomes",
            ",".join(included_chromosomes),
            str(anno_dir),
            str(
        source_variant_file_pattern + ".CLI.deepseapredict.diff.tsv"
                ).format(chr="{{chr}}", block="{{block}}"),
                "{input.pvcf_blocks_file}",
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
    shell:
        f"mkdir -p {pybedtools_tmp_path / 'parclip'} && python {annotation_python_file} scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path / 'parclip'} {saved_deepripe_models_path} {{threads}} 'parclip'"


rule deepRiPe_eclip_hg2:
    input:
        variants=anno_tmp_dir / (source_variant_file_pattern + "_variants.vcf"),
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / (source_variant_file_pattern + "_variants.eclip_hg2_deepripe.csv.gz"),
    threads: lambda wildcards, attempt: n_jobs_deepripe * attempt
    shell:
        f"mkdir -p {pybedtools_tmp_path / 'hg2'} && python {annotation_python_file} scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path / 'hg2'} {saved_deepripe_models_path} {{threads}} 'eclip_hg2'"


rule deepRiPe_eclip_k5:
    input:
        variants=anno_tmp_dir / (source_variant_file_pattern + "_variants.vcf"),
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / (source_variant_file_pattern + "_variants.eclip_k5_deepripe.csv.gz"),
    threads: lambda wildcards, attempt: n_jobs_deepripe * attempt
    shell:
        f"mkdir -p {pybedtools_tmp_path / 'k5'} && python {annotation_python_file} scorevariants-deepripe {{input.variants}} {anno_dir}  {{input.fasta}} {pybedtools_tmp_path / 'k5'} {saved_deepripe_models_path} {{threads}} 'eclip_k5'"


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
            ]+['--plugin '+i for i in config['additional_vep_plugin_cmds'].values()]
        )


rule extract_with_header:
    input:
        source_variant_dir
        / (source_variant_file_pattern + f".{config['source_variant_file_type']}"),
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
        / (source_variant_file_pattern + f".{config['source_variant_file_type']}"),
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
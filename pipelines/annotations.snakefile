import pandas as pd
from pathlib import Path


configfile: "config/deeprvat_annotation_config.yaml"


# init general

species = config.get("species") or "homo_sapiens"
genome_assembly = config.get("genome_assembly") or "GRCh38"
fasta_dir = Path(config["fasta_dir"])
fasta_file_name = config["fasta_file_name"]

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
cadd_shell_path = Path(config["cadd_src_dir"]) / "CADD.sh"
cadd_snv_file = config["cadd_snv_file"]
cadd_indel_file = config["cadd_indel_file"]

# init vep
vep_source_dir = Path(config["vep_src_dir"])
vep_cache_dir = Path(config["vep_cache_dir"])
vep_plugin_dir = Path(config.get("vep_plugin_dir")) or ""
vep_input_format = config.get("vep_input_format") or "vcf"
vep_nfork = config.get("vep_nfork") or 5

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
    pvcf_blocks_df["Chromosome"].isin([str(c) for c in config["included_chromosomes"]])
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
        expand(
            [
                anno_dir / (vcf_pattern + "_cadd_anno.tsv.gz"),
                anno_dir / (vcf_pattern + "_vep_anno"),
            ],
            zip,
            chr=chromosomes,
            block=block,
        ),


rule vep:
    input:
        vcf=anno_tmp_dir / (vcf_pattern + "_stripped.vcf.gz"),
        fasta=fasta_dir / fasta_file_name,
    output:
        anno_dir / (vcf_pattern + "_vep_anno"),
    threads: vep_nfork
    resources:
        runtime="8:00",
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
                "--vcf",
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


rule annototate:
    input:
        anno_tmp_dir / (vcf_pattern + "_stripped.vcf.gz"),
    output:
        anno_dir / (vcf_pattern + "_cadd_anno.tsv.gz"),
    shell:
        " ".join(
            [
                str(load_hts),
                str(load_perl),
                str(load_vep),
                str(cadd_shell_path),
                "-a",
                "-g",
                str(genome_assembly),
                "-o",
                "{output}",
                "{input}",
            ]
        )


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

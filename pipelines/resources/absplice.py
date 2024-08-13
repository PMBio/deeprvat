# codign_genes.py
# mmsplice_splicemap.py
# absplice_dna.py
import snakemake
import click


@click.group()
def cli():
    pass


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
def codign_genes(input, output):
    import pyranges as pr

    gr = pr.read_gtf(input["gtf_file"])
    gr = gr[(gr.Feature == "gene") & (gr.gene_type == "protein_coding")]
    df_genes = gr.df

    df_genes["gene_id_orig"] = df_genes["gene_id"]
    df_genes["PAR_Y"] = df_genes["gene_id"].apply(lambda x: "PAR_Y" in x)
    df_genes = df_genes[df_genes["PAR_Y"] == False]
    df_genes["gene_id"] = df_genes["gene_id"].apply(lambda x: x.split(".")[0])

    columns = [
        "Chromosome",
        "Start",
        "End",
        "Strand",
        "gene_id",
        "gene_id_orig",
        "gene_name",
        "gene_type",
    ]
    df_genes[columns].to_csv(output["coding_genes"], index=False)


@cli.command()
@click.argument("input_fasta", type=click.Path(exists=True))
@click.argument("input_vcf", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
def mmsplice_splicemap(input_fasta, input_vcf, output):
    from absplice import SpliceOutlier, SpliceOutlierDataloader

    dl = SpliceOutlierDataloader(
        input["fasta"],
        input["vcf"],
        splicemap5=list(input["splicemap_5"]),
        splicemap3=list(input["splicemap_3"]),
    )

    model = SpliceOutlier()
    model.predict_save(dl, output["result"])


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
def absplice_dna(input, output, extra_info):
    from absplice import SplicingOutlierResult

    splicing_result = SplicingOutlierResult(
        df_mmsplice=snakemake.input["mmsplice_splicemap"],
        df_spliceai=snakemake.input["spliceai"],
    )
    splicing_result.predict_absplice_dna(extra_info=snakemake.params["extra_info"])
    splicing_result._absplice_dna.to_csv(snakemake.output["absplice_dna"])

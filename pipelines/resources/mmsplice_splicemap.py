from absplice import SpliceOutlier, SpliceOutlierDataloader

dl = SpliceOutlierDataloader(
    snakemake.input['fasta'], snakemake.input['vcf'],
    splicemap5=list(snakemake.input['splicemap_5']),
    splicemap3=list(snakemake.input['splicemap_3'])
)

model = SpliceOutlier()
model.predict_save(dl, snakemake.output['result'])
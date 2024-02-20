import pandas as pd
import pyranges as pr

gr = pr.read_gtf(snakemake.input['gtf_file'])
gr = gr[(gr.Feature == 'gene') & (gr.gene_type == 'protein_coding')]
df_genes = gr.df

df_genes['gene_id_orig'] = df_genes['gene_id']
df_genes['PAR_Y'] = df_genes['gene_id'].apply(lambda x: 'PAR_Y' in x)
df_genes = df_genes[df_genes['PAR_Y'] == False]
df_genes['gene_id'] = df_genes['gene_id'].apply(lambda x: x.split('.')[0])

columns = [
    'Chromosome', 'Start', 'End', 'Strand',
    'gene_id', 'gene_id_orig', 'gene_name', 'gene_type'
]
df_genes[columns].to_csv(snakemake.output['coding_genes'], index=False)
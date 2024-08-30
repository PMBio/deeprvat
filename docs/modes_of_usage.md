# Modes of usage

DeepRVAT can be applied in various modes, presented here in increasing levels of complexity. For each of these scenarios, we provide a corresponding Snakemake pipeline.

## Precomputed gene impairment scores

_Note: Precomputed gene impairment scores are not yet available. They will be made available upon publication of the DeepRVAT manuscript._ 

For users running association testing on UKBB WES data, we provide precomputed gene impairment scores for all protein-coding genes with a qualifying variant within 300 bp of an exon. In this scenario, users are freed from processing of large WES data and may carry out highly computationally efficient association tests with the default DeepRVAT pipeline or the DeepRVAT+REGENIE integration.

Note that DeepRVAT scores are on a scale between 0 and 1, with a score closer to 0 indicating that the aggregate effect of variants in the gene is protective, and a score closer to 1 when the aggregate effect is deleterious.

For full details, see [here](precomputed_burdens).

## Pretrained models

Some users may wish to select variants or make variant-to-gene assignments differently from our methods, or to work on datasets other than UKBB. For this, we provide an ensemble of pretrained DeepRVAT gene impairment modules, which can be used for scoring individual-gene pairs for subsequent association testing. We also provide a pipeline for functional annotation of variants for compatibility with the pretrained modules.

For full details, see [here](pretrained_models).

## Model training

Other users may wish to exert full control over DeepRVAT scores, for example, to modify the model architecture, the set of annotations, or the set of training traits. For this, we provide pipelines for gene impairment module training, both in our CV and in a standard training/validation setup, with subsequent gene impairment score computation and association testing.

For full details, see [here](training_association).

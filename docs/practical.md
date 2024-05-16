# Practical recommendations for users


## Modes of usage

DeepRVAT can be applied in various modes, presented here in increasing levels of complexity. For each of these scenarios, we provide a corresponding Snakemake pipeline.

### Precomputed burden scores

_Note: Precomputed burden scores are not yet available. They will be made available upon publication of the DeepRVAT manuscript._ 

For users running association testing on UKBB WES data, we provide precomputed burden scores for all protein-coding genes with a qualifying variant within 300 bp of an exon. In this scenario, users are freed from processing of large WES data and may carry out highly computationally efficient association tests with the default DeepRVAT pipeline or the DeepRVAT+REGENIE integration.

Note that DeepRVAT scores are on a scale between 0 and 1, with a score closer to 0 indicating that the aggregate effect of variants in the gene is protective, and a score closer to 1 when the aggregate effect is deleterious.

### Pretrained models

Some users may wish to select variants or make variant-to-gene assigments differently from our methods, or to work on datasets other than UKBB. For this, we provide an ensemble of pretrained DeepRVAT gene impairment modules, which can be used for scoring individual-gene pairs for subsequent association testing. We also provide a pipeline for functional annotation of variants for compatibility with the pretrained modules.

### Model training

Other users may wish to exert full control over DeepRVAT scores, for example, to modify the model architecture, the set of annotations, or the set of training traits. For this, we provide pipelines for gene impairment module training, both in our CV and in a standard training/validation setup, with subsequent gene impairment score computation and association testing.


## Gene impairment module training

For users wishing to train a custom DeepRVAT model, we provide here some practical suggestions based on our experiences.

### Model architecture

We found no benefit to using architectures larger than that used in this work, though we conjecture that larger architectures may provide some benefit with larger training data and more annotations. We performed limited experimentation with the aggregation function used and found the maximum to give better results than the sum. However, exploring other choices or a learned aggregation remains open.

### Training traits and seed genes

We found that multiphenotype training improved performance, however, on our dataset, adding traits with fewer than three seed genes provided modest to no benefit. We also saw poor performance when including seed genes based on prior knowledge, e.g., known GWAS or RVAS associations, rather than the seed gene discovery methods. We hypothesize that this is because an informative seed gene must have driver rare variants in the training dataset itself, which may not be the case for associations known from other cohorts.

### Variant selection

While association testing was carried out on variants with MAF < 0.1%, we saw improved results when including a greater number of variants (we used MAF < 1%) for training.

### Variant annotations

We found that the best performance was achieved when including the full set of annotations, including correlated annotations. We thus recommend including annotations fairly liberally. However, we did find limits, for example, increasing the number of DeepSEA PCs from the 6 we used provided no benefit and eventually degraded model performance.

### Model ensembling

We found little to no benefit, but also no harm, from using more than 6 DeepRVAT gene impairment modules per CV fold in our ensemble. Therefore, we chose this number as the most computationally efficient to achieve optimal results.


# Practical recommendations for training

For users wishing to train a custom DeepRVAT model, we provide here some practical suggestions based on our experiences.

## Model architecture

We found no benefit to using architectures larger than that used in the [DeepRVAT pupblication](https://www.biorxiv.org/content/10.1101/2023.07.12.548506v2) which can be found here `deeprvat/pretrained_models/config.yaml`. Though we conjecture that larger architectures may provide some benefit with larger training data and more annotations. We performed limited experimentation with the aggregation function used and found the maximum to give better results than the sum. However, exploring other choices or a learned aggregation remains open.

## Training traits and seed genes

We found that multiphenotype training improved performance, however, on our dataset, adding traits with fewer than three seed genes provided modest to no benefit. We also saw poor performance when including seed genes based on prior knowledge, e.g., known GWAS or RVAS associations, rather than the seed gene discovery methods. We hypothesize that this is because an informative seed gene must have driver rare variants in the training dataset itself, which may not be the case for associations known from other cohorts.

## Variant selection

While association testing was carried out on variants with MAF < 0.1%, we saw improved results when including a greater number of variants (we used MAF < 1%) for training.

## Variant annotations

We found that the best performance was achieved when including the full set of annotations, including correlated annotations. We thus recommend including annotations fairly liberally. However, we did find limits, for example, increasing the number of DeepSEA PCs from the 6 we used provided no benefit and eventually degraded model performance.

## Model ensembling

We found little to no benefit, but also no harm, from using more than 6 DeepRVAT gene impairment modules per CV fold in our ensemble. Therefore, we chose this number as the most computationally efficient to achieve optimal results.


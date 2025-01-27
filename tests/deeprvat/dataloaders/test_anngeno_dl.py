from deeprvat.data.anngeno_dl import AnnGenoDataset
import hypothesis.strategies as st
from pathlib import Path
import tempfile
from typing import Any, Dict
from anngeno import AnnGeno
from anngeno.test_utils import anngeno_args_and_genotypes
from hypothesis import Phase, given, settings


# TODO: Implement
# Check that entries with 1 in AnnGenoDataset.variant_gene_mask correspond
# to variants in the gene in question
@settings(phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target])
def test_variant_gene_mask():
    # get_region for each training region
    pass


# TODO: Implement
# Check that entries with 1 in AnnGenoDataset.gene_phenotype_mask correspond to genes
# associated with the phenotype in question according to the training_regions argument
@settings(phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target])
def test_gene_covariatephenotype_mask():
    pass


# TODO: Implement
# Check output of __getitem__
# Sometimes use sample_set
# Sometimes use cache_regions
@given(
    anngeno_args_and_genotypes=anngeno_args_and_genotypes(
        min_phenotypes=1, min_annotations=1, region_set=True
    ),
    batch_proportion=st.floats(min_value=0, max_value=1, exclude_min=True),
)
@settings(phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target])
def test_getitem_training(anngeno_args_and_genotypes: Dict[str, Any]):
    anngeno_args = anngeno_args_and_genotypes["anngeno_args"]
    genotypes = anngeno_args_and_genotypes["genotypes"]

    variant_ids = anngeno_args["variant_metadata"]["id"]
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = Path(tmpdirname) / anngeno_args["filename"]
        anngeno_args["filename"] = filename
        ag = AnnGeno(**anngeno_args)

        ag.set_samples(
            slice(None),
            genotypes,
            variant_ids=variant_ids,
        )

        # Can only use ag.subset_samples in read-only mode
        del ag
        ag = AnnGeno(filename=anngeno_args["filename"])

        if sample_subset := anngeno_args_and_genotypes.get("sample_subset", None):
            ag.subset_samples(sample_subset)

        ag.subset_annotations(
            annotation_columns=anngeno_args_and_genotypes.get(
                "annotation_columns", None
            ),
            variant_set=anngeno_args_and_genotypes.get("variant_set", None),
        )

        # TODO: construct dataaset and iterate through it
        batch_size = math.ceil(batch_proportion * n_samples)
        agd = AnnGenoDataset(
            filename=filename,
            sample_batch_size=batch_size,
            mask_type="sum",  # TODO: Test max
            training_mode=True,
            training_regions=anngeno_args_and_genotypes["training_regions"],

        )

        # TODO: reconstruct each region using variant_gene_mask

        # TODO: compare to results from using AnnGeno.get_region(), AnnGeno.phenotypes, AnnGeno.annotations


# TODO: Implement
# Check output of __getitem__
# Sometimes use sample_set
# Sometimes use cache_regions
@given(
    anngeno_args_and_genotypes=anngeno_args_and_genotypes(),
    batch_proportion=st.floats(min_value=0, max_value=1, exclude_min=True),
)
@settings(phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target])
def test_getitem_testing():
    # use __getitem__
    # compare to results from using AnnGeno.get_region(), AnnGeno.phenotypes, AnnGeno.annotations
    pass

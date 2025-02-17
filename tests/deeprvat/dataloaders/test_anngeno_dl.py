from deeprvat.data.anngeno_dl import AnnGenoDataset
from torch.utils.data import DataLoader
import numpy as np
import math
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
# @settings(phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target])
# def test_variant_gene_mask():
#     # get_region for each training region
#     pass


# TODO: Implement
# Check that entries with 1 in AnnGenoDataset.gene_phenotype_mask correspond to genes
# associated with the phenotype in question according to the training_regions argument
# @settings(phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target])
# def test_gene_covariatephenotype_mask():
#     pass


# TODO: Check that regions are correct
#       Check that all samples are iterated over
# Check output of __getitem__
# Sometimes use sample_set
# Sometimes use cache_regions
@given(
    anngeno_args_and_genotypes=anngeno_args_and_genotypes(
        min_phenotypes=1, min_annotations=1, covariates=True, training_regions=True
    ),
    batch_proportion=st.floats(min_value=0, max_value=1, exclude_min=True),
)
@settings(
    deadline=4_000, phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target]
)
def test_getitem_training(
    anngeno_args_and_genotypes: Dict[str, Any], batch_proportion: float
):
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
        ag = AnnGeno(filename=filename)

        if sample_subset := anngeno_args_and_genotypes.get("sample_subset", None):
            ag.subset_samples(sample_subset)

        if (
            annotation_columns := anngeno_args_and_genotypes.get(
                "annotation_columns", None
            )
        ) is not None:
            ag.subset_annotations(
                annotation_columns,
            )

        if (
            variant_set := anngeno_args_and_genotypes.get("variant_set", None)
        ) is not None:
            ag.subset_variants(variant_set)

        # TODO: construct dataaset and iterate through it
        batch_size = math.ceil(batch_proportion * ag.sample_count)
        training_regions = anngeno_args_and_genotypes["training_regions"]
        covariates = anngeno_args_and_genotypes["covariates"]
        agd = AnnGenoDataset(
            filename=filename,
            sample_batch_size=batch_size,
            mask_type="sum",  # TODO: Test max
            training_mode=True,
            training_regions=training_regions,
            covariates=covariates,
            standardize_covariates=False,  # TODO: test this function
            quantile_transform_phenotypes=False,  # TODO: test this function
            annotation_columns=anngeno_args_and_genotypes.get(
                "annotation_columns", None
            ),
            variant_set=anngeno_args_and_genotypes.get("variant_set", None),
            sample_set=anngeno_args_and_genotypes.get("sample_subset", None),
        )

        dl = DataLoader(
            agd,
            batch_size=None,  # No automatic batching
            batch_sampler=None,  # No automatic batching
        )

        # # Check that covariates are used for all phenotypes
        # gene_covariatephenotype_mask = agd.gene_covariatephenotype_mask.numpy()
        # gene_phenotype_mask = agd.gene_phenotype_mask.numpy()
        # assert np.all(gene_covariatephenotype_mask[:, : len(covariates)] == 1)
        # assert np.array_equal(
        #     gene_covariatephenotype_mask[:, len(covariates) :], agd.gene_phenotype_mask
        # )

        # # Reconstruct training_regions using agd.gene_covariatephenotype_mask
        # agd_phenotypes = list(agd.training_regions.keys())
        # reconstructed_training_regions = {p: [] for p in agd_phenotypes}
        # for i in range(gene_phenotype_mask.shape[0]):
        #     for j in range(gene_phenotype_mask.shape[1]):
        #         if gene_phenotype_mask[i, j]:
        #             reconstructed_training_regions[agd_phenotypes[i]].append(
        #                 int(agd.regions[j])
        #             )
        # assert set(training_regions.keys()) == set(agd_phenotypes)
        # for p, v in training_regions.items():
        #     assert len(set(reconstructed_training_regions[p])) == len(
        #         reconstructed_training_regions[p]
        #     )
        #     assert set(reconstructed_training_regions[p]) == set(v)

        # reconstruct each region using variant_gene_mask
        for batch in dl:
            for i, region in enumerate(agd.regions):
                # compare to results from using AnnGeno.get_region(), AnnGeno.phenotypes, AnnGeno.annotations
                reference = ag.get_region(region, batch["sample_slice"])
                region_mask = (
                    batch["variant_gene_mask"][:, i].cpu().numpy().astype(np.bool)
                )

                assert batch["variant_gene_mask"].shape == (
                    batch["genotypes"].shape[1],
                    sum([len(v) for v in training_regions.values()]),
                )
                assert np.array_equal(
                    batch["genotypes"][:, region_mask],
                    reference["genotypes"].astype(np.float32),
                )
                assert np.allclose(
                    batch["annotations"][region_mask],
                    reference["annotations"].astype(np.float32),
                )
                assert np.allclose(
                    batch["covariates"],
                    ag.phenotypes[anngeno_args_and_genotypes["covariates"]]
                    .iloc[batch["sample_slice"]]
                    .to_numpy()
                    .astype(np.float32),
                    equal_nan=True,
                )
                assert np.allclose(
                    batch["phenotypes"],
                    ag.phenotypes[
                        list(anngeno_args_and_genotypes["training_regions"].keys())
                    ]
                    .iloc[batch["sample_slice"]]
                    .to_numpy()
                    .astype(np.float32),
                    equal_nan=True,
                )


# TODO: check that all samples and regions are iterated over
# Check output of __getitem__
# Sometimes use sample_set
# Sometimes use cache_regions - but not yet, this has a BUG
@given(
    anngeno_args_and_genotypes=anngeno_args_and_genotypes(min_annotations=1),
    batch_proportion=st.floats(min_value=0, max_value=1, exclude_min=True),
    # cache_genotypes=st.booleans(),
)
@settings(
    deadline=4_000, phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target]
)
def test_getitem_gis_computation(anngeno_args_and_genotypes, batch_proportion):
    # use __getitem__
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
        ag = AnnGeno(filename=filename)

        if sample_subset := anngeno_args_and_genotypes.get("sample_subset", None):
            ag.subset_samples(sample_subset)

        if (
            annotation_columns := anngeno_args_and_genotypes.get(
                "annotation_columns", None
            )
        ) is not None:
            ag.subset_annotations(
                annotation_columns,
            )

        if (
            variant_set := anngeno_args_and_genotypes.get("variant_set", None)
        ) is not None:
            ag.subset_variants(variant_set)

        # TODO: construct dataaset and iterate through it
        batch_size = math.ceil(batch_proportion * ag.sample_count)
        agd = AnnGenoDataset(
            filename=filename,
            sample_batch_size=batch_size,
            mask_type="sum",  # TODO: Test max
            quantile_transform_phenotypes=False,  # TODO: test this function
            annotation_columns=anngeno_args_and_genotypes.get(
                "annotation_columns", None
            ),
            variant_set=anngeno_args_and_genotypes.get("variant_set", None),
            sample_set=anngeno_args_and_genotypes.get("sample_subset", None),
        )

        # if cache_genotypes:
        #     agd.cache_regions(compress=True)

        dl = DataLoader(
            agd,
            batch_size=None,  # No automatic batching
            batch_sampler=None,  # No automatic batching
        )

        for batch in dl:
            # compare to results from using AnnGeno.get_region()
            reference = ag.get_region(batch["region"], batch["sample_slice"])

            assert np.array_equal(
                batch["genotypes"], reference["genotypes"].astype(np.float32)
            )
            assert np.allclose(
                batch["annotations"], reference["annotations"].astype(np.float32)
            )

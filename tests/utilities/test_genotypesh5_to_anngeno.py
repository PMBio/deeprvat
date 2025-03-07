import hypothesis.strategies as st
from anngeno import AnnGeno
import string
import pandas as pd
from pathlib import Path
import tempfile
import h5py
import numpy as np
from hypothesis import Phase, assume, given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import column, data_frames, indexes

from deeprvat.utilities.genotypesh5_to_anngeno import _convert_genotypes_h5
from anngeno.test_utils import (
    anngeno_args_and_genotypes,
    indexed_array_equal,
)


@given(
    anngeno_args_and_genotypes=anngeno_args_and_genotypes(),
    batch_size=st.integers(min_value=1, max_value=110),
)
@settings(
    deadline=2_000, phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target]
)
def test_genotypesh5_to_anngeno(anngeno_args_and_genotypes, batch_size):
    anngeno_args = anngeno_args_and_genotypes["anngeno_args"]
    genotype_matrix = anngeno_args_and_genotypes["genotypes"]
    samples = anngeno_args["samples"]
    variant_metadata = anngeno_args["variant_metadata"]

    variant_metadata["id"] = np.arange(len(variant_metadata))

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmppath = Path(tmpdirname)
        ### Create genotypes.h5
        # Compute sparse representation
        sparse_variant_matrix = np.full_like(genotype_matrix, -1, dtype=np.int32)
        sparse_genotype_matrix = np.full_like(genotype_matrix, -1, dtype=np.int8)
        for i in range(genotype_matrix.shape[0]):
            variant_list = []
            genotype_list = []
            for j in range(genotype_matrix.shape[1]):
                if (g_ij := genotype_matrix[i, j]) > 0:
                    variant_list.append(j)
                    genotype_list.append(g_ij)

            sparse_variant_matrix[i, : len(variant_list)] = variant_list
            sparse_genotype_matrix[i, : len(genotype_list)] = genotype_list

        # Write genotypes.h5 file
        with h5py.File(tmppath / "genotypes.h5", "w") as f:
            f.create_dataset(
                "samples", data=samples.astype(object), dtype=h5py.string_dtype()
            )
            f.create_dataset(
                "variant_matrix", data=sparse_variant_matrix, dtype=np.int32
            )
            f.create_dataset(
                "genotype_matrix", data=sparse_genotype_matrix, dtype=np.int8
            )

        # Write variants.parquet and phenotypes.parquet
        variant_metadata.to_parquet(tmppath / "variants.parquet")
        pd.DataFrame({"sample": samples}, index=samples).to_parquet(
            tmppath / "phenotypes.parquet"
        )

        ### Run _convert_genotypes_h5
        _convert_genotypes_h5(
            tmppath / "variants.parquet",
            tmppath / "phenotypes.parquet",
            tmppath / "genotypes.h5",
            tmppath / "genotypes.ag",
            batch_size,
        )

        ### Compare results when reading from genotypes.h5 versus genotypes.ag
        with h5py.File(tmppath / "genotypes.h5") as f:
            h5_variant_matrix = f["variant_matrix"][:]
            h5_genotype_matrix = f["genotype_matrix"][:]

        h5_genotypes = np.zeros((len(samples), len(variant_metadata)), dtype=np.uint8)
        for i in range(h5_variant_matrix.shape[0]):
            cols = h5_variant_matrix[i, h5_variant_matrix[i] != -1]
            vals = h5_genotype_matrix[i, h5_genotype_matrix[i] != -1]
            h5_genotypes[i, cols] = vals

        assert np.array_equal(h5_genotypes, genotype_matrix)

        assert (tmppath / "genotypes.ag").exists()
        assert (tmppath / "genotypes.ag").is_dir()
        ag = AnnGeno(tmppath / "genotypes.ag")

        assert indexed_array_equal(
            h5_genotypes.T,
            ag.make_variant_ids(variant_metadata),
            ag[:, :].T,
            ag.variant_metadata["id"].to_numpy(),
        )

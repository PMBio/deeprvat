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


# NOTE: This is somewhat arbritrary but a little more readable
BASIC_ALPHABET = string.ascii_letters + string.digits + "_-:.<>|"


def sort_by_index(x, index):
    index_argsort = np.argsort(index)
    return x[index_argsort]


def indexed_array_equal(
    x: np.ndarray, x_index: np.ndarray, y: np.ndarray, y_index: np.ndarray
) -> bool:
    assert len(x_index.shape) == 1
    assert len(y_index.shape) == 1
    assert len(np.unique(x_index)) == len(x_index)
    assert len(np.unique(y_index)) == len(y_index)

    return (
        x_index.shape[0] == x.shape[0]
        and y_index.shape[0] == y.shape[0]
        and x.shape == y.shape
        and set(x_index) == set(y_index)
        and (
            np.array_equal(sort_by_index(x, x_index), sort_by_index(y, y_index))
            if (x.dtype == np.dtype("O") or y.dtype == np.dtype("O"))
            else np.array_equal(
                sort_by_index(x, x_index), sort_by_index(y, y_index), equal_nan=True
            )
        )
    )


@st.composite
def genotypes(draw):
    n_samples = draw(st.integers(min_value=1, max_value=100))
    n_variants = draw(st.integers(min_value=1, max_value=1000))

    samples = draw(
        arrays(
            dtype=object,
            shape=n_samples,
            elements=st.text(alphabet=BASIC_ALPHABET, min_size=1),
            unique=True,
        )
    )

    variant_metadata = draw(
        data_frames(
            columns=[
                column(
                    name="chrom",
                    elements=st.one_of(
                        st.just("chr1"),
                        st.just("chr2"),
                        st.just("4"),
                        st.just("5"),
                        st.just("chrX"),
                    ),
                ),
                column(name="pos", elements=st.integers(min_value=0, max_value=1e9)),
                column(
                    name="ref",
                    elements=st.text(alphabet=("A", "C", "G", "T"), min_size=1),
                ),
                column(
                    name="alt",
                    elements=st.text(alphabet=("A", "C", "G", "T"), min_size=1),
                ),
                # variant_ids,
            ],
            index=indexes(
                elements=st.integers(min_value=0, max_value=2000),
                min_size=n_variants,
                max_size=n_variants,
            ),
        )
    )

    genotype_matrix = draw(
        arrays(
            np.uint8,
            (n_samples, n_variants),
            elements=st.integers(min_value=0, max_value=2),
        )
    )

    return samples, variant_metadata, genotype_matrix


@given(genotypes=genotypes(), batch_size=st.integers(min_value=1, max_value=110))
@settings(
    deadline=2_000, phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target]
)
def test_genotypesh5_to_anngeno(genotypes, batch_size):
    samples, variant_metadata, genotype_matrix = genotypes
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
            f.create_dataset("samples", data=samples, dtype=h5py.string_dtype())
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
            variant_metadata["id"].to_numpy(),
            ag[:, :].T,
            ag.variant_metadata["id"].to_numpy(),
        )

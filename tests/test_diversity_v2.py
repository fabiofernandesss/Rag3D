import math
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from rag3d.backend import DEFAULT_RETRIEVAL_LIMITS
import rag3d.diversity as diversity_module
from rag3d.diversity import diversify, shifted_cosine_kernel


class BombVectors(dict):
    def get(self, *_args, **_kwargs):
        raise AssertionError("none must not inspect vectors")

    def values(self):
        raise AssertionError("none must not inspect vectors")


class OversizedExplodingItems(Sequence):
    def __len__(self):
        return DEFAULT_RETRIEVAL_LIMITS.max_pool + 1

    def __getitem__(self, index):
        raise AssertionError("oversized items must fail before indexing")

    def __iter__(self) -> Iterator:
        raise AssertionError("oversized items must fail before iteration")


class LyingItems(Sequence):
    def __init__(self):
        self.yielded = 0

    def __len__(self):
        return 0

    def __getitem__(self, index):
        if 0 <= index <= DEFAULT_RETRIEVAL_LIMITS.max_pool:
            self.yielded += 1
            return (index, 1.0)
        raise IndexError

    def __iter__(self) -> Iterator:
        for index in range(DEFAULT_RETRIEVAL_LIMITS.max_pool + 1):
            self.yielded += 1
            yield (index, 1.0)


def test_none_is_an_exact_order_identity_and_does_not_read_vectors():
    items = [(9, 0.1), (2, 99.0), (7, -4.0)]

    assert diversify(items, BombVectors(), 2, method="none") == [9, 2]
    assert diversify(items, BombVectors(), 0, method="none") == []
    assert diversify([], BombVectors(), 10, method="none") == []


def test_diversity_rejects_honest_oversized_items_without_iteration():
    with pytest.raises(ValueError, match="items.*maximum"):
        diversify(OversizedExplodingItems(), BombVectors(), 1, method="none")


def test_diversity_rejects_a_lying_container_at_the_first_excess_item():
    items = LyingItems()

    with pytest.raises(ValueError, match="items.*maximum"):
        diversify(items, BombVectors(), 1, method="none")

    assert items.yielded == DEFAULT_RETRIEVAL_LIMITS.max_pool + 1


def test_mmr_uses_relevance_minus_maximum_cosine_similarity():
    items = [(1, 1.0), (2, 0.9), (3, 0.8)]
    vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([0.0, 1.0]),
    }

    assert diversify(items, vectors, 2, method="mmr", mmr_lambda=0.5) == [1, 3]
    assert diversify(items, vectors, 3, method="mmr", mmr_lambda=1.0) == [1, 2, 3]


def test_mmr_maintains_incremental_max_similarity_instead_of_rescanning_selected(
    monkeypatch,
):
    size = 18
    top_k = 7
    items = [(index, float(size - index)) for index in range(size)]
    vectors = {
        index: np.array([1.0, float(index), float(index * index)], dtype=np.float64)
        for index in range(size)
    }
    original_dot = diversity_module.np.dot
    dot_calls = 0

    def counted_dot(left, right):
        nonlocal dot_calls
        dot_calls += 1
        return original_dot(left, right)

    monkeypatch.setattr(diversity_module.np, "dot", counted_dot)

    selected = diversify(items, vectors, top_k, method="mmr")

    assert len(selected) == top_k
    assert dot_calls <= top_k


def test_greedy_dpp_penalizes_exact_duplicates_with_a_psd_shifted_cosine_kernel():
    items = [(1, 1.0), (2, 0.99), (3, 0.9)]
    vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([0.0, 1.0]),
    }

    selected = diversify(items, vectors, 2, method="dpp", dpp_alpha=1.0)

    assert selected == [1, 3]
    assert len(selected) == len(set(selected))


def test_shifted_cosine_kernel_is_psd_for_antiparallel_and_duplicate_vectors():
    vectors = np.array(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )

    kernel = shifted_cosine_kernel(vectors, jitter=1e-9)

    assert kernel.dtype == np.float64
    assert np.allclose(kernel, kernel.T)
    assert np.linalg.eigvalsh(kernel).min() >= -1e-12


def test_diversity_rejects_vectors_above_shared_dimension_and_product_bounds():
    limits = DEFAULT_RETRIEVAL_LIMITS
    with pytest.raises(ValueError, match="dimension"):
        diversify(
            [(1, 1.0)],
            {1: np.ones(limits.max_dense_dim + 1, dtype=np.float32)},
            1,
            method="mmr",
        )

    rows = limits.max_diversity_values // limits.max_dense_dim + 1
    items = [(index, 1.0) for index in range(rows)]
    vectors = {
        index: np.ones(limits.max_dense_dim, dtype=np.float32)
        for index in range(rows)
    }
    with pytest.raises(ValueError, match="diversity.*values"):
        diversify(items, vectors, 1, method="dpp")


def test_public_shifted_cosine_kernel_rejects_unbounded_shapes():
    limits = DEFAULT_RETRIEVAL_LIMITS
    with pytest.raises(ValueError, match="rows"):
        shifted_cosine_kernel(np.ones((limits.max_pool + 1, 1)))
    with pytest.raises(ValueError, match="dimension"):
        shifted_cosine_kernel(np.ones((1, limits.max_dense_dim + 1)))


def test_shifted_cosine_kernel_remains_finite_for_extreme_finite_vectors():
    vectors = np.array(
        [[1e308, 1e308], [1e308, -1e308]], dtype=np.float64
    )

    kernel = shifted_cosine_kernel(vectors, jitter=1e-9)

    assert np.isfinite(kernel).all()
    assert kernel == pytest.approx(
        np.array([[1.0 + 1e-9, 0.5], [0.5, 1.0 + 1e-9]])
    )


def test_incremental_dpp_matches_naive_greedy_log_determinant():
    vectors = np.array(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-0.6, 0.8]],
        dtype=np.float64,
    )
    quality = np.array([1.0, 0.9, 0.75, 0.6], dtype=np.float64)
    similarity = shifted_cosine_kernel(vectors, jitter=1e-9)
    kernel = quality[:, None] * similarity * quality[None, :]
    chosen = []
    remaining = list(range(len(vectors)))
    for _ in range(3):
        scored = []
        for candidate in remaining:
            subset = chosen + [candidate]
            sign, logdet = np.linalg.slogdet(kernel[np.ix_(subset, subset)])
            scored.append((-(logdet if sign > 0 else -math.inf), candidate))
        _, best = min(scored)
        chosen.append(best)
        remaining.remove(best)

    selected = diversify(
        [(index + 1, float(quality[index])) for index in range(len(quality))],
        {index + 1: vector for index, vector in enumerate(vectors)},
        3,
        method="dpp",
    )

    assert selected == [index + 1 for index in chosen]


def test_dpp_selector_uses_on_demand_similarity_without_a_full_n_by_n_kernel(
    monkeypatch,
):
    size = 24
    items = [(index, float(size - index)) for index in range(size)]
    vectors = {
        index: np.array([1.0, float(index), float(index % 3)], dtype=np.float64)
        for index in range(size)
    }

    def forbidden_full_kernel(*_args, **_kwargs):
        raise AssertionError("DPP selection must not materialize an N x N kernel")

    monkeypatch.setattr(
        diversity_module, "shifted_cosine_kernel", forbidden_full_kernel
    )

    selected = diversify(items, vectors, 6, method="dpp")

    assert len(selected) == 6
    assert len(selected) == len(set(selected))


@pytest.mark.parametrize(
    "vectors",
    [
        {1: np.array([1.0, 0.0])},
        {1: np.array([0.0, 0.0]), 2: np.array([0.0, 1.0])},
        {1: np.array([math.nan, 0.0]), 2: np.array([0.0, 1.0])},
        {1: np.array([1.0]), 2: np.array([0.0, 1.0])},
    ],
)
@pytest.mark.parametrize("method", ["mmr", "dpp"])
def test_invalid_or_missing_vectors_fall_back_to_the_input_ranking(vectors, method):
    items = [(1, 1.0), (2, 0.5)]

    assert diversify(items, vectors, 2, method=method) == [1, 2]


def test_duplicate_ids_and_nonfinite_relevance_never_create_nan_or_duplicate_output():
    items = [(1, math.nan), (1, 2.0), (2, 1.0)]
    vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}

    assert diversify(items, vectors, 3, method="dpp") == [1, 2]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "unknown"}, "method"),
        ({"method": "mmr", "mmr_lambda": -0.1}, "mmr_lambda"),
        ({"method": "mmr", "mmr_lambda": 1.1}, "mmr_lambda"),
        ({"method": "dpp", "dpp_alpha": -1.0}, "dpp_alpha"),
        ({"method": "dpp", "jitter": -1.0}, "jitter"),
    ],
)
def test_diversity_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        diversify([], {}, 1, **kwargs)


def test_diversity_rejects_negative_or_excessive_top_k():
    with pytest.raises(ValueError, match="top_k"):
        diversify([], {}, -1)
    with pytest.raises(ValueError, match="top_k"):
        diversify([], {}, DEFAULT_RETRIEVAL_LIMITS.max_pool + 1)


try:
    from hypothesis import given, strategies as st
except ImportError:  # optional test dependency
    given = None


if given is not None:

    @given(
        st.lists(
            st.tuples(st.integers(min_value=0), st.floats(allow_nan=True)),
            max_size=50,
        ),
        st.integers(min_value=0, max_value=50),
    )
    def test_none_property_preserves_first_occurrences_and_bound(items, top_k):
        expected = []
        for cid, _ in items:
            if cid not in expected:
                expected.append(cid)

        assert diversify(items, BombVectors(), top_k, method="none") == expected[:top_k]

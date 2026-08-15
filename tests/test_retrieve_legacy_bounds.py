from __future__ import annotations

import numpy as np

from rag3d.backend import DEFAULT_RETRIEVAL_LIMITS
from rag3d.config import TriRagConfig
from rag3d.encoders import TriVec
from rag3d.retrieve import TriRetriever


class _Encoder:
    def __init__(self):
        self.batches = []

    def encode(self, texts, is_query=False):
        self.batches.append(list(texts))
        return [
            TriVec(
                dense=np.array([1.0, 0.0], dtype=np.float32),
                sparse={1: 1.0},
                tokens=np.array([[1.0, 0.0]], dtype=np.float32),
            )
            for _ in texts
        ]


class _DisjointStore:
    def __init__(self):
        limit = DEFAULT_RETRIEVAL_LIMITS.max_channel_k
        self.dense = [(index, 1.0) for index in range(limit)]
        self.sparse = [(limit + index, 1.0) for index in range(limit)]
        self.structural_candidates = None

    def dense_search(self, _vector, k):
        return self.dense[:k]

    def sparse_search(self, _weights, k):
        return self.sparse[:k]

    def colbert_scores(self, _tokens, candidates):
        self.structural_candidates = list(candidates)
        return []

    def get_chunks(self, _ids):
        return []


def test_legacy_structural_union_is_deterministic_and_bounded() -> None:
    store = _DisjointStore()
    cfg = TriRagConfig(
        encoder="hash",
        top_k=1,
        channel_k=DEFAULT_RETRIEVAL_LIMITS.max_channel_k,
        fusion="rrf",
        diversity=0.0,
        stitch_radius=0,
    )

    TriRetriever(store, _Encoder(), cfg).search("consulta")

    expected = []
    half = DEFAULT_RETRIEVAL_LIMITS.max_pool // 2
    for index in range(half):
        expected.extend((index, DEFAULT_RETRIEVAL_LIMITS.max_channel_k + index))
    assert store.structural_candidates == expected
    assert len(store.structural_candidates) == DEFAULT_RETRIEVAL_LIMITS.max_pool


def test_legacy_oversized_llm_expansion_falls_back_before_split() -> None:
    class OversizedExpansionLLM:
        def available(self):
            return True

        def complete(self, *_args, **_kwargs):
            return "x" * (DEFAULT_RETRIEVAL_LIMITS.max_query_bytes + 1)

    store = _DisjointStore()
    encoder = _Encoder()
    cfg = TriRagConfig(
        encoder="hash",
        top_k=1,
        channel_k=1,
        fusion="rrf",
        diversity=0.0,
        stitch_radius=0,
        expand_query=True,
    )

    TriRetriever(store, encoder, cfg, llm=OversizedExpansionLLM()).search("consulta")

    assert encoder.batches == [["consulta"]]

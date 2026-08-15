import math

import numpy as np
import pytest

from rag3d.backend import (
    BackendCapabilities,
    DEFAULT_RETRIEVAL_LIMITS,
    SearchFilters,
    SearchScope,
)
from rag3d.config import TriRagConfig
from rag3d.encoders import TriVec
from rag3d.memory import ChatMemory
from rag3d.rerank import CrossEncoderReranker, LLMListwiseReranker
from rag3d.retrieval_v2 import RetrievalV2


class FakeEncoder:
    name = "fake_encoder"

    def __init__(self, events):
        self.events = events
        self.texts = []

    def encode(self, texts, is_query=False):
        self.events.append("encode")
        self.texts = list(texts)
        return [
            TriVec(
                dense=np.array([1.0, 0.0], dtype=np.float32),
                sparse={index + 1: 1.0},
                tokens=np.array([[1.0, 0.0]], dtype=np.float32),
            )
            for index, _ in enumerate(texts)
        ]


class FakeBackend:
    backend_name = "fake-backend"

    def __init__(self, events, dense=None, sparse=None, structural=None, rows=None, enabled=True):
        self.events = events
        self._dense = list(dense or [])
        self._sparse = list(sparse or [])
        self._structural = list(structural or [])
        self.structural_ids = []
        self.filters_seen = []
        self.get_chunk_requests = []
        self.capabilities = BackendCapabilities(
            exact_dense_search=True,
            sparse_search=True,
            structural_rerank=enabled,
        )
        ids = {cid for cid, _ in self._dense + self._sparse + self._structural}
        self.rows = rows or {cid: make_row(cid) for cid in ids}

    def dense_search(self, _vector, k, *, filters=None, exact=None):
        self.events.append("dense")
        self.filters_seen.append(filters)
        return self._dense[:k]

    def sparse_search(self, _weights, k, *, filters=None):
        self.events.append("sparse")
        self.filters_seen.append(filters)
        return self._sparse[:k]

    def structural_rerank(self, _vectors, candidate_ids, k, *, filters=None):
        self.events.append("structural")
        self.structural_ids = list(candidate_ids)
        self.filters_seen.append(filters)
        allowed = set(candidate_ids)
        return [(cid, score) for cid, score in self._structural if cid in allowed][:k]

    def dense_vectors(self, ids):
        self.events.append("dense_vectors")
        return {cid: np.array([float(cid == ids[0]), float(cid != ids[0])]) for cid in ids}

    def get_chunks(self, ids):
        self.events.append("get_chunks")
        self.get_chunk_requests.append(tuple(ids))
        return [dict(self.rows[cid]) for cid in ids if cid in self.rows]

    def neighbors(self, doc_id, positions):
        self.events.append("neighbors")
        return [
            dict(row)
            for row in self.rows.values()
            if row.get("doc_id") == doc_id
            and row.get("kind") == "chunk"
            and row.get("pos") in positions
        ]


class FakeReranker:
    name = "fake_reranker"

    def __init__(self, events, reverse=True):
        self.events = events
        self.reverse = reverse

    def available(self):
        return True

    def rerank(self, _query, hits, top_k=None):
        self.events.append("rerank")
        result = [dict(hit) for hit in hits]
        if self.reverse:
            result.reverse()
        return result[:top_k] if top_k is not None else result


class FakeLLM:
    def available(self):
        return True

    def complete(self, *_args, **_kwargs):
        return "  variante   um  \nvariante dois"


class StrictBoundedRanking:
    def __init__(self, allowed_reads, start=1):
        self.allowed_reads = allowed_reads
        self.start = start

    def __iter__(self):
        for offset in range(self.allowed_reads):
            yield self.start + offset, float(self.allowed_reads - offset)


def make_row(cid, *, parent_id=None, doc_id=1, pos=0, text=None, kind="chunk"):
    return {
        "id": cid,
        "kind": kind,
        "doc_id": doc_id,
        "pos": pos,
        "parent_id": parent_id,
        "text": text or f"texto {cid}",
        "n_tokens": 2,
        "turn_no": None,
        "accessed_turn": None,
        "created": "2026-01-01T00:00:00Z",
        "importance": 0.5,
    }


def make_cfg(**changes):
    values = {
        "retrieval_pipeline": "v2",
        "fusion": "rrf",
        "dense_dim": 2,
        "colbert_dim": 2,
        "max_colbert_tokens": 8,
        "top_k": 2,
        "channel_k": 5,
        "structural_rerank": True,
        "reranker": "none",
        "diversity_method": "none",
        "rerank_pool": 4,
        "diversity_pool": 4,
    }
    values.update(changes)
    return TriRagConfig(**values)


def test_v2_stage_order_and_structural_scope_are_explicit(monkeypatch):
    events = []
    backend = FakeBackend(
        events,
        dense=[(1, 1.0), (2, 0.8)],
        sparse=[(3, 1.0), (2, 0.7)],
        structural=[(3, 999999.0), (1, -999999.0)],
    )
    reranker = FakeReranker(events)
    cfg = make_cfg(reranker="llm", diversity_method="mmr")

    def spy_diversify(items, vectors, top_k, **kwargs):
        events.append("diversity")
        assert list(vectors) == [cid for cid, _ in items]
        return [cid for cid, _ in items[:top_k]]

    monkeypatch.setattr("rag3d.retrieval_v2.diversify", spy_diversify)

    result = RetrievalV2(backend, FakeEncoder(events), cfg, reranker=reranker).search(
        "consulta"
    )

    assert events.index("encode") < events.index("dense") < events.index("sparse")
    assert events.index("sparse") < events.index("structural") < events.index("rerank")
    assert events.index("rerank") < events.index("dense_vectors") < events.index("diversity")
    assert set(backend.structural_ids) <= {1, 2, 3}
    assert len(backend.structural_ids) <= DEFAULT_RETRIEVAL_LIMITS.max_pool
    assert set(hit["id"] for hit in result.fused) <= {1, 2, 3}


def test_rrf_default_is_equal_weight_dense_plus_sparse_only():
    events = []
    backend = FakeBackend(
        events,
        dense=[(1, 1000.0), (2, 1.0)],
        sparse=[(2, 999999.0)],
        enabled=False,
    )
    cfg = make_cfg(
        structural_rerank=False,
        channel_weights=(100.0, 0.0001, 999.0),
        coherence_strength=1.0,
    )

    result = RetrievalV2(backend, FakeEncoder(events), cfg).search("consulta")

    assert [hit["id"] for hit in result.fused] == [2, 1]
    by_id = {hit["id"]: hit for hit in result.fused}
    assert by_id[2]["fusion_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert by_id[1]["fusion_score"] == pytest.approx(1 / 61)
    assert by_id[1]["channels"] == ["semantico"]
    assert by_id[2]["channels"] == ["semantico", "lexico"]


def test_quantum_v2_keeps_fusion_fields_and_still_uses_only_two_generators():
    events = []
    backend = FakeBackend(
        events,
        dense=[(1, 1.0), (2, 0.2)],
        sparse=[(1, 2.0), (2, 0.1)],
        structural=[(2, 5.0)],
    )

    result = RetrievalV2(backend, FakeEncoder(events), make_cfg(fusion="quantum")).search(
        "consulta"
    )

    assert all("classical" in hit for hit in result.fused)
    assert all("interference" in hit for hit in result.fused)
    assert all("per_channel" in hit for hit in result.fused)
    assert set(channel for hit in result.fused for channel in hit["channels"]) <= {
        "semantico",
        "lexico",
    }


def test_structural_blend_depends_on_rank_not_raw_maxsim_score():
    def run(scores):
        events = []
        backend = FakeBackend(
            events,
            dense=[(1, 1.0), (2, 0.8), (3, 0.7)],
            sparse=[],
            structural=list(zip([3, 2], scores)),
        )
        result = RetrievalV2(backend, FakeEncoder(events), make_cfg(top_k=3)).search("q")
        return result, backend

    low, low_backend = run([-1e30, 1e30])
    high, high_backend = run([1e300, -1e300])

    assert [hit["id"] for hit in low.fused] == [3, 2, 1]
    assert [hit["id"] for hit in high.fused] == [3, 2, 1]
    assert [hit["blend_score"] for hit in low.fused[:2]] == pytest.approx(
        [hit["blend_score"] for hit in high.fused[:2]]
    )
    assert low_backend.structural_ids == high_backend.structural_ids == [1, 2, 3]
    assert low.fused[-1]["id"] == 1


def test_zero_structural_rank_weight_is_an_exact_fusion_order_rollback():
    events = []
    backend = FakeBackend(
        events,
        dense=[(1, 1.0), (2, 0.8), (3, 0.7)],
        structural=[(3, 100.0), (2, 90.0)],
    )

    result = RetrievalV2(
        backend,
        FakeEncoder(events),
        make_cfg(top_k=3),
        structural_rank_weight=0.0,
    ).search("consulta")

    assert [hit["id"] for hit in result.fused] == [1, 2, 3]


def test_structural_is_skipped_when_backend_capability_is_false():
    events = []
    backend = FakeBackend(events, dense=[(1, 1.0)], structural=[(1, 2.0)], enabled=False)

    result = RetrievalV2(backend, FakeEncoder(events), make_cfg()).search("consulta")

    assert "structural" not in events
    assert result.views["estrutural"] == []


def test_structural_failure_preserves_the_fusion_order():
    events = []

    class FailingStructuralBackend(FakeBackend):
        def structural_rerank(self, *_args, **_kwargs):
            self.events.append("structural")
            raise RuntimeError("structural unavailable")

    backend = FailingStructuralBackend(
        events,
        dense=[(1, 1.0), (2, 0.5)],
        sparse=[(2, 2.0)],
    )

    result = RetrievalV2(backend, FakeEncoder(events), make_cfg()).search("consulta")

    assert [hit["id"] for hit in result.fused] == [2, 1]
    assert result.views["estrutural"] == []
    assert result.stats["structural_status"] == "fallback"
    assert result.stats["structural_reason"] == "stage-error"
    assert result.stats["structural_candidates_attempted"] == 2
    assert result.stats["structural_candidates_evaluated"] == 0
    assert "unavailable" not in repr(result.stats)


@pytest.mark.parametrize(
    "invalid_rows",
    [[(999, 1.0)], [(1, 1.0), (1, 0.5)]],
)
def test_structural_unknown_or_duplicate_ids_fail_closed(invalid_rows):
    class InvalidStructuralBackend(FakeBackend):
        def structural_rerank(self, *_args, **_kwargs):
            return invalid_rows

    backend = InvalidStructuralBackend(
        [], dense=[(1, 1.0), (2, 0.5)], enabled=True
    )

    result = RetrievalV2(backend, FakeEncoder([]), make_cfg()).search("consulta")

    assert [hit["id"] for hit in result.fused] == [1, 2]
    assert result.stats["structural_status"] == "fallback"
    assert result.stats["structural_reason"] == "invalid-output"
    assert result.stats["structural_candidates_evaluated"] == 0


def test_reranker_exception_preserves_the_structural_order():
    events = []

    class FailingReranker(FakeReranker):
        def rerank(self, *_args, **_kwargs):
            self.events.append("rerank")
            raise RuntimeError("reranker unavailable")

    backend = FakeBackend(
        events,
        dense=[(1, 1.0), (2, 0.5)],
        sparse=[(2, 2.0)],
        structural=[(1, 9.0), (2, 8.0)],
    )
    cfg = make_cfg(reranker="llm")

    expected = RetrievalV2(
        FakeBackend([], dense=backend._dense, sparse=backend._sparse, structural=backend._structural),
        FakeEncoder([]),
        make_cfg(),
    ).search("consulta")
    actual = RetrievalV2(
        backend,
        FakeEncoder(events),
        cfg,
        reranker=FailingReranker(events),
    ).search("consulta")

    assert [hit["id"] for hit in actual.fused] == [hit["id"] for hit in expected.fused]
    assert actual.stats["reranker_status"] == "fallback"
    assert actual.stats["reranker_reason"] == "stage-error"
    assert "unavailable" not in repr(actual.stats)


@pytest.mark.parametrize(
    "invalid_rows",
    [[{"id": 999}], [{"id": 1}, {"id": 1}]],
)
def test_reranker_unknown_or_duplicate_ids_fail_closed(invalid_rows):
    class InvalidReranker(FakeReranker):
        def rerank(self, *_args, **_kwargs):
            return invalid_rows

    backend = FakeBackend(
        [], dense=[(1, 1.0), (2, 0.5)], enabled=False
    )
    result = RetrievalV2(
        backend,
        FakeEncoder([]),
        make_cfg(structural_rerank=False, reranker="llm"),
        reranker=InvalidReranker([]),
    ).search("consulta")

    assert [hit["id"] for hit in result.fused] == [1, 2]
    assert result.stats["reranker_status"] == "fallback"
    assert result.stats["reranker_reason"] == "invalid-output"


def test_reranker_reconciliation_stops_at_expected_cardinality_plus_one():
    class UnboundedRows:
        def __init__(self):
            self.consumed = 0

        def __iter__(self):
            for _ in range(100_000):
                self.consumed += 1
                yield {"id": 1}

    rows = UnboundedRows()

    assert RetrievalV2._reconcile_reranker([{"id": 1}], rows) is None
    assert rows.consumed == 2


def test_prefetch_rejects_sized_oversize_without_iteration():
    class HonestOversizeRows:
        def __len__(self):
            return 2

        def __iter__(self):
            raise AssertionError("oversized rows must be rejected before iteration")

    class OversizeBackend(FakeBackend):
        def get_chunks(self, ids):
            return HonestOversizeRows()

    retriever = RetrievalV2(
        OversizeBackend([], dense=[(1, 1.0)], enabled=False),
        FakeEncoder([]),
        make_cfg(structural_rerank=False),
    )

    with pytest.raises(ValueError, match="chunk lookup"):
        retriever._prefetch([1])


def test_prefetch_stops_lying_backend_at_expected_plus_one():
    class LyingRows:
        def __init__(self):
            self.consumed = 0

        def __len__(self):
            return 0

        def __iter__(self):
            for _ in range(100_000):
                self.consumed += 1
                yield make_row(1)

    rows = LyingRows()

    class LyingBackend(FakeBackend):
        def get_chunks(self, ids):
            return rows

    retriever = RetrievalV2(
        LyingBackend([], dense=[(1, 1.0)], enabled=False),
        FakeEncoder([]),
        make_cfg(structural_rerank=False),
    )

    with pytest.raises(ValueError, match="chunk lookup"):
        retriever._prefetch([1])

    assert rows.consumed == 2


def test_prefetch_rejects_unexpected_chunk_without_overwriting_cache():
    cached = {1: make_row(1, text="legitimate")}

    class UnexpectedBackend(FakeBackend):
        def get_chunks(self, ids):
            assert ids == [2]
            return [make_row(1, text="wrong-scope")]

    retriever = RetrievalV2(
        UnexpectedBackend([], enabled=False),
        FakeEncoder([]),
        make_cfg(structural_rerank=False),
    )

    with pytest.raises(ValueError, match="chunk lookup"):
        retriever._prefetch([1, 2], cached)
    assert cached[1]["text"] == "legitimate"


def test_prefetch_rejects_duplicate_chunk_and_unexpected_parent_ids():
    class DuplicateBackend(FakeBackend):
        def get_chunks(self, ids):
            return [make_row(2), make_row(2)]

    duplicate = RetrievalV2(
        DuplicateBackend([], enabled=False),
        FakeEncoder([]),
        make_cfg(structural_rerank=False),
    )
    with pytest.raises(ValueError, match="chunk lookup"):
        duplicate._prefetch([2, 3])

    class UnexpectedParentBackend(FakeBackend):
        def get_chunks(self, ids):
            if ids == [1]:
                return [make_row(1, doc_id=7, parent_id=10)]
            assert ids == [10]
            return [make_row(11, doc_id=7, kind="parent")]

    unexpected_parent = RetrievalV2(
        UnexpectedParentBackend([], enabled=False),
        FakeEncoder([]),
        make_cfg(structural_rerank=False),
    )
    with pytest.raises(ValueError, match="parent lookup"):
        unexpected_parent._prefetch([1])


def test_stitch_stops_lying_neighbor_backend_at_requested_plus_one():
    class LyingNeighbors:
        def __init__(self):
            self.consumed = 0

        def __len__(self):
            return 0

        def __iter__(self):
            for position in range(100_000):
                self.consumed += 1
                yield make_row(position + 1, doc_id=7, pos=position)

    rows = LyingNeighbors()

    class LyingBackend(FakeBackend):
        def neighbors(self, doc_id, positions):
            return rows

    retriever = RetrievalV2(
        LyingBackend([], enabled=False),
        FakeEncoder([]),
        make_cfg(structural_rerank=False, stitch_radius=1),
    )
    hits = [
        {
            "id": 1,
            "doc_id": 7,
            "pos": 1,
            "kind": "chunk",
            "wide": "original",
        }
    ]

    assert retriever._stitch(hits) == hits
    assert hits[0]["wide"] == "original"
    assert rows.consumed == DEFAULT_RETRIEVAL_LIMITS.max_pool + 1


def test_query_encoder_output_stops_at_expected_cardinality_plus_one():
    class UnboundedEncoder(FakeEncoder):
        def __init__(self):
            super().__init__([])
            self.consumed = 0

        def encode(self, texts, is_query=False):
            vector = TriVec(
                dense=np.array([1.0, 0.0], dtype=np.float32),
                sparse={1: 1.0},
                tokens=np.array([[1.0, 0.0]], dtype=np.float32),
            )
            for _ in range(100_000):
                self.consumed += 1
                yield vector

    encoder = UnboundedEncoder()
    retriever = RetrievalV2(FakeBackend([], enabled=False), encoder, make_cfg())

    with pytest.raises(ValueError, match="exactly one vector"):
        retriever._encode_query(["consulta"])

    assert encoder.consumed == 2


@pytest.mark.parametrize("invalid_part", ["dense", "sparse", "tokens"])
def test_query_encoder_rejects_nonfinite_or_invalid_vector_payloads(
    invalid_part: str,
):
    class InvalidEncoder(FakeEncoder):
        def encode(self, texts, is_query=False):
            dense = np.array([1.0, 0.0], dtype=np.float32)
            sparse = {1: 1.0}
            tokens = np.array([[1.0, 0.0]], dtype=np.float32)
            if invalid_part == "dense":
                dense[0] = np.nan
            elif invalid_part == "sparse":
                sparse = {True: True}
            else:
                tokens[0, 0] = np.nan
            return [TriVec(dense=dense, sparse=sparse, tokens=tokens)]

    retriever = RetrievalV2(
        FakeBackend([], dense=[(1, 1.0)], enabled=False),
        InvalidEncoder([]),
        make_cfg(structural_rerank=False, dense_dim=2, colbert_dim=2),
    )

    with pytest.raises((TypeError, ValueError)):
        retriever.search("consulta")


def test_listwise_pipeline_reranker_receives_real_bounded_snippets():
    events = []

    class CapturingLLM:
        def __init__(self):
            self.prompt = ""

        def available(self):
            return True

        def complete(self, _system, messages, **_kwargs):
            self.prompt = messages[0]["content"]
            return "[1, 0]"

    llm = CapturingLLM()
    backend = FakeBackend(
        events,
        dense=[(1, 1.0), (2, 0.9), (3, 0.8)],
        enabled=False,
    )
    cfg = make_cfg(
        top_k=2,
        structural_rerank=False,
        reranker="llm",
        rerank_pool=2,
    )

    result = RetrievalV2(
        backend,
        FakeEncoder(events),
        cfg,
        reranker=LLMListwiseReranker(llm),
    ).search("consulta")

    assert [hit["id"] for hit in result.fused] == [2, 1]
    assert "texto 1" in llm.prompt and "texto 2" in llm.prompt
    assert "[] " not in llm.prompt
    assert len(backend.get_chunk_requests[0]) <= cfg.rerank_pool
    assert len(backend.get_chunk_requests[0]) <= DEFAULT_RETRIEVAL_LIMITS.max_pool


def test_cross_encoder_pipeline_reranker_receives_real_snippet_pairs():
    events = []

    class CapturingModel:
        def __init__(self):
            self.pairs = []

        def predict(self, pairs):
            self.pairs = list(pairs)
            return [0.1, 0.9]

    model = CapturingModel()
    backend = FakeBackend(
        events,
        dense=[(1, 1.0), (2, 0.9)],
        enabled=False,
    )
    cfg = make_cfg(structural_rerank=False, reranker="cross-encoder", rerank_pool=2)

    result = RetrievalV2(
        backend,
        FakeEncoder(events),
        cfg,
        reranker=CrossEncoderReranker(model=model),
    ).search("consulta")

    assert [hit["id"] for hit in result.fused] == [2, 1]
    assert model.pairs == [("consulta", "texto 1"), ("consulta", "texto 2")]
    assert len(backend.get_chunk_requests[0]) == 2


def test_reranker_snippet_lookup_failure_preserves_order_and_search_continues():
    events = []

    class OneShotLookupFailureBackend(FakeBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.failed_once = False

        def get_chunks(self, ids):
            if not self.failed_once:
                self.failed_once = True
                self.events.append("get_chunks")
                self.get_chunk_requests.append(tuple(ids))
                raise RuntimeError("snippet lookup unavailable")
            return super().get_chunks(ids)

    backend = OneShotLookupFailureBackend(
        events,
        dense=[(1, 1.0), (2, 0.9)],
        enabled=False,
    )
    cfg = make_cfg(structural_rerank=False, reranker="llm", rerank_pool=2)

    result = RetrievalV2(
        backend,
        FakeEncoder(events),
        cfg,
        reranker=LLMListwiseReranker(FakeLLM()),
    ).search("consulta")

    assert [hit["id"] for hit in result.fused] == [1, 2]


def test_builtin_cross_encoder_failure_is_reported_as_fallback():
    class FailingModel:
        def predict(self, _pairs):
            raise RuntimeError("model failed with a secret")

    result = RetrievalV2(
        FakeBackend([], dense=[(1, 1.0), (2, 0.9)], enabled=False),
        FakeEncoder([]),
        make_cfg(
            structural_rerank=False,
            reranker="cross-encoder",
            rerank_pool=2,
        ),
        reranker=CrossEncoderReranker(model=FailingModel()),
    ).search("consulta")

    assert [hit["id"] for hit in result.fused] == [1, 2]
    assert result.stats["reranker_status"] == "fallback"
    assert result.stats["reranker_reason"] == "stage-error"
    assert "secret" not in repr(result.stats)


def test_post_reranker_rank_is_the_relevance_source_for_diversity(monkeypatch):
    events = []
    backend = FakeBackend(events, dense=[(1, 100.0), (2, 1.0)], enabled=False)
    reranker = FakeReranker(events, reverse=True)
    captured = {}

    def capture(items, _vectors, top_k, **_kwargs):
        events.append("diversity")
        captured["items"] = items
        return [cid for cid, _ in items[:top_k]]

    monkeypatch.setattr("rag3d.retrieval_v2.diversify", capture)
    cfg = make_cfg(
        structural_rerank=False,
        reranker="llm",
        diversity_method="dpp",
    )

    RetrievalV2(backend, FakeEncoder(events), cfg, reranker=reranker).search("consulta")

    assert [cid for cid, _ in captured["items"]] == [2, 1]
    assert captured["items"][0][1] > captured["items"][1][1]


def test_final_score_is_ordinal_and_chat_memory_preserves_the_v2_order():
    events = []
    backend = FakeBackend(
        events,
        dense=[(1, 1.0), (2, 0.5)],
        enabled=False,
    )
    cfg = make_cfg(
        structural_rerank=False,
        reranker="llm",
        w_relevance=1.0,
        w_recency=0.0,
        w_importance=0.0,
    )
    result = RetrievalV2(
        backend,
        FakeEncoder(events),
        cfg,
        reranker=FakeReranker(events, reverse=True),
    ).search("consulta")

    assert [hit["id"] for hit in result.fused] == [2, 1]
    assert [hit["score"] for hit in result.fused] == pytest.approx([1.0, 0.5])
    assert [hit["final_rank"] for hit in result.fused] == [1, 2]
    assert result.fused[0]["fusion_score"] < result.fused[1]["fusion_score"]

    class MemoryStore:
        @staticmethod
        def last_turn_no():
            return 0

    rescored = ChatMemory(MemoryStore(), FakeEncoder([]), cfg)._memory_rescore(result)

    assert [hit["id"] for hit in rescored] == [2, 1]


def test_reranker_cannot_overwrite_the_preserved_raw_fusion_score():
    events = []

    class AnnotatingReranker(FakeReranker):
        def rerank(self, _query, hits, top_k=None):
            rows = [dict(hit, fusion_score=-999.0) for hit in reversed(hits)]
            return rows[:top_k] if top_k is not None else rows

    result = RetrievalV2(
        FakeBackend(events, dense=[(1, 1.0), (2, 0.5)], enabled=False),
        FakeEncoder(events),
        make_cfg(structural_rerank=False, reranker="llm"),
        reranker=AnnotatingReranker(events),
    ).search("consulta")

    assert [hit["id"] for hit in result.fused] == [2, 1]
    assert [hit["fusion_score"] for hit in result.fused] == pytest.approx(
        [1 / 62, 1 / 61]
    )


def test_empty_query_and_empty_corpus_are_safe():
    events = []
    backend = FakeBackend(events)
    retriever = RetrievalV2(backend, FakeEncoder(events), make_cfg())

    empty_query = retriever.search(" \t ")
    events_after_empty_query = list(events)
    empty_corpus = retriever.search("pergunta")

    assert empty_query.fused == []
    assert empty_query.views == {"semantico": [], "lexico": [], "estrutural": []}
    assert empty_query.stats["structural_candidate_depth_requested"] == 100
    assert empty_query.stats["structural_candidates_evaluated"] == 0
    assert events_after_empty_query == []
    assert empty_corpus.fused == []
    assert empty_corpus.stats["final_candidates"] == 0


def test_top_k_zero_returns_empty_without_backend_work():
    events = []
    retriever = RetrievalV2(
        FakeBackend(events, dense=[(1, 1.0)]), FakeEncoder(events), make_cfg()
    )

    result = retriever.search("consulta", top_k=0)

    assert result.fused == []
    assert events == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": -1}, "top_k"),
        ({"top_k": DEFAULT_RETRIEVAL_LIMITS.max_top_k + 1}, "top_k"),
        ({"channel_k": -1}, "channel_k"),
        ({"channel_k": DEFAULT_RETRIEVAL_LIMITS.max_channel_k + 1}, "channel_k"),
    ],
)
def test_public_search_limits_are_validated(kwargs, message):
    retriever = RetrievalV2(FakeBackend([]), FakeEncoder([]), make_cfg())

    with pytest.raises(ValueError, match=message):
        retriever.search("consulta", **kwargs)


def test_backend_results_and_structural_pool_are_clamped():
    events = []
    many = [(cid, float(2000 - cid)) for cid in range(1, 1501)]
    backend = FakeBackend(events, dense=many, sparse=many, structural=many)
    retriever = RetrievalV2(
        backend,
        FakeEncoder(events),
        make_cfg(top_k=10, channel_k=1000, rerank_pool=1000, diversity_pool=1000),
    )

    retriever.search("consulta")

    assert len(backend.structural_ids) <= DEFAULT_RETRIEVAL_LIMITS.max_pool


def test_structural_candidate_depth_is_independent_and_can_promote_a_deep_hit():
    events = []
    dense = [(cid, float(11 - cid)) for cid in range(1, 11)]
    backend = FakeBackend(
        events,
        dense=dense,
        structural=[(10, 1.0)],
    )
    cfg = make_cfg(
        top_k=2,
        channel_k=10,
        structural_candidate_depth=10,
        reranker="none",
        diversity_method="none",
    )

    result = RetrievalV2(backend, FakeEncoder(events), cfg).search("consulta")

    assert backend.structural_ids == list(range(1, 11))
    assert [hit["id"] for hit in result.fused] == [10, 1]
    assert result.stats["structural_candidate_depth_requested"] == 10
    assert result.stats["structural_candidates_evaluated"] == 10


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"structural_candidate_depth": True}, TypeError),
        ({"structural_candidate_depth": 0}, ValueError),
        (
            {
                "structural_candidate_depth": (
                    DEFAULT_RETRIEVAL_LIMITS.max_pool + 1
                )
            },
            ValueError,
        ),
        ({"stitch_radius": True}, TypeError),
        ({"stitch_radius": -1}, ValueError),
        (
            {"stitch_radius": DEFAULT_RETRIEVAL_LIMITS.max_top_k + 1},
            ValueError,
        ),
    ],
)
def test_v2_structural_and_stitch_configuration_is_bounded(kwargs, error):
    with pytest.raises(error):
        make_cfg(**kwargs)


def test_candidate_ranking_does_not_materialize_an_unbounded_backend_iterator():
    events = []

    class StreamingBackend(FakeBackend):
        def dense_search(self, _vector, k, *, filters=None, exact=None):
            self.events.append("dense")
            return StrictBoundedRanking(k)

    backend = StreamingBackend(events, enabled=False)

    result = RetrievalV2(
        backend,
        FakeEncoder(events),
        make_cfg(channel_k=5, structural_rerank=False),
    ).search("consulta")

    assert result.stats["dense_candidates"] == 5


def test_candidate_ranking_rejects_sized_oversize_before_iteration():
    class HonestOversize:
        def __len__(self):
            return 6

        def __iter__(self):
            raise AssertionError("oversize preflight must not iterate")

    with pytest.raises(ValueError, match="candidate ranking"):
        RetrievalV2._candidate_ranking(HonestOversize(), 5)


def test_candidate_ranking_stops_lying_iterator_at_limit_plus_one():
    class LyingRanking:
        def __init__(self):
            self.consumed = 0

        def __len__(self):
            return 0

        def __iter__(self):
            for index in range(100_000):
                self.consumed += 1
                yield (index, 1.0)

    ranking = LyingRanking()
    with pytest.raises(ValueError, match="candidate ranking"):
        RetrievalV2._candidate_ranking(ranking, 5)

    assert ranking.consumed == 6


def test_structural_ranking_does_not_materialize_past_its_bounded_prefix():
    events = []

    class StreamingStructuralBackend(FakeBackend):
        def structural_rerank(self, _vectors, candidate_ids, k, *, filters=None):
            self.events.append("structural")
            self.structural_ids = list(candidate_ids)
            return StrictBoundedRanking(k, start=min(candidate_ids))

    backend = StreamingStructuralBackend(
        events,
        dense=[(1, 1.0), (2, 0.5)],
        structural=[(1, 1.0)],
    )

    result = RetrievalV2(backend, FakeEncoder(events), make_cfg()).search("consulta")

    assert result.stats["structural_candidates"] == 2


def test_structural_ranking_rejects_oversize_at_limit_plus_one():
    class LyingRanking:
        def __init__(self):
            self.consumed = 0

        def __len__(self):
            return 0

        def __iter__(self):
            for index in range(100_000):
                self.consumed += 1
                yield (index + 1, 1.0)

    ranking = LyingRanking()

    assert RetrievalV2._structural_ranking(ranking, [1, 2], 2) is None
    assert ranking.consumed == 3


def test_diagnostics_are_complete_finite_aggregate_only_and_counted():
    events = []
    backend = FakeBackend(
        events,
        dense=[(1, 1.0), (2, 0.5)],
        sparse=[(2, 2.0), (3, 0.1)],
        structural=[(3, 100.0), (2, 50.0)],
    )
    filters = SearchFilters(metadata={"tenant": "secret-filter-value"})

    result = RetrievalV2(backend, FakeEncoder(events), make_cfg()).search(
        "secret query text", filters=filters
    )

    timing_keys = {
        "normalize_ms",
        "expand_ms",
        "encode_ms",
        "dense_ms",
        "sparse_ms",
        "union_ms",
        "fusion_ms",
        "structural_ms",
        "rerank_ms",
        "diversity_ms",
        "hydrate_ms",
        "stitch_ms",
        "total_retrieval_ms",
        "total_ms",
    }
    count_keys = {
        "dense_candidates",
        "sparse_candidates",
        "union_candidates",
        "fused_candidates",
        "structural_candidates",
        "reranked_candidates",
        "final_candidates",
    }
    assert timing_keys | count_keys <= result.stats.keys()
    assert all(math.isfinite(result.stats[key]) and result.stats[key] >= 0 for key in timing_keys)
    assert result.stats["dense_candidates"] == 2
    assert result.stats["sparse_candidates"] == 2
    assert result.stats["union_candidates"] == 3
    assert result.stats["backend"] == "unknown"
    assert result.stats["pipeline"] == "v2"
    assert result.stats["filters_applied"] is True
    assert result.stats["filter_count"] == 1
    assert result.stats["structural_status"] == "applied"
    assert result.stats["structural_reason"] == "none"
    assert result.stats["reranker_status"] == "skipped"
    assert result.stats["reranker_reason"] == "not-configured"
    serialized = repr(result.stats)
    assert "secret query text" not in serialized
    assert "secret-filter-value" not in serialized
    assert "dsn" not in serialized.lower()
    assert backend.filters_seen and all(item is filters for item in backend.filters_seen)


def test_diagnostic_identifiers_use_an_allowlist_and_never_normalize_secrets():
    events = []
    backend = FakeBackend(events, dense=[(1, 1.0)], enabled=False)
    backend.backend_name = "postgresql://alice:supersecret@db/private"
    encoder = FakeEncoder(events)
    encoder.name = "token=encoder-secret"
    reranker = FakeReranker(events, reverse=False)
    reranker.name = "https://user:reranker-secret@example.invalid"
    cfg = make_cfg(structural_rerank=False, reranker="llm")

    result = RetrievalV2(backend, encoder, cfg, reranker=reranker).search("consulta")

    assert result.stats["backend"] == "unknown"
    assert result.stats["encoder"] == "unknown"
    assert result.stats["reranker"] == "unknown"
    serialized = repr(result.stats).lower()
    assert "supersecret" not in serialized
    assert "encoder-secret" not in serialized
    assert "reranker-secret" not in serialized
    assert "postgresql" not in serialized


def test_total_retrieval_time_encloses_every_search_stage_without_double_counting():
    events = []
    result = RetrievalV2(
        FakeBackend(
            events,
            dense=[(1, 1.0), (2, 0.5)],
            sparse=[(2, 2.0)],
            structural=[(1, 4.0), (2, 3.0)],
        ),
        FakeEncoder(events),
        make_cfg(stitch_radius=1),
    ).search("consulta")
    individual = [
        "normalize_ms",
        "expand_ms",
        "encode_ms",
        "dense_ms",
        "sparse_ms",
        "union_ms",
        "fusion_ms",
        "structural_ms",
        "rerank_ms",
        "diversity_ms",
        "hydrate_ms",
        "stitch_ms",
    ]

    assert result.stats["total_retrieval_ms"] >= sum(
        result.stats[name] for name in individual
    )
    assert result.stats["total_ms"] >= result.stats["total_retrieval_ms"]


def test_expansion_is_normalized_and_timed_separately_from_encoding():
    events = []
    encoder = FakeEncoder(events)
    cfg = make_cfg(expand_query=True, expand_query_max=2)
    retriever = RetrievalV2(FakeBackend(events), encoder, cfg, llm=FakeLLM())

    result = retriever.search("  número 13. 243  ")

    assert encoder.texts == ["número 13.243", "variante um", "variante dois"]
    assert result.stats["expand_ms"] >= 0
    assert result.stats["encode_ms"] >= 0


def test_v2_oversized_llm_expansion_falls_back_before_splitting_or_encoding_it():
    events = []
    encoder = FakeEncoder(events)

    class OversizedExpansionLLM(FakeLLM):
        def complete(self, *_args, **_kwargs):
            return "x" * (DEFAULT_RETRIEVAL_LIMITS.max_query_bytes + 1)

    retriever = RetrievalV2(
        FakeBackend(events),
        encoder,
        make_cfg(expand_query=True, expand_query_max=2),
        llm=OversizedExpansionLLM(),
    )

    retriever.search("consulta")

    assert encoder.texts == ["consulta"]


def test_query_encoder_must_return_exactly_one_vector_per_requested_text():
    events = []

    class ExtraVectorEncoder(FakeEncoder):
        def encode(self, texts, is_query=False):
            return super().encode([*texts, "unexpected-extra"], is_query=is_query)

    retriever = RetrievalV2(
        FakeBackend(events),
        ExtraVectorEncoder(events),
        make_cfg(structural_rerank=False),
    )

    with pytest.raises(ValueError, match="exactly one vector per query text"):
        retriever.search("consulta")


def test_v2_rejects_oversized_query_before_encoding() -> None:
    events = []
    encoder = FakeEncoder(events)
    retriever = RetrievalV2(
        FakeBackend(events),
        encoder,
        make_cfg(structural_rerank=False),
    )

    with pytest.raises(ValueError, match="query.*bytes"):
        retriever.search("x" * (DEFAULT_RETRIEVAL_LIMITS.max_query_bytes + 1))

    assert encoder.texts == []


def test_hydration_small_to_big_and_stitch_preserve_reader_hit_shape():
    events = []
    rows = {
        1: make_row(1, parent_id=10, doc_id=7, pos=1, text="meio"),
        10: make_row(10, doc_id=7, pos=0, text="janela pai", kind="parent"),
        11: make_row(11, doc_id=7, pos=0, text="antes"),
        12: make_row(12, doc_id=7, pos=2, text="depois"),
    }
    backend = FakeBackend(events, dense=[(1, 1.0)], rows=rows, enabled=False)

    result = RetrievalV2(
        backend,
        FakeEncoder(events),
        make_cfg(top_k=1, structural_rerank=False, stitch_radius=1),
    ).search("consulta")

    hit = result.fused[0]
    assert hit["wide"] == "antes meio depois"
    assert {
        "id",
        "kind",
        "doc_id",
        "pos",
        "text",
        "wide",
        "n_tokens",
        "turn_no",
        "accessed_turn",
        "created",
        "importance",
        "score",
        "fusion_score",
    } <= hit.keys()
    assert events.count("get_chunks") == 2
    assert "neighbors" in events


def test_stitch_never_adds_neighbor_text_outside_an_active_filter_scope():
    events = []
    rows = {
        1: make_row(
            1,
            parent_id=10,
            doc_id=7,
            pos=1,
            text="ALLOWED",
        ),
        2: make_row(
            2,
            parent_id=20,
            doc_id=7,
            pos=2,
            text="OUT_OF_SCOPE_PARENT",
        ),
    }
    backend = FakeBackend(events, dense=[(1, 1.0)], rows=rows, enabled=False)
    filters = SearchFilters(scope=SearchScope(parent_ids=(10,)))

    result = RetrievalV2(
        backend,
        FakeEncoder(events),
        make_cfg(top_k=1, structural_rerank=False, stitch_radius=1),
    ).search("consulta", filters=filters)

    assert result.fused[0]["wide"] == "ALLOWED"
    assert "OUT_OF_SCOPE_PARENT" not in repr(result.fused)
    assert "neighbors" not in events
    assert result.stats["stitch_status"] == "skipped"
    assert result.stats["stitch_reason"] == "filters-applied"


def test_hydration_rejects_a_parent_from_another_document_scope():
    rows = {
        1: make_row(1, parent_id=10, doc_id=7, text="ALLOWED"),
        10: make_row(
            10,
            doc_id=99,
            text="SECRET_OTHER_DOCUMENT",
            kind="parent",
        ),
    }
    backend = FakeBackend([], dense=[(1, 1.0)], rows=rows, enabled=False)
    filters = SearchFilters(scope=SearchScope(document_ids=(7,)))

    result = RetrievalV2(
        backend,
        FakeEncoder([]),
        make_cfg(top_k=1, structural_rerank=False, stitch_radius=0),
    ).search("consulta", filters=filters)

    assert result.fused[0]["wide"] == "ALLOWED"
    assert "SECRET_OTHER_DOCUMENT" not in repr(result.fused)


def test_stitch_adapter_failure_is_all_or_nothing_and_diagnosed():
    class PartiallyFailingNeighbors(FakeBackend):
        def neighbors(self, doc_id, positions):
            if doc_id == 8:
                raise RuntimeError("neighbor secret")
            return [
                make_row(11, doc_id=doc_id, pos=0, text="SHOULD_NOT_APPLY")
            ]

    rows = {
        1: make_row(1, doc_id=7, pos=0, text="first"),
        2: make_row(2, doc_id=8, pos=0, text="second"),
    }
    backend = PartiallyFailingNeighbors(
        [], dense=[(1, 1.0), (2, 0.9)], rows=rows, enabled=False
    )

    result = RetrievalV2(
        backend,
        FakeEncoder([]),
        make_cfg(top_k=2, structural_rerank=False, stitch_radius=1),
    ).search("consulta")

    assert [hit["wide"] for hit in result.fused] == ["first", "second"]
    assert result.stats["stitch_status"] == "fallback"
    assert result.stats["stitch_reason"] == "stage-error"
    assert "secret" not in repr(result.stats)


@pytest.mark.parametrize(
    "neighbor_rows",
    [
        [make_row(11, doc_id=999, pos=0, text="OTHER_DOCUMENT")],
        [make_row(11, doc_id=7, pos=99, text="UNREQUESTED_POSITION")],
        [
            make_row(11, doc_id=7, pos=0, text="DUPLICATE_A"),
            make_row(12, doc_id=7, pos=0, text="DUPLICATE_B"),
        ],
    ],
)
def test_stitch_rejects_cross_document_unrequested_or_duplicate_neighbors(
    neighbor_rows,
):
    class InvalidNeighbors(FakeBackend):
        def neighbors(self, doc_id, positions):
            return [dict(row) for row in neighbor_rows]

    rows = {1: make_row(1, doc_id=7, pos=0, text="ALLOWED")}
    backend = InvalidNeighbors([], dense=[(1, 1.0)], rows=rows, enabled=False)

    result = RetrievalV2(
        backend,
        FakeEncoder([]),
        make_cfg(top_k=1, structural_rerank=False, stitch_radius=1),
    ).search("consulta")

    assert result.fused[0]["wide"] == "ALLOWED"
    assert result.stats["stitch_status"] == "fallback"
    assert result.stats["stitch_reason"] == "invalid-output"


def test_diversity_vector_failure_preserves_order_and_is_diagnosed():
    class FailingVectors(FakeBackend):
        def dense_vectors(self, ids):
            raise RuntimeError("vector secret")

    backend = FailingVectors(
        [], dense=[(1, 1.0), (2, 0.9)], enabled=False
    )

    result = RetrievalV2(
        backend,
        FakeEncoder([]),
        make_cfg(
            top_k=2,
            structural_rerank=False,
            diversity_method="dpp",
        ),
    ).search("consulta")

    assert [hit["id"] for hit in result.fused] == [1, 2]
    assert result.stats["diversity_status"] == "fallback"
    assert result.stats["diversity_reason"] == "stage-error"
    assert "secret" not in repr(result.stats)


def test_diversity_zero_vectors_preserve_order_and_report_invalid_output():
    class ZeroVectors(FakeBackend):
        def dense_vectors(self, ids):
            return {cid: np.zeros(2, dtype=np.float32) for cid in ids}

    result = RetrievalV2(
        ZeroVectors([], dense=[(1, 1.0), (2, 0.9)], enabled=False),
        FakeEncoder([]),
        make_cfg(
            top_k=2,
            structural_rerank=False,
            diversity_method="dpp",
        ),
    ).search("consulta")

    assert [hit["id"] for hit in result.fused] == [1, 2]
    assert result.stats["diversity_status"] == "fallback"
    assert result.stats["diversity_reason"] == "invalid-output"


def test_diversity_overflowing_norm_preserves_order_and_reports_invalid_output():
    class OverflowingNormVectors(FakeBackend):
        def dense_vectors(self, ids):
            values = {
                1: np.array([1e308, 1e308], dtype=np.float64),
                2: np.array([1e308, -1e308], dtype=np.float64),
            }
            return {cid: values[cid] for cid in ids}

    result = RetrievalV2(
        OverflowingNormVectors(
            [], dense=[(1, 1.0), (2, 0.9)], enabled=False
        ),
        FakeEncoder([]),
        make_cfg(
            dense_dim=2,
            top_k=2,
            structural_rerank=False,
            diversity_method="dpp",
        ),
    ).search("consulta")

    assert [hit["id"] for hit in result.fused] == [1, 2]
    assert result.stats["diversity_status"] == "fallback"
    assert result.stats["diversity_reason"] == "invalid-output"


def test_hydration_backfills_missing_selected_ids_in_fused_and_views():
    events = []
    rows = {
        2: make_row(2, text="segundo"),
        3: make_row(3, text="terceiro"),
    }
    backend = FakeBackend(
        events,
        dense=[(1, 3.0), (2, 2.0), (3, 1.0)],
        rows=rows,
        enabled=False,
    )

    result = RetrievalV2(
        backend,
        FakeEncoder(events),
        make_cfg(top_k=2, structural_rerank=False),
    ).search("consulta")

    assert [hit["id"] for hit in result.fused] == [2, 3]
    assert [hit["id"] for hit in result.views["semantico"]] == [2, 3]
    assert [hit["score"] for hit in result.fused] == pytest.approx([1.0, 0.5])


def test_stitch_aggregate_neighbor_positions_are_bounded():
    events = []

    class RecordingNeighborsBackend(FakeBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.neighbor_requests = []

        def neighbors(self, doc_id, positions):
            self.neighbor_requests.append((doc_id, tuple(positions)))
            return super().neighbors(doc_id, positions)

    dense = [(cid, float(101 - cid)) for cid in range(1, 101)]
    rows = {
        cid: make_row(cid, doc_id=cid, pos=0)
        for cid in range(1, 101)
    }
    backend = RecordingNeighborsBackend(
        events,
        dense=dense,
        rows=rows,
        enabled=False,
    )

    RetrievalV2(
        backend,
        FakeEncoder(events),
        make_cfg(
            top_k=100,
            channel_k=100,
            structural_rerank=False,
            stitch_radius=DEFAULT_RETRIEVAL_LIMITS.max_top_k,
        ),
    ).search("consulta")

    assert sum(len(positions) for _doc_id, positions in backend.neighbor_requests) <= (
        DEFAULT_RETRIEVAL_LIMITS.max_pool
    )


def test_stage_observer_receives_copied_ids_and_cannot_break_retrieval():
    events = []
    observed = {}

    def observer(stage, ids):
        assert isinstance(ids, tuple)
        observed[stage] = ids
        if stage == "sparse":
            raise RuntimeError("benchmark observer failure")

    backend = FakeBackend(
        events,
        dense=[(1, 1.0), (2, 0.5)],
        sparse=[(2, 2.0), (3, 1.0)],
        structural=[(3, 7.0), (1, 6.0)],
    )

    result = RetrievalV2(
        backend,
        FakeEncoder(events),
        make_cfg(),
        stage_observer=observer,
    ).search("consulta")

    assert set(observed) == {
        "dense",
        "sparse",
        "union",
        "fusion",
        "structural",
        "reranker",
        "final",
    }
    assert observed["union"] == tuple(sorted({1, 2, 3}))
    assert result.fused
    assert "stage_observer" not in result.stats


def test_default_diagnostics_do_not_retain_per_document_stage_ids():
    result = RetrievalV2(
        FakeBackend([], dense=[(987654321, 1.0)], enabled=False),
        FakeEncoder([]),
        make_cfg(top_k=1, structural_rerank=False),
    ).search("consulta")

    assert "987654321" not in repr(result.stats)

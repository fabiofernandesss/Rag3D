from __future__ import annotations

import hashlib
import json
import sys
import types
from contextlib import contextmanager

import pytest

import rag3d.engine as engine_module
from rag3d.backend import (
    DEFAULT_RETRIEVAL_LIMITS,
    BackendCapabilities,
    FingerprintMismatchError,
    SearchFilters,
)
from rag3d.config import TriRagConfig
from rag3d.engine import TriRag
from rag3d.encoders import HashEncoder
from rag3d.llm import NoLLM
from rag3d.rerank import (
    CrossEncoderReranker,
    LLMListwiseReranker,
    NoOpReranker,
    Reranker,
)
from rag3d.retrieve import TriRetriever
from rag3d.retrieval_v2 import RetrievalV2
from rag3d.store import TriStore


def _cfg(tmp_path, **overrides):
    values = {
        "data_dir": tmp_path,
        "backend": "sqlite",
        "retrieval_pipeline": "v2",
        "encoder": "hash",
        "dense_dim": 8,
        "colbert_dim": 4,
        "max_colbert_tokens": 8,
        "contextual_enrich": False,
    }
    values.update(overrides)
    return TriRagConfig(**values)


class _FakeRemoteStore:
    capabilities = BackendCapabilities(transactions=True)

    def __init__(self, dsn, dense_dim, colbert_dim):
        self.received = (dsn, dense_dim, colbert_dim)
        self.metadata = {}
        self.backend_name = "postgres-holo"
        self.events = []
        self.close_calls = 0

    @contextmanager
    def transaction(self):
        self.events.append("transaction")
        yield

    def lock_fingerprint(self):
        self.events.append("fingerprint-lock")

    def get_meta(self, key):
        self.events.append("get-meta")
        return self.metadata.get(key)

    def set_meta(self, key, value):
        self.metadata[key] = value

    def n_chunks(self):
        return 0

    def corpus_tokens(self, kinds=("chunk", "turn")):
        return 0

    def last_turn_no(self):
        return 0

    def health(self):
        return {"status": "ok", "backend": self.backend_name}

    def close(self):
        self.close_calls += 1


def _install_fake_store_module(monkeypatch, module_name, class_name, store_class):
    module = types.ModuleType(module_name)
    setattr(module, class_name, store_class)
    monkeypatch.setitem(sys.modules, module_name, module)


def test_engine_resolves_config_then_honors_explicit_sqlite_over_dsn(tmp_path):
    secret_dsn = "postgresql://admin:super-secret@example.invalid/prod"
    cfg = _cfg(tmp_path, backend="sqlite", pg_dsn=secret_dsn)
    original_resolve = cfg.resolve
    calls = []

    def recording_resolve():
        calls.append("resolve")
        return original_resolve()

    cfg.resolve = recording_resolve
    rag = TriRag(cfg, llm=NoLLM())
    try:
        assert calls == ["resolve"]
        assert isinstance(rag.store, TriStore)
        assert rag.store.backend_name == "sqlite"
        stats = rag.stats()
        assert stats["backend"] == "sqlite"
        assert secret_dsn not in repr(stats)
        assert "super-secret" not in repr(stats)
    finally:
        rag.store.close()


def test_legacy_facade_rejects_oversized_query_before_encoding(tmp_path):
    cfg = _cfg(tmp_path, retrieval_pipeline="legacy")
    rag = TriRag(cfg, llm=NoLLM())
    try:
        with pytest.raises(ValueError, match="query.*bytes"):
            rag.search("x" * (DEFAULT_RETRIEVAL_LIMITS.max_query_bytes + 1))
    finally:
        rag.store.close()


@pytest.mark.parametrize("backend", ["postgres-holo", "pgvector"])
def test_engine_requires_dsn_for_postgres_backends_without_echoing_secrets(tmp_path, backend):
    cfg = _cfg(tmp_path, backend=backend, pg_dsn="")
    with pytest.raises(ValueError, match="DSN|required") as raised:
        TriRag(cfg, llm=NoLLM())
    assert repr(tmp_path) not in str(raised.value)


def test_engine_selects_postgres_holo_lazily_and_masks_stats(monkeypatch, tmp_path):
    _install_fake_store_module(
        monkeypatch, "rag3d.pgstore", "PgHoloStore", _FakeRemoteStore
    )
    secret_dsn = "postgresql://admin:super-secret@example.invalid/prod"
    cfg = _cfg(tmp_path, backend="postgres-holo", pg_dsn=secret_dsn)

    rag = TriRag(cfg, llm=NoLLM())

    assert rag.store.received == (secret_dsn, 8, 4)
    stats = rag.stats()
    assert stats["backend"] == "postgres-holo"
    assert secret_dsn not in repr(stats)
    assert "super-secret" not in repr(stats)


def test_engine_selects_pgvector_lazily_when_adapter_exists(monkeypatch, tmp_path):
    class FakePgVector(_FakeRemoteStore):
        def __init__(self, dsn, dense_dim, colbert_dim, **kwargs):
            super().__init__(dsn, dense_dim, colbert_dim)
            self.backend_name = "pgvector"
            self.received_options = kwargs
            self.search_mode = kwargs["search_mode"]

    _install_fake_store_module(
        monkeypatch, "rag3d.pgvector_store", "PgVectorStore", FakePgVector
    )
    secret_dsn = "postgresql://admin:super-secret@example.invalid/prod"
    rag = TriRag(
        _cfg(tmp_path, backend="pgvector", pg_dsn=secret_dsn), llm=NoLLM()
    )

    assert rag.store.backend_name == "pgvector"
    fingerprint = rag.store.received_options["fingerprint"]
    assert fingerprint.backend == "pgvector"
    assert fingerprint.encoder == "hash"
    assert rag.store.received_options["search_mode"] == "exact"
    assert rag.store.received_options["statement_timeout_ms"] == 5_000
    assert rag.stats()["pgvector_search_mode"] == "exact"
    assert secret_dsn not in repr(rag.stats())


def test_engine_pgvector_import_error_is_clear_and_secret_safe(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "rag3d.pgvector_store", None)
    secret_dsn = "postgresql://admin:super-secret@example.invalid/prod"

    with pytest.raises(RuntimeError, match="pgvector.*unavailable") as raised:
        TriRag(
            _cfg(tmp_path, backend="pgvector", pg_dsn=secret_dsn), llm=NoLLM()
        )

    assert secret_dsn not in str(raised.value)
    assert "super-secret" not in str(raised.value)
    assert raised.value.__cause__ is not None


def test_engine_chains_backend_constructor_cause_without_echoing_dsn(
    monkeypatch, tmp_path
):
    secret = "postgresql://admin:super-secret@example.invalid/prod"

    class FailingStore:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("container dns lookup failed")

        def close(self):
            pass

    _install_fake_store_module(
        monkeypatch, "rag3d.pgstore", "PgHoloStore", FailingStore
    )

    with pytest.raises(RuntimeError, match="failed to initialize backend") as raised:
        TriRag(
            _cfg(tmp_path, backend="postgres-holo", pg_dsn=secret), llm=NoLLM()
        )

    assert secret not in str(raised.value)
    assert "super-secret" not in str(raised.value)
    assert str(raised.value.__cause__) == "container dns lookup failed"


def test_engine_preserves_pgvector_fingerprint_mismatch_type(monkeypatch, tmp_path):
    class MismatchedPgVector:
        def __init__(self, *args, **kwargs):
            raise FingerprintMismatchError(
                "incompatible index fingerprint; differing fields: model"
            )

    _install_fake_store_module(
        monkeypatch, "rag3d.pgvector_store", "PgVectorStore", MismatchedPgVector
    )

    with pytest.raises(FingerprintMismatchError, match="differing fields: model"):
        TriRag(
            _cfg(
                tmp_path,
                backend="pgvector",
                pg_dsn="postgresql://example.invalid/rag3d_test",
            ),
            llm=NoLLM(),
        )


def test_engine_passes_resolved_fallback_policy_to_encoder(monkeypatch, tmp_path):
    captured = {}

    def fake_make_encoder(kind, dense_dim, colbert_dim, max_tokens, allow_fallback=True):
        captured["allow_fallback"] = allow_fallback
        return HashEncoder(dense_dim, colbert_dim, max_tokens)

    monkeypatch.setattr(engine_module, "make_encoder", fake_make_encoder)
    cfg = _cfg(
        tmp_path,
        encoder="auto",
        retrieval_pipeline="v2",
        allow_encoder_fallback=False,
    )
    rag = TriRag(cfg, llm=NoLLM())
    try:
        assert captured["allow_fallback"] is False
    finally:
        rag.store.close()


def test_engine_preserves_legacy_encoder_key_and_persists_separate_v2_fingerprint(tmp_path):
    cfg = _cfg(tmp_path)
    rag = TriRag(cfg, llm=NoLLM())
    try:
        legacy = rag.store.get_meta("encoder")
        payload = rag.store.get_meta("retrieval_v2_fingerprint")
        digest = rag.store.get_meta("retrieval_v2_fingerprint_sha256")

        assert legacy == "hash:8:4"
        assert isinstance(payload, str)
        parsed = json.loads(payload)
        assert parsed["backend"] == "sqlite"
        assert parsed["encoder"] == "hash"
        assert parsed["dense_dim"] == 8
        assert parsed["structural_dim"] == 4
        assert parsed["max_structural_tokens"] == 8
        assert parsed["structural_projection"] == "hash-token-ngrams-2-4-v1"
        assert parsed["query_max_tokens"] == 8
        assert parsed["passage_max_tokens"] == 8
        assert parsed["sparse_version"] == "crc32-unicode-word-v1"
        assert parsed["schema_version"] == "rag3d-trivec-v2"
        assert digest == hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert str(tmp_path) not in payload
    finally:
        rag.store.close()

    reopened = TriRag(_cfg(tmp_path), llm=NoLLM())
    reopened.store.close()


def test_engine_fingerprint_mismatch_error_omits_storage_location(tmp_path):
    secret_dir = tmp_path / "do-not-leak-this-location"
    first = TriRag(_cfg(secret_dir), llm=NoLLM())
    first.store.close()

    with pytest.raises(RuntimeError, match="fingerprint|encoder") as raised:
        TriRag(_cfg(secret_dir, dense_dim=16), llm=NoLLM())

    assert str(secret_dir) not in str(raised.value)
    assert "do-not-leak-this-location" not in str(raised.value)


def test_engine_rejects_structural_token_limit_mismatch(tmp_path):
    first = TriRag(_cfg(tmp_path, max_colbert_tokens=8), llm=NoLLM())
    first.store.close()

    with pytest.raises(RuntimeError, match="max_structural_tokens|fingerprint"):
        TriRag(_cfg(tmp_path, max_colbert_tokens=9), llm=NoLLM())


def test_legacy_rollback_validates_any_existing_v2_fingerprint(tmp_path):
    first = TriRag(_cfg(tmp_path, max_colbert_tokens=8), llm=NoLLM())
    first.ingest("Certified Retrieval V2 content.")
    first.store.close()

    compatible = TriRag(
        _cfg(tmp_path, retrieval_pipeline="legacy", max_colbert_tokens=8),
        llm=NoLLM(),
    )
    compatible.store.close()

    with pytest.raises(FingerprintMismatchError, match="max_structural_tokens"):
        TriRag(
            _cfg(tmp_path, retrieval_pipeline="legacy", max_colbert_tokens=7),
            llm=NoLLM(),
        )


def test_engine_locks_first_fingerprint_publication_before_metadata_read(
    monkeypatch, tmp_path
):
    _install_fake_store_module(
        monkeypatch, "rag3d.pgstore", "PgHoloStore", _FakeRemoteStore
    )
    rag = TriRag(
        _cfg(
            tmp_path,
            backend="postgres-holo",
            pg_dsn="postgresql://test-only.invalid/db",
        ),
        llm=NoLLM(),
    )

    assert rag.store.events[:3] == [
        "transaction",
        "fingerprint-lock",
        "get-meta",
    ]


def test_engine_closes_open_store_once_when_encoder_construction_fails(
    monkeypatch, tmp_path
):
    store = _FakeRemoteStore("test-only", 8, 4)
    monkeypatch.setattr(TriRag, "_make_store", lambda self, data: store)

    def fail_encoder(*args, **kwargs):
        raise RuntimeError("encoder construction failed")

    monkeypatch.setattr(engine_module, "make_encoder", fail_encoder)
    with pytest.raises(RuntimeError, match="encoder construction"):
        TriRag(_cfg(tmp_path), llm=NoLLM())

    assert store.close_calls == 1


def test_engine_closes_open_store_once_when_consumer_construction_fails(
    monkeypatch, tmp_path
):
    store = _FakeRemoteStore("test-only", 8, 4)
    monkeypatch.setattr(TriRag, "_make_store", lambda self, data: store)

    class BrokenIngestor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("consumer construction failed")

    monkeypatch.setattr(engine_module, "Ingestor", BrokenIngestor)
    with pytest.raises(RuntimeError, match="consumer construction"):
        TriRag(_cfg(tmp_path), llm=NoLLM())

    assert store.close_calls == 1


def test_v2_refuses_to_guess_missing_fingerprint_for_populated_legacy_index(tmp_path):
    legacy = TriRag(
        _cfg(tmp_path, retrieval_pipeline="legacy"), llm=NoLLM()
    )
    legacy.ingest("Legacy content that already belongs to an index.")
    legacy.store.close()

    with pytest.raises(RuntimeError, match="legacy|reindex|fingerprint") as raised:
        TriRag(_cfg(tmp_path, retrieval_pipeline="v2"), llm=NoLLM())

    assert str(tmp_path) not in str(raised.value)
    compatible_legacy = TriRag(
        _cfg(tmp_path, retrieval_pipeline="legacy"), llm=NoLLM()
    )
    assert compatible_legacy.store.n_chunks() == 1
    compatible_legacy.store.close()


def test_v2_rejects_tampered_fingerprint_digest(tmp_path):
    rag = TriRag(_cfg(tmp_path), llm=NoLLM())
    rag.store.close()
    store = TriStore(tmp_path / "trirag.db")
    store.set_meta("retrieval_v2_fingerprint_sha256", "0" * 64)
    store.close()

    with pytest.raises(RuntimeError, match="digest|fingerprint") as raised:
        TriRag(_cfg(tmp_path), llm=NoLLM())

    assert str(tmp_path) not in str(raised.value)


@pytest.mark.parametrize(
    "reranker_name, expected_type",
    [
        ("none", NoOpReranker),
        ("llm", LLMListwiseReranker),
        ("cross-encoder", CrossEncoderReranker),
    ],
)
def test_engine_selects_v2_pipeline_and_injects_configured_reranker(
    tmp_path, reranker_name, expected_type
):
    rag = TriRag(_cfg(tmp_path, reranker=reranker_name), llm=NoLLM())
    try:
        assert isinstance(rag.retriever, RetrievalV2)
        assert isinstance(rag.retriever.reranker, expected_type)
        assert rag.retriever.backend is rag.store
        assert rag.retriever.encoder is rag.encoder
        assert rag.retriever.llm is rag.llm
    finally:
        rag.store.close()


def test_engine_preserves_legacy_retriever_and_reranker_contract(tmp_path):
    cfg = _cfg(
        tmp_path,
        retrieval_pipeline="legacy",
        fusion="quantum",
        rerank=True,
    )
    rag = TriRag(cfg, llm=NoLLM())
    try:
        assert type(rag.retriever) is TriRetriever
        assert isinstance(rag.retriever.reranker, Reranker)
    finally:
        rag.store.close()


def test_public_search_forwards_v2_channel_depth_and_filters(tmp_path):
    rag = TriRag(_cfg(tmp_path), llm=NoLLM())
    calls = []

    class RecordingRetriever:
        def search(self, query, top_k=None, channel_k=None, filters=None):
            calls.append((query, top_k, channel_k, filters))
            return "sentinel-result"

    filters = SearchFilters()
    rag.retriever = RecordingRetriever()
    try:
        assert (
            rag.search("consulta", top_k=3, channel_k=7, filters=filters)
            == "sentinel-result"
        )
        assert calls == [("consulta", 3, 7, filters)]
    finally:
        rag.store.close()


def test_legacy_search_rejects_nonempty_v2_filters_explicitly(tmp_path):
    rag = TriRag(
        _cfg(tmp_path, retrieval_pipeline="legacy", fusion="quantum"),
        llm=NoLLM(),
    )
    try:
        with pytest.raises(NotImplementedError, match="filters.*V2"):
            rag.search(
                "consulta",
                filters=SearchFilters(metadata={"tenant": "acme"}),
            )
    finally:
        rag.store.close()


def test_stats_identifies_pipeline_and_v2_stages_without_sensitive_values(tmp_path):
    rag = TriRag(
        _cfg(tmp_path, reranker="cross-encoder", diversity_method="mmr"),
        llm=NoLLM(),
    )
    try:
        stats = rag.stats()
        assert stats["pipeline"] == "v2"
        assert stats["fusion"] == "rrf"
        assert stats["reranker"] == "cross-encoder"
        assert stats["diversity"] == "mmr"
        assert str(tmp_path) not in repr(stats)
    finally:
        rag.store.close()


def test_v2_engine_ingests_and_retrieves_through_public_facade(tmp_path):
    rag = TriRag(
        _cfg(
            tmp_path,
            top_k=2,
            channel_k=8,
            structural_candidate_depth=8,
            diversity_method="none",
            stitch_radius=0,
        ),
        llm=NoLLM(),
    )
    try:
        rag.ingest("O contrato Atlas vence em setembro de 2031.", title="atlas")
        rag.ingest("A política Boreal trata apenas de férias.", title="boreal")

        result = rag.search("quando vence o contrato Atlas?", top_k=1)

        assert len(result.fused) == 1
        assert "Atlas" in result.fused[0]["text"]
        assert result.stats["pipeline"] == "v2"
        assert result.stats["fusion"] == "rrf"
        assert result.stats["final_candidates"] == 1
        assert result.fused[0]["score"] == pytest.approx(1.0)
        assert result.fused[0]["final_rank"] == 1
    finally:
        rag.store.close()

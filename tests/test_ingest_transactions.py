from __future__ import annotations

from contextlib import contextmanager

import pytest

from rag3d.config import TriRagConfig
from rag3d.encoders import HashEncoder, TriVec
from rag3d.ingest import (
    MAX_EMBEDDING_BATCH,
    MAX_INGEST_CHUNKS,
    MAX_INGEST_DOCUMENT_BYTES,
    Ingestor,
)
from rag3d.store import TriStore

MAX_TEST_INGEST_LABEL_BYTES = 4 * 1024


class _FailingEncoder:
    def encode(self, texts, is_query=False):
        raise RuntimeError("encoder failure")


class _ShortEncoder:
    def encode(self, texts, is_query=False):
        import numpy as np

        return [
            TriVec(
                dense=np.ones(8, dtype=np.float32),
                sparse={7: 1.0},
                tokens=np.ones((1, 4), dtype=np.float32),
            )
        ]


class _SingletonCardinalityEncoder:
    def __init__(self, count):
        self.count = count
        self.healthy = HashEncoder(8, 4, 8)

    def encode(self, texts, is_query=False):
        vector = self.healthy.encode([texts[0]], is_query=is_query)[0]
        return [vector] * self.count


class _SummaryCardinalityEncoder:
    def __init__(self, count):
        self.count = count
        self.healthy = HashEncoder(8, 4, 8)

    def encode(self, texts, is_query=False):
        vectors = self.healthy.encode(texts, is_query=is_query)
        if len(texts) == 1 and texts[0].startswith("[resumo de "):
            return [vectors[0]] * self.count
        return vectors


class _SummaryLLM:
    def available(self):
        return True

    def complete(self, system, messages, max_tokens=1500):
        return "Resumo verificável do documento."


class _OversizedLLM:
    def available(self):
        return True

    def complete(self, *_args, **_kwargs):
        return "x" * (64 * 1024 + 1)


class _CapturingEncoder:
    def __init__(self):
        self.inputs = []
        self.delegate = HashEncoder(8, 4, 8)

    def encode(self, texts, is_query=False):
        self.inputs.extend(texts)
        return self.delegate.encode(texts, is_query=is_query)


class _TransactionProbeStore:
    """Minimal store proving slow preparation stays outside the write transaction."""

    def __init__(self):
        self.in_transaction = False
        self.events = []
        self._next_id = 1

    @contextmanager
    def transaction(self):
        assert self.in_transaction is False
        self.in_transaction = True
        self.events.append("transaction:start")
        try:
            yield self
        finally:
            self.events.append("transaction:end")
            self.in_transaction = False

    def _write(self, name):
        assert self.in_transaction is True
        self.events.append(name)
        value = self._next_id
        self._next_id += 1
        return value

    def add_doc(self, source, title, n_tokens):
        return self._write("add_doc")

    def add_parent(self, doc_id, text, token_count, pos):
        return self._write("add_parent")

    def add_chunk(self, *args, **kwargs):
        return self._write("add_chunk")


class _OutsideTransactionEncoder:
    def __init__(self, store):
        self.store = store
        self.delegate = HashEncoder(8, 4, 8)

    def encode(self, texts, is_query=False):
        assert self.store.in_transaction is False
        self.store.events.append("encode")
        return self.delegate.encode(texts, is_query=is_query)


class _OutsideTransactionLLM:
    def __init__(self, store):
        self.store = store

    def available(self):
        assert self.store.in_transaction is False
        return True

    def complete(self, system, messages, max_tokens=1500):
        assert self.store.in_transaction is False
        self.store.events.append("llm")
        return "Contexto ou resumo verificável."


class _BatchRecordingEncoder:
    def __init__(self):
        self.batch_sizes = []
        self.delegate = HashEncoder(8, 4, 8)

    def encode(self, texts, is_query=False):
        self.batch_sizes.append(len(texts))
        return self.delegate.encode(texts, is_query=is_query)


class _ExplodingEncoder:
    def encode(self, texts, is_query=False):
        raise AssertionError("oversized document must fail before encoding")


class _UnboundedCardinalityEncoder:
    def __init__(self):
        self.consumed = 0
        self.delegate = HashEncoder(8, 4, 8)

    def encode(self, texts, is_query=False):
        vector = self.delegate.encode([texts[0]], is_query=is_query)[0]
        for _ in range(100_000):
            self.consumed += 1
            yield vector


def _cfg(tmp_path, **overrides):
    values = {
        "data_dir": tmp_path,
        "backend": "sqlite",
        "encoder": "hash",
        "dense_dim": 8,
        "colbert_dim": 4,
        "max_colbert_tokens": 8,
        "contextual_enrich": False,
        "tiny_doc_tokens": 500,
    }
    values.update(overrides)
    return TriRagConfig(**values)


def _counts(store):
    return {
        table: store.db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
        for table in ("docs", "chunks", "dvecs", "postings", "colvecs")
    }


def test_ingest_encoder_failure_rolls_back_document_and_connection_is_reusable(tmp_path):
    store = TriStore(tmp_path / "encoder-failure.db")
    cfg = _cfg(tmp_path)
    ingestor = Ingestor(store, _FailingEncoder(), cfg)

    with pytest.raises(RuntimeError, match="encoder failure"):
        ingestor.ingest_text("A document that must not become orphaned.")

    assert _counts(store) == {
        "docs": 0,
        "chunks": 0,
        "dvecs": 0,
        "postings": 0,
        "colvecs": 0,
    }

    healthy = Ingestor(store, HashEncoder(8, 4, 8), cfg)
    result = healthy.ingest_text("A document that must persist.")
    assert result["doc_id"] is not None
    assert store.n_chunks() == 1
    store.close()


def test_ingest_second_chunk_failure_rolls_back_doc_parents_and_first_chunk(
    monkeypatch, tmp_path
):
    store = TriStore(tmp_path / "chunk-failure.db")
    cfg = _cfg(
        tmp_path,
        tiny_doc_tokens=1,
        chunk_tokens=3,
        chunk_overlap=0,
        parent_tokens=6,
    )
    ingestor = Ingestor(store, HashEncoder(8, 4, 8), cfg)
    original_add_chunk = store.add_chunk
    calls = 0

    def fail_second_chunk(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second chunk failure")
        return original_add_chunk(*args, **kwargs)

    monkeypatch.setattr(store, "add_chunk", fail_second_chunk)
    text = (
        "Primeira sentença com conteúdo suficiente. "
        "Segunda sentença com conteúdo suficiente. "
        "Terceira sentença com conteúdo suficiente."
    )

    with pytest.raises(RuntimeError, match="second chunk failure"):
        ingestor.ingest_text(text)

    assert calls >= 2
    assert _counts(store) == {
        "docs": 0,
        "chunks": 0,
        "dvecs": 0,
        "postings": 0,
        "colvecs": 0,
    }

    monkeypatch.setattr(store, "add_chunk", original_add_chunk)
    result = ingestor.ingest_text(text)
    assert result["chunks"] >= 2
    assert store.n_chunks() == result["chunks"]
    store.close()


def test_empty_ingest_remains_a_noop(tmp_path):
    store = TriStore(tmp_path / "empty.db")
    ingestor = Ingestor(store, HashEncoder(8, 4, 8), _cfg(tmp_path))

    assert ingestor.ingest_text(" \n\t ") == {"doc_id": None, "chunks": 0}
    assert _counts(store) == {
        "docs": 0,
        "chunks": 0,
        "dvecs": 0,
        "postings": 0,
        "colvecs": 0,
    }
    store.close()


def test_ingest_rejects_encoder_cardinality_mismatch_without_partial_document(tmp_path):
    store = TriStore(tmp_path / "short-encoder.db")
    cfg = _cfg(
        tmp_path,
        tiny_doc_tokens=1,
        chunk_tokens=3,
        chunk_overlap=0,
        parent_tokens=6,
    )
    ingestor = Ingestor(store, _ShortEncoder(), cfg)
    text = (
        "Primeira sentença com conteúdo suficiente. "
        "Segunda sentença com conteúdo suficiente. "
        "Terceira sentença com conteúdo suficiente."
    )

    with pytest.raises(ValueError, match="one vector|cardinality|count"):
        ingestor.ingest_text(text)

    assert _counts(store) == {
        "docs": 0,
        "chunks": 0,
        "dvecs": 0,
        "postings": 0,
        "colvecs": 0,
    }
    store.close()


@pytest.mark.parametrize("count", [0, 2])
def test_tiny_ingest_requires_exactly_one_vector_and_rolls_back(tmp_path, count):
    store = TriStore(tmp_path / f"tiny-cardinality-{count}.db")
    ingestor = Ingestor(
        store,
        _SingletonCardinalityEncoder(count),
        _cfg(tmp_path, tiny_doc_tokens=500),
    )

    with pytest.raises(ValueError, match="exactly one|cardinality|count"):
        ingestor.ingest_text("Tiny document with one embedding request.")

    assert _counts(store) == {
        "docs": 0,
        "chunks": 0,
        "dvecs": 0,
        "postings": 0,
        "colvecs": 0,
    }
    store.close()


def test_ingest_stops_unbounded_encoder_output_at_expected_cardinality_plus_one(
    tmp_path,
):
    store = TriStore(tmp_path / "unbounded-cardinality.db")
    encoder = _UnboundedCardinalityEncoder()
    ingestor = Ingestor(store, encoder, _cfg(tmp_path, tiny_doc_tokens=500))

    with pytest.raises(ValueError, match="exactly one vector"):
        ingestor.ingest_text("Tiny document with one embedding request.")

    assert encoder.consumed == 2
    assert _counts(store) == {
        "docs": 0,
        "chunks": 0,
        "dvecs": 0,
        "postings": 0,
        "colvecs": 0,
    }
    store.close()


@pytest.mark.parametrize("count", [0, 2])
def test_summary_ingest_requires_exactly_one_vector_and_rolls_back(tmp_path, count):
    store = TriStore(tmp_path / f"summary-cardinality-{count}.db")
    cfg = _cfg(
        tmp_path,
        tiny_doc_tokens=1,
        huge_doc_tokens=1,
        chunk_tokens=4,
        chunk_overlap=0,
        parent_tokens=8,
    )
    ingestor = Ingestor(
        store,
        _SummaryCardinalityEncoder(count),
        cfg,
        _SummaryLLM(),
    )
    text = (
        "Primeira sentença extensa para criar um chunk. "
        "Segunda sentença extensa para criar outro chunk."
    )

    with pytest.raises(ValueError, match="exactly one|cardinality|count"):
        ingestor.ingest_text(text)

    assert _counts(store) == {
        "docs": 0,
        "chunks": 0,
        "dvecs": 0,
        "postings": 0,
        "colvecs": 0,
    }
    store.close()


def test_llm_and_encoder_work_finish_before_the_write_transaction(tmp_path):
    store = _TransactionProbeStore()
    cfg = _cfg(
        tmp_path,
        tiny_doc_tokens=1,
        huge_doc_tokens=1,
        chunk_tokens=4,
        chunk_overlap=0,
        parent_tokens=8,
        contextual_enrich=True,
    )
    ingestor = Ingestor(
        store,
        _OutsideTransactionEncoder(store),
        cfg,
        _OutsideTransactionLLM(store),
    )

    result = ingestor.ingest_text(
        "Primeira sentença extensa para criar um chunk. "
        "Segunda sentença extensa para criar outro chunk."
    )

    assert result["chunks"] >= 2
    transaction_start = store.events.index("transaction:start")
    assert all(
        event in {"encode", "llm"}
        for event in store.events[:transaction_start]
    )
    assert "encode" in store.events[:transaction_start]
    assert "llm" in store.events[:transaction_start]
    assert all(
        event in {"add_doc", "add_parent", "add_chunk", "transaction:end"}
        for event in store.events[transaction_start + 1 :]
    )


def test_chunk_embeddings_are_batched_outside_the_transaction(tmp_path):
    store = TriStore(tmp_path / "batched-embeddings.db")
    encoder = _BatchRecordingEncoder()
    cfg = _cfg(
        tmp_path,
        tiny_doc_tokens=1,
        chunk_tokens=1,
        chunk_overlap=0,
        parent_tokens=8,
    )
    ingestor = Ingestor(store, encoder, cfg)
    text = " ".join(f"Sentença {index}." for index in range(MAX_EMBEDDING_BATCH * 2 + 3))

    result = ingestor.ingest_text(text)

    assert result["chunks"] == MAX_EMBEDDING_BATCH * 2 + 3
    assert len(encoder.batch_sizes) >= 3
    assert max(encoder.batch_sizes) <= MAX_EMBEDDING_BATCH
    store.close()


def test_oversized_chunk_count_fails_before_encoder_or_database_write(tmp_path):
    store = TriStore(tmp_path / "oversized-chunks.db")
    cfg = _cfg(
        tmp_path,
        tiny_doc_tokens=1,
        chunk_tokens=1,
        chunk_overlap=0,
    )
    ingestor = Ingestor(store, _ExplodingEncoder(), cfg)
    text = " ".join(f"Sentença {index}." for index in range(MAX_INGEST_CHUNKS + 1))

    with pytest.raises(ValueError, match="chunks.*maximum"):
        ingestor.ingest_text(text)

    assert _counts(store) == {
        "docs": 0,
        "chunks": 0,
        "dvecs": 0,
        "postings": 0,
        "colvecs": 0,
    }
    store.close()


def test_document_byte_limit_applies_before_normalization_and_file_read(tmp_path):
    store = TriStore(tmp_path / "oversized-document.db")
    ingestor = Ingestor(store, _ExplodingEncoder(), _cfg(tmp_path))

    with pytest.raises(ValueError, match="document.*bytes"):
        ingestor.ingest_text("x" * (MAX_INGEST_DOCUMENT_BYTES + 1))

    document = tmp_path / "too-large.txt"
    with document.open("wb") as handle:
        handle.seek(MAX_INGEST_DOCUMENT_BYTES)
        handle.write(b"x")
    with pytest.raises(ValueError, match="document.*bytes"):
        ingestor.ingest_file(document)

    assert _counts(store) == {
        "docs": 0,
        "chunks": 0,
        "dvecs": 0,
        "postings": 0,
        "colvecs": 0,
    }
    store.close()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("title", "t" * (MAX_TEST_INGEST_LABEL_BYTES + 1), "title.*bytes"),
        ("source", "s" * (MAX_TEST_INGEST_LABEL_BYTES + 1), "source.*bytes"),
    ],
)
def test_ingest_labels_are_bounded_before_encoder_or_database_write(
    tmp_path, field, value, message
):
    store = TriStore(tmp_path / f"oversized-{field}.db")
    ingestor = Ingestor(store, _ExplodingEncoder(), _cfg(tmp_path))
    kwargs = {field: value}

    with pytest.raises(ValueError, match=message):
        ingestor.ingest_text("texto curto", **kwargs)

    assert _counts(store) == {
        "docs": 0,
        "chunks": 0,
        "dvecs": 0,
        "postings": 0,
        "colvecs": 0,
    }
    store.close()


def test_oversized_context_enrichment_is_discarded_before_encoding(tmp_path):
    store = TriStore(tmp_path / "oversized-enrichment.db")
    encoder = _CapturingEncoder()
    cfg = _cfg(
        tmp_path,
        tiny_doc_tokens=1,
        chunk_tokens=4,
        chunk_overlap=0,
        contextual_enrich=True,
        huge_doc_tokens=1_000_000,
    )
    ingestor = Ingestor(store, encoder, cfg, _OversizedLLM())

    result = ingestor.ingest_text(
        "Primeira sentença curta. Segunda sentença curta.", title="bounded"
    )

    assert result["chunks"] >= 1
    assert encoder.inputs
    assert all(len(value) <= 64 * 1024 for value in encoder.inputs)
    assert all("x" * 100 not in value for value in encoder.inputs)
    store.close()


def test_oversized_summary_is_omitted_before_encoding_or_persistence(tmp_path):
    store = TriStore(tmp_path / "oversized-summary.db")
    encoder = _CapturingEncoder()
    cfg = _cfg(
        tmp_path,
        tiny_doc_tokens=1,
        chunk_tokens=4,
        chunk_overlap=0,
        contextual_enrich=False,
        huge_doc_tokens=1,
    )
    ingestor = Ingestor(store, encoder, cfg, _OversizedLLM())

    result = ingestor.ingest_text(
        "Primeira sentença curta. Segunda sentença curta.", title="bounded"
    )

    assert result["chunks"] >= 1
    assert all(len(value) <= 64 * 1024 for value in encoder.inputs)
    assert not store.all_texts(kinds=("summary",))
    store.close()

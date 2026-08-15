from __future__ import annotations

import math
import sqlite3
from collections.abc import Mapping

import numpy as np
import pytest

from rag3d.backend import DEFAULT_RETRIEVAL_LIMITS
from rag3d.encoders import HashEncoder, TriVec
from rag3d.pgstore import PgHoloStore
from rag3d.store import TriStore


class _OversizedExplodingMapping(Mapping[int, float]):
    def __getitem__(self, key: int) -> float:
        raise KeyError(key)

    def __iter__(self):
        raise AssertionError("oversized mapping was iterated")

    def __len__(self) -> int:
        return 8_193

    def items(self):
        raise AssertionError("oversized mapping items were requested")


class _LyingLengthMapping(Mapping[int, float]):
    def __init__(self) -> None:
        self.yielded = 0

    def __getitem__(self, key: int) -> float:
        if 0 <= key <= 8_192:
            return 1.0
        raise KeyError(key)

    def __iter__(self):
        return iter(range(8_193))

    def __len__(self) -> int:
        return 1

    def items(self):
        for term in range(8_193):
            self.yielded += 1
            yield term, 1.0


def _vector(sparse: Mapping[int, float]) -> TriVec:
    return TriVec(
        dense=np.asarray([1.0, 0.0], dtype=np.float32),
        sparse=dict(sparse),
        tokens=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )


def _store_with_postings(tmp_path) -> tuple[TriStore, list[int]]:
    store = TriStore(tmp_path / "sparse-batch.db")
    with store.transaction():
        document_id = store.add_doc("fixture", "fixture", 4)
        chunk_ids = [
            store.add_chunk(document_id, "alpha", "", 1, _vector({10: 1.0, 20: 0.5})),
            store.add_chunk(document_id, "beta", "", 1, _vector({10: 1.0, 30: 2.0})),
            store.add_chunk(document_id, "gamma", "", 1, _vector({20: 1.5, 30: 0.25})),
            store.add_chunk(document_id, "tie", "", 1, _vector({10: 1.0})),
        ]
    return store, chunk_ids


def _legacy_reference(
    store: TriStore, query: Mapping[int, float], k: int
) -> list[tuple[int, float]]:
    """Frozen pre-optimization semantics, including float accumulation order."""
    n_docs = max(1, store.n_chunks())
    scores: dict[int, float] = {}
    for term, query_weight in query.items():
        rows = store.db.execute(
            "SELECT p.chunk_id, p.weight FROM postings p JOIN chunks c ON c.id=p.chunk_id"
            " WHERE p.term=? AND c.kind IN ('chunk','turn','summary')",
            (term,),
        ).fetchall()
        document_frequency = len(rows)
        if document_frequency == 0:
            continue
        inverse_document_frequency = float(
            np.log(
                1.0
                + (n_docs - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
        )
        for chunk_id, document_weight in rows:
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + query_weight * document_weight * inverse_document_frequency
            )
    return [
        (int(chunk_id), float(score))
        for chunk_id, score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:k]
    ]


def test_sparse_search_aggregates_and_limits_in_sql_beyond_variable_limit(
    tmp_path,
) -> None:
    store, _ = _store_with_postings(tmp_path)
    query = {term: 0.25 + term / 100.0 for term in range(1, 26)}
    expected = _legacy_reference(store, query, 4)

    previous_limit = store.db.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 7)
    statements: list[str] = []
    store.db.set_trace_callback(statements.append)
    try:
        actual = store.sparse_search(query, 4)
    finally:
        store.db.set_trace_callback(None)
        store.db.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous_limit)

    selects = [statement for statement in statements if statement.lstrip().upper().startswith("SELECT")]
    posting_selects = [statement for statement in selects if "JOIN postings" in statement]
    assert actual == expected
    assert len(posting_selects) == 1
    normalized = " ".join(posting_selects[0].split()).upper()
    assert "GROUP BY" in normalized
    assert "ORDER BY SCORE DESC" in normalized
    assert "LIMIT 4" in normalized
    assert len(selects) == 1


def test_sparse_search_matches_legacy_math_ties_and_order_exactly(tmp_path) -> None:
    store, chunk_ids = _store_with_postings(tmp_path)
    query = {30: 0.75, 10: 1.25, 20: -0.5, 999: 9.0}

    actual = store.sparse_search(query, 10)

    assert actual == _legacy_reference(store, query, 10)
    assert all(math.isfinite(score) for _, score in actual)
    tie_query = {10: 1.0}
    tie_result = store.sparse_search(tie_query, 10)
    assert tie_result == _legacy_reference(store, tie_query, 10)
    assert [chunk_id for chunk_id, _ in tie_result] == [
        chunk_ids[0],
        chunk_ids[1],
        chunk_ids[3],
    ]


def test_sparse_sql_aggregation_preserves_legacy_float_accumulation_order(
    tmp_path,
) -> None:
    rng = np.random.default_rng(20260814)
    store = TriStore(tmp_path / "sparse-random-exact.db")
    with store.transaction():
        document_id = store.add_doc("fixture", "fixture", 20)
        for position in range(20):
            terms = {
                int(term): float(weight)
                for term, weight in zip(
                    rng.choice(15, size=6, replace=False), rng.normal(size=6)
                )
            }
            store.add_chunk(
                document_id,
                f"chunk-{position}",
                "",
                1,
                _vector(terms),
                pos=position,
            )

    for _ in range(12):
        query = {
            int(term): float(weight)
            for term, weight in zip(
                rng.choice(18, size=8, replace=False), rng.normal(size=8)
            )
        }
        assert store.sparse_search(query, 17) == _legacy_reference(store, query, 17)
    store.close()


def test_sparse_search_includes_summary_but_excludes_orphan_postings(tmp_path) -> None:
    store = TriStore(tmp_path / "sparse-filtered-universe.db")
    with store.transaction():
        document_id = store.add_doc("fixture", "fixture", 2)
        store.add_chunk(
            document_id, "chunk", "", 1, _vector({10: 1.0}), kind="chunk"
        )
        summary_id = store.add_chunk(
            document_id,
            "summary",
            "",
            1,
            _vector({20: 2.0}),
            kind="summary",
            pos=-1,
        )

    # Corrupt this disposable fixture after ingestion to prove the defensive
    # JOIN excludes an orphan from both DF and scored rows. Normal stores cannot
    # create this state because both a declarative FK and a trigger reject it.
    store.db.execute("DROP TRIGGER rag3d_postings_chunk_insert_fk")
    store.db.commit()
    store.db.execute("PRAGMA foreign_keys=OFF")
    store.db.execute("INSERT INTO postings VALUES(20,?,0.0)", (summary_id,))
    store.db.execute("INSERT INTO postings VALUES(20,999999999,100.0)")
    store.db.commit()

    statements: list[str] = []
    store.db.set_trace_callback(statements.append)
    try:
        result = store.sparse_search({20: 1.0}, 10)
    finally:
        store.db.set_trace_callback(None)

    assert [candidate for candidate, _ in result] == [summary_id]
    assert result[0][1] == pytest.approx(2.0 * math.log(2.0))
    sparse_sql = next(
        statement for statement in statements if "JOIN postings" in statement
    )
    assert "COUNT(DISTINCT p2.chunk_id)" in sparse_sql
    store.close()


def test_sparse_search_zero_and_empty_do_not_query_postings(tmp_path) -> None:
    store, _ = _store_with_postings(tmp_path)
    statements: list[str] = []
    store.db.set_trace_callback(statements.append)
    try:
        assert store.sparse_search({10: 1.0}, 0) == []
        assert store.sparse_search({}, 10) == []
    finally:
        store.db.set_trace_callback(None)

    assert not any("FROM postings" in statement for statement in statements)


def test_sparse_search_rejects_unbounded_query_terms_before_sql(tmp_path) -> None:
    store, _ = _store_with_postings(tmp_path)
    oversized = {
        term: 1.0
        for term in range(DEFAULT_RETRIEVAL_LIMITS.max_sparse_terms + 1)
    }
    statements: list[str] = []
    store.db.set_trace_callback(statements.append)
    try:
        with pytest.raises(ValueError, match="sparse terms exceed maximum of 8192"):
            store.sparse_search(oversized, 10)
    finally:
        store.db.set_trace_callback(None)

    assert not any("FROM postings" in statement for statement in statements)


@pytest.mark.parametrize("validator", [TriStore._sparse_input, PgHoloStore._sparse_input])
def test_sparse_limit_preflights_len_before_items(validator) -> None:
    weights = _OversizedExplodingMapping()

    with pytest.raises(ValueError, match="sparse terms exceed maximum of 8192"):
        validator(weights)


@pytest.mark.parametrize("validator", [TriStore._sparse_input, PgHoloStore._sparse_input])
def test_sparse_limit_counts_items_from_a_lying_mapping(validator) -> None:
    weights = _LyingLengthMapping()

    with pytest.raises(ValueError, match="sparse terms exceed maximum of 8192"):
        validator(weights)

    assert weights.yielded == DEFAULT_RETRIEVAL_LIMITS.max_sparse_terms + 1


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo"])
@pytest.mark.parametrize(
    "mapping_factory", [_OversizedExplodingMapping, _LyingLengthMapping]
)
def test_sparse_ingest_and_search_use_the_same_limit(
    tmp_path, adapter, mapping_factory
) -> None:
    if adapter == "sqlite":
        store = TriStore(tmp_path / f"{mapping_factory.__name__}.db")
    else:
        store = PgHoloStore.__new__(PgHoloStore)
        store._dense_dim = 2
        store._colbert_dim = 2
    vector = TriVec(
        dense=np.asarray([1.0, 0.0], dtype=np.float32),
        sparse=mapping_factory(),  # type: ignore[arg-type]
        tokens=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError) as search_error:
        store.sparse_search(mapping_factory(), 1)
    with pytest.raises(ValueError) as ingest_error:
        store.add_chunk(None, "oversized", "", 1, vector)

    assert str(search_error.value) == str(ingest_error.value)
    assert str(search_error.value) == "sparse terms exceed maximum of 8192"


def test_hash_encoder_sparse_vector_with_1100_terms_remains_accepted() -> None:
    encoder = HashEncoder(dense_dim=8, colbert_dim=4, max_tokens=256)
    text = " ".join(f"unique_term_{term}" for term in range(1_100))
    vector = encoder.encode([text], is_query=True)[0]

    assert len(vector.sparse) == 1_100
    assert vector.tokens.shape[0] == 256
    assert TriStore._sparse_input(vector.sparse) == vector.sparse
    assert PgHoloStore._sparse_input(vector.sparse) == vector.sparse


def test_sparse_search_rejects_non_integer_terms_without_sql_interpolation(tmp_path) -> None:
    store, _ = _store_with_postings(tmp_path)
    statements: list[str] = []
    store.db.set_trace_callback(statements.append)
    try:
        with pytest.raises(TypeError, match="term IDs"):
            store.sparse_search({"1) OR 1=1 --": 1.0}, 10)  # type: ignore[dict-item]
    finally:
        store.db.set_trace_callback(None)

    assert not any("FROM postings" in statement for statement in statements)

from __future__ import annotations

import math
import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from rag3d.backend import DEFAULT_RETRIEVAL_LIMITS
from rag3d.encoders import TriVec
from rag3d.holo import Holographer
from rag3d.pgstore import PgHoloStore
from rag3d.pgvector_store import (
    PgVectorStore,
    _validate_ann_options,
    _validate_sparse_weights,
)
from rag3d.store import (
    TriStore,
    _SQLITE_HISTORICAL_VARIABLE_LIMIT,
    _sqlite_variable_limit,
)


class _Rows:
    def __init__(self, *, one=(1,), rows=()):
        self._one = one
        self._rows = list(rows)

    def fetchone(self):
        return self._one

    def fetchall(self):
        return list(self._rows)


class _RecordingDatabase:
    def __init__(self):
        self.calls = []
        self.commits = 0

    @contextmanager
    def transaction(self):
        yield

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        return _Rows()

    def commit(self):
        self.commits += 1

    def close(self):
        pass


class _OversizedExplodingSequence(Sequence[int]):
    def __init__(self) -> None:
        self.iterated = False

    def __len__(self) -> int:
        return DEFAULT_RETRIEVAL_LIMITS.max_pool + 1

    def __getitem__(self, index):
        raise AssertionError("oversized sequence was indexed")

    def __iter__(self) -> Iterator[int]:
        self.iterated = True
        raise AssertionError("oversized sequence was iterated")


class _LyingLengthSequence(Sequence[int]):
    def __init__(self) -> None:
        self.yielded = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index):
        if 0 <= index < DEFAULT_RETRIEVAL_LIMITS.max_pool + 100:
            return index
        raise IndexError(index)

    def __iter__(self) -> Iterator[int]:
        for value in range(DEFAULT_RETRIEVAL_LIMITS.max_pool + 100):
            self.yielded += 1
            yield value


class _LyingDocumentMetadata(Mapping[str, int]):
    def __init__(self) -> None:
        self.yielded = 0

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[str]:
        for index in range(DEFAULT_RETRIEVAL_LIMITS.max_metadata_json_bytes + 100):
            self.yielded += 1
            yield f"key_{index}"

    def __getitem__(self, key: str) -> int:
        return int(key.removeprefix("key_"))

    def items(self):
        for index in range(DEFAULT_RETRIEVAL_LIMITS.max_metadata_json_bytes + 100):
            self.yielded += 1
            yield f"key_{index}", index


class _OversizedExplodingTokenRows(Sequence[list[float]]):
    def __init__(self, size: int = 8_193) -> None:
        self.size = size
        self.iterated = False

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index):
        raise AssertionError("oversized structural rows were indexed")

    def __iter__(self):
        self.iterated = True
        raise AssertionError("oversized structural rows were iterated")


class _LyingLengthTokenRows(Sequence[list[float]]):
    def __init__(self, rows: int = 3) -> None:
        self.rows = rows
        self.yielded = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index):
        if 0 <= index < self.rows:
            return [1.0, 0.0]
        raise IndexError(index)

    def __iter__(self):
        for _ in range(self.rows):
            self.yielded += 1
            yield [1.0, 0.0]


class _OversizedExplodingDenseVector(Sequence[float]):
    def __init__(self) -> None:
        self.iterated = False

    def __len__(self) -> int:
        return DEFAULT_RETRIEVAL_LIMITS.max_dense_dim + 1

    def __getitem__(self, index):
        raise AssertionError("oversized dense vector was indexed")

    def __iter__(self):
        self.iterated = True
        raise AssertionError("oversized dense vector was iterated")


class _OversizedExplodingStructuralRow(Sequence[float]):
    def __init__(self) -> None:
        self.iterated = False

    def __len__(self) -> int:
        return DEFAULT_RETRIEVAL_LIMITS.max_structural_dim + 1

    def __getitem__(self, index):
        raise AssertionError("oversized structural row was indexed")

    def __iter__(self):
        self.iterated = True
        raise AssertionError("oversized structural row was iterated")


def _vec(*, sparse=None) -> TriVec:
    return TriVec(
        dense=np.asarray([1.0, 0.0], dtype=np.float32),
        sparse={7: 1.0} if sparse is None else sparse,
        tokens=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )


def _make_store(adapter: str, tmp_path):
    if adapter == "sqlite":
        store = TriStore(tmp_path / "boundary.db")
        statements = []
        store.db.set_trace_callback(statements.append)
        return store, statements
    database = _RecordingDatabase()
    if adapter == "postgres-holo":
        store = PgHoloStore.__new__(PgHoloStore)
        store.db = database
        store._dense_dim = 2
        store._colbert_dim = 2
        store._transaction_depth = 0
        store.holo = Holographer(2, 2)
        return store, database.calls
    store = object.__new__(PgVectorStore)
    store.db = database
    store.dense_dim = 2
    store.colbert_dim = 2
    store._fingerprint_verified = True
    store._max_structural_tokens = DEFAULT_RETRIEVAL_LIMITS.max_structural_tokens
    store._query_max_tokens = DEFAULT_RETRIEVAL_LIMITS.max_structural_tokens
    store.search_mode = "exact"
    return store, database.calls


def _close_if_sqlite(adapter: str, store: Any) -> None:
    if adapter == "sqlite":
        store.db.set_trace_callback(None)
        store.close()


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo", "pgvector"])
@pytest.mark.parametrize("method_name", ["delete_chunk", "delete_document"])
def test_delete_rejects_boolean_ids_without_mutating_id_one(
    tmp_path, adapter: str, method_name: str
) -> None:
    store, calls = _make_store(adapter, tmp_path)
    try:
        if adapter == "sqlite":
            with store.transaction():
                document_id = store.add_doc("source", "title", 1)
                chunk_id = store.add_chunk(document_id, "text", "text", 1, _vec())
            assert (document_id, chunk_id) == (1, 1)
            calls.clear()

        with pytest.raises(TypeError, match="integer|bool"):
            getattr(store, method_name)(True)

        if adapter == "sqlite":
            assert store.db.execute("SELECT COUNT(*) FROM docs WHERE id=1").fetchone()[0] == 1
            assert store.db.execute("SELECT COUNT(*) FROM chunks WHERE id=1").fetchone()[0] == 1
        else:
            assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo", "pgvector"])
@pytest.mark.parametrize("sequence_type", [_OversizedExplodingSequence, _LyingLengthSequence])
def test_chunk_id_sequences_are_bounded_while_they_are_consumed(
    tmp_path, adapter: str, sequence_type
) -> None:
    store, calls = _make_store(adapter, tmp_path)
    values = sequence_type()
    try:
        with pytest.raises(ValueError, match="maximum|more than|pool"):
            store.get_chunks(values)
        if isinstance(values, _OversizedExplodingSequence):
            assert values.iterated is False
        else:
            assert values.yielded == DEFAULT_RETRIEVAL_LIMITS.max_pool + 1
        assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo", "pgvector"])
@pytest.mark.parametrize("sequence_type", [_OversizedExplodingSequence, _LyingLengthSequence])
def test_neighbor_positions_are_bounded_while_they_are_consumed(
    tmp_path, adapter: str, sequence_type
) -> None:
    store, calls = _make_store(adapter, tmp_path)
    values = sequence_type()
    try:
        with pytest.raises(ValueError, match="maximum|more than"):
            store.neighbors(7, values)
        if isinstance(values, _OversizedExplodingSequence):
            assert values.iterated is False
        else:
            assert values.yielded == DEFAULT_RETRIEVAL_LIMITS.max_pool + 1
        assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


def test_sqlite_batches_all_id_and_position_apis_under_variable_limit(tmp_path) -> None:
    store = TriStore(tmp_path / "sqlite-batches.db")
    with store.transaction():
        document_id = store.add_doc("source", "title", 25)
        chunk_ids = [
            store.add_chunk(
                document_id,
                f"chunk {position}",
                f"chunk {position}",
                1,
                _vec(),
                pos=position,
            )
            for position in range(25)
        ]

    previous_limit = store.db.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 7)
    requested = list(reversed(chunk_ids))
    try:
        assert [row["id"] for row in store.get_chunks(requested)] == requested
        assert set(store.dense_vecs(requested)) == set(chunk_ids)
        store.touch_access(requested, 9)
        assert store.db.execute(
            "SELECT COUNT(*) FROM chunks WHERE accessed_turn=9"
        ).fetchone()[0] == 25
        structural = store.colbert_scores(
            np.asarray([[1.0, 0.0]], dtype=np.float32), requested
        )
        assert [chunk_id for chunk_id, _ in structural] == chunk_ids
        neighbors = store.neighbors(document_id, list(reversed(range(25))))
        assert [row["pos"] for row in neighbors] == list(range(25))
    finally:
        store.db.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous_limit)
        store.close()


def test_sqlite_variable_limit_falls_back_when_getlimit_is_missing() -> None:
    class _LegacyConnection:
        pass

    class _ModernConnection:
        def getlimit(self, limit: int) -> int:
            assert limit == sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER
            return 7

    assert _sqlite_variable_limit(_LegacyConnection()) == (
        _SQLITE_HISTORICAL_VARIABLE_LIMIT
    )
    assert _sqlite_variable_limit(_ModernConnection()) == 7


def test_sqlite_batches_fall_back_to_historical_limit_without_getlimit(
    tmp_path,
) -> None:
    store = TriStore(tmp_path / "legacy-sqlite-limit.db")

    class _NoGetlimitConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def __getattr__(self, name: str):
            if name == "getlimit":
                raise AttributeError(name)
            return getattr(self._connection, name)

    store.db = _NoGetlimitConnection(store.db)
    values = list(range(1_050))
    try:
        batches = list(store._sqlite_batches(values, reserved_variables=1))
        assert max(len(batch) for batch in batches) == (
            _SQLITE_HISTORICAL_VARIABLE_LIMIT - 1
        )
        assert [item for batch in batches for item in batch] == values
    finally:
        store.close()


_INVALID_CHUNK_CASES = [
    pytest.param({"kind": "evil"}, ValueError, id="unknown-kind"),
    pytest.param({"kind": "parent"}, ValueError, id="parent-through-add-chunk"),
    pytest.param({"kind": "chunk", "pos": -1}, ValueError, id="negative-chunk-pos"),
    pytest.param(
        {"kind": "rolling_summary", "pos": -1},
        ValueError,
        id="negative-rolling-summary-pos",
    ),
    pytest.param({"kind": "summary", "pos": -2}, ValueError, id="summary-below-minus-one"),
    pytest.param({"n_tokens": True}, TypeError, id="boolean-token-count"),
    pytest.param({"n_tokens": -1}, ValueError, id="negative-token-count"),
    pytest.param({"n_tokens": 2**31}, ValueError, id="token-count-int32-overflow"),
    pytest.param({"importance": True}, TypeError, id="boolean-importance"),
    pytest.param({"importance": math.nan}, ValueError, id="nan-importance"),
    pytest.param({"importance": -0.1}, ValueError, id="negative-importance"),
    pytest.param({"importance": 1.1}, ValueError, id="importance-above-one"),
    pytest.param({"doc_id": True}, TypeError, id="boolean-document-id"),
    pytest.param({"parent_id": True}, TypeError, id="boolean-parent-id"),
    pytest.param({"turn_no": -1}, ValueError, id="negative-turn-number"),
]


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo", "pgvector"])
@pytest.mark.parametrize("overrides,error_type", _INVALID_CHUNK_CASES)
def test_chunk_scalar_validation_is_uniform_before_database_io(
    tmp_path, adapter: str, overrides, error_type
) -> None:
    store, calls = _make_store(adapter, tmp_path)
    arguments = {
        "doc_id": None,
        "text": "text",
        "ctx": "context",
        "n_tokens": 1,
        "vec": _vec(),
        "kind": "chunk",
        "pos": 0,
        "parent_id": None,
        "importance": 0.5,
        "turn_no": None,
    }
    arguments.update(overrides)
    try:
        with pytest.raises(error_type):
            store.add_chunk(**arguments)
        assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo", "pgvector"])
def test_summary_minus_one_remains_a_valid_stored_position(tmp_path, adapter: str) -> None:
    store, _calls = _make_store(adapter, tmp_path)
    try:
        chunk_id = store.add_chunk(
            None,
            "summary",
            "summary",
            1,
            _vec(),
            kind="summary",
            pos=-1,
        )
        assert chunk_id == 1
        if adapter == "sqlite":
            assert store.db.execute(
                "SELECT kind,pos FROM chunks WHERE id=?", (chunk_id,)
            ).fetchone() == ("summary", -1)
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo", "pgvector"])
@pytest.mark.parametrize(
    "method_name,args,error_type",
    [
        ("add_doc", ("source", "title", -1), ValueError),
        ("add_doc", ("source", "title", True), TypeError),
        ("add_doc", ("source", "title", 2**31), ValueError),
        ("add_parent", (True, "parent", 1, 0), TypeError),
        ("add_parent", (1, "parent", -1, 0), ValueError),
        ("add_parent", (1, "parent", 1, -1), ValueError),
        ("add_parent", (1, "parent", 1, True), TypeError),
        ("add_parent", (1, "parent", 1, 2**31), ValueError),
    ],
)
def test_document_and_parent_scalar_validation_is_uniform(
    tmp_path, adapter: str, method_name: str, args, error_type
) -> None:
    store, calls = _make_store(adapter, tmp_path)
    try:
        with pytest.raises(error_type):
            getattr(store, method_name)(*args)
        assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize(
    "validator",
    [TriStore._sparse_input, PgHoloStore._sparse_input, _validate_sparse_weights],
)
@pytest.mark.parametrize("term", [-(2**63) - 1, 2**63])
def test_sparse_terms_must_fit_a_signed_bigint_on_every_adapter(validator, term) -> None:
    with pytest.raises(ValueError, match="bigint"):
        validator({term: 1.0})


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo", "pgvector"])
@pytest.mark.parametrize("method_name", ["corpus_tokens", "all_texts"])
def test_kind_collections_reject_unknown_values_before_database_io(
    tmp_path, adapter: str, method_name: str
) -> None:
    store, calls = _make_store(adapter, tmp_path)
    try:
        with pytest.raises(ValueError, match="kind"):
            getattr(store, method_name)(("evil",))
        assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo", "pgvector"])
def test_protocol_search_keywords_work_on_every_adapter(tmp_path, adapter: str) -> None:
    store, calls = _make_store(adapter, tmp_path)
    try:
        assert store.dense_search(
            qvec=np.asarray([1.0, 0.0], dtype=np.float32), k=0
        ) == []
        assert store.sparse_search(qsparse={}, k=0) == []
        assert store.structural_rerank(
            qtokens=np.empty((0, 2), dtype=np.float32),
            candidate_ids=[],
            k=0,
        ) == []
        assert store.colbert_scores(
            qtokens=np.empty((0, 2), dtype=np.float32), candidate_ids=[]
        ) == []
        assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo", "pgvector"])
@pytest.mark.parametrize(
    "operation,error_type",
    [
        ("dense-invalid-filters", TypeError),
        ("dense-invalid-exact", TypeError),
        ("sparse-empty-invalid-filters", TypeError),
        ("sparse-boolean-term-zero-k", TypeError),
        ("structural-boolean-id-zero-k", TypeError),
        ("structural-empty-invalid-filters", TypeError),
    ],
)
def test_search_inputs_are_validated_before_empty_or_zero_short_circuits(
    tmp_path, adapter: str, operation: str, error_type
) -> None:
    store, calls = _make_store(adapter, tmp_path)
    empty_tokens = np.empty((0, 2), dtype=np.float32)
    try:
        with pytest.raises(error_type):
            if operation == "dense-invalid-filters":
                store.dense_search(np.asarray([1.0, 0.0]), 0, filters="bad")
            elif operation == "dense-invalid-exact":
                store.dense_search(np.asarray([1.0, 0.0]), 0, exact="bad")
            elif operation == "sparse-empty-invalid-filters":
                store.sparse_search({}, 1, filters="bad")
            elif operation == "sparse-boolean-term-zero-k":
                store.sparse_search({True: 1.0}, 0)
            elif operation == "structural-boolean-id-zero-k":
                store.structural_rerank(empty_tokens, [True], 0)
            else:
                store.structural_rerank(empty_tokens, [], 0, filters="bad")
        assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo"])
@pytest.mark.parametrize("operation", ["ingest", "query"])
def test_structural_token_rows_preflight_sized_inputs_before_iteration(
    tmp_path, adapter: str, operation: str
) -> None:
    store, _calls = _make_store(adapter, tmp_path)
    store._max_structural_tokens = 2
    rows = _OversizedExplodingTokenRows(size=3)
    try:
        with pytest.raises(ValueError, match="structural token rows exceed maximum of 2"):
            if operation == "ingest":
                store.add_chunk(
                    None,
                    "text",
                    "context",
                    1,
                    TriVec(
                        dense=np.asarray([1.0, 0.0], dtype=np.float32),
                        sparse={7: 1.0},
                        tokens=rows,  # type: ignore[arg-type]
                    ),
                )
            else:
                store.colbert_scores(rows, [])  # type: ignore[arg-type]
        assert rows.iterated is False
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo"])
@pytest.mark.parametrize("operation", ["ingest", "query"])
def test_structural_token_rows_count_a_lying_sized_input_during_consumption(
    tmp_path, adapter: str, operation: str
) -> None:
    store, _calls = _make_store(adapter, tmp_path)
    store._max_structural_tokens = 2
    rows = _LyingLengthTokenRows(rows=3)
    try:
        with pytest.raises(ValueError, match="structural token rows exceed maximum of 2"):
            if operation == "ingest":
                store.add_chunk(
                    None,
                    "text",
                    "context",
                    1,
                    TriVec(
                        dense=np.asarray([1.0, 0.0], dtype=np.float32),
                        sparse={7: 1.0},
                        tokens=rows,  # type: ignore[arg-type]
                    ),
                )
            else:
                store.colbert_scores(rows, [])  # type: ignore[arg-type]
        assert rows.yielded == 3
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo"])
@pytest.mark.parametrize("operation", ["ingest", "query"])
def test_structural_value_budget_is_enforced_before_database_io(
    tmp_path, adapter: str, operation: str
) -> None:
    width = 512
    rows = DEFAULT_RETRIEVAL_LIMITS.max_structural_values // width + 1
    vectors = np.ones((rows, width), dtype=np.float32)
    assert vectors.shape[0] <= 8_192
    assert vectors.size > DEFAULT_RETRIEVAL_LIMITS.max_structural_values

    store, calls = _make_store(adapter, tmp_path)
    if adapter == "postgres-holo":
        store._colbert_dim = width
        store.holo = Holographer(2, width)
    try:
        with pytest.raises(
            ValueError,
            match=(
                "structural vector values exceed maximum of "
                f"{DEFAULT_RETRIEVAL_LIMITS.max_structural_values}"
            ),
        ):
            if operation == "ingest":
                store.add_chunk(
                    None,
                    "text",
                    "context",
                    1,
                    TriVec(
                        dense=np.asarray([1.0, 0.0], dtype=np.float32),
                        sparse={7: 1.0},
                        tokens=vectors,
                    ),
                )
            else:
                store.colbert_scores(vectors, [])
        assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo"])
def test_dense_dimension_is_preflighted_before_iteration_or_database_io(
    tmp_path, adapter: str
) -> None:
    store, calls = _make_store(adapter, tmp_path)
    dense = _OversizedExplodingDenseVector()
    try:
        with pytest.raises(
            ValueError,
            match=(
                "dense vector dimension exceeds maximum of "
                f"{DEFAULT_RETRIEVAL_LIMITS.max_dense_dim}"
            ),
        ):
            store.add_chunk(
                None,
                "text",
                "context",
                1,
                TriVec(
                    dense=dense,  # type: ignore[arg-type]
                    sparse={7: 1.0},
                    tokens=np.asarray([[1.0, 0.0]], dtype=np.float32),
                ),
            )
        assert dense.iterated is False
        assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo"])
def test_dense_vector_rejects_nested_rows_before_numpy_materialization(
    tmp_path, adapter: str
) -> None:
    store, calls = _make_store(adapter, tmp_path)
    nested = _OversizedExplodingStructuralRow()
    try:
        with pytest.raises(ValueError, match="dense vector must have one dimension"):
            store.add_chunk(
                None,
                "text",
                "context",
                1,
                TriVec(
                    dense=[nested],  # type: ignore[list-item]
                    sparse={7: 1.0},
                    tokens=np.asarray([[1.0, 0.0]], dtype=np.float32),
                ),
            )
        assert nested.iterated is False
        assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo"])
def test_structural_dimension_is_preflighted_before_row_iteration_or_database_io(
    tmp_path, adapter: str
) -> None:
    store, calls = _make_store(adapter, tmp_path)
    row = _OversizedExplodingStructuralRow()
    try:
        with pytest.raises(
            ValueError,
            match=(
                "structural vector dimension exceeds maximum of "
                f"{DEFAULT_RETRIEVAL_LIMITS.max_structural_dim}"
            ),
        ):
            store.add_chunk(
                None,
                "text",
                "context",
                1,
                TriVec(
                    dense=np.asarray([1.0, 0.0], dtype=np.float32),
                    sparse={7: 1.0},
                    tokens=[row],  # type: ignore[arg-type]
                ),
            )
        assert row.iterated is False
        assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


def test_sqlite_maxsim_batches_pairwise_allocations_without_changing_scores(
    tmp_path, monkeypatch
) -> None:
    token_rows = 600
    store = TriStore(tmp_path / "bounded-maxsim.db")
    tokens = np.tile(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        (token_rows // 2, 1),
    )
    with store.transaction():
        document_id = store.add_doc("source", "title", token_rows)
        chunk_id = store.add_chunk(
            document_id,
            "text",
            "context",
            token_rows,
            TriVec(
                dense=np.asarray([1.0, 0.0], dtype=np.float32),
                sparse={7: 1.0},
                tokens=tokens,
            ),
        )

    original_matmul = np.matmul
    pair_counts = []

    def bounded_matmul(left, right, *args, **kwargs):
        pair_counts.append(int(left.shape[0]) * int(right.shape[1]))
        return original_matmul(left, right, *args, **kwargs)

    monkeypatch.setattr(np, "matmul", bounded_matmul)
    try:
        scores = store.colbert_scores(tokens, [chunk_id])
    finally:
        store.close()

    assert scores == [(chunk_id, 1.0)]
    assert len(pair_counts) > 1
    assert max(pair_counts) <= DEFAULT_RETRIEVAL_LIMITS.max_structural_values


def test_postgres_holo_binary_maxsim_batches_pairwise_allocations() -> None:
    token_rows = 600
    document_constellation = b"\0" * (token_rows * 16)

    class RowsDatabase(_RecordingDatabase):
        def execute(self, sql, params=()):
            self.calls.append((sql, params))
            return _Rows(rows=[(7, document_constellation)])

    class GuardedHolographer:
        def __init__(self):
            self.pair_counts = []

        def sign_tokens(self, tokens):
            return b"\0" * (len(tokens) * 16)

        def binary_maxsim(self, query, document):
            pairs = (len(query) // 16) * (len(document) // 16)
            self.pair_counts.append(pairs)
            if pairs > DEFAULT_RETRIEVAL_LIMITS.max_structural_values:
                raise AssertionError("unbounded MaxSim pair allocation")
            return 0.25

    database = RowsDatabase()
    store = PgHoloStore.__new__(PgHoloStore)
    store.db = database
    store._dense_dim = 2
    store._colbert_dim = 2
    store._transaction_depth = 0
    store._max_structural_tokens = DEFAULT_RETRIEVAL_LIMITS.max_structural_tokens
    store._fingerprint_structural_tokens = None
    store.holo = GuardedHolographer()

    query = np.ones((token_rows, 2), dtype=np.float32)
    assert store.colbert_scores(query, [7]) == [(7, 0.25)]
    assert len(store.holo.pair_counts) > 1
    assert max(store.holo.pair_counts) <= (
        DEFAULT_RETRIEVAL_LIMITS.max_structural_values
    )


def test_sqlite_structural_token_limit_uses_config_fingerprint_and_hard_bound(
    tmp_path,
) -> None:
    configured = TriStore(
        tmp_path / "configured-structural-limit.db", max_structural_tokens=3
    )
    configured.set_meta(
        "retrieval_v2_fingerprint",
        json.dumps({"max_structural_tokens": 2}),
    )
    with pytest.raises(ValueError, match="structural token rows exceed maximum of 2"):
        configured.colbert_scores(
            np.ones((3, 2), dtype=np.float32), []
        )
    configured.close()

    hard_bounded = TriStore(tmp_path / "hard-structural-limit.db")
    rows = _OversizedExplodingTokenRows(
        size=DEFAULT_RETRIEVAL_LIMITS.max_structural_tokens + 1
    )
    try:
        with pytest.raises(
            ValueError,
            match=(
                "structural token rows exceed maximum of "
                f"{DEFAULT_RETRIEVAL_LIMITS.max_structural_tokens}"
            ),
        ):
            hard_bounded.colbert_scores(rows, [])  # type: ignore[arg-type]
        assert rows.iterated is False
    finally:
        hard_bounded.close()


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo", "pgvector"])
@pytest.mark.parametrize(
    "metadata,error_type",
    [
        ([], TypeError),
        ({"not_finite": math.nan}, ValueError),
        ({"oversized": "x" * (16 * 1024 + 1)}, ValueError),
    ],
)
def test_document_metadata_is_uniform_finite_bounded_json_before_database_io(
    tmp_path, adapter: str, metadata, error_type
) -> None:
    store, calls = _make_store(adapter, tmp_path)
    try:
        with pytest.raises(error_type, match="metadata|mapping|JSON"):
            store.add_doc("source", "title", 1, metadata)
        assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo", "pgvector"])
def test_document_metadata_accepts_protocol_mapping_implementations(
    tmp_path, adapter: str
) -> None:
    store, _calls = _make_store(adapter, tmp_path)
    try:
        document_id = store.add_doc(
            "source", "title", 1, MappingProxyType({"tenant": "alpha"})
        )
        assert document_id == 1
        if adapter == "sqlite":
            encoded = store.db.execute(
                "SELECT meta FROM docs WHERE id=?", (document_id,)
            ).fetchone()[0]
            assert encoded == '{"tenant":"alpha"}'
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo", "pgvector"])
def test_document_metadata_counts_a_lying_mapping_before_database_io(
    tmp_path, adapter: str
) -> None:
    store, calls = _make_store(adapter, tmp_path)
    metadata = _LyingDocumentMetadata()
    try:
        with pytest.raises(ValueError, match="metadata.*16 KiB"):
            store.add_doc("source", "title", 1, metadata)
        assert (
            0
            < metadata.yielded
            <= DEFAULT_RETRIEVAL_LIMITS.max_metadata_json_bytes + 1
        )
        assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo", "pgvector"])
@pytest.mark.parametrize(
    "method_name,args",
    [
        ("add_doc", (1, "title", 1)),
        ("add_doc", ("source", 1, 1)),
        ("add_parent", (1, 7, 1, 0)),
        ("set_meta", ("key", 1)),
        ("set_meta", ("", "value")),
        ("get_meta", (1,)),
    ],
)
def test_text_and_meta_scalar_types_are_uniform_before_database_io(
    tmp_path, adapter: str, method_name: str, args
) -> None:
    store, calls = _make_store(adapter, tmp_path)
    try:
        with pytest.raises((TypeError, ValueError), match="string|non-empty"):
            getattr(store, method_name)(*args)
        assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize("adapter", ["sqlite", "postgres-holo", "pgvector"])
@pytest.mark.parametrize("field", ["text", "ctx"])
def test_chunk_text_fields_are_uniform_strings_before_database_io(
    tmp_path, adapter: str, field: str
) -> None:
    store, calls = _make_store(adapter, tmp_path)
    arguments = {
        "doc_id": None,
        "text": "text",
        "ctx": "context",
        "n_tokens": 1,
        "vec": _vec(),
    }
    arguments[field] = 1
    try:
        with pytest.raises(TypeError, match="string"):
            store.add_chunk(**arguments)
        assert calls == []
    finally:
        _close_if_sqlite(adapter, store)


@pytest.mark.parametrize(
    "validator",
    [TriStore._sparse_input, PgHoloStore._sparse_input, _validate_sparse_weights],
)
@pytest.mark.parametrize("weight", [1e100, -(1e100)])
def test_sparse_weights_fit_postgresql_real_on_every_adapter(validator, weight) -> None:
    with pytest.raises(ValueError, match="PostgreSQL real|sparse weights"):
        validator({1: weight})


def test_sqlite_neighbors_break_duplicate_position_ties_by_chunk_id(tmp_path) -> None:
    store = TriStore(tmp_path / "neighbor-ties.db")
    try:
        document_id = store.add_doc("source", "title", 2)
        first = store.add_chunk(document_id, "z-first", "ctx", 1, _vec(), pos=3)
        second = store.add_chunk(document_id, "a-second", "ctx", 1, _vec(), pos=3)
        store.add_chunk(document_id, "not-requested", "ctx", 1, _vec(), pos=4)
        other_document = store.add_doc("other", "other", 1)
        store.add_chunk(other_document, "wrong-document", "ctx", 1, _vec(), pos=3)
        assert first < second

        assert store.neighbors(document_id, [3]) == [
            {
                "id": first,
                "doc_id": document_id,
                "kind": "chunk",
                "pos": 3,
                "text": "z-first",
            },
            {
                "id": second,
                "doc_id": document_id,
                "kind": "chunk",
                "pos": 3,
                "text": "a-second",
            },
        ]
    finally:
        store.close()


@pytest.mark.parametrize("adapter", ["postgres-holo", "pgvector"])
def test_postgres_neighbors_sql_has_a_stable_id_tie_break(
    tmp_path, adapter: str
) -> None:
    store, calls = _make_store(adapter, tmp_path)

    assert store.neighbors(1, [3]) == []

    sql = " ".join(calls[-1][0].split()).lower()
    assert "select id" in sql
    assert "order by pos asc, id asc" in sql


def test_pgvector_0_7_off_mode_emits_only_the_supported_hnsw_guc() -> None:
    database = _RecordingDatabase()
    store = object.__new__(PgVectorStore)
    store.db = database
    store._pgvector_version_tuple = (0, 7, 4)
    store.ef_search = 40
    store.iterative_scan = "off"
    store.max_scan_tuples = 20_000
    store.scan_mem_multiplier = 1.0

    store._set_local_ann_options(10)

    assert database.calls == [
        ("SELECT set_config(%s,%s,true)", ("hnsw.ef_search", "40"))
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_scan_tuples": 40_000},
        {"scan_mem_multiplier": 2},
    ],
)
def test_pgvector_0_7_rejects_controls_it_cannot_apply(overrides) -> None:
    options = {
        "ef_search": 40,
        "iterative_scan": "off",
        "max_scan_tuples": 20_000,
        "scan_mem_multiplier": 1,
        "extension_version": (0, 7, 4),
    }
    options.update(overrides)

    with pytest.raises(ValueError, match="pgvector >= 0.8.0"):
        _validate_ann_options(**options)

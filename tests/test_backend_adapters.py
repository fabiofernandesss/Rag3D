from __future__ import annotations

import math
import os
import re
import sqlite3
import sys
import threading
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from rag3d.backend import (
    DEFAULT_RETRIEVAL_LIMITS,
    BackendCapabilities,
    RetrievalBackend,
    SearchFilters,
    SearchScope,
)
from rag3d.encoders import TriVec
from rag3d.pgstore import SCHEMA as PG_HOLO_SCHEMA
from rag3d.pgstore import PgHoloStore
from rag3d.store import TriStore


def _vec(
    dense=(1.0, 0.0),
    sparse=None,
    tokens=((1.0, 0.0),),
) -> TriVec:
    return TriVec(
        dense=np.asarray(dense, dtype=np.float32),
        sparse=dict(sparse or {7: 1.0}),
        tokens=np.asarray(tokens, dtype=np.float32),
    )


def test_sqlite_is_a_truthful_runtime_backend(tmp_path):
    store = TriStore(tmp_path / "adapter.db")

    assert isinstance(store, RetrievalBackend)
    assert store.backend_name == "sqlite"
    assert store.capabilities == BackendCapabilities(
        exact_dense_search=True,
        ann_dense_search=False,
        sparse_search=True,
        structural_rerank=True,
        metadata_filters=False,
        transactions=True,
        native_vector=False,
        quantized_vector=False,
        cross_language_index=False,
    )
    assert store.health()["status"] == "ok"
    assert store.health()["backend"] == "sqlite"
    store.close()


def test_sqlite_contract_persists_searches_orders_and_deletes(tmp_path):
    path = tmp_path / "contract.db"
    store = TriStore(path)
    with store.transaction():
        first_doc = store.add_doc("first", "First", 5, {"group": "a"})
        parent = store.add_parent(first_doc, "parent", 1, 0)
        first = store.add_chunk(
            first_doc, "first", "first", 1, _vec(), pos=0, parent_id=parent
        )
        second_doc = store.add_doc("second", "Second", 5)
        second = store.add_chunk(second_doc, "second", "second", 1, _vec(), pos=0)

    assert [row["id"] for row in store.get_chunks([second, first])] == [second, first]
    assert [cid for cid, _ in store.dense_search(np.array([1.0, 0.0]), 2)] == [
        first,
        second,
    ]
    assert [cid for cid, _ in store.sparse_search({7: 1.0}, 2)] == [first, second]
    assert [
        cid
        for cid, _ in store.structural_rerank(
            np.array([[1.0, 0.0]], dtype=np.float32), [second, first], 2
        )
    ] == [first, second]
    assert set(store.dense_vectors([first, second])) == {first, second}

    store.close()
    reopened = TriStore(path)
    assert reopened.n_chunks() == 2
    reopened.delete_document(first_doc)
    assert reopened.get_chunks([first, second]) == [reopened.get_chunks([second])[0]]
    assert reopened.db.execute("SELECT COUNT(*) FROM docs WHERE id=?", (first_doc,)).fetchone()[0] == 0
    assert reopened.db.execute("SELECT COUNT(*) FROM postings WHERE chunk_id=?", (first,)).fetchone()[0] == 0
    reopened.close()


def test_sqlite_transaction_rolls_back_all_document_state_and_remains_usable(tmp_path):
    store = TriStore(tmp_path / "rollback.db")

    with pytest.raises(RuntimeError, match="injected"):
        with store.transaction():
            doc_id = store.add_doc("partial", "Partial", 2)
            chunk_id = store.add_chunk(doc_id, "partial", "partial", 2, _vec())
            store.set_meta("inside", "must rollback")
            store.touch_access([chunk_id], 3)
            raise RuntimeError("injected")

    assert store.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM dvecs").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM postings").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM colvecs").fetchone()[0] == 0
    assert store.get_meta("inside") is None

    with store.transaction():
        persisted = store.add_doc("ok", "OK", 1)
        try:
            with store.transaction():
                store.add_chunk(persisted, "rolled back", "rolled back", 1, _vec())
                raise RuntimeError("nested")
        except RuntimeError:
            pass
        kept = store.add_chunk(persisted, "kept", "kept", 1, _vec())

    assert [row["id"] for row in store.get_chunks([kept])] == [kept]
    assert store.n_chunks() == 1
    store.close()


def test_sqlite_meta_does_not_commit_a_preexisting_implicit_transaction(tmp_path):
    store = TriStore(tmp_path / "implicit-meta.db")
    store.add_doc("source", "pending", 1)
    assert store.db.in_transaction is True

    store.set_meta("pending-key", "pending-value")
    store.db.rollback()

    assert store.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 0
    assert store.get_meta("pending-key") is None
    store.close()


def test_sqlite_touch_does_not_commit_a_preexisting_implicit_transaction(tmp_path):
    store = TriStore(tmp_path / "implicit-touch.db")
    with store.transaction():
        doc_id = store.add_doc("source", "original", 1)
        chunk_id = store.add_chunk(doc_id, "text", "text", 1, _vec())

    store.db.execute("UPDATE docs SET title=? WHERE id=?", ("pending", doc_id))
    assert store.db.in_transaction is True
    store.touch_access([chunk_id], 9)
    store.db.rollback()

    assert store.db.execute("SELECT title FROM docs WHERE id=?", (doc_id,)).fetchone()[0] == "original"
    assert store.db.execute(
        "SELECT accessed_turn FROM chunks WHERE id=?", (chunk_id,)
    ).fetchone()[0] is None
    store.close()


def test_sqlite_add_chunk_rolls_back_its_whole_unit_when_caller_catches_error(
    tmp_path,
):
    store = TriStore(tmp_path / "atomic-add-chunk.db")
    with store.transaction():
        document_id = store.add_doc("source", "title", 1)
    store.db.execute(
        "CREATE TRIGGER fail_structural_insert BEFORE INSERT ON colvecs "
        "BEGIN SELECT RAISE(FAIL, 'injected structural failure'); END"
    )
    store.db.commit()
    store.db.execute(
        "UPDATE docs SET title=? WHERE id=?", ("unrelated pending work", document_id)
    )

    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        store.add_chunk(document_id, "partial", "partial", 1, _vec())
    # This is the hostile caller pattern: catch the write error, then commit
    # unrelated work. No earlier step of add_chunk may leak through that commit.
    store.db.commit()

    assert store.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM dvecs").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM postings").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM colvecs").fetchone()[0] == 0
    assert store.db.execute(
        "SELECT title FROM docs WHERE id=?", (document_id,)
    ).fetchone()[0] == "unrelated pending work"
    store.close()


def test_sqlite_atomic_add_chunk_preserves_legacy_explicit_commit_visibility(
    tmp_path,
):
    path = tmp_path / "atomic-add-chunk-commit.db"
    store = TriStore(path)
    peer = sqlite3.connect(path)
    try:
        store.add_chunk(None, "pending", "ctx", 1, _vec())
        assert peer.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        store.commit()
        assert peer.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
    finally:
        peer.close()
        store.close()


def test_sqlite_touch_access_rolls_back_earlier_batches_when_a_later_batch_fails(
    tmp_path,
):
    store = TriStore(tmp_path / "atomic-touch.db")
    with store.transaction():
        document_id = store.add_doc("source", "title", 3)
        chunk_ids = [
            store.add_chunk(
                document_id, f"chunk-{position}", "ctx", 1, _vec(), pos=position
            )
            for position in range(3)
        ]
    store.db.execute(
        f"CREATE TRIGGER fail_late_touch BEFORE UPDATE OF accessed_turn ON chunks "
        f"WHEN OLD.id={chunk_ids[-1]} "
        "BEGIN SELECT RAISE(FAIL, 'injected late batch failure'); END"
    )
    store.db.commit()
    store.db.execute(
        "UPDATE docs SET title=? WHERE id=?", ("unrelated pending work", document_id)
    )
    previous_limit = store.db.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 3)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            store.touch_access(chunk_ids, 9)
        store.db.commit()
    finally:
        store.db.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous_limit)

    assert store.db.execute(
        "SELECT COUNT(*) FROM chunks WHERE accessed_turn IS NOT NULL"
    ).fetchone()[0] == 0
    assert store.db.execute(
        "SELECT title FROM docs WHERE id=?", (document_id,)
    ).fetchone()[0] == "unrelated pending work"
    store.close()


def test_sqlite_schema_enforces_document_and_vector_cascades(tmp_path):
    store = TriStore(tmp_path / "foreign-keys.db")
    try:
        assert store.db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        chunk_targets = {
            row[2]: row[6] for row in store.db.execute("PRAGMA foreign_key_list(chunks)")
        }
        assert chunk_targets["docs"] == "CASCADE"
        for table in ("dvecs", "postings", "colvecs"):
            targets = {
                row[2]: row[6]
                for row in store.db.execute(f"PRAGMA foreign_key_list({table})")
            }
            assert targets["chunks"] == "CASCADE"
    finally:
        store.close()


def _create_legacy_sqlite_schema_without_foreign_keys(path: Path) -> None:
    database = sqlite3.connect(path)
    database.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE docs(
          id INTEGER PRIMARY KEY, source TEXT, title TEXT,
          created REAL, n_tokens INTEGER, meta TEXT
        );
        CREATE TABLE chunks(
          id INTEGER PRIMARY KEY, doc_id INTEGER, parent_id INTEGER,
          kind TEXT NOT NULL DEFAULT 'chunk', pos INTEGER, text TEXT NOT NULL,
          ctx TEXT, n_tokens INTEGER, created REAL, importance REAL DEFAULT 0.5,
          turn_no INTEGER, accessed_turn INTEGER
        );
        CREATE TABLE dvecs(chunk_id INTEGER PRIMARY KEY, data BLOB NOT NULL);
        CREATE TABLE postings(term INTEGER NOT NULL, chunk_id INTEGER NOT NULL, weight REAL NOT NULL);
        CREATE TABLE colvecs(chunk_id INTEGER PRIMARY KEY, n_tok INTEGER, dim INTEGER, data BLOB NOT NULL);
        """
    )
    database.close()


def test_sqlite_legacy_schema_gets_compatible_validated_integrity_triggers(tmp_path):
    path = tmp_path / "legacy-no-fks.db"
    _create_legacy_sqlite_schema_without_foreign_keys(path)
    store = TriStore(path)
    try:
        assert store.db.execute("PRAGMA foreign_key_list(chunks)").fetchall() == []
        with store.transaction():
            document_id = store.add_doc("source", "title", 1)
            chunk_id = store.add_chunk(document_id, "text", "ctx", 1, _vec())
        store.delete_document(document_id)
        assert store.get_chunks([chunk_id]) == []
        assert store.db.execute("SELECT COUNT(*) FROM dvecs").fetchone()[0] == 0
        assert store.db.execute("SELECT COUNT(*) FROM postings").fetchone()[0] == 0
        assert store.db.execute("SELECT COUNT(*) FROM colvecs").fetchone()[0] == 0

        with pytest.raises(sqlite3.IntegrityError, match="unknown document"):
            store.add_chunk(999, "orphan", "ctx", 1, _vec())
        store.db.rollback()
    finally:
        store.close()


def test_sqlite_legacy_schema_refuses_preexisting_orphans(tmp_path):
    path = tmp_path / "legacy-orphan.db"
    _create_legacy_sqlite_schema_without_foreign_keys(path)
    database = sqlite3.connect(path)
    database.execute("INSERT INTO postings VALUES(7,999,1.0)")
    database.commit()
    database.close()

    with pytest.raises(RuntimeError, match="orphaned"):
        TriStore(path)


def test_sqlite_outer_transaction_takes_immediate_write_lock(tmp_path):
    store = TriStore(tmp_path / "fingerprint-lock.db")
    statements = []
    store.db.set_trace_callback(statements.append)

    with store.transaction():
        pass

    assert "BEGIN IMMEDIATE" in statements
    store.close()


@pytest.mark.parametrize("method", ["dense", "sparse", "structural"])
@pytest.mark.parametrize(
    "k,error",
    [
        (-1, ValueError),
        (True, TypeError),
        (DEFAULT_RETRIEVAL_LIMITS.max_channel_k + 1, ValueError),
    ],
)
def test_sqlite_search_rejects_invalid_k(tmp_path, method, k, error):
    store = TriStore(tmp_path / "limits.db")
    with pytest.raises(error):
        if method == "dense":
            store.dense_search(np.array([1.0, 0.0]), k)
        elif method == "sparse":
            store.sparse_search({7: 1.0}, k)
        else:
            store.structural_rerank(np.array([[1.0, 0.0]]), [], k)
    store.close()


def test_sqlite_search_handles_zero_empty_and_rejects_unsupported_modes(tmp_path):
    store = TriStore(tmp_path / "validation.db")
    assert store.dense_search(np.array([], dtype=np.float32), 0) == []
    assert store.sparse_search({}, 0) == []
    assert store.structural_rerank(np.empty((0, 2), dtype=np.float32), [], 0) == []

    non_empty_filter = SearchFilters(scope=SearchScope(kinds=("chunk",)))
    calls = (
        lambda: store.dense_search(np.array([1.0, 0.0]), 1, filters=non_empty_filter),
        lambda: store.sparse_search({7: 1.0}, 1, filters=non_empty_filter),
        lambda: store.structural_rerank(
            np.array([[1.0, 0.0]]), [], 1, filters=non_empty_filter
        ),
    )
    for call in calls:
        with pytest.raises(NotImplementedError, match="filter"):
            call()

    with pytest.raises(NotImplementedError, match="exact|ANN"):
        store.dense_search(np.array([1.0, 0.0]), 1, exact=False)

    with store.transaction():
        doc_id = store.add_doc("source", "title", 1)
        store.add_chunk(doc_id, "text", "text", 1, _vec())
    with pytest.raises(ValueError, match="dimension"):
        store.dense_search(np.array([1.0, 0.0, 0.0]), 1)
    with pytest.raises(ValueError, match="finite"):
        store.dense_search(np.array([math.nan, 0.0]), 1)
    with pytest.raises(ValueError, match="finite"):
        store.sparse_search({7: math.inf}, 1)
    store.close()


class _RecordingDb:
    def __init__(self, health_error=None):
        self.calls = []
        self.transactions = 0
        self.commits = 0
        self.health_error = health_error

    @contextmanager
    def transaction(self):
        self.transactions += 1
        yield

    def execute(self, sql, params=None):
        if sql == "SELECT 1" and self.health_error is not None:
            raise self.health_error
        self.calls.append((sql, params))
        return self

    def commit(self):
        self.commits += 1

    def fetchone(self):
        return (1,)

    def fetchall(self):
        return []

    def close(self):
        self.calls.append(("CLOSE", None))


def _bare_pg(db=None) -> PgHoloStore:
    store = PgHoloStore.__new__(PgHoloStore)
    store.db = db or _RecordingDb()
    store._dense_dim = 2
    store._colbert_dim = 2
    store._transaction_depth = 0
    return store


def test_postgres_holo_is_a_truthful_runtime_backend_and_uses_transactions():
    store = _bare_pg()

    assert isinstance(store, RetrievalBackend)
    assert store.backend_name == "postgres-holo"
    assert store.capabilities == BackendCapabilities(
        exact_dense_search=False,
        ann_dense_search=True,
        sparse_search=True,
        structural_rerank=True,
        metadata_filters=False,
        transactions=True,
        native_vector=False,
        quantized_vector=True,
        cross_language_index=False,
    )
    with store.transaction():
        pass
    assert store.db.transactions == 1


def test_postgres_holo_commit_bridge_is_noop_inside_managed_transaction():
    store = _bare_pg()

    with store.transaction():
        assert store._transaction_depth == 1
        store.commit()

    assert store._transaction_depth == 0
    assert store.db.commits == 0
    store.commit()
    assert store.db.commits == 1


def test_postgres_holo_fingerprint_lock_is_transaction_scoped_and_parameterized():
    store = _bare_pg()

    with pytest.raises(RuntimeError, match="transaction"):
        store.lock_fingerprint()
    with store.transaction():
        store.lock_fingerprint()

    assert store.db.calls[-2] == (
        f"SET LOCAL lock_timeout = '{store.FINGERPRINT_LOCK_TIMEOUT_MS}ms'",
        None,
    )
    assert store.db.calls[-1] == (
        "SELECT pg_advisory_xact_lock(%s)",
        (store.FINGERPRINT_LOCK_ID,),
    )


@pytest.mark.parametrize("failing_sql", ["SET LOCAL", "pg_advisory_xact_lock"])
def test_postgres_holo_fingerprint_lock_failure_is_fixed_and_secret_safe(
    failing_sql,
):
    secret = "postgresql://admin:super-secret@example.invalid/prod"

    class FailingLockDb(_RecordingDb):
        def execute(self, sql, params=None):
            if failing_sql in sql:
                raise RuntimeError(secret)
            return super().execute(sql, params)

    store = _bare_pg(FailingLockDb())
    with pytest.raises(
        RuntimeError, match="^postgres-holo fingerprint lock acquisition failed$"
    ) as raised:
        with store.transaction():
            store.lock_fingerprint()

    assert secret not in str(raised.value)
    assert "super-secret" not in str(raised.value)


def test_postgres_holo_ann_orders_hamming_ties_by_id_before_limit():
    class FakeHolographer:
        def sign_dense(self, vector):
            return b"signature"

        def sig_to_bitstring(self, signature):
            return "0" * 1024

        def bands_of(self, signature):
            return [1]

    store = _bare_pg()
    store.holo = FakeHolographer()

    assert store.dense_search(np.array([1.0, 0.0]), 1) == []
    scan_sql = store.db.calls[-1][0]
    assert "ORDER BY ham ASC, id ASC LIMIT %s" in scan_sql


def test_postgres_holo_deletes_document_with_one_fk_cascaded_statement():
    db = _RecordingDb()
    store = _bare_pg(db)

    store.delete_document(42)

    assert db.transactions == 1
    assert db.calls == [("DELETE FROM holo_docs WHERE id=%s", (42,))]


def test_postgres_holo_schema_has_idempotent_validated_cascade_constraints():
    normalized = " ".join(PG_HOLO_SCHEMA.split()).lower()

    assert "references holo_docs(id) on delete cascade" in normalized
    assert "references holo_grams(id) on delete cascade" in normalized
    assert "validate constraint" in normalized
    assert "contype='f'" in normalized
    assert "confdeltype='c'" in normalized


def test_postgres_holo_constructor_bounds_schema_lock_and_statement_time(
    monkeypatch,
):
    calls = []

    class Result:
        def fetchone(self):
            return None

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def execute(self, sql, params=None):
            calls.append((sql, params))
            return Result()

    class Database:
        def __init__(self):
            self.transactions = 0
            self.closed = False

        @contextmanager
        def transaction(self):
            self.transactions += 1
            yield

        def cursor(self):
            return Cursor()

        def execute(self, sql, params=None):
            calls.append((sql, params))
            return Result()

        def close(self):
            self.closed = True

    database = Database()
    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = lambda *args, **kwargs: database
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    store = PgHoloStore("postgresql://secret.invalid/prod", 2, 2)
    try:
        assert database.transactions == 1
        assert calls[:3] == [
            (
                f"SET LOCAL lock_timeout = '{store.SCHEMA_LOCK_TIMEOUT_MS}ms'",
                None,
            ),
            (
                "SET LOCAL statement_timeout = "
                f"'{store.SCHEMA_STATEMENT_TIMEOUT_MS}ms'",
                None,
            ),
            (PG_HOLO_SCHEMA, None),
        ]
        assert 0 < store.SCHEMA_LOCK_TIMEOUT_MS <= 60_000
        assert 0 < store.SCHEMA_STATEMENT_TIMEOUT_MS <= 60_000
    finally:
        store.close()


@pytest.mark.parametrize(
    "failing_sql", ["lock_timeout", "statement_timeout", "CREATE TABLE"]
)
def test_postgres_holo_schema_bootstrap_failure_is_fixed_and_secret_safe(
    monkeypatch, failing_sql,
):
    secret = "postgresql://admin:super-secret@example.invalid/prod"

    class Result:
        def fetchone(self):
            return None

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def execute(self, sql, params=None):
            if failing_sql in sql:
                raise RuntimeError(secret)
            return Result()

    class Database:
        def __init__(self):
            self.closed = False

        @contextmanager
        def transaction(self):
            yield

        def cursor(self):
            return Cursor()

        def close(self):
            self.closed = True

    database = Database()
    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = lambda *args, **kwargs: database
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    with pytest.raises(
        RuntimeError, match="^failed to initialize postgres-holo schema$"
    ) as raised:
        PgHoloStore(secret, 2, 2)

    assert database.closed is True
    assert secret not in str(raised.value)
    assert "super-secret" not in str(raised.value)
    assert raised.value.__cause__ is not None
    assert secret in str(raised.value.__cause__)


@pytest.mark.parametrize(
    "relative_path",
    [
        "rag3d-js/src/pgstore.js",
        "rag3d-java/src/main/java/io/rag3d/core/PgHoloStore.java",
    ],
)
def test_cross_language_holo_schema_declares_the_same_cascade_fks(relative_path):
    source = (Path(__file__).resolve().parents[1] / relative_path).read_text(
        encoding="utf-8"
    )
    normalized = " ".join(source.split()).lower()

    assert "references holo_docs(id) on delete cascade" in normalized
    assert "references holo_grams(id) on delete cascade" in normalized
    assert "contype='f'" in normalized
    assert "confdeltype='c'" in normalized


def test_postgres_holo_n_chunks_observes_each_database_snapshot():
    class ChangingCountDb(_RecordingDb):
        def __init__(self):
            super().__init__()
            self.counts = iter(((1,), (2,)))

        def fetchone(self):
            return next(self.counts)

    store = _bare_pg(ChangingCountDb())

    assert store.n_chunks() == 1
    assert store.n_chunks() == 2
    assert sum("SELECT COUNT(*) FROM holo_grams" in sql for sql, _ in store.db.calls) == 2


def test_postgres_holo_sparse_is_filtered_aggregated_and_limited_in_sql():
    store = _bare_pg()

    assert store.sparse_search({7: 1.0, 9: 0.5}, 3) == []

    assert len(store.db.calls) == 1
    sql, params = store.db.calls[0]
    normalized = " ".join(sql.split()).lower()
    assert "join holo_grams" in normalized
    assert "%s::double precision[]" in normalized
    assert "kind in ('chunk','turn','summary')" in normalized
    assert "count(distinct p.gram_id)" in normalized
    assert "group by" in normalized
    assert "order by score desc" in normalized
    assert "limit %s" in normalized
    assert params[-1] == 3


def test_postgres_holo_rejects_exact_and_filters_before_touching_database():
    store = _bare_pg()
    filters = SearchFilters(scope=SearchScope(sources=("source",)))

    with pytest.raises(NotImplementedError, match="exact"):
        store.dense_search(np.array([1.0, 0.0]), 1, exact=True)
    with pytest.raises(NotImplementedError, match="filter"):
        store.sparse_search({7: 1.0}, 1, filters=filters)
    assert store.db.calls == []


def test_postgres_health_never_exposes_driver_error_or_dsn():
    secret = "postgresql://admin:super-secret@example.invalid/prod"
    store = _bare_pg(_RecordingDb(RuntimeError(secret)))

    health = store.health()

    assert health["status"] == "error"
    assert health["backend"] == "postgres-holo"
    assert secret not in repr(health)
    assert "super-secret" not in repr(health)


def test_postgres_constructor_masks_connection_error_dsn(monkeypatch):
    secret = "postgresql://admin:super-secret@example.invalid/prod"
    fake_psycopg = types.ModuleType("psycopg")

    def fail_connect(*args, **kwargs):
        raise RuntimeError(secret)

    fake_psycopg.connect = fail_connect
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    with pytest.raises(RuntimeError, match="postgres-holo|connect") as raised:
        PgHoloStore(secret, dense_dim=2, colbert_dim=2)

    assert secret not in str(raised.value)
    assert "super-secret" not in str(raised.value)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert secret in str(raised.value.__cause__)


@pytest.mark.parametrize(
    "name,value,error_type",
    [
        ("dense_dim", True, TypeError),
        ("dense_dim", 0, ValueError),
        (
            "dense_dim",
            DEFAULT_RETRIEVAL_LIMITS.max_dense_dim + 1,
            ValueError,
        ),
        ("colbert_dim", True, TypeError),
        ("colbert_dim", 0, ValueError),
        (
            "colbert_dim",
            DEFAULT_RETRIEVAL_LIMITS.max_structural_dim + 1,
            ValueError,
        ),
    ],
)
def test_postgres_constructor_validates_dimensions_before_connecting(
    monkeypatch, name, value, error_type
):
    connected = False
    fake_psycopg = types.ModuleType("psycopg")

    def connect(*args, **kwargs):
        nonlocal connected
        connected = True
        raise AssertionError("invalid dimensions must not reach psycopg")

    fake_psycopg.connect = connect
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    dimensions = {"dense_dim": 2, "colbert_dim": 2}
    dimensions[name] = value

    with pytest.raises(error_type, match="dimension|dense_dim|colbert_dim|between"):
        PgHoloStore("postgresql://example.invalid/test", **dimensions)

    assert connected is False


def test_postgres_constructor_closes_connection_when_fingerprint_read_fails(
    monkeypatch,
):
    secret = "postgresql://admin:super-secret@example.invalid/prod"

    class Result:
        def fetchone(self):
            return None

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def execute(self, sql, params=None):
            return Result()

    class Database:
        def __init__(self):
            self.closed = False

        @contextmanager
        def transaction(self):
            yield

        def cursor(self):
            return Cursor()

        def execute(self, sql, params=None):
            raise RuntimeError(secret)

        def close(self):
            self.closed = True

    database = Database()
    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = lambda *args, **kwargs: database
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    with pytest.raises(
        RuntimeError, match="^failed to initialize postgres-holo backend$"
    ) as raised:
        PgHoloStore(secret, dense_dim=2, colbert_dim=2)

    assert database.closed is True
    assert secret not in str(raised.value)
    assert "super-secret" not in str(raised.value)
    assert raised.value.__cause__ is not None
    assert secret in str(raised.value.__cause__)


def test_postgres_neighbors_accepts_negative_probe_positions_from_legacy_stitching():
    store = _bare_pg()

    assert store.neighbors(5, [-1, 0, 1]) == []
    assert store.db.calls[-1][1] == (5, [-1, 0, 1])


def test_postgres_holo_neighbors_return_identity_fields_in_stable_order():
    class NeighborDb(_RecordingDb):
        def fetchall(self):
            return [
                (12, 5, "chunk", 3, "first"),
                (19, 5, "chunk", 3, "second"),
            ]

    store = _bare_pg(NeighborDb())

    assert store.neighbors(5, [3]) == [
        {"id": 12, "doc_id": 5, "kind": "chunk", "pos": 3, "text": "first"},
        {"id": 19, "doc_id": 5, "kind": "chunk", "pos": 3, "text": "second"},
    ]
    sql, params = store.db.calls[-1]
    assert "SELECT id, doc_id, kind, pos, text" in sql
    assert "ORDER BY pos ASC, id ASC" in sql
    assert params == (5, [3])


def test_sqlite_rejects_values_that_overflow_structural_storage(tmp_path):
    store = TriStore(tmp_path / "float16-overflow.db")
    oversized = _vec(tokens=((100_000.0, 0.0),))

    with pytest.raises(ValueError, match="finite|float16"):
        with store.transaction():
            doc_id = store.add_doc("source", "title", 1)
            store.add_chunk(doc_id, "text", "text", 1, oversized)

    assert store.n_chunks() == 0
    store.close()


@pytest.mark.parametrize(
    "incompatible",
    [
        _vec(dense=(1.0, 0.0, 0.0)),
        _vec(tokens=((1.0, 0.0, 0.0),)),
    ],
)
def test_sqlite_rejects_mixed_storage_dimensions_before_writing(
    tmp_path, incompatible
):
    store = TriStore(tmp_path / "mixed-dimensions.db")
    with store.transaction():
        doc_id = store.add_doc("source", "title", 2)
        store.add_chunk(doc_id, "first", "first", 1, _vec())

    with pytest.raises(ValueError, match="dimension"):
        with store.transaction():
            store.add_chunk(doc_id, "second", "second", 1, incompatible)

    assert store.n_chunks() == 1
    store.close()


def _safe_pg_store_or_skip():
    dsn = os.environ.get("RAG3D_TEST_PG_DSN", "")
    allowed = os.environ.get("RAG3D_TEST_PG_ALLOW_DESTRUCTIVE") == "1"
    if not dsn or not allowed:
        pytest.skip(
            "requires RAG3D_TEST_PG_DSN and RAG3D_TEST_PG_ALLOW_DESTRUCTIVE=1"
        )

    psycopg = pytest.importorskip("psycopg")
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    database = str(conninfo_to_dict(dsn).get("dbname", ""))
    if re.search(r"(?:^|[_-])test(?:$|[_-])", database.casefold()) is None:
        pytest.skip("requires a database name with a delimited test token")

    schema = "rag3d_test_" + uuid.uuid4().hex
    admin = psycopg.connect(dsn, autocommit=True)
    admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    isolated_dsn = make_conninfo(dsn, options="-c search_path=" + schema)
    try:
        store = PgHoloStore(isolated_dsn, dense_dim=2, colbert_dim=2)
    except BaseException:
        admin.execute(
            sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
        )
        admin.close()
        raise
    store._test_isolated_dsn = isolated_dsn
    return store, admin, schema, sql


def test_postgres_holo_contract_in_isolated_test_schema():
    store, admin, schema, sql = _safe_pg_store_or_skip()
    try:
        with pytest.raises(RuntimeError, match="injected"):
            with store.transaction():
                doc_id = store.add_doc("source", "title", 1)
                store.add_chunk(doc_id, "text", "text", 1, _vec())
                raise RuntimeError("injected")
        assert store.n_chunks() == 0

        with store.transaction():
            doc_id = store.add_doc("source", "title", 1)
            chunk_id = store.add_chunk(doc_id, "text", "text", 1, _vec())
            summary_id = store.add_chunk(
                doc_id,
                "summary",
                "summary",
                1,
                _vec(sparse={9: 1.0}),
                kind="summary",
                pos=-1,
            )
        assert store.get_chunks([chunk_id])[0]["text"] == "text"
        assert store.dense_search(np.array([1.0, 0.0]), 1)[0][0] == chunk_id
        store.db.execute(
            "ALTER TABLE holo_spectrum DROP CONSTRAINT holo_spectrum_gram_fk"
        )
        orphan_id = 9_000_000_001
        store.db.execute(
            "INSERT INTO holo_spectrum(term,gram_id,weight) VALUES(%s,%s,%s)",
            (9, summary_id, 0.0),
        )
        store.db.execute(
            "INSERT INTO holo_spectrum(term,gram_id,weight) VALUES(%s,%s,%s)",
            (9, orphan_id, 100.0),
        )
        sparse = store.sparse_search({9: 1.0}, 10)
        assert [candidate for candidate, _ in sparse] == [summary_id]
        assert sparse[0][1] == pytest.approx(math.log(2.0))
        store.db.execute(
            "DELETE FROM holo_spectrum WHERE gram_id=%s", (orphan_id,)
        )
        store.db.execute(
            "ALTER TABLE holo_spectrum ADD CONSTRAINT holo_spectrum_gram_fk "
            "FOREIGN KEY(gram_id) REFERENCES holo_grams(id) ON DELETE CASCADE "
            "NOT VALID"
        )
        store.db.execute(
            "ALTER TABLE holo_spectrum VALIDATE CONSTRAINT holo_spectrum_gram_fk"
        )

        constraints = store.db.execute(
            "SELECT conname, convalidated, confdeltype "
            "FROM pg_constraint "
            "WHERE connamespace = ("
            "  SELECT oid FROM pg_namespace WHERE nspname=current_schema()"
            ") AND conname = ANY(%s) ORDER BY conname",
            (["holo_grams_doc_fk", "holo_spectrum_gram_fk"],),
        ).fetchall()
        assert constraints == [
            ("holo_grams_doc_fk", True, "c"),
            ("holo_spectrum_gram_fk", True, "c"),
        ]
        store.delete_document(doc_id)
        assert store.n_chunks() == 0
    finally:
        store.close()
        admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        admin.close()


def test_postgres_holo_advisory_lock_blocks_second_session_in_test_database():
    store, admin, schema, sql = _safe_pg_store_or_skip()
    try:
        with store.transaction():
            store.lock_fingerprint()
            acquired = admin.execute(
                "SELECT pg_try_advisory_xact_lock(%s)",
                (store.FINGERPRINT_LOCK_ID,),
            ).fetchone()[0]
            assert acquired is False
            store.commit()

        assert store._transaction_depth == 0
        acquired_after = admin.execute(
            "SELECT pg_try_advisory_xact_lock(%s)",
            (store.FINGERPRINT_LOCK_ID,),
        ).fetchone()[0]
        assert acquired_after is True
    finally:
        store.close()
        admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        admin.close()


def test_postgres_holo_advisory_lock_times_out_bounded_in_test_database():
    store, admin, schema, sql = _safe_pg_store_or_skip()
    store.FINGERPRINT_LOCK_TIMEOUT_MS = 100
    try:
        with admin.transaction():
            admin.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (store.FINGERPRINT_LOCK_ID,),
            )
            started = time.monotonic()
            with pytest.raises(
                RuntimeError,
                match="^postgres-holo fingerprint lock acquisition failed$",
            ):
                with store.transaction():
                    store.lock_fingerprint()
            elapsed = time.monotonic() - started
            assert elapsed < 2.0
    finally:
        store.close()
        admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        admin.close()


@pytest.mark.parametrize("legacy_state", ["wrong-named-constraint", "orphan"])
def test_postgres_holo_schema_hardening_fails_closed_on_invalid_legacy_state(
    legacy_state,
):
    store, admin, schema, sql = _safe_pg_store_or_skip()
    isolated_dsn = store._test_isolated_dsn
    try:
        if legacy_state == "wrong-named-constraint":
            store.db.execute(
                "ALTER TABLE holo_grams DROP CONSTRAINT holo_grams_doc_fk"
            )
            store.db.execute(
                "ALTER TABLE holo_grams ADD CONSTRAINT holo_grams_doc_fk CHECK (TRUE)"
            )
        else:
            store.db.execute(
                "ALTER TABLE holo_spectrum DROP CONSTRAINT holo_spectrum_gram_fk"
            )
            store.db.execute(
                "INSERT INTO holo_spectrum(term,gram_id,weight) VALUES(7,999,1.0)"
            )
        store.close()

        with pytest.raises(RuntimeError, match="initialize.*schema"):
            PgHoloStore(isolated_dsn, dense_dim=2, colbert_dim=2)
    finally:
        if not store.db.closed:
            store.close()
        admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        admin.close()


def test_postgres_holo_delete_document_race_cascades_concurrent_chunk():
    store, admin, schema, sql = _safe_pg_store_or_skip()
    peer = PgHoloStore(store._test_isolated_dsn, dense_dim=2, colbert_dim=2)
    try:
        document_id = store.add_doc("source", "title", 1)
        inserted = threading.Event()
        allow_insert_commit = threading.Event()

        def add_while_delete_starts():
            with peer.transaction():
                chunk_id = peer.add_chunk(
                    document_id, "concurrent", "ctx", 1, _vec()
                )
                inserted.set()
                assert allow_insert_commit.wait(timeout=2.0)
                return chunk_id

        with ThreadPoolExecutor(max_workers=2) as executor:
            addition = executor.submit(add_while_delete_starts)
            assert inserted.wait(timeout=2.0)
            deletion = executor.submit(store.delete_document, document_id)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                waiting = admin.execute(
                    "SELECT wait_event_type='Lock' FROM pg_stat_activity WHERE pid=%s",
                    (store.db.info.backend_pid,),
                ).fetchone()
                if waiting and waiting[0]:
                    break
                time.sleep(0.01)
            else:
                pytest.fail("delete_document did not reach the expected FK row lock")
            allow_insert_commit.set()
            concurrent_chunk = addition.result(timeout=2.0)
            deletion.result(timeout=2.0)

        assert peer.get_chunks([concurrent_chunk]) == []
        assert peer.db.execute(
            "SELECT COUNT(*) FROM holo_spectrum s "
            "LEFT JOIN holo_grams g ON g.id=s.gram_id WHERE g.id IS NULL"
        ).fetchone()[0] == 0
    finally:
        peer.close()
        store.close()
        admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        admin.close()


def test_capability_payloads_contain_only_boolean_claims(tmp_path):
    stores = [TriStore(tmp_path / "caps.db"), _bare_pg()]
    try:
        for store in stores:
            assert all(isinstance(value, bool) for value in asdict(store.capabilities).values())
    finally:
        stores[0].close()

"""Real PostgreSQL contract tests for :mod:`rag3d.pgvector_store`.

The suite is intentionally inert unless both environment gates are explicit:

* ``RAG3D_TEST_PG_DSN`` identifies an isolated test database; and
* ``RAG3D_TEST_PG_ALLOW_DESTRUCTIVE=1`` authorizes truncation/index rebuilds.

A durable sentinel in ``rag3d_v2_meta`` is verified before every destructive
statement as a third guard against pointing the suite at an unrelated schema.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag3d.backend import FingerprintMismatchError, IndexFingerprint, SearchFilters, SearchScope
from rag3d.config import TriRagConfig
from rag3d.engine import TriRag
from rag3d.pgvector_store import (
    PgVectorError,
    PgVectorHnswError,
    PgVectorSchemaError,
    PgVectorStore,
    _FINGERPRINT_LOCK_KEY,
    _MIGRATION_LOCK_KEY,
)


DSN = os.environ.get("RAG3D_TEST_PG_DSN")
DESTRUCTIVE_ALLOWED = os.environ.get("RAG3D_TEST_PG_ALLOW_DESTRUCTIVE") == "1"
_SENSITIVE_DETAIL = "opaque-sensitive-detail"


def _safe_test_database(dsn: object) -> bool:
    if not isinstance(dsn, str) or not dsn.strip():
        return False
    try:
        from psycopg.conninfo import conninfo_to_dict

        database = str(conninfo_to_dict(dsn).get("dbname", ""))
    except Exception:
        return False
    return bool(re.search(r"(?:^|[_-])test(?:$|[_-])", database.casefold()))


INTEGRATION_ENABLED = (
    bool(DSN) and DESTRUCTIVE_ALLOWED and _safe_test_database(DSN)
)
SENTINEL_KEY = "integration_test_sentinel"
SENTINEL_VALUE = "rag3d-pgvector-integration-v1"
HNSW_INDEX = "rag3d_v2_chunks_embedding_hnsw"
_PSQL_CONNECT_TIMEOUT_SECONDS = 5
_PSQL_STATEMENT_TIMEOUT_MS = 60_000
_PSQL_LOCK_TIMEOUT_MS = 5_000
_PSQL_PROCESS_TIMEOUT_SECONDS = 75

def _run_psql(command):
    environment = os.environ.copy()
    environment["PGCONNECT_TIMEOUT"] = str(_PSQL_CONNECT_TIMEOUT_SECONDS)
    existing_options = environment.get("PGOPTIONS", "").strip()
    bounded_options = (
        f"-c statement_timeout={_PSQL_STATEMENT_TIMEOUT_MS} "
        f"-c lock_timeout={_PSQL_LOCK_TIMEOUT_MS}"
    )
    environment["PGOPTIONS"] = " ".join(
        option for option in (existing_options, bounded_options) if option
    )
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            timeout=_PSQL_PROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("psql integration command timed out") from None


def _bounded_index_ddl(store: PgVectorStore, statement: str) -> None:
    with store.db.transaction():
        store.db.execute(
            "SELECT set_config('lock_timeout',%s,true)",
            (str(store.lock_timeout_ms),),
        )
        store.db.execute(
            "SELECT set_config('statement_timeout',%s,true)",
            (str(store.statement_timeout_ms),),
        )
        store.db.execute(statement)


def _vector(dense, sparse=None, tokens=None):
    return SimpleNamespace(
        dense=np.asarray(dense, dtype=np.float32),
        sparse=dict(sparse or {}),
        tokens=np.asarray(tokens if tokens is not None else [[1.0, 0.0]], dtype=np.float32),
    )


def _fingerprint(*, model: str = "integration-model") -> IndexFingerprint:
    return IndexFingerprint(
        backend="pgvector",
        encoder="integration",
        model=model,
        revision="test",
        dense_dim=3,
        structural_dim=2,
        max_structural_tokens=32,
        structural_projection="identity-test-v1",
        query_max_tokens=32,
        passage_max_tokens=128,
        sparse_version="test-sparse-v1",
        schema_version="pgvector-v1",
        normalization="l2",
        quantization="none",
        chunk_size=128,
        overlap=16,
        chunking_version="test-v1",
        pipeline_version="retrieval-v2",
    )


def _assert_or_install_sentinel(store: PgVectorStore) -> None:
    value = store.get_meta(SENTINEL_KEY)
    counts = store.health(include_metrics=True)["counts"]
    if value is None:
        if any(int(count) != 0 for count in counts.values()):
            pytest.fail("refusing to install integration sentinel over non-empty rag3d_v2 tables")
        store.set_meta(SENTINEL_KEY, SENTINEL_VALUE)
        value = store.get_meta(SENTINEL_KEY)
    if value != SENTINEL_VALUE:
        pytest.fail("refusing destructive integration setup: sentinel mismatch")


def _wipe_after_sentinel(store: PgVectorStore) -> None:
    if store.get_meta(SENTINEL_KEY) != SENTINEL_VALUE:
        pytest.fail("refusing destructive integration setup: sentinel missing")
    _bounded_index_ddl(store, f"DROP INDEX IF EXISTS {HNSW_INDEX}")
    store.db.execute("TRUNCATE rag3d_v2_documents RESTART IDENTITY CASCADE")
    store.refresh_capabilities()


@pytest.fixture
def store():
    if not INTEGRATION_ENABLED:
        pytest.skip("real pgvector integration is not explicitly enabled")
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    instance = PgVectorStore(
        DSN,
        dense_dim=3,
        colbert_dim=2,
        ef_search=100,
        iterative_scan="strict_order",
        max_scan_tuples=10_000,
        scan_mem_multiplier=2,
        fingerprint=_fingerprint(),
        search_mode="exact",
        statement_timeout_ms=30_000,
    )
    _assert_or_install_sentinel(instance)
    _wipe_after_sentinel(instance)
    try:
        yield instance
    finally:
        if not instance.closed:
            _wipe_after_sentinel(instance)
            instance.close()


def test_migration_is_idempotent_verified_and_restart_safe(store: PgVectorStore) -> None:
    assert store.get_meta("schema_version") == "1"
    assert store.get_meta("dense_dim") == "3"
    assert store.get_meta("structural_dim") == "2"

    restarted = PgVectorStore(
        DSN, dense_dim=3, colbert_dim=2, fingerprint=_fingerprint()
    )
    try:
        assert restarted.health()["schema_version"] == "1"
        assert restarted.health(include_metrics=True)["counts"] == {
            "documents": 0,
            "chunks": 0,
            "sparse_postings": 0,
        }
    finally:
        restarted.close()

    with pytest.raises(PgVectorSchemaError, match="dense_dim"):
        PgVectorStore(DSN, dense_dim=4, colbert_dim=2)


def test_operator_migration_executes_twice_and_runtime_accepts_it(
    store: PgVectorStore,
) -> None:
    migration = ROOT / "migrations" / "pgvector" / "001_retrieval_v2.sql"
    command = [
        "psql",
        "--no-psqlrc",
        "--dbname",
        DSN,
        "--set=dense_dim=3",
        "--set=structural_dim=2",
        "--file",
        str(migration),
    ]
    for _ in range(2):
        completed = _run_psql(command)
        assert completed.returncode == 0, completed.stderr

    verified = PgVectorStore(
        DSN, dense_dim=3, colbert_dim=2, fingerprint=_fingerprint()
    )
    verified.close()


def test_operator_migration_canonicalizes_zero_padded_dimensions(
    store: PgVectorStore,
) -> None:
    migration = ROOT / "migrations" / "pgvector" / "001_retrieval_v2.sql"
    command = [
        "psql",
        "--no-psqlrc",
        "--dbname",
        DSN,
        "--set=dense_dim=0003",
        "--set=structural_dim=0002",
        "--file",
        str(migration),
    ]

    for _ in range(2):
        completed = _run_psql(command)
        assert completed.returncode == 0, completed.stderr

    assert store.get_meta("dense_dim") == "3"
    assert store.get_meta("structural_dim") == "2"
    reopened = PgVectorStore(
        DSN, dense_dim=3, colbert_dim=2, fingerprint=_fingerprint()
    )
    reopened.close()


@pytest.mark.parametrize(
    "variable_args",
    [[], ["--set=dense_dim=3"]],
)
def test_operator_migration_missing_dimensions_exits_nonzero(
    store: PgVectorStore, variable_args
) -> None:
    migration = ROOT / "migrations" / "pgvector" / "001_retrieval_v2.sql"
    completed = _run_psql(
        [
            "psql",
            "--no-psqlrc",
            "--dbname",
            DSN,
            *variable_args,
            "--file",
            str(migration),
        ]
    )

    assert completed.returncode == 3
    assert "required" in completed.stdout


@pytest.mark.parametrize("state", ["document", "parent", "rolling_summary"])
def test_first_fingerprint_is_not_published_over_any_populated_state(
    store: PgVectorStore, state: str
) -> None:
    payload = _fingerprint().canonical_json()
    digest = _fingerprint().digest
    document_id = store.add_doc(f"legacy-{state}", "Legacy", 1)
    if state == "parent":
        store.add_parent(document_id, "legacy parent", 1, 0)
    elif state == "rolling_summary":
        store.add_chunk(
            document_id,
            "legacy rolling summary",
            "legacy rolling summary",
            1,
            _vector([1, 0, 0], {}, [[1, 0]]),
            kind="rolling_summary",
        )
    store.db.execute(
        "DELETE FROM rag3d_v2_meta WHERE key = ANY(%s::text[])",
        (["retrieval_v2_fingerprint", "retrieval_v2_fingerprint_sha256"],),
    )
    try:
        with pytest.raises(FingerprintMismatchError, match="populated|reindex"):
            PgVectorStore(
                DSN, dense_dim=3, colbert_dim=2, fingerprint=_fingerprint()
            )
        rows = store.db.execute(
            "SELECT key FROM rag3d_v2_meta WHERE key = ANY(%s::text[])",
            (["retrieval_v2_fingerprint", "retrieval_v2_fingerprint_sha256"],),
        ).fetchall()
        assert rows == []
    finally:
        store.db.execute("TRUNCATE rag3d_v2_documents RESTART IDENTITY CASCADE")
        store.db.execute(
            "INSERT INTO rag3d_v2_meta(key,value) VALUES(%s,%s),(%s,%s) "
            "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (
                "retrieval_v2_fingerprint",
                payload,
                "retrieval_v2_fingerprint_sha256",
                digest,
            ),
        )


@pytest.mark.parametrize(
    ("constraint_name", "adulterated", "restored"),
    [
        (
            "rag3d_v2_chunks_position_ck",
            "CHECK (TRUE)",
            "CHECK (pos >= 0 OR (kind = 'summary' AND pos = -1))",
        ),
        (
            "rag3d_v2_chunks_kind_ck",
            "CHECK (kind IN ('chunk','parent','summary','turn',"
            "'rolling_summary','evil'))",
            "CHECK (kind IN ('chunk','parent','summary','turn',"
            "'rolling_summary'))",
        ),
        (
            "rag3d_v2_chunks_position_ck",
            "CHECK (pos >= 0 OR (kind = 'summary' AND pos = -1)) NOT VALID",
            "CHECK (pos >= 0 OR (kind = 'summary' AND pos = -1))",
        ),
    ],
)
def test_runtime_and_operator_migration_reject_constraint_semantic_drift(
    store: PgVectorStore,
    constraint_name: str,
    adulterated: str,
    restored: str,
) -> None:
    store.db.execute(
        f"ALTER TABLE rag3d_v2_chunks DROP CONSTRAINT {constraint_name}"
    )
    store.db.execute(
        f"ALTER TABLE rag3d_v2_chunks ADD CONSTRAINT {constraint_name} "
        f"{adulterated}"
    )
    try:
        with pytest.raises(PgVectorSchemaError, match="constraint catalog"):
            PgVectorStore(
                DSN, dense_dim=3, colbert_dim=2, fingerprint=_fingerprint()
            )

        migration = ROOT / "migrations" / "pgvector" / "001_retrieval_v2.sql"
        completed = _run_psql(
            [
                "psql",
                "--no-psqlrc",
                "--dbname",
                DSN,
                "--set=dense_dim=3",
                "--set=structural_dim=2",
                "--file",
                str(migration),
            ]
        )
        assert completed.returncode != 0
        assert "constraint catalog" in completed.stderr
    finally:
        store.db.execute(
            f"ALTER TABLE rag3d_v2_chunks DROP CONSTRAINT {constraint_name}"
        )
        store.db.execute(
            f"ALTER TABLE rag3d_v2_chunks ADD CONSTRAINT {constraint_name} "
            f"{restored}"
        )


def test_runtime_and_operator_migration_reject_partial_unique_base_index(
    store: PgVectorStore,
) -> None:
    index_name = "rag3d_v2_chunks_turn_idx"
    _bounded_index_ddl(store, f"DROP INDEX {index_name}")
    _bounded_index_ddl(
        store,
        f"CREATE UNIQUE INDEX {index_name} ON rag3d_v2_chunks(turn_no) "
        "WHERE turn_no IS NOT NULL",
    )
    try:
        with pytest.raises(PgVectorSchemaError, match="index"):
            PgVectorStore(
                DSN, dense_dim=3, colbert_dim=2, fingerprint=_fingerprint()
            )

        migration = ROOT / "migrations" / "pgvector" / "001_retrieval_v2.sql"
        completed = _run_psql(
            [
                "psql",
                "--no-psqlrc",
                "--dbname",
                DSN,
                "--set=dense_dim=3",
                "--set=structural_dim=2",
                "--file",
                str(migration),
            ]
        )
        assert completed.returncode != 0
        assert "base index catalog" in completed.stderr
    finally:
        _bounded_index_ddl(store, f"DROP INDEX IF EXISTS {index_name}")
        _bounded_index_ddl(
            store, f"CREATE INDEX {index_name} ON rag3d_v2_chunks(turn_no)"
        )


@pytest.mark.parametrize(
    ("mutation", "restoration"),
    [
        (
            "ALTER TABLE rag3d_v2_documents ADD COLUMN unexpected TEXT",
            "ALTER TABLE rag3d_v2_documents DROP COLUMN unexpected",
        ),
        (
            "ALTER TABLE rag3d_v2_documents ALTER COLUMN title TYPE VARCHAR(255)",
            "ALTER TABLE rag3d_v2_documents ALTER COLUMN title TYPE TEXT",
        ),
        (
            "ALTER TABLE rag3d_v2_documents ALTER COLUMN source DROP NOT NULL",
            "ALTER TABLE rag3d_v2_documents ALTER COLUMN source SET NOT NULL",
        ),
        (
            "ALTER TABLE rag3d_v2_documents ALTER COLUMN metadata DROP DEFAULT",
            "ALTER TABLE rag3d_v2_documents ALTER COLUMN metadata "
            "SET DEFAULT '{}'::jsonb",
        ),
        (
            "ALTER TABLE rag3d_v2_documents ALTER COLUMN id DROP DEFAULT",
            "ALTER TABLE rag3d_v2_documents ALTER COLUMN id SET DEFAULT "
            "nextval('rag3d_v2_documents_id_seq'::regclass)",
        ),
        (
            "ALTER TABLE rag3d_v2_chunks ALTER COLUMN id DROP DEFAULT",
            "ALTER TABLE rag3d_v2_chunks ALTER COLUMN id SET DEFAULT "
            "nextval('rag3d_v2_chunks_id_seq'::regclass)",
        ),
    ],
    ids=[
        "extra-column",
        "type",
        "not-null",
        "declared-default",
        "documents-id-default",
        "chunks-id-default",
    ],
)
def test_runtime_and_operator_migration_reject_complete_column_manifest_drift(
    store: PgVectorStore, mutation: str, restoration: str
) -> None:
    store.db.execute(mutation)
    try:
        with pytest.raises(PgVectorSchemaError, match="schema definition|column"):
            PgVectorStore(
                DSN, dense_dim=3, colbert_dim=2, fingerprint=_fingerprint()
            )

        migration = ROOT / "migrations" / "pgvector" / "001_retrieval_v2.sql"
        completed = _run_psql(
            [
                "psql",
                "--no-psqlrc",
                "--dbname",
                DSN,
                "--set=dense_dim=3",
                "--set=structural_dim=2",
                "--file",
                str(migration),
            ]
        )
        assert completed.returncode != 0
        assert "column catalog" in completed.stderr
    finally:
        store.db.execute(restoration)


def test_fingerprint_is_persisted_and_mismatch_is_rejected(store: PgVectorStore) -> None:
    first = PgVectorStore(DSN, dense_dim=3, colbert_dim=2, fingerprint=_fingerprint())
    first.close()

    compatible = PgVectorStore(DSN, dense_dim=3, colbert_dim=2, fingerprint=_fingerprint())
    compatible.close()

    with pytest.raises(FingerprintMismatchError, match="differing fields: model"):
        PgVectorStore(
            DSN,
            dense_dim=3,
            colbert_dim=2,
            fingerprint=_fingerprint(model="different-model"),
        )

    original_digest = store.get_meta("retrieval_v2_fingerprint_sha256")
    assert original_digest is not None
    store.set_meta("retrieval_v2_fingerprint_sha256", "0" * 64)
    try:
        with pytest.raises(FingerprintMismatchError, match="digest"):
            PgVectorStore(
                DSN, dense_dim=3, colbert_dim=2, fingerprint=_fingerprint()
            )
    finally:
        store.set_meta("retrieval_v2_fingerprint_sha256", original_digest)

    with pytest.raises(FingerprintMismatchError, match="expected.*fingerprint"):
        PgVectorStore(DSN, dense_dim=3, colbert_dim=2)


def test_crud_exact_sparse_structural_filters_and_deletes(store: PgVectorStore) -> None:
    alpha = store.add_doc("alpha-source", "Alpha", 20, {"tenant": "a", "active": True})
    beta = store.add_doc("beta-source", "Beta", 10, {"tenant": "b", "active": False})
    parent = store.add_parent(alpha, "alpha parent", 5, 0)
    first = store.add_chunk(
        alpha,
        "alpha first",
        "alpha context",
        4,
        _vector([1, 0, 0], {10: 2.0}, [[1, 0], [0, 1]]),
        pos=0,
        parent_id=parent,
    )
    second = store.add_chunk(
        alpha,
        "alpha second",
        "alpha context",
        5,
        _vector([0, 1, 0], {10: 0.25, 20: 1.0}, [[0, 1], [-1, 0]]),
        pos=1,
        parent_id=parent,
    )
    third = store.add_chunk(
        beta,
        "beta first",
        "beta context",
        6,
        _vector([0.9, 0.1, 0], {20: 2.0}, [[0.8, 0.2], [0, 1]]),
        pos=0,
    )
    summary = store.add_chunk(
        alpha,
        "alpha summary",
        "alpha summary context",
        3,
        _vector([0, 0, 1], {30: 1.0}, [[1, 0]]),
        kind="summary",
        pos=-1,
    )
    store.commit()

    dense = store.dense_search([1, 0, 0], 3, exact=True)
    assert [chunk_id for chunk_id, _ in dense] == [first, third, second]
    assert dense[0][1] == pytest.approx(1.0, abs=1e-6)
    assert all(math.isfinite(score) for _, score in dense)

    alpha_filter = SearchFilters(scope=SearchScope(sources=("alpha-source",)))
    assert [row[0] for row in store.dense_search([1, 0, 0], 5, filters=alpha_filter)] == [
        first,
        second,
        summary,
    ]
    beta_filter = SearchFilters(metadata={"tenant": "b", "active": False})
    assert store.dense_search([1, 0, 0], 5, filters=beta_filter) == pytest.approx(
        [(third, dense[1][1])]
    )
    parent_filter = SearchFilters(scope=SearchScope(parent_ids=(parent,)))
    assert [row[0] for row in store.dense_search([1, 0, 0], 5, filters=parent_filter)] == [
        first,
        second,
    ]

    sparse = store.sparse_search({10: 1.0}, 5)
    assert [row[0] for row in sparse] == [first, second]
    assert sparse[0][1] > sparse[1][1] > 0
    assert store.sparse_search({20: 1.0}, 5, filters=alpha_filter)[0][0] == second

    structural = store.structural_rerank([[1, 0]], [second, first, third], 3)
    assert structural[0][0] == first
    assert [row[0] for row in store.structural_rerank([[1, 0]], [first, third], 3, filters=beta_filter)] == [third]
    assert store.colbert_scores([[1, 0]], [first, second]) == store.structural_rerank(
        [[1, 0]], [first, second], 2
    )

    hydrated = store.get_chunks([second, 999_999, first])
    assert [row["id"] for row in hydrated] == [second, first]
    assert hydrated[0]["ctx"] == "alpha context"
    vectors = store.dense_vectors([third, first])
    assert set(vectors) == {first, third}
    assert np.linalg.norm(vectors[first]) == pytest.approx(1.0, abs=1e-6)
    assert store.dense_vecs([first]).keys() == {first}
    assert store.neighbors(alpha, [1, 0]) == [
        {
            "id": first,
            "doc_id": alpha,
            "kind": "chunk",
            "pos": 0,
            "text": "alpha first",
        },
        {
            "id": second,
            "doc_id": alpha,
            "kind": "chunk",
            "pos": 1,
            "text": "alpha second",
        },
    ]
    assert store.neighbors(alpha, [-1, 0, 1]) == [
        {
            "id": first,
            "doc_id": alpha,
            "kind": "chunk",
            "pos": 0,
            "text": "alpha first",
        },
        {
            "id": second,
            "doc_id": alpha,
            "kind": "chunk",
            "pos": 1,
            "text": "alpha second",
        },
    ]
    assert store.corpus_tokens() == 15
    assert [row["id"] for row in store.all_texts()] == [first, second, third]
    assert store.n_chunks() == 4
    assert store.get_chunks([summary])[0]["pos"] == -1
    assert store.last_turn_no() == 0

    store.touch_access([first], 7)
    assert store.get_chunks([first])[0]["accessed_turn"] == 7

    store.delete_chunk(second)
    assert store.get_chunks([second]) == []
    store.delete_document(beta)
    assert store.get_chunks([third]) == []
    assert store.health(include_metrics=True)["counts"]["documents"] == 1


def test_transaction_rolls_back_a_composite_write(store: PgVectorStore) -> None:
    before = store.health(include_metrics=True)["counts"].copy()
    with pytest.raises(RuntimeError, match="force rollback"):
        with store.transaction():
            document_id = store.add_doc("rollback-source", "Rollback", 3)
            store.add_chunk(
                document_id,
                "must disappear",
                "must disappear",
                3,
                _vector([1, 0, 0], {1: 1}, [[1, 0]]),
            )
            raise RuntimeError("force rollback")
    assert store.health(include_metrics=True)["counts"] == before


def _hold_advisory_lock(lock_key: int, ready: threading.Event, release: threading.Event) -> None:
    import psycopg

    connection = psycopg.connect(DSN, autocommit=True)
    try:
        with connection.transaction():
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key,))
            ready.set()
            release.wait(timeout=2.0)
    finally:
        connection.close()


def test_runtime_migration_advisory_lock_wait_is_bounded(
    store: PgVectorStore,
) -> None:
    ready = threading.Event()
    release = threading.Event()
    blocker = threading.Thread(
        target=_hold_advisory_lock,
        args=(_MIGRATION_LOCK_KEY, ready, release),
        daemon=True,
    )
    blocker.start()
    assert ready.wait(timeout=2.0)
    started = time.monotonic()
    try:
        with pytest.raises(PgVectorError, match="migration lock"):
            PgVectorStore(
                DSN,
                dense_dim=3,
                colbert_dim=2,
                fingerprint=_fingerprint(),
                lock_timeout_ms=100,
            )
    finally:
        release.set()
        blocker.join(timeout=2.0)

    assert time.monotonic() - started < 1.5


def test_fingerprint_advisory_lock_wait_is_bounded(
    store: PgVectorStore,
) -> None:
    ready = threading.Event()
    release = threading.Event()
    blocker = threading.Thread(
        target=_hold_advisory_lock,
        args=(_FINGERPRINT_LOCK_KEY, ready, release),
        daemon=True,
    )
    blocker.start()
    assert ready.wait(timeout=2.0)
    store.lock_timeout_ms = 100
    started = time.monotonic()
    try:
        with pytest.raises(PgVectorError, match="fingerprint lock"):
            with store.transaction():
                store.lock_fingerprint()
    finally:
        release.set()
        blocker.join(timeout=2.0)

    assert time.monotonic() - started < 1.5


def test_public_trirag_v2_pgvector_short_large_summary_search_stats_and_close(
    store: PgVectorStore, tmp_path: Path
) -> None:
    fingerprint_keys = [
        "retrieval_v2_fingerprint",
        "retrieval_v2_fingerprint_sha256",
        "encoder",
    ]
    original_meta = {
        key: store.get_meta(key)
        for key in fingerprint_keys
        if store.get_meta(key) is not None
    }
    store.db.execute(
        "DELETE FROM rag3d_v2_meta WHERE key = ANY(%s::text[])",
        (fingerprint_keys,),
    )

    rag = None
    try:
        cfg = TriRagConfig(
            data_dir=tmp_path,
            pg_dsn=DSN,
            backend="pgvector",
            retrieval_pipeline="v2",
            encoder="hash",
            dense_dim=3,
            colbert_dim=2,
            max_colbert_tokens=8,
            chunk_tokens=12,
            chunk_overlap=2,
            parent_tokens=24,
            tiny_doc_tokens=16,
            huge_doc_tokens=40,
            contextual_enrich=False,
            top_k=3,
            channel_k=10,
            structural_candidate_depth=10,
            structural_rerank=True,
            reranker="none",
            diversity_method="none",
            allow_encoder_fallback=False,
            pgvector_search_mode="exact",
            remember_chat=False,
        )

        def summarize(_system, _messages, _max_tokens):
            return "Deterministic orbital catalog summary marker."

        rag = TriRag(cfg, llm=summarize)
        short = rag.ingest(
            "Mercury is the closest planet to the Sun.",
            source="e2e-short",
            title="Short planets",
        )
        large_text = " ".join(
            f"Section {index} records deterministic orbital catalog entry {index}."
            for index in range(24)
        )
        large = rag.ingest(
            large_text,
            source="e2e-large",
            title="Large orbital catalog",
        )

        assert short["chunks"] == 1
        assert large["chunks"] > 1
        summaries = rag.store.all_texts(kinds=("summary",))
        assert len(summaries) == 1
        summary = rag.store.get_chunks([summaries[0]["id"]])[0]
        assert summary["kind"] == "summary"
        assert summary["pos"] == -1
        assert "summary marker" in summary["text"]

        result = rag.search("orbital catalog summary marker", top_k=3)
        assert result.fused
        stats = rag.stats()
        assert stats["backend"] == "pgvector"
        assert stats["pipeline"] == "v2"
        assert stats["pgvector_search_mode"] == "exact"
    finally:
        if rag is not None and not rag.store.closed:
            rag.store.close()
        store.db.execute("TRUNCATE rag3d_v2_documents RESTART IDENTITY CASCADE")
        store.db.execute(
            "DELETE FROM rag3d_v2_meta WHERE key = ANY(%s::text[])",
            (fingerprint_keys,),
        )
        for key, value in original_meta.items():
            store.set_meta(key, value)

    assert rag is not None
    assert rag.store.closed is True


def test_invalid_vectors_weights_candidates_and_limits_fail_closed(store: PgVectorStore) -> None:
    with pytest.raises(ValueError, match="non-zero norm"):
        store.dense_search([0, 0, 0], 1)
    with pytest.raises(ValueError, match="finite"):
        store.dense_search([1, float("nan"), 0], 1)
    with pytest.raises(ValueError, match="dimension"):
        store.dense_search([1, 0], 1)
    with pytest.raises(ValueError, match="non-negative"):
        store.dense_search([1, 0, 0], -1)
    with pytest.raises(ValueError, match="finite"):
        store.sparse_search({1: float("inf")}, 1)
    with pytest.raises(ValueError, match="structural_dim"):
        store.structural_rerank([[1, 0, 0]], [1], 1)
    with pytest.raises(ValueError, match="candidate_ids"):
        store.structural_rerank([[1, 0]], list(range(1_001)), 1)
    with pytest.raises(ValueError, match="structural vectors.*maximum of 32"):
        store.add_chunk(
            None,
            "too many passage tokens",
            "",
            1,
            _vector([1, 0, 0], {}, np.ones((33, 2), dtype=np.float32)),
        )
    with pytest.raises(ValueError, match="query vectors.*maximum of 32"):
        store.structural_rerank(
            np.ones((33, 2), dtype=np.float32), [1], 1
        )
    with pytest.raises(ValueError, match="summary"):
        store.add_chunk(
            None,
            "invalid negative position",
            "",
            1,
            _vector([1, 0, 0], {}, [[1, 0]]),
            kind="turn",
            pos=-1,
        )


def _populate_hnsw_corpus(store: PgVectorStore, count: int = 12_000):
    document_id = store.add_doc("hnsw-corpus", "HNSW corpus", count)
    structural = np.asarray([[1.0, 0.0]], dtype=np.float16).tobytes()
    ids = [
        int(row[0])
        for row in store.db.execute(
            """
            INSERT INTO rag3d_v2_chunks(
              document_id,parent_id,kind,pos,text,context,n_tokens,created,
              importance,turn_no,embedding,structural_n_tok,structural_dim,
              structural_data
            )
            SELECT %s,NULL,'chunk',point,
                   'point ' || point::text,'synthetic integration corpus',1,
                   extract(epoch FROM clock_timestamp()),0.5,NULL,
                   ARRAY[
                     1.0,
                     sqrt((point + 1)::double precision) / 100.0,
                     0.01
                   ]::vector(3),1,2,%s
            FROM generate_series(0,%s - 1) AS point
            RETURNING id
            """,
            (document_id, structural, count),
        ).fetchall()
    ]
    store.db.execute("ANALYZE rag3d_v2_chunks")
    return ids


def test_hnsw_build_capability_recall_and_local_gucs(store: PgVectorStore) -> None:
    ids = _populate_hnsw_corpus(store)
    assert store.capabilities.ann_dense_search is False

    status = store.create_hnsw_index(m=8, ef_construction=32)
    assert status["valid"] is True
    assert status["ready"] is True
    assert status["created_by_caller"] is True
    assert status["opclass"] == "vector_cosine_ops"
    assert store.capabilities.ann_dense_search is True
    existing = store.create_hnsw_index(m=8, ef_construction=32)
    assert existing["created_by_caller"] is False
    assert {
        key: value for key, value in existing.items() if key != "created_by_caller"
    } == {
        key: value for key, value in status.items() if key != "created_by_caller"
    }
    with pytest.raises(PgVectorSchemaError, match="options"):
        store.create_hnsw_index(m=16, ef_construction=64)

    before = {
        "ef_search": store.db.execute("SHOW hnsw.ef_search").fetchone()[0],
        "iterative_scan": store.db.execute("SHOW hnsw.iterative_scan").fetchone()[0],
        "max_scan_tuples": store.db.execute("SHOW hnsw.max_scan_tuples").fetchone()[0],
        "scan_mem_multiplier": store.db.execute("SHOW hnsw.scan_mem_multiplier").fetchone()[0],
        "enable_indexscan": store.db.execute("SHOW enable_indexscan").fetchone()[0],
        "statement_timeout": store.db.execute("SHOW statement_timeout").fetchone()[0],
    }
    exact = store.dense_search([1, 0, 0], 10, exact=True)
    assert [chunk_id for chunk_id, _ in exact] == ids[:10]
    # The monotonic fixture has unique distances, so ID recall is deterministic.
    # The high ef_search remains a test control, not a production recommendation.
    store.ef_search = 1_000
    approximate = store.dense_search([1, 0, 0], 10, exact=False)
    recall = len({row[0] for row in exact} & {row[0] for row in approximate}) / len(exact)
    assert recall >= 0.8
    after = {
        "ef_search": store.db.execute("SHOW hnsw.ef_search").fetchone()[0],
        "iterative_scan": store.db.execute("SHOW hnsw.iterative_scan").fetchone()[0],
        "max_scan_tuples": store.db.execute("SHOW hnsw.max_scan_tuples").fetchone()[0],
        "scan_mem_multiplier": store.db.execute("SHOW hnsw.scan_mem_multiplier").fetchone()[0],
        "enable_indexscan": store.db.execute("SHOW enable_indexscan").fetchone()[0],
        "statement_timeout": store.db.execute("SHOW statement_timeout").fetchone()[0],
    }
    assert after == before

    exact_audit = store.explain_dense([1, 0, 0], 10, exact=True)
    assert HNSW_INDEX not in exact_audit["plan"]["index_names"]
    assert exact_audit["settings"].get("enable_indexscan") == "off"
    assert "dsn" not in json.dumps(exact_audit).lower()


def test_natural_ann_plan_reports_hnsw_without_planner_forcing(store: PgVectorStore) -> None:
    _populate_hnsw_corpus(store, count=20_000)
    store.create_hnsw_index(m=8, ef_construction=32)
    audit = store.explain_dense([1, 0, 0], 10, exact=False)
    assert "enable_seqscan" not in audit["settings"]
    assert HNSW_INDEX in audit["plan"]["index_names"]
    assert audit["plan"]["hnsw_used"] is True


def test_concurrent_hnsw_build_runs_outside_a_transaction(store: PgVectorStore) -> None:
    _populate_hnsw_corpus(store, count=500)
    status = store.create_hnsw_index(m=8, ef_construction=32, concurrently=True)
    assert status["valid"] and status["ready"]


def test_health_and_errors_never_expose_the_dsn(store: PgVectorStore) -> None:
    health = store.health()
    serialized = json.dumps(health, sort_keys=True)
    assert health["backend"] == "pgvector"
    assert health["status"] == "ok"
    assert health["search_mode"] == "exact"
    assert "counts" not in health
    assert health["pgvector_version"] == "0.8.5"
    assert "postgresql://" not in serialized
    assert "host=" not in serialized
    assert "dsn" not in serialized.lower()


def test_psql_harness_applies_connection_server_and_process_timeouts(
    monkeypatch,
) -> None:
    recorded = {}

    def fake_run(command, **kwargs):
        recorded["command"] = command
        recorded.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    completed = _run_psql(["psql", "--file", "migration.sql"])

    assert completed.returncode == 0
    assert recorded["timeout"] == 75
    assert recorded["capture_output"] is True
    assert recorded["text"] is True
    assert recorded["env"]["PGCONNECT_TIMEOUT"] == "5"
    assert "statement_timeout=60000" in recorded["env"]["PGOPTIONS"]
    assert "lock_timeout=5000" in recorded["env"]["PGOPTIONS"]


def test_psql_harness_timeout_error_does_not_expose_command_secrets(
    monkeypatch,
) -> None:
    secret = _SENSITIVE_DETAIL

    def timeout(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, 75)

    monkeypatch.setattr(subprocess, "run", timeout)

    with pytest.raises(RuntimeError) as caught:
        _run_psql(["psql", "--dbname", secret])

    assert "timed out" in str(caught.value)
    assert secret not in str(caught.value)
    assert _SENSITIVE_DETAIL not in str(caught.value)


def test_fixture_index_ddl_uses_local_bounded_timeouts() -> None:
    class Transaction:
        def __init__(self, database):
            self.database = database

        def __enter__(self):
            self.database.calls.append(("BEGIN", ()))

        def __exit__(self, _exc_type, _exc, _traceback):
            self.database.calls.append(("END", ()))

    class Database:
        def __init__(self):
            self.calls = []

        def transaction(self):
            return Transaction(self)

        def execute(self, sql, params=()):
            self.calls.append((sql, params))

    database = Database()
    fake_store = SimpleNamespace(
        db=database,
        lock_timeout_ms=321,
        statement_timeout_ms=654,
    )

    _bounded_index_ddl(fake_store, "DROP INDEX IF EXISTS safe_index")

    assert database.calls == [
        ("BEGIN", ()),
        ("SELECT set_config('lock_timeout',%s,true)", ("321",)),
        ("SELECT set_config('statement_timeout',%s,true)", ("654",)),
        ("DROP INDEX IF EXISTS safe_index", ()),
        ("END", ()),
    ]


def test_concurrent_hnsw_ddl_has_session_timeouts_without_transaction() -> None:
    class Cursor:
        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    class Database:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params=()):
            self.calls.append((sql, params))
            if "current_setting('lock_timeout')" in sql:
                return Cursor(("0", "0"))
            return Cursor()

        def transaction(self):
            raise AssertionError("CONCURRENTLY must not open a transaction")

    store = object.__new__(PgVectorStore)
    store.db = Database()
    store._fingerprint_verified = True
    store._transaction_depth = 0
    store.lock_timeout_ms = 4321
    store.statement_timeout_ms = 54_321
    store._hnsw_status_cache = store._empty_hnsw_status()
    store._capabilities_cache = store._capabilities_for_status(
        store._hnsw_status_cache
    )
    ready = {
        **store._empty_hnsw_status(),
        "exists": True,
        "valid": True,
        "ready": True,
        "definition_valid": True,
        "method": "hnsw",
        "opclass": "vector_cosine_ops",
        "column": "embedding",
        "vector_type": "vector(3)",
        "options": {"m": 8, "ef_construction": 32},
    }
    statuses = iter([store._empty_hnsw_status(), ready])
    store._inspect_hnsw_index = lambda: next(statuses)

    status = store.create_hnsw_index(
        m=8,
        ef_construction=32,
        concurrently=True,
    )

    assert status["valid"] is True
    assert store.db.calls[0] == (
        "SELECT current_setting('lock_timeout'), "
        "current_setting('statement_timeout')",
        (),
    )
    assert store.db.calls[1] == (
        "SELECT set_config('lock_timeout',%s,false), "
        "set_config('statement_timeout',%s,false)",
        ("4321", "54321"),
    )
    assert "CREATE INDEX CONCURRENTLY" in store.db.calls[2][0]
    assert store.db.calls[3] == (
        "SELECT set_config('lock_timeout',%s,false), "
        "set_config('statement_timeout',%s,false)",
        ("0", "0"),
    )


def test_concurrent_hnsw_timeout_failure_restores_gucs_and_masks_secret() -> None:
    secret = _SENSITIVE_DETAIL

    class Cursor:
        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    class Database:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params=()):
            self.calls.append((sql, params))
            if "current_setting('lock_timeout')" in sql:
                return Cursor(("0", "0"))
            if "CREATE INDEX CONCURRENTLY" in sql:
                raise RuntimeError(secret)
            return Cursor()

        def transaction(self):
            raise AssertionError("CONCURRENTLY must not open a transaction")

    store = object.__new__(PgVectorStore)
    store.db = Database()
    store._fingerprint_verified = True
    store._transaction_depth = 0
    store.lock_timeout_ms = 100
    store.statement_timeout_ms = 200
    store._hnsw_status_cache = store._empty_hnsw_status()
    store._capabilities_cache = store._capabilities_for_status(
        store._hnsw_status_cache
    )
    statuses = iter(
        [store._empty_hnsw_status(), store._empty_hnsw_status()]
    )
    store._inspect_hnsw_index = lambda: next(statuses)

    with pytest.raises(PgVectorHnswError) as caught:
        store.create_hnsw_index(
            m=8,
            ef_construction=32,
            concurrently=True,
        )

    assert secret not in str(caught.value)
    assert _SENSITIVE_DETAIL not in str(caught.value)
    assert store.db.calls[-1] == (
        "SELECT set_config('lock_timeout',%s,false), "
        "set_config('statement_timeout',%s,false)",
        ("0", "0"),
    )

"""Unit tests for the optional PostgreSQL + pgvector backend.

These tests deliberately do not require PostgreSQL, psycopg, or pgvector-python.
The real-database contract lives in ``test_pgvector_integration.py``.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag3d.backend import (
    DEFAULT_RETRIEVAL_LIMITS,
    BackendCapabilities,
    FingerprintMismatchError,
    IndexFingerprint,
    SearchFilters,
    SearchScope,
)
from rag3d.pgvector_store import (
    PgVectorError,
    PgVectorExtensionError,
    PgVectorHnswError,
    PgVectorSchemaError,
    PgVectorStore,
    _EXPECTED_INDEX_COLUMNS,
    _build_filter_clause,
    _constraint_definition_matches,
    _expected_columns,
    _expected_constraint_catalog,
    _normalize_dense_vector,
    _parse_extension_version,
    _sanitize_explain_document,
    _schema_statements,
    _validate_ann_options,
    _validate_chunk_position,
    _validate_dimension,
    _validate_hnsw_build_options,
    _validate_k,
    _validate_search_mode,
    _validate_signed_positions,
    _validate_sparse_weights,
    _validate_statement_timeout,
)


class _OversizedExplodingSparseMapping(Mapping[int, float]):
    def __getitem__(self, key: int) -> float:
        raise KeyError(key)

    def __iter__(self):
        raise AssertionError("oversized sparse mapping was iterated")

    def __len__(self) -> int:
        return DEFAULT_RETRIEVAL_LIMITS.max_sparse_terms + 1

    def items(self):
        raise AssertionError("oversized sparse mapping items were requested")


class _LyingLengthSparseMapping(Mapping[int, float]):
    def __init__(self) -> None:
        self.yielded = 0

    def __getitem__(self, key: int) -> float:
        if 0 <= key <= DEFAULT_RETRIEVAL_LIMITS.max_sparse_terms:
            return 1.0
        raise KeyError(key)

    def __iter__(self):
        return iter(range(DEFAULT_RETRIEVAL_LIMITS.max_sparse_terms + 1))

    def __len__(self) -> int:
        return 1

    def items(self):
        for term in range(DEFAULT_RETRIEVAL_LIMITS.max_sparse_terms + 1):
            self.yielded += 1
            yield term, 1.0


class _OversizedExplodingStructural:
    def __init__(self, size: int) -> None:
        self.size = size
        self.iterated = False

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        self.iterated = True
        raise AssertionError("oversized structural vectors were iterated")


class _LyingLengthStructural:
    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.yielded = 0

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        for _ in range(self.rows):
            self.yielded += 1
            yield [1.0, 0.0]

def test_module_import_does_not_load_optional_database_packages() -> None:
    code = (
        "import json,sys; import rag3d.pgvector_store; "
        "print(json.dumps({'psycopg': 'psycopg' in sys.modules, "
        "'pgvector': any(k == 'pgvector' or k.startswith('pgvector.') for k in sys.modules)}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"psycopg": False, "pgvector": False}


def test_versioned_operator_migration_matches_runtime_schema_contract() -> None:
    migration = ROOT / "migrations" / "pgvector" / "001_retrieval_v2.sql"
    sql = migration.read_text(encoding="utf-8")
    runtime_sql = "\n".join(_schema_statements(1024, 128))

    assert "CREATE EXTENSION" not in sql.upper()
    assert "\\set ON_ERROR_STOP on" in sql
    assert "pg_advisory_xact_lock" in sql
    assert "SET LOCAL lock_timeout" in sql
    assert "SET LOCAL statement_timeout" in sql
    assert "pg_get_serial_sequence" in sql
    assert "attidentity" in sql
    assert "incompatible pgvector column catalog" in sql
    assert "indisunique" in sql
    assert "indpred IS NULL" in sql
    assert ":dense_dim" in sql
    assert ":structural_dim" in sql
    for table in (
        "rag3d_v2_meta",
        "rag3d_v2_documents",
        "rag3d_v2_chunks",
        "rag3d_v2_sparse_postings",
    ):
        assert f"CREATE TABLE IF NOT EXISTS public.{table}" in sql
        assert f"CREATE TABLE IF NOT EXISTS {table}" in runtime_sql
    for index_name in (
        "rag3d_v2_documents_source_idx",
        "rag3d_v2_chunks_document_idx",
        "rag3d_v2_chunks_parent_idx",
        "rag3d_v2_chunks_kind_idx",
        "rag3d_v2_chunks_turn_idx",
        "rag3d_v2_sparse_postings_chunk_idx",
    ):
        assert index_name in sql
        assert index_name in runtime_sql


@pytest.mark.parametrize("value", [0, -1, 2001, True, 3.0, "3"])
def test_dense_dimension_is_strictly_bounded(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="dense_dim"):
        _validate_dimension("dense_dim", value)


def test_dense_vector_is_l2_normalized_as_float32() -> None:
    vector = _normalize_dense_vector([3.0, 4.0, 0.0], 3, name="embedding")
    assert vector.dtype == np.float32
    assert vector.tolist() == pytest.approx([0.6, 0.8, 0.0], abs=1e-6)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "vector, message",
    [
        ([1.0, 2.0], "dimension"),
        ([[1.0, 2.0, 3.0]], "one-dimensional"),
        ([0.0, 0.0, 0.0], "non-zero norm"),
        ([math.nan, 0.0, 1.0], "finite"),
        ([math.inf, 0.0, 1.0], "finite"),
    ],
)
def test_dense_vector_rejects_invalid_input(vector: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _normalize_dense_vector(vector, 3, name="query_vector")


def test_pgvector_sparse_limit_is_independent_from_candidate_pool_limit() -> None:
    weights = {term: 1.0 for term in range(1_100)}

    assert _validate_sparse_weights(weights) == weights


def test_pgvector_sparse_limit_preflights_len_before_items() -> None:
    with pytest.raises(ValueError, match="sparse terms exceed maximum of 8192"):
        _validate_sparse_weights(_OversizedExplodingSparseMapping())


def test_pgvector_sparse_limit_counts_items_from_a_lying_mapping() -> None:
    weights = _LyingLengthSparseMapping()

    with pytest.raises(ValueError, match="sparse terms exceed maximum of 8192"):
        _validate_sparse_weights(weights)

    assert weights.yielded == DEFAULT_RETRIEVAL_LIMITS.max_sparse_terms + 1


@pytest.mark.parametrize("value", [-1, 1001, True, 1.5])
def test_top_k_is_strictly_bounded(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="k"):
        _validate_k(value)


def test_zero_top_k_is_an_empty_request() -> None:
    assert _validate_k(0) == 0


@pytest.mark.parametrize(
    "m, ef_construction, message",
    [
        (1, 64, "m"),
        (101, 204, "m"),
        (16, 3, "ef_construction"),
        (16, 1001, "ef_construction"),
        (16, 31, "at least 2 \u00d7 m"),
        (True, 64, "m"),
    ],
)
def test_hnsw_build_options_follow_pgvector_bounds(
    m: object, ef_construction: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _validate_hnsw_build_options(m, ef_construction)


def test_hnsw_build_options_accept_official_baseline() -> None:
    assert _validate_hnsw_build_options(16, 64) == (16, 64)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"ef_search": 0}, "ef_search"),
        ({"ef_search": 1001}, "ef_search"),
        ({"iterative_scan": "invalid"}, "iterative_scan"),
        ({"max_scan_tuples": 0}, "max_scan_tuples"),
        ({"scan_mem_multiplier": 0.99}, "scan_mem_multiplier"),
        ({"scan_mem_multiplier": math.inf}, "scan_mem_multiplier"),
    ],
)
def test_ann_query_options_are_bounded(kwargs: dict, message: str) -> None:
    defaults = {
        "ef_search": 40,
        "iterative_scan": "off",
        "max_scan_tuples": 20_000,
        "scan_mem_multiplier": 1.0,
        "extension_version": (0, 8, 6),
    }
    defaults.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        _validate_ann_options(**defaults)


def test_ann_max_scan_tuples_has_an_explicit_upper_bound() -> None:
    with pytest.raises(ValueError, match="max_scan_tuples"):
        _validate_ann_options(
            ef_search=40,
            iterative_scan="off",
            max_scan_tuples=1_000_001,
            scan_mem_multiplier=1,
            extension_version=(0, 8, 6),
        )


@pytest.mark.parametrize("value", [0, 60_001, True, 1.5])
def test_statement_timeout_is_strictly_bounded(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="statement_timeout"):
        _validate_statement_timeout(value)


@pytest.mark.parametrize("value", ["exact", "ann", "auto"])
def test_search_mode_is_closed_and_normalized(value: str) -> None:
    assert _validate_search_mode(value.upper()) == value


def test_search_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="search_mode"):
        _validate_search_mode("fast")


def test_summary_is_the_only_kind_allowed_to_use_negative_one_position() -> None:
    assert _validate_chunk_position("summary", -1) == -1
    assert _validate_chunk_position("summary", 0) == 0
    with pytest.raises(ValueError, match="summary"):
        _validate_chunk_position("chunk", -1)
    with pytest.raises(ValueError, match="pos"):
        _validate_chunk_position("summary", -2)


def test_neighbor_probes_accept_bounded_negative_positions() -> None:
    assert _validate_signed_positions([-1, 0, 1]) == [-1, 0, 1]
    with pytest.raises(ValueError, match="positions"):
        _validate_signed_positions([-(2**31) - 1])


def test_iterative_scan_requires_pgvector_0_8() -> None:
    with pytest.raises(ValueError, match="pgvector >= 0.8.0"):
        _validate_ann_options(
            ef_search=40,
            iterative_scan="strict_order",
            max_scan_tuples=20_000,
            scan_mem_multiplier=1,
            extension_version=(0, 7, 4),
        )


def test_ann_query_options_accept_supported_values() -> None:
    assert _validate_ann_options(
        ef_search=80,
        iterative_scan="relaxed_order",
        max_scan_tuples=40_000,
        scan_mem_multiplier=2,
        extension_version=(0, 8, 5),
    ) == (80, "relaxed_order", 40_000, 2.0)


@pytest.mark.parametrize(
    "raw, expected",
    [("0.8.6", (0, 8, 6)), ("0.8.0beta1", (0, 8, 0)), ("1.0", (1, 0, 0))],
)
def test_extension_version_parser(raw: str, expected: tuple) -> None:
    assert _parse_extension_version(raw) == expected


def test_filter_clause_binds_all_user_values() -> None:
    marker = "source-value-must-not-appear-in-sql"
    filters = SearchFilters(
        scope=SearchScope(
            kinds=("chunk",),
            document_ids=(7, 9),
            parent_ids=(3,),
            sources=(marker,),
        ),
        metadata={"tenant": "acme", "active": True},
    )
    sql, params = _build_filter_clause(filters)

    assert marker not in sql
    assert "acme" not in sql
    assert "ANY(%s::text[])" in sql
    assert "ANY(%s::bigint[])" in sql
    assert "@> %s::jsonb" in sql
    assert any(marker == value or marker in value for value in params)
    metadata_param = next(value for value in params if isinstance(value, str) and value.startswith("{"))
    assert json.loads(metadata_param) == {"tenant": "acme", "active": True}


def test_filter_clause_has_closed_default_retrieval_kinds() -> None:
    sql, params = _build_filter_clause(None)
    assert "c.kind IN ('chunk','turn','summary')" in sql
    assert params == []


def test_explain_sanitizer_keeps_audit_fields_and_drops_query_data() -> None:
    raw = [
        {
            "Plan": {
                "Node Type": "Limit",
                "Actual Rows": 4,
                "Shared Hit Blocks": 5,
                "Plans": [
                    {
                        "Node Type": "Index Scan",
                        "Index Name": "rag3d_v2_chunks_embedding_hnsw",
                        "Rows Removed by Filter": 2,
                        "Actual Total Time": 0.25,
                        "Filter": "metadata @> '{\"secret\": \"never-return\"}'",
                    }
                ],
            },
            "Planning Time": 0.1,
            "Execution Time": 0.4,
            "Settings": {
                "hnsw.ef_search": "80",
                "application_name": "private-value",
            },
            "Query Identifier": 123456,
        }
    ]

    audit = _sanitize_explain_document(raw, exact=False)
    serialized = json.dumps(audit, sort_keys=True)

    assert audit["mode"] == "ann"
    assert audit["plan"]["index_names"] == ["rag3d_v2_chunks_embedding_hnsw"]
    assert audit["plan"]["rows_removed_by_filter"] == 2
    assert audit["plan"]["buffers"]["shared_hit"] == 5
    assert audit["settings"] == {"hnsw.ef_search": "80"}
    assert "never-return" not in serialized
    assert "private-value" not in serialized
    assert "123456" not in serialized


def test_explain_sanitizer_uses_closed_index_and_setting_allowlists() -> None:
    raw = [
        {
            "Plan": {
                "Node Type": "Index Scan",
                "Index Name": "tenant_private_index",
            },
            "Settings": {
                "hnsw.future_private_setting": "secret",
                "hnsw.max_scan_tuples": "20000",
            },
        }
    ]

    audit = _sanitize_explain_document(raw, exact=False)

    assert audit["plan"]["index_names"] == []
    assert audit["settings"] == {"hnsw.max_scan_tuples": "20000"}


class _Rows:
    def __init__(self, *, one=None, all_rows=()):
        self._one = one
        self._all = list(all_rows)

    def fetchone(self):
        return self._one

    def fetchall(self):
        return list(self._all)


class _RecordingDatabase:
    def __init__(self, *, plan=None, rows=()):
        self.calls = []
        self.plan = plan
        self.rows = list(rows)

    @contextmanager
    def transaction(self):
        yield

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        if sql.startswith("EXPLAIN (FORMAT JSON)"):
            return _Rows(one=(self.plan,))
        if "SELECT chunk_id,score FROM scored" in sql:
            return _Rows(all_rows=self.rows)
        return _Rows(all_rows=self.rows)


def _bare_search_store(database: _RecordingDatabase) -> PgVectorStore:
    store = object.__new__(PgVectorStore)
    store.db = database
    store.dense_dim = 3
    store.colbert_dim = 2
    store._closed = False
    store._transaction_depth = 0
    store._fingerprint_verified = True
    store._max_structural_tokens = 8
    store._query_max_tokens = 8
    store.lock_timeout_ms = 250
    store.search_mode = "ann"
    store.statement_timeout_ms = 5_000
    store.ef_search = 40
    store.iterative_scan = "off"
    store.max_scan_tuples = 20_000
    store.scan_mem_multiplier = 1.0
    store._pgvector_version_tuple = (0, 8, 5)
    store._last_dense_mode = None
    store._capabilities_cache = BackendCapabilities(
        exact_dense_search=True,
        ann_dense_search=True,
        sparse_search=True,
        structural_rerank=True,
        metadata_filters=True,
        transactions=True,
        native_vector=True,
    )
    return store


def test_add_chunk_preflights_fingerprint_structural_token_limit() -> None:
    database = _RecordingDatabase()
    store = _bare_search_store(database)
    store._max_structural_tokens = 2
    tokens = _OversizedExplodingStructural(3)
    vector = type(
        "Vector",
        (),
        {"dense": [1.0, 0.0, 0.0], "sparse": {}, "tokens": tokens},
    )()

    with pytest.raises(ValueError, match="structural vectors.*maximum of 2"):
        store.add_chunk(None, "text", "context", 1, vector)

    assert tokens.iterated is False
    assert database.calls == []


def test_structural_rerank_counts_a_sized_iterable_that_lies_about_len() -> None:
    database = _RecordingDatabase()
    store = _bare_search_store(database)
    store._query_max_tokens = 2
    tokens = _LyingLengthStructural(3)

    with pytest.raises(ValueError, match="query vectors.*maximum of 2"):
        store.structural_rerank(tokens, [1], 1)

    assert tokens.yielded == 3
    assert database.calls == []


def test_structural_query_rejects_rows_times_dimension_before_database_io() -> None:
    maximum_values = getattr(
        DEFAULT_RETRIEVAL_LIMITS, "max_structural_values", 262_144
    )
    database = _RecordingDatabase()
    store = _bare_search_store(database)
    store._query_max_tokens = maximum_values
    rows = maximum_values // store.colbert_dim + 1
    query = np.ones((rows, store.colbert_dim), dtype=np.float32)

    with pytest.raises(ValueError, match="query vectors.*structural values"):
        store.structural_rerank(query, [1], 1)

    assert database.calls == []


def test_structural_rerank_rejects_oversized_stored_payload_before_copy() -> None:
    maximum_values = getattr(
        DEFAULT_RETRIEVAL_LIMITS, "max_structural_values", 262_144
    )

    class BombPayload:
        def __bytes__(self):
            raise AssertionError("oversized structural payload was copied")

    n_tokens = maximum_values // 2 + 1
    database = _RecordingDatabase(rows=[(1, n_tokens, 2, BombPayload())])
    store = _bare_search_store(database)
    store._max_structural_tokens = n_tokens

    with pytest.raises(PgVectorSchemaError, match="stored structural.*limit"):
        store.structural_rerank([[1.0, 0.0]], [1], 1)


def test_lock_fingerprint_requires_an_active_store_transaction() -> None:
    database = _RecordingDatabase()
    store = _bare_search_store(database)

    with pytest.raises(PgVectorError, match="active transaction"):
        store.lock_fingerprint()

    assert database.calls == []


def test_lock_fingerprint_sets_bounded_local_timeout_before_advisory_lock() -> None:
    database = _RecordingDatabase()
    store = _bare_search_store(database)

    with store.transaction():
        store.lock_fingerprint()

    assert "set_config('lock_timeout'" in database.calls[0][0]
    assert database.calls[0][1] == ("250",)
    assert "pg_advisory_xact_lock" in database.calls[1][0]


def test_lock_fingerprint_masks_driver_errors_and_secrets() -> None:
    secret = "host=private.invalid password=super-secret"

    class BrokenDatabase:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError(secret)

    store = _bare_search_store(BrokenDatabase())  # type: ignore[arg-type]
    store._transaction_depth = 1

    with pytest.raises(PgVectorError) as caught:
        store.lock_fingerprint()

    assert secret not in str(caught.value)
    assert "super-secret" not in str(caught.value)


def test_runtime_migration_sets_bounded_local_timeouts_before_ddl(monkeypatch) -> None:
    class MigrationCursor:
        def __init__(self) -> None:
            self.calls = []

        def execute(self, sql, params=()):
            self.calls.append((sql, params))

    class MigrationDatabase:
        def __init__(self) -> None:
            self.cursor_instance = MigrationCursor()

        @contextmanager
        def transaction(self):
            yield

        @contextmanager
        def cursor(self):
            yield self.cursor_instance

    database = MigrationDatabase()
    store = object.__new__(PgVectorStore)
    store.db = database
    store.dense_dim = 3
    store.colbert_dim = 2
    store.lock_timeout_ms = 250
    store.statement_timeout_ms = 5_000
    store._verify_schema = lambda _cursor: None
    store._verify_or_initialize_state = lambda _cursor, _fingerprint: None
    monkeypatch.setattr(
        "rag3d.pgvector_store._schema_statements",
        lambda *_args: ("CREATE TABLE bounded_migration()",),
    )

    store._migrate(None)

    calls = database.cursor_instance.calls
    assert "set_config('lock_timeout'" in calls[0][0]
    assert calls[0][1] == ("250",)
    assert "pg_advisory_xact_lock" in calls[1][0]
    statement_timeout_index = next(
        index
        for index, (sql, _params) in enumerate(calls)
        if "set_config('statement_timeout'" in sql
    )
    ddl_index = next(
        index
        for index, (sql, _params) in enumerate(calls)
        if sql == "CREATE TABLE bounded_migration()"
    )
    assert calls[statement_timeout_index][1] == ("5000",)
    assert statement_timeout_index < ddl_index


def test_runtime_migration_lock_error_is_fixed_and_secret_safe(monkeypatch) -> None:
    secret = "postgresql://admin:super-secret@private.invalid/prod"

    class MigrationCursor:
        def execute(self, sql, _params=()):
            if "pg_advisory_xact_lock" in sql:
                raise RuntimeError(secret)

    class MigrationDatabase:
        @contextmanager
        def transaction(self):
            yield

        @contextmanager
        def cursor(self):
            yield MigrationCursor()

    store = object.__new__(PgVectorStore)
    store.db = MigrationDatabase()
    store.dense_dim = 3
    store.colbert_dim = 2
    store.lock_timeout_ms = 250
    monkeypatch.setattr("rag3d.pgvector_store._schema_statements", lambda *_args: ())

    with pytest.raises(PgVectorError) as caught:
        store._migrate(None)

    assert "migration lock" in str(caught.value)
    assert secret not in str(caught.value)
    assert "super-secret" not in str(caught.value)


def test_capabilities_property_is_a_zero_io_snapshot() -> None:
    class BombDatabase:
        def execute(self, *_args, **_kwargs):
            raise AssertionError("capabilities must not query PostgreSQL")

    store = _bare_search_store(BombDatabase())  # type: ignore[arg-type]

    assert store.capabilities.ann_dense_search is True
    assert store.capabilities is store.capabilities


def test_required_ann_preflights_and_rejects_a_natural_sequential_plan() -> None:
    plan = [{"Plan": {"Node Type": "Seq Scan", "Relation Name": "private"}}]
    database = _RecordingDatabase(plan=plan, rows=[(1, 0.0)])
    store = _bare_search_store(database)

    with pytest.raises(PgVectorHnswError, match="natural PostgreSQL plan"):
        store.dense_search([1, 0, 0], 1, exact=False)

    sql_text = "\n".join(sql for sql, _ in database.calls)
    assert "EXPLAIN (FORMAT JSON)" in sql_text
    assert "enable_seqscan" not in sql_text
    assert not any(sql.lstrip().startswith("WITH nearest") for sql, _ in database.calls)


def test_auto_mode_routes_a_natural_sequential_plan_to_explicit_exact() -> None:
    plan = [{"Plan": {"Node Type": "Seq Scan"}}]
    database = _RecordingDatabase(plan=plan, rows=[(4, 0.25)])
    store = _bare_search_store(database)
    store.search_mode = "auto"

    assert store.dense_search([1, 0, 0], 1) == [(4, 0.75)]
    assert store._last_dense_mode == "exact"

    sql_text = "\n".join(sql for sql, _ in database.calls)
    assert "EXPLAIN (FORMAT JSON)" in sql_text
    assert "SET LOCAL enable_indexscan = off" in sql_text
    assert "enable_seqscan" not in sql_text


def test_ann_statement_uses_bounded_overfetch_then_stable_outer_tie_break() -> None:
    store = _bare_search_store(_RecordingDatabase())

    sql, params = store._dense_statement(
        np.asarray([1, 0, 0], dtype=np.float32), 3, None, exact=False
    )

    assert "WITH nearest AS MATERIALIZED" in sql
    assert "ORDER BY c.embedding <=> %s ASC" in sql
    assert "ORDER BY distance ASC, id ASC" in sql
    assert params[-2:] == [12, 3]


def test_sparse_search_aggregates_orders_and_limits_in_postgresql() -> None:
    database = _RecordingDatabase(rows=[(7, 1.25)])
    store = _bare_search_store(database)
    store.n_chunks = lambda: (_ for _ in ()).throw(
        AssertionError("sparse_search must not fetch corpus size separately")
    )

    assert store.sparse_search({11: 2.0}, 3) == [(7, 1.25)]

    statement = next(
        sql for sql, _ in database.calls if "SELECT chunk_id,score FROM scored" in sql
    )
    assert "SUM(" in statement
    assert "ORDER BY score DESC,chunk_id ASC" in statement
    assert "LIMIT %s" in statement


def test_neighbors_return_identity_fields_for_stitch_validation() -> None:
    database = _RecordingDatabase(rows=[(9, 7, "chunk", 1, "neighbor")])
    store = _bare_search_store(database)

    assert store.neighbors(7, [1]) == [
        {
            "id": 9,
            "doc_id": 7,
            "kind": "chunk",
            "pos": 1,
            "text": "neighbor",
        }
    ]


def test_unfingerprinted_store_refuses_mutation_before_database_io() -> None:
    class BombDatabase:
        def execute(self, *_args, **_kwargs):
            raise AssertionError("write reached PostgreSQL")

    store = object.__new__(PgVectorStore)
    store._fingerprint_verified = False
    store.db = BombDatabase()

    with pytest.raises(FingerprintMismatchError, match="fingerprint"):
        store.add_doc("source", "title", 1)


@pytest.mark.parametrize(
    "secret",
    [
        "postgresql://admin:super-secret@example.invalid/prod",
        "host=db.invalid user=admin password=super-secret dbname=private",
        "postgresql://admin:p%40ss%2Fword@example.invalid/private",
    ],
)
def test_health_is_cheap_and_connection_errors_are_secret_safe(secret: str) -> None:

    class BrokenDatabase:
        def __init__(self):
            self.calls = []

        def execute(self, sql, _params=()):
            self.calls.append(sql)
            raise RuntimeError(secret)

    store = object.__new__(PgVectorStore)
    store.db = BrokenDatabase()
    store._closed = False
    store._postgres_version = "17.6"
    store.pgvector_version = "0.8.5"
    store.search_mode = "exact"
    store._last_dense_mode = None
    store._hnsw_status_cache = store._empty_hnsw_status()

    health = store.health()

    assert health["status"] == "error"
    assert store.db.calls == ["SELECT 1"]
    assert "counts" not in health
    assert secret not in json.dumps(health)
    assert "password" not in json.dumps(health).lower()


def test_hnsw_recovery_masks_failures_from_initial_catalog_inspection() -> None:
    secret = "host=db.invalid password=super-secret dbname=private"
    store = object.__new__(PgVectorStore)
    store._fingerprint_verified = True
    store._transaction_depth = 0
    store._inspect_hnsw_index = lambda: (_ for _ in ()).throw(RuntimeError(secret))
    store._hnsw_status_cache = store._empty_hnsw_status()
    store._capabilities_cache = BackendCapabilities()

    with pytest.raises(PgVectorHnswError) as caught:
        store.create_hnsw_index(m=8, ef_construction=32)

    assert secret not in str(caught.value)
    assert "super-secret" not in str(caught.value)
    assert store.capabilities.ann_dense_search is False


def _ready_hnsw_status(store: PgVectorStore) -> dict:
    return {
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


def _bare_hnsw_store() -> PgVectorStore:
    store = object.__new__(PgVectorStore)
    store._fingerprint_verified = True
    store._transaction_depth = 0
    store._hnsw_status_cache = store._empty_hnsw_status()
    store._capabilities_cache = store._capabilities_for_status(
        store._hnsw_status_cache
    )
    return store


def test_hnsw_create_result_reports_caller_provenance() -> None:
    store = _bare_hnsw_store()
    ready = _ready_hnsw_status(store)
    statuses = iter((store._empty_hnsw_status(), ready))
    store._inspect_hnsw_index = lambda: next(statuses)
    store._execute_hnsw_ddl = lambda *_args, **_kwargs: None

    created = store.create_hnsw_index(m=8, ef_construction=32)

    assert created["created_by_caller"] is True

    store._inspect_hnsw_index = lambda: ready
    existing = store.create_hnsw_index(m=8, ef_construction=32)
    assert existing["created_by_caller"] is False


def test_hnsw_race_recovery_is_not_attributed_to_the_caller() -> None:
    store = _bare_hnsw_store()
    ready = _ready_hnsw_status(store)
    statuses = iter((store._empty_hnsw_status(), ready))
    store._inspect_hnsw_index = lambda: next(statuses)

    def lose_creation_race(*_args, **_kwargs):
        raise RuntimeError("duplicate relation")

    store._execute_hnsw_ddl = lose_creation_race

    recovered = store.create_hnsw_index(m=8, ef_construction=32)

    assert recovered["created_by_caller"] is False


def test_hnsw_post_create_inspection_recovery_preserves_caller_provenance() -> None:
    store = _bare_hnsw_store()
    ready = _ready_hnsw_status(store)
    inspections = iter(
        (
            store._empty_hnsw_status(),
            RuntimeError("transient catalog read"),
            ready,
        )
    )

    def inspect():
        result = next(inspections)
        if isinstance(result, Exception):
            raise result
        return result

    store._inspect_hnsw_index = inspect
    store._execute_hnsw_ddl = lambda *_args, **_kwargs: None

    recovered = store.create_hnsw_index(m=8, ef_construction=32)

    assert recovered["created_by_caller"] is True


def test_drop_hnsw_index_routes_through_bounded_ddl_executor() -> None:
    store = _bare_hnsw_store()
    calls = []
    store._execute_hnsw_ddl = lambda statement, *, concurrently: calls.append(
        (statement, concurrently)
    )

    store.drop_hnsw_index(concurrently=True)

    assert calls == [
        ("DROP INDEX CONCURRENTLY IF EXISTS rag3d_v2_chunks_embedding_hnsw", True)
    ]
    assert store.capabilities.ann_dense_search is False


class _NoExtensionCursor:
    def fetchone(self):
        return None


class _NoExtensionConnection:
    autocommit = True

    def execute(self, _sql, _params=()):
        return _NoExtensionCursor()

    def close(self):
        self.closed = True


class _FakePsycopg:
    @staticmethod
    def connect(_dsn, *, autocommit):
        assert autocommit is True
        return _NoExtensionConnection()


def test_missing_extension_error_is_actionable_and_dsn_safe(monkeypatch) -> None:
    dsn = "postgresql://secret-user:secret-password@db.invalid/private"
    monkeypatch.setattr(
        "rag3d.pgvector_store._load_optional_dependencies",
        lambda: (_FakePsycopg, lambda _connection: None),
    )

    with pytest.raises(PgVectorExtensionError) as caught:
        PgVectorStore(dsn, dense_dim=3, colbert_dim=2)

    message = str(caught.value)
    assert "CREATE EXTENSION vector" in message
    assert dsn not in message
    assert "secret-user" not in message
    assert "secret-password" not in message


def test_constructor_uses_fingerprint_structural_limits_and_bounded_fallback(
    monkeypatch,
) -> None:
    class ExtensionConnection:
        def execute(self, _sql, _params=()):
            return _Rows(one=("0.8.5", "17.6"))

        def close(self):
            pass

    class ExtensionPsycopg:
        @staticmethod
        def connect(_dsn, *, autocommit):
            assert autocommit is True
            return ExtensionConnection()

    monkeypatch.setattr(
        "rag3d.pgvector_store._load_optional_dependencies",
        lambda: (ExtensionPsycopg, lambda _connection: None),
    )
    monkeypatch.setattr(PgVectorStore, "_migrate", lambda *_args: None)
    monkeypatch.setattr(PgVectorStore, "refresh_capabilities", lambda _self: {})

    fingerprinted = PgVectorStore(
        "postgresql:///ignored",
        dense_dim=3,
        colbert_dim=2,
        fingerprint=_unit_fingerprint(),
    )
    fallback = PgVectorStore(
        "postgresql:///ignored", dense_dim=3, colbert_dim=2, fingerprint=None
    )

    assert fingerprinted._max_structural_tokens == 8
    assert fingerprinted._query_max_tokens == 8
    assert (
        fallback._max_structural_tokens
        == DEFAULT_RETRIEVAL_LIMITS.max_structural_tokens
    )
    assert (
        fallback._query_max_tokens
        == DEFAULT_RETRIEVAL_LIMITS.max_structural_tokens
    )


def test_base_migration_never_creates_the_server_extension() -> None:
    migration = "\n".join(_schema_statements(3, 2)).lower()
    assert "create extension" not in migration
    assert "rag3d_v2_documents" in migration
    assert "rag3d_v2_chunks" in migration
    assert "rag3d_v2_sparse_postings" in migration
    assert "embedding vector(3)" in migration


def test_schema_verification_checks_constraint_definitions_not_only_names() -> None:
    assert _constraint_definition_matches(
        "rag3d_v2_chunks_document_fk",
        "FOREIGN KEY (document_id) REFERENCES rag3d_v2_documents(id) ON DELETE CASCADE",
        2,
    )
    assert not _constraint_definition_matches(
        "rag3d_v2_chunks_document_fk",
        "FOREIGN KEY (document_id) REFERENCES rag3d_v2_documents(id) ON DELETE NO ACTION",
        2,
    )
    assert _constraint_definition_matches(
        "rag3d_v2_chunks_structural_ck",
        "CHECK ((kind = 'parent' AND structural_n_tok IS NULL AND "
        "structural_dim IS NULL AND structural_data IS NULL) OR "
        "(kind <> 'parent' AND structural_n_tok > 0 AND structural_dim = 2 "
        "AND structural_data IS NOT NULL AND octet_length(structural_data) = "
        "structural_n_tok * structural_dim * 2))",
        2,
    )
    assert not _constraint_definition_matches(
        "rag3d_v2_chunks_structural_ck",
        "CHECK (structural_n_tok > 0 AND structural_dim = 3)",
        2,
    )


@pytest.mark.parametrize(
    ("name", "definition"),
    [
        (
            "rag3d_v2_chunks_kind_ck",
            "CHECK (kind IN ('chunk','parent','summary','turn',"
            "'rolling_summary','evil'))",
        ),
        (
            "rag3d_v2_chunks_position_ck",
            "CHECK (pos >= 0 OR (kind = 'summary' AND pos = -1) OR true)",
        ),
        (
            "rag3d_v2_chunks_embedding_ck",
            "CHECK ((kind = 'parent' AND embedding IS NULL) OR "
            "embedding IS NOT NULL OR true)",
        ),
    ],
)
def test_constraint_definition_matcher_rejects_semantic_supersets(
    name: str, definition: str
) -> None:
    assert not _constraint_definition_matches(name, definition, 2)


class _SchemaCatalogCursor:
    def __init__(self, *, id_default_overrides=None) -> None:
        self.last_sql = ""
        self.id_default_overrides = dict(id_default_overrides or {})

    def execute(self, sql, _params=()):
        self.last_sql = sql

    def fetchall(self):
        if "pg_catalog.format_type(attr.atttypid" in self.last_sql:
            rows = []
            for (table, column), (data_type, not_null) in _expected_columns(3).items():
                default = ""
                identity = ""
                expected_serial = False
                declared_defaults = {
                    ("rag3d_v2_documents", "metadata"): "'{}'::jsonb",
                    ("rag3d_v2_chunks", "kind"): "'chunk'::text",
                    ("rag3d_v2_chunks", "pos"): "0",
                    ("rag3d_v2_chunks", "context"): "''::text",
                    ("rag3d_v2_chunks", "importance"): "0.5",
                }
                default = declared_defaults.get((table, column), default)
                if (table, column) in {
                    ("rag3d_v2_documents", "id"),
                    ("rag3d_v2_chunks", "id"),
                }:
                    default = f"nextval('{table}_id_seq'::regclass)"
                    expected_serial = True
                override = self.id_default_overrides.get((table, column))
                if override is not None:
                    default, identity, expected_serial = override
                rows.append(
                    (
                        table,
                        column,
                        data_type,
                        not_null,
                        default,
                        identity,
                        expected_serial,
                    )
                )
            return rows
        if "FROM pg_catalog.pg_constraint AS con" in self.last_sql:
            return [
                (name, *definition)
                for name, definition in _expected_constraint_catalog(2).items()
            ]
        if "FROM pg_catalog.pg_class AS idx" in self.last_sql:
            return [
                (name, table, "btree", column, True, True, False, True)
                for name, (table, column) in _EXPECTED_INDEX_COLUMNS.items()
            ]
        raise AssertionError("unexpected schema catalog query")


@pytest.mark.parametrize(
    "table",
    ["rag3d_v2_documents", "rag3d_v2_chunks"],
)
def test_runtime_schema_rejects_drifted_id_default_or_identity(table: str) -> None:
    cursor = _SchemaCatalogCursor(
        id_default_overrides={(table, "id"): ("7", "", False)}
    )
    store = object.__new__(PgVectorStore)
    store.dense_dim = 3
    store.colbert_dim = 2

    with pytest.raises(PgVectorSchemaError, match=rf"{table}\.id"):
        store._verify_schema(cursor)


def test_runtime_schema_accepts_owned_bigserial_id_defaults() -> None:
    store = object.__new__(PgVectorStore)
    store.dense_dim = 3
    store.colbert_dim = 2

    store._verify_schema(_SchemaCatalogCursor())


def _unit_fingerprint() -> IndexFingerprint:
    return IndexFingerprint(
        backend="pgvector",
        encoder="unit",
        model="unit-model",
        revision="test",
        dense_dim=3,
        structural_dim=2,
        max_structural_tokens=8,
        structural_projection="unit-v1",
        query_max_tokens=8,
        passage_max_tokens=8,
        sparse_version="unit-sparse-v1",
        schema_version="unit-schema-v1",
        normalization="l2",
        quantization="none",
        chunk_size=32,
        overlap=4,
        chunking_version="unit-chunk-v1",
        pipeline_version="retrieval-v2",
    )


class _FingerprintStateCursor:
    def __init__(self, *, documents: bool, chunks: bool):
        self.documents = documents
        self.chunks = chunks
        self.last_sql = ""
        self.last_params = ()
        self.calls = []

    def execute(self, sql, params=()):
        self.last_sql = sql
        self.last_params = params
        self.calls.append((sql, params))

    def fetchall(self):
        keys = self.last_params[0] if self.last_params else []
        if "schema_version" in keys:
            return [
                ("schema_version", "1"),
                ("dense_dim", "3"),
                ("structural_dim", "2"),
                ("normalization", "l2"),
                ("quantization", "none"),
            ]
        return []

    def fetchone(self):
        return (self.documents, self.chunks)


@pytest.mark.parametrize(
    ("documents", "chunks"), [(True, False), (False, True), (True, True)]
)
def test_first_fingerprint_publication_requires_completely_empty_state(
    documents: bool, chunks: bool
) -> None:
    cursor = _FingerprintStateCursor(documents=documents, chunks=chunks)
    store = object.__new__(PgVectorStore)
    store.dense_dim = 3
    store.colbert_dim = 2
    store._fingerprint_verified = False

    with pytest.raises(FingerprintMismatchError, match="populated|reindex"):
        store._verify_or_initialize_state(cursor, _unit_fingerprint())

    population_query = next(
        sql for sql, _ in cursor.calls if "SELECT EXISTS" in sql
    )
    assert "rag3d_v2_documents" in population_query
    assert "rag3d_v2_chunks" in population_query
    assert "kind IN" not in population_query
    assert not any(
        "INSERT INTO rag3d_v2_meta" in sql
        and "retrieval_v2_fingerprint" in repr(params)
        for sql, params in cursor.calls
    )


def test_pgvector_extra_preserves_python_3_9_compatibility() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'pgvector = [' in pyproject
    assert '"psycopg[binary]>=3.1,<4"' in pyproject
    assert '"pgvector>=0.4.2,<0.5; python_version<\'3.10\'"' in pyproject
    assert '"pgvector>=0.5,<0.6; python_version>=\'3.10\'"' in pyproject


def test_docker_pgvector_profile_is_pinned_and_legacy_service_is_preserved() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    legacy = """  db:
    image: postgres:16
    container_name: rag3d-pg
    environment:
      POSTGRES_PASSWORD: rag3d
      POSTGRES_DB: rag3d
    ports:
      - "5433:5432"
    volumes:
      - rag3d_pg:/var/lib/postgresql/data
"""
    assert legacy in compose
    assert "pgvector/pgvector:0.8.6-pg16-bookworm" in compose
    assert 'profiles: ["pgvector"]' in compose
    assert '"127.0.0.1:5434:5432"' in compose
    assert "POSTGRES_PASSWORD: ${RAG3D_PGVECTOR_PASSWORD" in compose
    assert "RAG3D_PGVECTOR_PASSWORD:?" not in compose
    assert "POSTGRES_PASSWORD: rag3d" not in compose.split("  pgvector:", 1)[1]
    assert "rag3d_pgvector:/var/lib/postgresql/data" in compose
    assert "pg_isready" in compose


def test_legacy_compose_config_does_not_require_pgvector_profile_password() -> None:
    environment = os.environ.copy()
    environment.pop("RAG3D_PGVECTOR_PASSWORD", None)
    completed = subprocess.run(
        ["docker-compose", "config", "--services"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "db" in completed.stdout.splitlines()

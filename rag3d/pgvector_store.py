"""Optional PostgreSQL + pgvector retrieval backend.

The module-level dependency surface intentionally stays limited to the RAG3D
core dependencies.  ``psycopg`` and ``pgvector-python`` are loaded only when a
``PgVectorStore`` is constructed.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Sized
from contextlib import contextmanager
from itertools import islice
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .backend import (
    DEFAULT_RETRIEVAL_LIMITS,
    BackendCapabilities,
    FingerprintMismatchError,
    IndexFingerprint,
    SearchFilters,
    normalize_sparse_weights,
    serialize_document_metadata as _serialize_document_metadata_common,
    validate_string_value,
)


_SEARCHABLE_KINDS_SQL = "('chunk','turn','summary')"
_ITERATIVE_SCAN_VALUES = frozenset({"off", "strict_order", "relaxed_order"})
_SEARCH_MODES = frozenset({"exact", "ann", "auto"})
_SAFE_EXPLAIN_SETTINGS = frozenset(
    {
        "enable_indexscan",
        "enable_bitmapscan",
        "statement_timeout",
        "hnsw.ef_search",
        "hnsw.iterative_scan",
        "hnsw.max_scan_tuples",
        "hnsw.scan_mem_multiplier",
    }
)
_SAFE_EXPLAIN_INDEXES = frozenset(
    {
        "rag3d_v2_meta_pk",
        "rag3d_v2_documents_pk",
        "rag3d_v2_documents_source_idx",
        "rag3d_v2_chunks_pk",
        "rag3d_v2_chunks_document_idx",
        "rag3d_v2_chunks_parent_idx",
        "rag3d_v2_chunks_kind_idx",
        "rag3d_v2_chunks_turn_idx",
        "rag3d_v2_sparse_postings_pk",
        "rag3d_v2_sparse_postings_chunk_idx",
        "rag3d_v2_chunks_embedding_hnsw",
    }
)
_VERSION_PREFIX = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?")
_SCHEMA_VERSION = 1
_MIGRATION_LOCK_KEY = 0x5241473344563201
_FINGERPRINT_LOCK_KEY = 0x5241473344465001
_HNSW_INDEX = "rag3d_v2_chunks_embedding_hnsw"
_ALL_CHUNK_KINDS = frozenset({"chunk", "parent", "summary", "turn", "rolling_summary"})
_FINGERPRINT_KEY = "retrieval_v2_fingerprint"
_FINGERPRINT_DIGEST_KEY = "retrieval_v2_fingerprint_sha256"
_MAX_SCAN_TUPLES = 1_000_000
_MAX_STATEMENT_TIMEOUT_MS = 60_000
_DEFAULT_LOCK_TIMEOUT_MS = 5_000
_MAX_LOCK_TIMEOUT_MS = 60_000
_ANN_OVERFETCH_FACTOR = 4
_DEFAULT_MAX_SCAN_TUPLES = 20_000
_DEFAULT_SCAN_MEM_MULTIPLIER = 1.0
_MAX_BIGINT = 2**63 - 1
_MAX_INTEGER = 2**31 - 1
_FALLBACK_MAX_STRUCTURAL_TOKENS = DEFAULT_RETRIEVAL_LIMITS.max_structural_tokens
_MAX_STRUCTURAL_VALUES = DEFAULT_RETRIEVAL_LIMITS.max_structural_values


class PgVectorError(RuntimeError):
    """Base error for the optional pgvector backend."""


class PgVectorDependencyError(PgVectorError):
    """Raised when the optional Python packages are unavailable."""


class PgVectorConnectionError(PgVectorError):
    """Raised when PostgreSQL cannot be reached without exposing the DSN."""


class PgVectorExtensionError(PgVectorError):
    """Raised when the server-side pgvector extension is unavailable."""


class PgVectorSchemaError(PgVectorError):
    """Raised when existing ``rag3d_v2_*`` objects do not match the contract."""


class PgVectorHnswError(PgVectorError):
    """Raised when ANN is requested without a verified HNSW index."""


def _validate_dimension(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool")
    if value < 1 or value > 2_000:
        raise ValueError(f"{name} must be between 1 and 2000")
    return value


def _normalize_dense_vector(value: object, dimension: int, *, name: str) -> np.ndarray:
    _validate_dimension("dense_dim", dimension)
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric one-dimensional vector") from exc
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if vector.shape[0] != dimension:
        raise ValueError(f"{name} dimension does not match dense_dim")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} values must be finite")
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must have non-zero norm")
    normalized = np.asarray(vector / norm, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError(f"{name} normalization must remain finite")
    return normalized


def _stored_vector_to_numpy(value: object, dimension: int) -> np.ndarray:
    """Convert pgvector-python 0.4/0.5 return values without importing it."""

    converter = getattr(value, "to_numpy", None)
    if callable(converter):
        value = converter()
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.shape[0] != dimension:
        raise PgVectorSchemaError("stored dense vector dimension is invalid")
    if not np.isfinite(vector).all():
        raise PgVectorSchemaError("stored dense vector contains non-finite values")
    return vector.copy()


def _validate_k(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("k must be an integer, not bool")
    if value < 0:
        raise ValueError("k must be non-negative")
    if value > DEFAULT_RETRIEVAL_LIMITS.max_channel_k:
        raise ValueError(
            f"k must not exceed {DEFAULT_RETRIEVAL_LIMITS.max_channel_k}"
        )
    return value


def _validate_non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    if value > _MAX_BIGINT:
        raise ValueError(f"{name} must fit a signed bigint")
    return value


def _validate_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    if value > _MAX_INTEGER:
        raise ValueError(f"{name} must fit a signed 32-bit integer")
    return value


def _validate_optional_count(name: str, value: object) -> Optional[int]:
    if value is None:
        return None
    return _validate_count(name, value)


def _validate_optional_id(name: str, value: object) -> Optional[int]:
    if value is None:
        return None
    return _validate_non_negative_int(name, value)


def _validate_ids(
    name: str, values: Sequence[int], *, maximum: int = DEFAULT_RETRIEVAL_LIMITS.max_pool
) -> List[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    if len(values) > maximum:
        raise ValueError(f"{name} must not contain more than {maximum} values")
    result = []
    for item_count, value in enumerate(values, start=1):
        if item_count > maximum:
            raise ValueError(
                f"{name} must not contain more than {maximum} values"
            )
        result.append(_validate_non_negative_int(name, value))
    return result


def _validate_signed_positions(values: Sequence[int]) -> List[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("positions must be a sequence of integers")
    if len(values) > DEFAULT_RETRIEVAL_LIMITS.max_pool:
        raise ValueError(
            "positions must not contain more than "
            f"{DEFAULT_RETRIEVAL_LIMITS.max_pool} values"
        )
    result = []
    maximum = DEFAULT_RETRIEVAL_LIMITS.max_pool
    for item_count, value in enumerate(values, start=1):
        if item_count > maximum:
            raise ValueError(
                "positions must not contain more than "
                f"{maximum} values"
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("positions must contain integers, not bool")
        if value < -(2**31) or value > _MAX_INTEGER:
            raise ValueError("positions must fit a PostgreSQL integer")
        result.append(value)
    return result


def _validate_sparse_weights(weights: Mapping[int, float]) -> Dict[int, float]:
    normalized = normalize_sparse_weights(weights)
    result: Dict[int, float] = {}
    max_real = float(np.finfo(np.float32).max)
    for term, raw_weight in normalized.items():
        if term < -(2**63) or term > 2**63 - 1:
            raise ValueError("sparse term ID must fit a signed bigint")
        weight = float(raw_weight)
        if abs(weight) > max_real:
            raise ValueError("sparse weights must be finite PostgreSQL real values")
        result[term] = weight
    return result


def _validate_structural_vectors(
    value: object,
    dimension: int,
    *,
    name: str,
    allow_empty: bool,
    max_tokens: int,
) -> np.ndarray:
    dimension = _validate_dimension("colbert_dim", dimension)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise TypeError("structural token limit must be an integer, not bool")
    if max_tokens < 1:
        raise ValueError("structural token limit must be positive")

    def check_row_count(row_count: int) -> None:
        if row_count > max_tokens:
            raise ValueError(
                f"{name} exceeds the maximum of {max_tokens} tokens"
            )
        if row_count * dimension > _MAX_STRUCTURAL_VALUES:
            raise ValueError(
                f"{name} exceeds the maximum number of structural values "
                f"({_MAX_STRUCTURAL_VALUES})"
            )

    if isinstance(value, np.ndarray):
        if value.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix")
        check_row_count(int(value.shape[0]))
        raw_vectors: object = value
    else:
        if isinstance(value, (str, bytes, bytearray)):
            raise ValueError(f"{name} must be a numeric matrix")
        if isinstance(value, Sized):
            try:
                declared_rows = len(value)
            except (OverflowError, TypeError, ValueError):
                raise ValueError(f"{name} must be a numeric matrix") from None
            check_row_count(declared_rows)
        try:
            iterator = iter(value)  # type: ignore[arg-type]
        except TypeError:
            raise ValueError(f"{name} must be a numeric matrix") from None
        rows = []
        for row_count, row in enumerate(iterator, start=1):
            check_row_count(row_count)
            if isinstance(row, (str, bytes, bytearray)):
                raise ValueError(f"{name} structural_dim does not match colbert_dim")
            if isinstance(row, np.ndarray):
                if row.ndim != 1 or row.shape[0] != dimension:
                    raise ValueError(
                        f"{name} structural_dim does not match colbert_dim"
                    )
                bounded_row: object = row
            else:
                if isinstance(row, Sized):
                    try:
                        if len(row) != dimension:
                            raise ValueError(
                                f"{name} structural_dim does not match colbert_dim"
                            )
                    except (OverflowError, TypeError):
                        raise ValueError(
                            f"{name} structural_dim does not match colbert_dim"
                        ) from None
                try:
                    bounded_values = list(islice(iter(row), dimension + 1))
                except TypeError:
                    raise ValueError(
                        f"{name} structural_dim does not match colbert_dim"
                    ) from None
                if len(bounded_values) != dimension:
                    raise ValueError(
                        f"{name} structural_dim does not match colbert_dim"
                    )
                bounded_row = bounded_values
            rows.append(bounded_row)
        if not rows:
            raw_vectors = np.empty((0, dimension), dtype=np.float32)
        else:
            raw_vectors = rows

    try:
        vectors = np.asarray(raw_vectors, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if vectors.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if vectors.shape[1] != dimension:
        raise ValueError(f"{name} structural_dim does not match colbert_dim")
    if not allow_empty and vectors.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one token")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{name} values must be finite")
    encoded = vectors.astype(np.float16)
    if not np.isfinite(encoded).all():
        raise ValueError(f"{name} values must remain finite as float16")
    return vectors


def _validate_kind(kind: object, *, allow_parent: bool = False) -> str:
    if not isinstance(kind, str) or kind not in _ALL_CHUNK_KINDS:
        raise ValueError("kind must be a supported RAG3D chunk kind")
    if kind == "parent" and not allow_parent:
        raise ValueError("parent rows must be created with add_parent")
    return kind


def _validate_chunk_position(kind: object, pos: object) -> int:
    chunk_kind = _validate_kind(kind)
    if isinstance(pos, bool) or not isinstance(pos, int):
        raise TypeError("pos must be an integer, not bool")
    if pos < -1 or pos > 2**31 - 1:
        raise ValueError("pos must fit the supported PostgreSQL integer range")
    if pos == -1 and chunk_kind != "summary":
        raise ValueError("pos=-1 is reserved for summary chunks")
    return pos


def _serialize_document_metadata(metadata: Optional[Mapping[str, Any]]) -> str:
    return _serialize_document_metadata_common(metadata)


def _validate_hnsw_build_options(m: object, ef_construction: object) -> Tuple[int, int]:
    if isinstance(m, bool) or not isinstance(m, int):
        raise TypeError("m must be an integer, not bool")
    if m < 2 or m > 100:
        raise ValueError("m must be between 2 and 100")
    if isinstance(ef_construction, bool) or not isinstance(ef_construction, int):
        raise TypeError("ef_construction must be an integer, not bool")
    if ef_construction < 4 or ef_construction > 1_000:
        raise ValueError("ef_construction must be between 4 and 1000")
    if ef_construction < 2 * m:
        raise ValueError("ef_construction must be at least 2 \u00d7 m")
    return m, ef_construction


def _validate_ann_options(
    *,
    ef_search: object,
    iterative_scan: object,
    max_scan_tuples: object,
    scan_mem_multiplier: object,
    extension_version: Tuple[int, int, int],
) -> Tuple[int, str, int, float]:
    if isinstance(ef_search, bool) or not isinstance(ef_search, int):
        raise TypeError("ef_search must be an integer, not bool")
    if ef_search < 1 or ef_search > 1_000:
        raise ValueError("ef_search must be between 1 and 1000")
    if not isinstance(iterative_scan, str) or iterative_scan not in _ITERATIVE_SCAN_VALUES:
        raise ValueError("iterative_scan must be off, strict_order, or relaxed_order")
    if iterative_scan != "off" and extension_version < (0, 8, 0):
        raise ValueError("iterative_scan requires pgvector >= 0.8.0")
    if isinstance(max_scan_tuples, bool) or not isinstance(max_scan_tuples, int):
        raise TypeError("max_scan_tuples must be an integer, not bool")
    if max_scan_tuples < 1 or max_scan_tuples > _MAX_SCAN_TUPLES:
        raise ValueError(
            f"max_scan_tuples must be between 1 and {_MAX_SCAN_TUPLES}"
        )
    if (
        isinstance(scan_mem_multiplier, bool)
        or not isinstance(scan_mem_multiplier, (int, float))
    ):
        raise TypeError("scan_mem_multiplier must be a real number, not bool")
    multiplier = float(scan_mem_multiplier)
    if not math.isfinite(multiplier) or multiplier < 1 or multiplier > 1_000:
        raise ValueError("scan_mem_multiplier must be finite and between 1 and 1000")
    if extension_version < (0, 8, 0) and (
        max_scan_tuples != _DEFAULT_MAX_SCAN_TUPLES
        or multiplier != _DEFAULT_SCAN_MEM_MULTIPLIER
    ):
        raise ValueError(
            "max_scan_tuples and scan_mem_multiplier require pgvector >= 0.8.0"
        )
    return ef_search, iterative_scan, max_scan_tuples, multiplier


def _validate_statement_timeout(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("statement_timeout_ms must be an integer, not bool")
    if value < 1 or value > _MAX_STATEMENT_TIMEOUT_MS:
        raise ValueError(
            "statement_timeout_ms must be between 1 and "
            f"{_MAX_STATEMENT_TIMEOUT_MS}"
        )
    return value


def _validate_lock_timeout(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("lock_timeout_ms must be an integer, not bool")
    if value < 1 or value > _MAX_LOCK_TIMEOUT_MS:
        raise ValueError(
            "lock_timeout_ms must be between 1 and "
            f"{_MAX_LOCK_TIMEOUT_MS}"
        )
    return value


def _validate_search_mode(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("search_mode must be a string")
    normalized = value.strip().lower()
    if normalized not in _SEARCH_MODES:
        raise ValueError("search_mode must be exact, ann, or auto")
    return normalized


def _parse_extension_version(raw: object) -> Tuple[int, int, int]:
    if not isinstance(raw, str):
        raise PgVectorExtensionError("pgvector reported an invalid extension version")
    match = _VERSION_PREFIX.match(raw)
    if match is None:
        raise PgVectorExtensionError("pgvector reported an invalid extension version")
    return tuple(int(part or 0) for part in match.groups())  # type: ignore[return-value]


def _validate_filters(filters: Optional[SearchFilters]) -> None:
    if filters is not None and not isinstance(filters, SearchFilters):
        raise TypeError("filters must be SearchFilters or None")


def _build_filter_clause(filters: Optional[SearchFilters]) -> Tuple[str, List[Any]]:
    _validate_filters(filters)
    clauses = []
    params: List[Any] = []
    scope = filters.scope if filters is not None else None

    if scope is not None and scope.kinds:
        clauses.append("c.kind = ANY(%s::text[])")
        params.append(list(scope.kinds))
    else:
        clauses.append(f"c.kind IN {_SEARCHABLE_KINDS_SQL}")
    if scope is not None and scope.document_ids:
        clauses.append("c.document_id = ANY(%s::bigint[])")
        params.append(list(scope.document_ids))
    if scope is not None and scope.parent_ids:
        clauses.append("c.parent_id = ANY(%s::bigint[])")
        params.append(list(scope.parent_ids))
    if scope is not None and scope.sources:
        clauses.append("d.source = ANY(%s::text[])")
        params.append(list(scope.sources))
    if filters is not None and filters.metadata:
        clauses.append("d.metadata @> %s::jsonb")
        params.append(
            json.dumps(
                dict(filters.metadata),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    return " AND ".join(clauses), params


def _sanitize_explain_document(document: object, *, exact: bool) -> dict:
    if not isinstance(document, list) or not document or not isinstance(document[0], dict):
        raise PgVectorError("PostgreSQL returned an invalid EXPLAIN document")
    root = document[0]
    raw_plan = root.get("Plan")
    if not isinstance(raw_plan, dict):
        raise PgVectorError("PostgreSQL returned an invalid EXPLAIN plan")

    node_types: List[str] = []
    index_names: List[str] = []
    rows_removed = 0
    buffer_keys = {
        "Shared Hit Blocks": "shared_hit",
        "Shared Read Blocks": "shared_read",
        "Shared Dirtied Blocks": "shared_dirtied",
        "Shared Written Blocks": "shared_written",
        "Temp Read Blocks": "temp_read",
        "Temp Written Blocks": "temp_written",
    }
    buffers = {safe: 0 for safe in buffer_keys.values()}

    def visit(node: dict) -> None:
        nonlocal rows_removed
        node_type = node.get("Node Type")
        if isinstance(node_type, str):
            node_types.append(node_type)
        index_name = node.get("Index Name")
        if isinstance(index_name, str) and index_name in _SAFE_EXPLAIN_INDEXES:
            index_names.append(index_name)
        removed = node.get("Rows Removed by Filter", 0)
        if isinstance(removed, (int, float)) and not isinstance(removed, bool):
            rows_removed += max(0, int(removed))
        for raw_key, safe_key in buffer_keys.items():
            value = node.get(raw_key, 0)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                buffers[safe_key] += max(0, int(value))
        children = node.get("Plans", [])
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    visit(child)

    visit(raw_plan)
    raw_settings = root.get("Settings", {})
    settings = {}
    if isinstance(raw_settings, dict):
        for key, value in raw_settings.items():
            if isinstance(key, str) and key in _SAFE_EXPLAIN_SETTINGS:
                settings[key] = str(value)

    planning = root.get("Planning Time", 0.0)
    execution = root.get("Execution Time", 0.0)
    return {
        "mode": "exact" if exact else "ann",
        "plan": {
            "node_types": node_types,
            "index_names": sorted(set(index_names)),
            "hnsw_used": _HNSW_INDEX in index_names,
            "actual_rows": int(raw_plan.get("Actual Rows", 0) or 0),
            "rows_removed_by_filter": rows_removed,
            "buffers": buffers,
        },
        "planning_time_ms": float(planning) if isinstance(planning, (int, float)) else 0.0,
        "execution_time_ms": float(execution) if isinstance(execution, (int, float)) else 0.0,
        "settings": settings,
    }


def _schema_statements(dense_dim: int, colbert_dim: int) -> Tuple[str, ...]:
    """Return the fixed base migration with only validated integers embedded."""

    dense_dim = _validate_dimension("dense_dim", dense_dim)
    colbert_dim = _validate_dimension("colbert_dim", colbert_dim)
    return (
        """
        CREATE TABLE IF NOT EXISTS rag3d_v2_meta(
          key TEXT CONSTRAINT rag3d_v2_meta_pk PRIMARY KEY,
          value TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS rag3d_v2_documents(
          id BIGSERIAL CONSTRAINT rag3d_v2_documents_pk PRIMARY KEY,
          source TEXT NOT NULL,
          title TEXT NOT NULL,
          created DOUBLE PRECISION NOT NULL,
          n_tokens INTEGER NOT NULL,
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
          CONSTRAINT rag3d_v2_documents_tokens_ck CHECK (n_tokens >= 0)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS rag3d_v2_chunks(
          id BIGSERIAL CONSTRAINT rag3d_v2_chunks_pk PRIMARY KEY,
          document_id BIGINT,
          parent_id BIGINT,
          kind TEXT NOT NULL DEFAULT 'chunk',
          pos INTEGER NOT NULL DEFAULT 0,
          text TEXT NOT NULL,
          context TEXT NOT NULL DEFAULT '',
          n_tokens INTEGER NOT NULL,
          created DOUBLE PRECISION NOT NULL,
          importance REAL NOT NULL DEFAULT 0.5,
          turn_no INTEGER,
          accessed_turn INTEGER,
          embedding vector({dense_dim}),
          structural_n_tok INTEGER,
          structural_dim INTEGER,
          structural_data BYTEA,
          CONSTRAINT rag3d_v2_chunks_document_fk FOREIGN KEY(document_id)
            REFERENCES rag3d_v2_documents(id) ON DELETE CASCADE,
          CONSTRAINT rag3d_v2_chunks_parent_fk FOREIGN KEY(parent_id)
            REFERENCES rag3d_v2_chunks(id) ON DELETE SET NULL,
          CONSTRAINT rag3d_v2_chunks_kind_ck CHECK (
            kind IN ('chunk','parent','summary','turn','rolling_summary')
          ),
          CONSTRAINT rag3d_v2_chunks_position_ck CHECK (
            pos >= 0 OR (kind = 'summary' AND pos = -1)
          ),
          CONSTRAINT rag3d_v2_chunks_tokens_ck CHECK (n_tokens >= 0),
          CONSTRAINT rag3d_v2_chunks_importance_ck CHECK (
            importance >= 0 AND importance <= 1
          ),
          CONSTRAINT rag3d_v2_chunks_embedding_ck CHECK (
            (kind = 'parent' AND embedding IS NULL) OR
            (kind <> 'parent' AND embedding IS NOT NULL)
          ),
          CONSTRAINT rag3d_v2_chunks_structural_ck CHECK (
            (kind = 'parent' AND structural_n_tok IS NULL AND
             structural_dim IS NULL AND structural_data IS NULL) OR
            (kind <> 'parent' AND structural_n_tok > 0 AND
             structural_dim = {colbert_dim} AND structural_data IS NOT NULL AND
             octet_length(structural_data) = structural_n_tok * structural_dim * 2)
          )
        )
        """,
        """
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_catalog.pg_constraint
            WHERE conname = 'rag3d_v2_chunks_position_ck'
              AND conrelid = 'rag3d_v2_chunks'::regclass
          ) THEN
            ALTER TABLE rag3d_v2_chunks
              ADD CONSTRAINT rag3d_v2_chunks_position_ck CHECK (
                pos >= 0 OR (kind = 'summary' AND pos = -1)
              );
          END IF;
        END
        $$
        """,
        """
        CREATE TABLE IF NOT EXISTS rag3d_v2_sparse_postings(
          term BIGINT NOT NULL,
          chunk_id BIGINT NOT NULL,
          weight REAL NOT NULL,
          CONSTRAINT rag3d_v2_sparse_postings_pk PRIMARY KEY(term, chunk_id),
          CONSTRAINT rag3d_v2_sparse_postings_chunk_fk FOREIGN KEY(chunk_id)
            REFERENCES rag3d_v2_chunks(id) ON DELETE CASCADE,
          CONSTRAINT rag3d_v2_sparse_postings_weight_ck CHECK (
            weight <> 'NaN'::real AND
            weight > '-Infinity'::real AND weight < 'Infinity'::real
          )
        )
        """,
        "CREATE INDEX IF NOT EXISTS rag3d_v2_documents_source_idx "
        "ON rag3d_v2_documents(source)",
        "CREATE INDEX IF NOT EXISTS rag3d_v2_chunks_document_idx "
        "ON rag3d_v2_chunks(document_id)",
        "CREATE INDEX IF NOT EXISTS rag3d_v2_chunks_parent_idx "
        "ON rag3d_v2_chunks(parent_id)",
        "CREATE INDEX IF NOT EXISTS rag3d_v2_chunks_kind_idx "
        "ON rag3d_v2_chunks(kind)",
        "CREATE INDEX IF NOT EXISTS rag3d_v2_chunks_turn_idx "
        "ON rag3d_v2_chunks(turn_no)",
        "CREATE INDEX IF NOT EXISTS rag3d_v2_sparse_postings_chunk_idx "
        "ON rag3d_v2_sparse_postings(chunk_id)",
    )


_EXPECTED_INDEX_COLUMNS = {
    "rag3d_v2_documents_source_idx": ("rag3d_v2_documents", "source"),
    "rag3d_v2_chunks_document_idx": ("rag3d_v2_chunks", "document_id"),
    "rag3d_v2_chunks_parent_idx": ("rag3d_v2_chunks", "parent_id"),
    "rag3d_v2_chunks_kind_idx": ("rag3d_v2_chunks", "kind"),
    "rag3d_v2_chunks_turn_idx": ("rag3d_v2_chunks", "turn_no"),
    "rag3d_v2_sparse_postings_chunk_idx": (
        "rag3d_v2_sparse_postings",
        "chunk_id",
    ),
}


def _expected_columns(dense_dim: int) -> Dict[Tuple[str, str], Tuple[str, bool]]:
    return {
        ("rag3d_v2_meta", "key"): ("text", True),
        ("rag3d_v2_meta", "value"): ("text", True),
        ("rag3d_v2_documents", "id"): ("bigint", True),
        ("rag3d_v2_documents", "source"): ("text", True),
        ("rag3d_v2_documents", "title"): ("text", True),
        ("rag3d_v2_documents", "created"): ("double precision", True),
        ("rag3d_v2_documents", "n_tokens"): ("integer", True),
        ("rag3d_v2_documents", "metadata"): ("jsonb", True),
        ("rag3d_v2_chunks", "id"): ("bigint", True),
        ("rag3d_v2_chunks", "document_id"): ("bigint", False),
        ("rag3d_v2_chunks", "parent_id"): ("bigint", False),
        ("rag3d_v2_chunks", "kind"): ("text", True),
        ("rag3d_v2_chunks", "pos"): ("integer", True),
        ("rag3d_v2_chunks", "text"): ("text", True),
        ("rag3d_v2_chunks", "context"): ("text", True),
        ("rag3d_v2_chunks", "n_tokens"): ("integer", True),
        ("rag3d_v2_chunks", "created"): ("double precision", True),
        ("rag3d_v2_chunks", "importance"): ("real", True),
        ("rag3d_v2_chunks", "turn_no"): ("integer", False),
        ("rag3d_v2_chunks", "accessed_turn"): ("integer", False),
        ("rag3d_v2_chunks", "embedding"): (f"vector({dense_dim})", False),
        ("rag3d_v2_chunks", "structural_n_tok"): ("integer", False),
        ("rag3d_v2_chunks", "structural_dim"): ("integer", False),
        ("rag3d_v2_chunks", "structural_data"): ("bytea", False),
        ("rag3d_v2_sparse_postings", "term"): ("bigint", True),
        ("rag3d_v2_sparse_postings", "chunk_id"): ("bigint", True),
        ("rag3d_v2_sparse_postings", "weight"): ("real", True),
    }


_DECLARED_COLUMN_DEFAULTS = {
    ("rag3d_v2_documents", "metadata"): "'{}'::jsonb",
    ("rag3d_v2_chunks", "kind"): "'chunk'::text",
    ("rag3d_v2_chunks", "pos"): "0",
    ("rag3d_v2_chunks", "context"): "''::text",
    ("rag3d_v2_chunks", "importance"): "0.5",
}
_SERIAL_ID_COLUMNS = frozenset(
    {
        ("rag3d_v2_documents", "id"),
        ("rag3d_v2_chunks", "id"),
    }
)


def _column_default_contract(
    table: str,
    column: str,
    default_expression: object,
    identity: object,
    owns_expected_sequence: object,
) -> Tuple[str, str]:
    normalized_default = _normalize_catalog_expression(default_expression)
    identity_kind = str(identity)
    key = (table, column)
    if key in _SERIAL_ID_COLUMNS:
        expected_sequence = re.escape(f"{table}_id_seq")
        serial_default = re.fullmatch(
            rf"nextval\('(?:[^']+\.)?{expected_sequence}'::regclass\)",
            normalized_default,
        )
        if identity_kind == "" and bool(owns_expected_sequence) and serial_default:
            return "serial", ""
    return normalized_default, identity_kind


_REQUIRED_CONSTRAINTS = frozenset(
    {
        "rag3d_v2_meta_pk",
        "rag3d_v2_documents_pk",
        "rag3d_v2_documents_tokens_ck",
        "rag3d_v2_chunks_pk",
        "rag3d_v2_chunks_document_fk",
        "rag3d_v2_chunks_parent_fk",
        "rag3d_v2_chunks_kind_ck",
        "rag3d_v2_chunks_position_ck",
        "rag3d_v2_chunks_tokens_ck",
        "rag3d_v2_chunks_importance_ck",
        "rag3d_v2_chunks_embedding_ck",
        "rag3d_v2_chunks_structural_ck",
        "rag3d_v2_sparse_postings_pk",
        "rag3d_v2_sparse_postings_chunk_fk",
        "rag3d_v2_sparse_postings_weight_ck",
    }
)


def _constraint_definition_matches(
    name: str, definition: str, colbert_dim: int
) -> bool:
    """Exact textual compatibility helper retained for focused unit probes.

    Runtime verification uses structured catalog fields below. This helper is
    intentionally equality-based so a semantic superset such as ``OR true``
    can never be accepted merely because it contains expected fragments.
    """

    if not isinstance(definition, str):
        return False

    def normalized(value: str) -> str:
        return re.sub(r"\s+", "", value.lower().replace('"', ""))

    definitions = {
        "rag3d_v2_meta_pk": "PRIMARY KEY (key)",
        "rag3d_v2_documents_pk": "PRIMARY KEY (id)",
        "rag3d_v2_documents_tokens_ck": "CHECK (n_tokens >= 0)",
        "rag3d_v2_chunks_pk": "PRIMARY KEY (id)",
        "rag3d_v2_chunks_document_fk": (
            "FOREIGN KEY (document_id) REFERENCES rag3d_v2_documents(id) "
            "ON DELETE CASCADE"
        ),
        "rag3d_v2_chunks_parent_fk": (
            "FOREIGN KEY (parent_id) REFERENCES rag3d_v2_chunks(id) "
            "ON DELETE SET NULL"
        ),
        "rag3d_v2_chunks_kind_ck": (
            "CHECK (kind IN ('chunk','parent','summary','turn',"
            "'rolling_summary'))"
        ),
        "rag3d_v2_chunks_position_ck": (
            "CHECK (pos >= 0 OR (kind = 'summary' AND pos = -1))"
        ),
        "rag3d_v2_chunks_tokens_ck": "CHECK (n_tokens >= 0)",
        "rag3d_v2_chunks_importance_ck": (
            "CHECK (importance >= 0 AND importance <= 1)"
        ),
        "rag3d_v2_chunks_embedding_ck": (
            "CHECK ((kind = 'parent' AND embedding IS NULL) OR "
            "(kind <> 'parent' AND embedding IS NOT NULL))"
        ),
        "rag3d_v2_chunks_structural_ck": (
            "CHECK ((kind = 'parent' AND structural_n_tok IS NULL AND "
            "structural_dim IS NULL AND structural_data IS NULL) OR "
            "(kind <> 'parent' AND structural_n_tok > 0 AND "
            f"structural_dim = {colbert_dim} AND structural_data IS NOT NULL AND "
            "octet_length(structural_data) = structural_n_tok * structural_dim * 2))"
        ),
        "rag3d_v2_sparse_postings_pk": "PRIMARY KEY (term, chunk_id)",
        "rag3d_v2_sparse_postings_chunk_fk": (
            "FOREIGN KEY (chunk_id) REFERENCES rag3d_v2_chunks(id) "
            "ON DELETE CASCADE"
        ),
        "rag3d_v2_sparse_postings_weight_ck": (
            "CHECK (weight <> 'NaN'::real AND weight > '-Infinity'::real "
            "AND weight < 'Infinity'::real)"
        ),
    }
    expected = definitions.get(name)
    if expected is None:
        return False
    candidate = normalized(definition)
    accepted = {normalized(expected)}
    if name == "rag3d_v2_chunks_document_fk":
        accepted.add(normalized(expected.replace("REFERENCES ", "REFERENCES public.")))
    return candidate in accepted


def _normalize_catalog_expression(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", "", value.lower())


def _expected_constraint_catalog(colbert_dim: int) -> Dict[str, Tuple[Any, ...]]:
    """Return the exact structured pg_catalog contract for base constraints."""

    dimension = _validate_dimension("colbert_dim", colbert_dim)
    checks = {
        "rag3d_v2_documents_tokens_ck": "(n_tokens >= 0)",
        "rag3d_v2_chunks_kind_ck": (
            "(kind = ANY (ARRAY['chunk'::text, 'parent'::text, "
            "'summary'::text, 'turn'::text, 'rolling_summary'::text]))"
        ),
        "rag3d_v2_chunks_position_ck": (
            "((pos >= 0) OR ((kind = 'summary'::text) AND "
            "(pos = '-1'::integer)))"
        ),
        "rag3d_v2_chunks_tokens_ck": "(n_tokens >= 0)",
        "rag3d_v2_chunks_importance_ck": (
            "((importance >= (0)::double precision) AND "
            "(importance <= (1)::double precision))"
        ),
        "rag3d_v2_chunks_embedding_ck": (
            "(((kind = 'parent'::text) AND (embedding IS NULL)) OR "
            "((kind <> 'parent'::text) AND (embedding IS NOT NULL)))"
        ),
        "rag3d_v2_chunks_structural_ck": (
            "(((kind = 'parent'::text) AND (structural_n_tok IS NULL) AND "
            "(structural_dim IS NULL) AND (structural_data IS NULL)) OR "
            "((kind <> 'parent'::text) AND (structural_n_tok > 0) AND "
            f"(structural_dim = {dimension}) AND (structural_data IS NOT NULL) "
            "AND (octet_length(structural_data) = "
            "((structural_n_tok * structural_dim) * 2))))"
        ),
        "rag3d_v2_sparse_postings_weight_ck": (
            "((weight <> 'NaN'::real) AND (weight > '-Infinity'::real) "
            "AND (weight < 'Infinity'::real))"
        ),
    }

    def spec(
        table: str,
        kind: str,
        columns: Tuple[str, ...],
        *,
        target: str = "",
        target_columns: Tuple[str, ...] = (),
        delete_action: str = "",
        expression: str = "",
    ) -> Tuple[Any, ...]:
        return (
            table,
            kind,
            True,
            columns,
            target,
            target_columns,
            delete_action,
            _normalize_catalog_expression(expression),
        )

    return {
        "rag3d_v2_meta_pk": spec("rag3d_v2_meta", "p", ("key",)),
        "rag3d_v2_documents_pk": spec("rag3d_v2_documents", "p", ("id",)),
        "rag3d_v2_documents_tokens_ck": spec(
            "rag3d_v2_documents",
            "c",
            ("n_tokens",),
            expression=checks["rag3d_v2_documents_tokens_ck"],
        ),
        "rag3d_v2_chunks_pk": spec("rag3d_v2_chunks", "p", ("id",)),
        "rag3d_v2_chunks_document_fk": spec(
            "rag3d_v2_chunks",
            "f",
            ("document_id",),
            target="rag3d_v2_documents",
            target_columns=("id",),
            delete_action="c",
        ),
        "rag3d_v2_chunks_parent_fk": spec(
            "rag3d_v2_chunks",
            "f",
            ("parent_id",),
            target="rag3d_v2_chunks",
            target_columns=("id",),
            delete_action="n",
        ),
        "rag3d_v2_chunks_kind_ck": spec(
            "rag3d_v2_chunks",
            "c",
            ("kind",),
            expression=checks["rag3d_v2_chunks_kind_ck"],
        ),
        "rag3d_v2_chunks_position_ck": spec(
            "rag3d_v2_chunks",
            "c",
            ("pos", "kind"),
            expression=checks["rag3d_v2_chunks_position_ck"],
        ),
        "rag3d_v2_chunks_tokens_ck": spec(
            "rag3d_v2_chunks",
            "c",
            ("n_tokens",),
            expression=checks["rag3d_v2_chunks_tokens_ck"],
        ),
        "rag3d_v2_chunks_importance_ck": spec(
            "rag3d_v2_chunks",
            "c",
            ("importance",),
            expression=checks["rag3d_v2_chunks_importance_ck"],
        ),
        "rag3d_v2_chunks_embedding_ck": spec(
            "rag3d_v2_chunks",
            "c",
            ("kind", "embedding"),
            expression=checks["rag3d_v2_chunks_embedding_ck"],
        ),
        "rag3d_v2_chunks_structural_ck": spec(
            "rag3d_v2_chunks",
            "c",
            ("kind", "structural_n_tok", "structural_dim", "structural_data"),
            expression=checks["rag3d_v2_chunks_structural_ck"],
        ),
        "rag3d_v2_sparse_postings_pk": spec(
            "rag3d_v2_sparse_postings", "p", ("term", "chunk_id")
        ),
        "rag3d_v2_sparse_postings_chunk_fk": spec(
            "rag3d_v2_sparse_postings",
            "f",
            ("chunk_id",),
            target="rag3d_v2_chunks",
            target_columns=("id",),
            delete_action="c",
        ),
        "rag3d_v2_sparse_postings_weight_ck": spec(
            "rag3d_v2_sparse_postings",
            "c",
            ("weight",),
            expression=checks["rag3d_v2_sparse_postings_weight_ck"],
        ),
    }


def _load_optional_dependencies():
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except (ImportError, ModuleNotFoundError) as exc:
        raise PgVectorDependencyError(
            "PgVectorStore requires the 'pgvector' optional dependency extra"
        ) from exc
    return psycopg, register_vector


class PgVectorStore:
    """Retrieval V2 backend backed by PostgreSQL and the pgvector extension."""

    def __init__(
        self,
        dsn: str,
        dense_dim: int,
        colbert_dim: int,
        *,
        fingerprint: Optional[IndexFingerprint] = None,
        ef_search: int = 40,
        iterative_scan: str = "off",
        max_scan_tuples: int = 20_000,
        scan_mem_multiplier: float = 1.0,
        search_mode: str = "exact",
        statement_timeout_ms: int = 5_000,
        lock_timeout_ms: int = _DEFAULT_LOCK_TIMEOUT_MS,
    ):
        self.dense_dim = _validate_dimension("dense_dim", dense_dim)
        self.colbert_dim = _validate_dimension("colbert_dim", colbert_dim)
        if fingerprint is not None and not isinstance(fingerprint, IndexFingerprint):
            raise TypeError("fingerprint must be IndexFingerprint or None")
        if fingerprint is not None:
            if fingerprint.dense_dim != self.dense_dim:
                raise PgVectorSchemaError("fingerprint dense_dim does not match dense_dim")
            if fingerprint.structural_dim != self.colbert_dim:
                raise PgVectorSchemaError(
                    "fingerprint structural_dim does not match colbert_dim"
                )
            if fingerprint.backend != "pgvector":
                raise PgVectorSchemaError("fingerprint backend must be pgvector")
            if fingerprint.normalization != "l2":
                raise PgVectorSchemaError("fingerprint normalization must be l2")

        self._max_structural_tokens = (
            fingerprint.max_structural_tokens
            if fingerprint is not None
            else _FALLBACK_MAX_STRUCTURAL_TOKENS
        )
        self._query_max_tokens = (
            fingerprint.query_max_tokens
            if fingerprint is not None
            else _FALLBACK_MAX_STRUCTURAL_TOKENS
        )

        self._closed = False
        self._transaction_depth = 0
        self.search_mode = _validate_search_mode(search_mode)
        self.statement_timeout_ms = _validate_statement_timeout(
            statement_timeout_ms
        )
        self.lock_timeout_ms = _validate_lock_timeout(lock_timeout_ms)
        self._last_dense_mode: Optional[str] = None
        self._fingerprint_verified = False
        self._hnsw_status_cache: dict = self._empty_hnsw_status()
        self._capabilities_cache = self._capabilities_for_status(
            self._hnsw_status_cache
        )
        psycopg, register_vector = _load_optional_dependencies()
        try:
            self.db = psycopg.connect(dsn, autocommit=True)
        except Exception as exc:
            raise PgVectorConnectionError(
                "could not connect to the PostgreSQL pgvector backend"
            ) from exc
        try:
            row = self.db.execute(
                "SELECT extversion,current_setting('server_version') "
                "FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if row is None:
                raise PgVectorExtensionError(
                    "pgvector extension is not installed; ask a database administrator "
                    "to run CREATE EXTENSION vector"
                )
            self.pgvector_version = str(row[0])
            self._postgres_version = str(row[1])
            self._pgvector_version_tuple = _parse_extension_version(self.pgvector_version)
            (
                self.ef_search,
                self.iterative_scan,
                self.max_scan_tuples,
                self.scan_mem_multiplier,
            ) = _validate_ann_options(
                ef_search=ef_search,
                iterative_scan=iterative_scan,
                max_scan_tuples=max_scan_tuples,
                scan_mem_multiplier=scan_mem_multiplier,
                extension_version=self._pgvector_version_tuple,
            )
            register_vector(self.db)
            self._migrate(fingerprint)
            self.refresh_capabilities()
        except Exception:
            self.db.close()
            self._closed = True
            raise

    @property
    def backend_name(self) -> str:
        return "pgvector"

    @property
    def closed(self) -> bool:
        return self._closed

    @staticmethod
    def _empty_hnsw_status() -> dict:
        return {
            "exists": False,
            "valid": False,
            "ready": False,
            "definition_valid": False,
            "method": None,
            "opclass": None,
            "column": None,
            "vector_type": None,
            "options": {},
            "index_name": _HNSW_INDEX,
        }

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities_cache

    @staticmethod
    def _capabilities_for_status(status: Mapping[str, Any]) -> BackendCapabilities:
        return BackendCapabilities(
            exact_dense_search=True,
            ann_dense_search=bool(
                status["valid"] and status["ready"] and status["definition_valid"]
            ),
            sparse_search=True,
            structural_rerank=True,
            metadata_filters=True,
            transactions=True,
            native_vector=True,
            quantized_vector=False,
            cross_language_index=False,
        )

    def _migrate(self, fingerprint: Optional[IndexFingerprint]) -> None:
        with self.db.transaction():
            with self.db.cursor() as cursor:
                self._acquire_advisory_xact_lock(
                    cursor, _MIGRATION_LOCK_KEY, purpose="migration"
                )
                try:
                    cursor.execute(
                        "SELECT set_config('statement_timeout',%s,true)",
                        (str(self.statement_timeout_ms),),
                    )
                except Exception as exc:
                    raise PgVectorError(
                        "could not configure the pgvector migration timeout"
                    ) from exc
                for statement in _schema_statements(self.dense_dim, self.colbert_dim):
                    cursor.execute(statement)
                self._verify_schema(cursor)
                self._verify_or_initialize_state(cursor, fingerprint)

    def _acquire_advisory_xact_lock(
        self, executor: Any, lock_key: int, *, purpose: str
    ) -> None:
        try:
            executor.execute(
                "SELECT set_config('lock_timeout',%s,true)",
                (str(self.lock_timeout_ms),),
            )
            executor.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key,))
        except Exception as exc:
            raise PgVectorError(
                f"could not acquire the pgvector {purpose} lock"
            ) from exc

    def _verify_schema(self, cursor: Any) -> None:
        table_names = [
            "rag3d_v2_meta",
            "rag3d_v2_documents",
            "rag3d_v2_chunks",
            "rag3d_v2_sparse_postings",
        ]
        cursor.execute(
            """
            SELECT cls.relname, attr.attname,
                   pg_catalog.format_type(attr.atttypid, attr.atttypmod),
                   attr.attnotnull,
                   COALESCE(
                     pg_catalog.pg_get_expr(def.adbin, def.adrelid, false), ''
                   ),
                   attr.attidentity::text,
                   COALESCE(
                     pg_catalog.pg_get_serial_sequence(
                       pg_catalog.format('%%I.%%I', ns.nspname, cls.relname),
                       attr.attname
                     ) = pg_catalog.format(
                       '%%I.%%I', ns.nspname,
                       cls.relname || '_' || attr.attname || '_seq'
                     ),
                     false
                   )
            FROM pg_catalog.pg_class AS cls
            JOIN pg_catalog.pg_namespace AS ns ON ns.oid = cls.relnamespace
            JOIN pg_catalog.pg_attribute AS attr ON attr.attrelid = cls.oid
            LEFT JOIN pg_catalog.pg_attrdef AS def
              ON def.adrelid = attr.attrelid AND def.adnum = attr.attnum
            WHERE ns.nspname = current_schema()
              AND cls.relname = ANY(%s::text[])
              AND cls.relkind = 'r'
              AND attr.attnum > 0
              AND NOT attr.attisdropped
            """,
            (table_names,),
        )
        actual_columns = {}
        actual_defaults = {}
        for (
            table,
            column,
            data_type,
            not_null,
            default_expression,
            identity,
            owns_expected_sequence,
        ) in cursor.fetchall():
            key = (str(table), str(column))
            actual_columns[key] = (str(data_type), bool(not_null))
            actual_defaults[key] = _column_default_contract(
                key[0],
                key[1],
                default_expression,
                identity,
                owns_expected_sequence,
            )
        expected_columns = _expected_columns(self.dense_dim)
        for key, expected in expected_columns.items():
            actual = actual_columns.get(key)
            if actual != expected:
                field = f"{key[0]}.{key[1]}"
                if key == ("rag3d_v2_chunks", "embedding"):
                    field += " dense_dim"
                raise PgVectorSchemaError(
                    f"incompatible pgvector schema definition for {field}"
                )
        unexpected = set(actual_columns) - set(expected_columns)
        if unexpected:
            raise PgVectorSchemaError(
                "incompatible pgvector schema contains unexpected columns"
            )
        for key in expected_columns:
            expected_default = (
                "serial"
                if key in _SERIAL_ID_COLUMNS
                else _normalize_catalog_expression(
                    _DECLARED_COLUMN_DEFAULTS.get(key, "")
                )
            )
            if actual_defaults.get(key) != (expected_default, ""):
                raise PgVectorSchemaError(
                    "incompatible pgvector schema definition for "
                    f"{key[0]}.{key[1]} default or identity"
                )

        cursor.execute(
            """
            SELECT con.conname, cls.relname, con.contype, con.convalidated,
                   COALESCE((
                     SELECT array_agg(attr.attname::text ORDER BY key.ord)
                     FROM unnest(con.conkey) WITH ORDINALITY AS key(attnum,ord)
                     JOIN pg_catalog.pg_attribute AS attr
                       ON attr.attrelid = con.conrelid
                      AND attr.attnum = key.attnum
                   ), ARRAY[]::text[]),
                   CASE
                     WHEN target.oid IS NULL THEN ''
                     WHEN target_ns.nspname = current_schema() THEN target.relname
                     ELSE target_ns.nspname || '.' || target.relname
                   END,
                   COALESCE((
                     SELECT array_agg(attr.attname::text ORDER BY key.ord)
                     FROM unnest(con.confkey) WITH ORDINALITY AS key(attnum,ord)
                     JOIN pg_catalog.pg_attribute AS attr
                       ON attr.attrelid = con.confrelid
                      AND attr.attnum = key.attnum
                   ), ARRAY[]::text[]),
                   CASE WHEN con.contype = 'f' THEN con.confdeltype::text ELSE '' END,
                   COALESCE(
                     pg_catalog.pg_get_expr(con.conbin, con.conrelid, false), ''
                   )
            FROM pg_catalog.pg_constraint AS con
            JOIN pg_catalog.pg_class AS cls ON cls.oid = con.conrelid
            JOIN pg_catalog.pg_namespace AS ns ON ns.oid = cls.relnamespace
            LEFT JOIN pg_catalog.pg_class AS target ON target.oid = con.confrelid
            LEFT JOIN pg_catalog.pg_namespace AS target_ns
              ON target_ns.oid = target.relnamespace
            WHERE ns.nspname = current_schema()
              AND cls.relname = ANY(%s::text[])
            """,
            (table_names,),
        )
        actual_constraints: Dict[str, List[Tuple[Any, ...]]] = {}
        for (
            name,
            table,
            constraint_type,
            validated,
            columns,
            target,
            target_columns,
            delete_action,
            expression,
        ) in cursor.fetchall():
            actual_constraints.setdefault(str(name), []).append(
                (
                    str(table),
                    str(constraint_type),
                    bool(validated),
                    tuple(str(column) for column in columns),
                    str(target),
                    tuple(str(column) for column in target_columns),
                    str(delete_action),
                    _normalize_catalog_expression(expression),
                )
            )
        expected_constraints = _expected_constraint_catalog(self.colbert_dim)
        for name, expected_definition in expected_constraints.items():
            if actual_constraints.get(name) != [expected_definition]:
                raise PgVectorSchemaError(
                    f"incompatible pgvector constraint catalog for {name}"
                )

        index_names = list(_EXPECTED_INDEX_COLUMNS)
        cursor.execute(
            """
            SELECT idx.relname, tbl.relname, am.amname, attr.attname,
                   ind.indisvalid, ind.indisready, ind.indisunique,
                   ind.indpred IS NULL
            FROM pg_catalog.pg_class AS idx
            JOIN pg_catalog.pg_namespace AS ns ON ns.oid = idx.relnamespace
            JOIN pg_catalog.pg_index AS ind ON ind.indexrelid = idx.oid
            JOIN pg_catalog.pg_class AS tbl ON tbl.oid = ind.indrelid
            JOIN pg_catalog.pg_am AS am ON am.oid = idx.relam
            JOIN pg_catalog.pg_attribute AS attr
              ON attr.attrelid = tbl.oid AND attr.attnum = ind.indkey[0]
            WHERE ns.nspname = current_schema()
              AND idx.relname = ANY(%s::text[])
              AND ind.indnatts = 1
            """,
            (index_names,),
        )
        actual_indexes = {
            str(name): (
                str(table),
                str(method),
                str(column),
                bool(valid),
                bool(ready),
                bool(unique),
                bool(no_predicate),
            )
            for name, table, method, column, valid, ready, unique, no_predicate
            in cursor.fetchall()
        }
        for name, (table, column) in _EXPECTED_INDEX_COLUMNS.items():
            if actual_indexes.get(name) != (
                table,
                "btree",
                column,
                True,
                True,
                False,
                True,
            ):
                raise PgVectorSchemaError(
                    f"incompatible pgvector schema definition for index {name}"
                )

    def _verify_or_initialize_state(
        self, cursor: Any, fingerprint: Optional[IndexFingerprint]
    ) -> None:
        expected = {
            "schema_version": str(_SCHEMA_VERSION),
            "dense_dim": str(self.dense_dim),
            "structural_dim": str(self.colbert_dim),
            "normalization": "l2",
            "quantization": "none",
        }
        cursor.execute(
            "SELECT key,value FROM rag3d_v2_meta WHERE key = ANY(%s::text[])",
            (list(expected),),
        )
        stored = {str(key): str(value) for key, value in cursor.fetchall()}
        for key, value in expected.items():
            if key in stored and stored[key] != value:
                raise PgVectorSchemaError(
                    f"incompatible pgvector schema state for {key}"
                )
            if key not in stored:
                cursor.execute(
                    "INSERT INTO rag3d_v2_meta(key,value) VALUES(%s,%s)",
                    (key, value),
                )

        cursor.execute(
            "SELECT key,value FROM rag3d_v2_meta "
            "WHERE key = ANY(%s::text[])",
            ([_FINGERPRINT_KEY, _FINGERPRINT_DIGEST_KEY],),
        )
        fingerprint_state = {
            str(key): str(value) for key, value in cursor.fetchall()
        }
        cursor.execute(
            "SELECT "
            "EXISTS(SELECT 1 FROM rag3d_v2_documents),"
            "EXISTS(SELECT 1 FROM rag3d_v2_chunks)"
        )
        population_row = cursor.fetchone()
        populated = bool(population_row[0] or population_row[1])

        if fingerprint is None:
            if populated or fingerprint_state:
                raise FingerprintMismatchError(
                    "an expected retrieval V2 fingerprint is required before "
                    "opening a populated or fingerprinted pgvector index"
                )
            self._fingerprint_verified = False
            return

        stored_payload = fingerprint_state.get(_FINGERPRINT_KEY)
        stored_digest = fingerprint_state.get(_FINGERPRINT_DIGEST_KEY)
        if stored_payload is None and stored_digest is None:
            if populated:
                raise FingerprintMismatchError(
                    "populated pgvector state has no verified retrieval V2 "
                    "fingerprint; reindex before opening it"
                )
            cursor.execute(
                "INSERT INTO rag3d_v2_meta(key,value) VALUES(%s,%s)",
                (_FINGERPRINT_KEY, fingerprint.canonical_json()),
            )
            cursor.execute(
                "INSERT INTO rag3d_v2_meta(key,value) VALUES(%s,%s)",
                (_FINGERPRINT_DIGEST_KEY, fingerprint.digest),
            )
            self._fingerprint_verified = True
            return
        if stored_payload is None or stored_digest is None:
            raise FingerprintMismatchError(
                "retrieval V2 fingerprint payload and digest must both exist"
            )
        try:
            payload = json.loads(stored_payload)
            stored_fingerprint = IndexFingerprint(**payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise FingerprintMismatchError(
                "stored retrieval V2 fingerprint is invalid"
            ) from None
        canonical_payload = stored_fingerprint.canonical_json()
        actual_digest = hashlib.sha256(stored_payload.encode("utf-8")).hexdigest()
        if stored_payload != canonical_payload or stored_digest != actual_digest:
            raise FingerprintMismatchError(
                "stored retrieval V2 fingerprint or digest is invalid"
            )
        fingerprint.assert_compatible(stored_fingerprint)
        self._fingerprint_verified = True

    def lock_fingerprint(self) -> None:
        if self._closed:
            raise PgVectorError("PgVectorStore is closed")
        if self._transaction_depth <= 0:
            raise PgVectorError(
                "fingerprint lock requires an active transaction"
            )
        self._acquire_advisory_xact_lock(
            self.db, _FINGERPRINT_LOCK_KEY, purpose="fingerprint"
        )

    def _require_verified_fingerprint(self) -> None:
        if not self._fingerprint_verified:
            raise FingerprintMismatchError(
                "a verified retrieval V2 fingerprint is required before writes"
            )

    def get_meta(self, key: str) -> Optional[str]:
        meta_key = validate_string_value("meta key", key, non_empty=True)
        row = self.db.execute(
            "SELECT value FROM rag3d_v2_meta WHERE key=%s", (meta_key,)
        ).fetchone()
        return str(row[0]) if row is not None else None

    def set_meta(self, key: str, value: str) -> None:
        self._require_verified_fingerprint()
        meta_key = validate_string_value("meta key", key, non_empty=True)
        meta_value = validate_string_value("meta value", value)
        self.db.execute(
            "INSERT INTO rag3d_v2_meta(key,value) VALUES(%s,%s) "
            "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (meta_key, meta_value),
        )

    def add_doc(
        self,
        source: str,
        title: str,
        n_tokens: int,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> int:
        self._require_verified_fingerprint()
        source_value = validate_string_value("source", source)
        title_value = validate_string_value("title", title)
        token_count = _validate_count("n_tokens", n_tokens)
        metadata = _serialize_document_metadata(meta)
        row = self.db.execute(
            """
            INSERT INTO rag3d_v2_documents(
              source,title,created,n_tokens,metadata
            ) VALUES(%s,%s,%s,%s,%s::jsonb)
            RETURNING id
            """,
            (source_value, title_value, time.time(), token_count, metadata),
        ).fetchone()
        return int(row[0])

    def add_parent(self, doc_id: int, text: str, n_tokens: int, pos: int) -> int:
        self._require_verified_fingerprint()
        document_id = _validate_non_negative_int("doc_id", doc_id)
        token_count = _validate_count("n_tokens", n_tokens)
        position = _validate_count("pos", pos)
        text_value = validate_string_value("text", text)
        row = self.db.execute(
            """
            INSERT INTO rag3d_v2_chunks(
              document_id,parent_id,kind,pos,text,context,n_tokens,created,
              importance,turn_no,embedding,structural_n_tok,structural_dim,
              structural_data
            ) VALUES(%s,NULL,'parent',%s,%s,%s,%s,%s,0.5,NULL,NULL,NULL,NULL,NULL)
            RETURNING id
            """,
            (
                document_id,
                position,
                text_value,
                text_value,
                token_count,
                time.time(),
            ),
        ).fetchone()
        return int(row[0])

    def add_chunk(
        self,
        doc_id: Optional[int],
        text: str,
        ctx: str,
        n_tokens: int,
        vec: Any,
        kind: str = "chunk",
        pos: int = 0,
        parent_id: Optional[int] = None,
        importance: float = 0.5,
        turn_no: Optional[int] = None,
    ) -> int:
        self._require_verified_fingerprint()
        document_id = _validate_optional_id("doc_id", doc_id)
        parent = _validate_optional_id("parent_id", parent_id)
        token_count = _validate_count("n_tokens", n_tokens)
        chunk_kind = _validate_kind(kind)
        position = _validate_chunk_position(chunk_kind, pos)
        turn = _validate_optional_count("turn_no", turn_no)
        text_value = validate_string_value("text", text)
        context_value = validate_string_value("ctx", ctx)
        if isinstance(importance, bool) or not isinstance(importance, (int, float)):
            raise TypeError("importance must be a real number, not bool")
        importance_value = float(importance)
        if not math.isfinite(importance_value) or not 0 <= importance_value <= 1:
            raise ValueError("importance must be finite and between 0 and 1")
        try:
            dense_value = vec.dense
            sparse_value = vec.sparse
            structural_value = vec.tokens
        except AttributeError:
            raise TypeError("vec must expose dense, sparse, and tokens") from None
        dense = _normalize_dense_vector(
            dense_value, self.dense_dim, name="embedding"
        )
        sparse = _validate_sparse_weights(sparse_value)
        structural = _validate_structural_vectors(
            structural_value,
            self.colbert_dim,
            name="structural vectors",
            allow_empty=False,
            max_tokens=self._max_structural_tokens,
        )
        structural_f16 = structural.astype(np.float16)
        terms = sorted(sparse)
        weights = [sparse[term] for term in terms]
        row = self.db.execute(
            f"""
            WITH new_chunk AS (
              INSERT INTO rag3d_v2_chunks(
                document_id,parent_id,kind,pos,text,context,n_tokens,created,
                importance,turn_no,embedding,structural_n_tok,structural_dim,
                structural_data
              ) VALUES(
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector({self.dense_dim}),
                %s,%s,%s
              )
              RETURNING id
            ), inserted_postings AS (
              INSERT INTO rag3d_v2_sparse_postings(term,chunk_id,weight)
              SELECT posting.term, (SELECT id FROM new_chunk), posting.weight
              FROM unnest(%s::bigint[], %s::real[]) AS posting(term,weight)
            )
            SELECT id FROM new_chunk
            """,
            (
                document_id,
                parent,
                chunk_kind,
                position,
                text_value,
                context_value,
                token_count,
                time.time(),
                importance_value,
                turn,
                dense,
                int(structural.shape[0]),
                self.colbert_dim,
                structural_f16.tobytes(),
                terms,
                weights,
            ),
        ).fetchone()
        return int(row[0])

    def delete_document(self, document_id: int) -> None:
        self._require_verified_fingerprint()
        identifier = _validate_non_negative_int("document_id", document_id)
        self.db.execute("DELETE FROM rag3d_v2_documents WHERE id=%s", (identifier,))

    def delete_chunk(self, chunk_id: int) -> None:
        self._require_verified_fingerprint()
        identifier = _validate_non_negative_int("chunk_id", chunk_id)
        self.db.execute("DELETE FROM rag3d_v2_chunks WHERE id=%s", (identifier,))

    _CHUNK_COLUMNS = (
        "id,document_id,parent_id,kind,pos,text,context,n_tokens,created,"
        "importance,turn_no,accessed_turn"
    )

    def get_chunks(self, ids: Sequence[int]) -> List[dict]:
        identifiers = _validate_ids("ids", ids)
        if not identifiers:
            return []
        rows = self.db.execute(
            f"SELECT {self._CHUNK_COLUMNS} FROM rag3d_v2_chunks "
            "WHERE id = ANY(%s::bigint[])",
            (identifiers,),
        ).fetchall()
        columns = [
            "id",
            "doc_id",
            "parent_id",
            "kind",
            "pos",
            "text",
            "ctx",
            "n_tokens",
            "created",
            "importance",
            "turn_no",
            "accessed_turn",
        ]
        by_id = {int(row[0]): dict(zip(columns, row)) for row in rows}
        return [by_id[identifier] for identifier in identifiers if identifier in by_id]

    def dense_vectors(self, ids: Sequence[int]) -> Dict[int, np.ndarray]:
        identifiers = _validate_ids("ids", ids)
        if not identifiers:
            return {}
        rows = self.db.execute(
            "SELECT id,embedding FROM rag3d_v2_chunks "
            "WHERE id = ANY(%s::bigint[]) AND embedding IS NOT NULL",
            (identifiers,),
        ).fetchall()
        return {
            int(chunk_id): _stored_vector_to_numpy(vector, self.dense_dim)
            for chunk_id, vector in rows
        }

    def dense_vecs(self, ids: Sequence[int]) -> Dict[int, np.ndarray]:
        return self.dense_vectors(ids)

    def touch_access(self, ids: Sequence[int], turn_no: int) -> None:
        self._require_verified_fingerprint()
        identifiers = _validate_ids("ids", ids)
        turn = _validate_count("turn_no", turn_no)
        if not identifiers:
            return
        self.db.execute(
            "UPDATE rag3d_v2_chunks SET accessed_turn=%s "
            "WHERE id = ANY(%s::bigint[])",
            (turn, identifiers),
        )

    @staticmethod
    def _validated_kinds(kinds: Sequence[str]) -> List[str]:
        if isinstance(kinds, (str, bytes)) or not isinstance(kinds, Sequence):
            raise TypeError("kinds must be a sequence of strings")
        maximum = DEFAULT_RETRIEVAL_LIMITS.max_filter_values
        if len(kinds) > maximum:
            raise ValueError("kinds contains too many values")
        checked = []
        for item_count, kind in enumerate(kinds, start=1):
            if item_count > maximum:
                raise ValueError("kinds contains too many values")
            checked.append(_validate_kind(kind, allow_parent=True))
        return checked

    def corpus_tokens(self, kinds: Sequence[str] = ("chunk", "turn")) -> int:
        checked = self._validated_kinds(kinds)
        if not checked:
            return 0
        row = self.db.execute(
            "SELECT COALESCE(SUM(n_tokens),0) FROM rag3d_v2_chunks "
            "WHERE kind = ANY(%s::text[])",
            (checked,),
        ).fetchone()
        return int(row[0])

    def all_texts(self, kinds: Sequence[str] = ("chunk", "turn")) -> List[dict]:
        checked = self._validated_kinds(kinds)
        if not checked:
            return []
        rows = self.db.execute(
            "SELECT id,kind,text,created,turn_no FROM rag3d_v2_chunks "
            "WHERE kind = ANY(%s::text[]) ORDER BY id ASC",
            (checked,),
        ).fetchall()
        columns = ["id", "kind", "text", "created", "turn_no"]
        return [dict(zip(columns, row)) for row in rows]

    def n_chunks(self) -> int:
        row = self.db.execute(
            f"SELECT COUNT(*) FROM rag3d_v2_chunks "
            f"WHERE kind IN {_SEARCHABLE_KINDS_SQL}"
        ).fetchone()
        return int(row[0])

    def last_turn_no(self) -> int:
        row = self.db.execute(
            "SELECT COALESCE(MAX(turn_no),0) FROM rag3d_v2_chunks WHERE kind='turn'"
        ).fetchone()
        return int(row[0])

    def neighbors(
        self, doc_id: Optional[int], positions: Sequence[int]
    ) -> List[dict]:
        if doc_id is None:
            return []
        document_id = _validate_non_negative_int("doc_id", doc_id)
        checked_positions = _validate_signed_positions(positions)
        if not checked_positions:
            return []
        rows = self.db.execute(
            "SELECT id,document_id,kind,pos,text FROM rag3d_v2_chunks "
            "WHERE document_id=%s AND kind='chunk' "
            "AND pos = ANY(%s::integer[]) ORDER BY pos ASC, id ASC",
            (document_id, checked_positions),
        ).fetchall()
        return [
            {
                "id": int(identifier),
                "doc_id": int(row_document_id),
                "kind": str(kind),
                "pos": int(pos),
                "text": str(text),
            }
            for identifier, row_document_id, kind, pos, text in rows
        ]

    def _dense_statement(
        self,
        query_vector: np.ndarray,
        k: int,
        filters: Optional[SearchFilters],
        *,
        exact: bool,
    ) -> Tuple[str, List[Any]]:
        filter_sql, filter_params = _build_filter_clause(filters)
        if exact:
            return (
                f"""
                SELECT c.id, c.embedding <=> %s AS distance
                FROM rag3d_v2_chunks AS c
                LEFT JOIN rag3d_v2_documents AS d ON d.id = c.document_id
                WHERE c.embedding IS NOT NULL AND {filter_sql}
                ORDER BY c.embedding <=> %s ASC, c.id ASC
                LIMIT %s
                """,
                [query_vector, *filter_params, query_vector, k],
            )
        # pgvector needs a pure distance ORDER BY in the inner query to keep
        # the HNSW path indexable.  We overfetch a hard-bounded candidate set
        # and apply the stable ID tie-break outside it. This is deterministic
        # within that candidate set; approximate retrieval cannot guarantee a
        # global tie boundary without reverting to an exact scan.
        candidate_limit = min(
            DEFAULT_RETRIEVAL_LIMITS.max_channel_k,
            max(k, k * _ANN_OVERFETCH_FACTOR),
        )
        return (
            f"""
            WITH nearest AS MATERIALIZED (
              SELECT c.id, c.embedding <=> %s AS distance
              FROM rag3d_v2_chunks AS c
              LEFT JOIN rag3d_v2_documents AS d ON d.id = c.document_id
              WHERE c.embedding IS NOT NULL AND {filter_sql}
              ORDER BY c.embedding <=> %s ASC
              LIMIT %s
            )
            SELECT id,distance FROM nearest
            ORDER BY distance ASC, id ASC
            LIMIT %s
            """,
            [query_vector, *filter_params, query_vector, candidate_limit, k],
        )

    def _set_local_statement_timeout(self) -> None:
        self.db.execute(
            "SELECT set_config('statement_timeout',%s,true)",
            (str(self.statement_timeout_ms),),
        )

    def _set_local_ann_options(self, k: int) -> None:
        effective_ef_search = max(self.ef_search, k)
        if effective_ef_search > 1_000:
            raise ValueError("effective ef_search must not exceed 1000")
        settings = [("hnsw.ef_search", str(effective_ef_search))]
        if self._pgvector_version_tuple >= (0, 8, 0):
            settings.extend(
                [
                    ("hnsw.iterative_scan", self.iterative_scan),
                    ("hnsw.max_scan_tuples", str(self.max_scan_tuples)),
                    (
                        "hnsw.scan_mem_multiplier",
                        format(self.scan_mem_multiplier, "g"),
                    ),
                ]
            )
        for name, value in settings:
            self.db.execute("SELECT set_config(%s,%s,true)", (name, value))

    def _resolve_dense_mode(self, exact: Optional[bool]) -> Tuple[str, bool]:
        if exact is not None and not isinstance(exact, bool):
            raise TypeError("exact must be bool or None")
        if exact is True:
            return "exact", False
        if exact is False:
            return "ann", True
        if self.search_mode == "exact":
            return "exact", False
        if self.search_mode == "ann":
            return "ann", True
        if self.capabilities.ann_dense_search:
            return "ann", False
        return "exact", False

    def _preflight_ann(
        self, statement: str, params: Sequence[Any]
    ) -> bool:
        row = self.db.execute(
            "EXPLAIN (FORMAT JSON) " + statement, params
        ).fetchone()
        audit = _sanitize_explain_document(row[0], exact=False)
        return bool(audit["plan"]["hnsw_used"])

    @staticmethod
    def _scored_dense_rows(rows: Sequence[Sequence[Any]]) -> List[Tuple[int, float]]:
        scored = []
        for chunk_id, raw_distance in rows:
            distance = float(raw_distance)
            score = 1.0 - distance
            if math.isfinite(score):
                scored.append((int(chunk_id), score))
        return scored

    def dense_search(
        self,
        qvec: Any,
        k: int,
        *,
        filters: Optional[SearchFilters] = None,
        exact: Optional[bool] = None,
    ) -> List[Tuple[int, float]]:
        _validate_filters(filters)
        selected_mode, ann_required = self._resolve_dense_mode(exact)
        limit = _validate_k(k)
        if limit == 0:
            return []
        query = _normalize_dense_vector(
            qvec, self.dense_dim, name="query_vector"
        )
        if selected_mode == "ann" and not self.capabilities.ann_dense_search:
            raise PgVectorHnswError(
                "ANN dense search requires a valid and ready HNSW index"
            )
        statement, params = self._dense_statement(
            query, limit, filters, exact=selected_mode == "exact"
        )
        with self.db.transaction():
            self._set_local_statement_timeout()
            if selected_mode == "exact":
                self.db.execute("SET LOCAL enable_indexscan = off")
                self.db.execute("SET LOCAL enable_bitmapscan = off")
            else:
                self._set_local_ann_options(limit)
                if not self._preflight_ann(statement, params):
                    if ann_required:
                        raise PgVectorHnswError(
                            "the natural PostgreSQL plan did not select the "
                            "verified HNSW index"
                        )
                    selected_mode = "exact"
                    statement, params = self._dense_statement(
                        query, limit, filters, exact=True
                    )
                    self.db.execute("SET LOCAL enable_indexscan = off")
                    self.db.execute("SET LOCAL enable_bitmapscan = off")
            rows = self.db.execute(statement, params).fetchall()
        self._last_dense_mode = selected_mode
        return self._scored_dense_rows(rows)

    def explain_dense(
        self,
        query_vector: Any,
        k: int,
        *,
        filters: Optional[SearchFilters] = None,
        exact: Optional[bool] = None,
    ) -> Mapping[str, Any]:
        limit = _validate_k(k)
        query = _normalize_dense_vector(
            query_vector, self.dense_dim, name="query_vector"
        )
        selected_mode, ann_required = self._resolve_dense_mode(exact)
        if selected_mode == "ann" and not self.capabilities.ann_dense_search:
            raise PgVectorHnswError(
                "ANN EXPLAIN requires a valid and ready HNSW index"
            )
        statement, params = self._dense_statement(
            query, limit, filters, exact=selected_mode == "exact"
        )
        with self.db.transaction():
            self._set_local_statement_timeout()
            if selected_mode == "exact":
                self.db.execute("SET LOCAL enable_indexscan = off")
                self.db.execute("SET LOCAL enable_bitmapscan = off")
            else:
                self._set_local_ann_options(limit)
                if not self._preflight_ann(statement, params):
                    if ann_required:
                        raise PgVectorHnswError(
                            "the natural PostgreSQL plan did not select the "
                            "verified HNSW index"
                        )
                    selected_mode = "exact"
                    statement, params = self._dense_statement(
                        query, limit, filters, exact=True
                    )
                    self.db.execute("SET LOCAL enable_indexscan = off")
                    self.db.execute("SET LOCAL enable_bitmapscan = off")
            explain = (
                "EXPLAIN (ANALYZE, BUFFERS, SETTINGS, FORMAT JSON) "
                + statement
            )
            row = self.db.execute(explain, params).fetchone()
        audit = _sanitize_explain_document(
            row[0], exact=selected_mode == "exact"
        )
        audit["backend"] = "pgvector"
        audit["configured_search_mode"] = self.search_mode
        self._last_dense_mode = selected_mode
        return audit

    def sparse_search(
        self,
        qsparse: Mapping[int, float],
        k: int,
        *,
        filters: Optional[SearchFilters] = None,
    ) -> List[Tuple[int, float]]:
        _validate_filters(filters)
        limit = _validate_k(k)
        weights = _validate_sparse_weights(qsparse)
        if limit == 0 or not weights:
            return []
        terms = sorted(weights)
        query_values = [weights[term] for term in terms]
        filter_sql, filter_params = _build_filter_clause(filters)
        statement = f"""
            WITH query_terms AS MATERIALIZED (
              SELECT query.term, query.query_weight
              FROM unnest(%s::bigint[], %s::double precision[])
                AS query(term,query_weight)
            ), corpus AS MATERIALIZED (
              SELECT GREATEST(COUNT(*)::double precision, 1.0) AS n
              FROM rag3d_v2_chunks
              WHERE kind IN {_SEARCHABLE_KINDS_SQL}
            ), term_df AS MATERIALIZED (
              SELECT posting.term, COUNT(*)::double precision AS df
              FROM rag3d_v2_sparse_postings AS posting
              JOIN query_terms AS query ON query.term = posting.term
              JOIN rag3d_v2_chunks AS eligible ON eligible.id = posting.chunk_id
              WHERE eligible.kind IN {_SEARCHABLE_KINDS_SQL}
              GROUP BY posting.term
            ), scored AS (
              SELECT posting.chunk_id,
                     SUM(
                       query.query_weight * posting.weight::double precision *
                       ln(1.0 + (corpus.n - term_df.df + 0.5) /
                         (term_df.df + 0.5))
                     ) AS score
              FROM query_terms AS query
              JOIN rag3d_v2_sparse_postings AS posting
                ON posting.term = query.term
              JOIN term_df ON term_df.term = posting.term
              CROSS JOIN corpus
              JOIN rag3d_v2_chunks AS c ON c.id = posting.chunk_id
              LEFT JOIN rag3d_v2_documents AS d ON d.id = c.document_id
              WHERE {filter_sql}
              GROUP BY posting.chunk_id
            )
            SELECT chunk_id,score FROM scored
            ORDER BY score DESC,chunk_id ASC
            LIMIT %s
        """
        with self.db.transaction():
            self._set_local_statement_timeout()
            rows = self.db.execute(
                statement,
                [terms, query_values, *filter_params, limit],
            ).fetchall()
        result = []
        for chunk_id, raw_score in rows:
            score = float(raw_score)
            if math.isfinite(score):
                result.append((int(chunk_id), score))
        return result

    def structural_rerank(
        self,
        qtokens: Any,
        candidate_ids: Sequence[int],
        k: int,
        *,
        filters: Optional[SearchFilters] = None,
    ) -> List[Tuple[int, float]]:
        _validate_filters(filters)
        limit = _validate_k(k)
        identifiers = _validate_ids("candidate_ids", candidate_ids)
        query = _validate_structural_vectors(
            qtokens,
            self.colbert_dim,
            name="query vectors",
            allow_empty=True,
            max_tokens=self._query_max_tokens,
        )
        if limit == 0 or not identifiers or query.shape[0] == 0:
            return []
        filter_sql, filter_params = _build_filter_clause(filters)
        with self.db.transaction():
            self._set_local_statement_timeout()
            rows = self.db.execute(
                f"""
                SELECT c.id,c.structural_n_tok,c.structural_dim,c.structural_data
                FROM rag3d_v2_chunks AS c
                LEFT JOIN rag3d_v2_documents AS d ON d.id = c.document_id
                WHERE c.id = ANY(%s::bigint[])
                  AND c.structural_data IS NOT NULL AND {filter_sql}
                """,
                [identifiers, *filter_params],
            ).fetchall()
        query_f32 = query.astype(np.float32, copy=False)
        scored: List[Tuple[int, float]] = []
        for chunk_id, n_tokens, dimension, payload in rows:
            stored_dimension = int(dimension)
            stored_tokens = int(n_tokens)
            if stored_dimension != self.colbert_dim or stored_tokens <= 0:
                raise PgVectorSchemaError("stored structural vector shape is invalid")
            if (
                stored_tokens > self._max_structural_tokens
                or stored_tokens * stored_dimension > _MAX_STRUCTURAL_VALUES
            ):
                raise PgVectorSchemaError(
                    "stored structural vector exceeds the configured resource limit"
                )
            expected_bytes = stored_tokens * stored_dimension * 2
            try:
                if len(payload) != expected_bytes:
                    raise PgVectorSchemaError(
                        "stored structural vector shape is invalid"
                    )
            except TypeError:
                raise PgVectorSchemaError(
                    "stored structural vector shape is invalid"
                ) from None
            document = np.frombuffer(bytes(payload), dtype=np.float16).reshape(
                stored_tokens, stored_dimension
            ).astype(np.float32)
            with np.errstate(all="ignore"):
                similarities = query_f32 @ document.T
                score = float(similarities.max(axis=1).mean())
            if math.isfinite(score):
                scored.append((int(chunk_id), score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]

    def colbert_scores(
        self, qtokens: Any, candidate_ids: Sequence[int]
    ) -> List[Tuple[int, float]]:
        return self.structural_rerank(
            qtokens, candidate_ids, len(candidate_ids)
        )

    def _inspect_hnsw_index(self) -> dict:
        row = self.db.execute(
            """
            SELECT ind.indisvalid, ind.indisready, am.amname, opc.opcname,
                   attr.attname,
                   pg_catalog.format_type(attr.atttypid, attr.atttypmod),
                   idx.reloptions, ind.indpred IS NULL, ind.indisunique
            FROM pg_catalog.pg_class AS idx
            JOIN pg_catalog.pg_namespace AS ns ON ns.oid = idx.relnamespace
            JOIN pg_catalog.pg_index AS ind ON ind.indexrelid = idx.oid
            JOIN pg_catalog.pg_class AS tbl ON tbl.oid = ind.indrelid
            JOIN pg_catalog.pg_am AS am ON am.oid = idx.relam
            JOIN pg_catalog.pg_opclass AS opc ON opc.oid = ind.indclass[0]
            JOIN pg_catalog.pg_attribute AS attr
              ON attr.attrelid = tbl.oid AND attr.attnum = ind.indkey[0]
            WHERE ns.nspname = current_schema()
              AND idx.relname = %s
              AND tbl.relname = 'rag3d_v2_chunks'
              AND ind.indnatts = 1
            """,
            (_HNSW_INDEX,),
        ).fetchone()
        if row is None:
            return self._empty_hnsw_status()
        (
            valid,
            ready,
            method,
            opclass,
            column,
            vector_type,
            reloptions,
            no_predicate,
            unique,
        ) = row
        options = {}
        for option in reloptions or []:
            name, separator, raw_value = str(option).partition("=")
            if separator and name in {"m", "ef_construction"}:
                try:
                    options[name] = int(raw_value)
                except ValueError:
                    options[name] = raw_value
        definition_valid = bool(
            method == "hnsw"
            and opclass == "vector_cosine_ops"
            and column == "embedding"
            and vector_type == f"vector({self.dense_dim})"
            and no_predicate
            and not unique
        )
        return {
            "exists": True,
            "valid": bool(valid),
            "ready": bool(ready),
            "definition_valid": definition_valid,
            "method": str(method),
            "opclass": str(opclass),
            "column": str(column),
            "vector_type": str(vector_type),
            "options": options,
            "index_name": _HNSW_INDEX,
        }

    def create_hnsw_index(
        self,
        m: int = 16,
        ef_construction: int = 64,
        *,
        concurrently: bool = False,
    ) -> Mapping[str, Any]:
        self._require_verified_fingerprint()
        m_value, ef_value = _validate_hnsw_build_options(m, ef_construction)
        if not isinstance(concurrently, bool):
            raise TypeError("concurrently must be bool")
        if concurrently and self._transaction_depth:
            raise PgVectorHnswError(
                "CREATE INDEX CONCURRENTLY cannot run inside transaction()"
            )

        requested_options = {"m": m_value, "ef_construction": ef_value}
        concurrent_sql = " CONCURRENTLY" if concurrently else ""
        statement = (
            f"CREATE INDEX{concurrent_sql} {_HNSW_INDEX} "
            "ON rag3d_v2_chunks USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {m_value}, ef_construction = {ef_value})"
        )
        creation_completed = False
        try:
            status = self._inspect_hnsw_index()
            if status["exists"] and status["valid"] and status["ready"]:
                if not status["definition_valid"]:
                    self._cache_hnsw_status(status)
                    raise PgVectorSchemaError(
                        "existing HNSW index has an incompatible definition"
                    )
                if status["options"] != requested_options:
                    self._cache_hnsw_status(status)
                    raise PgVectorSchemaError(
                        "existing HNSW index options differ from requested options"
                    )
                self._cache_hnsw_status(status)
                return {**dict(status), "created_by_caller": False}

            if status["exists"]:
                drop = (
                    "DROP INDEX CONCURRENTLY IF EXISTS "
                    if concurrently
                    else "DROP INDEX IF EXISTS "
                )
                self._execute_hnsw_ddl(
                    drop + _HNSW_INDEX,
                    concurrently=concurrently,
                )

            self._execute_hnsw_ddl(statement, concurrently=concurrently)
            creation_completed = True

            status = self._inspect_hnsw_index()
            if not (
                status["exists"]
                and status["valid"]
                and status["ready"]
                and status["definition_valid"]
                and status["options"] == requested_options
            ):
                try:
                    drop = (
                        "DROP INDEX CONCURRENTLY IF EXISTS "
                        if concurrently
                        else "DROP INDEX IF EXISTS "
                    )
                    self._execute_hnsw_ddl(
                        drop + _HNSW_INDEX,
                        concurrently=concurrently,
                    )
                except Exception:
                    pass
                self._cache_hnsw_status(self._empty_hnsw_status())
                raise PgVectorHnswError(
                    "HNSW index did not become valid, ready, and "
                    "definition-compatible"
                )
            self._cache_hnsw_status(status)
            return {**dict(status), "created_by_caller": True}
        except PgVectorSchemaError:
            raise
        except PgVectorHnswError:
            raise
        except Exception:
            # A concurrent creator may have won the race. Accept only the
            # complete requested catalog state; every recovery failure is
            # swallowed and replaced by a fixed, conninfo-safe error.
            try:
                recovered = self._inspect_hnsw_index()
                if (
                    recovered["exists"]
                    and recovered["valid"]
                    and recovered["ready"]
                    and recovered["definition_valid"]
                    and recovered["options"] == requested_options
                ):
                    self._cache_hnsw_status(recovered)
                    return {
                        **dict(recovered),
                        "created_by_caller": creation_completed,
                    }
                if recovered["exists"] and not (
                    recovered["valid"] and recovered["ready"]
                ):
                    drop = (
                        "DROP INDEX CONCURRENTLY IF EXISTS "
                        if concurrently
                        else "DROP INDEX IF EXISTS "
                    )
                    self._execute_hnsw_ddl(
                        drop + _HNSW_INDEX,
                        concurrently=concurrently,
                    )
            except Exception:
                pass
            self._cache_hnsw_status(self._empty_hnsw_status())
            raise PgVectorHnswError("HNSW index creation failed") from None

    def drop_hnsw_index(self, *, concurrently: bool = False) -> None:
        """Remove the fixed HNSW index through the bounded DDL executor."""

        self._require_verified_fingerprint()
        if not isinstance(concurrently, bool):
            raise TypeError("concurrently must be bool")
        if concurrently and self._transaction_depth:
            raise PgVectorHnswError(
                "DROP INDEX CONCURRENTLY cannot run inside transaction()"
            )
        statement = (
            "DROP INDEX CONCURRENTLY IF EXISTS "
            if concurrently
            else "DROP INDEX IF EXISTS "
        ) + _HNSW_INDEX
        try:
            self._execute_hnsw_ddl(statement, concurrently=concurrently)
        except Exception:
            raise PgVectorHnswError("HNSW index removal failed") from None
        self._cache_hnsw_status(self._empty_hnsw_status())

    @contextmanager
    def _hnsw_ddl_timeouts(self, *, concurrently: bool) -> Iterator[None]:
        if not concurrently:
            with self.db.transaction():
                self.db.execute(
                    "SELECT set_config('lock_timeout',%s,true)",
                    (str(self.lock_timeout_ms),),
                )
                self._set_local_statement_timeout()
                yield
            return

        previous = self.db.execute(
            "SELECT current_setting('lock_timeout'), "
            "current_setting('statement_timeout')"
        ).fetchone()
        if not previous or len(previous) != 2:
            raise PgVectorHnswError("could not read HNSW DDL timeout settings")
        configured = False
        try:
            self.db.execute(
                "SELECT set_config('lock_timeout',%s,false), "
                "set_config('statement_timeout',%s,false)",
                (str(self.lock_timeout_ms), str(self.statement_timeout_ms)),
            )
            configured = True
            yield
        finally:
            if configured:
                self.db.execute(
                    "SELECT set_config('lock_timeout',%s,false), "
                    "set_config('statement_timeout',%s,false)",
                    (str(previous[0]), str(previous[1])),
                )

    def _execute_hnsw_ddl(self, statement: str, *, concurrently: bool) -> None:
        with self._hnsw_ddl_timeouts(concurrently=concurrently):
            self.db.execute(statement)

    def _cache_hnsw_status(self, status: Mapping[str, Any]) -> None:
        self._hnsw_status_cache = dict(status)
        self._capabilities_cache = self._capabilities_for_status(
            self._hnsw_status_cache
        )

    def refresh_capabilities(self) -> dict:
        if self._closed:
            status = self._empty_hnsw_status()
        else:
            status = self._inspect_hnsw_index()
        self._cache_hnsw_status(status)
        return dict(self._hnsw_status_cache)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        if self._closed:
            raise PgVectorError("PgVectorStore is closed")
        self._transaction_depth += 1
        try:
            with self.db.transaction():
                yield
        finally:
            self._transaction_depth -= 1

    def commit(self) -> None:
        if self._closed:
            raise PgVectorError("PgVectorStore is closed")
        if self._transaction_depth == 0:
            self.db.commit()

    def metrics(self) -> Mapping[str, Any]:
        try:
            with self.db.transaction():
                self._set_local_statement_timeout()
                counts_row = self.db.execute(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM rag3d_v2_documents),
                      (SELECT COUNT(*) FROM rag3d_v2_chunks),
                      (SELECT COUNT(*) FROM rag3d_v2_sparse_postings)
                    """
                ).fetchone()
            return {
                "status": "ok",
                "backend": "pgvector",
                "counts": {
                    "documents": int(counts_row[0]),
                    "chunks": int(counts_row[1]),
                    "sparse_postings": int(counts_row[2]),
                },
            }
        except Exception:
            return {"status": "error", "backend": "pgvector"}

    def health(self, *, include_metrics: bool = False) -> Mapping[str, Any]:
        if not isinstance(include_metrics, bool):
            raise TypeError("include_metrics must be bool")
        result: Dict[str, Any] = {
            "status": "error",
            "backend": "pgvector",
            "postgres_version": self._postgres_version,
            "pgvector_version": self.pgvector_version,
            "schema_version": str(_SCHEMA_VERSION),
            "hnsw": dict(self._hnsw_status_cache),
            "search_mode": self.search_mode,
            "last_dense_mode": self._last_dense_mode,
        }
        if self._closed:
            return result
        try:
            self.db.execute("SELECT 1").fetchone()
            result["status"] = "ok"
        except Exception:
            return result
        if include_metrics:
            metrics = self.metrics()
            if metrics.get("status") == "ok":
                result["counts"] = metrics["counts"]
            else:
                result["status"] = "error"
        return result

    def close(self) -> None:
        if not self._closed:
            self.db.close()
            self._closed = True
            self._cache_hnsw_status(self._empty_hnsw_status())

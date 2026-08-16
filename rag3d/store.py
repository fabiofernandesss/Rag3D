"""Armazenamento tridimensional em SQLite + numpy.

Cada chunk é salvo nas três formas na hora da ingestão:
  dvecs    -> vetor denso (BLOB float32)
  postings -> termos esparsos invertidos (term -> chunk, peso)
  colvecs  -> matriz token a token (BLOB float16)

Sem servidor, sem dependência externa: um arquivo .db + cache em RAM.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Sized
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .backend import (
    DEFAULT_RETRIEVAL_LIMITS,
    BackendCapabilities,
    SearchFilters,
    normalize_sparse_weights,
    serialize_document_metadata,
    validate_string_value,
)
from .encoders import TriVec

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS docs(
  id INTEGER PRIMARY KEY, source TEXT, title TEXT,
  created REAL, n_tokens INTEGER, meta TEXT
);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY,
  doc_id INTEGER REFERENCES docs(id) ON DELETE CASCADE,
  parent_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
  kind TEXT NOT NULL DEFAULT 'chunk',   -- chunk | parent | summary | turn | rolling_summary
  pos INTEGER, text TEXT NOT NULL, ctx TEXT,
  n_tokens INTEGER, created REAL,
  importance REAL DEFAULT 0.5, turn_no INTEGER,
  accessed_turn INTEGER                 -- último turno em que a memória foi usada
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(kind);
CREATE TABLE IF NOT EXISTS dvecs(
  chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  data BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS postings(
  term INTEGER NOT NULL,
  chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  weight REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);
CREATE INDEX IF NOT EXISTS idx_postings_chunk ON postings(chunk_id);
CREATE TABLE IF NOT EXISTS colvecs(
  chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  n_tok INTEGER, dim INTEGER, data BLOB NOT NULL
);

-- Existing databases created before the FK schema cannot gain declarative
-- constraints through ALTER TABLE. These idempotent triggers provide the same
-- checks/cascades until an explicit offline table rebuild is performed.
CREATE TRIGGER IF NOT EXISTS rag3d_chunks_doc_insert_fk
BEFORE INSERT ON chunks
WHEN NEW.doc_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM docs WHERE id=NEW.doc_id)
BEGIN SELECT RAISE(ABORT, 'unknown document'); END;
CREATE TRIGGER IF NOT EXISTS rag3d_chunks_doc_update_fk
BEFORE UPDATE OF doc_id ON chunks
WHEN NEW.doc_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM docs WHERE id=NEW.doc_id)
BEGIN SELECT RAISE(ABORT, 'unknown document'); END;
CREATE TRIGGER IF NOT EXISTS rag3d_chunks_parent_insert_fk
BEFORE INSERT ON chunks
WHEN NEW.parent_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM chunks WHERE id=NEW.parent_id)
BEGIN SELECT RAISE(ABORT, 'unknown parent chunk'); END;
CREATE TRIGGER IF NOT EXISTS rag3d_chunks_parent_update_fk
BEFORE UPDATE OF parent_id ON chunks
WHEN NEW.parent_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM chunks WHERE id=NEW.parent_id)
BEGIN SELECT RAISE(ABORT, 'unknown parent chunk'); END;
CREATE TRIGGER IF NOT EXISTS rag3d_docs_delete_cascade
AFTER DELETE ON docs
BEGIN DELETE FROM chunks WHERE doc_id=OLD.id; END;
CREATE TRIGGER IF NOT EXISTS rag3d_chunks_delete_cascade
AFTER DELETE ON chunks
BEGIN
  DELETE FROM dvecs WHERE chunk_id=OLD.id;
  DELETE FROM postings WHERE chunk_id=OLD.id;
  DELETE FROM colvecs WHERE chunk_id=OLD.id;
  UPDATE chunks SET parent_id=NULL WHERE parent_id=OLD.id;
END;
CREATE TRIGGER IF NOT EXISTS rag3d_dvecs_chunk_insert_fk
BEFORE INSERT ON dvecs
WHEN NOT EXISTS(SELECT 1 FROM chunks WHERE id=NEW.chunk_id)
BEGIN SELECT RAISE(ABORT, 'unknown dense chunk'); END;
CREATE TRIGGER IF NOT EXISTS rag3d_postings_chunk_insert_fk
BEFORE INSERT ON postings
WHEN NOT EXISTS(SELECT 1 FROM chunks WHERE id=NEW.chunk_id)
BEGIN SELECT RAISE(ABORT, 'unknown sparse chunk'); END;
CREATE TRIGGER IF NOT EXISTS rag3d_colvecs_chunk_insert_fk
BEFORE INSERT ON colvecs
WHEN NOT EXISTS(SELECT 1 FROM chunks WHERE id=NEW.chunk_id)
BEGIN SELECT RAISE(ABORT, 'unknown structural chunk'); END;
"""

_ALL_CHUNK_KINDS = frozenset(
    {"chunk", "parent", "summary", "turn", "rolling_summary"}
)
_MAX_BIGINT = 2**63 - 1
_MAX_INTEGER = 2**31 - 1
_MAX_DENSE_DIM = DEFAULT_RETRIEVAL_LIMITS.max_dense_dim
_MAX_STRUCTURAL_DIM = DEFAULT_RETRIEVAL_LIMITS.max_structural_dim
_MAX_STRUCTURAL_TOKEN_ROWS = DEFAULT_RETRIEVAL_LIMITS.max_structural_tokens
_MAX_STRUCTURAL_VALUES = DEFAULT_RETRIEVAL_LIMITS.max_structural_values
# sqlite3.Connection.getlimit() exists only on Python 3.11+. Older runtimes
# keep the historical SQLite SQLITE_LIMIT_VARIABLE_NUMBER default of 999.
_SQLITE_HISTORICAL_VARIABLE_LIMIT = 999
_MAX_MAXSIM_PAIRS = DEFAULT_RETRIEVAL_LIMITS.max_structural_values


def _sqlite_variable_limit(connection: sqlite3.Connection) -> int:
    getter = getattr(connection, "getlimit", None)
    if getter is None:
        return _SQLITE_HISTORICAL_VARIABLE_LIMIT
    return int(getter(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))


def _validate_non_negative_bigint(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    if value > _MAX_BIGINT:
        raise ValueError(f"{name} must fit a signed bigint")
    return value


def _validate_optional_bigint(name: str, value: object) -> Optional[int]:
    if value is None:
        return None
    return _validate_non_negative_bigint(name, value)


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


def _validate_kind(kind: object, *, allow_parent: bool = False) -> str:
    if not isinstance(kind, str) or kind not in _ALL_CHUNK_KINDS:
        raise ValueError("kind must be a supported RAG3D chunk kind")
    if kind == "parent" and not allow_parent:
        raise ValueError("parent rows must be created with add_parent")
    return kind


def _validate_stored_position(kind: object, pos: object) -> int:
    chunk_kind = _validate_kind(kind)
    if isinstance(pos, bool) or not isinstance(pos, int):
        raise TypeError("pos must be an integer, not bool")
    if pos < -1 or pos > _MAX_INTEGER:
        raise ValueError("pos must fit the supported signed 32-bit integer range")
    if pos == -1 and chunk_kind != "summary":
        raise ValueError("pos=-1 is reserved for summary chunks")
    return pos


def _validate_parent_position(pos: object) -> int:
    position = _validate_count("pos", pos)
    return position


def _validate_importance(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("importance must be a real number, not bool")
    importance = float(value)
    if not math.isfinite(importance) or not 0 <= importance <= 1:
        raise ValueError("importance must be finite and between 0 and 1")
    return importance


def _bounded_integer_sequence(
    values: Sequence[int],
    name: str,
    *,
    allow_negative: bool = False,
    maximum: int = DEFAULT_RETRIEVAL_LIMITS.max_pool,
) -> List[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    if len(values) > maximum:
        raise ValueError(f"{name} exceed maximum of {maximum}")
    result = []
    for item_count, value in enumerate(values, start=1):
        if item_count > maximum:
            raise ValueError(f"{name} exceed maximum of {maximum}")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be integers, not bool")
        if allow_negative:
            if value < -(2**31) or value > _MAX_INTEGER:
                raise ValueError(f"{name} must fit a signed 32-bit integer")
        else:
            _validate_non_negative_bigint(name, value)
        result.append(value)
    return result


def _validated_kinds(kinds: Sequence[str]) -> List[str]:
    if isinstance(kinds, (str, bytes)) or not isinstance(kinds, Sequence):
        raise TypeError("kinds must be a sequence of strings")
    maximum = DEFAULT_RETRIEVAL_LIMITS.max_filter_values
    if len(kinds) > maximum:
        raise ValueError(f"kinds exceed maximum of {maximum}")
    result = []
    for item_count, kind in enumerate(kinds, start=1):
        if item_count > maximum:
            raise ValueError(f"kinds exceed maximum of {maximum}")
        result.append(_validate_kind(kind, allow_parent=True))
    return result


def _validated_sparse_weights(weights: Mapping[int, float]) -> Dict[int, float]:
    normalized = normalize_sparse_weights(weights)
    for term in normalized:
        if term < -(2**63) or term > _MAX_BIGINT:
            raise ValueError("sparse term ID must fit a signed bigint")
    return normalized


def _validate_dimension(name: str, value: object, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool")
    if value < 1 or value > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _validate_structural_token_limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_structural_tokens must be an integer, not bool")
    if value <= 0:
        raise ValueError("max_structural_tokens must be positive")
    return min(value, _MAX_STRUCTURAL_TOKEN_ROWS)


def _fingerprint_structural_token_limit(payload: object) -> Optional[int]:
    if not isinstance(payload, str) or len(payload) > 16 * 1024:
        return None
    try:
        decoded = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    value = decoded.get("max_structural_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return min(value, _MAX_STRUCTURAL_TOKEN_ROWS)


def _bounded_structural_vectors(vectors: object, maximum: int) -> np.ndarray:
    if isinstance(vectors, Sized) and len(vectors) > maximum:
        raise ValueError(
            f"structural token rows exceed maximum of {maximum}"
        )
    if isinstance(vectors, np.ndarray):
        if vectors.ndim == 2 and vectors.shape[1] > _MAX_STRUCTURAL_DIM:
            raise ValueError(
                "structural vector dimension exceeds maximum of "
                f"{_MAX_STRUCTURAL_DIM}"
            )
        if vectors.size > _MAX_STRUCTURAL_VALUES:
            raise ValueError(
                "structural vector values exceed maximum of "
                f"{_MAX_STRUCTURAL_VALUES}"
            )
        materialized = vectors
    else:
        if isinstance(vectors, (str, bytes)):
            raise TypeError("structural vectors must be numeric")
        try:
            iterator = iter(vectors)  # type: ignore[arg-type]
        except TypeError:
            raise TypeError("structural vectors must be an iterable of rows") from None
        rows = []
        value_count = 0
        for row_count, row in enumerate(iterator, start=1):
            if row_count > maximum:
                raise ValueError(
                    f"structural token rows exceed maximum of {maximum}"
                )
            if isinstance(row, (str, bytes)):
                raise TypeError("structural vectors must be numeric")
            if isinstance(row, np.ndarray):
                row_size = int(row.size)
                if row_size > _MAX_STRUCTURAL_DIM:
                    raise ValueError(
                        "structural vector dimension exceeds maximum of "
                        f"{_MAX_STRUCTURAL_DIM}"
                    )
                if row_size > _MAX_STRUCTURAL_VALUES - value_count:
                    raise ValueError(
                        "structural vector values exceed maximum of "
                        f"{_MAX_STRUCTURAL_VALUES}"
                    )
                value_count += row_size
                rows.append(row)
                continue
            if isinstance(row, Sized):
                row_size = len(row)
                if row_size > _MAX_STRUCTURAL_DIM:
                    raise ValueError(
                        "structural vector dimension exceeds maximum of "
                        f"{_MAX_STRUCTURAL_DIM}"
                    )
                if row_size > _MAX_STRUCTURAL_VALUES - value_count:
                    raise ValueError(
                        "structural vector values exceed maximum of "
                        f"{_MAX_STRUCTURAL_VALUES}"
                    )
            try:
                row_iterator = iter(row)
            except TypeError:
                raise TypeError("structural vectors must be numeric") from None
            bounded_row = []
            for scalar_count, scalar in enumerate(row_iterator, start=1):
                if scalar_count > _MAX_STRUCTURAL_DIM:
                    raise ValueError(
                        "structural vector dimension exceeds maximum of "
                        f"{_MAX_STRUCTURAL_DIM}"
                    )
                value_count += 1
                if value_count > _MAX_STRUCTURAL_VALUES:
                    raise ValueError(
                        "structural vector values exceed maximum of "
                        f"{_MAX_STRUCTURAL_VALUES}"
                    )
                bounded_row.append(scalar)
            rows.append(bounded_row)
        materialized = rows
    try:
        value = np.asarray(materialized, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("structural vectors must be numeric") from exc
    if value.ndim != 2:
        raise ValueError("structural vectors must have two dimensions")
    if value.shape[1] > _MAX_STRUCTURAL_DIM:
        raise ValueError(
            "structural vector dimension exceeds maximum of "
            f"{_MAX_STRUCTURAL_DIM}"
        )
    if value.shape[0] > maximum:
        raise ValueError(
            f"structural token rows exceed maximum of {maximum}"
        )
    if value.size > _MAX_STRUCTURAL_VALUES:
        raise ValueError(
            "structural vector values exceed maximum of "
            f"{_MAX_STRUCTURAL_VALUES}"
        )
    if not np.isfinite(value).all():
        raise ValueError("structural vector values must be finite")
    return value


def _bounded_float_maxsim(query: np.ndarray, document: np.ndarray) -> float:
    """Compute exact MaxSim while bounding every pairwise score matrix."""

    document_tokens = int(document.shape[0])
    query_tokens = int(query.shape[0])
    if query_tokens == 0 or document_tokens == 0:
        return 0.0
    query_batch = max(1, _MAX_MAXSIM_PAIRS // document_tokens)
    maxima = np.empty(query_tokens, dtype=np.float32)
    for offset in range(0, query_tokens, query_batch):
        end = min(offset + query_batch, query_tokens)
        with np.errstate(all="ignore"):
            pair_scores = np.matmul(query[offset:end], document.T)
        maxima[offset:end] = pair_scores.max(axis=1)
    return float(maxima.mean())


class _SparseScoreAggregate:
    """SQLite aggregate preserving the legacy Python float operation order."""

    def __init__(self) -> None:
        self.total = 0.0

    def step(
        self,
        query_weight: float,
        document_weight: float,
        universe: int,
        document_frequency: int,
    ) -> None:
        inverse_document_frequency = float(
            np.log(
                1.0
                + (float(universe) - float(document_frequency) + 0.5)
                / (float(document_frequency) + 0.5)
            )
        )
        self.total += (
            float(query_weight)
            * float(document_weight)
            * inverse_document_frequency
        )

    def finalize(self) -> float:
        return self.total


class TriStore:
    def __init__(
        self, path: Path, *, max_structural_tokens: int = _MAX_STRUCTURAL_TOKEN_ROWS
    ):
        structural_token_limit = _validate_structural_token_limit(
            max_structural_tokens
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self._transaction_depth = 0
        self._savepoint_sequence = 0
        self._max_structural_tokens = structural_token_limit
        self._fingerprint_structural_tokens: Optional[int] = None
        self.db.execute("PRAGMA foreign_keys=ON")
        if self.db.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            self.db.close()
            raise RuntimeError("SQLite foreign-key enforcement is unavailable")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        # migração: bases criadas antes da coluna accessed_turn
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(chunks)")}
        if "accessed_turn" not in cols:
            self.db.execute("ALTER TABLE chunks ADD COLUMN accessed_turn INTEGER")
            self.db.commit()
        self._validate_referential_integrity()
        stored_fingerprint = self.db.execute(
            "SELECT value FROM meta WHERE key='retrieval_v2_fingerprint'"
        ).fetchone()
        if stored_fingerprint:
            self._fingerprint_structural_tokens = (
                _fingerprint_structural_token_limit(stored_fingerprint[0])
            )
        self._stored_dense_dim: Optional[int] = None
        self._stored_structural_dim: Optional[int] = None
        self._refresh_storage_dimensions()
        self._dense_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (ids, matriz)
        self._dense_cache_ver: int = -1  # PRAGMA data_version do cache atual

    @property
    def backend_name(self) -> str:
        return "sqlite"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            exact_dense_search=True,
            sparse_search=True,
            structural_rerank=True,
            transactions=True,
        )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit a unit atomically, using a savepoint when already nested."""
        savepoint: Optional[str] = None
        if self._transaction_depth > 0 or self.db.in_transaction:
            self._savepoint_sequence += 1
            savepoint = f"rag3d_sp_{self._savepoint_sequence}"
            self.db.execute(f"SAVEPOINT {savepoint}")
        else:
            # Fingerprint publication and document ingest both need a writer
            # lock before their initial read.  This prevents two SQLite
            # processes from concurrently publishing incompatible metadata.
            self.db.execute("BEGIN IMMEDIATE")
        self._transaction_depth += 1
        try:
            yield
        except BaseException:
            if savepoint is None:
                self.db.rollback()
            else:
                self.db.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                self.db.execute(f"RELEASE SAVEPOINT {savepoint}")
            self._dense_cache = None
            self._refresh_storage_dimensions()
            raise
        else:
            try:
                if savepoint is None:
                    self.db.commit()
                else:
                    self.db.execute(f"RELEASE SAVEPOINT {savepoint}")
            except BaseException:
                if savepoint is None:
                    self.db.rollback()
                else:
                    self.db.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    self.db.execute(f"RELEASE SAVEPOINT {savepoint}")
                self._dense_cache = None
                self._refresh_storage_dimensions()
                raise
        finally:
            self._transaction_depth -= 1

    @staticmethod
    def _validate_k(k: int) -> int:
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer, not bool")
        if k < 0:
            raise ValueError("k must be non-negative")
        if k > DEFAULT_RETRIEVAL_LIMITS.max_channel_k:
            raise ValueError(
                f"k exceeds maximum of {DEFAULT_RETRIEVAL_LIMITS.max_channel_k}"
            )
        return k

    @staticmethod
    def _validate_filters(filters: Optional[SearchFilters]) -> None:
        if filters is None:
            return
        if not isinstance(filters, SearchFilters):
            raise TypeError("filters must be SearchFilters")
        if not filters.is_empty:
            raise NotImplementedError("search filters are not supported by sqlite")

    @staticmethod
    def _dense_input(vector: np.ndarray) -> np.ndarray:
        if isinstance(vector, Sized) and len(vector) > _MAX_DENSE_DIM:
            raise ValueError(
                f"dense vector dimension exceeds maximum of {_MAX_DENSE_DIM}"
            )
        if isinstance(vector, np.ndarray):
            materialized: object = vector
        else:
            if isinstance(vector, (str, bytes)):
                raise TypeError("dense vector must be numeric")
            try:
                iterator = iter(vector)
            except TypeError:
                raise TypeError("dense vector must be numeric") from None
            bounded = []
            for item_count, scalar in enumerate(iterator, start=1):
                if item_count > _MAX_DENSE_DIM:
                    raise ValueError(
                        "dense vector dimension exceeds maximum of "
                        f"{_MAX_DENSE_DIM}"
                    )
                if not np.isscalar(scalar):
                    raise ValueError("dense vector must have one dimension")
                bounded.append(scalar)
            materialized = bounded
        try:
            value = np.asarray(materialized, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("dense vector must be numeric") from exc
        if value.ndim != 1:
            raise ValueError("dense vector must have one dimension")
        if value.shape[0] > _MAX_DENSE_DIM:
            raise ValueError(
                f"dense vector dimension exceeds maximum of {_MAX_DENSE_DIM}"
            )
        if not np.isfinite(value).all():
            raise ValueError("dense vector values must be finite")
        return value

    def _token_input(self, vectors: np.ndarray) -> np.ndarray:
        maximum = min(
            getattr(self, "_max_structural_tokens", _MAX_STRUCTURAL_TOKEN_ROWS),
            getattr(
                self,
                "_fingerprint_structural_tokens",
                None,
            )
            or _MAX_STRUCTURAL_TOKEN_ROWS,
            _MAX_STRUCTURAL_TOKEN_ROWS,
        )
        return _bounded_structural_vectors(vectors, maximum)

    @staticmethod
    def _sparse_input(weights: Mapping[int, float]) -> Dict[int, float]:
        return _validated_sparse_weights(weights)

    @staticmethod
    def _candidate_ids(ids: Sequence[int]) -> List[int]:
        return _bounded_integer_sequence(ids, "candidate pool")

    def _sqlite_batches(
        self, values: Sequence[int], *, reserved_variables: int = 0
    ) -> Iterator[Sequence[int]]:
        variable_limit = _sqlite_variable_limit(self.db)
        batch_size = variable_limit - reserved_variables
        if batch_size < 1:
            raise sqlite3.OperationalError(
                "SQLite variable limit is too small for this operation"
            )
        for offset in range(0, len(values), batch_size):
            yield values[offset : offset + batch_size]

    def _base_vec(self, vec: TriVec) -> Tuple[np.ndarray, Dict[int, float], np.ndarray]:
        if not isinstance(vec, TriVec):
            raise TypeError("vec must be TriVec")
        dense = self._dense_input(vec.dense)
        tokens = self._token_input(vec.tokens)
        sparse = self._sparse_input(vec.sparse)
        if dense.size == 0:
            raise ValueError("dense vector dimension must be positive")
        if tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("structural vector dimensions must be positive")
        return dense, sparse, tokens

    def _validated_vec(self, vec: TriVec) -> Tuple[np.ndarray, Dict[int, float], np.ndarray]:
        dense, sparse, tokens = self._base_vec(vec)
        if (
            self._stored_dense_dim is not None
            and dense.shape[0] != self._stored_dense_dim
        ):
            raise ValueError(
                f"dense vector dimension mismatch; expected {self._stored_dense_dim}"
            )
        if (
            self._stored_structural_dim is not None
            and tokens.shape[1] != self._stored_structural_dim
        ):
            raise ValueError(
                "structural vector dimension mismatch; expected "
                f"{self._stored_structural_dim}"
            )
        return dense, sparse, tokens

    def _refresh_storage_dimensions(self) -> None:
        dense_bounds = self.db.execute(
            "SELECT MIN(length(data)), MAX(length(data)) FROM dvecs"
        ).fetchone()
        if dense_bounds[0] != dense_bounds[1]:
            raise ValueError("stored dense vectors have mixed dimensions")
        if dense_bounds[0] is not None and dense_bounds[0] % np.dtype(np.float32).itemsize:
            raise ValueError("stored dense vector has an invalid byte length")
        self._stored_dense_dim = (
            int(dense_bounds[0] // np.dtype(np.float32).itemsize)
            if dense_bounds[0] is not None
            else None
        )
        if self._stored_dense_dim is not None:
            _validate_dimension(
                "stored dense vector dimension",
                self._stored_dense_dim,
                _MAX_DENSE_DIM,
            )

        structural_bounds = self.db.execute(
            "SELECT MIN(dim), MAX(dim) FROM colvecs"
        ).fetchone()
        if structural_bounds[0] != structural_bounds[1]:
            raise ValueError("stored structural vectors have mixed dimensions")
        self._stored_structural_dim = (
            int(structural_bounds[0])
            if structural_bounds[0] is not None
            else None
        )
        if self._stored_structural_dim is not None:
            _validate_dimension(
                "stored structural vector dimension",
                self._stored_structural_dim,
                _MAX_STRUCTURAL_DIM,
            )

    def _validate_referential_integrity(self) -> None:
        checks = (
            "SELECT COUNT(*) FROM chunks c LEFT JOIN docs d ON d.id=c.doc_id "
            "WHERE c.doc_id IS NOT NULL AND d.id IS NULL",
            "SELECT COUNT(*) FROM chunks c LEFT JOIN chunks p ON p.id=c.parent_id "
            "WHERE c.parent_id IS NOT NULL AND p.id IS NULL",
            "SELECT COUNT(*) FROM dvecs d LEFT JOIN chunks c ON c.id=d.chunk_id "
            "WHERE c.id IS NULL",
            "SELECT COUNT(*) FROM postings p LEFT JOIN chunks c ON c.id=p.chunk_id "
            "WHERE c.id IS NULL",
            "SELECT COUNT(*) FROM colvecs v LEFT JOIN chunks c ON c.id=v.chunk_id "
            "WHERE c.id IS NULL",
        )
        if any(int(self.db.execute(statement).fetchone()[0]) for statement in checks):
            self.db.close()
            raise RuntimeError("SQLite index contains orphaned rows")

    @contextmanager
    def _write_savepoint(self) -> Iterator[None]:
        """Rollback one public compound write without committing caller state."""
        if not self.db.in_transaction:
            # Preserve the legacy commit bridge: add_chunk starts a pending
            # transaction but remains visible only after caller commit().
            self.db.execute("BEGIN")
        self._savepoint_sequence += 1
        savepoint = f"rag3d_write_{self._savepoint_sequence}"
        self.db.execute(f"SAVEPOINT {savepoint}")
        try:
            yield
        except BaseException:
            self.db.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            self.db.execute(f"RELEASE SAVEPOINT {savepoint}")
            self._dense_cache = None
            self._refresh_storage_dimensions()
            raise
        else:
            try:
                self.db.execute(f"RELEASE SAVEPOINT {savepoint}")
            except BaseException:
                self.db.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                self.db.execute(f"RELEASE SAVEPOINT {savepoint}")
                self._dense_cache = None
                self._refresh_storage_dimensions()
                raise

    # ------------------------------------------------------------- meta ---

    def get_meta(self, key: str) -> Optional[str]:
        meta_key = validate_string_value("meta key", key, non_empty=True)
        row = self.db.execute(
            "SELECT value FROM meta WHERE key=?", (meta_key,)
        ).fetchone()
        value = row[0] if row else None
        if meta_key == "retrieval_v2_fingerprint":
            self._fingerprint_structural_tokens = (
                _fingerprint_structural_token_limit(value)
            )
        return value

    def set_meta(self, key: str, value: str) -> None:
        meta_key = validate_string_value("meta key", key, non_empty=True)
        meta_value = validate_string_value("meta value", value)
        was_in_transaction = self.db.in_transaction
        self.db.execute(
            "INSERT OR REPLACE INTO meta VALUES(?,?)", (meta_key, meta_value)
        )
        if meta_key == "retrieval_v2_fingerprint":
            self._fingerprint_structural_tokens = (
                _fingerprint_structural_token_limit(meta_value)
            )
        if self._transaction_depth == 0 and not was_in_transaction:
            self.db.commit()

    def lock_fingerprint(self) -> None:
        """The outer ``BEGIN IMMEDIATE`` already owns SQLite's writer lock."""
        if self._transaction_depth == 0:
            raise RuntimeError("fingerprint lock requires an active transaction")

    # ------------------------------------------------------------ escrita ---

    def add_doc(
        self,
        source: str,
        title: str,
        n_tokens: int,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> int:
        source_value = validate_string_value("source", source)
        title_value = validate_string_value("title", title)
        token_count = _validate_count("n_tokens", n_tokens)
        metadata = serialize_document_metadata(meta)
        cur = self.db.execute(
            "INSERT INTO docs(source,title,created,n_tokens,meta) VALUES(?,?,?,?,?)",
            (source_value, title_value, time.time(), token_count, metadata),
        )
        return int(cur.lastrowid)

    def add_chunk(
        self,
        doc_id: Optional[int],
        text: str,
        ctx: str,
        n_tokens: int,
        vec: TriVec,
        kind: str = "chunk",
        pos: int = 0,
        parent_id: Optional[int] = None,
        importance: float = 0.5,
        turn_no: Optional[int] = None,
    ) -> int:
        document_id = _validate_optional_bigint("doc_id", doc_id)
        parent = _validate_optional_bigint("parent_id", parent_id)
        token_count = _validate_count("n_tokens", n_tokens)
        chunk_kind = _validate_kind(kind)
        position = _validate_stored_position(chunk_kind, pos)
        importance_value = _validate_importance(importance)
        turn = _validate_optional_count("turn_no", turn_no)
        text_value = validate_string_value("text", text)
        context_value = validate_string_value("ctx", ctx)
        dense, sparse, tokens = self._validated_vec(vec)
        with np.errstate(over="ignore", invalid="ignore"):
            col = tokens.astype(np.float16)
        if not np.isfinite(col).all():
            raise ValueError("structural vector values must remain finite in float16")
        with self._write_savepoint():
            cur = self.db.execute(
                "INSERT INTO chunks(doc_id,parent_id,kind,pos,text,ctx,n_tokens,created,importance,turn_no)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
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
                ),
            )
            cid = int(cur.lastrowid)
            self.db.execute(
                "INSERT INTO dvecs VALUES(?,?)", (cid, dense.tobytes())
            )
            self.db.executemany(
                "INSERT INTO postings VALUES(?,?,?)",
                [(t, cid, w) for t, w in sparse.items()],
            )
            self.db.execute(
                "INSERT INTO colvecs VALUES(?,?,?,?)",
                (cid, col.shape[0], col.shape[1], col.tobytes()),
            )
        self._stored_dense_dim = int(dense.shape[0])
        self._stored_structural_dim = int(tokens.shape[1])
        self._dense_cache = None
        return cid

    def add_parent(self, doc_id: int, text: str, n_tokens: int, pos: int) -> int:
        """Nó pai (small-to-big): não é indexado, só recuperado por expansão."""
        document_id = _validate_non_negative_bigint("doc_id", doc_id)
        token_count = _validate_count("n_tokens", n_tokens)
        position = _validate_parent_position(pos)
        text_value = validate_string_value("text", text)
        cur = self.db.execute(
            "INSERT INTO chunks(doc_id,kind,pos,text,ctx,n_tokens,created) VALUES(?,?,?,?,?,?,?)",
            (
                document_id,
                "parent",
                position,
                text_value,
                text_value,
                token_count,
                time.time(),
            ),
        )
        return int(cur.lastrowid)

    def commit(self) -> None:
        if self._transaction_depth == 0:
            self.db.commit()

    def delete_chunk(self, chunk_id: int) -> None:
        identifier = _validate_non_negative_bigint("chunk_id", chunk_id)
        with self.transaction():
            self.db.execute("DELETE FROM chunks WHERE id=?", (identifier,))
        self._dense_cache = None

    def delete_document(self, document_id: int) -> None:
        identifier = _validate_non_negative_bigint("document_id", document_id)
        with self.transaction():
            self.db.execute("DELETE FROM docs WHERE id=?", (identifier,))
        self._dense_cache = None

    # ------------------------------------------------------------ leitura ---

    def get_chunks(self, ids: Sequence[int]) -> List[dict]:
        identifiers = _bounded_integer_sequence(ids, "chunk IDs")
        if not identifiers:
            return []
        rows = []
        for batch in self._sqlite_batches(identifiers):
            q = ",".join("?" * len(batch))
            rows.extend(
                self.db.execute(
                    "SELECT id,doc_id,parent_id,kind,pos,text,ctx,n_tokens,"
                    "created,importance,turn_no,accessed_turn "
                    f"FROM chunks WHERE id IN ({q})",
                    list(batch),
                ).fetchall()
            )
        cols = ["id", "doc_id", "parent_id", "kind", "pos", "text", "ctx", "n_tokens", "created", "importance", "turn_no", "accessed_turn"]
        by_id = {r[0]: dict(zip(cols, r)) for r in rows}
        return [by_id[identifier] for identifier in identifiers if identifier in by_id]

    def dense_vecs(self, ids: Sequence[int]) -> Dict[int, np.ndarray]:
        """Vetores densos dos ids (para a seleção fermiônica/DPP)."""
        identifiers = _bounded_integer_sequence(ids, "chunk IDs")
        if not identifiers:
            return {}
        rows = []
        for batch in self._sqlite_batches(list(dict.fromkeys(identifiers))):
            q = ",".join("?" * len(batch))
            rows.extend(
                self.db.execute(
                    f"SELECT chunk_id, data FROM dvecs WHERE chunk_id IN ({q})",
                    list(batch),
                ).fetchall()
            )
        return {int(cid): np.frombuffer(blob, dtype=np.float32) for cid, blob in rows}

    def dense_vectors(self, ids: Sequence[int]) -> Dict[int, np.ndarray]:
        return self.dense_vecs(ids)

    def touch_access(self, ids: Sequence[int], turn_no: int) -> None:
        """Recência conta do último USO, não da criação (fórmula de Stanford)."""
        identifiers = _bounded_integer_sequence(ids, "chunk IDs")
        turn = _validate_count("turn_no", turn_no)
        if not identifiers:
            return
        with self.transaction():
            for batch in self._sqlite_batches(
                list(dict.fromkeys(identifiers)), reserved_variables=1
            ):
                q = ",".join("?" * len(batch))
                self.db.execute(
                    f"UPDATE chunks SET accessed_turn=? WHERE id IN ({q})",
                    [turn] + list(batch),
                )

    def corpus_tokens(self, kinds: Sequence[str] = ("chunk", "turn")) -> int:
        checked = list(dict.fromkeys(_validated_kinds(kinds)))
        if not checked:
            return 0
        total = 0
        for batch in self._sqlite_batches(checked):
            q = ",".join("?" * len(batch))
            row = self.db.execute(
                f"SELECT COALESCE(SUM(n_tokens),0) FROM chunks WHERE kind IN ({q})",
                list(batch),
            ).fetchone()
            total += int(row[0])
        return total

    def all_texts(self, kinds: Sequence[str] = ("chunk", "turn")) -> List[dict]:
        checked = list(dict.fromkeys(_validated_kinds(kinds)))
        if not checked:
            return []
        rows = []
        for batch in self._sqlite_batches(checked):
            q = ",".join("?" * len(batch))
            rows.extend(
                self.db.execute(
                    "SELECT id,kind,text,created,turn_no FROM chunks "
                    f"WHERE kind IN ({q}) ORDER BY id",
                    list(batch),
                ).fetchall()
            )
        rows.sort(key=lambda row: int(row[0]))
        return [dict(zip(["id", "kind", "text", "created", "turn_no"], r)) for r in rows]

    def n_chunks(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM chunks WHERE kind IN ('chunk','turn','summary')").fetchone()[0])

    def last_turn_no(self) -> int:
        row = self.db.execute("SELECT COALESCE(MAX(turn_no),0) FROM chunks WHERE kind='turn'").fetchone()
        return int(row[0])

    def neighbors(self, doc_id, positions):
        """Chunks vizinhos do mesmo doc em posições dadas (costura Rag3D)."""
        if doc_id is None:
            return []
        document_id = _validate_non_negative_bigint("doc_id", doc_id)
        checked_positions = _bounded_integer_sequence(
            positions, "positions", allow_negative=True
        )
        if not checked_positions:
            return []
        rows = []
        for batch in self._sqlite_batches(
            list(dict.fromkeys(checked_positions)), reserved_variables=1
        ):
            q = ",".join("?" * len(batch))
            rows.extend(
                self.db.execute(
                    "SELECT id, doc_id, kind, pos, text FROM chunks WHERE doc_id=? "
                    f"AND kind='chunk' AND pos IN ({q}) ORDER BY pos ASC, id ASC",
                    [document_id] + list(batch),
                ).fetchall()
            )
        rows.sort(key=lambda row: (int(row[3]), int(row[0])))
        return [
            dict(zip(["id", "doc_id", "kind", "pos", "text"], row))
            for row in rows
        ]

    # ------------------------------------------------- eixo 1: denso -------

    def _dense_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        # PRAGMA data_version muda quando OUTRA conexão grava no arquivo — assim
        # um `chat` de longa duração enxerga um `ingest` feito noutro terminal.
        dv = self.db.execute("PRAGMA data_version").fetchone()[0]
        if dv != self._dense_cache_ver:
            self._dense_cache = None
            self._dense_cache_ver = dv
        if self._dense_cache is None:
            rows = self.db.execute(
                "SELECT d.chunk_id, d.data FROM dvecs d JOIN chunks c ON c.id=d.chunk_id"
                " WHERE c.kind IN ('chunk','turn','summary') ORDER BY d.chunk_id"
            ).fetchall()
            if not rows:
                self._dense_cache = (np.zeros(0, dtype=np.int64), np.zeros((0, 1), dtype=np.float32))
            else:
                ids = np.array([r[0] for r in rows], dtype=np.int64)
                # concatena os BLOBs uma vez e reinterpreta — evita o loop
                # frombuffer+stack por linha (byte-idêntico ao np.stack). O
                # resultado é read-only, mas dense_search só faz `mat @ qvec`.
                buf = b"".join(r[1] for r in rows)
                mat = np.frombuffer(buf, dtype=np.float32).reshape(len(rows), -1)
                self._dense_cache = (ids, mat)
        return self._dense_cache

    def dense_search(
        self,
        qvec: np.ndarray,
        k: int,
        *,
        filters: Optional[SearchFilters] = None,
        exact: Optional[bool] = None,
    ) -> List[Tuple[int, float]]:
        self._validate_filters(filters)
        if exact is not None and not isinstance(exact, bool):
            raise TypeError("exact must be bool or None")
        if exact is False:
            raise NotImplementedError("ANN dense search is unavailable; use exact=True")
        k = self._validate_k(k)
        if k == 0:
            return []
        query = self._dense_input(qvec)
        ids, mat = self._dense_matrix()
        if len(ids) == 0:
            return []
        if query.shape[0] != mat.shape[1]:
            raise ValueError(
                f"dense vector dimension mismatch; expected {mat.shape[1]}"
            )
        # errstate: numpy+Accelerate no macOS emite flags FPE espúrias em matmul
        # (resultados corretos; ver numpy#25864-família). Valores validados finitos.
        with np.errstate(all="ignore"):
            sims = mat @ query
        if not np.isfinite(sims).all():
            raise ValueError("dense scores must be finite")
        order = sorted(
            range(len(ids)), key=lambda index: (-float(sims[index]), int(ids[index]))
        )[: min(k, len(ids))]
        return [(int(ids[index]), float(sims[index])) for index in order]

    # ------------------------------------------------ eixo 2: esparso ------

    def sparse_search(
        self,
        qsparse: Mapping[int, float],
        k: int,
        *,
        filters: Optional[SearchFilters] = None,
    ) -> List[Tuple[int, float]]:
        self._validate_filters(filters)
        k = self._validate_k(k)
        qsparse = self._sparse_input(qsparse)
        if k == 0 or not qsparse:
            return []
        query_payload = json.dumps(
            [[term, weight] for term, weight in qsparse.items()],
            separators=(",", ":"),
        )
        self.db.create_aggregate(
            "rag3d_sparse_score", 4, _SparseScoreAggregate
        )
        rows = self.db.execute(
            "SELECT contribution.chunk_id, rag3d_sparse_score("
            "contribution.query_weight, contribution.document_weight, "
            "contribution.n_docs, contribution.df) AS score FROM ("
            "  SELECT p.chunk_id, "
            "  CAST(json_extract(q.value,'$[1]') AS REAL) AS query_weight, "
            "  p.weight AS document_weight, u.n_docs, d.df "
            "  FROM json_each(?) q "
            "  JOIN postings p INDEXED BY idx_postings_term "
            "  ON p.term=CAST(json_extract(q.value,'$[0]') AS INTEGER) "
            "  JOIN chunks c ON c.id=p.chunk_id AND c.kind IN ('chunk','turn','summary') "
            "  JOIN ("
            "    SELECT p2.term, COUNT(DISTINCT p2.chunk_id) AS df "
            "    FROM json_each(?) q2 "
            "    JOIN postings p2 INDEXED BY idx_postings_term "
            "    ON p2.term=CAST(json_extract(q2.value,'$[0]') AS INTEGER) "
            "    JOIN chunks c2 ON c2.id=p2.chunk_id AND c2.kind IN ('chunk','turn','summary') "
            "    GROUP BY p2.term"
            "  ) d ON d.term=p.term "
            "  CROSS JOIN ("
            "    SELECT MAX(1,COUNT(*)) AS n_docs FROM chunks "
            "    WHERE kind IN ('chunk','turn','summary')"
            "  ) u ORDER BY CAST(q.key AS INTEGER) ASC, p.rowid ASC"
            ") contribution GROUP BY contribution.chunk_id "
            "ORDER BY score DESC, contribution.chunk_id ASC LIMIT ?",
            (query_payload, query_payload, k),
        ).fetchall()
        if any(not math.isfinite(float(score)) for _chunk_id, score in rows):
            raise ValueError("sparse scores must be finite")
        return [(int(chunk_id), float(score)) for chunk_id, score in rows]

    # ------------------------------------- eixo 3: estrutural (MaxSim) -----

    def colbert_scores(self, qtokens: np.ndarray, candidate_ids: Sequence[int]) -> List[Tuple[int, float]]:
        """MaxSim: para cada token da consulta, o melhor token do chunk; média."""
        candidate_ids = self._candidate_ids(candidate_ids)
        qtokens = self._token_input(qtokens)
        if len(candidate_ids) == 0 or qtokens.size == 0:
            return []
        rows = []
        for batch in self._sqlite_batches(list(dict.fromkeys(candidate_ids))):
            q = ",".join("?" * len(batch))
            rows.extend(
                self.db.execute(
                    "SELECT chunk_id,n_tok,dim,data FROM colvecs "
                    f"WHERE chunk_id IN ({q})",
                    list(batch),
                ).fetchall()
            )
        qt = qtokens.astype(np.float32)
        out: List[Tuple[int, float]] = []
        for cid, n_tok, dim, blob in rows:
            stored_tokens = _validate_dimension(
                "stored structural token rows",
                n_tok,
                _MAX_STRUCTURAL_TOKEN_ROWS,
            )
            stored_dimension = _validate_dimension(
                "stored structural vector dimension",
                dim,
                _MAX_STRUCTURAL_DIM,
            )
            if stored_tokens * stored_dimension > _MAX_STRUCTURAL_VALUES:
                raise ValueError(
                    "stored structural vector values exceed maximum of "
                    f"{_MAX_STRUCTURAL_VALUES}"
                )
            expected_bytes = (
                stored_tokens * stored_dimension * np.dtype(np.float16).itemsize
            )
            if not isinstance(blob, (bytes, bytearray, memoryview)) or len(blob) != expected_bytes:
                raise ValueError("stored structural vector has an invalid byte length")
            dt = np.frombuffer(blob, dtype=np.float16).reshape(
                stored_tokens, stored_dimension
            ).astype(np.float32)
            if not np.isfinite(dt).all():
                raise ValueError("stored structural vector values must be finite")
            if qtokens.shape[1] != stored_dimension:
                raise ValueError(
                    "structural vector dimension mismatch; expected "
                    f"{stored_dimension}"
                )
            score = _bounded_float_maxsim(qt, dt)
            if not math.isfinite(score):
                raise ValueError("structural scores must be finite")
            out.append((int(cid), score))
        out.sort(key=lambda x: (-x[1], x[0]))
        return out

    def structural_rerank(
        self,
        qtokens: np.ndarray,
        candidate_ids: Sequence[int],
        k: int,
        *,
        filters: Optional[SearchFilters] = None,
    ) -> List[Tuple[int, float]]:
        self._validate_filters(filters)
        k = self._validate_k(k)
        candidate_ids = self._candidate_ids(candidate_ids)
        qtokens = self._token_input(qtokens)
        if k == 0 or not candidate_ids or qtokens.size == 0:
            return []
        return self.colbert_scores(qtokens, candidate_ids)[:k]

    def health(self) -> dict:
        try:
            self.db.execute("SELECT 1").fetchone()
        except Exception:
            return {"status": "error", "backend": self.backend_name}
        return {"status": "ok", "backend": self.backend_name}

    def close(self) -> None:
        self._dense_cache = None
        self.db.close()

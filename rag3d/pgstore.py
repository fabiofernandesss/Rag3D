"""Backend Postgres PURO — sem pgvector, sem extensão nenhuma.

Cada chunk vira um Holograma Textual (ver holo.py). O que o Postgres guarda
é só o que ele já sabe indexar há 30 anos: BIGINT, INT[], BYTEA, REAL.

  eixo 1 (semântico)  : assinatura em coluna BIT(1024); distância de Hamming
                        = bit_count(sig # consulta) — XOR e popcount NATIVOS
                        do Postgres 14+. Em bases grandes, o pré-filtro por
                        facetas (INT[] + índice GIN, operador &&) corta a
                        varredura. Depois, re-pontuação exata pelo eco int8
                        em Python.
  eixo 2 (léxico)     : tabela invertida comum (term BIGINT, weight REAL)
                        com B-tree — SQL clássico.
  eixo 3 (estrutural) : constelação binária em BYTEA; MaxSim por XOR +
                        popcount sobre os candidatos, em numpy.

Interface idêntica ao TriStore (SQLite): o resto do TriRAG nem percebe.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Sized
from contextlib import contextmanager
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
from .holo import HOLO_BITS, TOKEN_BITS, Holographer

_KINDS_SEARCH = "('chunk','turn','summary')"
_KINDS_SPARSE = _KINDS_SEARCH

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS holo_meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS holo_docs(
  id BIGSERIAL PRIMARY KEY, source TEXT, title TEXT,
  created DOUBLE PRECISION, n_tokens INT, meta TEXT
);
CREATE TABLE IF NOT EXISTS holo_grams(
  id BIGSERIAL PRIMARY KEY,
  doc_id BIGINT CONSTRAINT holo_grams_doc_fk
    REFERENCES holo_docs(id) ON DELETE CASCADE,
  parent_id BIGINT CONSTRAINT holo_grams_parent_fk
    REFERENCES holo_grams(id) ON DELETE SET NULL,
  kind TEXT NOT NULL DEFAULT 'chunk',
  pos INT, text TEXT NOT NULL, ctx TEXT,
  n_tokens INT, created DOUBLE PRECISION,
  importance REAL DEFAULT 0.5, turn_no INT, accessed_turn INT,
  sig BIT({HOLO_BITS}),
  bands INT[],
  echo BYTEA,
  constellation BYTEA, n_tok INT
);
CREATE INDEX IF NOT EXISTS idx_grams_kind ON holo_grams(kind);
CREATE INDEX IF NOT EXISTS idx_grams_doc ON holo_grams(doc_id);
CREATE INDEX IF NOT EXISTS idx_grams_bands ON holo_grams USING GIN(bands);
CREATE TABLE IF NOT EXISTS holo_spectrum(
  term BIGINT NOT NULL,
  gram_id BIGINT NOT NULL CONSTRAINT holo_spectrum_gram_fk
    REFERENCES holo_grams(id) ON DELETE CASCADE,
  weight REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spectrum_term ON holo_spectrum(term);
CREATE INDEX IF NOT EXISTS idx_spectrum_gram ON holo_spectrum(gram_id);

-- Idempotent online hardening for schemas created by older Python/JS/Java
-- releases. NOT VALID avoids an exclusive validation scan while adding the
-- constraint; VALIDATE then fails closed if legacy orphans exist.
DO $rag3d$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid='holo_grams'::regclass AND conname='holo_grams_doc_fk'
  ) THEN
    ALTER TABLE holo_grams ADD CONSTRAINT holo_grams_doc_fk
      FOREIGN KEY(doc_id) REFERENCES holo_docs(id) ON DELETE CASCADE NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid='holo_grams'::regclass AND conname='holo_grams_parent_fk'
  ) THEN
    ALTER TABLE holo_grams ADD CONSTRAINT holo_grams_parent_fk
      FOREIGN KEY(parent_id) REFERENCES holo_grams(id) ON DELETE SET NULL NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid='holo_spectrum'::regclass AND conname='holo_spectrum_gram_fk'
  ) THEN
    ALTER TABLE holo_spectrum ADD CONSTRAINT holo_spectrum_gram_fk
      FOREIGN KEY(gram_id) REFERENCES holo_grams(id) ON DELETE CASCADE NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint c
    WHERE c.conrelid='holo_grams'::regclass
      AND c.conname='holo_grams_doc_fk' AND c.contype='f'
      AND c.confrelid='holo_docs'::regclass AND c.confdeltype='c'
      AND c.conkey=ARRAY[(SELECT attnum FROM pg_attribute
        WHERE attrelid='holo_grams'::regclass AND attname='doc_id')]
      AND c.confkey=ARRAY[(SELECT attnum FROM pg_attribute
        WHERE attrelid='holo_docs'::regclass AND attname='id')]
  ) THEN
    RAISE EXCEPTION 'incompatible holo_grams_doc_fk constraint';
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint c
    WHERE c.conrelid='holo_grams'::regclass
      AND c.conname='holo_grams_parent_fk' AND c.contype='f'
      AND c.confrelid='holo_grams'::regclass AND c.confdeltype='n'
      AND c.conkey=ARRAY[(SELECT attnum FROM pg_attribute
        WHERE attrelid='holo_grams'::regclass AND attname='parent_id')]
      AND c.confkey=ARRAY[(SELECT attnum FROM pg_attribute
        WHERE attrelid='holo_grams'::regclass AND attname='id')]
  ) THEN
    RAISE EXCEPTION 'incompatible holo_grams_parent_fk constraint';
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint c
    WHERE c.conrelid='holo_spectrum'::regclass
      AND c.conname='holo_spectrum_gram_fk' AND c.contype='f'
      AND c.confrelid='holo_grams'::regclass AND c.confdeltype='c'
      AND c.conkey=ARRAY[(SELECT attnum FROM pg_attribute
        WHERE attrelid='holo_spectrum'::regclass AND attname='gram_id')]
      AND c.confkey=ARRAY[(SELECT attnum FROM pg_attribute
        WHERE attrelid='holo_grams'::regclass AND attname='id')]
  ) THEN
    RAISE EXCEPTION 'incompatible holo_spectrum_gram_fk constraint';
  END IF;
END
$rag3d$;
ALTER TABLE holo_grams VALIDATE CONSTRAINT holo_grams_doc_fk;
ALTER TABLE holo_grams VALIDATE CONSTRAINT holo_grams_parent_fk;
ALTER TABLE holo_spectrum VALIDATE CONSTRAINT holo_spectrum_gram_fk;
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
_MAX_MAXSIM_PAIRS = DEFAULT_RETRIEVAL_LIMITS.max_structural_values
_TOKEN_SIGNATURE_BYTES = TOKEN_BITS // 8


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


def _validate_importance(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("importance must be a real number, not bool")
    importance = float(value)
    if not math.isfinite(importance) or not 0 <= importance <= 1:
        raise ValueError("importance must be finite and between 0 and 1")
    return importance


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


def _bounded_binary_maxsim(holo: Holographer, query: bytes, document: object) -> float:
    """Delegate binary MaxSim in batches with a bounded XOR tensor."""

    if not isinstance(document, (bytes, bytearray, memoryview)):
        raise ValueError("stored structural constellation must be bytes")
    query_bytes = len(query)
    document_bytes = len(document)
    if (
        query_bytes % _TOKEN_SIGNATURE_BYTES
        or document_bytes % _TOKEN_SIGNATURE_BYTES
    ):
        raise ValueError("stored structural constellation has an invalid byte length")
    query_tokens = query_bytes // _TOKEN_SIGNATURE_BYTES
    document_tokens = document_bytes // _TOKEN_SIGNATURE_BYTES
    if query_tokens == 0 or document_tokens == 0:
        return 0.0
    if query_tokens > _MAX_STRUCTURAL_TOKEN_ROWS:
        raise ValueError(
            "structural token rows exceed maximum of "
            f"{_MAX_STRUCTURAL_TOKEN_ROWS}"
        )
    if document_tokens > _MAX_STRUCTURAL_TOKEN_ROWS:
        raise ValueError(
            "stored structural token rows exceed maximum of "
            f"{_MAX_STRUCTURAL_TOKEN_ROWS}"
        )

    document_payload = bytes(document)
    query_batch = max(1, _MAX_MAXSIM_PAIRS // document_tokens)
    if query_tokens <= query_batch:
        score = float(holo.binary_maxsim(query, document_payload))
        if not math.isfinite(score):
            raise ValueError("structural scores must be finite")
        return score

    weighted_score = 0.0
    for offset in range(0, query_tokens, query_batch):
        end = min(offset + query_batch, query_tokens)
        query_slice = query[
            offset * _TOKEN_SIGNATURE_BYTES : end * _TOKEN_SIGNATURE_BYTES
        ]
        batch_score = float(holo.binary_maxsim(query_slice, document_payload))
        if not math.isfinite(batch_score):
            raise ValueError("structural scores must be finite")
        weighted_score += batch_score * (end - offset)
    score = weighted_score / query_tokens
    if not math.isfinite(score):
        raise ValueError("structural scores must be finite")
    return score


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


class PgHoloStore:
    """Mesma interface do TriStore, sobre Postgres comum."""

    # acima disso, a busca semântica usa o pré-filtro de facetas
    BAND_PREFILTER_THRESHOLD = 20000
    # candidatos re-pontuados pelo eco (multiplicador do k pedido)
    PREFETCH_FACTOR = 4
    PREFETCH_MIN = 200
    # Stable, non-secret lock namespace: ASCII "RAG3DFP2" as signed BIGINT.
    FINGERPRINT_LOCK_ID = 0x5241473344465032
    FINGERPRINT_LOCK_TIMEOUT_MS = 5_000
    SCHEMA_LOCK_TIMEOUT_MS = 5_000
    SCHEMA_STATEMENT_TIMEOUT_MS = 30_000

    def __init__(
        self,
        dsn: str,
        dense_dim: int,
        colbert_dim: int,
        *,
        max_structural_tokens: int = _MAX_STRUCTURAL_TOKEN_ROWS,
    ):
        dsn_value = validate_string_value("dsn", dsn, non_empty=True)
        dense_dimension = _validate_dimension(
            "dense_dim", dense_dim, _MAX_DENSE_DIM
        )
        structural_dimension = _validate_dimension(
            "colbert_dim", colbert_dim, _MAX_STRUCTURAL_DIM
        )
        structural_token_limit = _validate_structural_token_limit(
            max_structural_tokens
        )
        try:
            holographer = Holographer(dense_dimension, structural_dimension)
        except Exception as exc:
            raise RuntimeError(
                "failed to initialize postgres-holo projections"
            ) from exc
        import psycopg  # dependência só deste backend

        # autocommit=True: leituras não deixam a conexão "idle in transaction"
        # (um `chat` fica parado no prompt sem segurar transação aberta). As
        # escritas compostas usam `with self.db.transaction()` para atomicidade.
        try:
            self.db = psycopg.connect(dsn_value, autocommit=True)
        except Exception as exc:
            raise RuntimeError(
                "failed to connect postgres-holo backend "
                f"({type(exc).__name__})"
            ) from exc
        try:
            schema_lock_timeout_ms = int(self.SCHEMA_LOCK_TIMEOUT_MS)
            schema_statement_timeout_ms = int(
                self.SCHEMA_STATEMENT_TIMEOUT_MS
            )
            if not 0 < schema_lock_timeout_ms <= 60_000:
                raise ValueError("invalid schema lock timeout")
            if not 0 < schema_statement_timeout_ms <= 60_000:
                raise ValueError("invalid schema statement timeout")
            with self.db.transaction():
                with self.db.cursor() as cur:
                    cur.execute(
                        "SET LOCAL lock_timeout = "
                        f"'{schema_lock_timeout_ms}ms'"
                    )
                    cur.execute(
                        "SET LOCAL statement_timeout = "
                        f"'{schema_statement_timeout_ms}ms'"
                    )
                    cur.execute(SCHEMA)
        except Exception as exc:
            try:
                self.db.close()
            except Exception:
                pass
            raise RuntimeError(
                "failed to initialize postgres-holo schema"
            ) from exc
        self.holo = holographer
        self._dense_dim = dense_dimension
        self._colbert_dim = structural_dimension
        self._max_structural_tokens = structural_token_limit
        self._fingerprint_structural_tokens: Optional[int] = None
        self._transaction_depth = 0
        try:
            stored_fingerprint = self.db.execute(
                "SELECT value FROM holo_meta WHERE key=%s",
                ("retrieval_v2_fingerprint",),
            ).fetchone()
        except Exception as exc:
            try:
                self.db.close()
            except Exception:
                pass
            raise RuntimeError(
                "failed to initialize postgres-holo backend"
            ) from exc
        if stored_fingerprint:
            self._fingerprint_structural_tokens = (
                _fingerprint_structural_token_limit(stored_fingerprint[0])
            )

    @property
    def backend_name(self) -> str:
        return "postgres-holo"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            ann_dense_search=True,
            sparse_search=True,
            structural_rerank=True,
            transactions=True,
            quantized_vector=True,
            # The adapter cannot know whether its current corpus was encoded
            # with BGE-M3 or the cross-language Hash legacy mode.  Claim the
            # conservative capability; Hash parity remains an encoder trait.
            cross_language_index=False,
        )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Use Psycopg's transaction/savepoint context over autocommit."""
        with self.db.transaction():
            self._transaction_depth += 1
            try:
                yield
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
            raise NotImplementedError(
                "search filters are not supported by postgres-holo"
            )

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
    def _bounded_ids(
        ids: Sequence[int],
        name: str = "candidate IDs",
        *,
        allow_negative: bool = False,
    ) -> List[int]:
        if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
            raise TypeError(f"{name} must be a sequence of integers")
        maximum = DEFAULT_RETRIEVAL_LIMITS.max_pool
        if len(ids) > maximum:
            raise ValueError(
                f"{name} exceed maximum of {maximum}"
            )
        values = []
        for item_count, value in enumerate(ids, start=1):
            if item_count > maximum:
                raise ValueError(f"{name} exceed maximum of {maximum}")
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be integers, not bool")
            if allow_negative:
                if value < -(2**31) or value > _MAX_INTEGER:
                    raise ValueError(f"{name} must fit a signed 32-bit integer")
            else:
                _validate_non_negative_bigint(name, value)
            values.append(value)
        return values

    def _validated_vec(self, vec: TriVec) -> Tuple[np.ndarray, Dict[int, float], np.ndarray]:
        if not isinstance(vec, TriVec):
            raise TypeError("vec must be TriVec")
        dense = self._dense_input(vec.dense)
        tokens = self._token_input(vec.tokens)
        sparse = self._sparse_input(vec.sparse)
        if dense.shape[0] != self._dense_dim:
            raise ValueError(
                f"dense vector dimension mismatch; expected {self._dense_dim}"
            )
        if tokens.shape[0] == 0 or tokens.shape[1] != self._colbert_dim:
            raise ValueError(
                f"structural vector dimension mismatch; expected {self._colbert_dim}"
            )
        return dense, sparse, tokens

    # ------------------------------------------------------------- meta ---

    def get_meta(self, key: str) -> Optional[str]:
        meta_key = validate_string_value("meta key", key, non_empty=True)
        row = self.db.execute(
            "SELECT value FROM holo_meta WHERE key=%s", (meta_key,)
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
        self.db.execute(
            "INSERT INTO holo_meta VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (meta_key, meta_value),
        )
        if meta_key == "retrieval_v2_fingerprint":
            self._fingerprint_structural_tokens = (
                _fingerprint_structural_token_limit(meta_value)
            )

    def lock_fingerprint(self) -> None:
        """Serialize metadata read/compare/publish within the current xact."""
        if getattr(self, "_transaction_depth", 0) == 0:
            raise RuntimeError("fingerprint lock requires an active transaction")
        timeout_ms = int(self.FINGERPRINT_LOCK_TIMEOUT_MS)
        if timeout_ms <= 0 or timeout_ms > 60_000:
            raise RuntimeError("invalid postgres-holo fingerprint lock timeout")
        try:
            self.db.execute(f"SET LOCAL lock_timeout = '{timeout_ms}ms'")
            self.db.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (self.FINGERPRINT_LOCK_ID,),
            )
        except Exception as exc:
            raise RuntimeError(
                "postgres-holo fingerprint lock acquisition failed"
            ) from exc

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
        row = self.db.execute(
            "INSERT INTO holo_docs(source,title,created,n_tokens,meta) VALUES(%s,%s,%s,%s,%s) RETURNING id",
            (source_value, title_value, time.time(), token_count, metadata),
        ).fetchone()
        return int(row[0])

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
        sig = self.holo.sign_dense(dense)
        bits = self.holo.sig_to_bitstring(sig)
        bands = self.holo.bands_of(sig)
        echo = self.holo.quantize(dense)
        const = self.holo.sign_tokens(tokens)
        # gram + espectro numa ÚNICA query via CTE: um round-trip por chunk (era
        # 2) e atômica por si só (sem BEGIN/COMMIT avulso).
        terms = list(sparse.keys())
        weights = [sparse[t] for t in terms]
        row = self.db.execute(
            f"WITH g AS ("
            f"  INSERT INTO holo_grams(doc_id,parent_id,kind,pos,text,ctx,n_tokens,created,"
            f"    importance,turn_no,sig,bands,echo,constellation,n_tok) VALUES("
            f"    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::bit({HOLO_BITS}),%s,%s,%s,%s) RETURNING id"
            f"), s AS ("
            f"  INSERT INTO holo_spectrum(term, gram_id, weight)"
            f"  SELECT u.term, (SELECT id FROM g), u.weight"
            f"  FROM unnest(%s::bigint[], %s::real[]) AS u(term, weight)"
            f") SELECT id FROM g",
            (
                document_id, parent, chunk_kind, position, text_value, context_value,
                token_count, time.time(), importance_value, turn, bits, bands,
                echo, const, int(tokens.shape[0]),
                terms, weights,
            ),
        ).fetchone()
        return int(row[0])

    def add_parent(self, doc_id: int, text: str, n_tokens: int, pos: int) -> int:
        document_id = _validate_non_negative_bigint("doc_id", doc_id)
        token_count = _validate_count("n_tokens", n_tokens)
        position = _validate_count("pos", pos)
        text_value = validate_string_value("text", text)
        row = self.db.execute(
            "INSERT INTO holo_grams(doc_id,kind,pos,text,ctx,n_tokens,created)"
            " VALUES(%s,'parent',%s,%s,%s,%s,%s) RETURNING id",
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

    def commit(self) -> None:
        if getattr(self, "_transaction_depth", 0) == 0:
            self.db.commit()

    def delete_chunk(self, chunk_id: int) -> None:
        identifier = _validate_non_negative_bigint("chunk_id", chunk_id)
        with self.transaction():
            self.db.execute("DELETE FROM holo_grams WHERE id=%s", (identifier,))

    def delete_document(self, document_id: int) -> None:
        identifier = _validate_non_negative_bigint("document_id", document_id)
        with self.transaction():
            self.db.execute("DELETE FROM holo_docs WHERE id=%s", (identifier,))

    # ------------------------------------------------------------ leitura ---

    _COLS = "id,doc_id,parent_id,kind,pos,text,ctx,n_tokens,created,importance,turn_no,accessed_turn"

    def get_chunks(self, ids: Sequence[int]) -> List[dict]:
        ids = self._bounded_ids(ids, "chunk IDs")
        if not ids:
            return []
        rows = self.db.execute(
            f"SELECT {self._COLS} FROM holo_grams WHERE id = ANY(%s)", (list(ids),)
        ).fetchall()
        cols = self._COLS.split(",")
        by_id = {r[0]: dict(zip(cols, r)) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def dense_vecs(self, ids: Sequence[int]) -> Dict[int, np.ndarray]:
        """Vetores densos (eco int8 desquantizado) — para a seleção fermiônica."""
        ids = self._bounded_ids(ids, "chunk IDs")
        if not ids:
            return {}
        rows = self.db.execute(
            "SELECT id, echo FROM holo_grams WHERE id = ANY(%s) AND echo IS NOT NULL", (list(ids),)
        ).fetchall()
        return {int(cid): self.holo.dequantize(bytes(echo)) for cid, echo in rows}

    def dense_vectors(self, ids: Sequence[int]) -> Dict[int, np.ndarray]:
        return self.dense_vecs(ids)

    def touch_access(self, ids: Sequence[int], turn_no: int) -> None:
        ids = self._bounded_ids(ids, "chunk IDs")
        turn = _validate_count("turn_no", turn_no)
        if not ids:
            return
        self.db.execute(
            "UPDATE holo_grams SET accessed_turn=%s WHERE id = ANY(%s)",
            (turn, list(ids)),
        )

    def corpus_tokens(self, kinds: Sequence[str] = ("chunk", "turn")) -> int:
        checked = _validated_kinds(kinds)
        if not checked:
            return 0
        row = self.db.execute(
            "SELECT COALESCE(SUM(n_tokens),0) FROM holo_grams WHERE kind = ANY(%s)",
            (checked,),
        ).fetchone()
        return int(row[0])

    def all_texts(self, kinds: Sequence[str] = ("chunk", "turn")) -> List[dict]:
        checked = _validated_kinds(kinds)
        if not checked:
            return []
        rows = self.db.execute(
            "SELECT id,kind,text,created,turn_no FROM holo_grams WHERE kind = ANY(%s) ORDER BY id",
            (checked,),
        ).fetchall()
        return [dict(zip(["id", "kind", "text", "created", "turn_no"], r)) for r in rows]

    def n_chunks(self) -> int:
        row = self.db.execute(
            f"SELECT COUNT(*) FROM holo_grams WHERE kind IN {_KINDS_SEARCH}"
        ).fetchone()
        return int(row[0])

    def last_turn_no(self) -> int:
        row = self.db.execute(
            "SELECT COALESCE(MAX(turn_no),0) FROM holo_grams WHERE kind='turn'"
        ).fetchone()
        return int(row[0])

    def neighbors(self, doc_id, positions):
        """Chunks vizinhos do mesmo doc em posições dadas (costura Rag3D)."""
        if doc_id is None:
            return []
        document_id = _validate_non_negative_bigint("doc_id", doc_id)
        positions = self._bounded_ids(positions, "positions", allow_negative=True)
        if not positions:
            return []
        rows = self.db.execute(
            "SELECT id, doc_id, kind, pos, text FROM holo_grams WHERE doc_id=%s "
            "AND kind='chunk' AND pos = ANY(%s) ORDER BY pos ASC, id ASC",
            (document_id, list(positions)),
        ).fetchall()
        return [
            {
                "id": int(identifier),
                "doc_id": int(row_document_id),
                "kind": str(kind),
                "pos": int(position),
                "text": str(text),
            }
            for identifier, row_document_id, kind, position, text in rows
        ]

    # ------------------------------------------------- eixo 1: semântico ---

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
        if exact is True:
            raise NotImplementedError("exact dense search is unavailable on postgres-holo")
        k = self._validate_k(k)
        if k == 0:
            return []
        qvec = self._dense_input(qvec)
        if qvec.shape[0] != self._dense_dim:
            raise ValueError(
                f"dense vector dimension mismatch; expected {self._dense_dim}"
            )
        sig = self.holo.sign_dense(qvec)
        bits = self.holo.sig_to_bitstring(sig)
        prefetch = min(
            max(self.PREFETCH_FACTOR * k, self.PREFETCH_MIN),
            DEFAULT_RETRIEVAL_LIMITS.max_pool,
        )
        ham_expr = f"bit_count(sig # %s::bit({HOLO_BITS}))"

        def scan(use_bands: bool):
            where = f"kind IN {_KINDS_SEARCH} AND sig IS NOT NULL"
            params: list = [bits]
            if use_bands:
                # cast explícito: psycopg adapta int pequeno como smallint[],
                # mas a coluna bands é int[] (senão o operador && não casa)
                where += " AND bands && %s::int[]"
                params.append(self.holo.bands_of(sig))
            return self.db.execute(
                f"SELECT id, echo, ({ham_expr}) AS ham FROM holo_grams"
                f" WHERE {where} ORDER BY ham ASC, id ASC LIMIT %s",
                (*params, prefetch),
            ).fetchall()

        big = self.n_chunks() > self.BAND_PREFILTER_THRESHOLD
        rows = scan(use_bands=big)
        # o pré-filtro de facetas (LSH banding) pode descartar vizinhos reais
        # para similaridade moderada; se ele devolver poucos candidatos, cai
        # para o full-scan Hamming (que ainda usa só popcount nativo).
        if big and len(rows) < prefetch // 2:
            rows = scan(use_bands=False)
        if not rows:
            return []

        # re-pontuação exata pelo eco int8 (cosseno real, não Hamming).
        # concatena os bytes de todos os ecos e faz um frombuffer/reshape só,
        # em vez de dequantize+stack por linha (menos alocação).
        q = qvec.astype(np.float32)
        dim = q.shape[0]
        blob = b"".join(bytes(r[1]) for r in rows)
        echoes = np.frombuffer(blob, dtype=np.int8).reshape(len(rows), dim).astype(np.float32) / 127.0
        with np.errstate(all="ignore"):  # flags FPE espúrias do Accelerate/macOS
            sims = echoes @ q
        if not np.isfinite(sims).all():
            raise ValueError("dense scores must be finite")
        # desempate por id (asc) — argsort é instável; JS ordena estável. Mesmo
        # critério nas duas linguagens para o corte top-k coincidir no empate.
        ids = [int(r[0]) for r in rows]
        order = sorted(range(len(rows)), key=lambda i: (-float(sims[i]), ids[i]))[:k]
        return [(ids[i], float(sims[i])) for i in order]

    # --------------------------------------------------- eixo 2: léxico ----

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
        terms = list(qsparse.keys())
        weights = [qsparse[term] for term in terms]
        rows = self.db.execute(
            "WITH query_terms(term,qweight) AS ("
            "  SELECT * FROM unnest(%s::bigint[],%s::double precision[])"
            "), universe AS ("
            "  SELECT GREATEST(COUNT(*),1)::double precision AS n_docs "
            "  FROM holo_grams WHERE kind IN " + _KINDS_SPARSE +
            "), term_df AS ("
            "  SELECT p.term, COUNT(DISTINCT p.gram_id)::double precision AS df "
            "  FROM holo_spectrum p JOIN holo_grams g ON g.id=p.gram_id "
            "  JOIN query_terms q ON q.term=p.term "
            "  WHERE g.kind IN " + _KINDS_SPARSE + " GROUP BY p.term"
            "), scored AS ("
            "  SELECT p.gram_id, SUM(q.qweight::double precision * "
            "    p.weight::double precision * LN(1.0 + "
            "    (u.n_docs - d.df + 0.5) / (d.df + 0.5))) AS score "
            "  FROM holo_spectrum p JOIN holo_grams g ON g.id=p.gram_id "
            "  JOIN query_terms q ON q.term=p.term "
            "  JOIN term_df d ON d.term=p.term CROSS JOIN universe u "
            "  WHERE g.kind IN " + _KINDS_SPARSE + " GROUP BY p.gram_id"
            ") SELECT gram_id,score FROM scored "
            "ORDER BY score DESC,gram_id ASC LIMIT %s",
            (terms, weights, k),
        ).fetchall()
        if any(not math.isfinite(float(score)) for _gram_id, score in rows):
            raise ValueError("sparse scores must be finite")
        return [(int(gram_id), float(score)) for gram_id, score in rows]

    # ----------------------------------------------- eixo 3: estrutural ----

    def colbert_scores(self, qtokens: np.ndarray, candidate_ids: Sequence[int]) -> List[Tuple[int, float]]:
        candidate_ids = self._bounded_ids(candidate_ids)
        qtokens = self._token_input(qtokens)
        if len(candidate_ids) == 0 or qtokens.size == 0:
            return []
        if qtokens.shape[1] != self._colbert_dim:
            raise ValueError(
                f"structural vector dimension mismatch; expected {self._colbert_dim}"
            )
        q_const = self.holo.sign_tokens(qtokens)
        rows = self.db.execute(
            "SELECT id, constellation FROM holo_grams WHERE id = ANY(%s) AND constellation IS NOT NULL",
            (list(candidate_ids),),
        ).fetchall()
        out = [
            (int(gid), _bounded_binary_maxsim(self.holo, q_const, const))
            for gid, const in rows
        ]
        out.sort(key=lambda x: (-x[1], x[0]))  # desempate por id, igual ao JS
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
        candidate_ids = self._bounded_ids(candidate_ids)
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
        self.db.close()

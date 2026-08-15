"""Typed contracts shared by Retrieval Engine V2 backends.

This module deliberately has no database or NumPy dependency.  SQLite and
PostgreSQL adapters can therefore expose the same truthful capability surface
without making optional backends import requirements for local users.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sized
from dataclasses import asdict, dataclass, field, fields
from types import MappingProxyType
from typing import (
    Any,
    ContextManager,
    Generic,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)


@dataclass(frozen=True)
class RetrievalLimits:
    """Hard bounds applied at public retrieval boundaries.

    Keeping these limits in one dependency-free place prevents adapters from
    growing subtly different (or unbounded) candidate pools.
    """

    max_top_k: int = 100
    max_channel_k: int = 1_000
    max_pool: int = 1_000
    # Sparse representations are encoder features, not candidate pools.  BGE
    # passages may approach 8,192 tokens and custom/hash encoders can emit more,
    # so this bound is deliberately independent from ``max_channel_k``.
    max_sparse_terms: int = 8_192
    max_query_expansions: int = 10
    max_filter_values: int = 100
    max_metadata_filters: int = 32
    max_filter_string_length: int = 256
    max_metadata_json_bytes: int = 16 * 1024
    max_query_bytes: int = 64 * 1024
    # Vector/tensor shapes are public allocation boundaries.  They are shared
    # by config, encoder factories, adapters, and the benchmark runner so a
    # custom configuration cannot turn one query into an unbounded allocation.
    max_dense_dim: int = 4_096
    max_structural_dim: int = 1_024
    max_structural_tokens: int = 1_024
    max_structural_values: int = 262_144
    max_diversity_values: int = 262_144
    max_fusion_channels: int = 8
    max_rrf_k: int = 1_000_000


DEFAULT_RETRIEVAL_LIMITS = RetrievalLimits()
_MAX_SIGNED_BIGINT = 2**63 - 1
_MAX_FLOAT32 = float.fromhex("0x1.fffffep+127")


def validate_string_value(
    name: str, value: object, *, non_empty: bool = False
) -> str:
    """Return a string scalar after applying the shared adapter contract."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if non_empty and not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _preflight_json_value(
    value: Any, remaining: list[int], depth: int = 0
) -> None:
    """Walk JSON-shaped data under a strict byte/work budget.

    This happens before ``json.dumps`` so a nested hostile value cannot force
    an unbounded serialization allocation merely to be rejected afterwards.
    The final encoded-size check remains authoritative because escaping can
    expand otherwise bounded strings.
    """

    maximum = DEFAULT_RETRIEVAL_LIMITS.max_metadata_json_bytes

    def consume(amount: int) -> None:
        remaining[0] -= amount
        if remaining[0] < 0:
            raise ValueError("document metadata must not exceed 16 KiB")

    if depth > 32:
        raise ValueError("document metadata must be finite JSON data")
    consume(1)
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        if len(value) > remaining[0]:
            raise ValueError("document metadata must not exceed 16 KiB")
        encoded = value.encode("utf-8")
        consume(len(encoded))
        return
    if isinstance(value, int):
        # ceil(bit_length * log10(2)) is an allocation-free upper bound for
        # decimal digits (the rational coefficient is rounded upward).
        bits = abs(value).bit_length()
        digits = max(1, (bits * 30103 + 99_999) // 100_000)
        consume(digits + (1 if value < 0 else 0))
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("document metadata must be finite JSON data")
        consume(32)
        return
    if isinstance(value, Mapping):
        if len(value) > maximum or len(value) > remaining[0]:
            raise ValueError("document metadata must not exceed 16 KiB")
        for item_count, (key, child) in enumerate(value.items(), start=1):
            if item_count > maximum:
                raise ValueError("document metadata must not exceed 16 KiB")
            if not isinstance(key, str):
                raise ValueError("document metadata must be finite JSON data")
            _preflight_json_value(key, remaining, depth + 1)
            _preflight_json_value(child, remaining, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > maximum or len(value) > remaining[0]:
            raise ValueError("document metadata must not exceed 16 KiB")
        for item_count, child in enumerate(value, start=1):
            if item_count > maximum:
                raise ValueError("document metadata must not exceed 16 KiB")
            _preflight_json_value(child, remaining, depth + 1)
        return
    raise ValueError("document metadata must be finite JSON data")


def serialize_document_metadata(
    metadata: Optional[Mapping[str, Any]],
) -> str:
    """Canonicalize finite document metadata within the public JSON bound."""

    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError("document metadata must be a mapping")
    maximum = DEFAULT_RETRIEVAL_LIMITS.max_metadata_json_bytes
    # Every distinct JSON object entry costs at least one byte.  This generous
    # item ceiling cannot reject a valid document below the byte ceiling, while
    # still bounding a Mapping that lies about len().
    if len(metadata) > maximum:
        raise ValueError("document metadata must not exceed 16 KiB")
    normalized: dict[str, Any] = {}
    remaining = [maximum]
    for item_count, (key, value) in enumerate(metadata.items(), start=1):
        if item_count > maximum:
            raise ValueError("document metadata must not exceed 16 KiB")
        if not isinstance(key, str):
            raise TypeError("document metadata keys must be strings")
        _preflight_json_value(key, remaining)
        _preflight_json_value(value, remaining)
        normalized[key] = value
    try:
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("document metadata must be finite JSON data") from None
    if len(encoded.encode("utf-8")) > maximum:
        raise ValueError("document metadata must not exceed 16 KiB")
    return encoded


def validate_query_text(query: str) -> str:
    """Validate a public query without allocating an unbounded UTF-8 copy."""

    if not isinstance(query, str):
        raise TypeError("query must be a string")
    limit = DEFAULT_RETRIEVAL_LIMITS.max_query_bytes
    # Every Unicode code point occupies at least one UTF-8 byte.  This O(1)
    # preflight rejects huge strings before encoding; the remaining allocation
    # is bounded to at most four times ``limit``.
    if len(query) > limit or len(query.encode("utf-8")) > limit:
        raise ValueError(f"query exceeds maximum of {limit} UTF-8 bytes")
    return query


def normalize_sparse_weights(weights: Mapping[int, float]) -> dict[int, float]:
    """Validate and materialize a bounded sparse representation.

    ``Mapping`` implementations are not necessarily honest about ``len``.
    Check it first to reject ordinary oversized mappings without iteration,
    then enforce the same bound while consuming ``items()``.
    """
    if not isinstance(weights, Mapping):
        raise TypeError("sparse weights must be a mapping")
    limit = DEFAULT_RETRIEVAL_LIMITS.max_sparse_terms
    if len(weights) > limit:
        raise ValueError(f"sparse terms exceed maximum of {limit}")
    normalized: dict[int, float] = {}
    for item_count, (term, weight) in enumerate(weights.items(), start=1):
        if item_count > limit:
            raise ValueError(f"sparse terms exceed maximum of {limit}")
        if isinstance(term, bool) or not isinstance(term, int):
            raise TypeError("sparse term IDs must be integers")
        if term < -(2**63) or term > _MAX_SIGNED_BIGINT:
            raise ValueError("sparse term ID must fit a signed bigint")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError("sparse weights must be real numbers")
        try:
            score = float(weight)
        except (OverflowError, ValueError):
            raise ValueError(
                "sparse weights must be finite PostgreSQL real values"
            ) from None
        if not math.isfinite(score) or abs(score) > _MAX_FLOAT32:
            raise ValueError(
                "sparse weights must be finite PostgreSQL real values"
            )
        normalized[term] = score
    return normalized


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_non_negative_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not bool")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class BackendCapabilities:
    """Features a backend actually implements, never inferred by the pipeline."""

    exact_dense_search: bool = False
    ann_dense_search: bool = False
    sparse_search: bool = False
    structural_rerank: bool = False
    metadata_filters: bool = False
    transactions: bool = False
    native_vector: bool = False
    quantized_vector: bool = False
    cross_language_index: bool = False

    def __post_init__(self) -> None:
        for item in fields(self):
            if not isinstance(getattr(self, item.name), bool):
                raise TypeError(f"{item.name} capability must be bool")


def _normalized_scope_values(name: str, values: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} scope values must be a sequence, not a string")
    maximum = DEFAULT_RETRIEVAL_LIMITS.max_filter_values
    if len(values) > maximum:
        raise ValueError(
            f"{name} scope values exceed the configured maximum of {maximum}"
        )
    normalized = []
    for item_count, value in enumerate(values, start=1):
        if item_count > maximum:
            raise ValueError(
                f"{name} scope values exceed the configured maximum of {maximum}"
            )
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} scope values must be non-empty strings")
        if len(value) > DEFAULT_RETRIEVAL_LIMITS.max_filter_string_length:
            raise ValueError(
                f"{name} scope value length exceeds "
                f"{DEFAULT_RETRIEVAL_LIMITS.max_filter_string_length}"
            )
        normalized.append(value)
    return tuple(normalized)


def _normalized_scope_ids(name: str, values: Sequence[int]) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} scope values must be a sequence of integers")
    maximum = DEFAULT_RETRIEVAL_LIMITS.max_filter_values
    if len(values) > maximum:
        raise ValueError(
            f"{name} scope values exceed the configured maximum of {maximum}"
        )
    normalized = []
    for item_count, value in enumerate(values, start=1):
        if item_count > maximum:
            raise ValueError(
                f"{name} scope values exceed the configured maximum of {maximum}"
            )
        if isinstance(value, bool):
            raise TypeError(f"{name} scope ID must be an integer, not bool")
        if not isinstance(value, int):
            raise TypeError(f"{name} scope ID must be an integer")
        if value < 0:
            raise ValueError(f"{name} scope ID must be non-negative")
        if value > _MAX_SIGNED_BIGINT:
            raise ValueError(f"{name} scope ID must fit a signed bigint")
        normalized.append(value)
    return tuple(normalized)


@dataclass(frozen=True)
class SearchScope:
    """Optional bounded identity scope for a search.

    Each collection is an equality/``IN`` constraint.  There is intentionally
    no free-form SQL expression in this contract.
    """

    kinds: Tuple[str, ...] = ()
    document_ids: Tuple[int, ...] = ()
    parent_ids: Tuple[int, ...] = ()
    sources: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("kinds", "sources"):
            object.__setattr__(self, name, _normalized_scope_values(name, getattr(self, name)))
        for name in ("document_ids", "parent_ids"):
            object.__setattr__(self, name, _normalized_scope_ids(name, getattr(self, name)))
        invalid_kinds = sorted(set(self.kinds) - _RETRIEVABLE_KINDS)
        if invalid_kinds:
            raise ValueError(
                "invalid retrieval kind; expected chunk, summary, or turn"
            )
        count = len(self.kinds) + len(self.document_ids) + len(self.parent_ids) + len(self.sources)
        if count > DEFAULT_RETRIEVAL_LIMITS.max_filter_values:
            raise ValueError(
                "scope values exceed the configured maximum of "
                f"{DEFAULT_RETRIEVAL_LIMITS.max_filter_values}"
            )

    @property
    def value_count(self) -> int:
        return len(self.kinds) + len(self.document_ids) + len(self.parent_ids) + len(self.sources)

    @property
    def is_empty(self) -> bool:
        return self.value_count == 0


FilterScalar = Union[str, int, float, bool, None]
_METADATA_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,62}$")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$")
_RETRIEVABLE_KINDS = frozenset({"chunk", "turn", "summary"})


@dataclass(frozen=True)
class SearchFilters:
    """Bounded, typed equality predicates safe for adapter parameterization."""

    scope: SearchScope = field(default_factory=SearchScope)
    metadata: Mapping[str, FilterScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.scope, SearchScope):
            raise TypeError("scope must be a SearchScope")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata filters must be a mapping")
        if len(self.metadata) > DEFAULT_RETRIEVAL_LIMITS.max_metadata_filters:
            raise ValueError(
                "metadata filters exceed the configured maximum of "
                f"{DEFAULT_RETRIEVAL_LIMITS.max_metadata_filters}"
            )
        normalized = {}
        maximum = DEFAULT_RETRIEVAL_LIMITS.max_metadata_filters
        for item_count, (key, value) in enumerate(self.metadata.items(), start=1):
            if item_count > maximum:
                raise ValueError(
                    "metadata filters exceed the configured maximum of "
                    f"{maximum}"
                )
            if not isinstance(key, str) or not _METADATA_KEY.fullmatch(key):
                raise ValueError(f"invalid metadata key: {key!r}")
            if not isinstance(value, (str, int, float, bool)) and value is not None:
                raise TypeError(f"metadata filter {key!r} must use scalar equality")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"metadata filter {key!r} must be finite")
            if (
                isinstance(value, str)
                and len(value) > DEFAULT_RETRIEVAL_LIMITS.max_filter_string_length
            ):
                raise ValueError(
                    f"metadata filter {key!r} value length exceeds "
                    f"{DEFAULT_RETRIEVAL_LIMITS.max_filter_string_length}"
                )
            normalized[key] = value
        try:
            serialized = json.dumps(
                normalized,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError, OverflowError):
            raise ValueError("metadata filters must be bounded finite JSON") from None
        if (
            len(serialized.encode("utf-8"))
            > DEFAULT_RETRIEVAL_LIMITS.max_metadata_json_bytes
        ):
            raise ValueError("metadata filters must not exceed 16 KiB")
        object.__setattr__(self, "metadata", MappingProxyType(normalized))

    @property
    def is_empty(self) -> bool:
        return self.scope.is_empty and not self.metadata

    @property
    def predicate_count(self) -> int:
        return self.scope.value_count + len(self.metadata)


@dataclass(frozen=True)
class SearchDiagnostics:
    """Safe stage timing/count telemetry.

    Only aggregate measurements and non-sensitive implementation identifiers
    are accepted.  Queries, documents, vectors, DSNs and filter values have no
    field in this type and consequently cannot leak through :meth:`as_dict`.
    """

    normalize_ms: float = 0.0
    expand_ms: float = 0.0
    encode_ms: float = 0.0
    dense_ms: float = 0.0
    sparse_ms: float = 0.0
    union_ms: float = 0.0
    fusion_ms: float = 0.0
    structural_ms: float = 0.0
    rerank_ms: float = 0.0
    diversity_ms: float = 0.0
    hydrate_ms: float = 0.0
    stitch_ms: float = 0.0
    total_retrieval_ms: float = 0.0
    total_ms: float = 0.0

    dense_candidates: int = 0
    sparse_candidates: int = 0
    union_candidates: int = 0
    fused_candidates: int = 0
    structural_candidates: int = 0
    reranked_candidates: int = 0
    final_candidates: int = 0

    backend: str = "unknown"
    pipeline: str = "unknown"
    encoder: str = "unknown"
    fusion: str = "unknown"
    reranker: str = "none"
    diversity: str = "none"
    filters_applied: bool = False
    filter_count: int = 0
    structural_status: str = "skipped"
    structural_reason: str = "not-run"
    reranker_status: str = "skipped"
    reranker_reason: str = "not-run"
    diversity_status: str = "skipped"
    diversity_reason: str = "not-run"
    stitch_status: str = "skipped"
    stitch_reason: str = "not-run"

    _TIMINGS = (
        "normalize_ms",
        "expand_ms",
        "encode_ms",
        "dense_ms",
        "sparse_ms",
        "union_ms",
        "fusion_ms",
        "structural_ms",
        "rerank_ms",
        "diversity_ms",
        "hydrate_ms",
        "stitch_ms",
        "total_retrieval_ms",
        "total_ms",
    )
    _COUNTS = (
        "dense_candidates",
        "sparse_candidates",
        "union_candidates",
        "fused_candidates",
        "structural_candidates",
        "reranked_candidates",
        "final_candidates",
        "filter_count",
    )
    _IDENTIFIERS = ("backend", "pipeline", "encoder", "fusion", "reranker", "diversity")
    _STAGE_STATUSES = frozenset({"applied", "skipped", "fallback"})
    _STAGE_REASONS = frozenset(
        {
            "none",
            "not-run",
            "not-configured",
            "unsupported",
            "unavailable",
            "empty-pool",
            "invalid-output",
            "stage-error",
            "filters-applied",
            "missing-vectors",
        }
    )

    def __post_init__(self) -> None:
        for name in self._TIMINGS:
            _require_non_negative_finite(name, getattr(self, name))
        for name in self._COUNTS:
            _require_non_negative_int(name, getattr(self, name))
        for name in self._IDENTIFIERS:
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or not _SAFE_IDENTIFIER.fullmatch(value)
            ):
                raise ValueError(f"{name} must be a non-empty identifier")
        if not isinstance(self.filters_applied, bool):
            raise TypeError("filters_applied must be bool")
        for name in (
            "structural_status",
            "reranker_status",
            "diversity_status",
            "stitch_status",
        ):
            if getattr(self, name) not in self._STAGE_STATUSES:
                raise ValueError(f"{name} is not an allowed stage status")
        for name in (
            "structural_reason",
            "reranker_reason",
            "diversity_reason",
            "stitch_reason",
        ):
            if getattr(self, name) not in self._STAGE_REASONS:
                raise ValueError(f"{name} is not an allowed stage reason")

    def as_dict(self) -> dict:
        return {
            "timings_ms": {
                name[: -len("_ms")]: float(getattr(self, name)) for name in self._TIMINGS
            },
            "candidate_counts": {
                name[: -len("_candidates")]: getattr(self, name)
                for name in self._COUNTS
                if name.endswith("_candidates")
            },
            "configuration": {
                name: getattr(self, name) for name in self._IDENTIFIERS
            }
            | {
                "filters_applied": self.filters_applied,
                "filter_count": self.filter_count,
            },
            "stage_outcomes": {
                "structural": {
                    "status": self.structural_status,
                    "reason": self.structural_reason,
                },
                "reranker": {
                    "status": self.reranker_status,
                    "reason": self.reranker_reason,
                },
                "diversity": {
                    "status": self.diversity_status,
                    "reason": self.diversity_reason,
                },
                "stitch": {
                    "status": self.stitch_status,
                    "reason": self.stitch_reason,
                },
            },
        }


HitT = TypeVar("HitT")


@dataclass(frozen=True)
class RetrievalStageResult(Generic[HitT]):
    """Shallow-immutable hit container and timing for one bounded stage.

    The outer tuple and dataclass fields cannot be replaced. Arbitrary hit
    objects are deliberately not copied or deep-frozen.
    """

    stage: str
    hits: Tuple[HitT, ...]
    elapsed_ms: float
    input_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("stage must be a non-empty identifier")
        if isinstance(self.hits, (str, bytes)):
            raise TypeError("hits must be a sequence of retrieval hits")
        maximum = DEFAULT_RETRIEVAL_LIMITS.max_pool
        if isinstance(self.hits, Sized) and len(self.hits) > maximum:
            raise ValueError(f"hits exceed maximum of {maximum}")
        bounded_hits = []
        for hit_count, hit in enumerate(self.hits, start=1):
            if hit_count > maximum:
                raise ValueError(f"hits exceed maximum of {maximum}")
            bounded_hits.append(hit)
        object.__setattr__(self, "hits", tuple(bounded_hits))
        _require_non_negative_finite("elapsed_ms", self.elapsed_ms)
        _require_non_negative_int("input_count", self.input_count)

    @property
    def output_count(self) -> int:
        return len(self.hits)


class FingerprintMismatchError(RuntimeError):
    """Raised before an incompatible encoder/index is read or written."""


@dataclass(frozen=True)
class IndexFingerprint:
    """Canonical identity of every index-affecting retrieval choice."""

    backend: str
    encoder: str
    model: str
    revision: str
    dense_dim: int
    structural_dim: int
    max_structural_tokens: int
    structural_projection: str
    query_max_tokens: int
    passage_max_tokens: int
    sparse_version: str
    schema_version: str
    normalization: str
    quantization: str
    chunk_size: int
    overlap: int
    chunking_version: str
    pipeline_version: str

    _TEXT_FIELDS = (
        "backend",
        "encoder",
        "model",
        "revision",
        "structural_projection",
        "sparse_version",
        "schema_version",
        "normalization",
        "quantization",
        "chunking_version",
        "pipeline_version",
    )

    def __post_init__(self) -> None:
        for name in self._TEXT_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "dense_dim",
            "structural_dim",
            "max_structural_tokens",
            "query_max_tokens",
            "passage_max_tokens",
            "chunk_size",
            "overlap",
        ):
            value = getattr(self, name)
            _require_non_negative_int(name, value)
        for name in (
            "dense_dim",
            "structural_dim",
            "max_structural_tokens",
            "query_max_tokens",
            "passage_max_tokens",
            "chunk_size",
        ):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

    def as_dict(self) -> dict:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def differing_fields(self, other: "IndexFingerprint") -> Tuple[str, ...]:
        if not isinstance(other, IndexFingerprint):
            raise TypeError("other fingerprint must be IndexFingerprint")
        return tuple(
            item.name for item in fields(self) if getattr(self, item.name) != getattr(other, item.name)
        )

    def assert_compatible(self, stored: "IndexFingerprint") -> None:
        differences = self.differing_fields(stored)
        if differences:
            # Values are intentionally omitted: configuration fields may be
            # extended in future and error messages must remain secret-safe.
            raise FingerprintMismatchError(
                "incompatible index fingerprint; differing fields: " + ", ".join(differences)
            )


@runtime_checkable
class RetrievalBackend(Protocol):
    """Dependency-inverted storage/retrieval contract for V2 adapters."""

    @property
    def backend_name(self) -> str:
        ...

    @property
    def capabilities(self) -> BackendCapabilities:
        ...

    def transaction(self) -> ContextManager[None]:
        ...

    def get_meta(self, key: str) -> Optional[str]:
        ...

    def set_meta(self, key: str, value: str) -> None:
        ...

    def add_doc(
        self, source: str, title: str, n_tokens: int, meta: Optional[Mapping[str, Any]] = None
    ) -> int:
        ...

    def add_parent(self, doc_id: int, text: str, n_tokens: int, pos: int) -> int:
        ...

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
        ...

    def delete_document(self, document_id: int) -> None:
        ...

    def delete_chunk(self, chunk_id: int) -> None:
        ...

    def get_chunks(self, ids: Sequence[int]) -> Sequence[Mapping[str, Any]]:
        ...

    def dense_search(
        self,
        qvec: Any,
        k: int,
        *,
        filters: Optional[SearchFilters] = None,
        exact: Optional[bool] = None,
    ) -> Sequence[Tuple[int, float]]:
        ...

    def sparse_search(
        self,
        qsparse: Mapping[int, float],
        k: int,
        *,
        filters: Optional[SearchFilters] = None,
    ) -> Sequence[Tuple[int, float]]:
        ...

    def structural_rerank(
        self,
        qtokens: Any,
        candidate_ids: Sequence[int],
        k: int,
        *,
        filters: Optional[SearchFilters] = None,
    ) -> Sequence[Tuple[int, float]]:
        ...

    def dense_vectors(self, ids: Sequence[int]) -> Mapping[int, Any]:
        ...

    def neighbors(self, doc_id: Optional[int], positions: Sequence[int]) -> Sequence[Any]:
        ...

    def health(self) -> Mapping[str, Any]:
        ...

    # Legacy bridge operations remain explicit until all consumers migrate.
    def commit(self) -> None:
        ...

    def colbert_scores(
        self, qtokens: Any, candidate_ids: Sequence[int]
    ) -> Sequence[Tuple[int, float]]:
        ...

    def dense_vecs(self, ids: Sequence[int]) -> Mapping[int, Any]:
        ...

    def touch_access(self, ids: Sequence[int], turn_no: int) -> None:
        ...

    def corpus_tokens(self, kinds: Sequence[str] = ("chunk", "turn")) -> int:
        ...

    def all_texts(self, kinds: Sequence[str] = ("chunk", "turn")) -> Sequence[Mapping[str, Any]]:
        ...

    def n_chunks(self) -> int:
        ...

    def last_turn_no(self) -> int:
        ...

    def close(self) -> None:
        ...

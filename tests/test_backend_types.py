from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from inspect import Parameter, signature
from dataclasses import FrozenInstanceError

import pytest

from rag3d.backend import (
    DEFAULT_RETRIEVAL_LIMITS,
    BackendCapabilities,
    FingerprintMismatchError,
    IndexFingerprint,
    RetrievalBackend,
    RetrievalStageResult,
    SearchDiagnostics,
    SearchFilters,
    SearchScope,
    normalize_sparse_weights,
    serialize_document_metadata,
    validate_query_text,
    validate_string_value,
)


class _OversizedExplodingScopeSequence(Sequence[str]):
    def __len__(self) -> int:
        return DEFAULT_RETRIEVAL_LIMITS.max_filter_values + 1

    def __getitem__(self, index):
        raise AssertionError("oversized scope sequence was indexed")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("oversized scope sequence was iterated")


def test_query_text_limit_is_enforced_in_utf8_bytes() -> None:
    byte_limit = DEFAULT_RETRIEVAL_LIMITS.max_query_bytes

    assert validate_query_text("a" * byte_limit) == "a" * byte_limit
    with pytest.raises(ValueError, match="query.*bytes"):
        validate_query_text("a" * (byte_limit + 1))
    with pytest.raises(ValueError, match="query.*bytes"):
        validate_query_text("😀" * (byte_limit // 4 + 1))
    with pytest.raises(TypeError, match="query must be a string"):
        validate_query_text(b"query")


def test_string_contract_distinguishes_type_and_empty_value_errors() -> None:
    assert validate_string_value("source", "") == ""
    assert validate_string_value("source", "manual", non_empty=True) == "manual"
    with pytest.raises(TypeError, match="source must be a string"):
        validate_string_value("source", 7)
    with pytest.raises(ValueError, match="source must be a non-empty string"):
        validate_string_value("source", "", non_empty=True)


@pytest.mark.parametrize(
    "metadata",
    [
        {"payload": "x" * (16 * 1024 + 1)},
        {"nested": [{"payload": "x" * (16 * 1024 + 1)}]},
    ],
)
def test_document_metadata_rejects_obvious_oversize_before_json_serialization(
    monkeypatch, metadata
) -> None:
    monkeypatch.setattr(
        "rag3d.backend.json.dumps",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("oversized metadata must fail before json.dumps")
        ),
    )

    with pytest.raises(ValueError, match="metadata.*16 KiB"):
        serialize_document_metadata(metadata)


def test_document_metadata_canonicalizes_all_supported_finite_json_shapes() -> None:
    assert serialize_document_metadata(None) == "{}"

    encoded = serialize_document_metadata(
        {
            "none": None,
            "bool": True,
            "positive": 7,
            "negative": -3,
            "float": 1.25,
            "object": {"nested": "ok"},
            "array": [False, ("tuple", 2)],
        }
    )

    assert json.loads(encoded) == {
        "array": [False, ["tuple", 2]],
        "bool": True,
        "float": 1.25,
        "negative": -3,
        "none": None,
        "object": {"nested": "ok"},
        "positive": 7,
    }
    assert encoded == serialize_document_metadata(json.loads(encoded))


def test_document_metadata_rejects_invalid_shapes_before_serialization() -> None:
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        serialize_document_metadata([("key", "value")])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="keys must be strings"):
        serialize_document_metadata({1: "value"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="finite JSON"):
        serialize_document_metadata({"nested": {1: "value"}})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="finite JSON"):
        serialize_document_metadata({"value": math.inf})
    with pytest.raises(ValueError, match="finite JSON"):
        serialize_document_metadata({"value": object()})

    recursive: object = None
    for _ in range(34):
        recursive = [recursive]
    with pytest.raises(ValueError, match="finite JSON"):
        serialize_document_metadata({"nested": recursive})


class _LyingScopeSequence(Sequence[str]):
    def __init__(self) -> None:
        self.yielded = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index):
        if 0 <= index < DEFAULT_RETRIEVAL_LIMITS.max_filter_values + 100:
            return "manual"
        raise IndexError(index)

    def __iter__(self) -> Iterator[str]:
        for _ in range(DEFAULT_RETRIEVAL_LIMITS.max_filter_values + 100):
            self.yielded += 1
            yield "manual"


class _LyingScopeIdSequence(Sequence[int]):
    def __init__(self) -> None:
        self.yielded = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index):
        raise AssertionError("lying scope IDs were indexed")

    def __iter__(self) -> Iterator[int]:
        for value in range(DEFAULT_RETRIEVAL_LIMITS.max_filter_values + 1):
            self.yielded += 1
            yield value


class _LyingMetadataMapping(Mapping[str, int]):
    def __init__(self) -> None:
        self.yielded = 0

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[str]:
        return (f"key_{index}" for index in range(132))

    def __getitem__(self, key: str) -> int:
        return int(key.removeprefix("key_"))

    def items(self):
        for index in range(132):
            self.yielded += 1
            yield f"key_{index}", index


class _OversizedMetadataMapping(Mapping[str, int]):
    def __len__(self) -> int:
        return DEFAULT_RETRIEVAL_LIMITS.max_metadata_json_bytes + 1

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("oversized metadata mapping was iterated")

    def __getitem__(self, key: str) -> int:
        raise AssertionError("oversized metadata mapping was indexed")

    def items(self):
        raise AssertionError("oversized metadata mapping items were read")


class _OversizedMetadataList(list):
    def __len__(self) -> int:
        return DEFAULT_RETRIEVAL_LIMITS.max_metadata_json_bytes + 1

    def __iter__(self):
        raise AssertionError("oversized metadata list was iterated")


class _OversizedExplodingHits(Sequence[dict[str, int]]):
    def __len__(self) -> int:
        return DEFAULT_RETRIEVAL_LIMITS.max_pool + 1

    def __getitem__(self, index):
        raise AssertionError("oversized hits were indexed")

    def __iter__(self) -> Iterator[dict[str, int]]:
        raise AssertionError("oversized hits were iterated")


class _LyingHits(Sequence[dict[str, int]]):
    def __init__(self) -> None:
        self.yielded = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index):
        raise AssertionError("lying hits were indexed")

    def __iter__(self) -> Iterator[dict[str, int]]:
        for index in range(DEFAULT_RETRIEVAL_LIMITS.max_pool + 1):
            self.yielded += 1
            yield {"id": index}
        raise AssertionError("RetrievalStageResult consumed beyond its bound")


def test_document_metadata_preflights_container_and_encoded_byte_limits() -> None:
    with pytest.raises(ValueError, match="metadata.*16 KiB"):
        serialize_document_metadata(_OversizedMetadataMapping())
    with pytest.raises(ValueError, match="metadata.*16 KiB"):
        serialize_document_metadata({"nested": _OversizedMetadataMapping()})
    with pytest.raises(ValueError, match="metadata.*16 KiB"):
        serialize_document_metadata({"nested": _OversizedMetadataList()})

    # Control characters are one UTF-8 byte before JSON escaping and six after
    # it. The authoritative post-serialization byte check must still reject it.
    with pytest.raises(ValueError, match="metadata.*16 KiB"):
        serialize_document_metadata({"escaped": "\x00" * 3_000})

    huge_integer = 1 << (DEFAULT_RETRIEVAL_LIMITS.max_metadata_json_bytes * 4)
    with pytest.raises(ValueError, match="metadata.*16 KiB"):
        serialize_document_metadata({"integer": huge_integer})


def test_backend_capabilities_are_explicit_and_immutable() -> None:
    capabilities = BackendCapabilities(
        exact_dense_search=True,
        sparse_search=True,
        transactions=True,
    )

    assert capabilities.exact_dense_search is True
    assert capabilities.ann_dense_search is False
    assert capabilities.structural_rerank is False
    with pytest.raises(FrozenInstanceError):
        capabilities.native_vector = True  # type: ignore[misc]

    with pytest.raises(TypeError, match="capability"):
        BackendCapabilities(sparse_search=1)  # type: ignore[arg-type]


def test_search_scope_normalizes_sequences_and_enforces_bounds() -> None:
    scope = SearchScope(
        kinds=["chunk", "summary"],  # type: ignore[arg-type]
        document_ids=[1, 2],  # type: ignore[arg-type]
        parent_ids=(3,),
        sources=["manual"],  # type: ignore[arg-type]
    )

    assert scope.kinds == ("chunk", "summary")
    assert scope.document_ids == (1, 2)
    assert scope.parent_ids == (3,)
    assert scope.sources == ("manual",)

    with pytest.raises(ValueError, match="scope values"):
        SearchScope(
            document_ids=tuple(range(DEFAULT_RETRIEVAL_LIMITS.max_filter_values + 1))
        )
    with pytest.raises(TypeError, match="integer"):
        SearchScope(document_ids=("1",))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        SearchScope(parent_ids=(True,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        SearchScope(document_ids=(-1,))
    assert SearchScope(document_ids=(2**63 - 1,)).document_ids == (2**63 - 1,)
    with pytest.raises(ValueError, match="bigint"):
        SearchScope(document_ids=(2**63,))
    with pytest.raises(ValueError, match="bigint"):
        SearchScope(parent_ids=(2**63,))
    with pytest.raises(ValueError, match="kind"):
        SearchScope(kinds=("parent",))
    with pytest.raises(ValueError, match="length"):
        SearchScope(
            sources=("x" * (DEFAULT_RETRIEVAL_LIMITS.max_filter_string_length + 1),)
        )


def test_search_scope_preflights_and_counts_hostile_sequences() -> None:
    with pytest.raises(ValueError, match="scope values.*maximum"):
        SearchScope(sources=_OversizedExplodingScopeSequence())

    values = _LyingScopeSequence()
    with pytest.raises(ValueError, match="scope values.*maximum"):
        SearchScope(sources=values)
    assert values.yielded == DEFAULT_RETRIEVAL_LIMITS.max_filter_values + 1

    ids = _LyingScopeIdSequence()
    with pytest.raises(ValueError, match="scope values.*maximum"):
        SearchScope(document_ids=ids)
    assert ids.yielded == DEFAULT_RETRIEVAL_LIMITS.max_filter_values + 1


def test_search_filters_accept_only_bounded_typed_equality_predicates() -> None:
    filters = SearchFilters(
        scope=SearchScope(document_ids=(1,)),
        metadata={"language": "pt", "published": True, "revision": 3, "rating": 4.5},
    )

    assert dict(filters.metadata) == {
        "language": "pt",
        "published": True,
        "revision": 3,
        "rating": 4.5,
    }
    with pytest.raises(TypeError, match="scalar equality"):
        SearchFilters(metadata={"tags": ["private"]})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="mapping"):
        SearchFilters(metadata=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="metadata key"):
        SearchFilters(metadata={"unsafe key": "x"})
    with pytest.raises(ValueError, match="metadata filters"):
        SearchFilters(
            metadata={
                f"key_{i}": i
                for i in range(DEFAULT_RETRIEVAL_LIMITS.max_metadata_filters + 1)
            }
        )
    with pytest.raises(ValueError, match="length"):
        SearchFilters(
            metadata={
                "source": "x" * (DEFAULT_RETRIEVAL_LIMITS.max_filter_string_length + 1)
            }
        )


def test_search_filters_counts_items_from_a_lying_mapping() -> None:
    metadata = _LyingMetadataMapping()

    with pytest.raises(ValueError, match="metadata filters.*maximum"):
        SearchFilters(metadata=metadata)

    assert metadata.yielded == DEFAULT_RETRIEVAL_LIMITS.max_metadata_filters + 1


def test_search_filters_bound_total_canonical_metadata_json_size() -> None:
    metadata = {f"field_{index}": "🧠" * 256 for index in range(32)}

    with pytest.raises(ValueError, match="metadata.*16 KiB"):
        SearchFilters(metadata=metadata)

    with pytest.raises(ValueError, match="metadata"):
        SearchFilters(metadata={"huge_integer": 10**10_000})


def test_sparse_normalization_converts_huge_integer_overflow_to_value_error() -> None:
    with pytest.raises(ValueError, match="sparse weights"):
        normalize_sparse_weights({1: 10**400})


def test_sparse_normalization_rejects_non_mapping_and_non_numeric_weights() -> None:
    assert normalize_sparse_weights({-1: 0, 2: 1.5}) == {-1: 0.0, 2: 1.5}
    with pytest.raises(TypeError, match="mapping"):
        normalize_sparse_weights([(1, 1.0)])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real numbers"):
        normalize_sparse_weights({1: True})
    with pytest.raises(ValueError, match="PostgreSQL real"):
        normalize_sparse_weights({1: math.inf})


def test_backend_boundary_objects_reject_wrong_container_and_scalar_types() -> None:
    with pytest.raises(TypeError, match="sequence, not a string"):
        SearchScope(kinds="chunk")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        SearchScope(sources=(" ",))
    with pytest.raises(TypeError, match="sequence of integers"):
        SearchScope(document_ids="1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scope values"):
        SearchScope(kinds=("chunk",) * 100, sources=("manual",))
    with pytest.raises(TypeError, match="SearchScope"):
        SearchFilters(scope={})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be finite"):
        SearchFilters(metadata={"score": math.nan})
    with pytest.raises(TypeError, match="filters_applied"):
        SearchDiagnostics(filters_applied=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="hits must be a sequence"):
        RetrievalStageResult(stage="dense", hits="secret", elapsed_ms=0.0)  # type: ignore[arg-type]


def test_numeric_and_fingerprint_boundaries_fail_closed() -> None:
    with pytest.raises(ValueError, match="filter_count must be non-negative"):
        SearchDiagnostics(filter_count=-1)
    with pytest.raises(TypeError, match="total_ms must be a real number"):
        SearchDiagnostics(total_ms="slow")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dense_dim must be positive"):
        _fingerprint(dense_dim=0)
    with pytest.raises(ValueError, match="overlap must be smaller"):
        _fingerprint(overlap=400)
    with pytest.raises(TypeError, match="other fingerprint"):
        _fingerprint().differing_fields({})  # type: ignore[arg-type]


def test_search_diagnostics_are_bounded_numeric_and_secret_free() -> None:
    diagnostics = SearchDiagnostics(
        normalize_ms=0.1,
        encode_ms=2.5,
        dense_ms=1.2,
        sparse_ms=0.8,
        total_retrieval_ms=5.0,
        total_ms=6.0,
        dense_candidates=20,
        sparse_candidates=15,
        union_candidates=28,
        final_candidates=10,
        backend="sqlite",
        pipeline="v2",
        encoder="hash",
        fusion="rrf",
        reranker="none",
        diversity="none",
        filters_applied=True,
        filter_count=2,
    )

    payload = diagnostics.as_dict()
    encoded = json.dumps(payload)
    assert payload["timings_ms"]["encode"] == 2.5
    assert payload["candidate_counts"]["final"] == 10
    assert payload["configuration"]["filter_count"] == 2
    for forbidden in ("query", "text", "vector", "dsn", "filter_values"):
        assert forbidden not in encoded.lower()

    with pytest.raises(TypeError, match="not bool"):
        SearchDiagnostics(final_candidates=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        SearchDiagnostics(total_ms=math.nan)
    with pytest.raises(ValueError, match="non-negative"):
        SearchDiagnostics(dense_ms=-0.1)
    with pytest.raises(ValueError, match="identifier"):
        SearchDiagnostics(backend="postgresql://user:secret@host/db")
    with pytest.raises(ValueError, match="structural_status"):
        SearchDiagnostics(structural_status="secret://unsafe")
    with pytest.raises(ValueError, match="reranker_reason"):
        SearchDiagnostics(reranker_reason="secret://unsafe")


def test_retrieval_stage_result_is_shallow_immutable_and_counted() -> None:
    first_hit = {"id": "a"}
    result = RetrievalStageResult(stage="dense", hits=(first_hit, {"id": "b"}), elapsed_ms=1.0)

    assert result.output_count == 2
    assert result.hits[0]["id"] == "a"
    assert "shallow" in (RetrievalStageResult.__doc__ or "").lower()
    first_hit["id"] = "changed"
    assert result.hits[0]["id"] == "changed"
    with pytest.raises(ValueError, match="non-empty"):
        RetrievalStageResult(stage="", hits=(), elapsed_ms=0.0)


def test_retrieval_stage_result_accepts_exactly_the_bounded_pool() -> None:
    hits = tuple({"id": index} for index in range(DEFAULT_RETRIEVAL_LIMITS.max_pool))

    result = RetrievalStageResult(stage="dense", hits=hits, elapsed_ms=0.0)

    assert result.output_count == DEFAULT_RETRIEVAL_LIMITS.max_pool


def test_retrieval_stage_result_preflights_an_oversized_sized_container() -> None:
    with pytest.raises(ValueError, match="hits exceed maximum"):
        RetrievalStageResult(
            stage="dense",
            hits=_OversizedExplodingHits(),
            elapsed_ms=0.0,
        )


def test_retrieval_stage_result_counts_a_lying_container_while_iterating() -> None:
    hits = _LyingHits()

    with pytest.raises(ValueError, match="hits exceed maximum"):
        RetrievalStageResult(stage="dense", hits=hits, elapsed_ms=0.0)

    assert hits.yielded == DEFAULT_RETRIEVAL_LIMITS.max_pool + 1


def test_retrieval_backend_add_chunk_has_an_explicit_legacy_compatible_signature() -> None:
    parameters = signature(RetrievalBackend.add_chunk).parameters

    assert tuple(parameters) == (
        "self",
        "doc_id",
        "text",
        "ctx",
        "n_tokens",
        "vec",
        "kind",
        "pos",
        "parent_id",
        "importance",
        "turn_no",
    )
    assert all(item.kind is not Parameter.VAR_POSITIONAL for item in parameters.values())
    assert all(item.kind is not Parameter.VAR_KEYWORD for item in parameters.values())


def test_retrieval_backend_search_signatures_preserve_legacy_keyword_names() -> None:
    assert "qvec" in signature(RetrievalBackend.dense_search).parameters
    assert "qsparse" in signature(RetrievalBackend.sparse_search).parameters
    assert "qtokens" in signature(RetrievalBackend.structural_rerank).parameters
    assert "qtokens" in signature(RetrievalBackend.colbert_scores).parameters


def test_retrieval_backend_declares_v2_and_legacy_bridge_operations() -> None:
    required = {
        "backend_name",
        "capabilities",
        "transaction",
        "get_meta",
        "set_meta",
        "add_doc",
        "add_parent",
        "add_chunk",
        "delete_document",
        "delete_chunk",
        "get_chunks",
        "dense_search",
        "sparse_search",
        "structural_rerank",
        "dense_vectors",
        "neighbors",
        "health",
        "commit",
        "colbert_scores",
        "dense_vecs",
        "touch_access",
        "corpus_tokens",
        "all_texts",
        "n_chunks",
        "last_turn_no",
    }

    assert not (required - set(dir(RetrievalBackend)))


def _fingerprint(**overrides: object) -> IndexFingerprint:
    values = {
        "backend": "sqlite",
        "encoder": "hash",
        "model": "rag3d-hash",
        "revision": "1",
        "dense_dim": 1024,
        "structural_dim": 128,
        "max_structural_tokens": 256,
        "structural_projection": "hash-token-ngrams-2-4-v1",
        "query_max_tokens": 256,
        "passage_max_tokens": 256,
        "sparse_version": "crc32-unicode-word-v1",
        "schema_version": "rag3d-trivec-v2",
        "normalization": "l2",
        "quantization": "none",
        "chunk_size": 400,
        "overlap": 60,
        "chunking_version": "adaptive-v1",
        "pipeline_version": "legacy",
    }
    values.update(overrides)
    return IndexFingerprint(**values)  # type: ignore[arg-type]


def test_index_fingerprint_is_canonical_complete_and_deterministic() -> None:
    first = _fingerprint()
    second = _fingerprint()
    payload = json.loads(first.canonical_json())

    assert set(payload) == {
        "backend",
        "encoder",
        "model",
        "revision",
        "dense_dim",
        "structural_dim",
        "max_structural_tokens",
        "structural_projection",
        "query_max_tokens",
        "passage_max_tokens",
        "sparse_version",
        "schema_version",
        "normalization",
        "quantization",
        "chunk_size",
        "overlap",
        "chunking_version",
        "pipeline_version",
    }
    assert first.digest == second.digest
    assert len(first.digest) == 64
    first.assert_compatible(second)


def test_index_fingerprint_mismatch_fails_with_non_secret_diff() -> None:
    current = _fingerprint(backend="pgvector", pipeline_version="v2")
    stored = _fingerprint()

    with pytest.raises(FingerprintMismatchError) as exc_info:
        current.assert_compatible(stored)

    message = str(exc_info.value)
    assert "backend" in message
    assert "pipeline_version" in message
    assert "postgresql://" not in message


@pytest.mark.parametrize(
    "field",
    [
        "dense_dim",
        "structural_dim",
        "max_structural_tokens",
        "query_max_tokens",
        "passage_max_tokens",
        "chunk_size",
    ],
)
def test_index_fingerprint_rejects_bool_as_integer(field: str) -> None:
    with pytest.raises(TypeError, match="not bool"):
        _fingerprint(**{field: True})


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_index_fingerprint_rejects_empty_text_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _fingerprint(**{field: " "})

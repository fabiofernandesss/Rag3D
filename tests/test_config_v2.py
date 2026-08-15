from __future__ import annotations

import warnings
from dataclasses import fields
from pathlib import Path

import pytest

from rag3d.backend import DEFAULT_RETRIEVAL_LIMITS
from rag3d.config import TriRagConfig


_LEGACY_POSITIONAL_FIELDS = (
    "data_dir",
    "pg_dsn",
    "encoder",
    "dense_dim",
    "colbert_dim",
    "max_colbert_tokens",
    "chunk_tokens",
    "chunk_overlap",
    "parent_tokens",
    "tiny_doc_tokens",
    "huge_doc_tokens",
    "contextual_enrich",
    "top_k",
    "channel_k",
    "rerank",
    "rerank_pool",
    "stitch_radius",
    "expand_query",
    "expand_query_max",
    "fusion",
    "rrf_k",
    "channel_weights",
    "interference_strength",
    "coherence_strength",
    "diversity",
    "diversity_pool",
    "memory_budget_tokens",
    "recency_half_life_turns",
    "w_relevance",
    "w_recency",
    "w_importance",
    "summary_every_turns",
    "llm_provider",
    "llm_model",
    "read_mode",
    "max_answer_tokens",
    "remember_chat",
    "small_corpus_tokens",
)


_ENV_KEYS = (
    "RAG3D_BACKEND",
    "RAG3D_RETRIEVAL_PIPELINE",
    "RAG3D_FUSION",
    "RAG3D_STRUCTURAL_RERANK",
    "RAG3D_RERANKER",
    "RAG3D_DIVERSITY_METHOD",
    "RAG3D_ALLOW_ENCODER_FALLBACK",
    "RAG3D_ENCODER",
    "RAG3D_PG",
    "RAG3D_PGVECTOR_SEARCH_MODE",
    "RAG3D_PGVECTOR_STATEMENT_TIMEOUT_MS",
    "TRIRAG_PG",
    "TRIRAG_ENCODER",
    "TRIRAG_FUSION",
)


@pytest.fixture(autouse=True)
def clean_retrieval_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_safe_default_preserves_legacy_sqlite_quantum_behavior() -> None:
    cfg = TriRagConfig()

    assert cfg.backend == "sqlite"
    assert cfg.retrieval_pipeline == "legacy"
    assert cfg.fusion == "quantum"
    assert cfg.allow_encoder_fallback is True
    assert cfg.pg_dsn == ""


def test_legacy_positional_constructor_keeps_original_field_order() -> None:
    cfg = TriRagConfig(Path("/tmp/rag3d-positional"), "", "fallback")

    assert cfg.data_dir == Path("/tmp/rag3d-positional")
    assert cfg.pg_dsn == ""
    assert cfg.encoder == "fallback"
    assert cfg.backend == "sqlite"
    assert tuple(item.name for item in fields(TriRagConfig)[:38]) == _LEGACY_POSITIONAL_FIELDS


def test_v2_defaults_to_rrf_and_disables_silent_encoder_fallback() -> None:
    cfg = TriRagConfig(retrieval_pipeline="v2")

    assert cfg.fusion == "rrf"
    assert cfg.allow_encoder_fallback is False
    assert cfg.structural_rerank is True
    assert cfg.reranker == "none"
    assert cfg.diversity_method == "none"
    assert cfg.pgvector_search_mode == "exact"
    assert cfg.pgvector_statement_timeout_ms == 5_000


def test_pgvector_execution_controls_are_explicit_environment_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG3D_PGVECTOR_SEARCH_MODE", "ANN")
    monkeypatch.setenv("RAG3D_PGVECTOR_STATEMENT_TIMEOUT_MS", "1250")

    cfg = TriRagConfig()

    assert cfg.pgvector_search_mode == "ann"
    assert cfg.pgvector_statement_timeout_ms == 1_250


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pgvector_search_mode": "maybe"}, "pgvector_search_mode"),
        ({"pgvector_statement_timeout_ms": 0}, "pgvector_statement_timeout_ms"),
        ({"pgvector_statement_timeout_ms": 60_001}, "pgvector_statement_timeout_ms"),
    ],
)
def test_pgvector_execution_controls_fail_closed(kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        TriRagConfig(**kwargs)


@pytest.mark.parametrize(
    ("field", "maximum"),
    [
        ("dense_dim", DEFAULT_RETRIEVAL_LIMITS.max_dense_dim),
        ("colbert_dim", DEFAULT_RETRIEVAL_LIMITS.max_structural_dim),
        (
            "max_colbert_tokens",
            DEFAULT_RETRIEVAL_LIMITS.max_structural_tokens,
        ),
    ],
)
def test_encoder_shape_limits_accept_boundary_and_reject_oversize(
    field: str, maximum: int
) -> None:
    assert getattr(TriRagConfig(**{field: maximum}), field) == maximum

    with pytest.raises(ValueError, match=field):
        TriRagConfig(**{field: maximum + 1})


def test_structural_tensor_product_is_bounded_before_encoder_allocation() -> None:
    limit = DEFAULT_RETRIEVAL_LIMITS.max_structural_values
    assert (
        TriRagConfig(colbert_dim=1_024, max_colbert_tokens=limit // 1_024)
        .max_colbert_tokens
        == limit // 1_024
    )

    with pytest.raises(ValueError, match="structural.*tensor"):
        TriRagConfig(colbert_dim=1_024, max_colbert_tokens=limit // 1_024 + 1)


def test_resolve_recomputes_only_implicit_values_after_public_mutation() -> None:
    cfg = TriRagConfig()
    cfg.retrieval_pipeline = "v2"
    cfg.pg_dsn = "postgresql://user:secret@localhost/rag3d"
    cfg.rerank = True

    resolved = cfg.resolve()

    assert resolved is cfg
    assert cfg.backend == "postgres-holo"
    assert cfg.fusion == "rrf"
    assert cfg.allow_encoder_fallback is False
    assert cfg.reranker == "llm"

    # Assignments made internally by resolve must remain implicit.
    cfg.retrieval_pipeline = "legacy"
    cfg.pg_dsn = ""
    cfg.rerank = False
    cfg.resolve()
    assert cfg.backend == "sqlite"
    assert cfg.fusion == "quantum"
    assert cfg.allow_encoder_fallback is True
    assert cfg.reranker == "none"


def test_resolve_preserves_explicit_constructor_values() -> None:
    cfg = TriRagConfig(
        backend="sqlite",
        retrieval_pipeline="legacy",
        fusion="quantum",
        allow_encoder_fallback=True,
        reranker="cross-encoder",
    )
    cfg.retrieval_pipeline = "v2"
    cfg.pg_dsn = "postgresql://localhost/rag3d"
    cfg.rerank = True

    cfg.resolve()

    assert cfg.backend == "sqlite"
    assert cfg.fusion == "quantum"
    assert cfg.allow_encoder_fallback is True
    assert cfg.reranker == "cross-encoder"


def test_resolve_preserves_explicit_environment_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG3D_BACKEND", "sqlite")
    monkeypatch.setenv("RAG3D_FUSION", "quantum")
    monkeypatch.setenv("RAG3D_ALLOW_ENCODER_FALLBACK", "true")
    monkeypatch.setenv("RAG3D_RERANKER", "cross-encoder")
    cfg = TriRagConfig()
    cfg.retrieval_pipeline = "v2"
    cfg.pg_dsn = "postgresql://localhost/rag3d"
    cfg.rerank = True

    cfg.resolve()

    assert cfg.backend == "sqlite"
    assert cfg.fusion == "quantum"
    assert cfg.allow_encoder_fallback is True
    assert cfg.reranker == "cross-encoder"


def test_resolve_recognizes_post_init_override_as_explicit() -> None:
    cfg = TriRagConfig()
    cfg.fusion = "rrf"
    cfg.resolve()
    cfg.retrieval_pipeline = "v2"
    cfg.resolve()
    cfg.retrieval_pipeline = "legacy"
    cfg.resolve()

    assert cfg.fusion == "rrf"


def test_resolve_tracks_explicit_assignment_even_when_equal_to_derived_value() -> None:
    cfg = TriRagConfig()
    cfg.backend = "sqlite"
    cfg.fusion = "quantum"
    cfg.allow_encoder_fallback = True
    cfg.reranker = "none"
    cfg.retrieval_pipeline = "v2"
    cfg.pg_dsn = "postgresql://localhost/rag3d"
    cfg.rerank = True

    cfg.resolve()

    assert cfg.backend == "sqlite"
    assert cfg.fusion == "quantum"
    assert cfg.allow_encoder_fallback is True
    assert cfg.reranker == "none"


def test_resolve_sentinels_restore_pipeline_derived_defaults() -> None:
    cfg = TriRagConfig(
        backend="pgvector",
        fusion="quantum",
        allow_encoder_fallback=True,
        reranker="cross-encoder",
    )
    cfg.backend = ""
    cfg.fusion = ""
    cfg.allow_encoder_fallback = None
    cfg.reranker = ""
    cfg.retrieval_pipeline = "v2"
    cfg.pg_dsn = "postgresql://localhost/rag3d"
    cfg.rerank = True

    cfg.resolve()

    assert cfg.backend == "postgres-holo"
    assert cfg.fusion == "rrf"
    assert cfg.allow_encoder_fallback is False
    assert cfg.reranker == "llm"


def test_failed_resolve_never_leaves_internal_resolution_guard_set() -> None:
    cfg = TriRagConfig()
    cfg.backend = "invalid"

    with pytest.raises(ValueError, match="backend"):
        cfg.resolve()

    assert cfg._resolving is False
    cfg.backend = ""
    assert cfg.resolve().backend == "sqlite"


def test_explicit_new_configuration_wins_over_legacy_pg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG3D_BACKEND", "sqlite")
    monkeypatch.setenv("RAG3D_PG", "postgresql://user:password@db/private")
    monkeypatch.setenv("RAG3D_RETRIEVAL_PIPELINE", "v2")
    monkeypatch.setenv("RAG3D_FUSION", "quantum")
    monkeypatch.setenv("RAG3D_ALLOW_ENCODER_FALLBACK", "true")

    cfg = TriRagConfig()

    assert cfg.backend == "sqlite"
    assert cfg.retrieval_pipeline == "v2"
    assert cfg.fusion == "quantum"
    assert cfg.allow_encoder_fallback is True
    assert "password" not in repr(cfg)


def test_config_repr_does_not_expose_the_data_directory_path() -> None:
    secret_path = Path("/private/tenant-secret/rag3d")

    cfg = TriRagConfig(data_dir=secret_path)

    assert str(secret_path) not in repr(cfg)


@pytest.mark.parametrize(
    "field",
    [
        "backend",
        "retrieval_pipeline",
        "fusion",
        "reranker",
        "diversity_method",
        "encoder",
        "pgvector_search_mode",
    ],
)
def test_invalid_configuration_choices_never_echo_the_supplied_value(field):
    secret = "postgresql://admin:do-not-print@db/private"

    with pytest.raises(ValueError) as captured:
        TriRagConfig(**{field: secret})

    assert secret not in str(captured.value)
    assert "do-not-print" not in str(captured.value)


@pytest.mark.parametrize("dsn_var", ["RAG3D_PG", "TRIRAG_PG"])
def test_legacy_pg_configuration_selects_holographic_backend(
    monkeypatch: pytest.MonkeyPatch, dsn_var: str
) -> None:
    monkeypatch.setenv(dsn_var, "postgresql://localhost/rag3d")

    cfg = TriRagConfig()

    assert cfg.backend == "postgres-holo"
    assert cfg.pg_dsn == "postgresql://localhost/rag3d"


def test_trirag_fallback_warns_without_exposing_its_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_dsn = "postgresql://user:do-not-print@localhost/private"
    monkeypatch.setenv("TRIRAG_PG", secret_dsn)

    with pytest.warns(DeprecationWarning) as warning_records:
        cfg = TriRagConfig()

    messages = [str(item.message) for item in warning_records]
    assert cfg.pg_dsn == secret_dsn
    assert any("TRIRAG_PG" in message for message in messages)
    assert all(secret_dsn not in message and "do-not-print" not in message for message in messages)
    assert all(item.filename != "<string>" for item in warning_records)


def test_rag3d_environment_wins_without_warning_for_unused_trirag_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG3D_PG", "postgresql://localhost/current")
    monkeypatch.setenv("TRIRAG_PG", "postgresql://user:old-secret@localhost/old")

    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        cfg = TriRagConfig()

    assert cfg.pg_dsn == "postgresql://localhost/current"
    assert warning_records == []


def test_explicit_empty_rag3d_value_blocks_legacy_fallback_without_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG3D_PG", "")
    monkeypatch.setenv("TRIRAG_PG", "postgresql://user:old-secret@localhost/old")

    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        cfg = TriRagConfig()

    assert cfg.pg_dsn == ""
    assert cfg.backend == "sqlite"
    assert warning_records == []


def test_new_reranker_setting_wins_and_old_boolean_remains_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert TriRagConfig(rerank=True).reranker == "llm"

    monkeypatch.setenv("RAG3D_RERANKER", "cross-encoder")
    cfg = TriRagConfig(rerank=True)
    assert cfg.reranker == "cross-encoder"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"backend": "qdrant"}, "backend"),
        ({"retrieval_pipeline": "v3"}, "retrieval_pipeline"),
        ({"fusion": "sum"}, "fusion"),
        ({"reranker": "magic"}, "reranker"),
        ({"diversity_method": "random"}, "diversity_method"),
        ({"encoder": "unknown"}, "encoder"),
    ],
)
def test_invalid_enums_fail_explicitly(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TriRagConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    [
        "backend",
        "retrieval_pipeline",
        "fusion",
        "reranker",
        "diversity_method",
        "encoder",
    ],
)
def test_non_string_enums_fail_intentionally_never_with_attribute_error(field: str) -> None:
    with pytest.raises((TypeError, ValueError), match=field) as exc_info:
        TriRagConfig(**{field: 1})  # type: ignore[arg-type]

    assert not isinstance(exc_info.value, AttributeError)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"top_k": True}, "not bool"),
        ({"top_k": 0}, "top_k"),
        ({"top_k": 101}, "top_k"),
        ({"channel_k": 1001}, "channel_k"),
        ({"rerank_pool": 1001}, "rerank_pool"),
        ({"diversity_pool": 1001}, "diversity_pool"),
        ({"expand_query_max": 11}, "expand_query_max"),
        ({"structural_rerank": 1}, "structural_rerank"),
        ({"allow_encoder_fallback": 1}, "allow_encoder_fallback"),
        ({"top_k": 20, "channel_k": 10}, "top_k cannot exceed channel_k"),
        ({"chunk_overlap": -1}, "chunk_overlap"),
        ({"chunk_tokens": 100, "chunk_overlap": 100}, "chunk_overlap"),
        ({"rrf_k": True}, "not bool"),
        ({"rrf_k": 0}, "rrf_k"),
        ({"rrf_k": DEFAULT_RETRIEVAL_LIMITS.max_rrf_k + 1}, "rrf_k"),
    ],
)
def test_retrieval_limits_are_enforced(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        TriRagConfig(**kwargs)  # type: ignore[arg-type]


def test_invalid_boolean_environment_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG3D_ALLOW_ENCODER_FALLBACK", "sometimes")

    with pytest.raises(ValueError, match="RAG3D_ALLOW_ENCODER_FALLBACK"):
        TriRagConfig()


def test_boolean_environment_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG3D_STRUCTURAL_RERANK", "OFF")
    monkeypatch.setenv("RAG3D_ALLOW_ENCODER_FALLBACK", "YeS")

    cfg = TriRagConfig(retrieval_pipeline="v2")

    assert cfg.structural_rerank is False
    assert cfg.allow_encoder_fallback is True

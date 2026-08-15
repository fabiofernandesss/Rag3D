from __future__ import annotations

import sys
import types

import pytest

import rag3d.encoders as encoders
from rag3d.backend import DEFAULT_RETRIEVAL_LIMITS


@pytest.mark.parametrize("kind", ["fallback", "hash"])
def test_explicit_hash_encoder_never_requires_fallback_permission(kind: str) -> None:
    encoder = encoders.make_encoder(kind, 64, 16, 8, allow_fallback=False)

    assert isinstance(encoder, encoders.HashEncoder)
    assert encoder.dense_dim == 64


def test_auto_encoder_falls_back_only_when_explicitly_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingBge:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("FlagEmbedding is unavailable")

    monkeypatch.setattr(encoders, "Bgem3Encoder", MissingBge)

    encoder = encoders.make_encoder("auto", 64, 16, 8, allow_fallback=True)
    assert isinstance(encoder, encoders.HashEncoder)

    with pytest.raises(ImportError, match="FlagEmbedding"):
        encoders.make_encoder("auto", 64, 16, 8, allow_fallback=False)


def test_explicit_bge_error_is_never_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenBge:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("model revision cannot be loaded")

    monkeypatch.setattr(encoders, "Bgem3Encoder", BrokenBge)

    with pytest.raises(RuntimeError, match="model revision"):
        encoders.make_encoder("bge-m3", 1024, 128, 256, allow_fallback=True)


def test_invalid_encoder_name_fails_instead_of_silently_hashing() -> None:
    with pytest.raises(ValueError, match="encoder"):
        encoders.make_encoder("typo", 64, 16, 8)


def test_invalid_encoder_factory_value_is_not_echoed_in_the_error() -> None:
    secret = "postgresql://admin:do-not-print@db/private"

    with pytest.raises(ValueError) as captured:
        encoders.make_encoder(secret, 64, 16, 8)

    assert secret not in str(captured.value)
    assert "do-not-print" not in str(captured.value)


def test_explicit_bge_requires_its_real_dense_dimension() -> None:
    with pytest.raises(ValueError, match="1024"):
        encoders.make_encoder("bge-m3", 768, 128, 256)


def test_auto_bge_dimension_mismatch_obeys_fallback_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AvailableBge:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(encoders, "Bgem3Encoder", AvailableBge)

    fallback = encoders.make_encoder("auto", 768, 128, 256, allow_fallback=True)
    assert isinstance(fallback, encoders.HashEncoder)
    with pytest.raises(ValueError, match="1024"):
        encoders.make_encoder("auto", 768, 128, 256, allow_fallback=False)


@pytest.mark.parametrize("value", [True, 0, -1])
def test_encoder_dimensions_reject_bool_and_non_positive_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        encoders.make_encoder("hash", value, 16, 8)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("argument", "maximum"),
    [
        ("dense_dim", DEFAULT_RETRIEVAL_LIMITS.max_dense_dim),
        ("colbert_dim", DEFAULT_RETRIEVAL_LIMITS.max_structural_dim),
        ("max_tokens", DEFAULT_RETRIEVAL_LIMITS.max_structural_tokens),
    ],
)
def test_encoder_factory_rejects_unbounded_shapes(
    argument: str, maximum: int
) -> None:
    values = {
        "dense_dim": 64,
        "colbert_dim": 16,
        "max_tokens": 8,
    }
    values[argument] = maximum + 1

    with pytest.raises(ValueError, match=argument):
        encoders.make_encoder("hash", **values)


def test_encoder_factory_accepts_shared_shape_boundaries() -> None:
    encoder = encoders.make_encoder(
        "hash",
        dense_dim=DEFAULT_RETRIEVAL_LIMITS.max_dense_dim,
        colbert_dim=DEFAULT_RETRIEVAL_LIMITS.max_structural_dim,
        max_tokens=(
            DEFAULT_RETRIEVAL_LIMITS.max_structural_values
            // DEFAULT_RETRIEVAL_LIMITS.max_structural_dim
        ),
    )

    assert encoder.dense_dim == DEFAULT_RETRIEVAL_LIMITS.max_dense_dim
    assert encoder.colbert_dim == DEFAULT_RETRIEVAL_LIMITS.max_structural_dim


def test_encoder_factory_rejects_oversized_structural_tensor_product() -> None:
    limit = DEFAULT_RETRIEVAL_LIMITS.max_structural_values

    with pytest.raises(ValueError, match="structural.*tensor"):
        encoders.make_encoder(
            "hash",
            dense_dim=64,
            colbert_dim=1_024,
            max_tokens=limit // 1_024 + 1,
        )


def test_bge_index_spec_pins_model_revision_loader_and_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = {}
    fake_module = types.ModuleType("FlagEmbedding")
    fake_hub = types.ModuleType("huggingface_hub")

    def fake_snapshot_download(**kwargs: object) -> str:
        received["snapshot"] = kwargs
        return "/verified-cache/bge-m3-immutable-snapshot"

    class FakeModel:
        def __init__(self, model: str, **kwargs: object) -> None:
            received["model"] = model
            received["model_kwargs"] = kwargs

    fake_module.BGEM3FlagModel = FakeModel  # type: ignore[attr-defined]
    fake_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    encoder = encoders.Bgem3Encoder(colbert_dim=16, max_tokens=19)
    spec = encoder.index_spec

    assert received == {
        "snapshot": {
            "repo_id": "BAAI/bge-m3",
            "revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "ignore_patterns": ["onnx/*", "imgs/*", "*.md", "*.DS_Store"],
        },
        "model": "/verified-cache/bge-m3-immutable-snapshot",
        "model_kwargs": {"use_fp16": False},
    }
    assert spec.model == "BAAI/bge-m3"
    assert spec.revision == "5617a9f61b028005a4858fdac845db406aefb181"
    assert spec.max_structural_tokens == 19
    assert spec.query_max_tokens == 256
    assert spec.passage_max_tokens == 1024
    prefix = "gaussian-jl-pcg64-seed42-sha256:"
    assert spec.structural_projection.startswith(prefix)
    assert len(spec.structural_projection.removeprefix(prefix)) == 64


def test_hash_index_spec_makes_cross_language_algorithm_identity_explicit() -> None:
    spec = encoders.HashEncoder(dense_dim=64, colbert_dim=16, max_tokens=23).index_spec

    assert spec.model == "rag3d/hash"
    assert spec.revision == "crc32-char-ngram-v1"
    assert spec.max_structural_tokens == 23
    assert spec.structural_projection == "hash-token-ngrams-2-4-v1"
    assert spec.query_max_tokens == 23
    assert spec.passage_max_tokens == 23
    assert spec.sparse_version == "crc32-unicode-word-v1"
    assert spec.schema_version == "rag3d-trivec-v2"

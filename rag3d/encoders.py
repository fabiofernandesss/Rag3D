"""Encoders tridimensionais.

Todo texto vira TRÊS representações de uma vez:
  dense   -> vetor denso (significado)
  sparse  -> pesos por termo (léxico exato)
  tokens  -> matriz token a token (estrutura fina, late-interaction/MaxSim)

Backend preferido: BGE-M3 (BAAI), que produz as três nativamente e cobre
100+ línguas. Fallback: encoder por hashing de n-gramas de caracteres —
qualidade menor, mas zero dependências e agnóstico de escrita/língua.
"""
from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .backend import DEFAULT_RETRIEVAL_LIMITS
from .textproc import char_ngrams, word_tokens


@dataclass
class TriVec:
    """As três projeções de um mesmo texto."""
    dense: np.ndarray            # (D,) float32, L2-normalizado
    sparse: Dict[int, float]     # id do termo -> peso
    tokens: np.ndarray           # (T, C) float32, linhas L2-normalizadas


@dataclass(frozen=True)
class EncoderIndexSpec:
    """Immutable identity of every encoder choice that changes stored vectors."""

    model: str
    revision: str
    max_structural_tokens: int
    structural_projection: str
    query_max_tokens: int
    passage_max_tokens: int
    sparse_version: str
    schema_version: str


class BaseEncoder:
    name = "base"
    index_spec: EncoderIndexSpec

    def encode(self, texts: List[str], is_query: bool = False) -> List[TriVec]:
        raise NotImplementedError


def _bounded_dimension(name: str, value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")


def _validate_structural_shape(colbert_dim: int, max_tokens: int) -> None:
    if (
        colbert_dim * max_tokens
        > DEFAULT_RETRIEVAL_LIMITS.max_structural_values
    ):
        raise ValueError("structural tensor exceeds the public value-count bound")


# ---------------------------------------------------------------- BGE-M3 ---

class Bgem3Encoder(BaseEncoder):
    """BGE-M3 via FlagEmbedding: denso(1024) + esparso + ColBERT em uma passada."""

    name = "bge-m3"

    MODEL = "BAAI/bge-m3"
    REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
    QUERY_MAX_TOKENS = 256
    PASSAGE_MAX_TOKENS = 1024

    def __init__(self, colbert_dim: int = 128, max_tokens: int = 256):
        limits = DEFAULT_RETRIEVAL_LIMITS
        _bounded_dimension(
            "colbert_dim", colbert_dim, limits.max_structural_dim
        )
        _bounded_dimension(
            "max_tokens", max_tokens, limits.max_structural_tokens
        )
        _validate_structural_shape(colbert_dim, max_tokens)
        from FlagEmbedding import BGEM3FlagModel  # import tardio: dependência opcional
        from huggingface_hub import snapshot_download

        # FlagEmbedding currently accepts arbitrary kwargs but does not
        # reliably forward ``revision`` to every tokenizer/model loader. Resolve
        # the immutable Hugging Face snapshot ourselves and give the encoder a
        # local path, so the revision recorded in the fingerprint is truthful.
        try:
            model_path = snapshot_download(
                repo_id=self.MODEL,
                revision=self.REVISION,
                ignore_patterns=["onnx/*", "imgs/*", "*.md", "*.DS_Store"],
            )
            self.model = BGEM3FlagModel(model_path, use_fp16=False)
        except Exception as exc:
            raise RuntimeError(
                "failed to load pinned BGE-M3 snapshot "
                f"({type(exc).__name__})"
            ) from None
        self.colbert_dim = colbert_dim
        self.max_tokens = max_tokens
        # projeção aleatória fixa 1024 -> colbert_dim (Johnson-Lindenstrauss preserva MaxSim)
        rng = np.random.default_rng(42)
        self._proj = (rng.standard_normal((1024, colbert_dim)) / np.sqrt(colbert_dim)).astype(np.float32)
        projection_digest = hashlib.sha256(
            self._proj.astype("<f4", copy=False).tobytes(order="C")
        ).hexdigest()
        self.index_spec = EncoderIndexSpec(
            model=self.MODEL,
            revision=self.REVISION,
            max_structural_tokens=max_tokens,
            structural_projection=(
                "gaussian-jl-pcg64-seed42-sha256:" + projection_digest
            ),
            query_max_tokens=self.QUERY_MAX_TOKENS,
            passage_max_tokens=self.PASSAGE_MAX_TOKENS,
            sparse_version="bge-m3-lexical-weights-v1",
            schema_version="rag3d-trivec-v2",
        )

    def encode(self, texts: List[str], is_query: bool = False) -> List[TriVec]:
        # max_length no tamanho real do conteúdo: o padrão 8192 deixa a
        # codificação em CPU muito mais lenta sem ganho (chunks têm ~400 tokens)
        out = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
            max_length=(
                self.QUERY_MAX_TOKENS if is_query else self.PASSAGE_MAX_TOKENS
            ),
        )
        res: List[TriVec] = []
        for i in range(len(texts)):
            dense = np.asarray(out["dense_vecs"][i], dtype=np.float32)
            dense /= (np.linalg.norm(dense) + 1e-9)
            sparse = {int(k): float(v) for k, v in out["lexical_weights"][i].items()}
            col = np.asarray(out["colbert_vecs"][i], dtype=np.float32)[: self.max_tokens]
            col = col @ self._proj
            col /= (np.linalg.norm(col, axis=1, keepdims=True) + 1e-9)
            res.append(TriVec(dense=dense, sparse=sparse, tokens=col))
        return res


# -------------------------------------------------------------- Fallback ---

def _stable_hash(s: str) -> int:
    """Hash estável entre processos (hash() do Python é randomizado)."""
    return zlib.crc32(s.encode("utf-8"))


class HashEncoder(BaseEncoder):
    """Encoder por hashing — zero dependências, qualquer língua/escrita.

    dense : hashing trick sobre n-gramas de caracteres (funciona até em CJK
            e árabe porque não depende de separação de palavras)
    sparse: palavras Unicode (CJK vira bigramas) com peso 1+log(tf)
    tokens: um vetor hasheado por palavra, para MaxSim aproximado
    """

    name = "hash"

    def __init__(self, dense_dim: int = 1024, colbert_dim: int = 128, max_tokens: int = 256):
        limits = DEFAULT_RETRIEVAL_LIMITS
        _bounded_dimension("dense_dim", dense_dim, limits.max_dense_dim)
        _bounded_dimension(
            "colbert_dim", colbert_dim, limits.max_structural_dim
        )
        _bounded_dimension(
            "max_tokens", max_tokens, limits.max_structural_tokens
        )
        _validate_structural_shape(colbert_dim, max_tokens)
        self.dense_dim = dense_dim
        self.colbert_dim = colbert_dim
        self.max_tokens = max_tokens
        self.index_spec = EncoderIndexSpec(
            model="rag3d/hash",
            revision="crc32-char-ngram-v1",
            max_structural_tokens=max_tokens,
            structural_projection="hash-token-ngrams-2-4-v1",
            query_max_tokens=max_tokens,
            passage_max_tokens=max_tokens,
            sparse_version="crc32-unicode-word-v1",
            schema_version="rag3d-trivec-v2",
        )

    def _dense(self, text: str) -> np.ndarray:
        v = np.zeros(self.dense_dim, dtype=np.float32)
        for g in char_ngrams(text):
            h = _stable_hash(g)
            v[h % self.dense_dim] += 1.0 if (h >> 31) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _sparse(self, text: str) -> Dict[int, float]:
        tf: Dict[int, int] = {}
        for w in word_tokens(text):
            tid = _stable_hash("w:" + w)
            tf[tid] = tf.get(tid, 0) + 1
        return {t: 1.0 + float(np.log(c)) for t, c in tf.items()}

    def _token_vec(self, word: str) -> np.ndarray:
        v = np.zeros(self.colbert_dim, dtype=np.float32)
        for g in char_ngrams(word, 2, 4):
            h = _stable_hash("t:" + g)
            v[h % self.colbert_dim] += 1.0 if (h >> 31) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode(self, texts: List[str], is_query: bool = False) -> List[TriVec]:
        res: List[TriVec] = []
        for t in texts:
            words = word_tokens(t)[: self.max_tokens]
            toks = (
                np.stack([self._token_vec(w) for w in words])
                if words
                else np.zeros((1, self.colbert_dim), dtype=np.float32)
            )
            res.append(TriVec(dense=self._dense(t), sparse=self._sparse(t), tokens=toks))
        return res


# --------------------------------------------------------------- factory ---


def make_encoder(
    kind: str,
    dense_dim: int,
    colbert_dim: int,
    max_tokens: int,
    allow_fallback: bool = True,
) -> BaseEncoder:
    """Create an encoder with an explicit fallback policy.

    ``allow_fallback=True`` remains the API default for legacy callers.  The V2
    pipeline passes its fail-closed configuration explicitly.  Asking for the
    hash/fallback encoder is never considered a fallback and therefore does not
    require this permission.
    """
    limits = DEFAULT_RETRIEVAL_LIMITS
    for name, value, maximum in (
        ("dense_dim", dense_dim, limits.max_dense_dim),
        ("colbert_dim", colbert_dim, limits.max_structural_dim),
        ("max_tokens", max_tokens, limits.max_structural_tokens),
    ):
        _bounded_dimension(name, value, maximum)
    _validate_structural_shape(colbert_dim, max_tokens)
    if not isinstance(allow_fallback, bool):
        raise TypeError("allow_fallback must be bool")
    if not isinstance(kind, str):
        raise TypeError("encoder kind must be a string")

    normalized = kind.strip().lower()
    if normalized in ("fallback", "hash"):
        return HashEncoder(
            dense_dim=dense_dim, colbert_dim=colbert_dim, max_tokens=max_tokens
        )
    if normalized not in ("auto", "bge-m3"):
        raise ValueError(
            "invalid encoder; expected auto, bge-m3, fallback, or hash"
        )
    if normalized == "bge-m3" and dense_dim != 1024:
        raise ValueError("bge-m3 requires dense_dim=1024")

    try:
        encoder = Bgem3Encoder(colbert_dim=colbert_dim, max_tokens=max_tokens)
        if dense_dim != 1024:
            raise ValueError("bge-m3 requires dense_dim=1024")
        return encoder
    except Exception:
        # Explicit BGE is a strict request.  Auto is strict too when V2's
        # production-safe fallback flag is disabled.
        if normalized == "bge-m3" or not allow_fallback:
            raise
        return HashEncoder(
            dense_dim=dense_dim, colbert_dim=colbert_dim, max_tokens=max_tokens
        )

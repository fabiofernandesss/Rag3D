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

import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .textproc import char_ngrams, word_tokens


@dataclass
class TriVec:
    """As três projeções de um mesmo texto."""
    dense: np.ndarray            # (D,) float32, L2-normalizado
    sparse: Dict[int, float]     # id do termo -> peso
    tokens: np.ndarray           # (T, C) float32, linhas L2-normalizadas


class BaseEncoder:
    name = "base"

    def encode(self, texts: List[str], is_query: bool = False) -> List[TriVec]:
        raise NotImplementedError


# ---------------------------------------------------------------- BGE-M3 ---

class Bgem3Encoder(BaseEncoder):
    """BGE-M3 via FlagEmbedding: denso(1024) + esparso + ColBERT em uma passada."""

    name = "bge-m3"

    def __init__(self, colbert_dim: int = 128, max_tokens: int = 256):
        from FlagEmbedding import BGEM3FlagModel  # import tardio: dependência opcional

        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
        self.colbert_dim = colbert_dim
        self.max_tokens = max_tokens
        # projeção aleatória fixa 1024 -> colbert_dim (Johnson-Lindenstrauss preserva MaxSim)
        rng = np.random.default_rng(42)
        self._proj = (rng.standard_normal((1024, colbert_dim)) / np.sqrt(colbert_dim)).astype(np.float32)

    def encode(self, texts: List[str], is_query: bool = False) -> List[TriVec]:
        # max_length no tamanho real do conteúdo: o padrão 8192 deixa a
        # codificação em CPU muito mais lenta sem ganho (chunks têm ~400 tokens)
        out = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
            max_length=256 if is_query else 1024,
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
        self.dense_dim = dense_dim
        self.colbert_dim = colbert_dim
        self.max_tokens = max_tokens

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

def make_encoder(kind: str, dense_dim: int, colbert_dim: int, max_tokens: int) -> BaseEncoder:
    if kind in ("auto", "bge-m3"):
        try:
            return Bgem3Encoder(colbert_dim=colbert_dim, max_tokens=max_tokens)
        except Exception:
            if kind == "bge-m3":
                raise
    return HashEncoder(dense_dim=dense_dim, colbert_dim=colbert_dim, max_tokens=max_tokens)

"""Holograma Textual — busca vetorial em banco de dados COMUM.

O problema: bancos normais (Postgres sem pgvector, MySQL, SQLite...) não têm
busca vetorial. A solução clássica é instalar uma extensão ou um banco
vetorial. A solução daqui é outra: transformar o vetor em algo que QUALQUER
banco sabe indexar — inteiros e bytes.

Cada texto (já projetado nos três eixos pelo encoder) vira um HOLOGRAMA:

  assinatura holográfica : o vetor denso é atravessado por 1024 hiperplanos
      aleatórios fixos (LSH por hiperplano aleatório, Charikar 2002); cada
      hiperplano dá 1 bit — de que lado o significado caiu. 1024 bits = 128
      bytes = 16 BIGINTs. A distância de Hamming entre assinaturas aproxima
      o ângulo entre os vetores: hamming/bits ~ theta/pi. Postgres 14+
      calcula isso NATIVAMENTE: bit_count(a # b) sobre os 16 bigints.

  facetas : a assinatura é fatiada em bandas de 8 bits (LSH banding).
      Dois textos parecidos compartilham ao menos uma faceta com altíssima
      probabilidade (~94% para cos=0.8 com 16 bandas). Facetas são inteiros
      pequenos -> índice GIN/B-tree comum -> pré-filtro sublinear.

  eco denso : o vetor original quantizado em int8 (1/4 do espaço, erro <1%)
      para re-pontuação exata dos candidatos que passaram no pré-filtro.

  constelação estrutural : cada token vira uma mini-assinatura de 128 bits;
      o MaxSim (eixo 3) é feito com XOR + popcount — inteiro puro.

  espectro léxico : os termos esparsos já são (int, float) — tabela invertida
      comum com B-tree resolve (eixo 2 nunca precisou de vetor).

Resultado: os três eixos do TriRAG rodam em Postgres puro — ou em qualquer
coisa que armazene inteiros — com a mesma matemática de fusão quântica.
"""
from __future__ import annotations

import struct
from typing import List, Tuple

import numpy as np

from .portable import hyperplanes

HOLO_BITS = 1024          # bits da assinatura semântica (16 bigints)
HOLO_WORDS = HOLO_BITS // 64
TOKEN_BITS = 128          # bits por token na constelação (16 bytes)
BAND_BITS = 8             # bits por faceta
N_BANDS = 16              # facetas por holograma

_SEED = 7742


class Holographer:
    """Projeta TriVec -> holograma (assinaturas, facetas, eco, constelação)."""

    def __init__(self, dense_dim: int, colbert_dim: int):
        # hiperplanos determinísticos e PORTÁVEIS (portable.hyperplanes) — os
        # mesmos bits em Python e JS, para hologramas compatíveis entre as
        # linguagens. Seeds distintos por matriz para não correlacionar.
        self._planes = hyperplanes(_SEED, dense_dim, HOLO_BITS)
        self._tok_planes = hyperplanes(_SEED ^ 0x5D, colbert_dim, TOKEN_BITS)
        # tabela popcount para bytes (MaxSim binário em numpy puro)
        self._popcnt = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    # ------------------------------------------------------ assinatura ----

    def sign_dense(self, vec: np.ndarray) -> bytes:
        """Vetor denso -> 128 bytes (1024 bits de LSH por hiperplano)."""
        with np.errstate(all="ignore"):  # flags FPE espúrias do Accelerate/macOS
            bits = (vec.astype(np.float32) @ self._planes) >= 0.0
        return np.packbits(bits).tobytes()

    @staticmethod
    def sig_to_words(sig: bytes) -> List[int]:
        """128 bytes -> 16 BIGINTs assinados (para bancos com BIT_COUNT(int),
        como MySQL). No Postgres usamos sig_to_bitstring + BIT(n)."""
        return list(struct.unpack(f">{HOLO_WORDS}q", sig))

    @staticmethod
    def sig_to_bitstring(sig: bytes) -> str:
        """128 bytes -> '0101...' (1024 chars) para coluna BIT(1024) do Postgres,
        onde bit_count(a # b) é a distância de Hamming nativa."""
        return "".join(np.unpackbits(np.frombuffer(sig, dtype=np.uint8)).astype(str))

    @staticmethod
    def bands_of(sig: bytes) -> List[int]:
        """Facetas: banda i = (i << 8) | byte_i — o índice entra no valor
        para facetas de bandas diferentes nunca colidirem."""
        return [(i << 8) | sig[i] for i in range(N_BANDS)]

    # ------------------------------------------------------ eco denso -----

    @staticmethod
    def quantize(vec: np.ndarray) -> bytes:
        """float32 normalizado -> int8 (eco para re-pontuação exata)."""
        return np.clip(np.round(vec * 127.0), -127, 127).astype(np.int8).tobytes()

    @staticmethod
    def dequantize(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.int8).astype(np.float32) / 127.0

    # ---------------------------------------------------- constelação -----

    def sign_tokens(self, tokens: np.ndarray) -> bytes:
        """(T, C) float -> T assinaturas de 128 bits, empacotadas."""
        with np.errstate(all="ignore"):  # flags FPE espúrias do Accelerate/macOS
            bits = (tokens.astype(np.float32) @ self._tok_planes) >= 0.0
        return np.packbits(bits, axis=1).tobytes()  # T * 16 bytes

    def binary_maxsim(self, q_const: bytes, d_const: bytes) -> float:
        """MaxSim inteiro: para cada token da consulta, o token do documento
        com menor Hamming; similaridade = 1 - 2*hamming/bits (cos aproximado)."""
        nb = TOKEN_BITS // 8
        q = np.frombuffer(q_const, dtype=np.uint8).reshape(-1, nb)
        d = np.frombuffer(d_const, dtype=np.uint8).reshape(-1, nb)
        if q.size == 0 or d.size == 0:
            return 0.0
        xor = q[:, None, :] ^ d[None, :, :]          # (Tq, Td, 16)
        ham = self._popcnt[xor].sum(axis=2).astype(np.int32)  # (Tq, Td)
        sim = 1.0 - 2.0 * ham.min(axis=1).astype(np.float32) / TOKEN_BITS
        return float(sim.mean())

    # --------------------------------------------------------- útil -------

    def hamming(self, sig_a: bytes, sig_b: bytes) -> int:
        a = np.frombuffer(sig_a, dtype=np.uint8)
        b = np.frombuffer(sig_b, dtype=np.uint8)
        return int(self._popcnt[a ^ b].sum())

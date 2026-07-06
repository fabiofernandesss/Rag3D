"""Núcleo determinístico portável — bit a bit igual em Python e JavaScript.

O objetivo é que um Holograma Textual gerado em Python seja idêntico ao
gerado em JS: mesmos hiperplanos LSH, mesma assinatura, mesmas facetas. Assim
um documento salvo por um roda em Python pode ser encontrado por uma consulta
em JS no MESMO banco. É o que faz o TriRAG "funcionar em todas as linguagens"
de verdade — humanas e de programação.

Peças que precisam ser idênticas entre linguagens:
  - PRNG: splitmix64 counter-based (vetorizável em numpy, trivial em BigInt)
  - normais: Box-Muller sobre o PRNG (hiperplanos determinísticos)
  - crc32: hash estável (já usado no encoder fallback)

Nada de numpy.random aqui: o gerador do numpy (PCG64 + ziggurat) não é
replicável em JS. Este módulo é a fonte única da aleatoriedade determinística.
"""
from __future__ import annotations

import numpy as np

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
C1 = 0xBF58476D1CE4E5B9
C2 = 0x94D049BB133111EB


def splitmix64_stream(seed: int, n: int) -> np.ndarray:
    """n inteiros uint64 do splitmix64 counter-based.

    splitmix64 sequencial no passo k produz mix(S + (k+1)*GOLDEN); como é só
    função do índice, dá para calcular todos de uma vez (vetorizado) — e a
    versão escalar em JS gera exatamente os mesmos valores.
    """
    seed &= MASK64
    idx = np.arange(1, n + 1, dtype=np.uint64)
    g = np.uint64(GOLDEN)
    # aritmética uint64 no numpy faz wraparound (definido para sem sinal)
    with np.errstate(over="ignore"):
        z = (np.uint64(seed) + idx * g).astype(np.uint64)
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(C1)).astype(np.uint64)
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(C2)).astype(np.uint64)
        z = (z ^ (z >> np.uint64(31))).astype(np.uint64)
    return z


def uniforms(seed: int, n: int) -> np.ndarray:
    """n floats em [0,1) — top 53 bits de cada uint64 (igual ao JS)."""
    z = splitmix64_stream(seed, n)
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def normals(seed: int, n: int) -> np.ndarray:
    """n normais padrão via Box-Muller determinístico.

    normal j usa uniform(2j) e uniform(2j+1); pega só o ramo do cosseno
    (metade do par) — simples e idêntico entre linguagens.
    """
    u = uniforms(seed, 2 * n)
    u1 = np.maximum(u[0::2], 1e-12)          # evita log(0)
    u2 = u[1::2]
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def hyperplanes(seed: int, rows: int, cols: int) -> np.ndarray:
    """Matriz (rows, cols) de normais, preenchida em ordem C (row-major),
    idêntica ao laço i*cols+j do JS."""
    return normals(seed, rows * cols).astype(np.float32).reshape(rows, cols)


# ---------------------------------------------------------------- crc32 ---

_CRC_TABLE = None


def _crc_table():
    global _CRC_TABLE
    if _CRC_TABLE is None:
        tbl = []
        for n in range(256):
            c = n
            for _ in range(8):
                c = (0xEDB88320 ^ (c >> 1)) if (c & 1) else (c >> 1)
            tbl.append(c & 0xFFFFFFFF)
        _CRC_TABLE = tbl
    return _CRC_TABLE


def crc32(data: bytes) -> int:
    """CRC-32 IEEE — mesmo resultado de zlib.crc32 e da versão JS."""
    tbl = _crc_table()
    c = 0xFFFFFFFF
    for b in data:
        c = tbl[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF

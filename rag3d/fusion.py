"""Fusão dos três eixos — o novo cálculo (interferência quântica).

Inspiração: modelos quânticos de recuperação de informação (van Rijsbergen,
"The Geometry of Information Retrieval"; Quantum Language Models de Sordoni
et al., CIKM 2013). Em vez de somar pontuações (CombSUM) ou ranks (RRF),
cada canal contribui uma AMPLITUDE COMPLEXA para o documento:

    a_c(d) = sqrt(w_c * s_c(d)) * exp(i * phi_c(d))

  - módulo: raiz da pontuação normalizada do canal (pontuação = probabilidade,
    amplitude = raiz da probabilidade, como na regra de Born)
  - fase phi_c(d): posição do documento no ranking do canal, mapeada em [0, pi]
    (topo -> fase 0, fundo -> fase pi)

A pontuação final é a probabilidade da superposição (regra de Born):

    P(d) = | sum_c a_c(d) |^2
         = sum_c w_c s_c(d)                                    <- parte clássica
         + 2 * sum_{c<c'} sqrt(w_c s_c w_c' s_c') * cos(phi_c - phi_c')   <- interferência

Leitura física do termo de interferência:
  - canais que CONCORDAM no rank têm fases próximas -> cos ~ +1 ->
    interferência CONSTRUTIVA: o documento sobe.
  - canais que discordam fortemente -> fases opostas -> cos ~ -1 ->
    interferência DESTRUTIVA: o documento desce.
  - documento visto por um canal só: amplitude única, sem termo cruzado ->
    recupera exatamente a pontuação clássica daquele canal.

Com interference_strength = 0 o cálculo colapsa no CombSUM ponderado
clássico; com 1, interferência plena. RRF incluído como linha de base.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

Ranking = List[Tuple[int, float]]  # [(chunk_id, score), ...] em ordem decrescente


@dataclass
class FusedHit:
    chunk_id: int
    score: float
    classical: float
    interference: float
    per_channel: Dict[str, float] = field(default_factory=dict)  # score normalizado por canal
    channels: List[str] = field(default_factory=list)            # canais que acharam o doc


def _minmax(ranking: Ranking) -> Dict[int, float]:
    """Normaliza pontuações do canal para [0,1] dentro do pool de candidatos."""
    if not ranking:
        return {}
    vals = [s for _, s in ranking]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {cid: 1.0 for cid, _ in ranking}
    return {cid: (s - lo) / (hi - lo) for cid, s in ranking}


def _phases(ranking: Ranking) -> Dict[int, float]:
    """Fase pela posição no ranking: topo -> 0, fundo -> pi."""
    n = len(ranking)
    if n <= 1:
        return {cid: 0.0 for cid, _ in ranking}
    return {cid: math.pi * i / (n - 1) for i, (cid, _) in enumerate(ranking)}


def quantum_fuse(
    channels: Dict[str, Ranking],
    weights: Dict[str, float],
    top_k: int,
    interference_strength: float = 1.0,
) -> List[FusedHit]:
    """Superposição dos três eixos com interferência entre canais."""
    norm = {name: _minmax(r) for name, r in channels.items()}
    phase = {name: _phases(r) for name, r in channels.items()}
    names = list(channels.keys())

    all_ids = set()
    for r in channels.values():
        all_ids.update(cid for cid, _ in r)

    hits: List[FusedHit] = []
    for cid in all_ids:
        # amplitude e fase por canal presente
        amps: List[Tuple[str, float, float]] = []  # (canal, amplitude, fase)
        per_channel: Dict[str, float] = {}
        for name in names:
            s = norm[name].get(cid)
            if s is None or s <= 0.0:
                continue
            per_channel[name] = s
            amps.append((name, math.sqrt(weights.get(name, 1.0) * s), phase[name][cid]))

        classical = sum(a * a for _, a, _ in amps)
        interf = 0.0
        for i in range(len(amps)):
            for j in range(i + 1, len(amps)):
                _, ai, pi_ = amps[i]
                _, aj, pj = amps[j]
                interf += 2.0 * ai * aj * math.cos(pi_ - pj)
        score = classical + interference_strength * interf
        hits.append(
            FusedHit(
                chunk_id=cid,
                score=score,
                classical=classical,
                interference=interf,
                per_channel=per_channel,
                channels=[n for n, _, _ in amps],
            )
        )

    # desempate determinístico por id: sets de Python e JS iteram em ordens
    # diferentes, então empates de pontuação precisam de um critério estável
    # e IGUAL nas duas linguagens (senão devolvem docs diferentes).
    hits.sort(key=lambda h: (-h.score, h.chunk_id))
    return hits[:top_k]


def rrf_fuse(
    channels: Dict[str, Ranking],
    weights: Dict[str, float],
    top_k: int,
    rrf_k: int = 60,
) -> List[FusedHit]:
    """Reciprocal Rank Fusion clássico — linha de base para comparação."""
    scores: Dict[int, float] = {}
    found: Dict[int, List[str]] = {}
    for name, ranking in channels.items():
        w = weights.get(name, 1.0)
        for rank, (cid, _) in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + w / (rrf_k + rank + 1)
            found.setdefault(cid, []).append(name)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return [
        FusedHit(chunk_id=cid, score=s, classical=s, interference=0.0, channels=found[cid])
        for cid, s in ranked
    ]


def fuse(
    channels: Dict[str, Ranking],
    weights: Sequence[float],
    top_k: int,
    method: str = "quantum",
    interference_strength: float = 1.0,
    rrf_k: int = 60,
) -> List[FusedHit]:
    names = list(channels.keys())
    wmap = {n: float(w) for n, w in zip(names, weights)}
    if method == "rrf":
        return rrf_fuse(channels, wmap, top_k, rrf_k=rrf_k)
    return quantum_fuse(channels, wmap, top_k, interference_strength=interference_strength)

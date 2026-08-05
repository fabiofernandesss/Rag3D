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


def coherence(norm_scores: Dict[int, float]) -> float:
    """Coerência do canal — quão DECIDIDO ele está nesta consulta.

    Trata as pontuações normalizadas como um estado misto (matriz densidade
    diagonal rho = diag(p)). A pureza Tr(rho^2) = sum(p^2) vale 1 para um
    estado puro (um pico só, canal decidido) e 1/n para o estado maximamente
    misto (distribuição chapada, canal sem opinião). Reescalada para [0,1]:

        kappa = (n * Tr(rho^2) - 1) / (n - 1)

    Um canal decohered (kappa ~ 0) não sabe de nada nesta consulta e deve
    pesar menos na superposição; um canal puro (kappa ~ 1) deve pesar mais.

    EXPERIMENTAL — desligado por padrão (coherence_strength = 0). Honestidade:
    (1) Tr(rho^2) numa rho diagonal é exatamente a entropia de Rényi-2 —
        "pureza" é entropia com outro nome, não um conceito novo;
    (2) a literatura de QPP mostra que decisão de pontuação prediz bem a
        DIFICULDADE da consulta e mal QUAL canal está certo (~0.09 de
        correlação) — um canal pode estar decidido e errado;
    (3) canal com poucos candidatos parece trivialmente decidido (por isso o
        piso de 3 abaixo: com 1-2 candidatos não dá para distinguir decisão
        de escassez, então não se concede o bônus de confiança).
    """
    vals = [s for s in norm_scores.values() if s > 0.0]
    n = len(vals)
    if n <= 2:
        return 0.0
    total = sum(vals)
    if total <= 0.0:
        return 0.0
    purity = sum((v / total) ** 2 for v in vals)
    return max(0.0, min(1.0, (purity * n - 1.0) / (n - 1.0)))


def _coherent_weights(
    base: Dict[str, float], norm: Dict[str, Dict[int, float]], strength: float
) -> Dict[str, float]:
    """Modula os pesos dos canais pela coerência de cada um NESTA consulta."""
    if strength <= 0.0:
        return base
    kappa = {name: coherence(scores) for name, scores in norm.items()}
    kmax = max(kappa.values()) if kappa else 0.0
    if kmax <= 0.0:
        return base
    return {
        name: base.get(name, 1.0) * ((1.0 - strength) + strength * (kappa[name] / kmax))
        for name in norm
    }


def quantum_fuse(
    channels: Dict[str, Ranking],
    weights: Dict[str, float],
    top_k: int,
    interference_strength: float = 1.0,
    coherence_strength: float = 0.0,
) -> List[FusedHit]:
    """Superposição dos três eixos com interferência entre canais."""
    norm = {name: _minmax(r) for name, r in channels.items()}
    phase = {name: _phases(r) for name, r in channels.items()}
    weights = _coherent_weights(weights, norm, coherence_strength)
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
    coherence_strength: float = 0.0,
) -> List[FusedHit]:
    names = list(channels.keys())
    wmap = {n: float(w) for n, w in zip(names, weights)}
    if coherence_strength > 0.0:
        wmap = _coherent_weights(wmap, {n: _minmax(r) for n, r in channels.items()}, coherence_strength)
    if method == "rrf":
        return rrf_fuse(channels, wmap, top_k, rrf_k=rrf_k)
    return quantum_fuse(channels, wmap, top_k, interference_strength=interference_strength)


# ---------------------------------------------- seleção fermiônica (DPP) ---
#
# A fusão acima é BOSÔNICA: as amplitudes dos eixos somam e a concordância
# gera interferência construtiva — ela decide QUAIS documentos são relevantes.
# Falta decidir QUAL CONJUNTO devolver. Aí entra o princípio oposto.
#
# Um conjunto de k documentos é um estado de k partículas. Se ele for
# ANTISSIMÉTRICO (determinante de Slater), a amplitude é
#
#     |psi_S|^2 = det(Gram(v_S)) = Vol^2(v_1, ..., v_k)
#
# ou seja, o volume² do paralelepípedo gerado pelos vetores. Dois documentos
# idênticos são duas partículas no mesmo estado: o determinante zera —
# EXCLUSÃO DE PAULI. Redundância é proibida por construção, não por heurística.
#
# Isto é exatamente um Determinantal Point Process (Kulesza & Taskar), e o
# argmax é obtido pelo guloso com atualização de Cholesky em O(k²N)
# (Chen et al. 2018). O ganho de cada candidato é combinado em log:
#
#     ganho(i) = theta * 2*log(relevância_i) + (1-theta) * log(volume residual_i)
#
# theta = 1 - diversidade. Com diversidade = 0 devolve exatamente o ranking
# fundido (trava de segurança, igual ao lambda=0 da interferência).


def fermionic_select(
    items: Sequence[Tuple[int, float]],
    vectors: Dict[int, "np.ndarray"],
    top_k: int,
    diversity: float = 0.0,
) -> List[int]:
    """Escolhe k ids maximizando relevância × volume (det de Slater / MAP-DPP).

    items: [(id, relevância)] já ordenado por relevância desc.
    vectors: id -> vetor denso (norma 1). Id sem vetor é tratado como
             ortogonal a todos (nunca é podado por redundância).
    """
    import numpy as np

    ids = [i for i, _ in items]
    n = len(ids)
    if diversity <= 0.0 or n <= 1 or top_k <= 0:
        return ids[:top_k]
    k = min(top_k, n)
    eps = 1e-12

    rel = np.array([s for _, s in items], dtype=np.float64)
    lo, hi = float(rel.min()), float(rel.max())
    q = (rel - lo) / (hi - lo) if hi - lo > eps else np.ones(n)
    log_q2 = 2.0 * np.log(np.maximum(q, eps))          # 2·log(relevância)

    # float64 em todo o cálculo — o JS usa float64 nativo; manter o mesmo tipo
    # garante que as duas linguagens escolham exatamente o mesmo conjunto.
    dim = next((len(v) for v in vectors.values() if v is not None), 0)
    V = np.zeros((n, dim), dtype=np.float64)
    for row, cid in enumerate(ids):
        v = vectors.get(cid)
        if v is not None and len(v) == dim:
            V[row] = v

    theta = max(0.0, min(1.0, 1.0 - diversity))
    d2 = np.ones(n, dtype=np.float64)                  # volume residual (diag da Cholesky de S)
    C = np.zeros((n, k), dtype=np.float64)
    avail = np.ones(n, dtype=bool)
    chosen: List[int] = []

    for t in range(k):
        gain = theta * log_q2 + (1.0 - theta) * np.log(np.maximum(d2, eps))
        gain[~avail] = -np.inf
        j = int(np.argmax(gain))
        if not avail[j]:
            break
        # d² é o volume que o candidato ainda acrescenta. Se acabou (posto
        # esgotado / só restam duplicatas), a atualização de Cholesky fica
        # numericamente instável — completa por relevância pura e sai.
        if d2[j] < eps:
            chosen.extend(cid for cid, ok in zip(ids, avail) if ok)
            return chosen[:k]
        chosen.append(ids[j])
        avail[j] = False
        if len(chosen) == k:
            break
        # atualização de Cholesky: remove a componente de j do espaço restante
        with np.errstate(all="ignore"):
            sim = V @ V[j]                             # similaridade de j com todos
            ej = (sim - C[:, :t] @ C[j, :t]) / math.sqrt(max(d2[j], eps))
        C[:, t] = ej
        d2 = np.maximum(d2 - ej * ej, 0.0)
    return chosen

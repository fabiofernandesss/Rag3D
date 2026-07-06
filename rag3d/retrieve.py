"""Busca tridimensional: a consulta é projetada nos três eixos.

  eixo 1 (semântico)  : cosseno denso sobre todo o índice
  eixo 2 (léxico)     : termos esparsos com IDF sobre todo o índice
  eixo 3 (estrutural) : MaxSim token a token sobre o pool de candidatos
                        dos eixos 1+2 (late-interaction como refinador —
                        prática padrão ColBERT: caro demais para varrer tudo)

Cada eixo devolve seu próprio ranking (as "três respostas"); a fusão
quântica produz o ranking combinado; small-to-big troca o chunk pequeno
pela janela-pai na hora de montar o contexto.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import TriRagConfig
from .encoders import BaseEncoder, TriVec
from .fusion import FusedHit, fuse
from .store import TriStore

CHANNELS = ("semantico", "lexico", "estrutural")

_SYS_EXPAND = (
    "Você reescreve consultas para busca em documentos normativos (editais, leis). "
    "Dada uma pergunta, gere até {N} variações com termos alternativos que provavelmente "
    "aparecem no documento (sinônimos, palavras-chave, números de seção/item, siglas expandidas). "
    "Uma variação por linha, sem numeração, sem explicações."
)


@dataclass
class TriResult:
    query: str
    fused: List[dict]                       # hits fundidos, com texto e janela ampla
    views: Dict[str, List[dict]] = field(default_factory=dict)  # top-k de CADA eixo
    stats: dict = field(default_factory=dict)


class TriRetriever:
    def __init__(self, store: TriStore, encoder: BaseEncoder, cfg: TriRagConfig, reranker=None, llm=None):
        self.store = store
        self.encoder = encoder
        self.cfg = cfg
        self.reranker = reranker  # opcional: reordena o topo fundido com um LLM
        self.llm = llm            # opcional: query expansion (Rag3D)

    def _expanded_query_vec(self, query: str) -> TriVec:
        """Gera variações da consulta (LLM) e combina os vetores como OR — mais
        recall, mesma saída de rankings. Espelha retrieve.js. Falha silenciosa."""
        if not self.cfg.expand_query or self.llm is None or not self.llm.available():
            return self.encoder.encode([query], is_query=True)[0]
        variants: List[str] = []
        try:
            raw = self.llm.complete(
                _SYS_EXPAND.format(N=self.cfg.expand_query_max),
                [{"role": "user", "content": query}], max_tokens=200,
            )
            variants = [s.strip() for s in raw.split("\n") if s.strip()][: self.cfg.expand_query_max]
        except Exception:
            pass
        if not variants:
            return self.encoder.encode([query], is_query=True)[0]
        vecs = self.encoder.encode([query] + variants, is_query=True)
        dense = np.sum([v.dense for v in vecs], axis=0).astype(np.float32)
        n = float(np.linalg.norm(dense))
        if n > 0:
            dense = dense / n
        sparse: Dict[int, float] = {}
        for v in vecs:
            for t, w in v.sparse.items():
                if t not in sparse or sparse[t] < w:
                    sparse[t] = w
        tokens = np.vstack([v.tokens for v in vecs])[: self.cfg.max_colbert_tokens]
        return TriVec(dense=dense, sparse=sparse, tokens=tokens)

    def _prefetch(self, ids: List[int]):
        """Busca os chunks e seus pais em DUAS queries (não uma por lista)."""
        by_id = {r["id"]: r for r in self.store.get_chunks(list(set(ids)))}
        parent_ids = list({r["parent_id"] for r in by_id.values() if r["parent_id"]})
        parents = {p["id"]: p for p in self.store.get_chunks(parent_ids)} if parent_ids else {}
        return by_id, parents

    @staticmethod
    def _assemble(cid, by_id, parents, score=None):
        """Monta um hit a partir dos mapas já buscados (sem tocar o banco)."""
        r = by_id.get(cid)
        if r is None:
            return None
        wide = parents.get(r["parent_id"], {}).get("text") if r["parent_id"] else None
        return {
            "id": r["id"], "kind": r["kind"], "doc_id": r.get("doc_id"), "pos": r.get("pos"),
            "text": r["text"], "wide": wide or r["text"],
            "n_tokens": r["n_tokens"], "turn_no": r["turn_no"], "accessed_turn": r.get("accessed_turn"),
            "created": r["created"], "importance": r["importance"], "score": score,
        }

    def _stitch(self, hits):
        """Costura de contiguidade (Rag3D): junta vizinhos pos±raio num bloco só."""
        R = int(self.cfg.stitch_radius or 0)
        if R <= 0:
            return hits
        by_doc = {}
        for h in hits:
            if h.get("doc_id") is None or h.get("pos") is None or h["kind"] != "chunk":
                continue
            s = by_doc.setdefault(h["doc_id"], set())
            for d in range(-R, R + 1):
                s.add(h["pos"] + d)
        text_by = {}
        for doc_id, positions in by_doc.items():
            for row in self.store.neighbors(doc_id, sorted(positions)):
                text_by[(doc_id, row["pos"])] = row["text"]
        for h in hits:
            if h.get("doc_id") is None or h.get("pos") is None or h["kind"] != "chunk":
                continue
            parts = [text_by[(h["doc_id"], h["pos"] + d)] for d in range(-R, R + 1)
                     if (h["doc_id"], h["pos"] + d) in text_by]
            if parts:
                h["wide"] = " ".join(parts)
        return hits

    def _hydrate(self, ids: List[int], scores: Optional[List[float]] = None) -> List[dict]:
        by_id, parents = self._prefetch(ids)
        sc = dict(zip(ids, scores)) if scores else {}
        return [h for h in (self._assemble(cid, by_id, parents, sc.get(cid)) for cid in ids) if h]

    def search(self, query: str, top_k: Optional[int] = None, channel_k: Optional[int] = None) -> TriResult:
        top_k = top_k or self.cfg.top_k
        channel_k = channel_k or self.cfg.channel_k

        q = self._expanded_query_vec(query)

        # três frentes
        dense = self.store.dense_search(q.dense, channel_k)
        sparse = self.store.sparse_search(q.sparse, channel_k)
        pool = list({cid for cid, _ in dense} | {cid for cid, _ in sparse})
        struct = self.store.colbert_scores(q.tokens, pool)[:channel_k]

        channels = {"semantico": dense, "lexico": sparse, "estrutural": struct}

        # se vai reranquear, funde um pool maior e deixa o reranker escolher o topo
        do_rerank = self.reranker is not None and self.reranker.available()
        fuse_k = max(top_k, self.cfg.rerank_pool) if do_rerank else top_k

        # fusão (quântica ou RRF)
        fused_hits: List[FusedHit] = fuse(
            channels,
            self.cfg.channel_weights,
            fuse_k,
            method=self.cfg.fusion,
            interference_strength=self.cfg.interference_strength,
            rrf_k=self.cfg.rrf_k,
        )
        # hidratação em lote: uma prefetch para fundido + as 3 visões juntas,
        # em vez de 4 hidratações separadas (cada uma batia o banco 2x).
        fused_ids = [h.chunk_id for h in fused_hits]
        view_rank = {name: ranking[:top_k] for name, ranking in channels.items()}
        all_ids = list(fused_ids) + [cid for r in view_rank.values() for cid, _ in r]
        by_id, parents = self._prefetch(all_ids)

        fused = [h for h in (self._assemble(cid, by_id, parents) for cid in fused_ids) if h]
        hit_by_id = {h.chunk_id: h for h in fused_hits}
        for row in fused:  # alinhado por id (assemble pode ter descartado ids)
            h = hit_by_id[row["id"]]
            row["score"] = h.score
            row["classical"] = h.classical
            row["interference"] = h.interference
            row["channels"] = h.channels
            row["per_channel"] = h.per_channel

        # reranker LLM-agnóstico: reordena o pool fundido e corta em top_k
        if do_rerank:
            fused = self.reranker.rerank(query, fused, top_k=top_k)
        else:
            fused = fused[:top_k]
        self._stitch(fused)  # costura de contiguidade (Rag3D) no top-k final

        # as três visões independentes (para leitores por eixo)
        views: Dict[str, List[dict]] = {}
        for name, ranking in view_rank.items():
            sc = dict(ranking)
            views[name] = [h for h in (self._assemble(cid, by_id, parents, sc.get(cid)) for cid, _ in ranking) if h]

        return TriResult(
            query=query,
            fused=fused,
            views=views,
            stats={
                "candidatos": {"semantico": len(dense), "lexico": len(sparse), "estrutural": len(struct)},
                "pool": len(pool),
                "fusao": self.cfg.fusion,
            },
        )

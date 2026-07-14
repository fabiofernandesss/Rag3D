"""Ablação REAL em base pública BEIR (nDCG@10, Recall@10) com encoder BGE-M3.

Responde à pergunta que o harness sintético não responde: com semântica de
verdade, a fusão quântica supera o RRF? E quanto cada eixo contribui?

  # baixe um dataset BEIR em bench_data/<nome>/ (corpus.jsonl, queries.jsonl, qrels/test.tsv)
  .venv/bin/python tests/beir_ablation.py nfcorpus [max_docs]

Usa o encoder BGE-M3 (RAG3D_ENCODER=bge-m3) e o store SQLite do próprio RAG3D.
Compara, na MESMA recuperação: eixos isolados, CombSUM, fusão quântica
(λ=0/0.5/1) e RRF. Números honestos, reproduzíveis.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rag3d.config import TriRagConfig
from rag3d.encoders import make_encoder
from rag3d.fusion import fuse
from rag3d.store import TriStore

DATA = Path(__file__).resolve().parents[1] / "bench_data"
CHANNEL_K = 100   # candidatos por eixo
TOPK = 10         # nDCG@10 / Recall@10


def load_beir(name: str):
    root = DATA / name
    corpus = {}
    for line in open(root / "corpus.jsonl", encoding="utf-8"):
        o = json.loads(line)
        txt = ((o.get("title") or "") + ". " + (o.get("text") or "")).strip()
        corpus[o["_id"]] = txt
    queries = {}
    for line in open(root / "queries.jsonl", encoding="utf-8"):
        o = json.loads(line)
        queries[o["_id"]] = o["text"]
    qrels = {}
    with open(root / "qrels" / "test.tsv", encoding="utf-8") as f:
        next(f)  # header: query-id, corpus-id, score
        for line in f:
            qid, cid, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[cid] = int(score)
    # só queries com julgamento de teste
    queries = {q: t for q, t in queries.items() if q in qrels}
    return corpus, queries, qrels


def ndcg_at_k(ranking, rel, k=TOPK):
    dcg = 0.0
    for i, cid in enumerate(ranking[:k]):
        g = rel.get(cid, 0)
        if g:
            dcg += (2 ** g - 1) / math.log2(i + 2)
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranking, rel, k=TOPK):
    rset = set(rel.keys())
    if not rset:
        return 0.0
    return len(set(ranking[:k]) & rset) / len(rset)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    max_docs = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    corpus, queries, qrels = load_beir(name)
    if max_docs:
        # mantém docs julgados + completa até max_docs (para caber no tempo/CPU)
        judged = {c for rel in qrels.values() for c in rel}
        keep = list(judged) + [c for c in corpus if c not in judged]
        keep = set(keep[:max(max_docs, len(judged))])
        corpus = {c: t for c, t in corpus.items() if c in keep}
    print(f"[{name}] corpus={len(corpus)} queries_teste={len(queries)}")

    cfg = TriRagConfig()
    cfg.encoder = "bge-m3"
    enc = make_encoder(cfg.encoder, cfg.dense_dim, cfg.colbert_dim, cfg.max_colbert_tokens)
    print(f"encoder: {enc.name}")

    tmp = Path(tempfile.mkdtemp())
    store = TriStore(tmp / "beir.db")

    # ---- indexação (batch-encode) ----
    cids = list(corpus.keys())
    texts = [corpus[c] for c in cids]
    chunk_to_cid = {}
    t0 = time.time()
    B = 16
    for i in range(0, len(texts), B):
        vecs = enc.encode(texts[i:i + B])
        for cid, text, vec in zip(cids[i:i + B], texts[i:i + B], vecs):
            doc_id = store.add_doc(cid, cid, 0)
            ch_id = store.add_chunk(doc_id, text, text, 0, vec)
            chunk_to_cid[ch_id] = cid
        if (i // B) % 20 == 0:
            print(f"  indexado {i+len(vecs)}/{len(texts)}  ({(time.time()-t0):.0f}s)", flush=True)
    store.commit()
    print(f"indexação: {time.time()-t0:.0f}s")

    # ---- avaliação: mesma recuperação, várias fusões ----
    W = cfg.channel_weights
    strategies = ["semantico", "lexico", "estrutural",
                  "CombSUM(λ=0)", "quantica(λ=0.5)", "quantica(λ=1)", "RRF(k=60)"]
    ndcg = {s: 0.0 for s in strategies}
    rec = {s: 0.0 for s in strategies}
    nq = 0
    t0 = time.time()
    for qid, qtext in queries.items():
        rel = qrels[qid]
        q = enc.encode([qtext], is_query=True)[0]
        dense = store.dense_search(q.dense, CHANNEL_K)
        sparse = store.sparse_search(q.sparse, CHANNEL_K)
        pool = list({c for c, _ in dense} | {c for c, _ in sparse})
        struct = store.colbert_scores(q.tokens, pool)[:CHANNEL_K]
        ch = {"semantico": dense, "lexico": sparse, "estrutural": struct}

        def cids_of(ranking):
            return [chunk_to_cid.get(cid) for cid, _ in ranking[:TOPK]]

        def fused_cids(method, lam=1.0):
            hits = fuse(ch, W, TOPK, method=method, interference_strength=lam, rrf_k=60)
            return [chunk_to_cid.get(h.chunk_id) for h in hits]

        rankings = {
            "semantico": cids_of(dense),
            "lexico": cids_of(sparse),
            "estrutural": cids_of(struct),
            "CombSUM(λ=0)": fused_cids("quantum", 0.0),
            "quantica(λ=0.5)": fused_cids("quantum", 0.5),
            "quantica(λ=1)": fused_cids("quantum", 1.0),
            "RRF(k=60)": fused_cids("rrf"),
        }
        for s, r in rankings.items():
            r = [c for c in r if c]
            ndcg[s] += ndcg_at_k(r, rel)
            rec[s] += recall_at_k(r, rel)
        nq += 1
        if nq % 50 == 0:
            print(f"  avaliadas {nq}/{len(queries)}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n== {name} · {nq} queries · encoder {enc.name} ==")
    print(f"{'estratégia':<18} {'nDCG@10':>8} {'Recall@10':>10}")
    for s in strategies:
        print(f"{s:<18} {ndcg[s]/nq:>8.4f} {rec[s]/nq:>10.4f}")


if __name__ == "__main__":
    main()

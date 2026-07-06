"""Benchmark: fusão quântica vs RRF vs CombSUM vs eixos isolados.

Corpus sintético multilíngue com "agulhas" rotuladas — cada consulta tem um
documento correto conhecido. Mede Recall@k e MRR de cada estratégia.

  python3 tests/bench_fusion.py [n_docs]

Isto NÃO substitui avaliação nos seus dados reais — é a ferramenta para
comparar as estratégias na sua máquina. Troque o corpus pelos seus documentos
e as consultas pelas suas perguntas reais quando puder rotular 50+ pares.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag3d.config import TriRagConfig
from rag3d.engine import TriRag
from rag3d.fusion import fuse
from rag3d.llm import NoLLM

TOPICS = [
    ("pt", "relatório financeiro do trimestre {i} com lucro de {i}00 mil reais"),
    ("pt", "processo judicial número {i} sobre disputa de terras no caso {i}"),
    ("en", "quarterly engineering report {i} describing microservice latency of {i} ms"),
    ("en", "clinical trial {i} results for the new drug candidate batch {i}"),
    ("zh", "第{i}号项目报告，关于北京数据中心的性能指标{i}"),
    ("pt", "ata da reunião {i} do conselho decidindo o orçamento do projeto {i}"),
]

QUERIES = [
    ("lucro do trimestre {i}", 0),
    ("disputa de terras caso {i}", 1),
    ("microservice latency report {i}", 2),
    ("drug candidate batch {i}", 3),
    ("北京数据中心 报告 {i}", 4),
    ("orçamento do projeto {i} na ata", 5),
]


def build(n_docs: int):
    cfg = TriRagConfig()
    cfg.data_dir = Path(tempfile.mkdtemp())
    cfg.encoder = "fallback"
    cfg.contextual_enrich = False
    cfg.small_corpus_tokens = 0
    rag = TriRag(cfg, llm=NoLLM())
    truth = {}  # (query_template_idx, i) -> texto esperado
    for i in range(n_docs // len(TOPICS)):
        for t_idx, (_, tpl) in enumerate(TOPICS):
            text = tpl.format(i=i)
            rag.ingest(text, title=f"d{t_idx}_{i}")
            truth[(t_idx, i)] = text
    return rag, truth


def evaluate(rag: TriRag, truth: dict, k: int = 5):
    strategies = {
        "eixo semântico": None, "eixo léxico": None, "eixo estrutural": None,
        "CombSUM (λ=0)": None, "quântica (λ=1)": None, "RRF k=60": None,
    }
    hits = {s: 0 for s in strategies}
    mrr = {s: 0.0 for s in strategies}
    n_q = 0

    n_per = len(truth) // len(TOPICS)
    for i in range(0, n_per, max(1, n_per // 20)):  # amostra de consultas
        for q_idx, (q_tpl, t_idx) in enumerate(QUERIES):
            query = q_tpl.format(i=i)
            expected = truth[(t_idx, i)]
            n_q += 1

            r = rag.search(query, top_k=k)
            ch = {name: [(row["id"], row["score"]) for row in rows]
                  for name, rows in r.views.items()}

            def rank_of(ranking_texts):
                for pos, t in enumerate(ranking_texts):
                    if t == expected:
                        return pos
                return None

            def texts_of(id_score):
                rows = rag.store.get_chunks([cid for cid, _ in id_score[:k]])
                return [row["text"] for row in rows]

            per = {
                "eixo semântico": texts_of(ch["semantico"]),
                "eixo léxico": texts_of(ch["lexico"]),
                "eixo estrutural": texts_of(ch["estrutural"]),
                "CombSUM (λ=0)": [h.chunk_id for h in fuse(ch, rag.cfg.channel_weights, k, "quantum", 0.0)],
                "quântica (λ=1)": [h.chunk_id for h in fuse(ch, rag.cfg.channel_weights, k, "quantum", 1.0)],
                "RRF k=60": [h.chunk_id for h in fuse(ch, rag.cfg.channel_weights, k, "rrf")],
            }
            for s in ("CombSUM (λ=0)", "quântica (λ=1)", "RRF k=60"):
                rows = rag.store.get_chunks(per[s])
                per[s] = [row["text"] for row in rows]

            for s, ranked in per.items():
                pos = rank_of(ranked)
                if pos is not None:
                    hits[s] += 1
                    mrr[s] += 1.0 / (pos + 1)

    print(f"\n{n_q} consultas, Recall@{k} e MRR:")
    print(f"{'estratégia':<20} {'recall':>8} {'MRR':>8}")
    for s in hits:
        print(f"{s:<20} {hits[s]/n_q:>8.1%} {mrr[s]/n_q:>8.3f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    print(f"construindo corpus sintético com ~{n} docs (encoder fallback)...")
    rag, truth = build(n)
    evaluate(rag, truth)

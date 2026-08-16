"""Teste de fumaça end-to-end: ingestão tri-eixo, busca multilíngue, fusão, memória.

Roda com o encoder fallback (zero dependências) e sem LLM — valida a
mecânica tridimensional inteira. Qualidade semântica real vem do BGE-M3.
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
from rag3d.textproc import estimate_tokens, split_sentences, word_tokens

FAILS = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FALHOU"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def make_rag(tmp: Path, **cfg_over) -> TriRag:
    cfg = TriRagConfig()
    cfg.data_dir = tmp
    cfg.encoder = "fallback"
    cfg.contextual_enrich = False
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    return TriRag(cfg, llm=NoLLM())


def test_textproc():
    print("\n== textproc (agnóstico de língua) ==")
    s = split_sentences("Olá mundo. Tudo bem? 你好。这是测试！النص العربي هنا.")
    check("segmenta multi-escrita", len(s) >= 4, f"{s}")
    toks = word_tokens("O contrato 移动电话 vence")
    check("CJK vira bigramas", "移动" in toks and "contrato" in toks, f"{toks}")
    check("estimativa tokens > 0", estimate_tokens("teste de tokens aqui") > 0)


def test_ingest_and_axes():
    print("\n== ingestão tri-eixo + busca ==")
    with tempfile.TemporaryDirectory() as td:
        rag = make_rag(Path(td), small_corpus_tokens=0)  # força retrieval
        rag.ingest("O contrato de aluguel vence em 15 de março de 2027 e o valor é R$ 3.500 por mês.", title="contrato")
        rag.ingest("The rocket launch is scheduled for July 2026 from Cape Canaveral.", title="rocket")
        rag.ingest("会议将于星期五上午十点在北京举行。", title="meeting")
        rag.ingest("A receita do bolo de fubá leva 3 ovos, fubá e erva-doce. Asse por 40 minutos.", title="bolo")

        r = rag.search("quando vence o contrato de aluguel?")
        check("3 visões presentes", set(r.views.keys()) == {"semantico", "lexico", "estrutural"})
        check("fusão devolve hits", len(r.fused) > 0)
        top = r.fused[0]["text"]
        check("acha o contrato (PT)", "contrato" in top, top[:80])
        check("hit tem interferência", "interference" in r.fused[0])

        r2 = rag.search("rocket launch date")
        check("acha o foguete (EN)", "rocket" in r2.fused[0]["text"].lower(), r2.fused[0]["text"][:80])

        r3 = rag.search("会议在哪里举行")
        check("acha a reunião (中文)", "北京" in r3.fused[0]["text"], r3.fused[0]["text"][:60])

        # RRF baseline também funciona
        rag.cfg.fusion = "rrf"
        r4 = rag.search("bolo de fubá receita")
        check("RRF acha o bolo", "fubá" in r4.fused[0]["text"], r4.fused[0]["text"][:80])


def test_adaptive_chunking():
    print("\n== chunking adaptativo (pequeno vs grande) ==")
    with tempfile.TemporaryDirectory() as td:
        rag = make_rag(Path(td))
        tiny = rag.ingest("Nota curta: senha do wifi é abc123.")
        check("doc minúsculo = 1 chunk", tiny["chunks"] == 1, str(tiny))

        big = "\n\n".join(
            f"Capítulo {i}. " + " ".join(f"Frase {j} do capítulo {i} sobre o assunto tal." for j in range(40))
            for i in range(12)
        )
        r = rag.ingest(big, title="livro")
        check("doc grande vira vários chunks", r["chunks"] > 3, str(r))

        res = rag.search("Frase 5 do capítulo 7")
        hit = res.fused[0]
        check("small-to-big: janela ampla >= chunk", len(hit["wide"]) >= len(hit["text"]))


def test_quantum_fusion_math():
    print("\n== matemática da fusão quântica ==")
    # dois canais concordam no doc 1; doc 2 só aparece num canal
    channels = {
        "a": [(1, 0.9), (2, 0.8), (3, 0.1)],
        "b": [(1, 0.85), (3, 0.2)],
        "c": [(1, 0.7), (2, 0.05)],
    }
    hits = fuse(channels, (1.0, 1.0, 1.0), 5, method="quantum", interference_strength=1.0)
    by_id = {h.chunk_id: h for h in hits}
    check("concordância -> topo", hits[0].chunk_id == 1)
    check("interferência construtiva no doc 1", by_id[1].interference > 0, f"{by_id[1].interference:.3f}")

    # com interferência 0 colapsa no clássico
    h0 = fuse(channels, (1.0, 1.0, 1.0), 5, method="quantum", interference_strength=0.0)
    for h in h0:
        check(f"λ=0: score==clássico (doc {h.chunk_id})", abs(h.score - h.classical) < 1e-9)

    # doc visto por 1 canal não ganha termo cruzado
    solo = [h for h in hits if len(h.channels) == 1]
    for h in solo:
        check(f"doc solo sem interferência (doc {h.chunk_id})", abs(h.interference) < 1e-9)


def test_fermionic_and_coherence():
    print("\n== diversidade greedy DPP + coerência ==")
    import numpy as np
    from rag3d.fusion import coherence, fermionic_select

    # 4 clusters de 4 vetores quase idênticos; relevância decrescente
    rng = np.random.default_rng(3)
    base = rng.standard_normal((4, 16))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    vecs, items = {}, []
    for g in range(4):
        for c in range(4):
            cid = g * 10 + c
            v = base[g] + 0.02 * rng.standard_normal(16)
            vecs[cid] = (v / np.linalg.norm(v)).astype(np.float32)
            items.append((cid, 1.0 - 0.01 * (g * 4 + c)))
    items.sort(key=lambda x: -x[1])

    puro = fermionic_select(items, vecs, 4, diversity=0.0)
    check("diversidade=0 devolve o ranking puro", puro == [i for i, _ in items[:4]], str(puro))

    div = fermionic_select(items, vecs, 4, diversity=0.5)
    grupos = {cid // 10 for cid in div}
    check("fermiônica cobre os 4 clusters (exclusão de Pauli)", len(grupos) == 4, str(div))
    check("fermiônica mantém o top-1 relevante", div[0] == items[0][0], str(div[0]))
    grupos_puro = {cid // 10 for cid in puro}
    check("ranking puro concentra num cluster só", len(grupos_puro) < 4, str(puro))

    # coerência: pico = decidido (~1), chapado = indeciso (0)
    check("coerência ~1 em distribuição de pico", coherence({1: 1.0, 2: 0.01, 3: 0.01}) > 0.9)
    check("coerência 0 em distribuição chapada", coherence({1: 1.0, 2: 1.0, 3: 1.0}) < 1e-9)


def test_memory_infinite():
    print("\n== memória de conversa (episódica + orçamento) ==")
    with tempfile.TemporaryDirectory() as td:
        rag = make_rag(Path(td), small_corpus_tokens=0, memory_budget_tokens=300)
        for i in range(30):
            rag.memory.record_turn(f"pergunta {i} sobre o tópico {i % 5}", f"resposta {i} detalhando o tópico {i % 5}")
        check("30 turnos gravados", rag.store.last_turn_no() == 30)

        ctx = rag.memory.build_context("me lembra do tópico 3?", rag.retriever)
        check("modo retrieval", ctx["mode"] == "retrieval", ctx["mode"])
        check("orçamento respeitado", ctx["tokens"] <= 300, str(ctx["tokens"]))
        check("recuperou blocos", len(ctx["blocks"]) > 0)
        check("blocos têm memory_score", all("memory_score" in b for b in ctx["blocks"]))

        # corpus pequeno -> contexto integral
        rag2 = make_rag(Path(td) / "b", small_corpus_tokens=100000)
        rag2.ingest("Fato único: o céu é azul.")
        ctx2 = rag2.memory.build_context("qual a cor do céu?", rag2.retriever)
        check("corpus pequeno -> integral", ctx2["mode"] == "corpus_integral", ctx2["mode"])


def test_persistence():
    print("\n== persistência ==")
    with tempfile.TemporaryDirectory() as td:
        rag = make_rag(Path(td), small_corpus_tokens=0)
        rag.ingest("O código do cofre é 7742.")
        del rag
        rag2 = make_rag(Path(td), small_corpus_tokens=0)
        r = rag2.search("código do cofre")
        check("reabre e acha", "7742" in r.fused[0]["text"], r.fused[0]["text"][:60])


if __name__ == "__main__":
    test_textproc()
    test_ingest_and_axes()
    test_adaptive_chunking()
    test_quantum_fusion_math()
    test_fermionic_and_coherence()
    test_memory_infinite()
    test_persistence()
    print(f"\n{'TUDO OK' if not FAILS else 'FALHAS: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)

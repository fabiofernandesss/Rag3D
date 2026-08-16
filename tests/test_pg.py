"""Suíte opt-in do backend Postgres holográfico (sem pgvector).

Requer um banco descartável cujo nome contenha ``test`` e dois guards:

  export RAG3D_TEST_PG_DSN=postgresql://postgres:rag3d@localhost:5433/rag3d_test
  export RAG3D_TEST_PG_ALLOW_DESTRUCTIVE=1

Exemplo de container:
  docker run -d --name rag3d-pg -e POSTGRES_PASSWORD=rag3d \
    -e POSTGRES_DB=rag3d_test -p 5433:5432 postgres:16
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rag3d.config import TriRagConfig
from rag3d.engine import TriRag
from rag3d.holo import HOLO_BITS, Holographer
from rag3d.llm import NoLLM

FAILS = []


def _test_dsn() -> str:
    """Return only the dedicated integration-test variable."""
    return os.environ.get("RAG3D_TEST_PG_DSN", "").strip()


def _database_name(dsn: str) -> str:
    try:
        from psycopg.conninfo import conninfo_to_dict

        return str(conninfo_to_dict(dsn).get("dbname", ""))
    except Exception:
        return ""


def _has_test_database_token(database: str) -> bool:
    return bool(re.search(r"(?:^|[_-])test(?:$|[_-])", database.casefold()))


def _safe_test_target() -> bool:
    dsn = _test_dsn()
    allowed = os.environ.get("RAG3D_TEST_PG_ALLOW_DESTRUCTIVE") == "1"
    database = _database_name(dsn).casefold()
    return bool(dsn and allowed and _has_test_database_token(database))


def _require_safe_test_target() -> str:
    if not _safe_test_target():
        raise RuntimeError(
            "refusing destructive PostgreSQL test: configure a test-only "
            "RAG3D_TEST_PG_DSN whose database name has a delimited 'test' token and set "
            "RAG3D_TEST_PG_ALLOW_DESTRUCTIVE=1"
        )
    return _test_dsn()


def _skip_unless_safe() -> None:
    if _safe_test_target():
        return
    try:
        import pytest
    except ImportError as exc:  # pragma: no cover - script mode checks in main
        raise RuntimeError("PostgreSQL integration target is not test-only") from exc
    pytest.skip(
        "requires test-only RAG3D_TEST_PG_DSN, database name with a delimited "
        "'test' token, and RAG3D_TEST_PG_ALLOW_DESTRUCTIVE=1"
    )


def check(name, cond, detail=""):
    status = "ok" if cond else "FALHOU"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def wipe(rag: TriRag):
    _require_safe_test_target()
    current_database = rag.store.db.execute("SELECT current_database()").fetchone()[0]
    if not _has_test_database_token(str(current_database)):
        raise RuntimeError("refusing destructive PostgreSQL test on a non-test database")
    with rag.store.db.cursor() as cur:
        cur.execute("TRUNCATE holo_grams, holo_spectrum, holo_docs, holo_meta")
    rag.store.db.commit()


def make_rag(**over) -> TriRag:
    dsn = _require_safe_test_target()
    cfg = TriRagConfig()
    cfg.data_dir = Path(tempfile.mkdtemp())
    cfg.encoder = "fallback"
    cfg.contextual_enrich = False
    cfg.pg_dsn = dsn
    cfg.small_corpus_tokens = 0
    for k, v in over.items():
        setattr(cfg, k, v)
    return TriRag(cfg, llm=NoLLM())


def test_holographer():
    print("\n== holograma: assinatura preserva vizinhança ==")
    h = Holographer(1024, 128)
    rng = np.random.default_rng(1)
    a = rng.standard_normal(1024).astype(np.float32)
    a /= np.linalg.norm(a)
    near = a + 0.15 * rng.standard_normal(1024).astype(np.float32)
    near /= np.linalg.norm(near)
    far = rng.standard_normal(1024).astype(np.float32)
    far /= np.linalg.norm(far)

    ham_near = h.hamming(h.sign_dense(a), h.sign_dense(near))
    ham_far = h.hamming(h.sign_dense(a), h.sign_dense(far))
    check("vizinho tem Hamming menor", ham_near < ham_far, f"{ham_near} vs {ham_far}")
    check("aleatório ~50% dos bits", abs(ham_far - HOLO_BITS / 2) < HOLO_BITS * 0.1, str(ham_far))

    q = h.dequantize(h.quantize(a))
    check("eco int8 fiel (cos>0.99)", float(q @ a) > 0.99, f"{float(q @ a):.4f}")

    words = h.sig_to_words(h.sign_dense(a))
    check("16 bigints", len(words) == 16)
    bands = h.bands_of(h.sign_dense(a))
    check("16 facetas sem colisão de banda", len(set(b >> 8 for b in bands)) == 16)


def test_pg_end_to_end():
    _skip_unless_safe()
    print("\n== Postgres puro: ingestão + 3 eixos + fusão ==")
    rag = make_rag()
    wipe(rag)
    rag.ingest("O contrato de aluguel vence em 15 de março de 2027 e custa R$ 3.500 mensais.", title="contrato")
    rag.ingest("The rocket launch is scheduled for July 2026 from Cape Canaveral.", title="rocket")
    rag.ingest("会议将于星期五上午十点在北京举行。", title="meeting")
    rag.ingest("A receita do bolo de fubá leva 3 ovos e erva-doce. Asse por 40 minutos.", title="bolo")

    r = rag.search("quando vence o contrato de aluguel?")
    check("3 visões", set(r.views.keys()) == {"semantico", "lexico", "estrutural"})
    check("acha contrato (PT)", "contrato" in r.fused[0]["text"], r.fused[0]["text"][:80])
    r2 = rag.search("rocket launch date")
    check("acha foguete (EN)", "rocket" in r2.fused[0]["text"].lower(), r2.fused[0]["text"][:80])
    r3 = rag.search("会议在哪里举行")
    check("acha reunião (中文)", "北京" in r3.fused[0]["text"], r3.fused[0]["text"][:60])
    check("interferência presente", "interference" in r.fused[0])


def test_pg_memory_and_persistence():
    _skip_unless_safe()
    print("\n== Postgres: memória + persistência entre conexões ==")
    rag = make_rag(memory_budget_tokens=300)
    wipe(rag)
    for i in range(20):
        rag.memory.record_turn(f"pergunta {i} sobre o tópico {i % 4}", f"resposta {i} sobre o tópico {i % 4}")
    check("20 turnos", rag.store.last_turn_no() == 20)
    ctx = rag.memory.build_context("tópico 2?", rag.retriever)
    check("retrieval + orçamento", ctx["mode"] == "retrieval" and ctx["tokens"] <= 300)
    used_ids = [b["id"] for b in ctx["blocks"]]
    if used_ids:
        row = rag.store.get_chunks(used_ids[:1])[0]
        check("acesso renovou recência", row["accessed_turn"] == 20, str(row["accessed_turn"]))

    rag2 = make_rag()  # nova conexão, mesmo banco
    r = rag2.search("tópico 2")
    check("persistiu entre conexões", len(r.fused) > 0)


def test_pg_scale_and_latency(n_docs: int = 400):
    _skip_unless_safe()
    print(f"\n== Postgres: escala ({n_docs} hologramas) + latência ==")
    rag = make_rag()
    wipe(rag)
    topics = ["finanças", "medicina", "futebol", "astronomia", "culinária", "direito"]
    t0 = time.time()
    for i in range(n_docs):
        t = topics[i % len(topics)]
        rag.ingest(f"Documento {i} sobre {t}: fato número {i} registrado no arquivo {i % 97}.", title=f"doc{i}")
    t_ing = time.time() - t0

    needle = n_docs // len(topics) - 2  # um i que existe no corpus gerado
    t0 = time.time()
    r = rag.search(f"fato número {needle}")
    t_q = time.time() - t0
    check(f"acha agulha exata em {n_docs}", any(f"número {needle} " in h["text"] for h in r.fused[:3]),
          r.fused[0]["text"][:60])
    check("consulta < 2s", t_q < 2.0, f"{t_q:.2f}s")
    print(f"  (ingestão {n_docs} docs: {t_ing:.1f}s | consulta: {t_q*1000:.0f}ms)")

    # força o caminho com pré-filtro de facetas (limiar baixo)
    rag.store.BAND_PREFILTER_THRESHOLD = 50
    t0 = time.time()
    r2 = rag.search(f"fato número {needle}")
    t_b = time.time() - t0
    check("facetas: ainda acha", any(f"número {needle} " in h["text"] for h in r2.fused[:5]),
          r2.fused[0]["text"][:60] if r2.fused else "vazio")
    print(f"  (consulta com pré-filtro de facetas: {t_b*1000:.0f}ms)")


if __name__ == "__main__":
    test_holographer()
    if not _safe_test_target():
        print(
            "\nPostgreSQL de teste não autorizado — defina "
            "RAG3D_TEST_PG_DSN para um banco com 'test' no nome e "
            "RAG3D_TEST_PG_ALLOW_DESTRUCTIVE=1."
        )
        sys.exit(2)
    dsn = _require_safe_test_target()
    try:
        import psycopg
        psycopg.connect(dsn, connect_timeout=3).close()
    except Exception as e:
        print(f"\nPostgres indisponível ({e}) — suba o container e rode de novo.")
        sys.exit(2)
    test_pg_end_to_end()
    test_pg_memory_and_persistence()
    test_pg_scale_and_latency()
    print(f"\n{'TUDO OK' if not FAILS else 'FALHAS: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)

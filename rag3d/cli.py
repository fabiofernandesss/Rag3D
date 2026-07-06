"""CLI do RAG3D.

  python3 -m rag3d ingest caminho/ou/arquivo [...]
  python3 -m rag3d ingest --text "qualquer texto"
  python3 -m rag3d ask "pergunta" [--tri] [--rrf] [--k 8]
  python3 -m rag3d search "consulta"            # só retrieval, mostra os 3 eixos
  python3 -m rag3d chat                          # conversa com memória infinita
  python3 -m rag3d stats
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import TriRagConfig
from .engine import TriRag


def _mk(args) -> TriRag:
    cfg = TriRagConfig()
    if getattr(args, "data", None):
        cfg.data_dir = Path(args.data).expanduser()
    if getattr(args, "pg", None):
        cfg.pg_dsn = args.pg
    if getattr(args, "rrf", False):
        cfg.fusion = "rrf"
    if getattr(args, "k", None):
        cfg.top_k = args.k
    if getattr(args, "tri", False):
        cfg.read_mode = "tri"
    if getattr(args, "rerank", False):
        cfg.rerank = True
    return TriRag(cfg)


def cmd_ingest(args) -> None:
    rag = _mk(args)
    if args.text:
        r = rag.ingest(args.text)
        print(f"ok: doc {r['doc_id']}, {r['chunks']} chunks, ~{r.get('tokens', 0)} tokens")
        return
    for p in args.paths:
        path = Path(p).expanduser()
        files = sorted(x for x in path.rglob("*") if x.is_file()) if path.is_dir() else [path]
        for f in files:
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".db", ".npy", ".pyc"}:
                continue
            try:
                r = rag.ingest_file(f)
                print(f"ok: {f.name} -> {r['chunks']} chunks (~{r.get('tokens', 0)} tokens)")
            except Exception as e:
                print(f"erro: {f}: {e}", file=sys.stderr)


def cmd_search(args) -> None:
    rag = _mk(args)
    r = rag.search(args.query)
    print(f"# consulta: {r.query}\n# {json.dumps(r.stats, ensure_ascii=False)}\n")
    for axis, rows in r.views.items():
        print(f"--- eixo {axis} ---")
        for row in rows[:5]:
            print(f"  [{row['id']}] {row['score']:.4f}  {row['text'][:90]!r}")
    print("\n=== fusão ({}) ===".format(rag.cfg.fusion))
    for row in r.fused:
        extra = f" interf={row['interference']:+.3f}" if "interference" in row else ""
        print(f"  [{row['id']}] {row['score']:.4f}{extra} eixos={','.join(row['channels'])}")
        print(f"      {row['text'][:120]!r}")


def cmd_ask(args) -> None:
    rag = _mk(args)
    out = rag.ask(args.query)
    _print_answer(out)


def cmd_chat(args) -> None:
    rag = _mk(args)
    print(f"RAG3D chat — {rag.stats()['llm']} | encoder {rag.encoder.name} | fusão {rag.cfg.fusion}")
    print("(vazio para sair)\n")
    while True:
        try:
            msg = input("você> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            break
        out = rag.chat(msg)
        _print_answer(out, prefix="yah> ")


def _print_answer(out: dict, prefix: str = "") -> None:
    ctx = out.get("context", {})
    mode = ctx.get("mode", "?")
    if out.get("sub_answers"):
        for axis, ans in out["sub_answers"].items():
            print(f"  ({axis}) {ans.strip()[:200]}")
        print()
    if out.get("answer"):
        print(f"{prefix}{out['answer'].strip()}\n")
    else:
        print(f"[{out.get('note', 'sem resposta')}] — evidências ({mode}):")
        for b in (ctx.get("blocks") or [])[:8]:
            t = b["chosen"] if isinstance(b, dict) and "chosen" in b else (b if isinstance(b, str) else b.get("text", ""))
            print(f"  - {t[:150]!r}")


def cmd_stats(args) -> None:
    print(json.dumps(_mk(args).stats(), indent=2, ensure_ascii=False))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="rag3d", description="RAG3D — RAG tridimensional com fusão quântica (sem pgvector)")
    # opções globais também aceitas DEPOIS do subcomando (posição convencional):
    # herdadas por cada subparser via `parents=[glob]`.
    # default=SUPPRESS: subparser não sobrescreve com None o valor já lido pelo
    # parser raiz (gotcha clássico do argparse quando a opção existe nos dois).
    glob = argparse.ArgumentParser(add_help=False)
    glob.add_argument("--data", default=argparse.SUPPRESS,
                      help="diretório de dados (padrão ~/.rag3d ou $RAG3D_DATA)")
    glob.add_argument("--pg", default=argparse.SUPPRESS,
                      help="DSN Postgres SEM pgvector (ou $TRIRAG_PG); vazio = SQLite local")
    ap.add_argument("--data", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    ap.add_argument("--pg", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", parents=[glob], help="ingerir arquivos/diretórios ou texto")
    p.add_argument("paths", nargs="*", help="arquivos ou diretórios")
    p.add_argument("--text", help="ingerir texto direto")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("search", parents=[glob], help="só retrieval: mostra os 3 eixos e a fusão")
    p.add_argument("query")
    p.add_argument("--rrf", action="store_true", help="usar RRF em vez da fusão quântica")
    p.add_argument("--rerank", action="store_true", help="reranker LLM sobre o topo fundido")
    p.add_argument("--k", type=int)
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("ask", parents=[glob], help="pergunta avulsa")
    p.add_argument("query")
    p.add_argument("--tri", action="store_true", help="3 leitores por eixo + leitora final")
    p.add_argument("--rerank", action="store_true", help="reranker LLM sobre o topo fundido")
    p.add_argument("--rrf", action="store_true")
    p.add_argument("--k", type=int)
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("chat", parents=[glob], help="conversa com memória infinita")
    p.add_argument("--tri", action="store_true")
    p.add_argument("--rerank", action="store_true", help="reranker LLM sobre o topo fundido")
    p.add_argument("--rrf", action="store_true")
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("stats", parents=[glob], help="estado do índice")
    p.set_defaults(fn=cmd_stats)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

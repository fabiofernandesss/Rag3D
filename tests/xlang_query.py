"""Cross-language: lado Python CONSULTA o Postgres (para provar JS->PY)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag3d.config import TriRagConfig
from rag3d.engine import TriRag
from rag3d.llm import NoLLM

DSN = "postgresql://postgres:rag3d@localhost:5433/rag3d"


def main():
    cfg = TriRagConfig()
    cfg.pg_dsn = DSN
    cfg.encoder = "fallback"
    cfg.contextual_enrich = False
    cfg.small_corpus_tokens = 0
    rag = TriRag(cfg, llm=NoLLM())
    query = sys.argv[1] if len(sys.argv) > 1 else "identidade visual da marca"
    r = rag.search(query, top_k=3)
    top = r.fused[0]["text"] if r.fused else "(vazio)"
    print(f"PY consulta {query!r} -> {top[:70]!r}")


if __name__ == "__main__":
    main()

"""Cross-language: lado Python INGERE no Postgres (backend holográfico).

Roda com o encoder fallback portável para casar bit a bit com o JS.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag3d.config import TriRagConfig
from rag3d.engine import TriRag
from rag3d.llm import NoLLM

DSN = "postgresql://postgres:rag3d@localhost:5433/rag3d"

DOCS = [
    ("contrato", "O contrato de aluguel vence em 15 de março de 2027 e custa R$ 3.500 por mês."),
    ("foguete", "The Artemis rocket launch is scheduled for 12 July 2026 from Cape Canaveral."),
    ("reuniao", "会议将于星期五上午十点在北京的数据中心举行，讨论季度业绩。"),
    ("bolo", "A receita do bolo de fubá cremoso leva 3 ovos, fubá, leite e erva-doce."),
    ("cofre", "O código de acesso ao cofre principal é 7742 e deve ser trocado todo mês."),
    ("medico", "Le patient présente une tension artérielle élevée; prescrire un régime pauvre en sel."),
]


def main():
    cfg = TriRagConfig()
    cfg.pg_dsn = DSN
    cfg.encoder = "fallback"
    cfg.contextual_enrich = False
    cfg.small_corpus_tokens = 0
    rag = TriRag(cfg, llm=NoLLM())
    for title, text in DOCS:
        rag.ingest(text, title=title)
    print(f"PY ingeriu {len(DOCS)} docs no Postgres (encoder {rag.encoder.name})")


if __name__ == "__main__":
    main()

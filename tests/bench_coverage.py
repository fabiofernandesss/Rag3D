"""Benchmark de COBERTURA: o top-k contém os fatos distintos que a resposta exige?

Mede a falha mais cara de RAG na prática: o ranking enche o contexto de
quase-duplicatas do mesmo trecho e os outros fatos necessários ficam de fora.
Se o fato não entra no contexto, a IA não tem como responder certo — por isso
cobertura é a métrica de "assertividade" que importa para o leitor final.

Corpus sintético: cada tópico tem 3 FATOS distintos (em 3 documentos) e várias
PARÁFRASES quase idênticas do fato 1 (a armadilha de redundância). A consulta
é genérica sobre o tópico.

  python3 tests/bench_coverage.py [n_topicos] [n_paráfrases]

Compara a seleção fermiônica (MAP-DPP / determinante de Slater) em vários
níveis de diversidade contra o ranking puro, e verifica que o topo não é
sacrificado (rank-1 continua relevante).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag3d.config import TriRagConfig
from rag3d.engine import TriRag
from rag3d.llm import NoLLM

_TEMAS = [
    "contrato de locação comercial", "obra de pavimentação urbana",
    "bolsa de pesquisa científica", "licença ambiental simplificada",
    "convênio de cooperação técnica", "programa de estágio supervisionado",
    "seguro de responsabilidade civil", "plano de manejo florestal",
    "termo de outorga de premiação", "chamamento público de inovação",
    "contrato de manutenção predial", "auxílio moradia estudantil",
    "concessão de uso de espaço público", "registro de preços para insumos",
    "credenciamento de prestadores de serviço", "termo de fomento cultural",
    "parceria público-privada de saneamento", "edital de residência artística",
    "convocação para avaliação técnica", "acordo de nível de serviço",
]
# 3 fatos distintos por tema — a resposta completa precisa dos três
_MOLDES = [
    "O prazo de vigência de {t} é de {a} meses, contados da assinatura.",
    "A multa por descumprimento em {t} corresponde a {b}% do valor total.",
    "O reajuste anual de {t} segue o índice oficial publicado em janeiro.",
]
# duplicatas do FATO 1 — é o que sobreposição de chunks e costura produzem:
# textos quase idênticos que ranqueiam igual e entopem o top-k
_DUPS = [
    "{f}",
    "Conforme o item anterior, {f}",
    "{f} (repetido no anexo)",
    "Nota: {f}",
    "{f} Ver seção correspondente.",
    "Errata — {f}",
    "Trecho reproduzido: {f}",
    "{f} Sem alteração.",
]


def _topicos(n: int):
    out = []
    for i, tema in enumerate(_TEMAS[:n]):
        fatos = [m.format(t=tema, a=12 + i, b=5 + i) for m in _MOLDES]
        out.append((f"t{i}", tema, fatos))
    return out


TOPICOS = _topicos(len(_TEMAS))


def build(n_topicos: int, n_paraf: int, **cfg_over):
    cfg = TriRagConfig()
    cfg.data_dir = Path(tempfile.mkdtemp())
    cfg.encoder = "fallback"
    cfg.contextual_enrich = False
    cfg.small_corpus_tokens = 0
    cfg.stitch_radius = 0
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    rag = TriRag(cfg, llm=NoLLM())

    gabarito = {}  # topico -> {fato_idx: set(textos)}
    for slug, tema, fatos in TOPICOS[:n_topicos]:
        gabarito[slug] = {}
        for i, fato in enumerate(fatos):
            rag.ingest(fato, title=f"{slug}_fato{i}")
            gabarito[slug][i] = fato
        for j in range(n_paraf):  # duplicatas quase idênticas do fato 1
            rag.ingest(_DUPS[j % len(_DUPS)].format(f=fatos[0]), title=f"{slug}_dup{j}")
    return rag, gabarito


def avalia(rag: TriRag, gabarito: dict, n_topicos: int, k: int = 6):
    """cobertura = fatos distintos presentes no top-k (a informação está lá,
    mesmo que dentro de uma duplicata); rank-1 = topo é do tópico certo."""
    cobertura = top1_ok = n = 0
    for slug, tema, fatos in TOPICOS[:n_topicos]:
        r = rag.search(f"informações sobre {tema}", top_k=k)
        textos = [h["text"] for h in r.fused[:k]]
        achados = {i for i, f in gabarito[slug].items() if any(f in t for t in textos)}
        cobertura += len(achados) / 3.0
        if textos and tema in textos[0]:
            top1_ok += 1
        n += 1
    return cobertura / n, top1_ok / n


def main():
    n_top = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    n_par = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    print(f"corpus: {n_top} tópicos × (3 fatos + {n_par} paráfrases do fato 1)\n")
    print(f"{'configuração':<34} {'cobertura@6':>12} {'rank-1 útil':>12}")
    print("-" * 60)
    for rotulo, over in [
        ("ranking puro (diversidade=0)", dict(diversity=0.0)),
        ("fermiônica 0.3", dict(diversity=0.3)),
        ("fermiônica 0.5", dict(diversity=0.5)),
        ("fermiônica 0.7", dict(diversity=0.7)),
        ("fermiônica 0.5 + coerência 0.5", dict(diversity=0.5, coherence_strength=0.5)),
        ("RRF puro", dict(diversity=0.0, fusion="rrf")),
        ("RRF + fermiônica 0.5", dict(diversity=0.5, fusion="rrf")),
    ]:
        rag, gab = build(n_top, n_par, **over)
        cob, t1 = avalia(rag, gab, n_top)
        print(f"{rotulo:<34} {cob:>11.1%} {t1:>11.1%}")


if __name__ == "__main__":
    main()

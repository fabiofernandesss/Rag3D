"""Camada de leitura — onde a IA lê o tridimensional.

Dois modos:

  fast : um único LLM recebe o contexto fundido + as três visões e responde.

  tri  : TRÊS leitores (um por eixo — semântico, léxico, estrutural), cada um
         responde só com a visão do seu eixo; depois o LEITOR FINAL recebe as
         três respostas e as evidências fundidas e faz a leitura definitiva.
         (É o fluxo pedido: as IAs leem as três respostas que vêm do
         tridimensional, e depois a leitora final sintetiza.)

Qualquer LLM serve para qualquer papel; dá para usar IAs diferentes por eixo.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from .config import TriRagConfig
from .llm import LLM, validate_llm_text

_AXIS_LABEL = {
    "semantico": "EIXO SEMÂNTICO (significado)",
    "lexico": "EIXO LÉXICO (termos exatos)",
    "estrutural": "EIXO ESTRUTURAL (correspondência fina)",
}

# REGRAS DE GROUNDING ESTRITO — para documentos normativos (editais, leis,
# contratos): só afirmar o que está TEXTUALMENTE nas evidências; na dúvida,
# dizer que não há base. Nunca inferir a partir da ausência.
_GROUNDING = (
    "REGRAS (documento normativo — siga à risca):\n"
    "1. Responda SOMENTE com o que está textualmente escrito nas EVIDÊNCIAS. Não use conhecimento externo.\n"
    "2. Se as evidências não contêm a resposta, diga exatamente: \"O documento não fornece base suficiente para responder.\" e PARE. É melhor não responder do que inferir.\n"
    "3. PROIBIDO inferir, deduzir ou completar. PROIBIDAS as expressões: \"pela lógica\", \"implicitamente\", \"é possível concluir\", \"infere-se\", \"inerente\", \"portanto obrigatório\".\n"
    "4. Concordância entre eixos/recuperadores NÃO é evidência. Se nenhum trecho apoia a afirmação, responda que não há base — mesmo que as três análises 'apontem' para algo.\n"
    "5. Ao listar itens (leis, critérios, requisitos, normas), liste TODOS os que aparecem nas evidências, sem omitir nem resumir. Em listas NUMERADAS (ex.: 3.3.1, 3.3.2, ...; a), b), c)), confira a sequência e inclua CADA item — nunca pule um número/letra intermediário. Números de lei (ex.: 13.243, 14.133) são itens da lista, não ruído; inclua-os. Se houver indício de que a lista continua fora do trecho, diga isso.\n"
    "6. Distinga PRINCÍPIO de OBRIGAÇÃO universal. Um \"princípio orientador\" ou uma \"característica\" que aparece na DEFINIÇÃO de um modelo NÃO significa que TODA solução é obrigada a exibir esse traço — só que quando aplica esse modelo o princípio vale. Respeite \"e/ou\": \"A ou B\" não vira \"A obrigatório\". Um item listado como PRIORIDADE (\"prioriza propostas de...\") não é requisito universal, é critério de preferência. Distinga DEFINIÇÃO conceitual de trecho operacional.\n"
    "7. Cite os trechos por [n]. Não mencione 'fusão', 'eixos' ou o método como justificativa — justifique só pelo texto citado.\n"
    "Responda na língua da pergunta."
)

_SYS_READER = "Você responde perguntas sobre um documento.\n" + _GROUNDING

_SYS_AXIS = (
    "Você é o leitor do {axis}. Use SOMENTE os trechos da sua visão. Direto "
    "(máx ~8 frases). Se sua visão não contém a resposta, diga apenas 'meu eixo "
    "não encontrou' — NÃO tente deduzir.\n" + _GROUNDING
)

_SYS_FINAL = (
    "Você é a leitora final. Três leitores buscaram por três eixos e você vê as "
    "três respostas e as EVIDÊNCIAS (trechos). REGRA CENTRAL: se QUALQUER um dos "
    "três leitores respondeu citando um trecho literal (com [n] ou entre aspas), "
    "essa resposta É fundamentada — USE-A, não descarte. Basta UM eixo ter achado. "
    "Só responda 'não há base suficiente' quando os TRÊS leitores disseram que não "
    "encontraram E nenhum trecho apoia. DESCARTE apenas conclusões que um leitor "
    "tirou SEM citar trecho (dedução). Nunca troque uma citação real por 'não há base'.\n"
    + _GROUNDING
)


def _fmt_blocks(blocks: List, max_chars: int = 6500) -> str:
    out = []
    for i, b in enumerate(blocks, 1):
        text = b["chosen"] if isinstance(b, dict) and "chosen" in b else (
            b["text"] if isinstance(b, dict) else str(b)
        )
        out.append(f"[{i}] {text[:max_chars]}")
    return "\n\n".join(out) if out else "(nenhuma evidência)"


class Reader:
    def __init__(self, cfg: TriRagConfig, llm: LLM, axis_llms: Optional[Dict[str, LLM]] = None):
        self.cfg = cfg
        self.llm = llm                       # leitora final ("fast" também usa esta)
        self.axis_llms = axis_llms or {}     # opcional: uma IA diferente por eixo

    # ------------------------------------------------------------------ fast

    def read_fast(self, query: str, context: dict) -> dict:
        parts: List[str] = []
        if context.get("summary"):
            parts.append(f"MEMÓRIA DA CONVERSA:\n{context['summary']}")
        if context.get("recent"):
            parts.append("TURNOS RECENTES:\n" + "\n".join(context["recent"]))
        if context["mode"] == "corpus_integral":
            parts.append("CONTEÚDO COMPLETO (corpus pequeno, sem retrieval):\n" + _fmt_blocks(context["blocks"]))
        else:
            parts.append("EVIDÊNCIAS (fusão dos 3 eixos):\n" + _fmt_blocks(context["blocks"]))
            views = context.get("views") or {}
            for axis, rows in views.items():
                extra = [r for r in rows[:3]]
                if extra:
                    parts.append(f"{_AXIS_LABEL.get(axis, axis)} — top 3:\n" + _fmt_blocks(extra, 600))
        prompt = "\n\n".join(parts) + f"\n\nPERGUNTA: {query}"
        answer = validate_llm_text(
            self.llm.complete(
                _SYS_READER,
                [{"role": "user", "content": prompt}],
                max_tokens=self.cfg.max_answer_tokens,
            )
        )
        return {"answer": answer, "read_mode": "fast", "sub_answers": None}

    # ------------------------------------------------------------------- tri

    def read_tri(self, query: str, context: dict) -> dict:
        if context["mode"] == "corpus_integral" or not context.get("views"):
            return self.read_fast(query, context)

        views: Dict[str, List[dict]] = context["views"]

        def ask_axis(axis: str) -> str:
            llm = self.axis_llms.get(axis, self.llm)
            body = _fmt_blocks(views.get(axis, []), 3000)
            prompt = f"VISÃO DO SEU EIXO:\n{body}\n\nPERGUNTA: {query}"
            try:
                return validate_llm_text(
                    llm.complete(
                        _SYS_AXIS.format(axis=_AXIS_LABEL.get(axis, axis)),
                        [{"role": "user", "content": prompt}],
                        max_tokens=500,
                    )
                )
            except Exception:
                return "(leitor do eixo falhou)"

        axes = list(views.keys())
        with ThreadPoolExecutor(max_workers=3) as ex:
            sub = dict(zip(axes, ex.map(ask_axis, axes)))

        parts = []
        if context.get("summary"):
            parts.append(f"MEMÓRIA DA CONVERSA:\n{context['summary']}")
        if context.get("recent"):
            parts.append("TURNOS RECENTES:\n" + "\n".join(context["recent"]))
        for axis, ans in sub.items():
            parts.append(f"RESPOSTA DO {_AXIS_LABEL.get(axis, axis)}:\n{ans.strip()}")
        parts.append("EVIDÊNCIAS MAIS FORTES (fusão quântica dos eixos):\n" + _fmt_blocks(context["blocks"], 6500))
        prompt = "\n\n".join(parts) + f"\n\nPERGUNTA ORIGINAL: {query}\n\nFaça a leitura final."

        final = validate_llm_text(
            self.llm.complete(
                _SYS_FINAL,
                [{"role": "user", "content": prompt}],
                max_tokens=self.cfg.max_answer_tokens,
            )
        )
        return {"answer": final, "read_mode": "tri", "sub_answers": sub}

    # ------------------------------------------------------------------ API

    def read(self, query: str, context: dict, mode: Optional[str] = None) -> dict:
        mode = mode or self.cfg.read_mode
        if not self.llm.available():
            return {
                "answer": None,
                "read_mode": "retrieval_only",
                "sub_answers": None,
                "note": "Sem LLM configurado: devolvendo só as evidências.",
            }
        return self.read_tri(query, context) if mode == "tri" else self.read_fast(query, context)

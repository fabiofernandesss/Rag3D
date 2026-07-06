"""Reranker — o maior ganho isolado depois do retrieval híbrido.

A pesquisa 2024-2026 é clara: um reranker sobre os candidatos fundidos rende
+5-15 nDCG@10, mais que qualquer outro passo. Rerankers clássicos são
cross-encoders (bge-reranker-v2-m3), que exigem um modelo dedicado. Aqui o
reranker é **agnóstico de IA**: usa o mesmo LLM da leitura, num prompt
listwise — recebe a consulta e os candidatos, devolve a ordem de relevância.

Barato: uma chamada, só sobre os top-N já fundidos. Cai de volta na ordem
original se o LLM falhar ou a resposta não fizer sentido — nunca piora o
retrieval, no pior caso mantém.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from .llm import LLM

_SYS = (
    "Você é um reordenador de relevância. Recebe uma pergunta e trechos "
    "numerados. Responda APENAS um array JSON com os números dos trechos, do "
    "mais relevante para o menos relevante, incluindo só os que ajudam a "
    "responder. Exemplo: [3, 1, 5]. Sem texto fora do array."
)


class Reranker:
    def __init__(self, llm: LLM, snippet_chars: int = 500):
        self.llm = llm
        self.snippet_chars = snippet_chars

    def available(self) -> bool:
        return self.llm is not None and self.llm.available()

    def rerank(self, query: str, hits: List[dict], top_k: Optional[int] = None) -> List[dict]:
        if not self.available() or len(hits) < 2:
            return hits[:top_k] if top_k else hits

        listing = "\n".join(
            f"[{i}] {h.get('text', '')[:self.snippet_chars]}" for i, h in enumerate(hits)
        )
        prompt = f"PERGUNTA: {query}\n\nTRECHOS:\n{listing}\n\nOrdem de relevância (array JSON):"
        try:
            raw = self.llm.complete(_SYS, [{"role": "user", "content": prompt}], max_tokens=200)
            order = self._parse(raw, len(hits))
        except Exception:
            return hits[:top_k] if top_k else hits

        if not order:
            return hits[:top_k] if top_k else hits

        reordered = [hits[i] for i in order]
        seen = set(order)
        reordered.extend(h for i, h in enumerate(hits) if i not in seen)  # não descarta ninguém
        for rank, h in enumerate(reordered):
            h["rerank"] = rank
        return reordered[:top_k] if top_k else reordered

    @staticmethod
    def _parse(raw: str, n: int) -> List[int]:
        """Extrai a lista de índices válidos, robusto a lixo em volta."""
        m = re.search(r"\[[\d,\s]*\]", raw)
        if not m:
            nums = re.findall(r"\d+", raw)
        else:
            try:
                nums = json.loads(m.group())
            except Exception:
                nums = re.findall(r"\d+", m.group())
        out: List[int] = []
        for x in nums:
            i = int(x)
            if 0 <= i < n and i not in out:
                out.append(i)
        return out

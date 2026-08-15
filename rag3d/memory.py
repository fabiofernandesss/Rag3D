"""Memória de conversa "infinita".

Arquitetura (MemGPT/Letta + agentes generativos de Stanford, simplificada):

  memória de trabalho : resumo rolante mantido por LLM (sempre no prompt)
  memória episódica   : cada turno vira um chunk tridimensional pesquisável
  memória semântica   : documentos ingeridos + resumos consolidados

Pontuação de recuperação de memória (fórmula dos agentes generativos):
  score = w_rel * relevância_tri + w_rec * recência + w_imp * importância
  recência = 0.5 ** (turnos_atrás / meia_vida)

Montagem de contexto com orçamento:
  corpus pequeno  -> entra INTEIRO no prompt (RAG desnecessário; crossover
                     long-context vs retrieval)
  corpus grande   -> resumo rolante + últimos turnos literais + memórias
                     recuperadas até caber no orçamento
"""
from __future__ import annotations

import json
import time
from typing import List, Optional

from .config import TriRagConfig
from .encoders import BaseEncoder
from .llm import LLM, validate_llm_text
from .retrieve import TriResult, TriRetriever
from .store import TriStore
from .textproc import estimate_tokens

_SUMMARY_PROMPT = (
    "Resumo atual da conversa:\n<resumo>\n{summary}\n</resumo>\n\n"
    "Novos turnos:\n<turnos>\n{turns}\n</turnos>\n\n"
    "Atualize o resumo: preserve fatos, decisões, nomes, números e preferências "
    "do usuário; descarte conversa fiada. Máximo ~15 frases, na língua da conversa. "
    "Responda só o resumo."
)


class ChatMemory:
    def __init__(self, store: TriStore, encoder: BaseEncoder, cfg: TriRagConfig, llm: Optional[LLM] = None):
        self.store = store
        self.encoder = encoder
        self.cfg = cfg
        self.llm = llm

    # ------------------------------------------------------------ gravação

    def record_turn(self, user_msg: str, assistant_msg: str) -> int:
        """Salva o turno como memória episódica tridimensional."""
        user_msg = validate_llm_text(user_msg)
        assistant_msg = validate_llm_text(assistant_msg)
        turn_no = self.store.last_turn_no() + 1
        text = f"[usuário] {user_msg}\n[assistente] {assistant_msg}"
        n_tok = estimate_tokens(text)
        importance = min(1.0, 0.35 + n_tok / 1500.0)  # heurística barata
        vec = self.encoder.encode([text])[0]
        self.store.add_chunk(
            None, text, text, n_tok, vec, kind="turn", turn_no=turn_no, importance=importance
        )
        self.store.commit()
        if turn_no % self.cfg.summary_every_turns == 0:
            self._consolidate(turn_no)
        return turn_no

    def _consolidate(self, upto_turn: int) -> None:
        """Funde os últimos turnos no resumo rolante (memória de trabalho)."""
        if self.llm is None or not self.llm.available():
            return
        old = self.store.get_meta("rolling_summary") or "(vazio)"
        turns = [
            t for t in self.store.all_texts(kinds=("turn",))
            if (t["turn_no"] or 0) > upto_turn - self.cfg.summary_every_turns
        ]
        body = "\n---\n".join(t["text"][:1200] for t in turns)
        try:
            raw = self.llm.complete(
                "Você mantém a memória de longo prazo de uma conversa.",
                [{"role": "user", "content": _SUMMARY_PROMPT.format(summary=old, turns=body)}],
                max_tokens=700,
            )
            new = validate_llm_text(raw).strip()
        except Exception:
            return
        # Slow/model work and vector allocation stay outside the write
        # transaction.  Publication of metadata and the searchable summary is
        # one atomic unit across SQLite, postgres-holo, and pgvector.
        vec = self.encoder.encode([f"[resumo da conversa] {new}"])[0]
        token_count = estimate_tokens(new)
        with self.store.transaction():
            old_id = self.store.get_meta("rolling_summary_chunk")
            self.store.set_meta("rolling_summary", new)
            self.store.set_meta("rolling_summary_turn", str(upto_turn))
            if old_id:
                self.store.delete_chunk(int(old_id))
            cid = self.store.add_chunk(
                None,
                new,
                f"[resumo da conversa] {new}",
                token_count,
                vec,
                kind="summary",
                importance=0.9,
            )
            self.store.set_meta("rolling_summary_chunk", str(cid))

    # ----------------------------------------------------------- contexto

    def _memory_rescore(self, result: TriResult) -> List[dict]:
        """Relevância tri-fundida + recência + importância (agentes generativos)."""
        now_turn = self.store.last_turn_no()
        hits = result.fused
        if not hits:
            return []
        smax = max(h["score"] for h in hits) or 1.0
        for h in hits:
            rel = h["score"] / smax if smax > 0 else 0.0
            if h["kind"] == "turn" and h["turn_no"] is not None:
                # recência conta do último USO (acesso renova a memória, como
                # na fórmula dos agentes generativos de Stanford)
                ref = max(h["turn_no"], h.get("accessed_turn") or 0)
                dist = max(0, now_turn - ref)
                rec = 0.5 ** (dist / self.cfg.recency_half_life_turns)
            else:
                rec = 0.5  # documentos não envelhecem como turnos
            h["memory_score"] = (
                self.cfg.w_relevance * rel
                + self.cfg.w_recency * rec
                + self.cfg.w_importance * float(h["importance"] or 0.5)
            )
        hits.sort(key=lambda h: -h["memory_score"])
        return hits

    def build_context(self, query: str, retriever: TriRetriever, recent_turns: int = 4) -> dict:
        """Decide o que entra no prompt. Devolve também as três visões."""
        total = self.store.corpus_tokens(kinds=("chunk", "turn"))

        # --- caso pequeno: contexto integral, sem retrieval -------------
        if total <= self.cfg.small_corpus_tokens:
            texts = self.store.all_texts(kinds=("chunk", "turn"))
            return {
                "mode": "corpus_integral",
                "summary": self.store.get_meta("rolling_summary"),
                "blocks": [t["text"] for t in texts],
                "views": None,
                "tokens": total,
            }

        # --- caso grande: resumo + turnos recentes + memórias tri -------
        result = retriever.search(query)
        hits = self._memory_rescore(result)

        last = self.store.all_texts(kinds=("turn",))[-recent_turns:]
        recent_ids = {t["id"] for t in last}

        budget = self.cfg.memory_budget_tokens
        blocks: List[dict] = []
        used = 0
        for h in hits:
            if h["id"] in recent_ids:
                continue  # já entra literal
            t = estimate_tokens(h["wide"])
            if used + t > budget:
                # tenta a versão estreita antes de desistir
                t2 = estimate_tokens(h["text"])
                if used + t2 > budget:
                    continue
                blocks.append({**h, "chosen": h["text"]})
                used += t2
            else:
                blocks.append({**h, "chosen": h["wide"]})
                used += t
            if used >= budget:
                break

        # acesso renova a recência das memórias usadas
        self.store.touch_access([b["id"] for b in blocks], self.store.last_turn_no())

        return {
            "mode": "retrieval",
            "summary": self.store.get_meta("rolling_summary"),
            "recent": [t["text"] for t in last],
            "blocks": blocks,
            "views": result.views,
            "stats": result.stats,
            "tokens": used,
        }

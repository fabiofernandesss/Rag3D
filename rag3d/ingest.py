"""Ingestão adaptativa — as três formas nascem na hora de salvar.

Estratégia por tamanho:
  minúsculo (<= tiny_doc_tokens) : salvo inteiro, um chunk só
  normal                         : chunks por sentença (agnóstico de língua)
                                   + nós "pai" (small-to-big)
  gigante  (>= huge_doc_tokens)  : tudo acima + nó de resumo do documento
                                   (RAPTOR-lite) quando há LLM

Enriquecimento contextual (Anthropic Contextual Retrieval): quando há LLM,
cada chunk ganha 1-2 frases que o situam no documento ANTES de ser
embutido — o texto embutido é `contexto + chunk`, o texto devolvido ao
leitor é o original.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .config import TriRagConfig
from .encoders import BaseEncoder
from .llm import LLM
from .store import TriStore
from .textproc import estimate_tokens, normalize, split_sentences


def _digest(text: str, budget: int = 20000) -> str:
    """Digesto cobrindo o documento INTEIRO (janelas espaçadas) — para o resumo
    de doc gigante refletir todas as páginas, não só o começo, com 1 chamada LLM."""
    if len(text) <= budget:
        return text
    parts, out = 8, []
    win, step = budget // parts, len(text) // parts
    for i in range(parts):
        out.append(text[i * step : i * step + win])
    return "\n[...]\n".join(out)

_ENRICH_PROMPT = (
    "Documento (início):\n<doc>\n{doc}\n</doc>\n\n"
    "Trecho:\n<chunk>\n{chunk}\n</chunk>\n\n"
    "Escreva 1-2 frases curtas, na língua do documento, situando este trecho "
    "dentro do documento (do que trata, a que parte pertence). Responda só as frases."
)


def chunk_text(text: str, target_tokens: int, overlap_tokens: int) -> List[str]:
    """Agrupa sentenças até ~target_tokens, com sobreposição de ~overlap_tokens."""
    sents = split_sentences(text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    for s in sents:
        st = estimate_tokens(s)
        if cur and cur_tok + st > target_tokens:
            chunks.append(" ".join(cur))
            # sobreposição: mantém sentenças do fim
            keep: List[str] = []
            kept = 0
            for prev in reversed(cur):
                pt = estimate_tokens(prev)
                if kept + pt > overlap_tokens:
                    break
                keep.insert(0, prev)
                kept += pt
            cur, cur_tok = keep, kept
        cur.append(s)
        cur_tok += st
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]


class Ingestor:
    def __init__(self, store: TriStore, encoder: BaseEncoder, cfg: TriRagConfig, llm: Optional[LLM] = None):
        self.store = store
        self.encoder = encoder
        self.cfg = cfg
        self.llm = llm

    # ------------------------------------------------------------------ API

    def ingest_text(self, text: str, source: str = "inline", title: str = "") -> dict:
        text = normalize(text)
        if not text:
            return {"doc_id": None, "chunks": 0}
        n_tok = estimate_tokens(text)
        title = title or (text[:60].replace("\n", " ") + ("…" if len(text) > 60 else ""))
        doc_id = self.store.add_doc(source, title, n_tok)

        if n_tok <= self.cfg.tiny_doc_tokens:
            n = self._ingest_tiny(doc_id, title, text, n_tok)
        else:
            n = self._ingest_chunked(doc_id, title, text, n_tok)
        self.store.commit()
        return {"doc_id": doc_id, "chunks": n, "tokens": n_tok}

    def ingest_file(self, path: Path) -> dict:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", "replace")
        return self.ingest_text(text, source=str(path), title=path.name)

    # ------------------------------------------------------------- interno

    def _ingest_tiny(self, doc_id: int, title: str, text: str, n_tok: int) -> int:
        ctx = f"[{title}] {text}" if title and title not in text else text
        vec = self.encoder.encode([ctx])[0]
        self.store.add_chunk(doc_id, text, ctx, n_tok, vec, pos=0)
        return 1

    def _ingest_chunked(self, doc_id: int, title: str, text: str, n_tok: int) -> int:
        chunks = chunk_text(text, self.cfg.chunk_tokens, self.cfg.chunk_overlap)

        # nós "pai": janelas maiores para devolver contexto amplo (small-to-big)
        per_parent = max(1, self.cfg.parent_tokens // max(1, self.cfg.chunk_tokens))
        parent_ids: List[Optional[int]] = []
        for i in range(0, len(chunks), per_parent):
            ptxt = " ".join(chunks[i : i + per_parent])
            pid = self.store.add_parent(doc_id, ptxt, estimate_tokens(ptxt), pos=i)
            parent_ids.extend([pid] * len(chunks[i : i + per_parent]))

        # enriquecimento contextual (opcional, com LLM)
        doc_head = text[:6000]
        ctxs: List[str] = []
        for c in chunks:
            prefix = f"[{title}] "
            if self.cfg.contextual_enrich and self.llm is not None and self.llm.available():
                try:
                    situ = self.llm.complete(
                        "Você situa trechos dentro de documentos. Seja breve.",
                        [{"role": "user", "content": _ENRICH_PROMPT.format(doc=doc_head, chunk=c[:2000])}],
                        max_tokens=120,
                    ).strip()
                    prefix = f"[{title}] {situ}\n"
                except Exception:
                    pass  # enriquecimento é acessório; nunca bloqueia a ingestão
            ctxs.append(prefix + c)

        # codifica em lote nas três formas
        vecs = self.encoder.encode(ctxs)
        for i, (c, ctx, vec) in enumerate(zip(chunks, ctxs, vecs)):
            self.store.add_chunk(
                doc_id, c, ctx, estimate_tokens(c), vec, pos=i, parent_id=parent_ids[i]
            )

        n = len(chunks)
        # documento gigante: nó de resumo pesquisável (RAPTOR-lite)
        if n_tok >= self.cfg.huge_doc_tokens and self.llm is not None and self.llm.available():
            try:
                summary = self.llm.complete(
                    "Você resume documentos com fidelidade, na língua do original.",
                    [{"role": "user", "content": f"Resuma em ~12 frases o documento (trechos ao longo dele):\n\n{_digest(text)}"}],
                    max_tokens=600,
                ).strip()
                svec = self.encoder.encode([f"[resumo de {title}] {summary}"])[0]
                self.store.add_chunk(
                    doc_id, summary, f"[resumo de {title}] {summary}",
                    estimate_tokens(summary), svec, kind="summary", pos=-1,
                )
                n += 1
            except Exception:
                pass
        return n

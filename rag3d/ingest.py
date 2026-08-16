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

from collections.abc import Sized
from itertools import islice
from pathlib import Path
from typing import List, Optional, Tuple

from .backend import DEFAULT_RETRIEVAL_LIMITS, RetrievalBackend
from .config import TriRagConfig
from .encoders import BaseEncoder, TriVec
from .llm import LLM, validate_llm_text
from .textproc import estimate_tokens, normalize, split_sentences

MAX_INGEST_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_INGEST_SOURCE_BYTES = 4 * 1024
MAX_INGEST_TITLE_BYTES = 4 * 1024
MAX_INGEST_LLM_RESPONSE_BYTES = DEFAULT_RETRIEVAL_LIMITS.max_query_bytes
MAX_INGEST_CHUNKS = 512
MAX_EMBEDDING_BATCH = 32


def _validate_document_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("document text must be a string")
    # O(1) character preflight prevents an unbounded UTF-8 allocation.  A
    # surviving string can allocate at most four times the byte limit here.
    if (
        len(text) > MAX_INGEST_DOCUMENT_BYTES
        or len(text.encode("utf-8")) > MAX_INGEST_DOCUMENT_BYTES
    ):
        raise ValueError(
            "document exceeds maximum of "
            f"{MAX_INGEST_DOCUMENT_BYTES} UTF-8 bytes"
        )
    return text


def _validate_ingest_label(value: str, name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if len(value) > maximum or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} exceeds maximum of {maximum} UTF-8 bytes")
    return value


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
    def __init__(self, store: RetrievalBackend, encoder: BaseEncoder, cfg: TriRagConfig, llm: Optional[LLM] = None):
        self.store = store
        self.encoder = encoder
        self.cfg = cfg
        self.llm = llm

    # ------------------------------------------------------------------ API

    def ingest_text(self, text: str, source: str = "inline", title: str = "") -> dict:
        text = _validate_document_text(text)
        source = _validate_ingest_label(source, "source", MAX_INGEST_SOURCE_BYTES)
        title = _validate_ingest_label(title, "title", MAX_INGEST_TITLE_BYTES)
        text = normalize(text)
        if not text:
            return {"doc_id": None, "chunks": 0}
        n_tok = estimate_tokens(text)
        title = title or (text[:60].replace("\n", " ") + ("…" if len(text) > 60 else ""))

        # Encoding and optional LLM calls are deliberately completed before the
        # write transaction. A slow external model must not extend database lock
        # lifetimes; only the bounded persistence phase below is atomic.
        if n_tok <= self.cfg.tiny_doc_tokens:
            tiny = self._prepare_tiny(title, text)
            chunked = None
        else:
            tiny = None
            chunked = self._prepare_chunked(title, text, n_tok)

        with self.store.transaction():
            doc_id = self.store.add_doc(source, title, n_tok)
            if tiny is not None:
                n = self._write_tiny(doc_id, text, n_tok, tiny)
            else:
                assert chunked is not None
                n = self._write_chunked(doc_id, chunked)
        return {"doc_id": doc_id, "chunks": n, "tokens": n_tok}

    def ingest_file(self, path: Path) -> dict:
        with path.open("rb") as handle:
            raw = handle.read(MAX_INGEST_DOCUMENT_BYTES + 1)
        if len(raw) > MAX_INGEST_DOCUMENT_BYTES:
            raise ValueError(
                "document exceeds maximum of "
                f"{MAX_INGEST_DOCUMENT_BYTES} UTF-8 bytes"
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", "replace")
        return self.ingest_text(text, source=str(path), title=path.name)

    # ------------------------------------------------------------- interno

    def _encode_exact(self, texts: List[str]) -> List[TriVec]:
        expected = len(texts)
        raw = self.encoder.encode(texts)
        if isinstance(raw, Sized) and len(raw) != expected:
            raise ValueError("encoder must return exactly one vector per input text")
        vecs = list(islice(iter(raw), expected + 1))
        if len(vecs) != expected:
            raise ValueError("encoder must return exactly one vector per input text")
        return vecs

    def _encode_batched(self, texts: List[str]) -> List[TriVec]:
        vectors: List[TriVec] = []
        for start in range(0, len(texts), MAX_EMBEDDING_BATCH):
            vectors.extend(
                self._encode_exact(texts[start : start + MAX_EMBEDDING_BATCH])
            )
        return vectors

    def _prepare_tiny(self, title: str, text: str) -> Tuple[str, TriVec]:
        ctx = f"[{title}] {text}" if title and title not in text else text
        vec = self._encode_exact([ctx])[0]
        return ctx, vec

    def _write_tiny(
        self,
        doc_id: int,
        text: str,
        n_tok: int,
        prepared: Tuple[str, TriVec],
    ) -> int:
        ctx, vec = prepared
        self.store.add_chunk(doc_id, text, ctx, n_tok, vec, pos=0)
        return 1

    def _prepare_chunked(
        self, title: str, text: str, n_tok: int
    ) -> Tuple[
        List[str],
        List[str],
        List[TriVec],
        List[Tuple[int, str, int]],
        Optional[Tuple[str, str, TriVec]],
    ]:
        chunks = chunk_text(text, self.cfg.chunk_tokens, self.cfg.chunk_overlap)
        if len(chunks) > MAX_INGEST_CHUNKS:
            raise ValueError(
                f"document chunks exceed maximum of {MAX_INGEST_CHUNKS}"
            )

        # Parent windows are planned here; their database IDs are assigned only
        # in the transaction after all external work has succeeded.
        per_parent = max(1, self.cfg.parent_tokens // max(1, self.cfg.chunk_tokens))
        parents: List[Tuple[int, str, int]] = []
        for i in range(0, len(chunks), per_parent):
            ptxt = " ".join(chunks[i : i + per_parent])
            parents.append((i, ptxt, estimate_tokens(ptxt)))

        # enriquecimento contextual (opcional, com LLM)
        doc_head = text[:6000]
        ctxs: List[str] = []
        for c in chunks:
            prefix = f"[{title}] "
            if self.cfg.contextual_enrich and self.llm is not None and self.llm.available():
                try:
                    raw_situ = self.llm.complete(
                        "Você situa trechos dentro de documentos. Seja breve.",
                        [{"role": "user", "content": _ENRICH_PROMPT.format(doc=doc_head, chunk=c[:2000])}],
                        max_tokens=120,
                    )
                    situ = validate_llm_text(
                        raw_situ, MAX_INGEST_LLM_RESPONSE_BYTES
                    ).strip()
                    prefix = f"[{title}] {situ}\n"
                except Exception:
                    pass  # enriquecimento é acessório; nunca bloqueia a ingestão
            ctxs.append(prefix + c)

        # External model calls stay outside the transaction and each request is
        # bounded.  The absolute chunk cap also bounds the prepared vectors
        # retained until the atomic persistence phase.
        vecs = self._encode_batched(ctxs)

        summary_payload: Optional[Tuple[str, str, TriVec]] = None
        if n_tok >= self.cfg.huge_doc_tokens and self.llm is not None and self.llm.available():
            summary: Optional[str]
            try:
                raw_summary = self.llm.complete(
                    "Você resume documentos com fidelidade, na língua do original.",
                    [{"role": "user", "content": f"Resuma em ~12 frases o documento (trechos ao longo dele):\n\n{_digest(text)}"}],
                    max_tokens=600,
                )
                summary = validate_llm_text(
                    raw_summary, MAX_INGEST_LLM_RESPONSE_BYTES
                ).strip()
            except Exception:
                summary = None
            if summary is not None:
                summary_context = f"[resumo de {title}] {summary}"
                svec = self._encode_exact([summary_context])[0]
                summary_payload = (summary, summary_context, svec)

        return chunks, ctxs, vecs, parents, summary_payload

    def _write_chunked(
        self,
        doc_id: int,
        prepared: Tuple[
            List[str],
            List[str],
            List[TriVec],
            List[Tuple[int, str, int]],
            Optional[Tuple[str, str, TriVec]],
        ],
    ) -> int:
        chunks, ctxs, vecs, parents, summary_payload = prepared
        per_parent = max(1, self.cfg.parent_tokens // max(1, self.cfg.chunk_tokens))
        parent_ids: List[Optional[int]] = [None] * len(chunks)
        for start, parent_text, parent_tokens in parents:
            parent_id = self.store.add_parent(
                doc_id, parent_text, parent_tokens, pos=start
            )
            end = min(len(chunks), start + per_parent)
            parent_ids[start:end] = [parent_id] * (end - start)

        for i, (chunk, context, vector) in enumerate(zip(chunks, ctxs, vecs)):
            self.store.add_chunk(
                doc_id,
                chunk,
                context,
                estimate_tokens(chunk),
                vector,
                pos=i,
                parent_id=parent_ids[i],
            )

        n = len(chunks)
        if summary_payload is not None:
            summary, summary_context, summary_vector = summary_payload
            self.store.add_chunk(
                doc_id,
                summary,
                summary_context,
                estimate_tokens(summary),
                summary_vector,
                kind="summary",
                pos=-1,
            )
            n += 1
        return n

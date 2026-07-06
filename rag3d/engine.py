"""Fachada TriRAG — junta as peças.

    from trirag import TriRag
    rag = TriRag()
    rag.ingest("qualquer texto, em qualquer língua")
    rag.ingest_file("docs/manual.md")
    r = rag.ask("qual o prazo do contrato?")          # pergunta avulsa
    r = rag.chat("e o que combinamos ontem?")         # com memória infinita
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Union

from .config import TriRagConfig
from .encoders import make_encoder
from .ingest import Ingestor
from .llm import LLM, CallableLLM, make_llm
from .memory import ChatMemory
from .reader import Reader
from .retrieve import TriResult, TriRetriever
from .store import TriStore


class TriRag:
    def __init__(
        self,
        cfg: Optional[TriRagConfig] = None,
        llm: Union[LLM, Callable, None] = None,
        axis_llms: Optional[Dict[str, LLM]] = None,
    ):
        self.cfg = cfg or TriRagConfig()
        data = self.cfg.ensure_dirs()
        if self.cfg.pg_dsn:
            from .pgstore import PgHoloStore  # import tardio: psycopg opcional
            self.store = PgHoloStore(self.cfg.pg_dsn, self.cfg.dense_dim, self.cfg.colbert_dim)
        else:
            self.store = TriStore(data / "trirag.db")
        self.encoder = make_encoder(
            self.cfg.encoder, self.cfg.dense_dim, self.cfg.colbert_dim, self.cfg.max_colbert_tokens
        )
        # não misturar índices incompatíveis: nome do encoder E dimensões
        # (dims diferentes com mesmo nome corromperiam os vetores silenciosamente)
        fingerprint = f"{self.encoder.name}:{self.cfg.dense_dim}:{self.cfg.colbert_dim}"
        prev = self.store.get_meta("encoder")
        if prev and prev != fingerprint:
            where = self.cfg.pg_dsn or f"{data}/trirag.db"
            raise RuntimeError(
                f"Índice foi criado com encoder '{prev}', agora ativo '{fingerprint}'. "
                f"Use a mesma configuração ou apague o índice ({where})."
            )
        self.store.set_meta("encoder", fingerprint)

        if callable(llm) and not isinstance(llm, LLM):
            self.llm: LLM = CallableLLM(llm)
        elif isinstance(llm, LLM):
            self.llm = llm
        else:
            self.llm = make_llm(self.cfg.llm_provider, self.cfg.llm_model)

        from .rerank import Reranker
        reranker = Reranker(self.llm) if self.cfg.rerank else None
        self.ingestor = Ingestor(self.store, self.encoder, self.cfg, self.llm)
        self.retriever = TriRetriever(self.store, self.encoder, self.cfg, reranker=reranker, llm=self.llm)
        self.memory = ChatMemory(self.store, self.encoder, self.cfg, self.llm)
        self.reader = Reader(self.cfg, self.llm, axis_llms)

    # ------------------------------------------------------------- ingestão

    def ingest(self, text: str, source: str = "inline", title: str = "") -> dict:
        return self.ingestor.ingest_text(text, source=source, title=title)

    def ingest_file(self, path: Union[str, Path]) -> dict:
        return self.ingestor.ingest_file(Path(path))

    # ---------------------------------------------------------------- busca

    def search(self, query: str, top_k: Optional[int] = None) -> TriResult:
        """Só retrieval: devolve ranking fundido + as três visões."""
        return self.retriever.search(query, top_k=top_k)

    # -------------------------------------------------------------- leitura

    def ask(self, query: str, mode: Optional[str] = None) -> dict:
        """Pergunta avulsa (sem gravar memória)."""
        ctx = self.memory.build_context(query, self.retriever)
        out = self.reader.read(query, ctx, mode=mode)
        out["context"] = ctx
        return out

    def chat(self, user_msg: str, mode: Optional[str] = None) -> dict:
        """Turno de conversa: recupera, lê, e grava o turno na memória."""
        ctx = self.memory.build_context(user_msg, self.retriever)
        out = self.reader.read(user_msg, ctx, mode=mode)
        out["context"] = ctx
        # só grava memória se habilitado (doc-QA não polui a busca com turnos)
        if out.get("answer") and getattr(self.cfg, "remember_chat", True):
            out["turn_no"] = self.memory.record_turn(user_msg, out["answer"])
        return out

    # ---------------------------------------------------------------- infra

    def stats(self) -> dict:
        return {
            "encoder": self.encoder.name,
            "llm": f"{self.llm.provider}:{self.llm.model}" if self.llm.available() else "nenhum",
            "chunks": self.store.n_chunks(),
            "tokens_no_corpus": self.store.corpus_tokens(),
            "turnos": self.store.last_turn_no(),
            "fusao": self.cfg.fusion,
            "backend": "postgres-holo (sem pgvector)" if self.cfg.pg_dsn else "sqlite",
            "dados": self.cfg.pg_dsn or str(self.cfg.data_dir),
        }

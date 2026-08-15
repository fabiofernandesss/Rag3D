"""Fachada TriRAG — junta as peças.

    from trirag import TriRag
    rag = TriRag()
    rag.ingest("qualquer texto, em qualquer língua")
    rag.ingest_file("docs/manual.md")
    r = rag.ask("qual o prazo do contrato?")          # pergunta avulsa
    r = rag.chat("e o que combinamos ontem?")         # com memória infinita
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from .backend import (
    FingerprintMismatchError,
    IndexFingerprint,
    RetrievalBackend,
    SearchFilters,
    validate_query_text,
)
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
        self.cfg.resolve()
        data = self.cfg.ensure_dirs()
        self.store = None
        try:
            if self.cfg.backend == "pgvector":
                self.encoder = make_encoder(
                    self.cfg.encoder,
                    self.cfg.dense_dim,
                    self.cfg.colbert_dim,
                    self.cfg.max_colbert_tokens,
                    allow_fallback=bool(self.cfg.allow_encoder_fallback),
                )
                fingerprint = self._v2_fingerprint(backend_name="pgvector")
                self.store = self._make_store(data, fingerprint=fingerprint)
            else:
                self.store = self._make_store(data)
                self.encoder = make_encoder(
                    self.cfg.encoder,
                    self.cfg.dense_dim,
                    self.cfg.colbert_dim,
                    self.cfg.max_colbert_tokens,
                    allow_fallback=bool(self.cfg.allow_encoder_fallback),
                )
            self._ensure_fingerprints()

            if callable(llm) and not isinstance(llm, LLM):
                self.llm: LLM = CallableLLM(llm)
            elif isinstance(llm, LLM):
                self.llm = llm
            else:
                self.llm = make_llm(self.cfg.llm_provider, self.cfg.llm_model)

            self.ingestor = Ingestor(self.store, self.encoder, self.cfg, self.llm)
            if self.cfg.retrieval_pipeline == "v2":
                from .rerank import (
                    CrossEncoderReranker,
                    LLMListwiseReranker,
                    NoOpReranker,
                )
                from .retrieval_v2 import RetrievalV2

                v2_rerankers = {
                    "none": lambda: NoOpReranker(),
                    "llm": lambda: LLMListwiseReranker(self.llm),
                    "cross-encoder": lambda: CrossEncoderReranker(),
                }
                v2_reranker = v2_rerankers[self.cfg.reranker]()
                self.retriever = RetrievalV2(
                    self.store,
                    self.encoder,
                    self.cfg,
                    reranker=v2_reranker,
                    llm=self.llm,
                )
            else:
                from .rerank import Reranker

                legacy_reranker = Reranker(self.llm) if self.cfg.rerank else None
                self.retriever = TriRetriever(
                    self.store,
                    self.encoder,
                    self.cfg,
                    reranker=legacy_reranker,
                    llm=self.llm,
                )
            self.memory = ChatMemory(self.store, self.encoder, self.cfg, self.llm)
            self.reader = Reader(self.cfg, self.llm, axis_llms)
        except BaseException:
            try:
                if self.store is not None:
                    self.store.close()
            except BaseException:
                pass
            raise

    def _make_store(
        self,
        data: Path,
        *,
        fingerprint: Optional[IndexFingerprint] = None,
    ) -> RetrievalBackend:
        backend = self.cfg.backend
        if backend == "sqlite":
            self.store = TriStore(data / "trirag.db")
            return self.store

        if backend not in {"postgres-holo", "pgvector"}:
            raise ValueError("unsupported retrieval backend")
        if not self.cfg.pg_dsn:
            raise ValueError(f"DSN is required for backend '{backend}'")

        if backend == "postgres-holo":
            try:
                from .pgstore import PgHoloStore
            except (ImportError, ModuleNotFoundError):
                raise RuntimeError("postgres-holo backend is unavailable") from None
            store_type = PgHoloStore
        else:
            try:
                from .pgvector_store import PgVectorStore
            except (ImportError, ModuleNotFoundError):
                raise RuntimeError("pgvector backend is unavailable") from None
            store_type = PgVectorStore

        try:
            if backend == "pgvector":
                return store_type(
                    self.cfg.pg_dsn,
                    self.cfg.dense_dim,
                    self.cfg.colbert_dim,
                    fingerprint=fingerprint,
                    search_mode=self.cfg.pgvector_search_mode,
                    statement_timeout_ms=self.cfg.pgvector_statement_timeout_ms,
                )
            return store_type(self.cfg.pg_dsn, self.cfg.dense_dim, self.cfg.colbert_dim)
        except FingerprintMismatchError:
            # This typed, secret-safe error is part of the backend contract:
            # callers use it to trigger an explicit reindex or rollback.
            raise
        except Exception as exc:
            # Driver exceptions may include conninfo. Preserve only the safe
            # implementation type, never the original DSN or exception text.
            raise RuntimeError(
                f"failed to initialize backend '{backend}' ({type(exc).__name__})"
            ) from None

    def _v2_fingerprint(
        self, *, backend_name: Optional[str] = None
    ) -> IndexFingerprint:
        encoder_name = self.encoder.name
        spec = self.encoder.index_spec
        resolved_backend = backend_name or self.store.backend_name
        quantization = {
            "sqlite": "dense-float32_structural-float16",
            "postgres-holo": "dense-int8_structural-binary",
            "pgvector": "native-vector",
        }[resolved_backend]
        return IndexFingerprint(
            backend=resolved_backend,
            encoder=encoder_name,
            model=spec.model,
            revision=spec.revision,
            dense_dim=self.cfg.dense_dim,
            structural_dim=self.cfg.colbert_dim,
            max_structural_tokens=spec.max_structural_tokens,
            structural_projection=spec.structural_projection,
            query_max_tokens=spec.query_max_tokens,
            passage_max_tokens=spec.passage_max_tokens,
            sparse_version=spec.sparse_version,
            schema_version=spec.schema_version,
            normalization="l2",
            quantization=quantization,
            chunk_size=self.cfg.chunk_tokens,
            overlap=self.cfg.chunk_overlap,
            chunking_version="sentence-v1",
            pipeline_version="retrieval-v2",
        )

    def _ensure_fingerprints(self) -> None:
        """Preserve the cross-language legacy key and add a canonical V2 key."""
        legacy = f"{self.encoder.name}:{self.cfg.dense_dim}:{self.cfg.colbert_dim}"
        with self.store.transaction():
            lock_fingerprint = getattr(self.store, "lock_fingerprint", None)
            if callable(lock_fingerprint):
                lock_fingerprint()
            previous = self.store.get_meta("encoder")
            if previous and previous != legacy:
                raise FingerprintMismatchError("incompatible encoder fingerprint")

            stored_payload = self.store.get_meta("retrieval_v2_fingerprint")
            stored_digest = self.store.get_meta(
                "retrieval_v2_fingerprint_sha256"
            )
            expected = self._v2_fingerprint()
            payload = expected.canonical_json()
            if stored_payload:
                try:
                    raw = json.loads(stored_payload)
                    if not isinstance(raw, dict):
                        raise TypeError
                    stored = IndexFingerprint(**raw)
                except (TypeError, ValueError, json.JSONDecodeError):
                    raise FingerprintMismatchError(
                        "invalid retrieval V2 fingerprint metadata"
                    ) from None
                if stored_payload != stored.canonical_json():
                    raise FingerprintMismatchError(
                        "retrieval V2 fingerprint metadata is not canonical"
                    )
                expected.assert_compatible(stored)
                actual_digest = hashlib.sha256(
                    stored_payload.encode("utf-8")
                ).hexdigest()
                if not stored_digest or stored_digest != actual_digest:
                    raise FingerprintMismatchError(
                        "invalid retrieval V2 fingerprint digest"
                    )
            elif stored_digest:
                raise FingerprintMismatchError(
                    "invalid retrieval V2 fingerprint metadata"
                )
            elif (
                self.cfg.retrieval_pipeline == "v2"
                and self.store.n_chunks() > 0
            ):
                raise FingerprintMismatchError(
                    "populated legacy index has no verified retrieval V2 "
                    "fingerprint; reindex before enabling V2"
                )

            if self.cfg.retrieval_pipeline == "v2":
                self.store.set_meta("retrieval_v2_fingerprint", payload)
                self.store.set_meta(
                    "retrieval_v2_fingerprint_sha256", expected.digest
                )

            self.store.set_meta("encoder", legacy)

    # ------------------------------------------------------------- ingestão

    def ingest(self, text: str, source: str = "inline", title: str = "") -> dict:
        return self.ingestor.ingest_text(text, source=source, title=title)

    def ingest_file(self, path: Union[str, Path]) -> dict:
        return self.ingestor.ingest_file(Path(path))

    # ---------------------------------------------------------------- busca

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        channel_k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
    ) -> TriResult:
        """Run retrieval while preserving the historical ``query/top_k`` API."""
        query = validate_query_text(query)
        if filters is not None and not isinstance(filters, SearchFilters):
            raise TypeError("filters must be SearchFilters")
        if self.cfg.retrieval_pipeline == "legacy":
            if filters is not None and not filters.is_empty:
                raise NotImplementedError("filters require Retrieval Engine V2")
            return self.retriever.search(
                query,
                top_k=top_k,
                channel_k=channel_k,
            )
        return self.retriever.search(
            query,
            top_k=top_k,
            channel_k=channel_k,
            filters=filters,
        )

    # -------------------------------------------------------------- leitura

    def ask(self, query: str, mode: Optional[str] = None) -> dict:
        """Pergunta avulsa (sem gravar memória)."""
        query = validate_query_text(query)
        ctx = self.memory.build_context(query, self.retriever)
        out = self.reader.read(query, ctx, mode=mode)
        out["context"] = ctx
        return out

    def chat(self, user_msg: str, mode: Optional[str] = None) -> dict:
        """Turno de conversa: recupera, lê, e grava o turno na memória."""
        user_msg = validate_query_text(user_msg)
        ctx = self.memory.build_context(user_msg, self.retriever)
        out = self.reader.read(user_msg, ctx, mode=mode)
        out["context"] = ctx
        # só grava memória se habilitado (doc-QA não polui a busca com turnos)
        if out.get("answer") and getattr(self.cfg, "remember_chat", True):
            out["turn_no"] = self.memory.record_turn(user_msg, out["answer"])
        return out

    # ---------------------------------------------------------------- infra

    def stats(self) -> dict:
        stats = {
            "encoder": self.encoder.name,
            "llm": f"{self.llm.provider}:{self.llm.model}" if self.llm.available() else "nenhum",
            "chunks": self.store.n_chunks(),
            "tokens_no_corpus": self.store.corpus_tokens(),
            "turnos": self.store.last_turn_no(),
            "fusao": self.cfg.fusion,
            "backend": self.store.backend_name,
            "dados": "local" if self.store.backend_name == "sqlite" else "remote",
            "pipeline": self.cfg.retrieval_pipeline,
            "fusion": self.cfg.fusion,
            "reranker": self.cfg.reranker,
            "diversity": self.cfg.diversity_method,
        }
        if self.store.backend_name == "pgvector":
            stats["pgvector_search_mode"] = self.cfg.pgvector_search_mode
        return stats

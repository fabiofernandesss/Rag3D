"""Configuração central do RAG3D."""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .backend import DEFAULT_RETRIEVAL_LIMITS


MAX_STITCH_RADIUS = DEFAULT_RETRIEVAL_LIMITS.max_top_k


def _env(name: str, default: str = "") -> str:
    """Lê RAG3D_<name>, com fallback para o antigo TRIRAG_<name>."""
    current_name = f"RAG3D_{name}"
    if current_name in os.environ:
        return os.environ[current_name]
    legacy_name = f"TRIRAG_{name}"
    if legacy_name in os.environ:
        warnings.warn(
            f"{legacy_name} is deprecated; use {current_name}",
            DeprecationWarning,
            stacklevel=4,
        )
        return os.environ[legacy_name]
    return default


def _new_env(name: str) -> str:
    """Read only the new namespace, used where legacy means a different setting."""
    return os.environ.get(f"RAG3D_{name}", "").strip()


def _new_env_int(name: str, default: int) -> int:
    raw = _new_env(name)
    if not raw:
        return default
    try:
        return int(raw, 10)
    except ValueError:
        raise ValueError(f"RAG3D_{name} must be an integer") from None


def _optional_env_bool(name: str) -> Optional[bool]:
    raw = _new_env(name)
    if not raw:
        return None
    normalized = raw.casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"RAG3D_{name} must be one of true/false, yes/no, on/off, or 1/0"
    )


def _env_bool(name: str, default: bool) -> bool:
    value = _optional_env_bool(name)
    return default if value is None else value


def _bounded_int(name: str, value: int, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _normalized_enum(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value.strip().lower()


@dataclass
class TriRagConfig:
    # --- armazenamento ---
    data_dir: Path = field(
        default_factory=lambda: Path(_env("DATA", "~/.rag3d")).expanduser(),
        repr=False,
    )
    # DSN Postgres (ex: postgresql://user:senha@host:5432/db). Vazio = SQLite
    # local. Postgres NÃO precisa de pgvector: backend holográfico puro.
    pg_dsn: str = field(default_factory=lambda: _env("PG", ""), repr=False)
    # --- encoder ---
    # "auto": tenta BGE-M3 (FlagEmbedding), senão fallback agnóstico de língua.
    encoder: str = field(default_factory=lambda: _env("ENCODER", "auto"))
    dense_dim: int = 1024          # BGE-M3; fallback projeta para este mesmo tamanho
    colbert_dim: int = 128         # dimensão reduzida dos vetores token a token
    max_colbert_tokens: int = 256  # limite de tokens armazenados por chunk no eixo 3
    # --- chunking adaptativo ---
    chunk_tokens: int = 400        # alvo por chunk (filho)
    chunk_overlap: int = 60        # sobreposição entre chunks
    parent_tokens: int = 1600      # janela "pai" (small-to-big)
    tiny_doc_tokens: int = 500     # doc menor que isso é salvo inteiro, sem quebrar
    huge_doc_tokens: int = 12000   # doc maior que isso ganha nó de resumo (se LLM disponível)
    contextual_enrich: bool = True # Contextual Retrieval (Anthropic): situa o chunk no doc via LLM

    # --- busca ---
    # pesquisa 2024-26: recuperar largo (100-200/canal) e entregar ~10-20 ao
    # leitor rende mais que top-5 estreito (Anthropic Contextual Retrieval).
    top_k: int = 10                # resultados finais entregues ao leitor
    channel_k: int = 100           # candidatos por canal antes da fusão
    # reranker LLM-agnóstico (maior ganho da pesquisa): reordena o topo fundido
    rerank: bool = False           # ligado sob demanda (custa 1 chamada de LLM)
    rerank_pool: int = 30          # quantos candidatos o reranker considera
    # Rag3D — costura de contiguidade: junta vizinhos (pos±raio) do mesmo doc,
    # reconstruindo listas/tabelas/seções partidas pelo chunking. 0 = off.
    stitch_radius: int = 0
    # Rag3D — query expansion via LLM (fecha recall em perguntas parafraseadas)
    expand_query: bool = False
    expand_query_max: int = 3
    # O sentinel vazio torna o padrão dependente da pipeline sem mudar o legado.
    fusion: str = field(default_factory=lambda: _env("FUSION", ""))
    rrf_k: int = 60                # constante clássica do RRF
    channel_weights: tuple = (1.0, 0.8, 0.9)  # pesos (semântico, léxico, estrutural)

    # --- fusão quântica ---
    interference_strength: float = 1.0  # 0 = clássico (sem interferência); 1 = interferência plena
    # coerência: modula o peso de cada eixo pela PUREZA da sua distribuição de
    # pontuações nesta consulta (canal indeciso pesa menos). 0 = pesos fixos.
    coherence_strength: float = 0.0
    # seleção fermiônica (greedy DPP / determinante de Slater): aproxima um
    # conjunto que equilibra relevância e volume — dois trechos quase idênticos
    # tendem a não ocupar duas vagas. 0 = ranking puro (comportamento clássico);
    # 0.3-0.6 aumenta o peso da cobertura, sem garantia de ótimo global.
    diversity: float = 0.35
    diversity_pool: int = 40           # candidatos considerados na seleção

    # --- memória de conversa infinita ---
    memory_budget_tokens: int = 6000    # orçamento de contexto recuperado por turno
    recency_half_life_turns: float = 40.0
    w_relevance: float = 1.0
    w_recency: float = 0.35
    w_importance: float = 0.25
    summary_every_turns: int = 12       # consolida resumo corrente a cada N turnos

    # --- leitura (LLM) ---
    llm_provider: str = field(default_factory=lambda: _env("LLM", "auto"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    read_mode: str = "fast"        # fast: 1 LLM lê as 3 visões | tri: 3 leitores + sintetizador
    max_answer_tokens: int = 1500
    # True: cada turno vira memória pesquisável (chat infinito). False: doc-QA
    # sem misturar histórico na busca (evita repetir respostas passadas).
    remember_chat: bool = True

    # ponto de cruzamento: corpus pequeno entra inteiro no prompt, sem retrieval
    small_corpus_tokens: int = 8000

    # --- Retrieval Engine V2 / rollout ---
    # Todos os campos novos ficam após a assinatura posicional histórica.
    # Strings vazias/None são sentinels internos para defaults derivados.
    backend: str = field(default_factory=lambda: _new_env("BACKEND"))
    retrieval_pipeline: str = field(
        default_factory=lambda: _env("RETRIEVAL_PIPELINE", "legacy")
    )
    structural_rerank: bool = field(
        default_factory=lambda: _env_bool("STRUCTURAL_RERANK", True)
    )
    reranker: str = field(default_factory=lambda: _env("RERANKER", ""))
    diversity_method: str = field(default_factory=lambda: _env("DIVERSITY_METHOD", "none"))
    # None é resolvido por pipeline: legado mantém a compatibilidade; V2 falha
    # fechado salvo autorização explícita.
    allow_encoder_fallback: Optional[bool] = field(
        default_factory=lambda: _optional_env_bool("ALLOW_ENCODER_FALLBACK")
    )
    # Late interaction continua pós-fusão, mas possui profundidade própria e
    # não depende de reranker/diversidade estarem habilitados.
    structural_candidate_depth: int = 100
    # pgvector is exact-by-default. ANN and automatic routing are explicit,
    # auditable rollout choices rather than consequences of index presence.
    pgvector_search_mode: str = field(
        default_factory=lambda: _new_env("PGVECTOR_SEARCH_MODE") or "exact"
    )
    pgvector_statement_timeout_ms: int = field(
        default_factory=lambda: _new_env_int(
            "PGVECTOR_STATEMENT_TIMEOUT_MS", 5_000
        )
    )

    # Origem e último snapshot permitem recomputar defaults depois das mutações
    # públicas legadas sem sobrescrever escolhas explícitas.
    _backend_explicit: bool = field(default=False, init=False, repr=False, compare=False)
    _fusion_explicit: bool = field(default=False, init=False, repr=False, compare=False)
    _fallback_explicit: bool = field(default=False, init=False, repr=False, compare=False)
    _reranker_explicit: bool = field(default=False, init=False, repr=False, compare=False)
    _last_resolved_backend: Optional[str] = field(
        default=None, init=False, repr=False, compare=False
    )
    _last_resolved_fusion: Optional[str] = field(
        default=None, init=False, repr=False, compare=False
    )
    _last_resolved_fallback: Optional[bool] = field(
        default=None, init=False, repr=False, compare=False
    )
    _last_resolved_reranker: Optional[str] = field(
        default=None, init=False, repr=False, compare=False
    )
    _tracking_enabled: bool = field(default=False, init=False, repr=False, compare=False)
    _resolving: bool = field(default=False, init=False, repr=False, compare=False)

    _TRACKED_DERIVED_FIELDS = {
        "backend": "_backend_explicit",
        "fusion": "_fusion_explicit",
        "allow_encoder_fallback": "_fallback_explicit",
        "reranker": "_reranker_explicit",
    }

    def __setattr__(self, name: str, value: object) -> None:
        flag_name = self._TRACKED_DERIVED_FIELDS.get(name)
        if (
            flag_name is not None
            and self.__dict__.get("_tracking_enabled", False)
            and not self.__dict__.get("_resolving", False)
        ):
            if name == "allow_encoder_fallback":
                explicit = value is not None
            else:
                explicit = not (
                    isinstance(value, str) and not value.strip()
                )
            object.__setattr__(self, flag_name, explicit)
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        self.backend = _normalized_enum("backend", self.backend)
        self.fusion = _normalized_enum("fusion", self.fusion)
        self.reranker = _normalized_enum("reranker", self.reranker)
        self._backend_explicit = bool(self.backend)
        self._fusion_explicit = bool(self.fusion)
        self._fallback_explicit = self.allow_encoder_fallback is not None
        self._reranker_explicit = bool(self.reranker)
        self.resolve()
        object.__setattr__(self, "_tracking_enabled", True)

    @staticmethod
    def _updated_string_origin(
        current: str,
        last_resolved: Optional[str],
        was_explicit: bool,
    ) -> bool:
        if last_resolved is not None and current != last_resolved:
            return bool(current)
        return was_explicit

    def resolve(self) -> "TriRagConfig":
        """Atomically resolve dependent defaults after public field mutation.

        Values supplied through constructor/environment, or changed since the
        last resolution, remain explicit. Empty string/``None`` resets a field
        to the pipeline-dependent default.
        """
        pipeline = _normalized_enum("retrieval_pipeline", self.retrieval_pipeline)
        backend_current = _normalized_enum("backend", self.backend)
        fusion_current = _normalized_enum("fusion", self.fusion)
        reranker_current = _normalized_enum("reranker", self.reranker)
        diversity_method = _normalized_enum("diversity_method", self.diversity_method)
        encoder = _normalized_enum("encoder", self.encoder)
        pgvector_search_mode = _normalized_enum(
            "pgvector_search_mode", self.pgvector_search_mode
        )

        backend_explicit = self._updated_string_origin(
            backend_current, self._last_resolved_backend, self._backend_explicit
        )
        fusion_explicit = self._updated_string_origin(
            fusion_current, self._last_resolved_fusion, self._fusion_explicit
        )
        reranker_explicit = self._updated_string_origin(
            reranker_current, self._last_resolved_reranker, self._reranker_explicit
        )

        fallback_current = self.allow_encoder_fallback
        if fallback_current is not None and not isinstance(fallback_current, bool):
            raise TypeError("allow_encoder_fallback must be bool")
        fallback_explicit = self._fallback_explicit
        if (
            self._last_resolved_fallback is not None
            and fallback_current != self._last_resolved_fallback
        ):
            fallback_explicit = fallback_current is not None

        backend = (
            backend_current
            if backend_explicit
            else ("postgres-holo" if self.pg_dsn else "sqlite")
        )
        fusion = fusion_current if fusion_explicit else ("rrf" if pipeline == "v2" else "quantum")
        fallback = (
            fallback_current if fallback_explicit else pipeline == "legacy"
        )
        reranker = (
            reranker_current
            if reranker_explicit
            else ("llm" if self.rerank else "none")
        )

        if not isinstance(self.structural_rerank, bool):
            raise TypeError("structural_rerank must be bool")
        choices = {
            "backend": (backend, {"sqlite", "postgres-holo", "pgvector"}),
            "retrieval_pipeline": (pipeline, {"legacy", "v2"}),
            "fusion": (fusion, {"rrf", "quantum"}),
            "reranker": (reranker, {"none", "llm", "cross-encoder"}),
            "diversity_method": (diversity_method, {"none", "dpp", "mmr"}),
            "encoder": (encoder, {"auto", "bge-m3", "fallback", "hash"}),
            "pgvector_search_mode": (
                pgvector_search_mode,
                {"exact", "ann", "auto"},
            ),
        }
        for name, (value, allowed) in choices.items():
            if value not in allowed:
                raise ValueError(
                    f"invalid {name}; expected one of {', '.join(sorted(allowed))}"
                )

        limits = DEFAULT_RETRIEVAL_LIMITS
        _bounded_int("top_k", self.top_k, 1, limits.max_top_k)
        _bounded_int("channel_k", self.channel_k, 1, limits.max_channel_k)
        _bounded_int("rerank_pool", self.rerank_pool, 1, limits.max_pool)
        _bounded_int("diversity_pool", self.diversity_pool, 1, limits.max_pool)
        _bounded_int(
            "structural_candidate_depth",
            self.structural_candidate_depth,
            1,
            limits.max_pool,
        )
        _bounded_int(
            "pgvector_statement_timeout_ms",
            self.pgvector_statement_timeout_ms,
            1,
            60_000,
        )
        _bounded_int("stitch_radius", self.stitch_radius, 0, MAX_STITCH_RADIUS)
        _bounded_int(
            "expand_query_max", self.expand_query_max, 1, limits.max_query_expansions
        )
        _bounded_int("dense_dim", self.dense_dim, 1, limits.max_dense_dim)
        _bounded_int(
            "colbert_dim", self.colbert_dim, 1, limits.max_structural_dim
        )
        _bounded_int(
            "max_colbert_tokens",
            self.max_colbert_tokens,
            1,
            limits.max_structural_tokens,
        )
        if self.colbert_dim * self.max_colbert_tokens > limits.max_structural_values:
            raise ValueError(
                "structural tensor exceeds the public value-count bound"
            )
        _positive_int("chunk_tokens", self.chunk_tokens)
        if isinstance(self.chunk_overlap, bool) or not isinstance(self.chunk_overlap, int):
            raise TypeError("chunk_overlap must be an integer, not bool")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_tokens")
        _bounded_int("rrf_k", self.rrf_k, 1, limits.max_rrf_k)

        if self.top_k > self.channel_k:
            raise ValueError("top_k cannot exceed channel_k")

        # Publish the complete resolved snapshot only after every validation
        # succeeds, avoiding a half-updated configuration on failure.
        object.__setattr__(self, "_resolving", True)
        try:
            self.backend = backend
            self.retrieval_pipeline = pipeline
            self.fusion = fusion
            self.allow_encoder_fallback = fallback
            self.reranker = reranker
            self.diversity_method = diversity_method
            self.encoder = encoder
            self.pgvector_search_mode = pgvector_search_mode
            self._backend_explicit = backend_explicit
            self._fusion_explicit = fusion_explicit
            self._fallback_explicit = fallback_explicit
            self._reranker_explicit = reranker_explicit
            self._last_resolved_backend = backend
            self._last_resolved_fusion = fusion
            self._last_resolved_fallback = fallback
            self._last_resolved_reranker = reranker
        finally:
            object.__setattr__(self, "_resolving", False)
        return self

    def ensure_dirs(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir

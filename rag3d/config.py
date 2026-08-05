"""Configuração central do RAG3D."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    """Lê RAG3D_<name>, com fallback para o antigo TRIRAG_<name>."""
    return os.environ.get(f"RAG3D_{name}") or os.environ.get(f"TRIRAG_{name}") or default


@dataclass
class TriRagConfig:
    # --- armazenamento ---
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA", "~/.rag3d")).expanduser())
    # DSN Postgres (ex: postgresql://user:senha@host:5432/db). Vazio = SQLite
    # local. Postgres NÃO precisa de pgvector: backend holográfico puro.
    pg_dsn: str = _env("PG", "")

    # --- encoder ---
    # "auto": tenta BGE-M3 (FlagEmbedding), senão fallback agnóstico de língua.
    encoder: str = _env("ENCODER", "auto")  # auto | bge-m3 | fallback
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
    fusion: str = "quantum"        # quantum | rrf
    rrf_k: int = 60                # constante clássica do RRF
    channel_weights: tuple = (1.0, 0.8, 0.9)  # pesos (semântico, léxico, estrutural)

    # --- fusão quântica ---
    interference_strength: float = 1.0  # 0 = clássico (sem interferência); 1 = interferência plena
    # coerência: modula o peso de cada eixo pela PUREZA da sua distribuição de
    # pontuações nesta consulta (canal indeciso pesa menos). 0 = pesos fixos.
    coherence_strength: float = 0.0
    # seleção fermiônica (MAP-DPP / determinante de Slater): escolhe o CONJUNTO
    # final maximizando relevância × volume — dois trechos quase idênticos não
    # ocupam duas vagas (exclusão de Pauli). 0 = ranking puro (comportamento
    # clássico); 0.3-0.6 = cobertura melhor sem perder o topo.
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
    llm_provider: str = _env("LLM", "auto")  # auto | anthropic | openai | ollama | none
    llm_model: str = _env("LLM_MODEL", "")
    read_mode: str = "fast"        # fast: 1 LLM lê as 3 visões | tri: 3 leitores + sintetizador
    max_answer_tokens: int = 1500
    # True: cada turno vira memória pesquisável (chat infinito). False: doc-QA
    # sem misturar histórico na busca (evita repetir respostas passadas).
    remember_chat: bool = True

    # ponto de cruzamento: corpus pequeno entra inteiro no prompt, sem retrieval
    small_corpus_tokens: int = 8000

    def ensure_dirs(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir

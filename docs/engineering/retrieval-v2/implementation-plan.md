# Plano de implementação da Retrieval Engine V2

Este plano transforma o grafo de requisitos em unidades isoladas de escrita. Cada unidade segue red-green-refactor, revisão de especificação e revisão de qualidade. Nenhum arquivo pode ter dois escritores simultâneos.

## Invariantes globais

- `legacy` preserva API, quantum, SQLite, PostgreSQL holográfico e campos públicos.
- `v2` usa RRF equal-weight dense+sparse por default; structural é reranker tardio.
- pgvector é extra/profile opt-in; nenhum import PostgreSQL no caminho SQLite.
- dados/colunas `holo_*` e seeds/bytes Hash permanecem compatíveis; FKs
  idempotentes são adicionadas/validadas nos três ports com preflight de órfãos.
- top-k/channel-k/pools são positivos e limitados; scores são finitos; empates usam id.
- falhas de reranker preservam a ordem; `diversity=none/0` é identidade.
- fingerprints incompatíveis falham explicitamente; fallback de encoder na V2 exige autorização.

## Task A — contratos, configuração e adapters existentes

**Requisitos:** RULE-001/002/007, IMPROVEMENT-001/006.

**Testes RED:** tipos/capabilities; precedência nova→legada→default; limites; fingerprint canônico/incompatível; SQLite sem extras; health/delete; fallback V2.

**Arquivos de escrita exclusivos:** `rag3d/backend.py` (novo), `rag3d/config.py`, `rag3d/engine.py`, `rag3d/store.py`, `rag3d/pgstore.py`, testes dedicados.

**Saída:** `RetrievalBackend` Protocol, `BackendCapabilities`, `SearchScope`, `SearchFilters`, `SearchDiagnostics`, `RetrievalStageResult`, `IndexFingerprint`; capabilities verdadeiras e seleção de backend compatível.

## Task B — pgvector exact/HNSW

**Requisitos:** RULE-003/004, IMPROVEMENT-002.

**Testes RED:** import opcional; extensão ausente; migration idempotente; add/get/delete; exact; HNSW; filtros parametrizados/limitados; `SET LOCAL`; health/version; `EXPLAIN` natural.

**Arquivos de escrita exclusivos:** `rag3d/pgvector_store.py` (novo), `migrations/pgvector/*.sql` (novo), `docker-compose.yml`, `pyproject.toml`, testes pgvector.

**Saída:** tabelas `rag3d_v2_*` separadas, busca IP/cosseno coerente, profile Docker pinado, extras Python e rollback por backend.

## Task C — pipeline V2, reranker e diversidade

**Requisitos:** RULE-001/002/004, IMPROVEMENT-003/004/005/007.

**Testes RED:** RRF equal-weight; structural subset da união; pipeline/order; legacy inalterada; query/corpus vazios; timings/contagens; NoOp/LLM/cross-encoder fallback; none/MMR/DPP; duplicatas/vetores ausentes/NaN; pools limitados.

**Arquivos de escrita exclusivos:** `rag3d/retrieval_v2.py` (novo), `rag3d/rerank.py`, `rag3d/diversity.py` (novo), alterações mínimas em `rag3d/fusion.py`, testes V2.

**Saída:** fluxo normalize→expand→encode→dense/sparse→union→fusion→structural→reranker→diversity→hydrate/small-to-big/stitch, com diagnostics sem conteúdo/DSN.

## Task D — avaliação e benchmark

**Requisitos:** RULE-005/006/007, IMPROVEMENT-008.

**Testes RED:** Recall/MRR/nDCG/percentis/duplicidade; bootstrap pareado determinístico; splits; JSON válido; benchmark não lê qrels antes de selecionar corpus; stage recall.

**Arquivos de escrita exclusivos:** `rag3d/evaluation.py` (novo), `benchmarks/run_retrieval_v2.py`, datasets pequenos versionáveis, testes de métricas.

**Saída:** runner configurável para legacy/v2 e backends disponíveis; qualidade, latência/QPS/RSS/ingest/index-size; resultados por query e IC95.

## Task E — documentação, revisão e entrega

**Requisitos:** RULE-008, IMPROVEMENT-009.

**Dependência:** Tasks A–D aprovadas e benchmarks reais concluídos.

**Arquivos:** grafos, ADR, arquitetura, guias, README/BENCHMARKS/CHANGELOG/.env.example, resultados e PR body.

**Gates:** spec review → quality/security review → testes completos → `git diff --check` → commits atômicos → push fork → PR upstream verificado.

## Rollback sistêmico

```bash
export RAG3D_RETRIEVAL_PIPELINE=legacy
export RAG3D_BACKEND=sqlite          # ou postgres-holo
export RAG3D_FUSION=quantum
```

As migrations pgvector são aditivas e separadas. As FKs holográficas também
são aditivas e permanecem após rollback; rollback de tráfego não apaga dados.

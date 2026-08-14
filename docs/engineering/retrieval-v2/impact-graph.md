# Grafo de impacto

Este grafo será atualizado a partir do diff final. Arestas indicam consumidores,
interfaces afetadas, testes e rollback; tabelas holográficas compartilhadas não
são alteradas.

```mermaid
flowchart TB
  BE[rag3d/backend.py] -->|é importado por| CFG[rag3d/config.py]
  BE -->|é implementado por| SQL[rag3d/store.py]
  BE -->|é implementado por| HOLO[rag3d/pgstore.py]
  BE -->|é implementado por| PGV[rag3d/pgvector_store.py]
  CFG -->|é lido por| ENG[rag3d/engine.py]
  ENG -->|instancia| SQL
  ENG -->|instancia| HOLO
  ENG -->|instancia| PGV
  ENG -->|instancia| RET[rag3d/retrieval_v2.py]
  RET -->|chama| DIV[rag3d/diversity.py]
  RET -->|chama| RER[rag3d/rerank.py]
  RET -->|mede com| EVAL[rag3d/evaluation.py]
  ING[rag3d/ingest.py] -->|grava via| BE
  MEM[rag3d/memory.py] -->|grava/lê via| BE
  PGV -->|grava| TBL[rag3d_v2_*]
  HOLO -->|preserva| HT[holo_*]
  HT -->|é consumido por| JS[rag3d-js]
  HT -->|é consumido por| JAVA[rag3d-java]

  T1[test_backend*] -->|valida| BE
  T2[test_retrieval_v2*] -->|valida| RET
  T3[test_pgvector*] -->|valida| PGV
  T4[npm test] -->|protege| JS
  T5[Java parity] -->|protege| JAVA
  BM[benchmark runner] -->|mede| RET
  BM -->|mede| PGV

  RB1[pipeline=legacy] -->|reverte| RET
  RB2[backend=postgres-holo] -->|reverte| PGV
  RB3[diversity=none] -->|reverte| DIV
```

| Área | Dependência anterior | Consumidor | Risco | Proteção | Rollback |
| --- | --- | --- | --- | --- | --- |
| Contratos | duck typing implícito | engine, ingest, memory, retriever | abstração excessiva | Protocol estrutural + contract tests | código legacy continua aceito |
| Configuração | DSN decide backend | CLI e `TriRag` | precedência/fallback | unitários de matriz de ambiente | remover flags e usar legado |
| SQLite | schema local | API Python | regressão offline | smoke + contract + reopen | `pipeline=legacy` |
| postgres-holo | `holo_*` compartilhado | Python/Node/Java | regressão cross-language | nenhum DDL destrutivo + parity | `backend=postgres-holo` |
| pgvector | novo namespace | V2 Python | planner/filtro/migration | PG real, exact ground truth, EXPLAIN | parar profile/backend |
| Pipeline V2 | recuperadores atuais | `search`, reader, memory | perda de recall | recall por estágio + flag | `pipeline=legacy` |
| Diversidade | DPP legado | top-k final | nDCG/MRR | `none` identidade + ablação | `diversity=none` |

# Evidence index — Retrieval Engine V2

| Evidência | Fonte | Sustenta | Limite |
| --- | --- | --- | --- |
| Baseline estruturado | `benchmarks/results/baseline.json` | estado anterior e falhas preexistentes | scripts históricos não medem IC/p95/nDCG |
| Smoke Python | `tests/test_smoke.py` executado no commit-base | API/ingest/retrieval/fusion/memória/persistência básicos | agregador manual, não pytest |
| Smoke Node | `npm test`, 8/8 no commit-base | comportamento local Node | não executa cross-language automaticamente |
| Paridade Java | `ParityCheck` compilado/executado no commit-base | matemática portátil Hash/holograma/fusão | não testa servidor PostgreSQL |
| Cartografia | relatório interno `AGENT-CARTOGRAPHER` | fluxo, schemas e divergências | análise read-only; não prova performance |
| Código estrutural | `rag3d/retrieve.py`, `store.py`, `pgstore.py` | MaxSim opera só na união | não mede recall perdido antes do pool |
| pgvector oficial | README/changelog oficial, consultado em 2026-08-13 | exact/HNSW, filtros e iterative scans | versão instalada deve ser confirmada |
| HNSW | Malkov & Yashunin, arXiv:1603.09320 | trade-off e ground truth | parâmetros ótimos dependem dos dados |
| RRF | Cormack, Clarke & Büttcher, SIGIR 2009 | baseline de rank fusion | `k0`/pesos ainda exigem validation |
| BEIR | Thakur et al., arXiv:2104.08663 | heterogeneidade e custo de reranking | corpus sintético local não substitui BEIR |
| DPP greedy | Chen, Zhang & Zhou, NeurIPS 2018 | Cholesky incremental e MAP difícil | greedy não é MAP global exato |
| Perfil legado | relatório interno `AGENT-PERFORMANCE` | structural+sparse = 74,38% do warm path observado; dense = 3,69% | Hash/SQLite sintético, quatro processos; diagnóstico, não claim |
| Auditoria RRF | relatórios internos `AGENT-FUSION` e `AGENT-IR-RESEARCH` | RRF local usa rank 1 e ausente zero; benchmark atual é weighted + structural condicionado | não prova superioridade de qualidade |
| Auditoria LSH/int8 | relatórios internos `AGENT-LSH-MATH` e `AGENT-EMBEDDINGS` | bandas usam 128/1024 bits; quantização pode alterar top-k | stress sintético, não corpus de produção |
| Auditoria de diversidade | relatório interno `AGENT-DIVERSITY` | greedy não é MAP global; kernel holo implícito pode ser indefinido | proposta V2 ainda requer implementação/teste |
| Auditoria de avaliação | relatório interno `AGENT-EVALUATION` | `--max_docs` tem leakage e nDCG atual aceita duplicatas | protocolo público completo ainda não executado |
| Integração PG legada | `tests/test_pg.py` em PostgreSQL 17.6 local | holo persiste/busca e o caminho de facetas executa | execução única; valores de tempo não são benchmark |

Evidências de implementação, testes, benchmark e PR serão anexadas somente depois de produzidas e verificadas.

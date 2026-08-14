# Rastreabilidade — Retrieval Engine V2

| Requisito | Subagente | Componente | Teste | Benchmark | Evidência | Status |
| --- | --- | --- | --- | --- | --- | --- |
| RULE-001 | AGENT-CARTOGRAPHER; AGENT-JS-JAVA-COMPAT | API/legacy/schema holo | smoke + cross-language | modos legacy | context manifest/cartografia | em andamento |
| RULE-002 | AGENT-BACKEND-ARCHITECT | contratos/capabilities/limites/fingerprint | unit + contract | stage recall | pendente | em andamento |
| RULE-003 | AGENT-PGVECTOR; AGENT-SECURITY | adapter/migrations | integração PG/segurança | exact/HNSW/filtros | pendente | em andamento |
| RULE-004 | AGENT-IR-RESEARCH; AGENT-HNSW-MATH; AGENT-LSH-MATH; AGENT-LATE-INTERACTION | fusion/ANN/LSH/structural | testes matemáticos | ablações | fontes primárias | em andamento |
| RULE-005 | AGENT-EVALUATION | runner estatístico | testes métricas/bootstrap | relatório comparativo | pendente | em andamento |
| RULE-006 | AGENT-PERFORMANCE | diagnostics/profiling | unit diagnostics | p50/p95/p99/QPS/RSS | baseline | em andamento |
| RULE-007 | AGENT-TESTS; AGENT-JS-JAVA-COMPAT | suíte/CI local | pytest/Node/Java/PG | quality gate | baseline | em andamento |
| RULE-008 | AGENT-GIT-PR | Git/PR | diff checks | resultados no PR | pendente | em andamento |
| IMPROVEMENT-001 | AGENT-BACKEND-ARCHITECT | backend contract/adapters | contract tests | modos por backend | pendente | em andamento |
| IMPROVEMENT-002 | AGENT-PGVECTOR; AGENT-HNSW-MATH | pgvector/HNSW | integration/explain | frontier ANN | pendente | em andamento |
| IMPROVEMENT-003 | AGENT-FUSION | pipeline V2/RRF | RRF/legacy/quantum | ablation | pendente | em andamento |
| IMPROVEMENT-004 | AGENT-LATE-INTERACTION; AGENT-LSH-MATH | structural stage | stage recall/MaxSim | structural ablation | cartografia | em andamento |
| IMPROVEMENT-005 | AGENT-DIVERSITY | none/MMR/DPP | propriedades/estabilidade | coverage/duplicate rate | pendente | em andamento |
| IMPROVEMENT-006 | AGENT-EMBEDDINGS | fallback/fingerprint | unit/cross-language | quantização | pendente | em andamento |
| IMPROVEMENT-007 | AGENT-PERFORMANCE | diagnostics | redaction/bounds/timings | latency decomposition | pendente | em andamento |
| IMPROVEMENT-008 | AGENT-EVALUATION; AGENT-PERFORMANCE | benchmark runner | metric tests | JSON/report/IC95 | baseline | em andamento |
| IMPROVEMENT-009 | AGENT-CARTOGRAPHER; AGENT-GIT-PR | docs/rollout/rollback | link/config checks | results | docs iniciais | em andamento |

“Concluído” exige relatório do subagente, código/decisão, teste, evidência e revisão independente nos requisitos de alto risco.

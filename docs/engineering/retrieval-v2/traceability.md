# Rastreabilidade — Retrieval Engine V2

| Requisito | Subagente | Componente | Teste | Benchmark | Evidência | Status |
| --- | --- | --- | --- | --- | --- | --- |
| RULE-001 | AGENT-CARTOGRAPHER; AGENT-JS-JAVA-COMPAT | API/legacy/schema holo | smoke; Node 24/24; Java parity/contract | legacy nos três backends | context manifest; review independente | concluído |
| RULE-002 | AGENT-BACKEND-ARCHITECT; AGENT-SECURITY | contratos/capabilities/limites/fingerprint | backend types/adapters/boundaries | lineage e pools nos JSONs | 89,09% linhas; 80,14% branches novos; review aprovado | concluído |
| RULE-003 | AGENT-PGVECTOR; AGENT-SECURITY | adapter/migration `rag3d_v2_*` | 34 integrações pgvector; migration twice/drift/locks | exact + grade HNSW | EXPLAIN natural; HNSW rejeitado por recall | concluído com HNSW experimental |
| RULE-004 | AGENT-IR-RESEARCH; AGENT-HNSW-MATH; AGENT-LSH-MATH; AGENT-LATE-INTERACTION | RRF/ANN/LSH/structural | fusion/diversity/retrieval properties | ablações e frontier ANN | fontes primárias registradas; claims limitados | concluído |
| RULE-005 | AGENT-EVALUATION; AGENT-EVALUATION-REVIEW | runner estatístico | 79 focais de avaliação + suíte global | validation lock -> test; IC95 | test JSON clean e lock canônico | concluído |
| RULE-006 | AGENT-PERFORMANCE; AGENT-OPTIMIZE-1/2; AGENT-VERIFY-1/2 | diagnostics/profiling/tuning | diagnostics e bounds | p50/p95/p99; ingest; RSS; storage | batching e HNSW rejeitados sem esconder regressões | concluído |
| RULE-007 | AGENT-TESTS; AGENT-JS-JAVA-COMPAT; AGENT-COVERAGE-GATE-FINAL | suíte/CI | Python 876; PG real; Node 24; Java checks | quality gates | workflow CI e cobertura consolidada | concluído |
| RULE-008 | AGENT-GIT-PR | commits/push/PR | diff checks e secret scan | artefatos no PR | 9 commits; branch no fork; [PR #1](https://github.com/fabiofernandesss/Rag3D/pull/1) verificado | concluído |
| IMPROVEMENT-001 | AGENT-BACKEND-ARCHITECT; AGENT-ADAPTERS-IMPL | backend contract/adapters | contract suite comum | modos por backend | capabilities truthfully testadas | concluído |
| IMPROVEMENT-002 | AGENT-PGVECTOR; AGENT-HNSW-MATH; AGENT-PGVECTOR-IMPL | pgvector exact/HNSW | integração/EXPLAIN/catálogo | exact 1k/10k; H100--H1000 | exact aceito; todos HNSW rejeitados | concluído com limitação registrada |
| IMPROVEMENT-003 | AGENT-FUSION; AGENT-PIPELINE-IMPL | pipeline V2/RRF/quantum | RRF properties; quantum parity | ablação RRF/quantum | RRF default apenas em V2 | concluído |
| IMPROVEMENT-004 | AGENT-LATE-INTERACTION; AGENT-LSH-MATH | structural late rerank | structural bounds/MaxSim/lineage | ablação structural | classificado e medido como reranker | concluído |
| IMPROVEMENT-005 | AGENT-DIVERSITY | none/MMR/greedy-DPP | finitude/PSD/ordem/fallback | MMR/DPP ablations | sem claim MAP global | concluído |
| IMPROVEMENT-006 | AGENT-EMBEDDINGS | fallback/fingerprint | encoder/fingerprint/xlang | stress int8 documentado | fallback V2 fail-closed; Hash-only xlang | concluído |
| IMPROVEMENT-007 | AGENT-PERFORMANCE | diagnostics/redaction/bounds | stage outcomes e segurança | decomposição de latência | nenhum conteúdo/DSN/embedding nos JSONs | concluído |
| IMPROVEMENT-008 | AGENT-EVALUATION; AGENT-PERFORMANCE | benchmark runner | métricas/bootstrap/lock | JSON 1k/10k/backends/ablações | relatório final; meta20 não atingida | concluído |
| IMPROVEMENT-009 | AGENT-CARTOGRAPHER; AGENT-GIT-PR | docs/rollout/rollback | links/config/diff checks | resultados versionados | arquitetura, ADR, guias e rollback | concluído |

“Concluído” exige relatório do subagente, código/decisão, teste, evidência e revisão independente nos requisitos de alto risco.

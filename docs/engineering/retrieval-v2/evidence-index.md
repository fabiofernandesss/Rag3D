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
| Auditoria de avaliação | relatórios internos `AGENT-EVALUATION` e `AGENT-EVALUATION-REVIEW` | encontrou leakage/duplicatas; seleção e métricas foram corrigidas | BEIR histórico continua diagnóstico não elegível |
| Integração PG legada | `tests/test_pg.py` em PostgreSQL 17.6 local | holo persiste/busca e o caminho de facetas executa | valores impressos pelo script não são benchmark |
| Contrato/backend V2 | `rag3d/backend.py`; testes types/adapters/boundaries | capabilities, filtros, fingerprints, bounds e erros seguros | ports externos devem respeitar o mesmo contrato |
| Pipeline V2 | `rag3d/retrieval_v2.py`; 1.530 linhas de testes focais | ordem de estágios, fallback explícito, limites e lineage | rerankers de modelo são opcionais e não foram avaliados externamente |
| PostgreSQL/pgvector real | suíte final: 876 passed; 34 integrações pgvector | migration, CRUD, exact, HNSW, restart, locks, filtros e plano natural | servidor local 17.6/pgvector 0.8.5 |
| Cobertura do delta | `/tmp/rag3d-coverage-final.json`; relatório AGENT-COVERAGE-GATE-FINAL | cinco módulos novos: 2.539/2.850 linhas e 1.013/1.264 branches | arquivo `/tmp` é evidência local; contagens reproduzíveis pela CI |
| Compatibilidade Node/Java | `npm test` 24/24; `ParityCheck`; `PgHoloStoreContractCheck` | Hash/legacy, limites e schema holo | V2 full fingerprint é Python-only e bloqueia ports legados |
| Teste bloqueado | `benchmarks/results/retrieval-v2-test-1k.json` + validation lock | nDCG/Recall/MRR/p50/p95/p99/IC em split não usado no tuning | 7 consultas sintéticas; sem generalização |
| Calibração 10k | `retrieval-v2-calibration-10k.json` e pgvector exact 10k | escala, ingestão, latência e tamanho | calibration; QPS/RSS não claimable |
| Ablações | `retrieval-v2-calibration-ablations-1k.json` | dense, sparse, RRF, quantum, structural, MMR e DPP | corpus sintético pequeno |
| Fronteira HNSW | JSONs `hnsw-ef{100,200,400,800,1000}-10k` | exact ground truth, plano natural, recall-latência | sem filtros; todos os pontos rejeitados pelo gate por consulta |
| Revisão independente | relatório interno `AGENT-INDEPENDENT-REVIEWER` | código spec-compliant após correções adversariais | relatório interno não é versionado |
| Meta de 20% | `docs/benchmarks/retrieval-v2-results.md` | cálculo, IC, guardrails e regressões | meta não atingida |
| Entrega upstream | [PR #2](https://github.com/fabiofernandesss/Rag3D/pull/2) | branch `rogerin:feat/retrieval-engine-v2-clean-20260815` para `main`, estado OPEN verificado | substitui o PR #1 após replay sem force-push e histórico sem findings locais |

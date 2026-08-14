# Retrieval Engine V2 — baseline anterior à implementação

Data da medição: 2026-08-13. Commit: `d39604dba98cbbf9d4f9c67781abf211aeb6a295` (`upstream/main`). Nenhum módulo de produção havia sido alterado quando estes comandos foram executados.

## Ambiente

| Item | Valor |
| --- | --- |
| Hardware | MacBook Pro Mac15,6, Apple M3 Pro, 11 cores, 18 GB RAM |
| Sistema | macOS 26.5.2, Darwin 25.5.0, arm64 |
| Python / NumPy | 3.14.5 / 2.5.1 |
| Node / npm | 25.2.1 / 11.6.2 |
| Java | OpenJDK 17.0.17 |
| PostgreSQL | cliente 17.6; servidor não iniciado |
| pgvector | pacote host 0.8.5 presente; sem servidor carregado/backend testado no instante inicial |
| Encoder | fallback por hashing, dimensões 1024/128 |

## Execuções existentes

| Comando | Resultado real |
| --- | --- |
| `PYTHONPATH=$PWD python3 tests/test_smoke.py` | sucesso; todas as asserções do script passaram; 2,0041 s de parede |
| `PYTHONPATH=$PWD python3 tests/bench_fusion.py 1800` | sucesso; 1.800 documentos, 120 consultas; 1,2481 s de parede |
| `PYTHONPATH=$PWD python3 tests/bench_coverage.py 20 8` | sucesso; 20 tópicos, 220 documentos, 20 consultas; 1,0174 s de parede |
| `PYTHONPATH=$PWD python3 tests/test_pg.py` | código 2; testes matemáticos passaram, integração indisponível sem `psycopg` |
| `cd rag3d-js && npm test` | 8 passaram, 0 falharam |
| `javac ... ParityCheck.java && java ... ParityCheck` | sucesso; executável de paridade portátil concluiu |
| `PYTHONPATH=$PWD python3 -m pytest -q` | indisponível: `pytest` não estava instalado |

Após o registro imutável acima, mas ainda antes de qualquer mudança no código
de produto, foi provisionado um cluster PostgreSQL 17.6 local e isolado com
pgvector 0.8.5, Psycopg 3.3.4 e o cliente Python pgvector 0.5.0. A suíte
holográfica legada passou integralmente, inclusive no caminho de 400
documentos. Seus valores impressos em execução única (0,3 s de ingestão, 7 ms
por consulta e 6 ms com pré-filtro) são somente diagnóstico de integração: não
houve repetições nem intervalo de confiança e eles não sustentam claim de
performance.

## Qualidade observada

No corpus sintético multilíngue de 1.800 documentos, dense, structural, CombSUM, quantum e RRF obtiveram `Recall@5 = 83,33%` e `MRR = 0,833`; sparse obteve `Recall@5 = 83,33%` e `MRR = 0,750`. Esse conjunto não separa as estratégias e não sustenta uma alegação de superioridade.

No corpus sintético de redundância, o ranking puro obteve `coverage@6 = 46,7%`, RRF puro `48,3%`, DPP 0,5 `100%` e RRF + DPP 0,5 `100%`; todos mantiveram rank-1 útil em `100%`. A métrica mede cobertura de fatos nesse cenário controlado, não qualidade global de RAG.

## Lacunas do baseline histórico

Os scripts anteriores não exportavam nDCG@10, Recall@20, distribuição de latência por consulta, QPS, RSS, tamanho do índice, throughput de ingestão nem intervalo de confiança pareado. Essas lacunas serão preenchidas pelo runner V2; não serão reconstruídas retroativamente nem tratadas como zero.

O registro estruturado e os valores completos estão em [`benchmarks/results/baseline.json`](../../benchmarks/results/baseline.json).

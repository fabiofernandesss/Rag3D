# Retrieval Engine V2 — resultados reproduzíveis

Data: 2026-08-14 (America/Recife). Código medido: commit
`11c2fd6fa4eadf9cc4f4b6f842eedd51824c360d`. Todas as execuções finais
registram `run.dirty=false`; os commits posteriores contêm somente estes
artefatos e o fechamento documental.

## Conclusão

A meta de 20% **não foi atingida**. No split de teste bloqueado, a V2 reduziu
a latência p95 SQLite em 11,05%, mas nDCG@10 caiu 1,79% e MRR@20 caiu 15,93%.
O ganho amostral de QPS foi 18,45%, abaixo de 20% e explicitamente não elegível
a claim. Em calibração, resultados melhores não são promovidos a evidência de
teste.

O pgvector exact funciona e permanece o modo seguro. A grade HNSW usou plano
natural em 35/35 probes, mas nenhum valor de `ef_search` passou o gate de
Recall ANN@20 >= 0,98 por consulta. HNSW permanece experimental.

## Ambiente e protocolo

| Item | Valor |
| --- | --- |
| Hardware | Apple M3 Pro, arm64, 11 CPUs lógicas, 18 GiB RAM |
| Sistema | macOS 26.5.2, Darwin 25.5.0 |
| Python / NumPy | 3.14.5 / 2.5.2 |
| PostgreSQL / pgvector | 17.6 / 0.8.5 |
| Psycopg / cliente pgvector | 3.3.4 / 0.5.0 |
| Dataset | `retrieval-v2-synthetic-v1` 1.0.0 |
| Seed dataset/bootstrap | 20260813 / 20260813 |
| Split principal | validation -> lock -> test, disjuntos por hash antes dos qrels |
| Corpus principal | 1.000 documentos/chunks; 7 consultas, 5 com qrels positivos |
| Repetições | 5 warm-ups + 10 repetições por consulta |
| IC | bootstrap pareado por consulta, 10.000 amostras, 95% |
| Ordem | intercalada e invertida em repetições alternadas |
| Encoder | Hash, dense 128, structural 32 x 64 |
| Retrieval | top_k 20, channel_k 100, structural depth 100, RRF k=60 |

O lock de teste registra `source_diff_sha256` igual ao SHA-256 do conteúdo
vazio, closure de todos os módulos Python e identidade canônica do dataset. O
arquivo de teste registra `config_sha256=6471768f14e99b3c932648a837fc3f53be2ded4371551209602472fd1bd63213`.

QPS, RSS e cold latency são diagnósticos: os sistemas compartilham processo,
a janela serial tem menos de cinco segundos e não há isolamento. Por isso o
JSON mantém `qps=null`, `claim_eligible=false`; a coluna abaixo mostra apenas
`successful_qps_sample`. RSS é high-water mark compartilhado e não sustenta
comparação de memória.

## Resultado bloqueado de teste — SQLite, 1k

| Pipeline | nDCG@10 | Recall@20 | MRR@20 | p50 ms | p95 ms | p99 ms | QPS amostral | RSS pico | Índice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy + SQLite | 0,820414 | 1,000000 | 0,866667 | 6,096 | 9,378 | 12,768 | 160,77 | 60.735.488 B | 8.647.024 B |
| V2 + SQLite | 0,805748 | 1,000000 | 0,728571 | 5,033 | 8,342 | 11,427 | 190,42 | 60.768.256 B | 8.647.024 B |

Diferenças relativas V2 contra legacy:

- nDCG@10: -1,7876%; IC95 [-22,1442%, +17,1416%];
- Recall@20: 0%; IC95 [0%, 0%];
- MRR@20: -15,9341%; IC95 [-39,0110%, 0%];
- latência p95: -11,0458% (melhora descritiva, sem IC pareado específico de
  razão de p95); o IC95 clusterizado da p95 é [7,102, 11,017] ms no legacy e
  [6,502, 10,561] ms na V2;
- QPS amostral: +18,4451%, não elegível a claim.

A queda de MRR viola o guardrail de 2%, independentemente da latência.

## Calibração por backend — 1k

Estes valores servem para diagnóstico e escolha de configuração, não para
claim. Cada par reutiliza o mesmo índice e corpus dentro da execução.

| Pipeline | Backend | nDCG@10 | Recall@20 | MRR@20 | p95 ms | QPS amostral | Índice |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | postgres-holo | 0,944011 | 1,000000 | 1,000000 | 11,706 | 103,59 | 4.784.128 B |
| V2 | postgres-holo | 0,983944 | 1,000000 | 1,000000 | 10,731 | 123,29 | 4.784.128 B |
| legacy | pgvector exact | 0,923212 | 1,000000 | 1,000000 | 12,034 | 109,83 | 27.746.304 B |
| V2 | pgvector exact | 1,000000 | 1,000000 | 1,000000 | 10,039 | 135,13 | 27.746.304 B |

Na calibração pgvector exact, a mediana por consulta melhorou 21,40% (IC95
[15,44%, 36,13%]), mas p95 melhorou 16,58%; o split é calibration e o QPS é
não elegível. Portanto isso não satisfaz a meta.

## Escala de 10k

| Pipeline | Backend/modo | nDCG@10 | Recall@20 | MRR@20 | p95 ms | Ingest chunks/s | Índice |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | SQLite | 0,907865 | 1,000000 | 1,000000 | 88,812 | 498,86 | 49.037.968 B |
| V2 | SQLite | 0,990289 | 1,000000 | 1,000000 | 94,064 | 498,86 | 49.037.968 B |
| legacy | pgvector exact | 0,907865 | 1,000000 | 1,000000 | 71,213 | 998,70 | 62.816.256 B |
| V2 | pgvector exact | 0,990289 | 1,000000 | 1,000000 | 71,134 | 998,70 | 62.816.256 B |

Em SQLite 10k, nDCG@10 subiu 9,08% (IC95 [1,64%, 17,69%]), Recall@20 e
MRR ficaram estáveis, e p95 piorou 5,91%. É calibration sintética e não alcança
20%.

## Fronteira HNSW — 10k, sem filtros

Configuração fixa: `m=16`, `ef_construction=128`, `iterative_scan=off`,
`max_scan_tuples=20000`, `scan_mem_multiplier=1`. Apenas `ef_search` variou.
Busca exact no mesmo snapshot foi o ground truth. O planner escolheu HNSW
naturalmente em 35/35 probes; não houve `enable_seqscan=off`.

| ef_search | Recall ANN@20 médio | Mínimo por consulta | Status do gate | V2 p95 ms | V2 nDCG@10 |
| ---: | ---: | ---: | --- | ---: | ---: |
| 100 | 0,742857 | 0,05 | falhou | 84,020 | 0,955201 |
| 200 | 0,878571 | 0,55 | falhou | 70,660 | 0,990289 |
| 400 | 0,885714 | 0,40 | falhou | 74,851 | 0,990289 |
| 800 | 0,985714 | 0,95 | falhou | 61,433 | 0,990289 |
| 1000 | 0,992857 | 0,95 | falhou | 75,923 | 0,990289 |

O menor recall por consulta não atingiu 0,98 em nenhum ponto. `ef=800`
reduziu p95 em 13,64% contra exact (71,134 ms), ainda abaixo de 20%, sem IC
inter-run válido e com falha de recall. Tamanho físico e ingestão variaram entre
execuções por bloat/cache do banco reutilizado; não são usados em claim de
armazenamento. Filtros por seletividade não foram medidos por este runner; a
integração cobre filtros e plano, mas não sustenta uma fronteira filtrada.

## Ablações — SQLite 1k calibration

| Variante V2 | nDCG@10 | Recall@20 | MRR@20 | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| pipeline completa (RRF + structural) | 1,000000 | 1,000000 | 1,000000 | 13,617 |
| dense only | 0,131998 | 0,566667 | 0,100087 | 2,021 |
| sparse only | 0,962913 | 1,000000 | 1,000000 | 9,208 |
| RRF sem structural | 0,916448 | 1,000000 | 1,000000 | 9,788 |
| quantum | 0,953363 | 1,000000 | 1,000000 | 9,115 |
| RRF + structural | 1,000000 | 1,000000 | 1,000000 | 9,998 |
| RRF + structural + MMR | 0,883568 | 1,000000 | 1,000000 | 12,522 |
| RRF + structural + greedy-DPP | 0,697691 | 0,733333 | 1,000000 | 12,303 |

Não há labels nem execução de modelo para cross-encoder/LLM no dataset
congelado; as interfaces e fallbacks são testados, mas essas variantes não têm
claim de qualidade.

## Ciclos de otimização

1. Batching da consulta sparse SQLite: rejeitado. Rankings ficaram idênticos,
   mas o candidato piorou sparse em 12,79% e total em 6,22% na medição do
   implementador; o código foi revertido. Permaneceu somente um limite
   defensivo compartilhado de 8.192 termos, sem claim de performance.
2. Tuning HNSW `ef_search`: rejeitado por recall por consulta. Exact permanece
   seguro; nenhuma configuração HNSW foi promovida.

## Meta de 20%

Baseline formal: `legacy + SQLite`, split test, 1.000 chunks. Candidato:
`V2 + SQLite` no mesmo índice e ordem balanceada.

`ganho_relativo = (novo - baseline) / baseline` para métricas higher-is-better;
para latência, `(baseline - novo) / baseline`.

Resultado: **meta atingida: não**. A evidência externa/heterogênea também não
foi executada; dados sintéticos não autorizam generalização.

## Artefatos

- [`retrieval-v2-validation-lock.json`](../../benchmarks/results/retrieval-v2-validation-lock.json)
- [`retrieval-v2-validation-1k.json`](../../benchmarks/results/retrieval-v2-validation-1k.json)
- [`retrieval-v2-test-1k.json`](../../benchmarks/results/retrieval-v2-test-1k.json)
- [`retrieval-v2-calibration-ablations-1k.json`](../../benchmarks/results/retrieval-v2-calibration-ablations-1k.json)
- [`retrieval-v2-calibration-10k.json`](../../benchmarks/results/retrieval-v2-calibration-10k.json)
- [`retrieval-v2-calibration-postgres-holo-1k.json`](../../benchmarks/results/retrieval-v2-calibration-postgres-holo-1k.json)
- [`retrieval-v2-calibration-pgvector-exact-1k.json`](../../benchmarks/results/retrieval-v2-calibration-pgvector-exact-1k.json)
- [`retrieval-v2-calibration-pgvector-exact-10k.json`](../../benchmarks/results/retrieval-v2-calibration-pgvector-exact-10k.json)
- `retrieval-v2-calibration-pgvector-hnsw-ef{100,200,400,800,1000}-10k.json`

## Limitações

- Apenas sete consultas sintéticas por split; cinco possuem qrels positivos.
- BEIR/NFCorpus/SciFact e avaliação RAG com respostas/citações não foram
  executados; `citation_precision`, `no_answer_accuracy` e ARES permanecem
  `null` sem labels adequados.
- QPS, RSS e cold-cache são diagnósticos não elegíveis a claim.
- A grade HNSW não mede seletividades de filtro, partial indexes nem
  particionamento; essas estratégias permanecem não avaliadas quantitativamente.
- Docker Compose foi validado estaticamente com `docker-compose config`; o
  daemon Docker não estava disponível para subir o profile.
- Python 3.9 foi validado por parsing de gramática; não havia runtime 3.9 local.

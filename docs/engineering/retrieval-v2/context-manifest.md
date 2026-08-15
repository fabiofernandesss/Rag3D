# Retrieval Engine V2 — context manifest

## Proveniência

| Campo | Valor |
| --- | --- |
| Commit-base | `d39604dba98cbbf9d4f9c67781abf211aeb6a295` |
| Branch-base | `upstream/main` |
| Branch de trabalho | `feat/retrieval-engine-v2` |
| Diretório | raiz do checkout RAG3D |
| CodeGraph | ausente; descoberta feita com `rg`, leitura numerada e inventário de símbolos |
| Instruções locais | não há `AGENTS.md`, `.harness/`, `CONTRIBUTING.md` ou CI no commit-base; existe `.claude/launch.json.example` |

## Árvore relevante

```text
rag3d/                 implementação Python e API pública
  engine.py            fachada TriRag/Rag3D e escolha de backend
  config.py            configuração e precedência RAG3D_*/TRIRAG_*
  retrieve.py          pipeline legado
  store.py             SQLite + NumPy
  pgstore.py           PostgreSQL holográfico sem extensão
  encoders.py          BGE-M3 opcional e Hash fallback
  fusion.py            quantum, RRF e greedy-DPP
  holo.py              LSH, quantização e MaxSim binário
  ingest.py            chunking, batch encoding e small-to-big
  rerank.py            LLM listwise
tests/                  scripts smoke, PG, ablação e cross-language
rag3d-js/               port Node; memória/JSON ou PostgreSQL holográfico
rag3d-java/             port Java 17; PostgreSQL holográfico
docker-compose.yml      PostgreSQL 16 comum na porta 5433
pyproject.toml          Python >=3.9; NumPy; extras BGE/PostgreSQL
```

Não existiam `schemas/`, `migrations/`, `benchmarks/`, `.github/workflows/` nem suíte pytest formal no commit-base.

## Linguagens, versões e dependências

| Superfície | Contrato declarado | Dependências principais |
| --- | --- | --- |
| Python | `>=3.9`, pacote 0.1.0 | `numpy>=1.24`; `FlagEmbedding>=1.2` opcional; `psycopg[binary]>=3.1` opcional |
| Node | `>=18`, pacote 0.1.0 | `pg` opcional; testes com `node:test` |
| Java | Java 17, artefato 0.1.0 | JDBC PostgreSQL 42.7.4; Spring Boot 3.3.5 opcional |
| Banco existente | PostgreSQL 16 no Compose | somente tipos nativos no backend holográfico |

O baseline local foi medido em Python 3.14.5/NumPy 2.5.1, Node 25.2.1, OpenJDK 17.0.17 e macOS arm64; detalhes em [`docs/benchmarks/retrieval-v2-baseline.md`](../../benchmarks/retrieval-v2-baseline.md).

## Entradas públicas e componentes

- Python exporta `Rag3D`, `Rag3DConfig`, `TriRag` e `TriRagConfig` em `rag3d/__init__.py`; a fachada oferece `ingest`, `ingest_file`, `search`, `ask`, `chat` e `stats`.
- `TriRetriever.search` é a entrada de retrieval legado; `TriResult` contém `query`, `fused`, `views` e `stats`.
- `TriStore` fornece SQLite sem servidor. `PgHoloStore` fornece PostgreSQL holográfico por duck typing; ainda não há Protocol ou capabilities.
- `HashEncoder` é portátil entre Python, Node e Java. `Bgem3Encoder` existe somente em Python.
- `fuse` seleciona quantum ou RRF; `fermionic_select` é diversidade gulosa com atualização incremental em float64.
- `Reranker` é listwise via LLM; exceção/parse vazio preserva a ordem, mas uma resposta válida incorreta pode piorá-la.
- Node usa `MemStore`/JSON sem PostgreSQL, não SQLite. Java oferece somente o schema holográfico.

## Backends e schemas atuais

| Backend | Persistência | Dense | Sparse | Estrutural | Observação |
| --- | --- | --- | --- | --- | --- |
| SQLite | `meta`, `docs`, `chunks`, `dvecs`, `postings`, `colvecs` | produto interno exato sobre matriz float32 em RAM | soma TF/IDF-like por postings | MaxSim float sobre candidatos | zero servidor |
| PostgreSQL holográfico | `holo_meta`, `holo_docs`, `holo_grams`, `holo_spectrum` | Hamming 1024-bit, depois eco int8 | postings em tabela B-tree | MaxSim binário 128-bit sobre candidatos | sem pgvector |
| Node local | memória/JSON | força bruta | força bruta | candidatos | não compartilha SQLite |

Os embeddings produzidos pelo código Python são L2-normalizados; para eles, produto interno e similaridade cosseno preservam a mesma ordenação. O sparse atual não é BM25 completo: não possui normalização por comprimento nem parâmetros `k1`/`b`.

## Fluxo real e invariantes

O retrieval real possui dois geradores globais de candidatos: dense e sparse. O estrutural recebe apenas `union(dense,sparse)` e, portanto, é late-interaction reranking. Ele não consegue recuperar um item ausente da união.

Invariantes públicos a preservar:

- API/aliases Python e modo retrieval-only;
- SQLite sem servidor e PostgreSQL holográfico sem pgvector;
- schema holográfico, seeds e bytes compartilhados por Hash em Python/Node/Java;
- pipeline legado e campos quantum `classical`, `interference`, `channels`, `per_channel`;
- ordenação determinística, persistência, LLM opcional, diversidade desligável;
- consulta/corpus vazios seguros, pools limitados, scores finitos e fallback técnico de reranker.

## Testes e benchmarks existentes

- `tests/test_smoke.py`: script manual; não é uma suíte pytest verdadeira.
- `tests/test_pg.py`: matemática holográfica e integração PostgreSQL, dependente de `psycopg`/servidor.
- `tests/bench_fusion.py`: corpus sintético multilíngue, Recall@5/MRR.
- `tests/bench_coverage.py`: cobertura de fatos sob redundância.
- `tests/beir_ablation.py`: BEIR/BGE-M3, mas sem splits calibration/validation/test próprios.
- `rag3d-js/test/smoke.test.mjs`: oito testes descobertos por `npm test`; scripts cross-language são separados.
- Java possui executáveis `ParityCheck`/`XlangCheck`, não JUnit.

## Divergências verificadas

1. Documentação recomenda RRF para produção, mas o default legado é quantum em Python/Node/Java.
2. O estrutural é descrito em alguns trechos como recuperador independente, mas só pontua a união dense+sparse.
3. O comentário de transação do backend PG Python não corresponde ao autocommit por operação.
4. “Reranker nunca piora” vale apenas para falha técnica; saída válida errada pode degradar o ranking.
5. O greedy-DPP não é solução MAP global exata.
6. “Eco int8 exato/erro <1%” não tem distribuição de erro nem Recall@k publicados.
7. Apenas 128 dos 1024 bits alimentam as 16 bandas de oito bits.
8. Paridade cross-language está comprovada para Hash, não para BGE-M3; Java não possui guard de fingerprint.
9. POM/package/pyproject permanecem em 0.1.0 enquanto o changelog abre em 0.2.0.
10. O DDL Java omite `idx_grams_doc`, presente em Python e Node.

## Riscos de mudança

- Alterar `holo_*`, seeds, hash, dimensões, byte order ou desempate quebra o índice compartilhado.
- Escolher pgvector implicitamente tornaria SQLite dependente de extras/servidor.
- Filtros pós-HNSW podem perder recall; exact search deve ser ground truth.
- Fingerprint incompleto permite consultar índice incompatível silenciosamente.
- Pools/arrays sem limite podem amplificar memória, placeholders e tempo de banco.
- O bypass de corpus pequeno ignora retrieval e precisa ser desativado/registrado em benchmarks.

## Estado integrado no fechamento

Código medido no benchmark: `11c2fd6fa4eadf9cc4f4b6f842eedd51824c360d`.
O delta final adiciona os seguintes componentes sem substituir o caminho
legacy:

| Superfície | Estado integrado | Evidência |
| --- | --- | --- |
| Contrato | `RetrievalBackend`, capabilities, filters, diagnostics, fingerprint e limites comuns | `tests/test_backend_types.py`, adapters e boundaries |
| SQLite | adapter legacy conformado, transações/savepoints, bounds e ordenação determinística | contract tests e smoke |
| PostgreSQL holo | adapter conformado, locks/timeout, FKs verificadas, sparse canônico e rollback | PostgreSQL 17.6 real + contratos Node/Java |
| pgvector | tabelas `rag3d_v2_*`, exact, HNSW, filtros, health, catálogo e migration idempotente | 34 testes de integração pgvector e EXPLAIN natural |
| Pipeline V2 | dense + sparse -> RRF -> structural late rerank -> reranker -> diversidade -> hidratação/stitch | testes de lineage, properties e fachada E2E |
| Avaliação | métricas deduplicadas, bootstrap pareado, validation lock, runner 1k/10k e ablações | JSONs em `benchmarks/results/` |
| Cross-language | caminho Hash legacy preservado; Node/Java recusam certificação V2 que não implementam | 24 testes Node e checks Java |
| CI | Python/PostgreSQL/pgvector, Node e Java | `.github/workflows/retrieval-v2.yml` |

A suíte final executou 876 testes Python com PostgreSQL/pgvector real. Nos
cinco módulos novos foram cobertas 2.539/2.850 linhas (89,09%) e 1.013/1.264
branches (80,14%). Node passou 24/24; Java compilou e passou `ParityCheck` e
`PgHoloStoreContractCheck`.

As divergências 1--5 e 8--10 acima foram tratadas pelo contrato, pelos defaults
condicionais, pela classificação explícita do structural, pelas transações e
pela documentação. As afirmações de erro int8 inferior a 1%, superioridade
global e paridade BGE-M3 continuam deliberadamente rejeitadas por falta de
evidência. A meta de 20% não foi atingida; consulte
[`docs/benchmarks/retrieval-v2-results.md`](../../benchmarks/retrieval-v2-results.md).

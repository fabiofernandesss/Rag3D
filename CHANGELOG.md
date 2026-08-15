# Changelog — RAG3D

Histórico das mudanças, da mais recente para a mais antiga. Cada entrada diz o
que mudou, **por quê** e **como foi verificado**.

---

## [Unreleased] — Retrieval Engine V2 (opt-in)

- Adiciona contrato tipado/capability-aware para SQLite, PostgreSQL holográfico
  e pgvector opcional, mantendo dados/colunas `holo_*` e o pipeline legado.
- Adiciona e valida FKs `holo_*` idempotentes nos ports Python, Node e Java
  para impedir órfãos; índices legados com órfãos falham fechado no startup e
  exigem preflight/correção administrativa antes do rollout.
- Adiciona pipeline V2 com dense+sparse RRF equal-weight, structural como
  late-interaction reranker, rerankers opcionais e diversidade
  `none`/MMR/greedy-DPP.
- Adiciona fingerprint fail-closed, fallback de encoder explícito, transações
  por documento, limites e diagnósticos seguros por estágio.
- Adiciona schema/migration `rag3d_v2_*`, busca pgvector exact e HNSW com modo
  explícito, filtros limitados, plano auditável e profile Docker separado.
- Adiciona testes pytest/property/contract/integration e runner reproduzível
  com métricas por consulta e bootstrap pareado. Resultados sintéticos são
  rotulados como mecanismo/performance local e não como ganho geral.
- Limita consultas a 64 KiB, documentos a 16 MiB, documentos chunked a 512
  chunks e lotes de embedding a 32 itens; enriquecimento e encoding terminam
  antes da transação curta de persistência.
- O benchmark remoto exige autorização dupla, nome de banco com token
  `test`/`bench`, relações vazias e pós-condição de limpeza antes de publicar
  qualquer relatório.
- Corrige o subset BEIR histórico para seleção por hash independente de qrels e
  corrige métricas com IDs duplicados.

Rollout: `RAG3D_RETRIEVAL_PIPELINE=v2`. Rollback composto:
`RAG3D_RETRIEVAL_PIPELINE=legacy` e backend `sqlite` ou `postgres-holo`.

---

## [0.2.0] — Java, Spring Boot e seleção fermiônica

### ☕ Java + Spring Boot (novo)

Terceira linguagem, com o **mesmo índice holográfico**. Não é um cliente HTTP:
é o algoritmo portado, produzindo assinaturas bit a bit idênticas às de
Python/JS.

- `rag3d-java/` — projeto Maven (`io.rag3d:rag3d:0.1.0`), Java 17+
  - `core/Portable.java` — splitmix64 + Box-Muller + CRC-32 (mesmos hiperplanos)
  - `core/TextProc.java` — normalização, sentenças e n-gramas por **code point**
  - `core/Encoders.java` — encoder por hashing (as três formas)
  - `core/Holo.java` — assinatura `BIT(1024)`, facetas, eco int8, constelação
  - `core/Fusion.java` — fusão quântica, RRF e diversidade **greedy-DPP**
  - `core/PgHoloStore.java` — Postgres puro via JDBC (mesmo esquema)
  - `Rag3D.java` — fachada (`connect` · `ingest` · `search`)
- `spring/Rag3DAutoConfiguration.java` — bean `Rag3D` injetável via
  `application.yml`; Spring Boot é **dependência opcional**
- `spring/Rag3DController.java` — REST opcional (`/rag3d/ingest`, `/search`,
  `/stats`), ativado com `rag3d.web.enabled=true`

**Verificado:**

- `ParityCheck` — assinatura/facetas/eco/constelação/sparse idênticos ao Python
  em 4 textos (PT/中文/leis/EN): **Hamming 0/1024**; fusão quântica numericamente
  idêntica.
- `XlangCheck` — Python ingere 6 docs (PT/EN/中文/francês) → **Java acha 6/6** no
  mesmo Postgres, sem reindexar.
- Núcleo compila **sem** Spring no classpath (dependência opcional de verdade).

### ⚛️ Diversidade greedy-DPP

A fusão quântica é **bosônica** — decide *quais* documentos são relevantes.
Faltava decidir *qual conjunto* devolver: o top-k enchia de quase-duplicatas e
os demais fatos ficavam fora do contexto.

Um conjunto de k documentos antissimetrizado (determinante de Slater) tem
amplitude `det(Gram) = Vol²`. Dois documentos idênticos = duas partículas no
mesmo estado → determinante zera → **exclusão de Pauli**. É um *Determinantal
Point Process*; o método greedy com Cholesky incremental aproxima o objetivo e
não é MAP global exato. O custo real também inclui similaridades vetoriais.

| | cobertura@6 | rank-1 útil |
|---|:--:|:--:|
| ranking puro | 46.7% | 100% |
| **fermiônica 0.35 (padrão)** | **100%** | 100% |
| RRF puro → RRF + fermiônica | 48.3% → **100%** | 100% |

**+53 pontos** de fatos distintos no corpus sintético de redundância, sem
perder o topo observado nessa execução. Isso não prova ausência de regressão
em outros corpora. No pipeline legado dessa versão, `diversity = 0.35` era o
padrão e `0` reproduzia o ranking anterior; a V2 usa `none` como rollout seguro.

### 🧪 Avaliação honesta

- `tests/bench_coverage.py` — benchmark de cobertura (redundância)
- `tests/beir_ablation.py` — diagnóstico exploratório em **BEIR com BGE-M3**
  (nDCG@10 e Recall@10; eixos isolados · CombSUM · quântica λ=0/0.5/1 · RRF).
  Como compara variantes no próprio split de teste, não é elegível para tuning,
  seleção de modelo ou claim.
- `BENCHMARKS.md` — registra que, no harness sintético, **a fusão quântica
  empata com o RRF**; não há evidência de superioridade. Documentado de
  propósito.
- **Pesos por coerência** (pureza Tr(ρ²) por eixo): implementado e **desligado
  por padrão** — `H₂ = -log Tr(ρ²)`, portanto pureza e entropia não são a mesma
  quantidade; a literatura de
  QPP mostra ~0.09 de correlação para prever qual canal está certo.

### 🔧 Correções e robustez

- DPP: preenchimento gracioso quando o volume acaba (Cholesky ficaria instável)
- Coerência: piso de 3 candidatos (canal com 1 hit parecia trivialmente decidido)
- DSN dos testes agora lê `RAG3D_PG` (roda em qualquer porta)

---

## [0.1.0] — RAG3D: base tridimensional

### Núcleo

- **Três eixos** por texto: semântico (denso), léxico (esparso invertido) e
  estrutural (multi-vetor, MaxSim). Cada eixo devolve sua própria resposta.
- **Fusão quântica**: amplitudes complexas + regra de Born; `λ=0` colapsa no
  CombSUM clássico; RRF (k=60) embutido como linha de base.
- **Hologramas Textuais**: assinatura LSH `BIT(1024)` (Hamming via `bit_count`
  nativo), facetas `INT[]`+GIN, eco int8 `BYTEA`, constelação de tokens —
  **busca vetorial em Postgres puro, sem `pgvector`**.
- **Paridade Python ↔ JavaScript** bit a bit (splitmix64 + CRC-32 + hiperplanos
  determinísticos): mesmo índice, duas linguagens.
- **Memória de conversa infinita**: resumo rolante + turnos episódicos + score
  `relevância + recência(uso) + importância`; corpus pequeno entra inteiro.
- **Encoders**: BGE-M3 (100+ línguas) ou fallback por hashing (zero dependência).
- **Leitura por qualquer IA**: Anthropic, OpenAI-compatível (DeepSeek, Ollama…)
  ou callable; modo `tri` (3 leitores por eixo + leitora final).

### Defesas para documentos normativos

Validadas num edital real de 15 páginas:

- **Grounding estrito** — só afirma o que está textual; sem base responde "não
  há base suficiente"; nunca infere; distingue princípio de obrigação; respeita
  "e/ou"; concordância entre eixos **não** é evidência.
- **Costura de contiguidade** — reconstrói listas/tabelas partidas pelo chunking
  (ex.: base legal com 6 leis espalhadas).
- **Normalização de números** — conserta "13.243" quebrado pela extração de PDF.
- **Query expansion + reranker** — fecha recall em perguntas parafraseadas.

### App web

- Upload de **PDF/TXT**, chat com **streaming** (SSE), resposta em **Markdown**,
  botão de copiar, as 3 respostas por eixo e as evidências com os eixos que
  acharam. Interface estilo ChatGPT com animações.
- Servidor **exige Postgres** (nada em cache local); botão de limpar índice.

### Performance

Otimizações medidas (todas preservando a paridade bit a bit):

| item | antes | depois |
|---|---|---|
| encode/chunk (JS) | 3.0 ms | 1.4 ms |
| ingest 1500 docs (JS) | 4052 ms | 2140 ms |
| query @1500 docs (JS) | 10.7 ms | 4.0 ms |
| query @1500 docs (Python) | 12.9 ms | 7.5 ms |

Registro histórico de execução única: doc de 200 páginas em **0,7 s** (202
chunks) e consulta em **11 ms**; sem p95/IC, não é claim de performance global.

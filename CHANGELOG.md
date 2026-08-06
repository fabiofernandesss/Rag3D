# Changelog — RAG3D

Histórico das mudanças, da mais recente para a mais antiga. Cada entrada diz o
que mudou, **por quê** e **como foi verificado**.

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
  - `core/Fusion.java` — fusão quântica, RRF e **seleção fermiônica (MAP-DPP)**
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

### ⚛️ Seleção fermiônica (MAP-DPP)

A fusão quântica é **bosônica** — decide *quais* documentos são relevantes.
Faltava decidir *qual conjunto* devolver: o top-k enchia de quase-duplicatas e
os demais fatos ficavam fora do contexto.

Um conjunto de k documentos antissimetrizado (determinante de Slater) tem
amplitude `det(Gram) = Vol²`. Dois documentos idênticos = duas partículas no
mesmo estado → determinante zera → **exclusão de Pauli**. É um *Determinantal
Point Process*; guloso com Cholesky incremental em O(k²N) (Chen et al., 2018).

| | cobertura@6 | rank-1 útil |
|---|:--:|:--:|
| ranking puro | 46.7% | 100% |
| **fermiônica 0.35 (padrão)** | **100%** | 100% |
| RRF puro → RRF + fermiônica | 48.3% → **100%** | 100% |

**+53 pontos** de fatos distintos no contexto, sem perder o topo e **sem
regressão** onde não há redundância. Ligada por padrão (`diversity = 0.35`);
`0` reproduz o comportamento anterior. Paridade Python/JS/Java verificada em 5
níveis de diversidade.

### 🧪 Avaliação honesta

- `tests/bench_coverage.py` — benchmark de cobertura (redundância)
- `tests/beir_ablation.py` — ablação em base pública **BEIR com BGE-M3**
  (nDCG@10 e Recall@10; eixos isolados · CombSUM · quântica λ=0/0.5/1 · RRF)
- `BENCHMARKS.md` — registra que, no harness sintético, **a fusão quântica
  empata com o RRF**; não há evidência de superioridade. Documentado de
  propósito.
- **Pesos por coerência** (pureza Tr(ρ²) por eixo): implementado e **desligado
  por padrão** — Tr(ρ²) diagonal *é* a entropia de Rényi-2, e a literatura de
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

Doc de 200 páginas: armazenamento **0.7 s** (202 chunks), consulta **11 ms**.

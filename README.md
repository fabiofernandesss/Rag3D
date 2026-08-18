<div align="center">

<img src="assets/rag3d-hero.svg" alt="RAG3D: a pergunta é projetada em três eixos — semântico, léxico e estrutural — que se fundem por interferência e devolvem um conjunto sem duplicatas" width="100%">

# 🔺 RAG3D

**Busca vetorial em Postgres comum. Sem `pgvector`, sem banco vetorial.**

Cada texto vira um objeto de **três dimensões**, buscado por três eixos ao mesmo
tempo e combinado por um cálculo de **interferência quântica**.
O mesmo índice funciona em **Python, JavaScript e Java** — assinaturas idênticas
bit a bit.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](#-começar-em-3-minutos)
[![Node](https://img.shields.io/badge/Node-18+-339933?logo=nodedotjs&logoColor=white)](#-começar-em-3-minutos)
[![Java](https://img.shields.io/badge/Java-17+_·_Spring_Boot-ED8B00?logo=openjdk&logoColor=white)](#-começar-em-3-minutos)
[![Postgres](https://img.shields.io/badge/Postgres-sem_pgvector-4169E1?logo=postgresql&logoColor=white)](#-backends)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 🎯 Em 30 segundos

|  | O que o RAG3D faz |
|---|---|
| 🗄️ | **Não precisa de banco vetorial.** Vetores viram bits (`BIT(1024)`), e o Postgres compara com `bit_count` nativo. Qualquer Postgres gerenciado serve. |
| 🔺 | **Três eixos, não um.** Significado, termos exatos e estrutura fina — cada um enxerga o que os outros erram. |
| 🌍 | **Três linguagens, um índice.** Indexe em Python, consulte em Java. Sem ETL, sem reindexar. |
| 📄 | **Feito para documento normativo.** Edital, lei, contrato: se não está no texto, ele responde que não há base. |

```bash
docker compose up -d                       # Postgres comum
cd rag3d-js && npm install && npm run web  # http://localhost:5178
```

Arraste um PDF, pergunte. Pronto.

---

## 🧠 Como funciona

```mermaid
flowchart LR
    Q([pergunta]) --> E[encoder]
    E --> S["🟣 semântico<br/>vetor denso"]
    E --> L["🔵 léxico<br/>esparso + IDF"]
    S --> F{{"⚛️ fusão<br/>interferência"}}
    L --> F
    F --> R["🟢 estrutural<br/>MaxSim rerank"]
    R --> D["conjunto final<br/>sem duplicatas"]
    D --> A([resposta da IA])
```

**Dense e léxico recuperam.** Eles varrem o índice inteiro e erram em lugares
diferentes: o denso entende paráfrase, o léxico acha `Lei 13.243/2016`.

**O estrutural refina.** Ele pontua só os candidatos que os dois trouxeram
(late-interaction MaxSim) — melhora a ordem, mas não resgata o que ficou fora.

**A fusão combina.** Cada eixo entra como uma amplitude complexa; eixos que
concordam se reforçam (interferência construtiva), eixos que discordam se
anulam. Com `λ=0` isso vira soma clássica; RRF (k=60) vem embutido como
alternativa segura.

<details>
<summary><b>🔬 A matemática, para quem quiser</b></summary>

**Fusão (regra de Born):**

```
a_c(d) = √(w_c · s_c(d)) · e^(i·φ_c(d))      ← amplitude e fase por eixo
P(d)   = |Σ_c a_c(d)|²  =  clássico + interferência
```

**Holograma Textual** — o que dispensa o banco vetorial. Cada texto vira:

| Peça | O que é | No Postgres |
|---|---|---|
| Assinatura | 1024 hiperplanos LSH → 1024 bits | `BIT(1024)` — Hamming via `bit_count` |
| Facetas | 16 bandas de 8 bits (pré-filtro) | `INT[]` + GIN |
| Eco denso | vetor int8 (re-pontuação exata) | `BYTEA` |
| Constelação | mini-assinatura por token | `BYTEA` (XOR + popcount) |
| Espectro | termos esparsos | tabela invertida |

**Diversidade greedy-DPP** — a fusão diz *quem* é relevante; falta decidir *qual
conjunto*. Um conjunto antissimetrizado (determinante de Slater) tem amplitude
`det(Gram) = Vol²`: dois textos idênticos zeram o determinante — exclusão de
Pauli. Guloso com Cholesky incremental; aproxima, não é MAP global exato.

</details>

<details>
<summary><b>🌍 Um índice, três linguagens (verificado)</b></summary>

Python, JavaScript e Java compartilham PRNG (splitmix64), hash (CRC-32) e
hiperplanos — então geram **o mesmo holograma**:

```
Python ─┐
JS ─────┼──► mesmo Postgres ──► assinatura BIT(1024): Hamming 0/1024
Java ───┘
```

Verificado: Python ingere 6 documentos (PT/EN/中文/francês) → JS acha **6/6** e
Java acha **6/6**, com assinatura recomputada batendo **0/1024** bits.

Vale para o índice Hash/holográfico. O fingerprint da Retrieval V2 e o
BGE-M3 são Python-only.

</details>

<details>
<summary><b>🛡️ Defesas para documentos normativos</b></summary>

Validadas num edital real de 15 páginas:

- **Grounding estrito** — só afirma o que está no texto. Sem base → *"o documento
  não fornece base suficiente"*. Distingue princípio de obrigação, respeita
  "e/ou", e concordância entre eixos **não** conta como evidência.
- **Costura de contiguidade** — junta chunks vizinhos e reconstrói listas
  partidas (a base legal com 6 leis espalhadas em 3 chunks).
- **Normalização de números** — conserta `13.243` quebrado na extração do PDF.
- **Query expansion + reranker** — fecha recall em perguntas parafraseadas.

</details>

<details>
<summary><b>💬 Memória de conversa infinita</b></summary>

Resumo rolante + turnos episódicos + score
`relevância + recência(uso) + importância`. Corpus pequeno entra inteiro no
prompt; grande vai por retrieval. Em doc-QA, `remember_chat=false` evita que o
histórico compita com os documentos.

</details>

---

## 🚀 Começar em 3 minutos

### 💬 App web (upload de PDF/TXT + chat)

```bash
docker compose up -d                          # 1. Postgres comum (sem pgvector)
cd rag3d-js && npm install && npm run build:web

export RAG3D_PG="postgresql://postgres:rag3d@localhost:5433/rag3d"
export OPENAI_BASE_URL="https://api.deepseek.com"   # ou OpenAI, Ollama…
export OPENAI_API_KEY="sua-chave"

npm run web                                   # http://localhost:5178
```

Resposta em **streaming**, formatada em Markdown, com botão de copiar e as três
respostas por eixo.

<details>
<summary><b>🐍 Python</b></summary>

```bash
pip install -e .            # + '.[postgres]' e '.[bge]' se quiser
```

```python
from rag3d import Rag3D

rag = Rag3D()                                    # SQLite local, encoder auto
rag.ingest("qualquer texto, em qualquer língua")

r = rag.search("pergunta")                       # r.views = 3 eixos · r.fused = fusão
print(rag.ask("qual o prazo do contrato?", mode="tri")["answer"])
```

</details>

<details>
<summary><b>🟨 JavaScript / Node</b></summary>

```bash
cd rag3d-js && npm install
```

```javascript
import { Rag3D, NoLLM } from "./src/index.js";

const rag = await Rag3D.create(
  { pgDsn: process.env.RAG3D_PG },     // vazio = store local
  { llm: new NoLLM() }
);
await rag.ingest("qualquer texto");
const r = await rag.search("pergunta");
console.log(r.fused[0].text, r.fused[0].interference);
await rag.close();
```

</details>

<details>
<summary><b>☕ Java / Spring Boot</b></summary>

```xml
<dependency>
  <groupId>io.rag3d</groupId>
  <artifactId>rag3d</artifactId>
  <version>0.1.0</version>
</dependency>
```

```java
try (Rag3D rag = Rag3D.connect("jdbc:postgresql://localhost:5432/rag3d", "postgres", senha)) {
    rag.ingest("O contrato de aluguel vence em 15 de março de 2027.");
    rag.search("quando vence?", 8).fused()
       .forEach(h -> System.out.printf("%.3f  %s%n", h.score(), h.text()));
}
```

**Spring Boot** — auto-configuração inclusa, só declarar no `application.yml`:

```yaml
rag3d:
  jdbc-url: jdbc:postgresql://localhost:5432/rag3d
  username: postgres
  password: ${DB_PASSWORD}
  diversity: 0.35
  web:
    enabled: true          # expõe /rag3d/search e /rag3d/ingest
```

```java
@Service
class BuscaService {
    private final Rag3D rag;                    // injetado
    BuscaService(Rag3D rag) { this.rag = rag; }
}
```

Spring é **opcional**: sem ele, o núcleo roda igual via `Rag3D.connect(...)`.

</details>

<details>
<summary><b>⌨️ CLI</b></summary>

```bash
python3 -m rag3d ingest docs/       # ou: node rag3d-js/src/cli.js ingest docs/
python3 -m rag3d search "prazo"     # mostra os 3 eixos + a fusão
python3 -m rag3d ask "..." --tri    # 3 leitores + leitora final
python3 -m rag3d chat               # memória infinita
```

</details>

---

## ⚙️ Configuração

Aceita `RAG3D_*` (fallback para `TRIRAG_*`). Detalhes em [`.env.example`](.env.example).

| Variável | Efeito |
|---|---|
| `RAG3D_PG` | DSN Postgres. Vazio = SQLite (Python) / JSON (JS) |
| `RAG3D_ENCODER` | `auto` (padrão), `bge-m3`, `fallback` |
| `RAG3D_LLM_MODEL` | modelo da IA leitora |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` | qualquer endpoint OpenAI-compatível |
| `RAG3D_RETRIEVAL_PIPELINE` | `legacy` (padrão) ou `v2` (opt-in) |

---

## 🔬 Testes

```bash
docker compose up -d

# Python
PYTHONPATH=$PWD python3 tests/test_smoke.py                  # tri-eixo, SQLite, zero-dep
RAG3D_TEST_PG_DSN=postgresql://postgres:rag3d@localhost:5433/rag3d_test \
RAG3D_TEST_PG_ALLOW_DESTRUCTIVE=1 python3 tests/test_pg.py   # backend holográfico

# JavaScript
cd rag3d-js && node --test 'test/*.test.mjs'

# Java (sem Maven)
cd rag3d-java && javac -d target/classes $(find src -name '*.java' -not -path '*/spring/*')
java -cp target/classes ParityCheck                          # holograma idêntico

# Paridade cross-language (prova Hamming 0/1024)
python3 tests/xlang_ingest.py && node rag3d-js/test/xlang_parity.mjs
```

> O teste de Postgres é destrutivo (`TRUNCATE`) e por isso exige um banco com
> `test` no nome e a flag explícita.

---

## 📊 Números honestos

**A fusão quântica não vence o RRF.** No harness sintético, todos os métodos de
fusão empatam. Está documentado de propósito — veja [BENCHMARKS.md](BENCHMARKS.md).

| estratégia | Recall@5 | MRR |
|---|:--:|:--:|
| eixo semântico / léxico / estrutural | 83.3% | 0.833 / 0.750 / 0.833 |
| CombSUM (λ=0) · **quântica (λ=1)** · RRF | 83.3% | 0.833 |

**Diversidade greedy-DPP** — no corpus sintético de redundância, cobertura de
fatos distintos vai de **46.7% → 100%** sem perder o rank-1. Não prova ausência
de regressão em outros corpora.

**Retrieval V2 (opt-in)** — o protocolo validation → lock → test em 1.000 chunks
**não atingiu a meta de 20%**: p95 caiu 11,05%, mas nDCG@10 caiu 1,79% e MRR@20
caiu 15,93%. Nenhum ponto da grade HNSW (`ef_search=100..1000`) passou
Recall ANN@20 ≥ 0,98. Por isso o padrão continua `legacy` e pgvector *exact* é o
modo seguro. [Relatório completo](docs/benchmarks/retrieval-v2-results.md).

---

## 🆚 Backends

|  | RAG3D | pgvector | Milvus / Qdrant / Pinecone |
|---|---|---|---|
| Extensão / serviço extra | **nenhum** | extensão | infra dedicada |
| Busca vetorial | `bit_count` + `INT[]`/`BYTEA` nativos | operador `<->` | índice ANN próprio |
| Mesmo índice em 3 linguagens | ✅ (Hash/holográfico) | — | — |
| Fricção operacional | mínima | média | alta |
| ANN em milhões de vetores | ⚠️ ainda não | ✅ HNSW | ✅ |

**Posicionamento honesto:** o forte é **eliminar o banco vetorial** e a
portabilidade entre linguagens — não vencer um HNSW em recall a milhões de
vetores. Para milhares a dezenas de milhares de chunks (a maioria dos casos de
doc-QA), o scan holográfico dá conta.

---

## 🧭 Limitações e roadmap

**Não provado ainda:** fusão quântica > RRF em base pública · curvas
recall × latência contra pgvector/FAISS · avaliação estatística repetida.

**Roadmap:** índice ANN sobre as assinaturas (multi-probe LSH / HNSW) ·
BEIR + MIRACL com BGE-M3 publicados (bons ou ruins) · reranker cross-encoder
nativo · extração de fatos com ADD/UPDATE/DELETE na memória.

---

## 📚 Mais

[CHANGELOG](CHANGELOG.md) ·
[Benchmarks](BENCHMARKS.md) ·
[Arquitetura V2](docs/architecture/retrieval-engine-v2.md) ·
[Guia de rollout](docs/guides/retrieval-v2-rollout.md)

**Fontes:** BGE-M3 (arXiv 2402.03216) · Contextual Retrieval (Anthropic) ·
Fusão híbrida (Bruch et al., TOIS 2023) · RRF (Cormack 2009) · IR quântico
(van Rijsbergen 2004; Sordoni SIGIR 2013) · DPP (Chen et al., NeurIPS 2018) ·
MemGPT · RAPTOR · LSH (Charikar STOC 2002)

---

<div align="center">

**RAG3D** — feito por **Fabio Fernandes Jesus**

📧 fabiofernandesjesus@hotmail.com · 📱 (81) 98888-6020 · 📸 [@fabinhodejesus](https://instagram.com/fabinhodejesus)

Licença [MIT](LICENSE) · contribuições e estrelas ⭐ são bem-vindas

</div>

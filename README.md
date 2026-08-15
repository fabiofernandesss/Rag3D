```
       ██████╗   █████╗   ██████╗  ██████╗ ██████╗
       ██╔══██╗ ██╔══██╗ ██╔════╝  ╚════██╗██╔══██╗
       ██████╔╝ ███████║ ██║  ███╗  █████╔╝██║  ██║
       ██╔══██╗ ██╔══██║ ██║   ██║  ╚═══██╗██║  ██║
       ██║  ██║ ██║  ██║ ╚██████╔╝ ██████╔╝██████╔╝
       ╚═╝  ╚═╝ ╚═╝  ╚═╝  ╚═════╝  ╚═════╝ ╚═════╝
            r a g   t r i d i m e n s i o n a l  ·  △
```

<div align="center">

# 🔺 RAG3D

### RAG tridimensional com fusão quântica, Hologramas Textuais e memória infinita

**Um método novo de RAG.** Cada texto vira um objeto de **três dimensões**, buscado por três eixos ao mesmo tempo, combinado por um cálculo de **interferência quântica**, lido por **qualquer IA** — e rodando até num **Postgres comum, sem `pgvector`**.

O índice holográfico **Hash** serve Python, JavaScript e Java com assinaturas
idênticas bit a bit. BGE-M3 e o backend pgvector da Retrieval V2 são Python-only.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](#)
[![Node](https://img.shields.io/badge/Node-18+-339933?logo=nodedotjs&logoColor=white)](#)
[![Java](https://img.shields.io/badge/Java-17+_·_Spring_Boot-ED8B00?logo=openjdk&logoColor=white)](#)
[![Postgres](https://img.shields.io/badge/Postgres-sem_pgvector-4169E1?logo=postgresql&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Autor: Fabio Fernandes Jesus**
📧 fabiofernandesjesus@hotmail.com · 📱 WhatsApp (81) 98888-6020 · 📸 [@fabinhodejesus](https://instagram.com/fabinhodejesus)

</div>

---

## 📑 Índice

[O método](#-o-método-sem-enrolação) ·
[Arquitetura](#️-arquitetura) ·
[Começando](#-começando-3-minutos) ·
[App web](#opção-a--app-web-upload-de-pdftxt--chat-) ·
[Java / Spring Boot](#opção-d--java--spring-boot-) ·
[Configuração](#️-configuração-variáveis-de-ambiente) ·
[Testes](#-rodando-os-testes) ·
[Avaliação e ablação](#-avaliação-e-ablação-honesto) ·
[Backends de armazenamento](#backends-de-armazenamento) ·
[Limitações e roadmap](#-limitações-e-roadmap) ·
[Honestidade](#-honestidade-científica)

---

## ✨ Em uma frase

> RAG3D representa cada texto em **3 eixos** (semântico + léxico + estrutural), funde os rankings com **interferência quântica**, guarda tudo como **Hologramas Textuais** que cabem num banco de dados comum, e responde com defesas de fidelidade pensadas para **documentos normativos** (editais, leis, contratos).

---

## 🧠 O método, sem enrolação

### 1. As três dimensões

Todo texto — documento **ou** turno de conversa — nasce em três formas ao ser salvo:

| Eixo | O que captura | Forma | Busca |
|---|---|---|---|
| 🟣 **Semântico** | significado, paráfrase | vetor denso 1024-d | cosseno |
| 🔵 **Léxico** | termos exatos, nomes, números, leis | esparso invertido | IDF |
| 🟢 **Estrutural** | ordem, frases, correspondência fina | multi-vetor (token a token) | MaxSim sobre candidatos dense+lexical |

Dense e lexical são os recuperadores globais. O estrutural pontua somente a
união limitada de candidatos produzida pelos dois e devolve uma terceira visão
como **late-interaction reranker**; não recupera um item ausente dessa união.

> Por que três sinais? Léxico e semântico erram em lugares diferentes; MaxSim
> estrutural refina a ordem dentro do pool que eles geraram.

### 2. A fusão quântica (o novo cálculo)

Em vez de somar pontuações, cada eixo entra como uma **amplitude complexa** (regra de Born):

```
a_c(d) = √(w_c · s_c(d)) · e^(i·φ_c(d))          ← amplitude e fase por eixo
P(d)   = |Σ_c a_c(d)|²  =  Σ clássico  +  Σ interferência
```

- Eixos que **concordam** → interferência **construtiva** → o documento sobe.
- Eixos que **discordam** → interferência **destrutiva** → o documento desce.
- Botão `λ` (interference_strength): `λ=0` colapsa no fusão clássica (CombSUM) — é a trava de segurança e a ablação. RRF (k=60) vem embutido como linha de base.

### 2b. Seleção fermiônica — a outra metade da física

A fusão acima é **bosônica**: amplitudes somam, concordância amplifica. Ela
decide *quais* documentos são relevantes — mas não *qual conjunto* devolver.
Sem isso, o top-k enche de quase-duplicatas e os outros fatos ficam de fora.

Entra o princípio oposto. Um conjunto de k documentos é um estado de k
partículas; se ele for **antissimétrico** (determinante de Slater):

```
|ψ_S|² = det(Gram(v_S)) = Vol²(v_1 … v_k)
```

Dois documentos idênticos são duas partículas no mesmo estado → o determinante
zera → **exclusão de Pauli**. Redundância é proibida por construção, não por
heurística. Formalmente é um *Determinantal Point Process*, resolvido pelo
guloso com Cholesky incremental (Chen et al., NeurIPS 2018). Ele aproxima o
objetivo log-determinante; não é MAP global exato, e o custo também inclui as
similaridades vetoriais.

**No harness sintético de redundância:** cobertura de fatos distintos no top-k
vai de **46.7% → 100%**, sem perder o rank-1 nesse corpus. Isso não é evidência
de ganho geral de retrieval nem substitui avaliação em dataset público.
No pipeline legado, era ligado por padrão (`diversity = 0.35`) e `0` reproduz
o ranking puro. A Retrieval V2 começa com `diversity=none`.
Detalhes em [BENCHMARKS.md](BENCHMARKS.md).

> Resumo da física: **bósons decidem a relevância, férmions decidem o conjunto.**

### 3. Hologramas Textuais (vetores em banco comum)

O truque que dispensa banco vetorial. Cada texto vira um **holograma** feito só de tipos que qualquer banco já sabe indexar:

| Peça | O que é | Tipo no Postgres |
|---|---|---|
| **Assinatura** | 1024 hiperplanos LSH → 1024 bits | `BIT(1024)` — distância = `bit_count(a # b)` nativo |
| **Facetas** | 16 bandas de 8 bits (pré-filtro) | `INT[]` + índice GIN |
| **Eco denso** | vetor quantizado int8 (re-pontuação) | `BYTEA` |
| **Constelação** | mini-assinatura por token | `BYTEA` (XOR + popcount) |
| **Espectro** | termos esparsos | tabela invertida |

Resultado: **busca vetorial em Postgres puro**, sem `pgvector`, sem extensão nenhuma.

### 4. Um índice Hash, TRÊS linguagens

Python (`rag3d/`), JavaScript (`rag3d-js/`) e **Java** (`rag3d-java/`) produzem
**hologramas idênticos bit a bit** — mesmo PRNG (splitmix64), mesmo hash
(CRC-32), mesmos hiperplanos (`portable.py` / `portable.js` / `Portable.java`).

Essa paridade vale para o encoder Hash/holográfico. JavaScript e Java não
geram embeddings BGE-M3 nem consultam as tabelas `rag3d_v2_*` de pgvector.

No modo holográfico legado, um documento Hash ingerido em Python é achado por
uma consulta em JS **ou em Java**, no mesmo banco, sem reindexar nada:

```
Python ─┐
JS ─────┼──► mesmo Postgres ──► assinatura BIT(1024) idêntica: Hamming 0/1024
Java ───┘
```

O contrato portátil Hash é verificado por `ParityCheck` e pelos testes Node:
assinatura, bandas, eco, sparse e constantes coincidem entre Python, JS e Java
nos vetores de teste. Os scripts `tests/xlang_ingest.py`, `XlangCheck` e
`xlang_parity.mjs` exercitam o índice PostgreSQL compartilhado quando o driver
JDBC e um banco de teste dedicado estão disponíveis; BGE-M3/pgvector não têm
essa paridade cross-language.

Um índice `postgres-holo` certificado pela Retrieval V2 inclui parâmetros de
pipeline/chunking que os ports Node e Java ainda não representam. Por isso eles
falham fechado ao encontrar esse fingerprint. Para tráfego trilíngue, mantenha
um índice Hash legado separado; para voltar de um índice V2, exporte/reingira os
documentos nesse índice em vez de editar metadados de fingerprint.

### 5. Defesas para documentos normativos

Validado num edital real de 15 páginas. Quatro defesas (Python **e** JS):

- **Grounding estrito** — só afirma o que está textual; sem base → *"não há base suficiente"*; nunca infere; distingue **princípio de obrigação** e respeita **"e/ou"**.
- **Costura de contiguidade** — junta chunks vizinhos, reconstruindo listas/tabelas partidas (ex.: a base legal com 6 leis espalhadas por 3 chunks).
- **Normalização de números** — conserta "13.243" quebrado pela extração de PDF.
- **Query expansion + reranker** — o LLM gera termos alternativos e reordena o pool (fecha recall em perguntas parafraseadas).

### 6. Memória de conversa infinita

Camadas (MemGPT + agentes de Stanford): resumo rolante + turnos episódicos + score `relevância + recência(uso) + importância`. Corpus pequeno entra inteiro no prompt; grande vai por retrieval.

---

## 🗺️ Arquitetura

```
INGESTÃO
texto ─► normalize/chunk ─► encoder ─► SQLite | postgres-holo | pgvector

RETRIEVAL V2
pergunta ─► normalize/expand/encode ─┬─► dense retriever ─┐
                                     └─► sparse retriever ─┴─► união limitada
                                                                  │
                                                                  ▼
                                                           RRF (padrão)
                                                                  │
                                                                  ▼
                                                   MaxSim estrutural (rerank)
                                                                  │
                                                                  ▼
                                         reranker opcional ─► none/MMR/DPP
                                                                  │
                                                                  ▼
                                              hydrate/small-to-big/stitch ─► leitor
```

O pipeline `legacy` continua disponível e preserva a fusão quântica e as três
visões históricas. Na V2, o estrutural aparece somente depois da fusão porque
não é um recuperador global.

---

## 🚀 Começando (3 minutos)

### Opção A — App web (upload de PDF/TXT + chat) 💬

```bash
# 1. Postgres comum (sem pgvector), via Docker
docker compose up -d

# 2. dependências e build do frontend
cd rag3d-js && npm install && npm run build:web

# 3. configure a IA (DeepSeek, OpenAI, Ollama... qualquer OpenAI-compatível)
export RAG3D_PG="postgresql://postgres:rag3d@localhost:5433/rag3d"
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_API_KEY="sua-chave"
export RAG3D_LLM_MODEL="deepseek-chat"

# 4. suba
npm run web      # abre em http://localhost:5178
```

Arraste um PDF/TXT, pergunte — a resposta vem **em streaming**, formatada em markdown, com botão de copiar e as **3 respostas por eixo**.

### Opção B — Biblioteca Python 🐍

```bash
pip install -e .            # extras: .[postgres], .[pgvector], .[bge], .[reranker]
```

```python
from rag3d import Rag3D, Rag3DConfig

rag = Rag3D()                                   # SQLite local, encoder auto, LLM auto
rag.ingest("qualquer texto, em qualquer língua")
r = rag.search("pergunta")                       # r.views = 3 respostas · r.fused = fusão
print(rag.ask("qual o prazo do contrato?", mode="tri")["answer"])
```

### Opção C — Biblioteca JavaScript / Node 🟨

```bash
cd rag3d-js && npm install
```

```javascript
import { Rag3D, NoLLM } from "./src/index.js";

const rag = await Rag3D.create(
  { pgDsn: process.env.RAG3D_PG },               // vazio = store em memória/JSON
  { llm: new NoLLM() }                           // ou deixe em branco p/ auto
);
await rag.ingest("qualquer texto");
const r = await rag.search("pergunta");
console.log(r.fused[0].text, r.fused[0].interference);
await rag.close();
```

### Opção D — Java / Spring Boot ☕

Para aplicação corporativa. O núcleo Java (`rag3d-java/`) é o mesmo algoritmo —
sem dependência de Python ou Node, só o driver JDBC.

```xml
<!-- pom.xml -->
<dependency>
  <groupId>io.rag3d</groupId>
  <artifactId>rag3d</artifactId>
  <version>0.1.0</version>
</dependency>
```

**Uso direto (sem framework):**

```java
import io.rag3d.Rag3D;

try (Rag3D rag = Rag3D.connect("jdbc:postgresql://localhost:5432/rag3d", "postgres", senha)) {
    rag.ingest("O contrato de aluguel vence em 15 de março de 2027.");

    Rag3D.Result r = rag.search("quando vence o contrato?", 8);
    r.fused().forEach(h ->
        System.out.printf("%.3f  %s  %s%n", h.score(), h.channels(), h.text()));

    r.views().forEach((eixo, hits) ->      // as 3 respostas do tridimensional
        System.out.println(eixo + " -> " + hits.size() + " hits"));
}
```

**Com Spring Boot** — auto-configuração inclusa, é só declarar no `application.yml`:

```yaml
rag3d:
  jdbc-url: jdbc:postgresql://localhost:5432/rag3d   # Postgres COMUM, sem pgvector
  username: postgres
  password: ${DB_PASSWORD}        # nunca a senha em texto no yml
  top-k: 10
  diversity: 0.35                 # seleção fermiônica (0 = ranking puro)
  fusion: quantum                 # quantum | rrf
  web:
    enabled: true                 # opcional: expõe /rag3d/search e /rag3d/ingest
```

```java
@Service
public class BuscaService {
    private final Rag3D rag;                       // injetado pela auto-configuração
    public BuscaService(Rag3D rag) { this.rag = rag; }

    public List<Rag3D.Hit> perguntar(String q) throws SQLException {
        return rag.search(q, 8).fused();
    }
}
```

Com `rag3d.web.enabled=true`, sobem os endpoints prontos:

```bash
curl -X POST localhost:8080/rag3d/ingest -H 'Content-Type: application/json' \
     -d '{"text":"...","title":"contrato.pdf"}'
curl 'localhost:8080/rag3d/search?q=prazo&k=8'
```

> O Spring Boot é **opcional** (`<optional>true</optional>` no pom): sem ele, o
> núcleo funciona igual via `Rag3D.connect(...)`.

**Compilar e testar o núcleo sem Maven** (Java 17+):

```bash
cd rag3d-java
javac -d target/classes \
  $(find src/main/java -name '*.java' -not -path '*/spring/*') \
  src/test/java/ParityCheck.java
java -cp target/classes ParityCheck
```

`XlangCheck` exige um PostgreSQL de teste e um driver JDBC fornecido pelo
operador; o JAR não é versionado neste repositório.

### CLI (Python e JS)

```bash
python3 -m rag3d ingest docs/            # ou: node rag3d-js/src/cli.js ingest docs/
python3 -m rag3d search "prazo"          # mostra os 3 eixos + a fusão
python3 -m rag3d ask "..." --tri         # 3 leitores + leitora final
python3 -m rag3d chat                     # memória infinita
```

---

## ⚙️ Configuração (variáveis de ambiente)

Aceita `RAG3D_*` (preferido) com fallback para `TRIRAG_*`. Ver [`.env.example`](.env.example).

| Variável | Efeito |
|---|---|
| `RAG3D_PG` | DSN PostgreSQL; sem backend explícito mantém o holográfico legado |
| `RAG3D_BACKEND` | `sqlite`, `postgres-holo` ou `pgvector` (Python) |
| `RAG3D_RETRIEVAL_PIPELINE` | `legacy` (padrão de rollout) ou `v2` |
| `RAG3D_FUSION` | V2 usa `rrf` por padrão; `quantum` permanece experimental/legado |
| `RAG3D_STRUCTURAL_RERANK` | habilita/desabilita MaxSim tardio na V2 |
| `RAG3D_RERANKER` | `none`, `llm` ou `cross-encoder` (`pip install 'rag3d[reranker]'`) |
| `RAG3D_DIVERSITY_METHOD` | `none`, `mmr` ou `dpp` |
| `RAG3D_ENCODER` | `auto`, `bge-m3`, `fallback`/`hash` |
| `RAG3D_ALLOW_ENCODER_FALLBACK` | V2 falha fechado salvo autorização explícita |
| `RAG3D_PGVECTOR_SEARCH_MODE` | `exact` (seguro), `ann` (fail-closed) ou `auto` |
| `RAG3D_LLM` / `RAG3D_LLM_MODEL` | provedor e modelo |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` | endpoint OpenAI-compatível (DeepSeek, Ollama, vLLM…) |

---

## 📁 Estrutura

```
rag3d/            núcleo Python  (portable, holo, fusion, encoders, store, retrieve, memory, reader, engine, cli)
rag3d-js/
  src/            port JavaScript (mesmos módulos, hologramas idênticos)
  web/            app web: server.js (Express) + client React (upload + chat streaming)
  test/           suíte JS + testes cross-language
rag3d-java/       port Java (Maven) — núcleo + auto-configuração Spring Boot
  src/main/java/io/rag3d/core/     Portable · TextProc · Encoders · Holo · Fusion · PgHoloStore
  src/main/java/io/rag3d/spring/   auto-config + controller REST opcional
  src/test/java/                   ParityCheck (bit a bit) · XlangCheck (tri-linguagem)
tests/            suíte Python + bench_fusion.py + beir_ablation.py + scripts cross-language
BENCHMARKS.md     ablação honesta + roteiro de avaliação BEIR
docker-compose.yml   Postgres comum (sem pgvector)
```

---

## 🔬 Rodando os testes

```bash
docker compose up -d
# Python
PYTHONPATH=$PWD python3 tests/test_smoke.py     # mecânica tri-eixo (SQLite, zero-dep)
PYTHONPATH=$PWD python3 tests/test_pg.py         # backend holográfico (Postgres)
python3 tests/bench_fusion.py                    # quântica vs RRF nos seus dados
# JavaScript
cd rag3d-js && node --test 'test/*.test.mjs'
# Java core (sem Maven e sem driver externo)
cd rag3d-java && javac -d target/classes $(find src/main/java -name '*.java' -not -path '*/spring/*') src/test/java/ParityCheck.java
java -cp target/classes ParityCheck
# Cross-language Python → Node
python3 tests/xlang_ingest.py && node rag3d-js/test/xlang_parity.mjs
```

O teste PostgreSQL com Java (`XlangCheck`) requer um JAR JDBC externo no
classpath e as mesmas proteções de banco de teste usadas pelos scripts Python.

---

## 📊 Avaliação e ablação (honesto)

**Resumo:** no harness sintético embutido, **a fusão quântica empata com o RRF**
— não há, hoje, evidência de que a supere. Detalhes completos em
[**BENCHMARKS.md**](BENCHMARKS.md). Isto é registrado de propósito, não escondido.

### Ablação de fusão (reproduzível)

```bash
python3 tests/bench_fusion.py 1800
```

| estratégia | Recall@5 | MRR |
|---|:--:|:--:|
| eixo semântico | 83.3% | 0.833 |
| eixo léxico | 83.3% | 0.750 |
| eixo estrutural | 83.3% | 0.833 |
| CombSUM (λ=0) | 83.3% | 0.833 |
| **quântica (λ=1)** | 83.3% | 0.833 |
| RRF (k=60) | 83.3% | 0.833 |

### Greedy DPP — cobertura no harness sintético

```bash
python3 tests/bench_coverage.py 20 8
```

| configuração | cobertura@6 | rank-1 útil |
|---|:--:|:--:|
| ranking puro | 46.7% | 100% |
| **fermiônica 0.35 (padrão)** | **100%** | 100% |
| RRF puro | 48.3% | 100% |
| RRF + fermiônica | 100% | 100% |

**+53 pontos** de fatos distintos no corpus sintético desenhado para
redundância, sem perder o topo observado. Esse resultado não demonstra ganho
geral de retrieval e não atende sozinho à meta de 20% da Retrieval V2.

Quântica = RRF = CombSUM. `λ=0` (sem interferência) dá o mesmo que `λ=1` — coerente
com a garantia de que a fusão colapsa no clássico. **Use RRF como padrão de
produção**; a fusão quântica é uma opção falsificável, não uma alegação de vitória.

> O corpus sintético é fácil (agulhas lexicalmente únicas), então todo método
> acha o alvo e nada os separa. Diferenciar de verdade exige encoder semântico
> real + base pública.

### Resultado fechado da Retrieval V2

O protocolo validation -> lock -> test em 1.000 chunks sintéticos não atingiu
a meta de 20%. Contra legacy + SQLite, a V2 reduziu p95 em 11,05%, enquanto
nDCG@10 caiu 1,79% e MRR@20 caiu 15,93%. A grade HNSW de `ef_search=100..1000`
usou o índice naturalmente, mas nenhum ponto passou Recall ANN@20 >= 0,98 em
todas as consultas. Por isso o rollout continua opt-in e pgvector exact é o
modo seguro. Metodologia, IC95, hardware, regressões e JSONs estão no
[relatório reproduzível](docs/benchmarks/retrieval-v2-results.md).

### Diagnóstico exploratório BEIR (BGE-M3, nDCG@10)

O script histórico abaixo compara estratégias e valores de λ diretamente no
split `test`. Ele é útil para inspeção, mas **não é elegível para tuning, claim
de qualidade ou meta de 20%**. O runner V2 incluído também usa somente o corpus
sintético congelado e sempre marca claims como `not_evaluated`; esta entrega
não inclui um adapter de dataset externo elegível a evidência publicável.

```bash
# 1. dependências pesadas (BGE-M3 = ~2.3GB; precisa de ~3GB de RAM livre)
python3 -m venv .venv && .venv/bin/pip install '.[bge]'

# 2. dataset BEIR (ex.: nfcorpus — 3.6k docs, 323 queries)
mkdir -p bench_data && cd bench_data
curl -sLO https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip && unzip -q nfcorpus.zip && cd ..

# 3. ablação completa (eixos isolados · CombSUM · quântica λ=0/0.5/1 · RRF), nDCG@10 + Recall@10
.venv/bin/python tests/beir_ablation.py nfcorpus
```

O diagnóstico ([`tests/beir_ablation.py`](tests/beir_ablation.py)) ingere o
corpus com **BGE-M3** e reporta nDCG@10/Recall@10 por estratégia na mesma
recuperação. Como todas as alternativas são observadas no `test`, seus números
não podem ser usados para escolher a alternativa vencedora.

---

<a id="backends-de-armazenamento"></a>

## 🆚 Backends de armazenamento

O backend holográfico sem extensão continua disponível. Retrieval Engine V2
também oferece pgvector exact/HNSW como extra opcional em tabelas separadas.
Os dados/colunas `holo_*` permanecem compatíveis, mas os três ports agora
adicionam e validam FKs nomeadas para impedir documentos/postings órfãos; faça
o [preflight de rollout](docs/guides/retrieval-v2-rollout.md) antes de atualizar
um índice legado.
Veja a [arquitetura V2](docs/architecture/retrieval-engine-v2.md) e o
[guia de rollout](docs/guides/retrieval-v2-rollout.md).

|  | RAG3D | pgvector | Milvus / Qdrant / Pinecone |
|---|---|---|---|
| Extensão / serviço extra | **nenhum** (Postgres puro) | extensão pgvector | serviço/infra dedicada |
| Busca vetorial | `bit_count` + `INT[]`/`BYTEA` nativos | `<->` operador da extensão | índice ANN próprio |
| Multilíngue (100+) | ✅ BGE-M3 | depende do encoder | depende do encoder |
| Mesmo índice em 3 linguagens | ✅ Hash/holográfico legado; fingerprint V2 é Python-only | — | — |
| Fricção operacional | mínima (qualquer Postgres gerenciado) | média | alta |
| ANN em escala de milhões | ⚠️ ainda não (scan + facetas) | ✅ HNSW | ✅ |

**Posicionamento honesto:** o forte do RAG3D é **eliminar o banco vetorial** e a
**portabilidade Python/JS**, não vencer um HNSW em recall a milhões de vetores.
O registro histórico observou ~11 ms/consulta em uma execução local, sem p95,
IC ou controle de carga. Trate-o como diagnóstico antigo, não como sizing de
produção; use o runner V2 no corpus e hardware de destino.

---

## 🧭 Limitações e roadmap

**O que ainda não está provado** (e não escondemos):

- Fusão quântica **> RRF** em benchmark público — ainda não demonstrado; o
  `beir_ablation.py` é apenas diagnóstico exploratório, não prova comparativa.
- Curvas recall × latência contra pgvector/FAISS/Qdrant.
- Avaliação estatística repetida (vários datasets, intervalos de confiança).

**Roadmap:**

- [x] Backend pgvector HNSW opcional na Retrieval V2 (ainda exige tuning por corpus).
- [ ] Rodar BEIR + MIRACL com BGE-M3 e publicar os números (bons ou ruins).
- [x] Interface cross-encoder opcional e fail-closed (`rag3d[reranker]`); ganho
  de qualidade ainda não avaliado.
- [ ] Extração de fatos ADD/UPDATE/DELETE na memória (padrão Mem0).

---

## 🎯 Honestidade científica

- A **fusão quântica** tem precedente real (van Rijsbergen 2004; Sordoni SIGIR'13; Gkoumas & Song) mas **não há prova publicada de que vença o RRF em benchmark de texto**. Por isso ela é falsificável (`λ=0` = clássico) e comparável (`--rrf`). **Meça nos seus dados.**
- Os **Hologramas** são LSH (aproximação). O eco int8 pode alterar o top-k;
  portanto não existe garantia universal de erro ou perda de recall abaixo de
  1% sem medição no corpus de destino.
- O **encoder embutido** (hash de n-gramas) é léxico-superficial: ótimo para dev/testes e zero-dep. Para busca semântica de verdade, `pip install 'rag3d[bge]'` liga o **BGE-M3** (100+ línguas, MIT).
- "Todas as línguas" = as ~100 do XLM-R (BGE-M3); fora delas, o fallback cobre qualquer escrita Unicode com qualidade léxica.

---

## 📚 Fontes principais

BGE-M3 (arXiv 2402.03216) · Contextual Retrieval (Anthropic) · Fusão híbrida (Bruch et al., TOIS 2023) · RRF (Cormack 2009) · IR quântico (van Rijsbergen 2004; Sordoni SIGIR 2013) · MemGPT (2310.08560) · RAPTOR (2401.18059) · LSH (Charikar STOC 2002)

---

<div align="center">

**RAG3D** — feito por **Fabio Fernandes Jesus**

📧 fabiofernandesjesus@hotmail.com · 📱 (81) 98888-6020 · 📸 [@fabinhodejesus](https://instagram.com/fabinhodejesus)

Licença [MIT](LICENSE) · contribuições e estrelas ⭐ são bem-vindas

</div>

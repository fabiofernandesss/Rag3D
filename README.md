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

Funciona em **qualquer língua** e o mesmo índice serve **Python e JavaScript** (assinaturas idênticas bit a bit).

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](#)
[![Node](https://img.shields.io/badge/Node-18+-339933?logo=nodedotjs&logoColor=white)](#)
[![Postgres](https://img.shields.io/badge/Postgres-sem_pgvector-4169E1?logo=postgresql&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Autor: Fabio Fernandes Jesus**
📧 fabiofernandesjesus@hotmail.com · 📱 WhatsApp (81) 98888-6020 · 📸 [@fabinhodejesus](https://instagram.com/fabinhodejesus)

</div>

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
| 🟢 **Estrutural** | ordem, frases, correspondência fina | multi-vetor (token a token) | MaxSim |

Na busca, a pergunta é projetada nos **mesmos três eixos** e cada um devolve **a sua própria resposta**. São as *"três respostas do tridimensional"*.

> Por que três? Léxico e semântico erram em lugares diferentes (MIRACL: BM25 39.3, denso 41.5, **híbrido 57.8** nDCG@10). O estrutural pega o que os dois deixam passar.

### 2. A fusão quântica (o novo cálculo)

Em vez de somar pontuações, cada eixo entra como uma **amplitude complexa** (regra de Born):

```
a_c(d) = √(w_c · s_c(d)) · e^(i·φ_c(d))          ← amplitude e fase por eixo
P(d)   = |Σ_c a_c(d)|²  =  Σ clássico  +  Σ interferência
```

- Eixos que **concordam** → interferência **construtiva** → o documento sobe.
- Eixos que **discordam** → interferência **destrutiva** → o documento desce.
- Botão `λ` (interference_strength): `λ=0` colapsa no fusão clássica (CombSUM) — é a trava de segurança e a ablação. RRF (k=60) vem embutido como linha de base.

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

### 4. Um índice, duas linguagens

Python (`rag3d/`) e JavaScript (`rag3d-js/`) produzem **hologramas idênticos bit a bit** (mesmo PRNG, mesmo hash, mesmos hiperplanos — ver `portable.py`/`portable.js`). Um documento ingerido em Python é achado por uma consulta em JS **no mesmo banco** — assinatura `BIT(1024)` com **Hamming 0/1024** entre as linguagens.

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
                          ┌──────────── INGESTÃO ────────────┐
   texto  ──normalize──►  chunk adaptativo  ──►  ENCODER  ──►  3 formas
                                                (BGE-M3 / hash)     │
                                                                    ▼
             ┌───────────────── HOLOGRAMA TEXTUAL ─────────────────┐
             │  BIT(1024) · facetas INT[] · eco BYTEA · constelação │  ──► Postgres
             └──────────────────────────────────────────────────────┘      (sem pgvector)
                                                                    ▲
                          ┌──────────── CONSULTA ────────────┐      │
  pergunta ─(expansão)─►  ENCODER  ─►  ┌── eixo semântico ──┐        │
                                       ├── eixo léxico  ─────┤─► FUSÃO QUÂNTICA ─► costura
                                       └── eixo estrutural ──┘         │              │
                                                                       ▼              ▼
                                       3 leitores (1 por eixo)  ──►  LEITORA FINAL (qualquer IA)
                                                                       │
                                                                       ▼
                                                                   resposta + fontes
```

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
pip install -e .            # + "pip install .[postgres]" e ".[bge]" se quiser
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
| `RAG3D_PG` | DSN Postgres → backend holográfico. Vazio = SQLite (Python) / JSON (JS) |
| `RAG3D_ENCODER` | `auto` (padrão), `bge-m3`, `fallback` |
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
tests/            suíte Python + benchmark de fusão + scripts cross-language
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
# Cross-language (prova Hamming 0/1024)
python3 tests/xlang_ingest.py && node rag3d-js/test/xlang_parity.mjs
```

---

## 🎯 Honestidade científica

- A **fusão quântica** tem precedente real (van Rijsbergen 2004; Sordoni SIGIR'13; Gkoumas & Song) mas **não há prova publicada de que vença o RRF em benchmark de texto**. Por isso ela é falsificável (`λ=0` = clássico) e comparável (`--rrf`). **Meça nos seus dados.**
- Os **Hologramas** são LSH (aproximação); o eco int8 re-pontua os candidatos com <1% de erro; o pré-filtro de facetas só liga acima de 20k hologramas.
- O **encoder embutido** (hash de n-gramas) é léxico-superficial: ótimo para dev/testes e zero-dep. Para busca semântica de verdade, `pip install FlagEmbedding` liga o **BGE-M3** (100+ línguas, MIT).
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

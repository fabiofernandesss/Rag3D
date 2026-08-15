# RAG3D (JavaScript)

Port em Node.js do [RAG3D](../README.md) — RAG tridimensional com fusão por
interferência quântica, Hologramas Textuais e memória infinita.

O diferencial: **hologramas compatíveis bit a bit com a versão Python**. Um
documento ingerido em Python fica gravado no mesmo Postgres com uma assinatura
que o JS recomputa idêntica (Hamming 0/1024) — então os dois consultam e
alimentam o **mesmo índice Hash legado**. Índices certificados pela Retrieval V2
são recusados pelo adapter JavaScript até existir fingerprint V2 portátil. RAG
que funciona entre linguagens de programação no contrato legado,
além de entre línguas humanas.

Zero dependências obrigatórias (usa `fetch`, `crypto` e o PRNG portável do
Node 18+). `pg` só é preciso para o backend Postgres.

## Instalação

```bash
cd rag3d-js
npm install            # instala pg (opcional; só p/ Postgres)
```

## Uso

```bash
# store em memória com persistência JSON (zero-dep)
node src/cli.js ingest --text "O contrato vence em março de 2027."
node src/cli.js search "prazo do contrato"

# Postgres puro (sem pgvector) — mesmo banco da versão Python
export TRIRAG_PG=postgresql://postgres:rag3d@localhost:5433/rag3d
node src/cli.js ingest docs/
node src/cli.js ask "qual o prazo?" --tri
node src/cli.js chat
```

```javascript
import { TriRag, NoLLM } from "./src/engine.js";

const rag = await TriRag.create(
  { pgDsn: process.env.TRIRAG_PG, contextualEnrich: false },
  { llm: new NoLLM() }            // ou deixe em branco p/ auto (Anthropic/OpenAI/Ollama)
);
await rag.ingest("qualquer texto, em qualquer língua");
const r = await rag.search("pergunta");   // r.views = 3 respostas; r.fused = fusão
console.log(r.fused[0].text, r.fused[0].interference);
await rag.close();
```

## O que é idêntico ao Python (`src/portable.js`)

- **PRNG** splitmix64 counter-based → os mesmos hiperplanos LSH
- **crc32** IEEE → os mesmos hashes do encoder fallback
- **tokenização** (normalize/word/char n-gramas) agnóstica de língua
- **Holograma**: assinatura `BIT(1024)`, facetas, eco int8, constelação

Consequência: `signDense(mesmo_texto)` em JS == `sign_dense(mesmo_texto)` em
Python, byte a byte. O esquema Postgres (`holo_grams`, `holo_spectrum`) é o
mesmo — as duas linguagens leem e escrevem as mesmas tabelas.

## Testes

```bash
node --test 'test/*.test.mjs'          # suíte em memória

# cross-language (precisa do Postgres de teste de pé):
python3 ../tests/xlang_ingest.py       # Python ingere
node test/xlang.mjs assert             # JS acha os docs do Python
node test/xlang_parity.mjs             # prova Hamming 0/1024 da assinatura
```

## Diferença para a versão Python

- JS traz só o encoder **fallback** portável (hash de n-gramas). O BGE-M3 (denso
  de verdade, 100+ línguas) vive no lado Python. Para busca semântica cruzada de
  alta qualidade, ingira/consulte pelo Python com BGE-M3; os hiperplanos são os
  mesmos, então dá para misturar desde que o modelo denso seja o mesmo.
- Backend em memória (JSON) no lugar do SQLite. Postgres é idêntico.

Detalhes completos da arquitetura e da matemática no [README raiz](../README.md).

// Configuração do RAG3D (JS) — espelha rag3d/config.py (camelCase).
import os from "node:os";
import path from "node:path";

// lê RAG3D_<name>, com fallback para o antigo TRIRAG_<name>
const E = (name) => process.env[`RAG3D_${name}`] || process.env[`TRIRAG_${name}`] || "";

export function defaultConfig(over = {}) {
  const cfg = {
    // armazenamento
    dataDir: E("DATA") ? expand(E("DATA")) : path.join(os.homedir(), ".rag3d"),
    pgDsn: E("PG"), // Postgres SEM pgvector; vazio = store em memória/JSON

    // encoder (JS traz o fallback portável; BGE-M3 é do lado Python)
    encoder: "hash",
    denseDim: 1024,
    colbertDim: 128,
    maxColbertTokens: 256,

    // chunking adaptativo
    chunkTokens: 400,
    chunkOverlap: 60,
    parentTokens: 1600,
    tinyDocTokens: 500,
    hugeDocTokens: 12000,
    contextualEnrich: true,

    // busca
    topK: 10,
    channelK: 100,
    fusion: "quantum", // quantum | rrf
    rrfK: 60,
    channelWeights: [1.0, 0.8, 0.9],
    interferenceStrength: 1.0,
    rerank: false,
    rerankPool: 30,
    // query expansion via LLM: gera termos alternativos (sinônimos, números
    // de seção, palavras que provavelmente aparecem no doc) para fechar recall
    // em consultas cujas palavras não batem com as do doc.
    expandQuery: false,
    expandQueryMax: 3,
    // costura de vizinhos (Rag3D contiguity): ao recuperar um chunk, junta os
    // vizinhos contíguos (pos±raio) do mesmo doc — reconstrói listas/tabelas/
    // seções que o chunking partiu (ex.: lista de leis, critérios). 0 = off.
    stitchRadius: 0,

    // memória infinita
    memoryBudgetTokens: 6000,
    recencyHalfLifeTurns: 40.0,
    wRelevance: 1.0,
    wRecency: 0.35,
    wImportance: 0.25,
    summaryEveryTurns: 12,

    // leitura
    llmProvider: E("LLM") || "auto",
    llmModel: E("LLM_MODEL"),
    readMode: "fast", // fast | tri
    maxAnswerTokens: 1500,
    smallCorpusTokens: 8000,
    // true: cada turno de chat vira memória pesquisável (chat com memória
    // infinita). false: perguntas-sobre-documento sem misturar histórico —
    // evita a leitora repetir respostas passadas (uso doc-QA, ex.: web).
    rememberChat: true,
  };
  return { ...cfg, ...over };
}

function expand(p) {
  return p.startsWith("~") ? path.join(os.homedir(), p.slice(1)) : p;
}

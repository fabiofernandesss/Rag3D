// Servidor web do TriRAG — upload de PDF/TXT + chat.
//
// Ingere os arquivos no índice holográfico (Postgres, sem pgvector) e responde
// no chat com o LLM configurado (DeepSeek via OpenAI-compat, por exemplo).
//
// Variáveis de ambiente:
//   RAG3D_PG         DSN Postgres (senão usa store em memória/JSON)
//   OPENAI_BASE_URL  ex: https://api.deepseek.com
//   OPENAI_API_KEY   chave do provedor
//   RAG3D_LLM_MODEL  ex: deepseek-chat
//   PORT             porta do servidor (padrão 5173)
import express from "express";
import multer from "multer";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { PDFParse } from "pdf-parse";
import { TriRag } from "../src/engine.js";
import { OpenAICompatLLM, NoLLM } from "../src/llm.js";

async function extractPdf(buffer) {
  const parser = new PDFParse({ data: buffer });
  try {
    return (await parser.getText()).text;
  } finally {
    await parser.destroy();
  }
}

// exige Postgres: tudo persiste no banco, nada em cache/JSON local
const PG_DSN = process.env.RAG3D_PG || process.env.TRIRAG_PG;
if (!PG_DSN) {
  console.error("RAG3D_PG não definido. Este servidor exige Postgres (nada é salvo em cache local).");
  console.error("Ex: RAG3D_PG=postgresql://postgres:senha@localhost:5432/db node web/server.js");
  process.exit(1);
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 5173;
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 25 * 1024 * 1024 } });

function buildLLM() {
  if (process.env.OPENAI_API_KEY || process.env.OPENAI_BASE_URL) {
    return new OpenAICompatLLM(
      process.env.RAG3D_LLM_MODEL || process.env.TRIRAG_LLM_MODEL || "deepseek-chat",
      process.env.OPENAI_BASE_URL || "https://api.deepseek.com",
      process.env.OPENAI_API_KEY || ""
    );
  }
  return new NoLLM();
}

const rag = await TriRag.create(
  {
    pgDsn: PG_DSN, encoder: "hash", contextualEnrich: false,
    readMode: "tri",
    // corpus pequeno (um doc ou poucos) entra INTEIRO no prompt — resumo e
    // perguntas abrangentes ficam corretos. Acima disso, retrieval tri-eixo.
    smallCorpusTokens: Number(process.env.RAG3D_SMALL_CORPUS || process.env.TRIRAG_SMALL_CORPUS || 8000),
    // doc-QA: não indexar turnos de chat (senão o histórico compete com os
    // documentos e a leitora pode repetir respostas passadas)
    rememberChat: false,
    // CHUNK FINO: docs normativos (editais) têm seções curtas e específicas
    // (2.6 eixos, 1.5 exclusões, 3.3 base legal). Chunks de ~200 tokens em vez
    // dos 400 default fazem o retrieval encostar exatamente na seção certa em
    // vez de trazer sempre o mesmo "chunk quente" grande.
    chunkTokens: 200, chunkOverlap: 80, parentTokens: 900,
    // COBERTURA: recupera largo e reranqueia (LLM) para o trecho certo subir.
    channelK: 200, rerankPool: 60, topK: 18, memoryBudgetTokens: 12000,
    rerank: (process.env.RAG3D_RERANK ?? process.env.TRIRAG_RERANK) !== "0",
    // resposta longa o bastante para várias perguntas de uma vez (o usuário
    // costuma colar a bateria inteira num só envio)
    maxAnswerTokens: Number(process.env.RAG3D_MAX_ANSWER || process.env.TRIRAG_MAX_ANSWER || 3000),
    // Rag3D: costura de contiguidade — reconstrói listas/tabelas partidas pelo
    // chunking (ex.: as 6 leis da base legal, critérios, eixos).
    stitchRadius: Number(process.env.RAG3D_STITCH || process.env.TRIRAG_STITCH || 3),
    // QUERY EXPANSION: DeepSeek gera 2-3 termos alternativos antes de buscar
    // (fecha recall em consultas que usam palavras diferentes das do doc).
    expandQuery: (process.env.RAG3D_EXPAND ?? process.env.TRIRAG_EXPAND) !== "0",
  },
  { llm: buildLLM() }
);

const app = express();
app.use(express.json());
// favicon inline (evita o 404 que o browser gera pedindo /favicon.ico)
app.get("/favicon.ico", (_req, res) =>
  res.type("image/svg+xml").send(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><text y="26" font-size="26">🔺</text></svg>'
  )
);
app.use(express.static(path.join(__dirname, "public")));

// --- upload: PDF ou TXT -> ingestão tridimensional -----------------------
app.post("/api/upload", upload.array("files"), async (req, res) => {
  const results = [];
  for (const f of req.files || []) {
    try {
      let text;
      const ext = path.extname(f.originalname).toLowerCase();
      if (ext === ".pdf") text = await extractPdf(f.buffer);
      else text = f.buffer.toString("utf8"); // .txt, .md, etc.
      if (!text || !text.trim()) throw new Error("sem texto extraível");
      const r = await rag.ingest(text, f.originalname, f.originalname);
      results.push({ file: f.originalname, ok: true, chunks: r.chunks, tokens: r.tokens });
    } catch (e) {
      results.push({ file: f.originalname, ok: false, error: e.message });
    }
  }
  res.json({ results, stats: await rag.stats() });
});

// --- chat: consulta tridimensional + leitura -----------------------------
app.post("/api/chat", async (req, res) => {
  const message = (req.body?.message || "").trim();
  if (!message) return res.status(400).json({ error: "mensagem vazia" });
  try {
    const out = await rag.chat(message);
    // no modo corpus_integral os blocks são strings; no retrieval são objetos
    const sources = (out.context?.blocks || []).slice(0, 5).map((b) =>
      typeof b === "string"
        ? { text: b.slice(0, 240), channels: [] }
        : { id: b.id, text: (b.chosen ?? b.text ?? "").slice(0, 240), channels: b.channels || [], interference: b.interference }
    );
    res.json({
      answer: out.answer,
      note: out.note || null,
      readMode: out.readMode,
      subAnswers: out.subAnswers || null,
      mode: out.context?.mode,
      sources,
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// chat com streaming (SSE): tokens chegam ao vivo, fontes no fim
app.post("/api/chat/stream", async (req, res) => {
  const message = (req.body?.message || "").trim();
  if (!message) return res.status(400).json({ error: "mensagem vazia" });
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();
  const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);
  try {
    const out = await rag.chatStream(message, (tok) => send({ token: tok }));
    const ctx = out.context || {};
    const sources = (ctx.blocks || []).slice(0, 5).map((b) =>
      typeof b === "string"
        ? { text: b.slice(0, 240), channels: [] }
        : { id: b.id, text: (b.chosen ?? b.text ?? "").slice(0, 240), channels: b.channels || [], interference: b.interference }
    );
    send({ done: true, sources, subAnswers: out.subAnswers || null, readMode: out.readMode, mode: ctx.mode, note: out.note || null });
  } catch (e) {
    send({ error: e.message });
  }
  res.end();
});

app.get("/api/stats", async (_req, res) => res.json(await rag.stats()));

// limpa o índice (remove todos os documentos)
app.post("/api/reset", async (_req, res) => {
  await rag.reset();
  res.json({ ok: true, stats: await rag.stats() });
});

app.listen(PORT, () => {
  console.log(`RAG3D web em http://localhost:${PORT}`);
  rag.stats().then((s) => console.log("backend:", s.backend, "| llm:", s.llm, "| chunks:", s.chunks));
});

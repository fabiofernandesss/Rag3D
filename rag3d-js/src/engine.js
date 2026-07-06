// Fachada TriRAG (JS) — espelha trirag/engine.py. Junta as peças.
import path from "node:path";
import { defaultConfig } from "./config.js";
import { makeEncoder } from "./encoders.js";
import { Ingestor } from "./ingest.js";
import { CallableLLM, makeLLM, NoLLM } from "./llm.js";
import { ChatMemory } from "./memory.js";
import { MemStore } from "./memstore.js";
import { Reader } from "./reader.js";
import { Reranker } from "./rerank.js";
import { TriRetriever } from "./retrieve.js";

export class TriRag {
  constructor(cfg, store, llm, axisLlms = {}) {
    this.cfg = cfg; this.store = store;
    this.encoder = makeEncoder(cfg);
    this.llm = llm;
    const reranker = cfg.rerank ? new Reranker(llm) : null;
    this.ingestor = new Ingestor(store, this.encoder, cfg, llm);
    this.retriever = new TriRetriever(store, this.encoder, cfg, reranker, llm);
    this.memory = new ChatMemory(store, this.encoder, cfg, llm);
    this.reader = new Reader(cfg, llm, axisLlms);
  }

  // fábrica assíncrona (store/llm podem exigir await)
  static async create(over = {}, { llm = null, axisLlms = {} } = {}) {
    const cfg = defaultConfig(over);
    let store;
    if (cfg.pgDsn) {
      const { PgHoloStore } = await import("./pgstore.js");
      store = await PgHoloStore.connect(cfg.pgDsn, cfg.denseDim, cfg.colbertDim);
    } else {
      const { mkdirSync } = await import("node:fs");
      mkdirSync(cfg.dataDir, { recursive: true });
      store = new MemStore(path.join(cfg.dataDir, "trirag.json"));
    }
    // guarda de encoder (nome + dims)
    const fingerprint = `hash:${cfg.denseDim}:${cfg.colbertDim}`;
    const prev = await store.getMeta("encoder");
    if (prev && prev !== fingerprint)
      throw new Error(`Índice criado com encoder '${prev}', agora '${fingerprint}'. Use a mesma config ou apague o índice.`);
    await store.setMeta("encoder", fingerprint);

    let realLlm;
    if (typeof llm === "function") realLlm = new CallableLLM(llm);
    else if (llm) realLlm = llm;
    else realLlm = await makeLLM(cfg.llmProvider, cfg.llmModel);

    return new TriRag(cfg, store, realLlm, axisLlms);
  }

  ingest(text, source = "inline", title = "") { return this.ingestor.ingestText(text, source, title); }
  ingestFile(fp) { return this.ingestor.ingestFile(fp); }
  search(query, topK = null) { return this.retriever.search(query, topK); }

  async ask(query, mode = null) {
    const ctx = await this.memory.buildContext(query, this.retriever);
    const out = await this.reader.read(query, ctx, mode);
    out.context = ctx;
    return out;
  }

  async chat(userMsg, mode = null) {
    const ctx = await this.memory.buildContext(userMsg, this.retriever);
    const out = await this.reader.read(userMsg, ctx, mode);
    out.context = ctx;
    // só grava memória de conversa se habilitado (doc-QA não polui a busca)
    if (out.answer && this.cfg.rememberChat !== false)
      out.turnNo = await this.memory.recordTurn(userMsg, out.answer);
    return out;
  }

  // versão streaming: onToken(delta) é chamado enquanto a resposta final sai
  async chatStream(userMsg, onToken, mode = null) {
    const ctx = await this.memory.buildContext(userMsg, this.retriever);
    const out = await this.reader.readStream(userMsg, ctx, onToken, mode);
    out.context = ctx;
    if (out.answer && this.cfg.rememberChat !== false)
      out.turnNo = await this.memory.recordTurn(userMsg, out.answer);
    return out;
  }

  async stats() {
    return {
      encoder: this.encoder.name,
      llm: this.llm.available() ? `${this.llm.provider}:${this.llm.model}` : "nenhum",
      chunks: await this.store.nChunks(),
      tokens_no_corpus: await this.store.corpusTokens(),
      turnos: await this.store.lastTurnNo(),
      fusao: this.cfg.fusion,
      backend: this.cfg.pgDsn ? "postgres-holo (sem pgvector)" : "memoria/json",
      dados: this.cfg.pgDsn || this.cfg.dataDir,
    };
  }

  async reset() { if (this.store.reset) await this.store.reset(); }

  async close() { if (this.store.close) await this.store.close(); }
}

export { NoLLM, CallableLLM };

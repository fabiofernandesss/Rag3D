// Suíte JS (store em memória) — espelha tests/test_smoke.py.
import { test } from "node:test";
import assert from "node:assert";
import { rmSync } from "node:fs";
import { TriRag, NoLLM } from "../src/engine.js";
import { Ingestor } from "../src/ingest.js";
import { HashEncoder } from "../src/encoders.js";
import { MemStore } from "../src/memstore.js";
import { defaultConfig } from "../src/config.js";
import { fuse, rrfFuse, fermionicSelect, coherence } from "../src/fusion.js";
import { normalize, wordTokens, splitSentences, estimateTokens } from "../src/textproc.js";

function tmp() { return "/tmp/trirag_js_" + Math.random().toString(36).slice(2); }
async function mk(over = {}) {
  const dir = tmp();
  rmSync(dir, { recursive: true, force: true });
  return TriRag.create({ dataDir: dir, contextualEnrich: false, smallCorpusTokens: 0, ...over }, { llm: new NoLLM() });
}

test("textproc agnóstico de língua", () => {
  const s = splitSentences("Olá mundo. Tudo bem? 你好。这是测试！");
  assert.ok(s.length >= 3, JSON.stringify(s));
  const toks = wordTokens("O contrato 移动电话 vence");
  assert.ok(toks.includes("移动") && toks.includes("contrato"), JSON.stringify(toks));
  assert.ok(estimateTokens("teste de tokens aqui") > 0);
  assert.equal(normalize("  a   b  "), "a b");
});

test("consulta pública respeita limite UTF-8", async () => {
  const rag = await mk();
  try {
    await assert.rejects(
      rag.search("😀".repeat(16385)),
      /query.*65536.*UTF-8 bytes/i,
    );
  } finally {
    await rag.close();
  }
});

test("ask, chat e chatStream validam a consulta antes de memória ou LLM", async () => {
  const rag = await mk({ smallCorpusTokens: 1_000_000 });
  let contextCalls = 0;
  let readerCalls = 0;
  rag.memory.buildContext = async () => {
    contextCalls += 1;
    return { mode: "corpus_integral", blocks: [], tokens: 0 };
  };
  rag.reader.read = async () => {
    readerCalls += 1;
    return { answer: "unexpected" };
  };
  rag.reader.readStream = async () => {
    readerCalls += 1;
    return { answer: "unexpected" };
  };
  const oversized = "x".repeat(65_537);
  try {
    await assert.rejects(rag.ask(oversized), /query.*65536.*UTF-8 bytes/i);
    await assert.rejects(rag.chat(oversized), /query.*65536.*UTF-8 bytes/i);
    await assert.rejects(rag.chatStream(oversized, () => {}), /query.*65536.*UTF-8 bytes/i);
    assert.equal(contextCalls, 0, "invalid input must not reach memory");
    assert.equal(readerCalls, 0, "invalid input must not reach the reader/LLM");
  } finally {
    await rag.close();
  }
});

test("stats mascara DSN PostgreSQL", async () => {
  const rag = await mk();
  const secret = "postgresql://admin:super-secret@db.internal/production";
  rag.cfg.pgDsn = secret;
  try {
    const stats = await rag.stats();
    assert.equal(stats.dados, "postgresql (redacted)");
    assert.ok(!JSON.stringify(stats).includes(secret));
    assert.ok(!JSON.stringify(stats).includes("super-secret"));
  } finally {
    await rag.close();
  }
});

test("expansão LLM excessiva cai para a consulta original antes de split", async () => {
  const rag = await mk({ expandQuery: true, expandQueryMax: 3 });
  const batches = [];
  const originalEncode = rag.encoder.encode.bind(rag.encoder);
  rag.encoder.encode = (texts, isQuery = false) => {
    batches.push([...texts]);
    return originalEncode(texts, isQuery);
  };
  rag.retriever.llm = {
    available: () => true,
    complete: async () => "x".repeat(65_537),
  };
  try {
    await rag.search("consulta");
    assert.deepEqual(batches, [["consulta"]]);
  } finally {
    await rag.close();
  }
});

test("ingestão tri-eixo + busca multilíngue", async () => {
  const rag = await mk();
  await rag.ingest("O contrato de aluguel vence em 15 de março de 2027.", "x", "contrato");
  await rag.ingest("The rocket launch is scheduled for July 2026.", "x", "rocket");
  await rag.ingest("会议将于星期五上午十点在北京举行。", "x", "meeting");
  const r = await rag.search("quando vence o contrato?");
  assert.deepEqual(Object.keys(r.views).sort(), ["estrutural", "lexico", "semantico"]);
  assert.ok(r.fused[0].text.includes("contrato"), r.fused[0].text);
  assert.ok("interference" in r.fused[0]);
  const r2 = await rag.search("rocket launch date");
  assert.ok(r2.fused[0].text.toLowerCase().includes("rocket"));
  const r3 = await rag.search("会议在哪里举行");
  assert.ok(r3.fused[0].text.includes("北京"));
  await rag.close();
});

test("rollback de ingestão MemStore remove documento e chunks parciais", async () => {
  class FailingMemStore extends MemStore {
    constructor() { super(null); this.chunkWrites = 0; }
    async addChunk(...args) {
      this.chunkWrites += 1;
      if (this.chunkWrites === 2) throw new Error("injected chunk failure");
      return super.addChunk(...args);
    }
  }

  const store = new FailingMemStore();
  const cfg = defaultConfig({
    denseDim: 8,
    colbertDim: 4,
    maxColbertTokens: 8,
    tinyDocTokens: 1,
    chunkTokens: 4,
    chunkOverlap: 0,
    parentTokens: 8,
    contextualEnrich: false,
  });
  const ingestor = new Ingestor(store, new HashEncoder(cfg), cfg, new NoLLM());

  await assert.rejects(
    ingestor.ingestText(
      "Primeira sentença curta. Segunda sentença curta. Terceira sentença curta.",
    ),
    /injected chunk failure/,
  );

  assert.equal(store.docs.length, 0);
  assert.equal(store.grams.length, 0);
  assert.equal(store.seq, 0);
});

test("chunking adaptativo pequeno vs grande", async () => {
  const rag = await mk();
  const tiny = await rag.ingest("Nota curta: senha do wifi é abc123.");
  assert.equal(tiny.chunks, 1);
  const big = Array.from({ length: 12 }, (_, i) =>
    `Capítulo ${i}. ` + Array.from({ length: 40 }, (_, j) => `Frase ${j} do capítulo ${i} sobre o assunto.`).join(" ")
  ).join("\n\n");
  const r = await rag.ingest(big, "x", "livro");
  assert.ok(r.chunks > 3, String(r.chunks));
  const res = await rag.search("Frase 5 do capítulo 7");
  assert.ok(res.fused[0].wide.length >= res.fused[0].text.length);
  await rag.close();
});

test("matemática da fusão quântica", () => {
  const ch = { a: [[1, 0.9], [2, 0.85], [3, 0.4]], b: [[1, 0.88], [3, 0.42]], c: [[1, 0.8], [2, 0.05]] };
  const hits = fuse(ch, [1, 1, 1], 5, "quantum", 1.0);
  assert.equal(hits[0].chunkId, 1, "concordância -> topo");
  assert.ok(hits.find((h) => h.chunkId === 1).interference > 0, "interferência construtiva");
  const h0 = fuse(ch, [1, 1, 1], 5, "quantum", 0.0);
  for (const h of h0) assert.ok(Math.abs(h.score - h.classical) < 1e-9, "λ=0 = clássico");
  const solo = hits.filter((h) => h.channels.length === 1);
  for (const h of solo) assert.ok(Math.abs(h.interference) < 1e-9, "solo sem interferência");

  const nearTie = fuse({ a: [[2, 1.0], [1, 1.0 - 5e-13]] }, [1], 2, "quantum", 1.0);
  assert.deepEqual(nearTie.map((h) => h.chunkId), [1, 2], "near-flat usa desempate histórico por id");

  const extreme = fuse({ a: [[1, 1e308], [2, -1e308]] }, [1], 2, "quantum", 1.0);
  assert.ok(extreme.every((h) => Number.isFinite(h.score)), "extremos finitos permanecem finitos");
});

test("RRF conta ID duplicado apenas no primeiro rank do canal", () => {
  const hits = rrfFuse(
    { a: [[1, 0.9], [1, 0.8], [2, 0.7]], b: [[2, 0.9]] },
    { a: 1, b: 1 },
    2,
    60,
  );

  assert.deepEqual(hits.map((hit) => hit.chunkId), [2, 1]);
  assert.equal(hits.find((hit) => hit.chunkId === 1).channels.length, 1);
});

test("memória episódica + orçamento", async () => {
  const rag = await mk({ memoryBudgetTokens: 300 });
  for (let i = 0; i < 30; i++) await rag.memory.recordTurn(`pergunta ${i} tópico ${i % 5}`, `resposta ${i} tópico ${i % 5}`);
  assert.equal(await rag.store.lastTurnNo(), 30);
  const ctx = await rag.memory.buildContext("me lembra do tópico 3?", rag.retriever);
  assert.equal(ctx.mode, "retrieval");
  assert.ok(ctx.tokens <= 300, String(ctx.tokens));
  assert.ok(ctx.blocks.length > 0);
  assert.ok(ctx.blocks.every((b) => "memoryScore" in b));
  await rag.close();
});

test("persistência JSON entre instâncias", async () => {
  const dir = tmp();
  rmSync(dir, { recursive: true, force: true });
  let rag = await TriRag.create({ dataDir: dir, contextualEnrich: false, smallCorpusTokens: 0 }, { llm: new NoLLM() });
  await rag.ingest("O código do cofre é 7742.");
  await rag.close();
  rag = await TriRag.create({ dataDir: dir, contextualEnrich: false, smallCorpusTokens: 0 }, { llm: new NoLLM() });
  const r = await rag.search("código do cofre");
  assert.ok(r.fused[0].text.includes("7742"), r.fused[0].text);
  await rag.close();
});

test("greedy DPP evita redundância e mantém o topo", () => {
  // 4 clusters de 4 vetores quase idênticos
  const base = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const vecs = new Map(); const items = [];
  for (let g = 0; g < 4; g++) for (let c = 0; c < 4; c++) {
    const id = g * 10 + c;
    const v = Float64Array.from(base[g].map((x, d) => x + (d === (g + 1) % 4 ? 0.02 * c : 0)));
    const n = Math.hypot(...v); for (let d = 0; d < v.length; d++) v[d] /= n;
    vecs.set(id, v); items.push([id, 1.0 - 0.01 * (g * 4 + c)]);
  }
  items.sort((a, b) => b[1] - a[1]);

  const puro = fermionicSelect(items, vecs, 4, 0.0);
  assert.deepEqual(puro, items.slice(0, 4).map(([i]) => i), "diversidade=0 = ranking puro");

  const div = fermionicSelect(items, vecs, 4, 0.5);
  assert.equal(new Set(div.map((i) => Math.floor(i / 10))).size, 4, "cobre os 4 clusters");
  assert.equal(div[0], items[0][0], "mantém o top-1 relevante");
  assert.ok(new Set(puro.map((i) => Math.floor(i / 10))).size < 4, "ranking puro concentra");
});

test("coerência: pico ~1, chapado 0", () => {
  assert.ok(coherence(new Map([[1, 1.0], [2, 0.01], [3, 0.01]])) > 0.9);
  assert.ok(coherence(new Map([[1, 1.0], [2, 1.0], [3, 1.0]])) < 1e-9);
});

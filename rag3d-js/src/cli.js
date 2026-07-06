#!/usr/bin/env node
// CLI do RAG3D (JS) — espelha trirag/cli.py.
//   node src/cli.js ingest <arquivos...> | --text "..."
//   node src/cli.js search "consulta"    [--rrf] [--rerank] [--k N]
//   node src/cli.js ask "pergunta"       [--tri] [--rerank] [--rrf]
//   node src/cli.js chat
//   node src/cli.js stats
// Flags globais: --pg <dsn>  --data <dir>
import { createInterface } from "node:readline";
import { statSync, readdirSync } from "node:fs";
import path from "node:path";
import { TriRag } from "./engine.js";

function parseArgs(argv) {
  const flags = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--tri" || a === "--rrf" || a === "--rerank") flags[a.slice(2)] = true;
    else if (a === "--k") flags.k = parseInt(argv[++i], 10);
    else if (a === "--pg") flags.pg = argv[++i];
    else if (a === "--data") flags.data = argv[++i];
    else if (a === "--text") flags.text = argv[++i];
    else flags._.push(a);
  }
  return flags;
}

async function mk(flags) {
  const over = {};
  if (flags.pg || process.env.TRIRAG_PG) over.pgDsn = flags.pg || process.env.TRIRAG_PG;
  if (flags.data) over.dataDir = flags.data;
  if (flags.rrf) over.fusion = "rrf";
  if (flags.k) over.topK = flags.k;
  if (flags.tri) over.readMode = "tri";
  if (flags.rerank) over.rerank = true;
  return TriRag.create(over);
}

function walk(p) {
  const st = statSync(p);
  if (!st.isDirectory()) return [p];
  return readdirSync(p).flatMap((f) => walk(path.join(p, f)));
}

const SKIP = new Set([".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".db", ".json", ".pyc"]);

async function main() {
  const [cmd, ...rest] = process.argv.slice(2);
  const flags = parseArgs(rest);
  if (!cmd) { console.error("uso: ingest|search|ask|chat|stats"); process.exit(1); }
  const rag = await mk(flags);

  // chat mantém a conexão viva pelo loop e fecha sozinho; os demais fecham no
  // finally para não vazar o cliente Postgres em caminhos de erro.
  if (cmd === "chat") { await runChat(rag); return; }
  try {
    await dispatch(cmd, flags, rag);
  } finally {
    await rag.close();
  }
}

async function dispatch(cmd, flags, rag) {
  if (cmd === "ingest") {
    if (flags.text) {
      const r = await rag.ingest(flags.text);
      console.log(`ok: doc ${r.docId}, ${r.chunks} chunks, ~${r.tokens || 0} tokens`);
    } else {
      for (const p of flags._) for (const f of walk(p)) {
        if (SKIP.has(path.extname(f).toLowerCase())) continue;
        try { const r = await rag.ingestFile(f); console.log(`ok: ${path.basename(f)} -> ${r.chunks} chunks`); }
        catch (e) { console.error(`erro: ${f}: ${e.message}`); }
      }
    }
  } else if (cmd === "search") {
    const r = await rag.search(flags._[0]);
    console.log(`# consulta: ${r.query}\n# ${JSON.stringify(r.stats)}\n`);
    for (const [axis, rows] of Object.entries(r.views)) {
      console.log(`--- eixo ${axis} ---`);
      for (const row of rows.slice(0, 5)) console.log(`  [${row.id}] ${(row.score ?? 0).toFixed(4)}  ${JSON.stringify(row.text.slice(0, 90))}`);
    }
    console.log(`\n=== fusão (${rag.cfg.fusion}) ===`);
    for (const row of r.fused) {
      const ex = row.interference !== undefined ? ` interf=${row.interference >= 0 ? "+" : ""}${row.interference.toFixed(3)}` : "";
      console.log(`  [${row.id}] ${row.score.toFixed(4)}${ex} eixos=${(row.channels || []).join(",")}`);
      console.log(`      ${JSON.stringify(row.text.slice(0, 120))}`);
    }
  } else if (cmd === "ask") {
    printAnswer(await rag.ask(flags._[0]));
  } else if (cmd === "stats") {
    console.log(JSON.stringify(await rag.stats(), null, 2));
  } else { console.error("comando desconhecido:", cmd); process.exit(1); }
}

async function runChat(rag) {
  const s = await rag.stats();
  console.log(`RAG3D chat — ${s.llm} | encoder ${s.encoder} | fusão ${rag.cfg.fusion}\n(vazio para sair)\n`);
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  const ask = () => rl.question("você> ", async (msg) => {
    if (!msg.trim()) { rl.close(); await rag.close(); return; }
    // erro num turno não derruba a sessão nem vaza a conexão pg
    try { printAnswer(await rag.chat(msg), "yah> "); }
    catch (e) { console.error("erro no turno:", e.message); }
    finally { ask(); }
  });
  ask();
}

function printAnswer(out, prefix = "") {
  const ctx = out.context || {};
  if (out.subAnswers) { for (const [ax, ans] of Object.entries(out.subAnswers)) console.log(`  (${ax}) ${ans.trim().slice(0, 200)}`); console.log(); }
  if (out.answer) console.log(`${prefix}${out.answer.trim()}\n`);
  else {
    console.log(`[${out.note || "sem resposta"}] — evidências (${ctx.mode}):`);
    for (const b of (ctx.blocks || []).slice(0, 8)) {
      const t = typeof b === "string" ? b : b.chosen ?? b.text ?? "";
      console.log(`  - ${JSON.stringify(t.slice(0, 150))}`);
    }
  }
}

main().catch((e) => { console.error(e); process.exit(1); });

// Cross-language: lado JS conecta no MESMO Postgres do Python.
//
//   node test/xlang.mjs query "<consulta>"      -> busca (prova PY->JS)
//   node test/xlang.mjs ingest                   -> ingere 1 doc (prova JS->PY)
//   node test/xlang.mjs assert                   -> bateria de asserts PY->JS

import { TriRag, NoLLM } from "../src/engine.js";

const DSN = "postgresql://postgres:rag3d@localhost:5433/rag3d";
const OVER = { pgDsn: DSN, encoder: "hash", contextualEnrich: false, smallCorpusTokens: 0 };

async function mk() { return TriRag.create(OVER, { llm: new NoLLM() }); }

async function query(q) {
  const rag = await mk();
  const r = await rag.search(q, 3);
  const top = r.fused.length ? r.fused[0].text : "(vazio)";
  console.log(`JS consulta ${JSON.stringify(q)} -> ${JSON.stringify(top.slice(0, 70))}`);
  await rag.close();
}

async function ingest() {
  const rag = await mk();
  await rag.ingest(
    "The quarterly brand identity guidelines define the logo, the colour palette and the typography.",
    "js", "branding"
  );
  console.log("JS ingeriu 1 doc (branding) no Postgres");
  await rag.close();
}

async function assertAll() {
  const rag = await mk();
  const cases = [
    ["quando vence o contrato de aluguel?", "contrato"],
    ["rocket launch date Cape Canaveral", "rocket"],
    ["会议在北京哪里举行", "北京"],
    ["código do cofre principal", "7742"],
    ["receita de bolo de fubá", "fubá"],
    ["tension artérielle du patient", "tension"],
  ];
  let ok = 0;
  for (const [q, needle] of cases) {
    const r = await rag.search(q, 3);
    const found = r.fused.slice(0, 3).some((h) => h.text.includes(needle));
    console.log(`  [${found ? "ok" : "FALHOU"}] JS acha doc PY: ${JSON.stringify(q)} -> ${JSON.stringify((r.fused[0]?.text || "").slice(0, 50))}`);
    if (found) ok++;
  }
  await rag.close();
  console.log(`JS achou ${ok}/${cases.length} docs ingeridos pelo Python`);
  process.exit(ok === cases.length ? 0 : 1);
}

const [cmd, arg] = process.argv.slice(2);
if (cmd === "query") await query(arg);
else if (cmd === "ingest") await ingest();
else if (cmd === "assert") await assertAll();
else { console.error("uso: query|ingest|assert"); process.exit(1); }

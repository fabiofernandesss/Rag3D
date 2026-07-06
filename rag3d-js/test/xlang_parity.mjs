// Prova bit-a-bit: a assinatura BIT(1024) que o PYTHON gravou no Postgres é
// idêntica à recomputada em JS para o mesmo texto.
import pg from "pg";
import { HashEncoder } from "../src/encoders.js";
import { Holographer } from "../src/holo.js";

const c = new pg.Client({ connectionString: "postgresql://postgres:rag3d@localhost:5433/rag3d" });
await c.connect();
// usa a coluna ctx (o texto EXATO que o Python embutiu) — sem adivinhar
const row = (await c.query(
  "SELECT ctx, sig::text AS sig FROM holo_grams WHERE text LIKE $1 LIMIT 1", ["O contrato%"]
)).rows[0];

const enc = new HashEncoder({ denseDim: 1024, colbertDim: 128, maxTokens: 256 });
const h = new Holographer(1024, 128);
const sigJs = Holographer.sigToBitstring(h.signDense(enc.encode([row.ctx])[0].dense));

let ham = 0;
for (let i = 0; i < row.sig.length; i++) if (row.sig[i] !== sigJs[i]) ham++;
console.log("assinatura BIT(1024) gravada por Python vs recomputada por JS:");
console.log("  comprimento:", row.sig.length, "bits");
console.log("  Hamming(PY-no-banco, JS-recomputado):", ham, "/ 1024");
console.log(ham === 0
  ? "  => IDÊNTICA bit a bit: o mesmo índice holográfico serve as duas linguagens"
  : "  => diferente (investigar)");
await c.end();
process.exit(ham === 0 ? 0 : 1);

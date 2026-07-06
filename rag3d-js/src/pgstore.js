// Backend Postgres PURO (JS) — sem pgvector. Espelha trirag/pgstore.py e usa a
// MESMA tabela/esquema, então lê e escreve os hologramas do lado Python.
//
// Interface assíncrona (pg é async). Mesmos nomes de tabela/coluna do Python:
// um doc ingerido por qualquer das linguagens é encontrado pela outra.

import { HOLO_BITS, Holographer } from "./holo.js";

const KINDS = "('chunk','turn','summary')";
const now = () => Date.now() / 1000;

const SCHEMA = `
CREATE TABLE IF NOT EXISTS holo_meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS holo_docs(
  id BIGSERIAL PRIMARY KEY, source TEXT, title TEXT,
  created DOUBLE PRECISION, n_tokens INT, meta TEXT);
CREATE TABLE IF NOT EXISTS holo_grams(
  id BIGSERIAL PRIMARY KEY, doc_id BIGINT, parent_id BIGINT,
  kind TEXT NOT NULL DEFAULT 'chunk',
  pos INT, text TEXT NOT NULL, ctx TEXT,
  n_tokens INT, created DOUBLE PRECISION,
  importance REAL DEFAULT 0.5, turn_no INT, accessed_turn INT,
  sig BIT(${HOLO_BITS}), bands INT[], echo BYTEA,
  constellation BYTEA, n_tok INT);
CREATE INDEX IF NOT EXISTS idx_grams_kind ON holo_grams(kind);
CREATE INDEX IF NOT EXISTS idx_grams_doc ON holo_grams(doc_id);
CREATE INDEX IF NOT EXISTS idx_grams_bands ON holo_grams USING GIN(bands);
CREATE TABLE IF NOT EXISTS holo_spectrum(
  term BIGINT NOT NULL, gram_id BIGINT NOT NULL, weight REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_spectrum_term ON holo_spectrum(term);
CREATE INDEX IF NOT EXISTS idx_spectrum_gram ON holo_spectrum(gram_id);`;

const COLS = "id,doc_id,parent_id,kind,pos,text,ctx,n_tokens,created,importance,turn_no,accessed_turn";

export class PgHoloStore {
  constructor(client, denseDim, colbertDim) {
    this.db = client;
    this.holo = new Holographer(denseDim, colbertDim);
    this._nCache = null;
    this._inBatch = false; // uma transação aberta englobando várias escritas
    this.BAND_PREFILTER_THRESHOLD = 20000;
    this.PREFETCH_FACTOR = 4;
    this.PREFETCH_MIN = 200;
  }

  // agrupa muitas escritas numa transação só (ingestão de doc grande) — 1 fsync
  async begin() { await this.db.query("BEGIN"); this._inBatch = true; }
  async commitBatch() { await this.db.query("COMMIT"); this._inBatch = false; this._nCache = null; }
  async rollbackBatch() { await this.db.query("ROLLBACK"); this._inBatch = false; this._nCache = null; }

  static async connect(dsn, denseDim, colbertDim) {
    const { default: pg } = await import("pg");
    const client = new pg.Client({ connectionString: dsn });
    await client.connect();
    const store = new PgHoloStore(client, denseDim, colbertDim);
    await client.query(SCHEMA);
    return store;
  }

  async close() { await this.db.end(); }

  // limpa todos os documentos (mantém holo_meta com o fingerprint do encoder)
  async reset() {
    await this.db.query("TRUNCATE holo_grams, holo_spectrum, holo_docs");
    this._nCache = null;
  }

  // ------------------------------------------------------------- meta ---
  async getMeta(key) {
    const r = await this.db.query("SELECT value FROM holo_meta WHERE key=$1", [key]);
    return r.rows.length ? r.rows[0].value : null;
  }
  async setMeta(key, value) {
    await this.db.query(
      "INSERT INTO holo_meta VALUES($1,$2) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
      [key, value]
    );
  }

  // ------------------------------------------------------------ escrita ---
  async addDoc(source, title, nTokens, meta = {}) {
    const r = await this.db.query(
      "INSERT INTO holo_docs(source,title,created,n_tokens,meta) VALUES($1,$2,$3,$4,$5) RETURNING id",
      [source, title, now(), nTokens, JSON.stringify(meta)]
    );
    return Number(r.rows[0].id);
  }

  async addChunk(docId, text, ctx, nTokens, vec, opts = {}) {
    const { kind = "chunk", pos = 0, parentId = null, importance = 0.5, turnNo = null } = opts;
    const sig = this.holo.signDense(vec.dense);
    const bits = Holographer.sigToBitstring(sig);
    const bands = Holographer.bandsOf(sig);
    const echo = Buffer.from(Holographer.quantize(vec.dense));
    const konst = Buffer.from(this.holo.signTokens(vec.tokens));
    // gram + espectro numa ÚNICA query via CTE: um round-trip por chunk (era 2),
    // e a instrução é atômica por si só (não precisa de BEGIN/COMMIT avulso).
    const terms = [], weights = [];
    for (const [t, w] of vec.sparse) { terms.push(t); weights.push(w); }
    const r = await this.db.query(
      `WITH g AS (
         INSERT INTO holo_grams(doc_id,parent_id,kind,pos,text,ctx,n_tokens,created,importance,turn_no,
           sig,bands,echo,constellation,n_tok)
         VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::bit(${HOLO_BITS}),$12,$13,$14,$15) RETURNING id
       ), s AS (
         INSERT INTO holo_spectrum(term, gram_id, weight)
         SELECT u.term, (SELECT id FROM g), u.weight
         FROM unnest($16::bigint[], $17::real[]) AS u(term, weight)
       )
       SELECT id FROM g`,
      [docId, parentId, kind, pos, text, ctx, nTokens, now(), importance, turnNo,
       bits, bands, echo, konst, vec.tokens.length, terms, weights]
    );
    this._nCache = null;
    return Number(r.rows[0].id);
  }

  async addParent(docId, text, nTokens, pos) {
    const r = await this.db.query(
      "INSERT INTO holo_grams(doc_id,kind,pos,text,ctx,n_tokens,created) VALUES($1,'parent',$2,$3,$4,$5,$6) RETURNING id",
      [docId, pos, text, text, nTokens, now()]
    );
    return Number(r.rows[0].id);
  }

  async commit() {} // autocommit: no-op (mantém interface)

  async deleteChunk(chunkId) {
    await this.db.query("DELETE FROM holo_grams WHERE id=$1", [chunkId]);
    await this.db.query("DELETE FROM holo_spectrum WHERE gram_id=$1", [chunkId]);
    this._nCache = null;
  }

  // ------------------------------------------------------------ leitura ---
  async getChunks(ids) {
    if (!ids.length) return [];
    const r = await this.db.query(`SELECT ${COLS} FROM holo_grams WHERE id = ANY($1)`, [ids]);
    const byId = new Map(r.rows.map((row) => [Number(row.id), normalizeRow(row)]));
    return ids.map((i) => byId.get(i)).filter(Boolean);
  }

  async touchAccess(ids, turnNo) {
    if (!ids.length) return;
    await this.db.query("UPDATE holo_grams SET accessed_turn=$1 WHERE id = ANY($2)", [turnNo, ids]);
  }

  async corpusTokens(kinds = ["chunk", "turn"]) {
    const r = await this.db.query(
      "SELECT COALESCE(SUM(n_tokens),0) AS s FROM holo_grams WHERE kind = ANY($1)", [kinds]
    );
    return Number(r.rows[0].s);
  }

  async allTexts(kinds = ["chunk", "turn"]) {
    const r = await this.db.query(
      "SELECT id,kind,text,created,turn_no FROM holo_grams WHERE kind = ANY($1) ORDER BY id", [kinds]
    );
    return r.rows.map((x) => ({ id: Number(x.id), kind: x.kind, text: x.text, created: x.created, turn_no: x.turn_no }));
  }

  async nChunks() {
    if (this._nCache === null) {
      const r = await this.db.query(`SELECT COUNT(*) AS c FROM holo_grams WHERE kind IN ${KINDS}`);
      this._nCache = Number(r.rows[0].c);
    }
    return this._nCache;
  }

  async lastTurnNo() {
    const r = await this.db.query("SELECT COALESCE(MAX(turn_no),0) AS m FROM holo_grams WHERE kind='turn'");
    return Number(r.rows[0].m);
  }

  // chunks vizinhos do mesmo doc em posições dadas (costura de contiguidade)
  async neighbors(docId, positions) {
    if (docId == null || !positions.length) return [];
    const r = await this.db.query(
      "SELECT pos, text FROM holo_grams WHERE doc_id=$1 AND kind='chunk' AND pos = ANY($2) ORDER BY pos",
      [docId, positions]
    );
    return r.rows.map((x) => ({ pos: x.pos, text: x.text }));
  }

  // ------------------------------------------------- eixo 1: semântico ---
  async denseSearch(qvec, k) {
    const sig = this.holo.signDense(qvec);
    const bits = Holographer.sigToBitstring(sig);
    const prefetch = Math.max(this.PREFETCH_FACTOR * k, this.PREFETCH_MIN);
    const hamExpr = `bit_count(sig # $1::bit(${HOLO_BITS}))`;

    const scan = async (useBands) => {
      let where = `kind IN ${KINDS} AND sig IS NOT NULL`;
      const params = [bits];
      if (useBands) {
        where += " AND bands && $2::int[]";
        params.push(Holographer.bandsOf(sig));
      }
      params.push(prefetch);
      const r = await this.db.query(
        `SELECT id, echo, (${hamExpr}) AS ham FROM holo_grams WHERE ${where} ORDER BY ham ASC LIMIT $${params.length}`,
        params
      );
      return r.rows;
    };

    const big = (await this.nChunks()) > this.BAND_PREFILTER_THRESHOLD;
    let rows = await scan(big);
    if (big && rows.length < prefetch / 2) rows = await scan(false); // fallback recall
    if (!rows.length) return [];

    // re-pontuação exata pelo eco int8
    const q = qvec;
    const scored = rows.map((row) => {
      const echo = Holographer.dequantize(new Uint8Array(row.echo));
      let dot = 0;
      for (let d = 0; d < q.length; d++) dot += echo[d] * q[d];
      return [Number(row.id), dot];
    });
    scored.sort((a, b) => b[1] - a[1] || a[0] - b[0]); // desempate por id, igual ao Python
    return scored.slice(0, k);
  }

  // --------------------------------------------------- eixo 2: léxico ----
  async sparseSearch(qsparse, k) {
    if (!qsparse.size) return [];
    const terms = [...qsparse.keys()];
    const nDocs = Math.max(1, await this.nChunks());
    // uma query só, SEM join: só grams pesquisáveis têm postings (parents não
    // recebem spectrum), então o join+filtro de kind era redundante.
    const r = await this.db.query(
      `SELECT term, gram_id, weight, COUNT(*) OVER (PARTITION BY term) AS df
       FROM holo_spectrum WHERE term = ANY($1)`, [terms]
    );
    const scores = new Map();
    for (const row of r.rows) {
      const t = Number(row.term), df = Number(row.df);
      const idf = Math.log(1.0 + (nDocs - df + 0.5) / (df + 0.5));
      const gid = Number(row.gram_id);
      scores.set(gid, (scores.get(gid) || 0) + qsparse.get(t) * Number(row.weight) * idf);
    }
    return [...scores.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, k);
  }

  // ----------------------------------------------- eixo 3: estrutural ----
  async colbertScores(qtokens, candidateIds) {
    if (!candidateIds.length || !qtokens.length) return [];
    const qConst = this.holo.signTokens(qtokens);
    const r = await this.db.query(
      "SELECT id, constellation FROM holo_grams WHERE id = ANY($1) AND constellation IS NOT NULL",
      [candidateIds]
    );
    const out = r.rows.map((row) => [
      Number(row.id),
      this.holo.binaryMaxsim(qConst, new Uint8Array(row.constellation)),
    ]);
    out.sort((a, b) => b[1] - a[1] || a[0] - b[0]); // desempate por id, igual ao Python
    return out;
  }
}

function normalizeRow(row) {
  return {
    id: Number(row.id),
    doc_id: row.doc_id === null ? null : Number(row.doc_id),
    parent_id: row.parent_id === null ? null : Number(row.parent_id),
    kind: row.kind, pos: row.pos, text: row.text, ctx: row.ctx,
    n_tokens: row.n_tokens, created: row.created,
    importance: row.importance, turn_no: row.turn_no, accessed_turn: row.accessed_turn,
  };
}

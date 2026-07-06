// Store em memória com persistência JSON — o caminho zero-dependência (sem
// Postgres). Mesma interface assíncrona do PgHoloStore, força bruta nos três
// eixos sobre a saída crua do encoder. Bom para corpora pequenos/médios e demos.
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";

const KINDS = new Set(["chunk", "turn", "summary"]);
const now = () => Date.now() / 1000;

export class MemStore {
  constructor(file) {
    this.file = file;
    this.meta = new Map();
    this.docs = [];
    this.grams = [];
    this.seq = 0;
    if (file && existsSync(file)) this._load();
  }

  _load() {
    const d = JSON.parse(readFileSync(this.file, "utf8"));
    this.meta = new Map(d.meta);
    this.seq = d.seq;
    this.grams = d.grams.map((g) => ({
      ...g,
      dense: g.dense ? Float32Array.from(g.dense) : null,
      sparse: g.sparse ? new Map(g.sparse) : new Map(),
      tokens: g.tokens ? g.tokens.map((t) => Float32Array.from(t)) : [],
    }));
  }

  _persist() {
    if (!this.file) return;
    mkdirSync(path.dirname(this.file), { recursive: true });
    writeFileSync(this.file, JSON.stringify({
      meta: [...this.meta], seq: this.seq,
      grams: this.grams.map((g) => ({
        ...g,
        dense: g.dense ? Array.from(g.dense) : null,
        sparse: [...g.sparse],
        tokens: g.tokens.map((t) => Array.from(t)),
      })),
    }));
  }

  async getMeta(k) { return this.meta.has(k) ? this.meta.get(k) : null; }
  async setMeta(k, v) { this.meta.set(k, v); this._persist(); }

  async addDoc(source, title, nTokens, meta = {}) {
    const id = ++this.seq;
    this.docs.push({ id, source, title, created: now(), n_tokens: nTokens, meta });
    return id;
  }

  async addChunk(docId, text, ctx, nTokens, vec, opts = {}) {
    const { kind = "chunk", pos = 0, parentId = null, importance = 0.5, turnNo = null } = opts;
    const id = ++this.seq;
    this.grams.push({
      id, doc_id: docId, parent_id: parentId, kind, pos, text, ctx,
      n_tokens: nTokens, created: now(), importance, turn_no: turnNo, accessed_turn: null,
      dense: vec.dense, sparse: vec.sparse, tokens: vec.tokens,
    });
    return id;
  }

  async addParent(docId, text, nTokens, pos) {
    const id = ++this.seq;
    this.grams.push({
      id, doc_id: docId, parent_id: null, kind: "parent", pos, text, ctx: text,
      n_tokens: nTokens, created: now(), importance: 0.5, turn_no: null, accessed_turn: null,
      dense: null, sparse: new Map(), tokens: [],
    });
    return id;
  }

  // batch é no-op em memória (mantém a mesma interface do PgHoloStore)
  async begin() {}
  async commitBatch() { this._persist(); }
  async rollbackBatch() {}

  async commit() { this._persist(); }

  // limpa todos os documentos (mantém meta com o fingerprint do encoder)
  async reset() {
    this.grams = [];
    this.docs = [];
    this.seq = 0;
    this._persist();
  }

  async deleteChunk(id) {
    this.grams = this.grams.filter((g) => g.id !== id);
    this._persist();
  }

  _pub(g) {
    const { dense, sparse, tokens, ...rest } = g;
    return rest;
  }

  async getChunks(ids) {
    const byId = new Map(this.grams.map((g) => [g.id, g]));
    return ids.map((i) => byId.get(i)).filter(Boolean).map((g) => this._pub(g));
  }

  async touchAccess(ids, turnNo) {
    const set = new Set(ids);
    for (const g of this.grams) if (set.has(g.id)) g.accessed_turn = turnNo;
    this._persist();
  }

  async corpusTokens(kinds = ["chunk", "turn"]) {
    const s = new Set(kinds);
    return this.grams.filter((g) => s.has(g.kind)).reduce((a, g) => a + (g.n_tokens || 0), 0);
  }

  async allTexts(kinds = ["chunk", "turn"]) {
    const s = new Set(kinds);
    return this.grams.filter((g) => s.has(g.kind))
      .map((g) => ({ id: g.id, kind: g.kind, text: g.text, created: g.created, turn_no: g.turn_no }));
  }

  _searchable() { return this.grams.filter((g) => KINDS.has(g.kind)); }
  async nChunks() { return this._searchable().length; }
  async lastTurnNo() {
    return this.grams.filter((g) => g.kind === "turn").reduce((m, g) => Math.max(m, g.turn_no || 0), 0);
  }

  async neighbors(docId, positions) {
    if (docId == null || !positions.length) return [];
    const set = new Set(positions);
    return this.grams
      .filter((g) => g.doc_id === docId && g.kind === "chunk" && set.has(g.pos))
      .sort((a, b) => a.pos - b.pos)
      .map((g) => ({ pos: g.pos, text: g.text }));
  }

  async denseSearch(qvec, k) {
    const out = [];
    for (const g of this._searchable()) {
      if (!g.dense) continue;
      let dot = 0;
      for (let d = 0; d < qvec.length; d++) dot += qvec[d] * g.dense[d];
      out.push([g.id, dot]);
    }
    out.sort((a, b) => b[1] - a[1]);
    return out.slice(0, k);
  }

  async sparseSearch(qsparse, k) {
    if (!qsparse.size) return [];
    const docs = this._searchable();
    const nDocs = Math.max(1, docs.length);
    const df = new Map();
    for (const t of qsparse.keys()) {
      let c = 0;
      for (const g of docs) if (g.sparse.has(t)) c++;
      if (c) df.set(t, c);
    }
    const scores = new Map();
    for (const g of docs) {
      let s = 0;
      for (const [t, qw] of qsparse) {
        if (!g.sparse.has(t)) continue;
        const idf = Math.log(1.0 + (nDocs - df.get(t) + 0.5) / (df.get(t) + 0.5));
        s += qw * g.sparse.get(t) * idf;
      }
      if (s > 0) scores.set(g.id, s);
    }
    return [...scores.entries()].sort((a, b) => b[1] - a[1]).slice(0, k);
  }

  async colbertScores(qtokens, candidateIds) {
    if (!candidateIds.length || !qtokens.length) return [];
    const set = new Set(candidateIds);
    const out = [];
    for (const g of this.grams) {
      if (!set.has(g.id) || !g.tokens.length) continue;
      let sum = 0;
      for (const qt of qtokens) {
        let best = -Infinity;
        for (const dt of g.tokens) {
          let dot = 0;
          for (let d = 0; d < qt.length; d++) dot += qt[d] * dt[d];
          if (dot > best) best = dot;
        }
        sum += best;
      }
      out.push([g.id, sum / qtokens.length]);
    }
    out.sort((a, b) => b[1] - a[1]);
    return out;
  }
}

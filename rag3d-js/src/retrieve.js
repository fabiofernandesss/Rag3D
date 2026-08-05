// Busca tridimensional — espelha trirag/retrieve.py. Três frentes + fusão.
import { fermionicSelect, fuse } from "./fusion.js";

const SYS_EXPAND =
  "Você reescreve consultas para busca em documentos normativos (editais, leis). " +
  "Dada uma pergunta, gere até {N} variações com termos alternativos que provavelmente " +
  "aparecem no documento (sinônimos, palavras-chave, números de seção/item, siglas expandidas). " +
  "Uma variação por linha, sem numeração, sem explicações.";

export class TriRetriever {
  constructor(store, encoder, cfg, reranker = null, llm = null) {
    this.store = store; this.encoder = encoder; this.cfg = cfg;
    this.reranker = reranker; this.llm = llm;
  }

  // gera variações da consulta (LLM) e combina os vetores como OR — mesma
  // saída de rankings do backend, só que com recall maior. Falha silenciosa.
  async _expandedQueryVec(query) {
    if (!this.cfg.expandQuery || !this.llm || !this.llm.available()) {
      return this.encoder.encode([query], true)[0];
    }
    let variants = [];
    try {
      const raw = await this.llm.complete(
        SYS_EXPAND.replace("{N}", this.cfg.expandQueryMax || 3),
        [{ role: "user", content: query }], 200
      );
      variants = raw.split("\n").map((s) => s.trim()).filter(Boolean).slice(0, this.cfg.expandQueryMax || 3);
    } catch { /* acessório */ }
    if (!variants.length) return this.encoder.encode([query], true)[0];
    const all = [query, ...variants];
    const vecs = this.encoder.encode(all, true);
    // combina: denso = média normalizada; esparso = união com peso máximo; tokens = pool.
    const dim = vecs[0].dense.length;
    const dense = new Float32Array(dim);
    for (const v of vecs) for (let i = 0; i < dim; i++) dense[i] += v.dense[i];
    let n = 0; for (let i = 0; i < dim; i++) n += dense[i] * dense[i];
    n = Math.sqrt(n); if (n > 0) for (let i = 0; i < dim; i++) dense[i] /= n;
    const sparse = new Map();
    for (const v of vecs) for (const [t, w] of v.sparse) if (!sparse.has(t) || sparse.get(t) < w) sparse.set(t, w);
    const tokens = vecs.flatMap((v) => v.tokens).slice(0, this.encoder.maxTokens || 256);
    return { dense, sparse, tokens };
  }

  // busca chunks + pais em DUAS queries (não uma por lista de ids)
  async _prefetch(ids) {
    const rows = await this.store.getChunks([...new Set(ids)]);
    const byId = new Map(rows.map((r) => [r.id, r]));
    const parentIds = [...new Set(rows.filter((r) => r.parent_id).map((r) => r.parent_id))];
    const parents = parentIds.length
      ? new Map((await this.store.getChunks(parentIds)).map((p) => [p.id, p]))
      : new Map();
    return { byId, parents };
  }

  // monta um hit dos mapas já buscados (sem tocar o banco)
  _assemble(id, byId, parents, score = null) {
    const r = byId.get(id);
    if (!r) return null;
    const wide = r.parent_id && parents.has(r.parent_id) ? parents.get(r.parent_id).text : null;
    return {
      id: r.id, kind: r.kind, doc_id: r.doc_id, pos: r.pos, text: r.text, wide: wide || r.text,
      n_tokens: r.n_tokens, turn_no: r.turn_no, accessed_turn: r.accessed_turn,
      created: r.created, importance: r.importance, score: score ?? null,
    };
  }

  // costura de contiguidade: para cada hit, junta os vizinhos (pos±raio) do
  // mesmo doc num bloco contíguo — reconstrói listas/tabelas/seções partidas
  // pelo chunking. Vira o campo `wide` (o que a leitora enxerga).
  async _stitch(hits) {
    const R = this.cfg.stitchRadius | 0;
    if (R <= 0) return hits;
    // agrupa posições por doc numa query só
    const byDoc = new Map();
    for (const h of hits) {
      if (h.doc_id == null || h.pos == null || h.kind !== "chunk") continue;
      const set = byDoc.get(h.doc_id) || new Set();
      for (let d = -R; d <= R; d++) set.add(h.pos + d);
      byDoc.set(h.doc_id, set);
    }
    const textByDocPos = new Map(); // `${doc}:${pos}` -> text
    for (const [docId, set] of byDoc) {
      const rows = await this.store.neighbors(docId, [...set]);
      for (const row of rows) textByDocPos.set(`${docId}:${row.pos}`, row.text);
    }
    for (const h of hits) {
      if (h.doc_id == null || h.pos == null || h.kind !== "chunk") continue;
      const parts = [];
      for (let d = -R; d <= R; d++) {
        const t = textByDocPos.get(`${h.doc_id}:${h.pos + d}`);
        if (t) parts.push(t);
      }
      if (parts.length) h.wide = parts.join(" ");
    }
    return hits;
  }

  async _hydrate(ids, scores = null) {
    const { byId, parents } = await this._prefetch(ids);
    const sc = scores ? new Map(ids.map((id, i) => [id, scores[i]])) : new Map();
    return ids.map((id) => this._assemble(id, byId, parents, sc.get(id))).filter(Boolean);
  }

  async search(query, topK = null, channelK = null) {
    topK = topK || this.cfg.topK;
    channelK = channelK || this.cfg.channelK;
    const q = await this._expandedQueryVec(query);

    const dense = await this.store.denseSearch(q.dense, channelK);
    const sparse = await this.store.sparseSearch(q.sparse, channelK);
    const pool = [...new Set([...dense.map((x) => x[0]), ...sparse.map((x) => x[0])])];
    const struct = (await this.store.colbertScores(q.tokens, pool)).slice(0, channelK);
    const channels = { semantico: dense, lexico: sparse, estrutural: struct };

    const doRerank = this.reranker && this.reranker.available();
    const doDiv = (this.cfg.diversity || 0) > 0;
    let fuseK = topK;
    if (doRerank) fuseK = Math.max(fuseK, this.cfg.rerankPool);
    if (doDiv) fuseK = Math.max(fuseK, this.cfg.diversityPool);
    const fusedHits = fuse(channels, this.cfg.channelWeights, fuseK, this.cfg.fusion,
      this.cfg.interferenceStrength, this.cfg.rrfK, this.cfg.coherenceStrength);

    // hidratação em lote: uma prefetch p/ fundido + as 3 visões juntas
    const fusedIds = fusedHits.map((h) => h.chunkId);
    const viewRank = {};
    for (const [name, ranking] of Object.entries(channels)) viewRank[name] = ranking.slice(0, topK);
    const allIds = [...fusedIds, ...Object.values(viewRank).flatMap((r) => r.map((x) => x[0]))];
    const { byId, parents } = await this._prefetch(allIds);

    let fused = fusedIds.map((id) => this._assemble(id, byId, parents)).filter(Boolean);
    const hitById = new Map(fusedHits.map((h) => [h.chunkId, h]));
    for (const row of fused) {
      const h = hitById.get(row.id);
      row.score = h.score; row.classical = h.classical; row.interference = h.interference;
      row.channels = h.channels; row.perChannel = h.perChannel;
    }
    if (doRerank) fused = await this.reranker.rerank(query, fused, doDiv ? this.cfg.diversityPool : topK);

    // seleção fermiônica (MAP-DPP): escolhe o CONJUNTO de k, sem redundância
    if (doDiv && fused.length > topK) {
      const pool = fused.slice(0, this.cfg.diversityPool);
      const vecs = await this.store.denseVecs(pool.map((h) => h.id));
      const keep = fermionicSelect(pool.map((h) => [h.id, h.score ?? 0]), vecs, topK, this.cfg.diversity);
      const order = new Map(keep.map((id, i) => [id, i]));
      fused = pool.filter((h) => order.has(h.id)).sort((a, b) => order.get(a.id) - order.get(b.id));
    } else {
      fused = fused.slice(0, topK);
    }
    await this._stitch(fused); // costura de vizinhos no top-k final

    const views = {};
    for (const [name, ranking] of Object.entries(viewRank)) {
      views[name] = ranking.map(([id, s]) => this._assemble(id, byId, parents, s)).filter(Boolean);
    }

    return {
      query, fused, views,
      stats: {
        candidatos: { semantico: dense.length, lexico: sparse.length, estrutural: struct.length },
        pool: pool.length, fusao: this.cfg.fusion,
      },
    };
  }
}

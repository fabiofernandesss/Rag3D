// Ingestão adaptativa — espelha trirag/ingest.py. As três formas nascem ao salvar.
//
// Toda a escrita de um documento acontece numa transação só (rápido para docs
// grandes: 1 fsync em vez de 1 por chunk). As chamadas de LLM (enriquecimento
// contextual e resumo) ficam FORA da transação — nunca seguram o banco aberto.
import { estimateTokens, normalize, splitSentences } from "./textproc.js";

const ENRICH = (doc, chunk) =>
  `Documento (início):\n<doc>\n${doc}\n</doc>\n\nTrecho:\n<chunk>\n${chunk}\n</chunk>\n\n` +
  "Escreva 1-2 frases curtas, na língua do documento, situando este trecho dentro do documento. Responda só as frases.";

// digesto que cobre o documento INTEIRO (janelas espaçadas), não só o começo —
// para o resumo de doc gigante refletir as 200 páginas, com 1 chamada de LLM.
function digest(text, budget = 20000) {
  if (text.length <= budget) return text;
  const parts = 8;
  const win = Math.floor(budget / parts);
  const step = Math.floor(text.length / parts);
  let out = "";
  for (let i = 0; i < parts; i++) out += text.slice(i * step, i * step + win) + "\n[...]\n";
  return out;
}

export function chunkText(text, targetTokens, overlapTokens) {
  const sents = splitSentences(text);
  const chunks = [];
  let cur = [], curTok = 0;
  for (const s of sents) {
    const st = estimateTokens(s);
    if (cur.length && curTok + st > targetTokens) {
      chunks.push(cur.join(" "));
      const keep = []; let kept = 0;
      for (let i = cur.length - 1; i >= 0; i--) {
        const pt = estimateTokens(cur[i]);
        if (kept + pt > overlapTokens) break;
        keep.unshift(cur[i]); kept += pt;
      }
      cur = keep; curTok = kept;
    }
    cur.push(s); curTok += st;
  }
  if (cur.length) chunks.push(cur.join(" "));
  return chunks.filter((c) => c.trim());
}

export class Ingestor {
  constructor(store, encoder, cfg, llm = null) {
    this.store = store; this.encoder = encoder; this.cfg = cfg; this.llm = llm;
  }

  async ingestText(text, source = "inline", title = "") {
    text = normalize(text);
    if (!text) return { docId: null, chunks: 0 };
    const nTok = estimateTokens(text);
    const cps = [...text];
    title = title || cps.slice(0, 60).join("").replace(/\n/g, " ") + (cps.length > 60 ? "…" : "");
    return nTok <= this.cfg.tinyDocTokens
      ? this._tiny(source, title, text, nTok)
      : this._chunked(source, title, text, nTok);
  }

  async _tiny(source, title, text, nTok) {
    const ctx = title && !text.includes(title) ? `[${title}] ${text}` : text;
    const vec = this.encoder.encode([ctx])[0];
    await this.store.begin();
    try {
      const docId = await this.store.addDoc(source, title, nTok);
      await this.store.addChunk(docId, text, ctx, nTok, vec, { pos: 0 });
      await this.store.commitBatch();
      await this.store.commit();
      return { docId, chunks: 1, tokens: nTok };
    } catch (e) { await this.store.rollbackBatch(); throw e; }
  }

  async _chunked(source, title, text, nTok) {
    const chunks = chunkText(text, this.cfg.chunkTokens, this.cfg.chunkOverlap);
    const perParent = Math.max(1, Math.floor(this.cfg.parentTokens / Math.max(1, this.cfg.chunkTokens)));

    // --- preparação FORA da transação: enriquecimento (LLM) + encode (CPU) ---
    const head = text.slice(0, 6000);
    const enrich = this.cfg.contextualEnrich && this.llm && this.llm.available();
    const ctxs = [];
    for (const c of chunks) {
      let prefix = `[${title}] `;
      if (enrich) {
        try {
          const situ = (await this.llm.complete(
            "Você situa trechos dentro de documentos. Seja breve.",
            [{ role: "user", content: ENRICH(head, c.slice(0, 2000)) }], 120
          )).trim();
          prefix = `[${title}] ${situ}\n`;
        } catch { /* acessório */ }
      }
      ctxs.push(prefix + c);
    }
    const vecs = this.encoder.encode(ctxs);

    // resumo de doc gigante (RAPTOR-lite): 1 chamada de LLM, cobrindo o doc todo
    let summary = null;
    if (nTok >= this.cfg.hugeDocTokens && this.llm && this.llm.available()) {
      try {
        const s = (await this.llm.complete(
          "Você resume documentos com fidelidade, na língua do original.",
          [{ role: "user", content: `Resuma em ~12 frases o documento a seguir (trechos ao longo dele):\n\n${digest(text)}` }],
          600
        )).trim();
        const sctx = `[resumo de ${title}] ${s}`;
        summary = { text: s, ctx: sctx, vec: this.encoder.encode([sctx])[0] };
      } catch { /* acessório */ }
    }

    // --- escrita: uma transação só para o documento inteiro ---
    await this.store.begin();
    try {
      const docId = await this.store.addDoc(source, title, nTok);
      const parentIds = [];
      for (let i = 0; i < chunks.length; i += perParent) {
        const slice = chunks.slice(i, i + perParent);
        const ptxt = slice.join(" ");
        const pid = await this.store.addParent(docId, ptxt, estimateTokens(ptxt), i);
        for (let j = 0; j < slice.length; j++) parentIds.push(pid);
      }
      for (let i = 0; i < chunks.length; i++) {
        await this.store.addChunk(docId, chunks[i], ctxs[i], estimateTokens(chunks[i]), vecs[i],
          { pos: i, parentId: parentIds[i] });
      }
      if (summary) {
        await this.store.addChunk(docId, summary.text, summary.ctx,
          estimateTokens(summary.text), summary.vec, { kind: "summary", pos: -1 });
      }
      await this.store.commitBatch();
      await this.store.commit();
      return { docId, chunks: chunks.length + (summary ? 1 : 0), tokens: nTok };
    } catch (e) { await this.store.rollbackBatch(); throw e; }
  }

  async ingestFile(fp) {
    const { readFileSync } = await import("node:fs");
    const path = await import("node:path");
    const text = readFileSync(fp, "utf8");
    return this.ingestText(text, fp, path.basename(fp));
  }
}

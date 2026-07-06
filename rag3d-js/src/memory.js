// Memória de conversa "infinita" — espelha trirag/memory.py.
import { estimateTokens } from "./textproc.js";

const SUMMARY = (summary, turns) =>
  `Resumo atual da conversa:\n<resumo>\n${summary}\n</resumo>\n\nNovos turnos:\n<turnos>\n${turns}\n</turnos>\n\n` +
  "Atualize o resumo: preserve fatos, decisões, nomes, números e preferências do usuário; descarte conversa fiada. " +
  "Máximo ~15 frases, na língua da conversa. Responda só o resumo.";

export class ChatMemory {
  constructor(store, encoder, cfg, llm = null) {
    this.store = store; this.encoder = encoder; this.cfg = cfg; this.llm = llm;
  }

  async recordTurn(userMsg, assistantMsg) {
    const turnNo = (await this.store.lastTurnNo()) + 1;
    const text = `[usuário] ${userMsg}\n[assistente] ${assistantMsg}`;
    const nTok = estimateTokens(text);
    const importance = Math.min(1.0, 0.35 + nTok / 1500.0);
    const vec = this.encoder.encode([text])[0];
    await this.store.addChunk(null, text, text, nTok, vec, { kind: "turn", turnNo, importance });
    await this.store.commit();
    if (turnNo % this.cfg.summaryEveryTurns === 0) await this._consolidate(turnNo);
    return turnNo;
  }

  async _consolidate(uptoTurn) {
    if (!this.llm || !this.llm.available()) return;
    const old = (await this.store.getMeta("rolling_summary")) || "(vazio)";
    const turns = (await this.store.allTexts(["turn"]))
      .filter((t) => (t.turn_no || 0) > uptoTurn - this.cfg.summaryEveryTurns);
    const body = turns.map((t) => t.text.slice(0, 1200)).join("\n---\n");
    let neu;
    try {
      neu = (await this.llm.complete("Você mantém a memória de longo prazo de uma conversa.",
        [{ role: "user", content: SUMMARY(old, body) }], 700)).trim();
    } catch { return; }
    await this.store.setMeta("rolling_summary", neu);
    await this.store.setMeta("rolling_summary_turn", String(uptoTurn));
    const oldId = await this.store.getMeta("rolling_summary_chunk");
    if (oldId) await this.store.deleteChunk(Number(oldId));
    const sctx = `[resumo da conversa] ${neu}`;
    const vec = this.encoder.encode([sctx])[0];
    const cid = await this.store.addChunk(null, neu, sctx, estimateTokens(neu), vec,
      { kind: "summary", importance: 0.9 });
    await this.store.setMeta("rolling_summary_chunk", String(cid));
    await this.store.commit();
  }

  async _rescore(result, nowTurn) {
    const hits = result.fused;
    if (!hits.length) return [];
    const smax = Math.max(...hits.map((h) => h.score)) || 1.0;
    for (const h of hits) {
      const rel = smax > 0 ? h.score / smax : 0;
      let rec;
      if (h.kind === "turn" && h.turn_no != null) {
        const ref = Math.max(h.turn_no, h.accessed_turn || 0);
        rec = Math.pow(0.5, Math.max(0, nowTurn - ref) / this.cfg.recencyHalfLifeTurns);
      } else rec = 0.5;
      h.memoryScore = this.cfg.wRelevance * rel + this.cfg.wRecency * rec
        + this.cfg.wImportance * (h.importance ?? 0.5);
    }
    hits.sort((a, b) => b.memoryScore - a.memoryScore);
    return hits;
  }

  async buildContext(query, retriever, recentTurns = 4) {
    const total = await this.store.corpusTokens(["chunk", "turn"]);

    if (total <= this.cfg.smallCorpusTokens) {
      const texts = await this.store.allTexts(["chunk", "turn"]);
      return {
        mode: "corpus_integral",
        summary: await this.store.getMeta("rolling_summary"),
        blocks: texts.map((t) => t.text), views: null, tokens: total,
      };
    }

    const result = await retriever.search(query);
    const nowTurn = await this.store.lastTurnNo();
    const hits = await this._rescore(result, nowTurn);
    const last = (await this.store.allTexts(["turn"])).slice(-recentTurns);
    const recentIds = new Set(last.map((t) => t.id));

    const budget = this.cfg.memoryBudgetTokens;
    const blocks = []; let used = 0;
    for (const h of hits) {
      if (recentIds.has(h.id)) continue;
      const tW = estimateTokens(h.wide);
      if (used + tW > budget) {
        const tN = estimateTokens(h.text);
        if (used + tN > budget) continue;
        blocks.push({ ...h, chosen: h.text }); used += tN;
      } else { blocks.push({ ...h, chosen: h.wide }); used += tW; }
      if (used >= budget) break;
    }
    await this.store.touchAccess(blocks.map((b) => b.id), nowTurn);

    return {
      mode: "retrieval",
      summary: await this.store.getMeta("rolling_summary"),
      recent: last.map((t) => t.text),
      blocks, views: result.views, stats: result.stats, tokens: used,
    };
  }
}

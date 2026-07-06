// Reranker LLM-agnóstico — espelha trirag/rerank.py. Maior ganho da pesquisa.
const SYS =
  "Você é um reordenador de relevância. Recebe uma pergunta e trechos numerados. " +
  "Responda APENAS um array JSON com os números dos trechos, do mais relevante para o menos " +
  "relevante, incluindo só os que ajudam. Exemplo: [3, 1, 5]. Sem texto fora do array.";

export class Reranker {
  constructor(llm, snippetChars = 500) { this.llm = llm; this.snippetChars = snippetChars; }
  available() { return this.llm && this.llm.available(); }

  async rerank(query, hits, topK = null) {
    if (!this.available() || hits.length < 2) return topK ? hits.slice(0, topK) : hits;
    const listing = hits.map((h, i) => `[${i}] ${(h.text || "").slice(0, this.snippetChars)}`).join("\n");
    const prompt = `PERGUNTA: ${query}\n\nTRECHOS:\n${listing}\n\nOrdem de relevância (array JSON):`;
    let order;
    try {
      const raw = await this.llm.complete(SYS, [{ role: "user", content: prompt }], 200);
      order = parseOrder(raw, hits.length);
    } catch { return topK ? hits.slice(0, topK) : hits; }
    if (!order.length) return topK ? hits.slice(0, topK) : hits;

    const seen = new Set(order);
    const reordered = order.map((i) => hits[i]);
    hits.forEach((h, i) => { if (!seen.has(i)) reordered.push(h); });
    reordered.forEach((h, rank) => (h.rerank = rank));
    return topK ? reordered.slice(0, topK) : reordered;
  }
}

function parseOrder(raw, n) {
  const m = raw.match(/\[[\d,\s]*\]/);
  let nums;
  if (m) { try { nums = JSON.parse(m[0]); } catch { nums = m[0].match(/\d+/g) || []; } }
  else nums = raw.match(/\d+/g) || [];
  const out = [];
  for (const x of nums) { const i = parseInt(x, 10); if (i >= 0 && i < n && !out.includes(i)) out.push(i); }
  return out;
}

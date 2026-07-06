// Camada de leitura — espelha trirag/reader.py. Modo fast e tri (3 leitores + Yah).
const AXIS = {
  semantico: "EIXO SEMÂNTICO (significado)",
  lexico: "EIXO LÉXICO (termos exatos)",
  estrutural: "EIXO ESTRUTURAL (correspondência fina)",
};
// REGRAS DE GROUNDING ESTRITO — compartilhadas por todos os leitores. Para
// documentos normativos (editais, leis, contratos) a regra de ouro é: só
// afirmar o que está TEXTUALMENTE nas evidências; na dúvida, dizer que não há
// base. Nunca inferir a partir da ausência.
const GROUNDING =
  "REGRAS (documento normativo — siga à risca):\n" +
  "1. Responda SOMENTE com o que está textualmente escrito nas EVIDÊNCIAS. Não use conhecimento externo.\n" +
  "2. Se as evidências não contêm a resposta, diga exatamente: \"O documento não fornece base suficiente para responder.\" e PARE. É melhor não responder do que inferir.\n" +
  "3. PROIBIDO inferir, deduzir ou completar. PROIBIDAS as expressões: \"pela lógica\", \"implicitamente\", \"é possível concluir\", \"infere-se\", \"inerente\", \"portanto obrigatório\".\n" +
  "4. Concordância entre eixos/recuperadores NÃO é evidência. Se nenhum trecho apoia a afirmação, responda que não há base — mesmo que as três análises 'apontem' para algo.\n" +
  "5. Ao listar itens (leis, critérios, requisitos, normas), liste TODOS os que aparecem nas evidências, sem omitir nem resumir. Em listas NUMERADAS (ex.: 3.3.1, 3.3.2, ...; a), b), c)), confira a sequência e inclua CADA item — nunca pule um número/letra intermediário. Números de lei (ex.: 13.243, 14.133) são itens da lista, não ruído; inclua-os. Se houver indício de que a lista continua fora do trecho, diga isso.\n" +
  "6. Distinga PRINCÍPIO de OBRIGAÇÃO universal. Um \"princípio orientador\" ou uma \"característica\" que aparece na DEFINIÇÃO de um modelo NÃO significa que TODA solução é obrigada a exibir esse traço — só que quando aplica esse modelo o princípio vale. Respeite \"e/ou\": \"A ou B\" não vira \"A obrigatório\". Um item listado como PRIORIDADE (\"prioriza propostas de...\") não é requisito universal, é critério de preferência. Distinga DEFINIÇÃO conceitual de trecho operacional.\n" +
  "7. Cite os trechos por [n]. Não mencione 'fusão', 'eixos' ou o método como justificativa — justifique só pelo texto citado.\n" +
  "Responda na língua da pergunta.";

const SYS_READER = "Você responde perguntas sobre um documento.\n" + GROUNDING;
const SYS_AXIS = (axis) =>
  `Você é o leitor do ${axis}. Use SOMENTE os trechos da sua visão. Direto (máx ~8 frases). ` +
  "Se sua visão não contém a resposta, diga apenas 'meu eixo não encontrou' — NÃO tente deduzir.\n" + GROUNDING;
const SYS_FINAL =
  "Você é a leitora final. Três leitores buscaram por três eixos e você vê as três respostas e as EVIDÊNCIAS (trechos). " +
  "REGRA CENTRAL: se QUALQUER um dos três leitores respondeu citando um trecho literal (com [n] ou entre aspas), essa resposta É fundamentada — USE-A, não descarte. Basta UM eixo ter achado. " +
  "Só responda 'não há base suficiente' quando os TRÊS leitores disseram que não encontraram E nenhum trecho apoia. " +
  "DESCARTE apenas conclusões que um leitor tirou SEM citar trecho (dedução). Nunca troque uma citação real por 'não há base'.\n" + GROUNDING;

// maxChars generoso: a janela costurada (contiguity) pode ter ~6k chars e não
// pode ser cortada no meio de uma lista. O orçamento de tokens já limita o total.
function fmtBlocks(blocks, maxChars = 6500) {
  if (!blocks || !blocks.length) return "(nenhuma evidência)";
  return blocks.map((b, i) => {
    const text = typeof b === "string" ? b : b.chosen ?? b.text ?? "";
    return `[${i + 1}] ${text.slice(0, maxChars)}`;
  }).join("\n\n");
}

export class Reader {
  constructor(cfg, llm, axisLlms = {}) { this.cfg = cfg; this.llm = llm; this.axisLlms = axisLlms; }

  async readFast(query, ctx) {
    const parts = [];
    if (ctx.summary) parts.push(`MEMÓRIA DA CONVERSA:\n${ctx.summary}`);
    if (ctx.recent) parts.push("TURNOS RECENTES:\n" + ctx.recent.join("\n"));
    if (ctx.mode === "corpus_integral")
      parts.push("CONTEÚDO COMPLETO (corpus pequeno, sem retrieval):\n" + fmtBlocks(ctx.blocks));
    else {
      parts.push("EVIDÊNCIAS (fusão dos 3 eixos):\n" + fmtBlocks(ctx.blocks));
      for (const [axis, rows] of Object.entries(ctx.views || {})) {
        const extra = rows.slice(0, 3);
        if (extra.length) parts.push(`${AXIS[axis] || axis} — top 3:\n` + fmtBlocks(extra, 600));
      }
    }
    const prompt = parts.join("\n\n") + `\n\nPERGUNTA: ${query}`;
    const answer = await this.llm.complete(SYS_READER, [{ role: "user", content: prompt }], this.cfg.maxAnswerTokens);
    return { answer, readMode: "fast", subAnswers: null };
  }

  async readTri(query, ctx) {
    if (ctx.mode === "corpus_integral" || !ctx.views) return this.readFast(query, ctx);
    const views = ctx.views;
    const axes = Object.keys(views);
    const subArr = await Promise.all(axes.map(async (axis) => {
      const llm = this.axisLlms[axis] || this.llm;
      const body = fmtBlocks(views[axis], 3000);
      const prompt = `VISÃO DO SEU EIXO:\n${body}\n\nPERGUNTA: ${query}`;
      try {
        return [axis, await llm.complete(SYS_AXIS(AXIS[axis] || axis), [{ role: "user", content: prompt }], 500)];
      } catch (e) { return [axis, `(leitor do eixo falhou: ${e.message})`]; }
    }));
    const sub = Object.fromEntries(subArr);

    const parts = [];
    if (ctx.summary) parts.push(`MEMÓRIA DA CONVERSA:\n${ctx.summary}`);
    if (ctx.recent) parts.push("TURNOS RECENTES:\n" + ctx.recent.join("\n"));
    for (const [axis, ans] of Object.entries(sub)) parts.push(`RESPOSTA DO ${AXIS[axis] || axis}:\n${ans.trim()}`);
    parts.push("EVIDÊNCIAS MAIS FORTES (fusão quântica):\n" + fmtBlocks(ctx.blocks, 6500));
    const prompt = parts.join("\n\n") + `\n\nPERGUNTA ORIGINAL: ${query}\n\nFaça a leitura final.`;
    const final = await this.llm.complete(SYS_FINAL, [{ role: "user", content: prompt }], this.cfg.maxAnswerTokens);
    return { answer: final, readMode: "tri", subAnswers: sub };
  }

  async read(query, ctx, mode = null) {
    mode = mode || this.cfg.readMode;
    if (!this.llm.available())
      return { answer: null, readMode: "retrieval_only", subAnswers: null,
        note: "Sem LLM configurado: devolvendo só as evidências." };
    return mode === "tri" ? this.readTri(query, ctx) : this.readFast(query, ctx);
  }

  // --- streaming: eixos rodam primeiro, a resposta final faz stream ---------

  _fastPrompt(query, ctx) {
    const parts = [];
    if (ctx.summary) parts.push(`MEMÓRIA DA CONVERSA:\n${ctx.summary}`);
    if (ctx.recent) parts.push("TURNOS RECENTES:\n" + ctx.recent.join("\n"));
    if (ctx.mode === "corpus_integral")
      parts.push("CONTEÚDO COMPLETO (corpus pequeno, sem retrieval):\n" + fmtBlocks(ctx.blocks));
    else {
      parts.push("EVIDÊNCIAS (fusão dos 3 eixos):\n" + fmtBlocks(ctx.blocks));
      for (const [axis, rows] of Object.entries(ctx.views || {})) {
        const extra = rows.slice(0, 3);
        if (extra.length) parts.push(`${AXIS[axis] || axis} — top 3:\n` + fmtBlocks(extra, 600));
      }
    }
    return parts.join("\n\n") + `\n\nPERGUNTA: ${query}`;
  }

  async readStream(query, ctx, onToken, mode = null) {
    mode = mode || this.cfg.readMode;
    if (!this.llm.available()) {
      return { answer: null, readMode: "retrieval_only", subAnswers: null,
        note: "Sem LLM configurado: devolvendo só as evidências." };
    }
    // modo tri: roda os 3 leitores por eixo (não-stream), depois faz stream da final
    let sub = null, sys = SYS_READER, prompt;
    if (mode === "tri" && ctx.mode !== "corpus_integral" && ctx.views) {
      const axes = Object.keys(ctx.views);
      const subArr = await Promise.all(axes.map(async (axis) => {
        const llm = this.axisLlms[axis] || this.llm;
        const body = fmtBlocks(ctx.views[axis], 3000);
        try {
          return [axis, await llm.complete(SYS_AXIS(AXIS[axis] || axis),
            [{ role: "user", content: `VISÃO DO SEU EIXO:\n${body}\n\nPERGUNTA: ${query}` }], 500)];
        } catch (e) { return [axis, `(leitor do eixo falhou: ${e.message})`]; }
      }));
      sub = Object.fromEntries(subArr);
      const parts = [];
      if (ctx.summary) parts.push(`MEMÓRIA DA CONVERSA:\n${ctx.summary}`);
      if (ctx.recent) parts.push("TURNOS RECENTES:\n" + ctx.recent.join("\n"));
      for (const [axis, ans] of Object.entries(sub)) parts.push(`RESPOSTA DO ${AXIS[axis] || axis}:\n${ans.trim()}`);
      parts.push("EVIDÊNCIAS MAIS FORTES (fusão quântica):\n" + fmtBlocks(ctx.blocks, 6500));
      prompt = parts.join("\n\n") + `\n\nPERGUNTA ORIGINAL: ${query}\n\nFaça a leitura final.`;
      sys = SYS_FINAL;
    } else {
      prompt = this._fastPrompt(query, ctx);
    }
    let answer;
    try {
      answer = await this.llm.completeStream(sys, [{ role: "user", content: prompt }],
        this.cfg.maxAnswerTokens, onToken);
    } catch (e) {
      // stream falhou de vez: cai para não-streaming e emite a resposta inteira
      answer = await this.llm.complete(sys, [{ role: "user", content: prompt }], this.cfg.maxAnswerTokens);
      onToken(answer);
    }
    return { answer, readMode: sub ? "tri" : "fast", subAnswers: sub };
  }
}
